# Training functions for EEG classification
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from tqdm.auto import tqdm
from typing import Optional
# import sklearn.metrics

from os.path import join as pjoin
from os.path import dirname


@dataclass
class TrainingConfig:
    epochs: int = 100
    opt_log_every: int = 100
    val_every_epochs: int = 1
    clip_grad_norm: float = 1.0
    swa_start: int = 2
    task: Optional[str] = None


def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)


# Define training and evaluation functions for sequence & class variants
def train_seq(args, model, swa_model, train_data, val_data, optimizer, scheduler, swa_scheduler, criterion, device, tblogger):
    def do_permute(output):
        return output.permute(0, 2, 1)
    return _train(do_permute, args, model, swa_model, train_data, val_data, optimizer, scheduler, swa_scheduler, criterion, device, tblogger)


def train_class(args, model, swa_model, train_data, val_data, optimizer, scheduler, swa_scheduler,
                criterion, device, tblogger, comment, i, save_model=True):
    def no_permute(output):
        return output
    return _train(no_permute, args, model, swa_model, train_data, val_data, optimizer, scheduler, swa_scheduler, criterion, device, tblogger, comment, i, save_model=save_model)


def evaluate_seq(model, iterator, criterion, device, tblogger, step, task):
    def do_permute(output):
        return output.permute(0, 2, 1)
    return _evaluate(do_permute, model, iterator, criterion, device, tblogger, step, task="gender")


def evaluate_class(model, iterator, criterion, device, tblogger, step, task):
    def no_permute(output):
        return output
    return _evaluate(no_permute, model, iterator, criterion, device, tblogger, step, task="gender")


# True training and evaluation functions
def _train(output_permuter, args, model, swa_model, train_data, val_data, optimizer,
           scheduler, swa_scheduler, criterion, device, tblogger, comment, count, save_model=True):
    global_step = 0
    losses = []
    accuracies = []
    val_accuracies = []
    lrs = []
    progress = tqdm(total=len(train_data) * args.epochs)

    val_swa_acc = 0
    val_acc = 0
    best_val_acc = 0
    best_swa_val_acc = 0
    running_loss = 0
    running_acc = 0

    for epoch in range(args.epochs):
        model.train()

        for i, (src, trg) in enumerate(train_data):
            # Send to device
            src = src.to(device)
            trg = trg.to(device)
            # Run classifier & take step
            output = model(src, task=args.task)
            optimizer.zero_grad()
            loss = criterion(output_permuter(output), trg)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

            # Log loss, accuracy
            losses.append(loss.item())
            running_loss = 0.99 * running_loss + 0.01 * loss.item()
            y_hat = torch.argmax(output, dim=-1, keepdim=False)
            accuracy = torch.sum(y_hat == trg) / y_hat.nelement()
            running_acc = 0.99 * running_acc + 0.01 * accuracy.item()
            accuracies.append(accuracy.item())
            tblogger.add_scalar("Loss/train", loss.item(), global_step=global_step)
            tblogger.add_scalar("Accuracy/train", accuracy.item(), global_step=global_step)
            tblogger.add_scalar("Grads/norm", grad_norm.item(), global_step=global_step)
            # Update progress bar
            progress.set_postfix({"loss": round(running_loss, 5), "acc": round(running_acc, 3)})
            progress.update(1)
            global_step += 1

        # END OF EPOCH
        # Update SWA model
        if epoch == args.swa_start and swa_model is None:
            swa_model = torch.optim.swa_utils.AveragedModel(model)
        elif swa_model and epoch >= args.swa_start:
            swa_model.update_parameters(model)  # push into if statement

        # Take scheduler step
        if epoch >= args.swa_start and swa_scheduler:
            swa_scheduler.step()
        elif scheduler:
            scheduler.step()
        lrs.append(optimizer.param_groups[0]['lr'])

        # Validate
        if epoch % args.val_every_epochs == 0 or epoch == args.epochs - 1:
            _, val_acc = _evaluate(output_permuter, model, val_data, criterion, device,
                                   tblogger, global_step, args.task)
            val_accuracies.append(val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if save_model:
                    best_base_file = pjoin("models", f"best_model-{comment}.pt")
                    torch.save(model.state_dict(), best_base_file)

            if swa_model:
                _, val_swa_acc = _evaluate(output_permuter, swa_model, val_data, criterion, device,
                                           tblogger, global_step, args.task, swa=True)
                if val_swa_acc > best_swa_val_acc:
                    best_swa_val_acc = val_swa_acc
                    if save_model:
                        best_swa_file = pjoin("models", f"best_swa_model-{comment}.pt")
                        torch.save(swa_model.module.state_dict(), best_swa_file)

        progress.set_postfix({"Epoch": epoch + 1, "TAcc": round(running_acc, 3), "VAcc": round(val_acc, 3)})
        tblogger.add_scalar("LR", optimizer.param_groups[0]['lr'], global_step=global_step)

    # SWA batch norm
    if swa_model:
        torch.optim.swa_utils.update_bn(train_data, swa_model)

    # End of training, save last model
    progress.close()
    if save_model:
        last_base_dir = pjoin("models", "last_model-" + comment + ".pt")
        torch.save(model.state_dict(), last_base_dir)
        if swa_model:
            last_swa_dir = pjoin("models", "last_swa_model-" + comment + ".pt")
            torch.save(swa_model.module.state_dict(), last_swa_dir)

    if swa_model:
        if val_swa_acc > 0.68:
            # ADD TO RESULTS
            with open(pjoin(dirname(__file__), "sweep_results.txt"), 'a') as out:
                out_tuple = ("MODEL: " + str(count), "swa_model", comment, val_swa_acc)
                swa_line = str(out_tuple) + "\n"
                out.write(swa_line)

    if val_acc > 0.68:
        # ADD TO RESULTS
        with open(pjoin(dirname(__file__), "sweep_results.txt"), 'a') as out:
            out_tuple = ("MODEL: " + str(count), "base_model", comment, val_acc)
            base_line = str(out_tuple) + "\n"
            out.write(base_line)

    return losses, accuracies, val_accuracies


def _evaluate(output_permuter, model, data_loader, criterion, device, tblogger, step, task, swa=False):
    model.eval()
    eval_loss = 0
    eval_accuracy = 0
    N = len(data_loader)
    # classification_reports = []
    with torch.no_grad():
        for i, (src, trg) in enumerate(data_loader):
            src = src.to(device)
            trg = trg.to(device)
            output = model(src, task=task)
            loss = criterion(output_permuter(output), trg)
            eval_loss += loss.item()
            y_hat = torch.argmax(output, dim=-1, keepdim=False)
            accuracy = torch.sum(y_hat == trg) / y_hat.nelement()
            eval_accuracy += accuracy.item()
            # classification_reports.append(sklearn.metrics.classification_report(y_hat, trg, [0, 1]))

    # classification_report = {}
    # for key in classification_reports[0].keys():
    #     classification_report[key] = 0
    #     for report in classification_reports:
    #         classification_report[key] += report[key]

    #     classification_report[key] = classification_report[key] / N

    loss_name = "Loss/validate"
    acc_name = "Accuracy/validate"
    if swa:
        loss_name = "Loss/validate_swa"
        acc_name = "Accuracy/validate_swa"
    tblogger.add_scalar(loss_name, eval_loss / N, global_step=step)
    tblogger.add_scalar(acc_name, eval_accuracy / N, global_step=step)
    return eval_loss / N, eval_accuracy / N
