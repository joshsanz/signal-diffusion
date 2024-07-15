# Training functions for EEG classification
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from tqdm.auto import tqdm
from typing import Optional
# import sklearn.metrics

from os.path import join as pjoin


@dataclass
class TrainingConfig:
    epochs: int = 100
    opt_log_every: int = 100
    val_every_epochs: int = 1
    clip_grad_norm: float = 1.0
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
def train_seq(args, model, ema_model, train_data, val_data,
              optimizer, scheduler, criterion, device, tblogger):
    def do_permute(output):
        return output.permute(0, 2, 1)
    return _train(do_permute, args, model, ema_model, train_data, val_data,
                  optimizer, scheduler, criterion, device, tblogger)


def train_class(args, model, ema_model, train_data, val_data,
                optimizer, scheduler, criterion, device, tblogger, run_name,
                save_model=True):
    def no_permute(output):
        return output
    return _train(no_permute, args, model, ema_model, train_data, val_data,
                  optimizer, scheduler, criterion, device, tblogger, run_name,
                  save_model=save_model)


def evaluate_seq(model, iterator, criterion, device, tblogger, step, task):
    def do_permute(output):
        return output.permute(0, 2, 1)
    return _evaluate(do_permute, model, iterator, criterion, device, tblogger, step, task=task)


def evaluate_class(model, iterator, criterion, device, tblogger, step, task):
    def no_permute(output):
        return output
    return _evaluate(no_permute, model, iterator, criterion, device, tblogger, step, task=task)


# True training and evaluation functions
def _train(output_permuter, args, model, ema_model, train_data, val_data, optimizer,
           scheduler, criterion, device, tblogger, run_name, save_model=True):
    global_step = 0
    losses = []
    accuracies = []
    val_accuracies = []
    ema_val_accuracies = []
    lrs = []
    progress = tqdm(total=len(train_data) * args.epochs)

    val_ema_acc = 0
    val_acc = 0
    best_val_acc = 0
    best_ema_val_acc = 0
    running_loss = 0
    running_acc = 0

    for epoch in range(args.epochs):
        model.train()

        for i, xy in enumerate(train_data):
            # Extract and send to device
            if isinstance(xy, list):
                src, trg = xy
            elif isinstance(xy, dict):
                src, trg = xy['image'], xy['class']
            else:
                raise ValueError(f"Unknown data format {type(xy)}")
            src = src.to(device)
            trg = trg.to(device)
            # Run classifier & take step
            output = model(src, task=args.task)
            optimizer.zero_grad()
            loss = criterion(output_permuter(output), trg)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            if ema_model:
                ema_model.update_parameters(model)

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
        # Take scheduler step
        if scheduler:
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
                    best_base_file = pjoin("models", f"best_model-{run_name}.pt")
                    torch.save(model.state_dict(), best_base_file)

            if ema_model:
                _, val_ema_acc = _evaluate(output_permuter, ema_model, val_data, criterion,
                                           device, tblogger, global_step, args.task, ema=True)
                ema_val_accuracies.append(val_ema_acc)
                if val_ema_acc > best_ema_val_acc:
                    best_ema_val_acc = val_ema_acc
                    if save_model:
                        best_ema_file = pjoin("models", f"best_ema_model-{run_name}.pt")
                        torch.save(ema_model.module.state_dict(), best_ema_file)
            else:
                ema_val_accuracies.append(0)

        progress.set_postfix({"Epoch": epoch + 1, "TAcc": round(running_acc, 3),
                              "VAcc": round(val_acc, 3)})
        tblogger.add_scalar("Scheduling/LR", optimizer.param_groups[0]['lr'],
                            global_step=global_step)

    # SWA batch norm
    if ema_model:
        torch.optim.swa_utils.update_bn(train_data, ema_model)

    # End of training, save last model
    progress.close()
    if save_model:
        last_base_dir = pjoin("models", "last_model-" + run_name + ".pt")
        torch.save(model.state_dict(), last_base_dir)
        if ema_model:
            last_ema_dir = pjoin("models", "last_ema_model-" + run_name + ".pt")
            torch.save(ema_model.module.state_dict(), last_ema_dir)

    return losses, accuracies, val_accuracies, ema_val_accuracies


def _evaluate(output_permuter, model, data_loader, criterion, device, tblogger,
              step, task, ema=False):
    model.eval()
    eval_loss = 0
    eval_accuracy = 0
    N = len(data_loader)
    with torch.no_grad():
        for i, xy in enumerate(data_loader):
            # Extract and send to device
            if isinstance(xy, list):
                src, trg = xy
            elif isinstance(xy, dict):
                src, trg = xy['image'], xy['class']
            else:
                raise ValueError(f"Unknown data format {type(xy)}")
            src = src.to(device)
            trg = trg.to(device)
            # Check output
            output = model(src, task=task)
            loss = criterion(output_permuter(output), trg)
            eval_loss += loss.item()
            y_hat = torch.argmax(output, dim=-1, keepdim=False)
            accuracy = torch.sum(y_hat == trg) / y_hat.nelement()
            eval_accuracy += accuracy.item()
    loss_name = "Loss/validate"
    acc_name = "Accuracy/validate"
    if ema:
        loss_name = "Loss/validate_ema"
        acc_name = "Accuracy/validate_ema"
    tblogger.add_scalar(loss_name, eval_loss / N, global_step=step)
    tblogger.add_scalar(acc_name, eval_accuracy / N, global_step=step)
    return eval_loss / N, eval_accuracy / N
