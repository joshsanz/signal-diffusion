# Training functions for EEG classification
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from tqdm.auto import tqdm
from typing import Optional
from common.dog import DoG, PDoG


@dataclass
class TrainingConfig:
    epochs: int = 100
    opt_restart_every: int = 200
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


def log_etas(tblogger, opt, step):
    state = opt.state_dict()
    scalars = {}
    for i, p in enumerate(state['param_groups']):
        if isinstance(opt, PDoG):
            etas = torch.stack([eta.norm() for eta in p['eta']]).detach().cpu()
        else:
            etas = torch.stack(p['eta']).detach().cpu()
        tblogger.add_histogram(f"Eta.{i}", etas, global_step=step)
    return scalars


# Define training and evaluation functions for sequence & class variants
def train_seq(args, model, swa_model, train_data, val_data, optimizer, scheduler, swa_scheduler, criterion, device, tblogger):
    def do_permute(output):
        return output.permute(0, 2, 1)
    return _train(do_permute, args, model, swa_model, train_data, val_data, optimizer, scheduler, swa_scheduler, criterion, device, tblogger)


def train_class(args, model, swa_model, train_data, val_data, optimizer, scheduler, swa_scheduler, 
criterion, device, tblogger, comment, i, save_model=True):
    def no_permute(output):
        return output
    return _train(no_permute, args, model, swa_model, train_data, val_data, optimizer, scheduler, swa_scheduler, criterion, device, tblogger, comment, i, save_model)


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
    scheduler, swa_scheduler, criterion, device, tblogger, comment, county, save_model=True):
    global_step = 0
    losses = []
    accuracies = []
    val_accuracies = []
    best_valid_acc = 0
    progress = tqdm(total=len(train_data) * args.epochs)

    val_swa_acc = 0
    val_acc = 0

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

            # Take scheduler step
            if epoch > args.swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            elif scheduler:
                scheduler.step()

            # Log loss, accuracy
            losses.append(loss.item())
            y_hat = torch.argmax(output, dim=-1, keepdim=False)
            accuracy = torch.sum(y_hat == trg) / y_hat.nelement()
            accuracies.append(accuracy.item())
            tblogger.add_scalar("Loss/train", loss.item(), global_step=global_step)
            tblogger.add_scalar("Accuracy/train", accuracy.item(), global_step=global_step)
            tblogger.add_scalar("Grads/norm", grad_norm.item(), global_step=global_step)
            if isinstance(optimizer, DoG) and global_step % args.opt_log_every == 0:
                log_etas(tblogger, optimizer, global_step)
            # Update progress bar
            progress.set_postfix({"loss": round(loss.item(), 5), "acc": round(accuracy.item(), 3)})
            progress.update(1)
            global_step += 1
        # End of epoch, validate
        if args.opt_restart_every > 0 and isinstance(optimizer, DoG) and (epoch + 1) % args.opt_restart_every == 0:
            optimizer.reset(keep_etas=False)

        if epoch % args.val_every_epochs == 0 or epoch == args.epochs - 1:
            _, val_acc = _evaluate(output_permuter, model, val_data, criterion, device,
                                   tblogger, global_step, args.task)
            if swa_model:
                _, val_swa_acc = _evaluate(output_permuter, swa_model, val_data, criterion, device,
                                        tblogger, global_step, args.task, swa=True)
            val_accuracies.append(val_acc)
            if val_acc > best_valid_acc:
                best_valid_acc = val_acc
                if save_model == True:
                    torch.save(model.state_dict(), "best_model.pt")

        progress.set_postfix({"Epoch": epoch + 1, "TAcc": round(accuracies[-1], 3), "VAcc": round(val_acc, 3)})
        tblogger.add_scalar("LR", optimizer.param_groups[0]['lr'], global_step=global_step)


    # SWA batch norm
    if swa_model: 
        torch.optim.swa_utils.update_bn(train_data, swa_model)
    
    # End of training, save last model
    progress.close()
    if save_model == True:
        torch.save(model.state_dict(), "last_model.pt")

    if swa_model:
        if val_swa_acc > 0.68:
           # ADD TO RESULTS
           with open("/home/abastani/signal-diffusion/eeg_classification/sweep_results.txt", 'a') as out:
            out_tuple = ("MODEL: "+str(county), "swa_model", comment, val_swa_acc)
            swa_line = str(out_tuple) + "\n"
            out.write(swa_line)
            out.close()

    if val_acc > 0.68:
       # ADD TO RESULTS
       with open("/home/abastani/signal-diffusion/eeg_classification/sweep_results.txt", 'a') as out:
            out_tuple = ("MODEL: "+str(county), "base_model", comment, val_acc)
            base_line = str(out_tuple) + "\n"
            out.write(base_line)
            out.close()


    # torch.save(model.state_dict(), "last_model.pt")
    return losses, accuracies, val_accuracies


def _evaluate(output_permuter, model, data_loader, criterion, device, tblogger, step, task, swa=False):
    model.eval()
    eval_loss = 0
    eval_accuracy = 0
    N = len(data_loader)
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
    loss_name = "Loss/validate"
    acc_name = "Accuracy/validate"
    if swa:
        loss_name = "Loss/validate_swa"
        acc_name = "Accuracy/validate_swa"
    tblogger.add_scalar(loss_name, eval_loss / N, global_step=step)
    tblogger.add_scalar(acc_name, eval_accuracy / N, global_step=step)
    return eval_loss / N, eval_accuracy / N
