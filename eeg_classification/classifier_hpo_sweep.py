# ## Imports and Setup
import numpy as np
import os
# import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
import time
from functools import reduce
import warnings

import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
import torchvision.transforms.v2 as transforms
from itertools import product

sys.path.append("../")

from cnn_models import CNNClassifierLight, LabelSmoothingCrossEntropy
# from cnn_models import EfficientNet, ShuffleNet, ResNet, CNNClassifier
from data_processing.math import MathDataset
from data_processing.parkinsons import ParkinsonsDataset
from data_processing.seed import SEEDDataset
from data_processing.general_dataset import GeneralPreprocessor, GeneralDataset, GeneralSampler
# from data_processing.general_dataset import general_class_labels, general_dataset_map
from training import train_class, TrainingConfig


class SummaryWriter(SummaryWriter):
    """Override SummaryWriter so it doesn't place hparams in a separate subdirectory."""
    def add_hparams(self, hparam_dict, metric_dict):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        logdir = self._get_file_writer().get_logdir()

        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v)


@dataclass
class DataLoaders:
    real_train_loader: torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader


def timefmt(t):
    t = int(t)
    h = t // 3600
    m = (t % 3600) // 60
    s = t % 60
    fmtd = f"{h:02d}:" if h > 0 else ""
    fmtd += f"{m:02d}:" if m > 0 else "00:"
    fmtd += f"{s:02d}"
    return fmtd


def print_run_header(t0, run, num_runs):
    t1 = time.time()
    t_run = timefmt(t1 - t0)
    t_est = "??" if run == 0 else timefmt((t1 - t0) / run * num_runs)
    print(f"Training run {run + 1}/{num_runs} [{t_run}/{t_est}]")


pjoin = os.path.join
os.makedirs("models", exist_ok=True)
os.makedirs("tensorboard_logs", exist_ok=True)

random_seed = 205  # 205 Gave a good split for training
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# Datapaths
datadirs = {}
datahome = '/data/shared/signal-diffusion'
# datahome = '/mnt/d/data/signal-diffusion'

# Math dataset
datadirs['math'] = f'{datahome}/eeg_math'
datadirs['math-stft'] = os.path.join(datadirs['math'], 'stfts')

# Parkinsons dataset
datadirs['parkinsons'] = f'{datahome}/parkinsons/'
datadirs['parkinsons-stft'] = os.path.join(datadirs['parkinsons'], 'stfts')

# SEED dataset
datadirs['seed'] = f'{datahome}/seed/'
datadirs['seed-stft'] = os.path.join(datadirs['seed'], "stfts")

# # Data Preprocessing (run once)

nsamps = 2000

preprocessor = GeneralPreprocessor(datadirs, nsamps, ovr_perc=0.5, fs=125)
# preprocessor.preprocess(resolution=256, train_frac=0.8, val_frac=0.2, test_frac=0.0)

# # Train on Real Data

# ## Dataloader Setup

# Parameters
BATCH_SIZE = 32
SHUFFLE = True
NUM_WORKERS = 4
N_TOKENS = 128
RESOLUTION = 256
HOP_LENGTH = 80
persistent = NUM_WORKERS > 0

# Data augmentation
randtxfm = transforms.Compose([
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# Datasets, excluding math
dataloaders = dict()
for transform_type in [None, "trivial_augment_wide"]:
    if transform_type == "trivial_augment_wide":
        transform_fn = randtxfm
    else:
        transform_fn = None

    math_val_dataset = MathDataset(datadirs['math-stft'], split="val")
    parkinsons_val_dataset = ParkinsonsDataset(datadirs['parkinsons-stft'], split="val")
    seed_val_dataset = SEEDDataset(datadirs['seed-stft'], split="val")
    val_datasets = [parkinsons_val_dataset, seed_val_dataset]

    math_real_train_dataset = MathDataset(datadirs['math-stft'], split="train", transform=transform_fn)
    parkinsons_real_train_dataset = ParkinsonsDataset(datadirs['parkinsons-stft'], split="train", transform=randtxfm)
    seed_real_train_dataset = SEEDDataset(datadirs['seed-stft'], split="train", transform=transform_fn)
    real_train_datasets = [parkinsons_real_train_dataset, seed_real_train_dataset]

    val_set = GeneralDataset(val_datasets, split='val')
    real_train_set = GeneralDataset(real_train_datasets, split='train')

    # Sampler for balanced training data
    train_samp = GeneralSampler(real_train_datasets, BATCH_SIZE, split='train')

    # Dataloaders
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=SHUFFLE,
                                             num_workers=NUM_WORKERS, pin_memory=True,
                                             persistent_workers=persistent)
    real_train_loader = torch.utils.data.DataLoader(real_train_set, batch_size=BATCH_SIZE,
                                                    num_workers=NUM_WORKERS, pin_memory=True,
                                                    persistent_workers=persistent, sampler=train_samp)
    dataloaders[transform_type] = DataLoaders(real_train_loader, val_loader)

# Define hyper paramters
OUTPUT_DIM = 2
# DROPOUT = 0.5
BATCH_FIRST = True  # True: (batch, seq, feature). False: (seq, batch, feature)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Tune primary model first, thn weight averaging model; Done
# Label smoothing don't need every .1 -> .3, .5, .9, 0; Done
# Only save models to disk, the best models dict is being kept in RAM prob
# Do a .cpu at the end of training and save to disk

# Itertools.product instead of a bunch of for loops
# Make sure you're saving models into models dir

# Scheduler
# epochs = [5,10,15]
# optimizers = [torch.optim.SGD] #torch.optim.AdamW,
# base_learning_rates = [1e-2, 1e-3, 1e-4, 3e-2, 3e-3, 3e-4]
# decays = [0.1, 0.01, 0.001, 0.3, 0.03, 0.003, 0.5, 0.05, 0.005]
# label_smoothing_epsilons = [0.9, 0.5, 0.3, 0.1, 0.0]
# schedulers = [torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, None]

epochs = [15]
transform_types = [None, "trivial_augment_wide"]
optimizers = [torch.optim.AdamW]
base_learning_rates = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]  # for adam 1e-2 -> 1e-4
decays = [0.1, 0.003, 0.0]
label_smoothing_epsilons = [0.3, 0.1, 0.0]
dropouts = [0, 0.25, 0.5, 0.75]
schedulers = [None]
poolings = ["mean"]  # "max"
save_model = True

num_combos = (len(epochs) * len(transform_types) * len(optimizers) * len(base_learning_rates) *
              len(decays) * len(dropouts) * len(poolings) *
              len(label_smoothing_epsilons) * len(schedulers))
print(f"*** Total number of combinations: {num_combos} ***")
t0 = time.time()
run = 0
for params in product(epochs, transform_types, optimizers, base_learning_rates,
                      decays, dropouts, poolings, label_smoothing_epsilons, schedulers):
    print("#" * 80)
    print_run_header(t0, run, num_combos)
    EPOCHS, TRANSFORM_TYPE, optimizer, BASE_LEARNING_RATE, L2_REG_DECAY, DROPOUT, POOLING, EPSILON, scheduler = params
    random_seed = 205  # 205 Gave a good split for training
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Data
    real_train_loader, val_loader = dataloaders[TRANSFORM_TYPE]

    # Loss function
    criterion = LabelSmoothingCrossEntropy(epsilon=EPSILON)

    # Create model instance
    model = CNNClassifierLight(1, OUTPUT_DIM, dropout=DROPOUT, pooling=POOLING)
    model = model.to(device)

    # Optimizer
    optimizer = optimizer(model.parameters(), lr=BASE_LEARNING_RATE, weight_decay=L2_REG_DECAY)

    # Scheduler
    if scheduler == torch.optim.lr_scheduler.ExponentialLR:
        exp_sched_gamma = 0.9
        scheduler = scheduler(optimizer, exp_sched_gamma, verbose=False, last_epoch=-1)
    elif scheduler == torch.optim.lr_scheduler.CosineAnnealingWarmRestarts:
        scheduler = scheduler(optimizer, T_0=EPOCHS // 3, T_mult=1, eta_min=1e-5, last_epoch=-1)
        warnings.warn("CosineAnnealingWarmRestarts is not used correctly in the training function")

    # EMA model
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))

    # Create training configuration
    ARGS = TrainingConfig(epochs=EPOCHS, val_every_epochs=1)

    # Log statistics
    postfix = ""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"run{run}-{model.name}-{timestamp}"
    print("Run name:", run_name)
    tbsw = SummaryWriter(log_dir=pjoin("tensorboard_logs/cnn", run_name))

    # Training loop
    losses, accs, val_accs, ema_val_accs = train_class(
        ARGS, model, ema_model, real_train_loader, val_loader,
        optimizer, scheduler, criterion, device, tbsw, run_name, run,
        save_model=save_model
    )
    tbsw.add_hparams(dict(
        epochs=ARGS.epochs, task=ARGS.task,
        clip_grad_norm=ARGS.clip_grad_norm,
        dropout=model.dropout,
        optimizer=optimizer.__class__.__name__,
        scheduler=scheduler.__class__.__name__,
        lr=optimizer.param_groups[0]['lr'],
        decay=optimizer.param_groups[0]['weight_decay'],
        pooling=POOLING,
        criterion=criterion.__class__.__name__,
        label_smoothing=0 if not isinstance(criterion, LabelSmoothingCrossEntropy) else criterion.epsilon,
        transform=transform_type,
    ), {
        'hparams/val_acc': max(val_accs),
        'hparams/ema_val_acc': max(ema_val_accs),
        'hparams/train_acc': reduce(lambda x1, x2: 0.99 * x1 + 0.01 * x2, accs, 0),  # Smoothed final estimate
        'hparams/train_loss': reduce(lambda x1, x2: 0.99 * x1 + 0.01 * x2, losses, 0),  # Smoothed final estimate
    })
    tbsw.flush()
    tbsw.close()

    # load best model and evaluate on test set
    # model.load_state_dict(torch.load(f'best_model-{run_name}.pt'))
    # test_loss, test_acc = evaluate_class(model, test_loader, criterion, device,
    #                                      tbsw, ARGS.epochs * len(real_train_loader) + 1)
    # print(f'Test loss={test_loss:.3f}; test accuracy={test_acc:.3f}')

    run += 1
