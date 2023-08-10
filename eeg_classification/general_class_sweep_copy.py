# %% [markdown]
# ## Imports and Setup

# %% [markdown]
# 

# %%
# Imports for Tensor
import csv
import itertools
import math
import numpy as np
import os
import pandas as pd
import shutil
import sys
from collections import OrderedDict
from datetime import datetime
from tempfile import TemporaryDirectory
from typing import Tuple

from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.tensorboard import SummaryWriter
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from torchvision import transforms
import torchcontrib
from torchcontrib.optim import SWA


from diffusers import StableDiffusionPipeline
from datasets import load_dataset

#sys.path.append("../")
#sys.path.append("/home/abastani/signal-diffusion/common")
#sys.path.append("/home/abastani/signal-diffusion/data_processing")

sys.path.insert(0, '/home/abastani/signal-diffusion/')




print("THI: ", sys.path)


# %%
from common.dog import DoG, LDoG, PDoG
from cnn_models import model_size
from cnn_models import CNNClassifier, CNNClassifierLight, LabelSmoothingCrossEntropy
from cnn_models import EfficientNet, ShuffleNet, ResNet
from data_processing.math import MathDataset
from data_processing.parkinsons import ParkinsonsDataset
from data_processing.seed import SEEDDataset
from data_processing.general_dataset import GeneralPreprocessor, GeneralDataset, GeneralSampler
from data_processing.general_dataset import general_class_labels, general_dataset_map
from training import train_class, evaluate_class, TrainingConfig
from visualization import *

# %%
random_seed = 205 #205 Gave a good split for training
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# %%
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

#SEED dataset
datadirs['seed'] = f'{datahome}/seed/'
datadirs['seed-stft'] = os.path.join(datadirs['seed'], "stfts")

# %% [markdown]
# # Data Preprocessing (run once)

# %%
nsamps = 2000

preprocessor = GeneralPreprocessor(datadirs, nsamps, ovr_perc=0.5, fs=125) 
#preprocessor.preprocess(resolution=256, train_frac=0.8, val_frac=0.2, test_frac=0.0)

# %% [markdown]
# # Train on Real Data

# %% [markdown]
# ## Dataloader Setup

# %%
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

# %%
# Datasets, excluding math
math_val_dataset = MathDataset(datadirs['math-stft'], split="val")
parkinsons_val_dataset = ParkinsonsDataset(datadirs['parkinsons-stft'], split="val")
seed_val_dataset = SEEDDataset(datadirs['seed-stft'], split="val")
val_datasets = [parkinsons_val_dataset, seed_val_dataset]

math_real_train_dataset = MathDataset(datadirs['math-stft'], split="train")
parkinsons_real_train_dataset = ParkinsonsDataset(datadirs['parkinsons-stft'], split="train", transform=None)
seed_real_train_dataset = SEEDDataset(datadirs['seed-stft'], split="train")
real_train_datasets = [parkinsons_real_train_dataset, seed_real_train_dataset]


val_set = GeneralDataset(val_datasets, split='val')
real_train_set = GeneralDataset(real_train_datasets, split='train')

# Sampler for balanced training data
train_samp = GeneralSampler(real_train_datasets, BATCH_SIZE, split='train')

# %%
# Dataloaders
val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=SHUFFLE,
                                         num_workers=NUM_WORKERS, pin_memory=True, 
                                         persistent_workers=persistent)
real_train_loader = torch.utils.data.DataLoader(real_train_set, batch_size=BATCH_SIZE, 
                                                num_workers=NUM_WORKERS, pin_memory=True, 
                                                persistent_workers=persistent, sampler=train_samp)

# %%
# Define hyper paramters
OUTPUT_DIM = 2
DROPOUT = 0.5
BATCH_FIRST = True # True: (batch, seq, feature). False: (seq, batch, feature)
#BASE_LEARNING_RATE = 5e-4
#L2_REG_DECAY = 0.00
#SWA_LEARNING_RATE = 1e-4
#EPOCHS = 20
#SWA_START = 15

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True


# Scheduler
epochs = list(range(1, 4))
swa_start_percs = [0.5]
optimizers = [torch.optim.AdamW]
base_learning_rates = [1e-2]
swa_learning_rates = [1e-2]
decays = [0.1]
label_smoothing_epsilons = [1.0]
schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR]
best_swa_models_dict = {}
best_models_dict = {}

# %%
for epoch in epochs:
    for swa_start_perc in swa_start_percs:
        swa_start = int(swa_start_perc * epoch)
        print("START: ", swa_start)
        for optimizer in optimizers:
            for base_learning_rate in base_learning_rates:
                for swa_learning_rate in swa_learning_rates:
                    for decay in decays:
                        for epsilon in label_smoothing_epsilons:
                            for scheduler in schedulers:
                                EPOCHS = epoch
                                SWA_START = swa_start
                                BASE_LEARNING_RATE = base_learning_rate
                                SWA_LEARNING_RATE = swa_learning_rate
                                L2_REG_DECAY = decay
                                EPSILON = epsilon

                                random_seed = 205 #205 Gave a good split for training
                                np.random.seed(random_seed)
                                torch.manual_seed(random_seed)
                            
                                # Loss function
                                criterion = LabelSmoothingCrossEntropy(epsilon=EPSILON)

                                # Optimizer
                                opt, decay, restart, max_eta, decouple = (optimizer, L2_REG_DECAY, 0, None, True)

                                # Create model instance
                                model = CNNClassifierLight(1, OUTPUT_DIM, dropout=DROPOUT,pooling="max")
                                model = model.to(device) 

                                if opt == torch.optim.AdamW:
                                    optimizer = opt(model.parameters(), lr=BASE_LEARNING_RATE, weight_decay=decay)
                                elif opt == torch.optim.SGD:
                                    BASE_LEARNING_RATE = BASE_LEARNING_RATE * 100
                                    # optimizer = opt(model.parameters(), lr=BASE_LEARNING_RATE, weight_decay=decay, momentum = 0.5)
                                    optimizer = opt(model.parameters(), lr=BASE_LEARNING_RATE)

                                if scheduler == torch.optim.lr_scheduler.ExponentialLR:
                                    exp_sched_gamma = 0.9
                                    scheduler = scheduler(optimizer, exp_sched_gamma, verbose=False, last_epoch=- 1)
                                elif scheduler == torch.optim.lr_scheduler.CosineAnnealingLR:
                                    scheduler = scheduler(optimizer, T_max=EPOCHS, last_epoch=- 1)

                                # SWA model instance
                                if SWA_START < EPOCHS:
                                    swa_model = torch.optim.swa_utils.AveragedModel(model)
                                    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=SWA_LEARNING_RATE)
                                else:
                                    swa_model = None
                                    swa_scheduler = None

                                # Create training configuration
                                ARGS = TrainingConfig(epochs=EPOCHS, val_every_epochs=1, opt_restart_every=restart, swa_start=SWA_START)

                                # Log statistics
                                postfix = ""
                                if isinstance(optimizer, DoG):
                                    postfix = f"_restart{restart}_etamax{max_eta}_decouple{str(int(decouple))}"
                                comment = f"cnnclass_{model.name}_{str(type(optimizer)).split('.')[-1][:-2]}_decay{decay}{postfix}_epoch:{EPOCHS},swa_start:{SWA_START},base_lr:{BASE_LEARNING_RATE},swa_lr:{SWA_LEARNING_RATE},epsilon:{EPSILON},sched:{scheduler}"
                                tbsw = SummaryWriter(log_dir="/home/abastani/signal-diffusion/eeg_classification/tensorboard_logs/cnn/" + comment + "-" + 
                                                    datetime.now().isoformat(sep='_'), 
                                                    comment=comment)
                                print("#" * 80)
                                print("Training", comment)

                                # Training loop
                                losses, accs, val_accs, swa_model = train_class(
                                    ARGS, model, swa_model,
                                    real_train_loader, val_loader,
                                    optimizer, scheduler, swa_scheduler, 
                                    criterion, device, tbsw, best_swa_models_dict,
                                    best_models_dict, comment
                                )

                                # load best model and evaluate on test set
                                #model.load_state_dict(torch.load(f'best_model.pt'))
                                # test_loss, test_acc = evaluate_class(model, test_loader, criterion, device, 
                                #                                      tbsw, ARGS.epochs * len(real_train_loader) + 1)
                                # print(f'Test loss={test_loss:.3f}; test accuracy={test_acc:.3f}')

                                # # Copy model to unique filename
                                # os.makedirs("models", exist_ok=True)
                                # shutil.copyfile("best_model.pt", f"models/best_model-{comment}.pt")
                                # shutil.copyfile("last_model.pt", f"models/last_model-{comment}.pt")
                                # print(f"Copied best model to models/best_model-{comment}.pt")

                                rmodel = model

swa_keys = list(best_swa_models_dict.keys())
swa_keys.sort()
swa_keys = swa_keys[:30]
base_keys = list(best_models_dict.keys())
base_keys.sort()
base_keys = base_keys[:30]


with open("/home/abastani/signal-diffusion/eeg_classification/sweep_results.txt", 'w') as out:
    for i in range(2):
        swa_line = "swa_val: " + str(swa_keys[i]) + "model comment: " + str(best_swa_models_dict[swa_keys[i]]) + "\n"
        out.write(swa_line)

        base_line = "base_val: " + str(base_keys[i]) + "model comment: " + str(best_models_dict[base_keys[i]]) + "\n"
        out.write(base_line)
