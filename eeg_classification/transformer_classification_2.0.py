#!/usr/bin/env python
# coding: utf-8

# # Imports and Setup

# In[1]:


# Imports for Tensor
import csv
import itertools
import math
import numpy as np
import os
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

sys.path.append("../")


# In[2]:


from models import TransformerClassifier, TransformerSequenceClassifier
from common.eeg_datasets import EarDataset, EarEEGPreprocessor, MathDataset, MathPreprocessor
from training import train_seq, train_class, evaluate_seq, evaluate_class, TrainingConfig
from visualization import *


# In[ ]:


#!ls 'gdrive/My Drive/Muller Group Drive/Ear EEG/Drowsiness_Detection/classifier_TBME'
# !ls C:\Users\arya_bastani\Documents\ear_eeg\data\ear_eeg_data
ear_eeg_base_path = '/data/shared/signal-diffusion/'
# ear_eeg_base_path = '/mnt/d/data/signal-diffusion/'
ear_eeg_data_path = ear_eeg_base_path + 'eeg_classification_data/ear_eeg_data/ear_eeg_clean'
math_data_path = '/data/shared/signal-diffusion/eeg_math/raw_eeg'
# math_data_path = '/mnt/d/data/signal-diffusion/eeg_math/raw_eeg'


# # Data Preprocessing (run once)

# In[ ]:


preprocessor = EarEEGPreprocessor(ear_eeg_base_path,)

seq_len = 2000
# %time preprocessor.preprocess(seq_len)


# In[ ]:


nsamps = 2000
preprocessor = MathPreprocessor(math_data_path, nsamps)

samps_per_file = 100
# %time preprocessor.preprocess(samps_per_file)


# # Models and DataLoaders

# In[ ]:


# # Math EEG Training

# Parameters
BATCH_SIZE = 64
SHUFFLE = True
NUM_WORKERS = 1
N_TOKENS = 128
CONTEXT_SAMPS = 20
N_SAMPS = CONTEXT_SAMPS * N_TOKENS
persistent = NUM_WORKERS > 0
use_pin_memory = torch.cuda.is_available()

# Datasets
train_set = MathDataset(math_data_path + "/train", CONTEXT_SAMPS)
val_set = MathDataset(math_data_path + "/val", CONTEXT_SAMPS)
test_set = MathDataset(math_data_path + "/test", CONTEXT_SAMPS)
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
    num_workers=NUM_WORKERS,
    pin_memory=use_pin_memory,
    persistent_workers=persistent,
)
val_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
    num_workers=NUM_WORKERS,
    pin_memory=use_pin_memory,
    persistent_workers=persistent,
)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
    num_workers=NUM_WORKERS,
    pin_memory=use_pin_memory,
    persistent_workers=persistent,
)

# In[ ]:


# define hyperparameters
INPUT_DIM = train_set.n_channels * CONTEXT_SAMPS
OUTPUT_DIM = 4
HID_DIM = INPUT_DIM
N_LAYERS = 4
N_HEADS = 4
FF_DIM = 256
DROPOUT = 0.1
BATCH_FIRST = True # True: (batch, seq, feature). False: (seq, batch, feature)
WEIGHT_DECAY = 0.0001

# CUDA for PyTorch
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
torch.backends.cudnn.benchmark = device.type == "cuda"

# In[ ]:


opt_options = [torch.optim.AdamW]
decay_options = [0.0001, 0.001]

for opt, decay in itertools.product(opt_options, decay_options):

    # Create model instance
    model = TransformerClassifier(INPUT_DIM, OUTPUT_DIM, HID_DIM, N_LAYERS, N_HEADS, FF_DIM, DROPOUT, BATCH_FIRST)
    model = model.to(device)    
    criterion = nn.CrossEntropyLoss()
    optimizer = opt(model.parameters(), lr=1e-3, weight_decay=decay)

    # Create training configuration
    ARGS = TrainingConfig(epochs=300, val_every_epochs=10)

    # Log statistics
    comment = f"txfmclass_{str(type(optimizer)).split('.')[-1][:-2]}_decay{decay}"
    tbsw = SummaryWriter(log_dir="./tensorboard_logs/" + comment + "-" + datetime.now().isoformat(sep='_'), 
                         comment=comment)
    print("#" * 80)
    print("Training", comment)

    # Training loop
    try:
        losses, accs, val_accs = train_class(
            ARGS, model, 
            train_loader, val_loader,
            optimizer, criterion,
            device, tbsw
        )
    except AssertionError:
        continue
    
    # load best model and evaluate on test set
    model.load_state_dict(torch.load(f'best_model.pt'))
    test_loss, test_acc = evaluate_class(model, test_loader, criterion, device, tbsw, ARGS.epochs * len(train_loader) + 1)
    print(f'Test loss={test_loss:.3f}; test accuracy={test_acc:.3f}')

    # Copy model to unique filename
    os.makedirs("models", exist_ok=True)
    shutil.copyfile("best_model.pt", f"models/best_model-{comment}.pt")
    shutil.copyfile("last_model.pt", f"models/last_model-{comment}.pt")
