# Imports for Tensor
import bisect
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil

from tqdm.auto import tqdm

import torch

# from data_processing.math import MathDataset
from data_processing.parkinsons import ParkinsonsDataset
from data_processing.seed import SEEDDataset
from data_processing.general_dataset import GeneralPreprocessor, GeneralDataset, GeneralSampler

random_seed = 205  # 205 gave a good split for training
np.random.seed(random_seed)
torch.manual_seed(random_seed)

#############
# Hyperparams
bin_spacing = "log"  # 'log' or 'linear'
nsamps = 2000
fs = 125
resolution = 256

###########
# Datapaths
datadirs = {}
datahome = '/data/shared/signal-diffusion'
# datahome = '/mnt/d/data/signal-diffusion'
out_dir = f'{datahome}/reweighted_meta_dataset_{bin_spacing}_n{nsamps}_fs{fs}'

# Math dataset
datadirs['math'] = f'{datahome}/eeg_math'
datadirs['math-stft'] = os.path.join(datadirs['math'], 'stfts')

# Parkinsons dataset
datadirs['parkinsons'] = f'{datahome}/parkinsons/'
datadirs['parkinsons-stft'] = os.path.join(datadirs['parkinsons'], 'stfts')

# SEED dataset
datadirs['seed'] = f'{datahome}/seed/'
datadirs['seed-stft'] = os.path.join(datadirs['seed'], "stfts")

#################
# Preprocess data
preprocessor = GeneralPreprocessor(datadirs, nsamps, ovr_perc=0.5, fs=fs, bin_spacing=bin_spacing)
# preprocessor.preprocess(resolution=resolution, train_frac=0.8, val_frac=0.2, test_frac=0.0)

# Load datasets
BATCH_SIZE = 32
parkinsons_real_train_dataset = ParkinsonsDataset(datadirs['parkinsons-stft'], split="train", transform=None)
seed_real_train_dataset = SEEDDataset(datadirs['seed-stft'], split="train")
real_train_datasets = [parkinsons_real_train_dataset, seed_real_train_dataset]
real_train_set = GeneralDataset(real_train_datasets, split='train')
train_samp = GeneralSampler(real_train_datasets, BATCH_SIZE, split='train')

# Print dataset sizes
print(f"SEED dataset size: {len(seed_real_train_dataset)}")
print(f"Parkinsons dataset size: {len(parkinsons_real_train_dataset)}")
print(f"Combined dataset size: {len(real_train_set)}")
print(f"Sampler size: {len(train_samp)}")

# Get weights
weights = train_samp.weights / torch.min(train_samp.weights)
plt.plot(weights)
plt.title('Sample Weights')
plt.tight_layout()
plt.savefig('weights.png')
plt.close()

print(set(weights.numpy()))
print(sum(weights.numpy()))
copy_dict = {}
for wgt in set(weights.numpy()):
    assert wgt >= 1, "Weights < 1 not allowed"
    add_n = int(np.floor(wgt))
    add_np1 = int(np.ceil(wgt))
    add_np1_every = int(np.floor(1 / (wgt - np.floor(wgt) + 1e-6)))
    print(f"{wgt:.4f}: add {np.ceil(wgt)} every {np.floor(1/(wgt - np.floor(wgt) + 1e-5))} else {np.floor(wgt)}")
    copy_dict[wgt] = (add_n, add_np1, add_np1_every)


###############
# Resample Data
#
# In order to have a class-balanced version on-disk:
# Select data indices by weight, then add an extra copy every i'th (or skip copy every i'th)
# to match fractional weight.

def make_copies(prog, idxs, metadata, out_dir, copy_fn):
    count = 0
    for i in idxs:
        i = i.item()
        # Get dataset & index
        dataset_idx = bisect.bisect_right(real_train_set.cumulative_sizes, i)
        if dataset_idx == 0:
            sample_idx = i
        else:
            sample_idx = i - real_train_set.cumulative_sizes[dataset_idx - 1]
        dataset = real_train_set.datasets[dataset_idx]
        # Get metadata
        md = dataset.metadata.iloc[sample_idx]
        # Make output directory
        os.makedirs(os.path.dirname(os.path.join(out_dir, dataset.name, md['file_name'])), exist_ok=True)
        # Number of copies
        copies = copy_fn(count)
        # Save copies
        for c in range(copies):
            fn = md['file_name']
            new_fn = f'{dataset.name}/{fn[:-4]}_{c}.png'
            mdc = md.copy()
            mdc['file_name'] = new_fn
            metadata.append(mdc)
            shutil.copyfile(os.path.join(dataset.datadir, fn), os.path.join(out_dir, new_fn))
        count += 1
        prog.update(1)
        prog.refresh()


metadata = []

prog = tqdm(total=len(weights))
for wgt in copy_dict:
    prog.set_description(f'Copying weight {wgt:.4f}')
    prog.refresh()
    idxs = torch.argwhere(weights == wgt).reshape(-1)
    copy_fn = lambda count: add_np1 if count % add_np1_every == 0 else add_n
    make_copies(prog, idxs, metadata, out_dir, copy_fn)
prog.close()

print(f"Final dataset size: {len(metadata)}")

###############
# Save metadata
metapd = pd.DataFrame(metadata)
print(metapd)
metapd.to_csv(os.path.join(out_dir, 'metadata.csv'), index=False)
