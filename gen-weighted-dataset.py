import bisect
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from signal_diffusion.config import load_settings
from signal_diffusion.data.meta import MetaDataset, MetaPreprocessor, MetaSampler


random_seed = 205  # 205 gave a good split for training
np.random.seed(random_seed)
torch.manual_seed(random_seed)

#############
# Hyperparams
bin_spacing = "log"  # 'log' or 'linear'
nsamps = 2000
fs = 125
resolution = 256


#############
# Configuration
dataset_names = ("parkinsons", "seed")
settings = load_settings()
out_dir = settings.output_root / f"reweighted_meta_dataset_{bin_spacing}_n{nsamps}_fs{fs}"
out_dir.mkdir(parents=True, exist_ok=True)

#################
# Preprocess data
preprocessor = MetaPreprocessor(
    settings,
    dataset_names,
    nsamps=nsamps,
    ovr_perc=0.5,
    fs=fs,
    bin_spacing=bin_spacing,
)
# preprocessor.preprocess(resolution=resolution, train_frac=0.8, val_frac=0.2, test_frac=0.0)

# Load datasets
real_train_set = MetaDataset(settings, dataset_names, split="train", tasks=("gender",))
train_sampler = MetaSampler(real_train_set, num_samples=len(real_train_set))

for dataset in real_train_set.datasets:
    print(f"{dataset.dataset_settings.name} dataset size: {len(dataset)}")
print(f"Combined dataset size: {len(real_train_set)}")
print(f"Sampler size: {len(train_sampler)}")

# Get weights
weights = train_sampler.weights / torch.min(train_sampler.weights)
plt.plot(weights)
plt.title("Sample Weights")
plt.tight_layout()
plt.savefig("weights.png")
plt.close()

print(set(weights.numpy()))
print(sum(weights.numpy()))
copy_params = {}
for wgt in set(weights.numpy()):
    assert wgt >= 1, "Weights < 1 not allowed"
    add_n = int(np.floor(wgt))
    add_np1 = int(np.ceil(wgt))
    fractional = wgt - np.floor(wgt)
    if fractional == 0:
        add_np1_every = 0
    else:
        add_np1_every = int(np.floor(1 / (fractional + 1e-6)))
    print(
        f"{wgt:.4f}: add {add_np1} every {add_np1_every if add_np1_every else 'N/A'} else {add_n}"
    )
    copy_params[wgt] = (add_n, add_np1, add_np1_every)


###############
# Resample Data
#
# In order to have a class-balanced version on-disk:
# Select data indices by weight, then add an extra copy every i'th (or skip copy every i'th)
# to match fractional weight.

def make_copies(prog, idxs, metadata_rows, output_dir, copy_fn, concat_dataset):
    count = 0
    cumulative_sizes = concat_dataset.cumulative_sizes
    datasets = concat_dataset.datasets

    for tensor_idx in idxs:
        idx = int(tensor_idx.item())
        dataset_idx = bisect.bisect_right(cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - cumulative_sizes[dataset_idx - 1]

        dataset = datasets[dataset_idx]
        dataset_name = dataset.dataset_settings.name
        metadata_row = dataset.metadata.iloc[sample_idx]

        relative_path = Path(str(metadata_row["file_name"]))
        source_path = dataset.root / relative_path

        copies = copy_fn(count)
        for copy_index in range(copies):
            new_filename = f"{relative_path.stem}_{copy_index}{relative_path.suffix}"
            relative_output = Path(dataset_name) / relative_path.parent / new_filename
            destination_path = output_dir / relative_output
            destination_path.parent.mkdir(parents=True, exist_ok=True)

            metadata_copy = metadata_row.copy()
            metadata_copy["file_name"] = relative_output.as_posix()
            metadata_rows.append(metadata_copy)
            shutil.copyfile(source_path, destination_path)

        count += 1
        prog.update(1)
        prog.refresh()


metadata_rows: list[pd.Series] = []

prog = tqdm(total=len(weights))
for wgt, params in copy_params.items():
    add_n, add_np1, add_np1_every = params

    def copy_fn(count, *, add_n=add_n, add_np1=add_np1, add_np1_every=add_np1_every):
        if add_np1_every <= 0:
            return add_n
        return add_np1 if count % add_np1_every == 0 else add_n

    prog.set_description(f"Copying weight {wgt:.4f}")
    prog.refresh()
    idxs = torch.argwhere(weights == wgt).reshape(-1)
    make_copies(prog, idxs, metadata_rows, out_dir, copy_fn, real_train_set)
prog.close()

print(f"Final dataset size: {len(metadata_rows)}")

###############
# Save metadata
metapd = pd.DataFrame(metadata_rows)
print(metapd)
metadata_csv = out_dir / "metadata.csv"
metapd.to_csv(metadata_csv, index=False)
