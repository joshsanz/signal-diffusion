# Take a synthetic dataset and split it into train/val chunks in a new folder.

import argparse
import glob
import os
from os.path import join as pjoin
import pandas as pd
import shutil as shu
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Split a synthetic dataset into train/val chunks.')
parser.add_argument('input', type=str, help='Path to the synthetic dataset.')
parser.add_argument("-o", "--output", type=str, help='Path to the output folder. Default `input`-split')
parser.add_argument("-f", "--val-frac", type=float, default=0.2, help='Size of the validation set as a fraction of the dataset size. Default 0.2')
parser.add_argument("-s", "--seed", type=int, default=42, help='Random seed for shuffling the dataset. Default 42')
args = parser.parse_args()
assert 0 < args.val_frac < 1
assert os.path.exists(args.input)
if not args.output:
    args.output = args.input + "-split"
print("Splitting dataset from\n\t{}\ninto train/val in\n\t{}".format(args.input, args.output))

# Load the dataset
df = pd.read_csv(pjoin(args.input, "metadata.csv"))
n = len(df)
n_val = int(n * args.val_frac)
n_train = n - n_val
# Shuffle the rows just in case, using the provided seed for reproducibility
df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

# Make the output folders
os.makedirs(args.output, exist_ok=True)
existing = glob.glob(pjoin(args.output, "*"))
for e in existing:
    shu.rmtree(e)
os.makedirs(pjoin(args.output, "train"), exist_ok=True)
os.makedirs(pjoin(args.output, "val"), exist_ok=True)


# Create the train split
df_train = df.head(n_train)
df_train.to_csv(pjoin(args.output, "train", "metadata.csv"), index=False)
for i, row in tqdm(df_train.iterrows(), total=n_train):
    shu.copy(pjoin(args.input, row["filename"]), pjoin(args.output, "train", row["filename"]))

# Create the val split
df_val = df.tail(n_val)
df_val.to_csv(pjoin(args.output, "val", "metadata.csv"), index=False)
for i, row in tqdm(df_val.iterrows(), total=n_val):
    shu.copy(pjoin(args.input, row["filename"]), pjoin(args.output, "val", row["filename"]))
