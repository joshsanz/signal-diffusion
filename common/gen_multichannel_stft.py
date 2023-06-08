import numpy as np
import os
from os.path import join as pjoin
import sys
import torch

sys.path.append(os.getcwd())
from common.eeg_datasets import MathPreprocessor, MathDataset
from common.multichannel_spectrograms import multichannel_spectrogram


def label_to_cats(y):
    math = y // 2
    female = y % 2
    return math, female


# settings
# 250 Hz sample rate after decimation in MathPreprocessor
# Expecting 2000 timesamples x 20 channels
# 2000 / 128 = 15.625 ==> 16 time bins, 512 freq bins per channel
# 16 * 20 = 320 time bins total
fs = 250  # Original is 500, but we decimate by 2
n_timesteps = 2000
n_channels = 20
time_steps = 512  # number of time-steps. Width of image
win_length = 512  # number of bins in spectrogram. Height of image
max_bins = win_length / n_channels
hop_length = 8  # number of samples per time-step in spectrogram
while n_timesteps / hop_length > max_bins:
    hop_length += 8
print(f"Best hop length {hop_length}, {np.floor(n_timesteps / hop_length)} bins per channel")

##########
# EEG Math
# data_dir = "/mnt/d/data/signal-diffusion/eeg_math/raw_eeg"
# output_dir = "/mnt/d/data/signal-diffusion/eeg_math/raw_stft"
data_dir = "/data/shared/signal-diffusion/eeg_math/raw_eeg"
output_dir = "/data/shared/signal-diffusion/eeg_math/raw_stft"

#########
# EEG AEP
# data_dir = "/data/shared/signal-diffusion/eeg_aep/raw_eeg"
# output_dir = "/data/shared/signal-diffusion/eeg_aep/raw_stft"

# Preprocess data -- only need to do this once
preprocessor = MathPreprocessor(data_dir, n_timesteps, fs=fs)
# preprocessor.preprocess(20, 1.0, 0.0, 0.0)  # train samples per stored file, train/val/test split

stfts = []
for split in ["train", "val", "test"]:
    idir = pjoin(data_dir, split)
    odir = pjoin(output_dir, split)
    os.makedirs(odir, exist_ok=True)
    dataset = MathDataset(idir, 1)  # Keep 2000 samples per channel (1 token total)
    captions = {}
    for i, (X, y) in enumerate(dataset):
        X = torch.transpose(X, 1, 0).contiguous().numpy()
        spec = multichannel_spectrogram(X, resolution=win_length, hop_length=hop_length, win_length=win_length,)
        # Organize into folders of 100 images each
        filename = "{:03d}/{:05d}.png".format(i % 100, i)
        savename = pjoin(odir, filename)
        os.makedirs(os.path.dirname(savename), exist_ok=True)
        spec.save(savename)
        is_math = y.item() // 2 > 0
        subject = dataset.get_subject(i)
        info = preprocessor.get_subject_info(subject)
        caption = "an EEG spectrogram from a {} {} year old subject".format(
            "female" if info['gender'] == 'F' else "male",
            info['age'],
        )
        if is_math:
            caption += " doing math"
        else:
            caption += " resting"
        captions[filename] = caption
    # Make image -> caption metadata file
    with open(pjoin(odir, "metadata.csv"), "w") as f:
        f.write("file_name,text\n")
        for filename, caption in captions.items():
            f.write(f"{filename},{caption}\n")
    print(f"Wrote {len(captions)} image-caption pairs to {odir}")
