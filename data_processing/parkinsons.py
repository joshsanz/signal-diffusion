# Dataset link: https://openneuro.org/datasets/ds004584/versions/1.0.0

# Datasets for EEG classification
import csv
import glob
import mne
import numpy as np
import os
import pandas as pd
import re
import torch
from bidict import bidict
from collections import OrderedDict
from PIL import Image
from scipy.signal import decimate
from torchvision import transforms
from torch.utils.data import WeightedRandomSampler
from os.path import join as pjoin
import bisect
import shutil
import math
from tqdm.auto import tqdm


# import support scripts: pull_data
import common.multichannel_spectrograms as mcs
from data_processing.channel_map import parkinsons_channels

mne.set_log_level("WARNING")

parkinsons_class_labels = bidict({
    0: "healthy_male",
    1: "healthy_female",
    2: "parkinsons_male",
    3: "parkinsons_female",
})


class ParkinsonsPreprocessor():
    # Originally 250, changed to 125
    def __init__(self, datadir, nsamps, ovr_perc=0, fs=250):
        self.datadir = datadir

        # Establish sampling constants
        orig_fs = 500
        self.decimation = orig_fs // fs
        print("Decimation factor {} new fs {}".format(self.decimation, orig_fs / self.decimation))
        self.fs = orig_fs / self.decimation
        self.nsamps = nsamps
        print("Decimation factor {} new number of samples {}".format(self.decimation, self.nsamps))
        self.ovr_perc = ovr_perc
        self.noverlap = int(np.floor(nsamps * ovr_perc))
        

        # Get subject info
        self.subjects = pd.read_csv(os.path.join(datadir, "participants.tsv"), sep="\t")
        self.n_channels = len(parkinsons_channels)


    def get_num_channels():
        return self.n_channels

    def get_gender(self, sd):
        sub_id = int(sd.split('-')[1]) - 1
        return self.subjects.iloc[sub_id].GENDER

    def get_health(self, sd):
        sub_id = int(sd.split('-')[1]) - 1
        return self.subjects.iloc[sub_id].GROUP

    def get_age(self, sd):
        sub_id = int(sd.split('-')[1]) - 1
        return self.subjects.iloc[sub_id].AGE

    @staticmethod
    def get_caption(gender, health, age):
        gender = "male" if gender == "M" else "female"
        health = "parkinsons disease diagnosed" if health == "PD" else "healthy"
        return f"an EEG spectrogram of a {age} year old, {health}, {gender} subject"

    def decimate(self, data):
        if self.decimation > 1:
            data = decimate(data, self.decimation, axis=1, zero_phase=True)
        return data

    def _generate_spectrograms(self, subject_dirs, outdir, resolution, hop_length):
        # Containers for sample metadata
        files = []
        genders = []
        ages = []
        pds = []
        # Load data file for each subject
        total_specs = 0
        for sd in tqdm(subject_dirs):
            setfile = glob.glob(pjoin(self.datadir, sd, "eeg", "*eeg.set"))[0]
            data = mne.io.read_raw_eeglab(setfile)
            data = data.get_data()
            # Get subset of channels we want
            chan_inds = [ch[1] for ch in parkinsons_channels]
            data = data[chan_inds, :]
            # Decimate data
            data = self.decimate(data)
            # Break data into chunks and save
            os.makedirs(pjoin(outdir, sd), exist_ok=True)
            N = data.shape[1]                
            shift_size = self.nsamps - self.noverlap

            if self.noverlap != 0:
                nblocks = math.floor((N - self.nsamps) / shift_size) + 1
            else:
                nblocks = math.floor(N / self.nsamps)
            assert nblocks > 1, (
                "File {} (T={}) is too short to be used with nsamps {} and decimation {}".format(
                    setfile, N, self.nsamps, self.decimation
                )
            )  
            total_specs += nblocks

            start_ind = 0
            end_ind = self.nsamps
            i = 0
            while end_ind <= N:
                blk = data[:, start_ind: end_ind]
                start_ind += shift_size
                end_ind += shift_size

                S = mcs.multichannel_spectrogram(
                    blk,
                    hop_length=hop_length,
                    resolution=resolution, win_length=resolution,
                )
                fname = pjoin(sd, f"spectrogram-{i}.png")
                files.append(fname)
                genders.append(self.get_gender(sd))
                ages.append(self.get_age(sd))
                pds.append(self.get_health(sd))
                S.save(pjoin(outdir, fname))
                i += 1

        assert total_specs == len(files), (
                    "{} spectrograms were generated, {} should've been made".format(
                        len(files), total_specs
                    )
                )
        return files, genders, ages, pds

    def make_tvt_splits(self, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=None):
        if seed is not None:
            np.random.seed(seed)
        # Get shuffled metadata
        metadata = pd.read_csv(pjoin(self.datadir, "stfts", "metadata.csv"))
        # Make individual age/gender/health columns
        metadata["gender"] = metadata["text"].str.contains("female")
        metadata["gender"] = metadata["gender"].map({True: "F", False: "M"})
        metadata["age"] = metadata["text"].apply(lambda s: int(re.search(r"\d+", s).group(0)))
        metadata["health"] = metadata["text"].str.contains("healthy")
        metadata["health"] = metadata["health"].map({True: "H", False: "PD"})
        # Empty per-split metadata dfs
        trainmeta = metadata.iloc[0:0]
        valmeta = metadata.iloc[0:0]
        testmeta = metadata.iloc[0:0]
        # Group by class
        gmeta = metadata.groupby(["gender", "health"])
        for group, gdf in gmeta:
            subjects = gdf["file_name"].apply(lambda s: s.split("/")[0])
            subjects = subjects.unique()
            np.random.shuffle(subjects)
            # Create train/val/test splits by subject to avoid data leakage
            ntrain = int(np.ceil(train_frac * len(subjects)))
            nval = int(np.floor(val_frac * len(subjects)))
            trainsubs = subjects[:ntrain]
            valsubs = subjects[ntrain:ntrain + nval]
            testsubs = subjects[ntrain + nval:]
            # Add splits to metadata
            trainmeta = pd.concat([trainmeta, gdf.loc[gdf["file_name"].apply(lambda s: s.split("/")[0]).isin(trainsubs)]])
            valmeta = pd.concat([valmeta, gdf.loc[gdf["file_name"].apply(lambda s: s.split("/")[0]).isin(valsubs)]])
            testmeta = pd.concat([testmeta, gdf.loc[gdf["file_name"].apply(lambda s: s.split("/")[0]).isin(testsubs)]])
        trainmeta.to_csv(pjoin(self.datadir, "stfts", "train-metadata.csv"), index=False)
        valmeta.to_csv(pjoin(self.datadir, "stfts", "val-metadata.csv"), index=False)
        testmeta.to_csv(pjoin(self.datadir, "stfts", "test-metadata.csv"), index=False)

    def preprocess(self, resolution=512, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=None):
        # Make output dir
        outdir = pjoin(self.datadir, "stfts")
         # Delete any premade stfts
        if os.path.isdir(outdir):
            shutil.rmtree(outdir)
        # Create new stft output directory
        os.makedirs(outdir, exist_ok=True)
        # Spectrogram parameters
        max_bins = math.floor(resolution / self.n_channels)
        hop_length = 8  # number of samples per time-step in spectrogram
        while self.nsamps / hop_length > max_bins:
            hop_length += 8
        # Get list of data directories
        subject_dirs = os.listdir(self.datadir)
        subject_dirs = list(filter(lambda d: os.path.isdir(pjoin(self.datadir, d)), subject_dirs))
        subject_dirs = list(filter(lambda d: d.startswith("sub"), subject_dirs))
        # Create spectrograms
        files, genders, ages, pds = self._generate_spectrograms(subject_dirs, outdir, resolution, hop_length)
        # Write metadata to file
        with open(pjoin(outdir, "metadata.csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerow(["file_name", "text"])
            for fn, g, a, p in zip(files, genders, ages, pds):
                caption = ParkinsonsPreprocessor.get_caption(g, p, a)
                writer.writerow([fn, caption])
        # Make splits
        self.make_tvt_splits(train_frac, val_frac, test_frac, seed)


class ParkinsonsDataset(torch.utils.data.Dataset):
    def __init__(self, datadir, split="train", transform=None):
        self.dataname = 'parkinsons'
        self.datadir = datadir
        self.split = split
        assert os.path.isfile(pjoin(datadir, f"{split}-metadata.csv")), "No metadata file found for split {}".format(split)
        self.metadata = pd.read_csv(pjoin(datadir, f"{split}-metadata.csv"))
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        fn = self.metadata.iloc[index]["file_name"]
        im = Image.open(pjoin(self.datadir, fn))
        im = self.transform(im)
        health = self.metadata.iloc[index]["health"]
        gender = self.metadata.iloc[index]["gender"]
        y = (health == "PD") * 2 + (gender == "F")

        return im, y % 2

    def caption(self, index):
        return self.metadata.iloc[index]["text"]

class ParkinsonsSampler(torch.utils.data.Sampler):
    def __init__(self, datadir, num_samples, split="train", replacement=True, generator=None):
        assert os.path.isfile(pjoin(datadir, f"{split}-metadata.csv")), "No metadata file found for split {}".format(split)
        self.metadata = pd.read_csv(pjoin(datadir, f"{split}-metadata.csv"))
        self.weights = torch.as_tensor(self.generate_weights(self.metadata), dtype=torch.double)
        self.num_samples = num_samples # Number of samples to draw not total
        self.split = split
        self.replacement = replacement
        self.generator = generator

    def __len__(self):
        return len(self.metadata)

    def __iter__(self):
        rand_tensor = torch.multinomial(self.weights, len(self.metadata), self.replacement, generator=self.generator)
        yield from iter(rand_tensor.tolist())

    def generate_weights(self, metadata):
        Y = []
        for i in range(len(metadata)):
            health = self.metadata.iloc[i]["health"]
            gender = self.metadata.iloc[i]["gender"]
            Y.append((health == "PD") * 2 + (gender == "F"))
        # Get the current weights of each class
        label_weights = [Y.count(i)/len(metadata) for i in range(4)]

        # Class rankings
        rankings = {}
        for label in range(4):
            label_weight = label_weights[label]
            rank = 0
            for weight in label_weights:
                if label_weight > weight:
                    rank += 1
            rankings[label] = rank

        # Flip weights so smaller classes are more prominent
        label_weights.sort(reverse=True)
        new_label_weights = [label_weights[rankings[i]] for i in range(4)]
        output_weights = [new_label_weights[i] for i in Y]

        # Normalize output weights
        norm_fact = sum(output_weights)
        output_weights = [weights / norm_fact for weights in output_weights]

        return output_weights
