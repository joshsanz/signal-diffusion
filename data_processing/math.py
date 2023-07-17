# MATH dataset link: https://physionet.org/content/eegmat/1.0.0/

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
from data_processing.channel_map import math_channels

mne.set_log_level("WARNING")


math_class_labels = bidict({
    0: "bkgnd_male",
    1: "bkgnd_female",
    2: "math_male",
    3: "math_female",
})


class MathPreprocessor():
    def __init__(self, datadir, nsamps, ovr_perc=0, fs=250):
        # Establish directories
        self.datadir = datadir
        self.eegdir = os.path.join(self.datadir, "raw_eeg")
        self.stfttdir = os.path.join(self.datadir, "stfts")

        # Establish sampling constants
        orig_fs = 500
        self.decimation = orig_fs // fs
        print("Decimation factor {} new fs {}".format(self.decimation, orig_fs / self.decimation))
        self.fs = orig_fs / self.decimation  # To compensate for decimation fact being an int
        self.nsamps = nsamps
        print("Decimation factor {} new number of samples {}".format(self.decimation, self.nsamps))
        self.ovr_perc = ovr_perc
        self.noverlap = int(np.floor(nsamps * ovr_perc))

        self.subjects = pd.read_csv(os.path.join(self.eegdir, "subject-info.csv"))
        # Get dataset descriptions
        self.files = list(filter(lambda f: f.endswith(".edf"), os.listdir(self.eegdir)))
        self.files.sort()
        assert len(self.files) == 72, "Expected 72 files, found {}".format(len(self.files))
        chans = mne.io.read_raw_edf(os.path.join(self.eegdir, self.files[0]), preload=False).ch_names
        self.n_channels = len(chans) - 1  # exclude EKG channel

    def get_label(self, file):
        info, _ = os.path.splitext(file)
        subject, m_or_bk = info.split("_")
        # age = self.subject_info[subject][0]
        index = int(subject[len(subject) - 2:])
        age = self.subjects.iloc[index][1]
        gender = self.subjects.iloc[index][2]
        isfemale = gender == "F"
        # subtractions = self.subject_info[subject][3]
        doing_math = m_or_bk == "2"
        label = torch.tensor(2 * doing_math + 1 * isfemale).long()
        return gender, doing_math, age, label

    @staticmethod
    def get_caption(gender, doingmath, age):
        gender = "male" if gender == "M" else "female"
        math = "doing math" if doingmath == 1 else "background"
        return f"an EEG spectrogram of a {age} year old, {math}, {gender} subject"

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
        subject_dirs = os.listdir(self.eegdir)
        subject_dirs = list(filter(lambda d: d.startswith("Subject"), subject_dirs))
        subject_dirs = np.unique([sub[:len(sub) - 6] for sub in subject_dirs])
        subject_dirs.sort()

        # Create spectrograms
        files, genders, maths, ages, labels = self._generate_spectrograms(subject_dirs, outdir, resolution, hop_length)
        # Write metadata to file
        with open(pjoin(outdir, "metadata.csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerow(["file_name", "text", "gender", "doingmath", "age"])
            for fn, g, m, a in zip(files, genders, maths, ages):
                caption = MathPreprocessor.get_caption(g, m, a)
                writer.writerow([fn, caption, g, m, a])
        # Make splits
        self.make_tvt_splits(train_frac, val_frac, test_frac, seed)

    def _generate_spectrograms(self, subject_dirs, outdir, resolution, hop_length):
        # Containers for sample metadata
        files = []
        genders = []
        maths = []
        ages = []
        labels = []

        total_specs = 0
        for sd in tqdm(subject_dirs):
            # Do one pass to get bkgrnd files and another for math, 1 = bkgnd, 2 = math
            i = 0
            for state in range(1,2): # To include math set range to (1,3)
                sub_file = sd + "_" + str(state) + ".edf"
                sub_dir = os.path.join(self.eegdir, sub_file)
                data = mne.io.read_raw_edf(sub_dir, preload=True)
                # Remove the EKG channel & decimate
                data, _ = data[:-1]
                # Decimate data
                data = self.decimate(data)
                # Break data into chunks and save
                sub_sd = "sub" + sd[len(sd)-2:]
                os.makedirs(pjoin(outdir, sub_sd), exist_ok=True)
                N = data.shape[1]
                shift_size = self.nsamps - self.noverlap
                # Conirm the spectrogram is long enough to split up
                if self.noverlap != 0:
                    nblocks = math.floor((N - self.nsamps) / shift_size) + 1
                else:
                    nblocks = math.floor(N / self.nsamps)
                assert nblocks > 1, (
                    "File {} (T={}) is too short to be used with nsamps {} and decimation {}".format(
                        sub_file, N, self.nsamps, self.decimation
                    )
                )
                total_specs += nblocks
                # Generate spectrograms
                start_ind = 0
                end_ind = self.nsamps
                while end_ind <= N:
                    start_ind, end_ind = int(math.floor(start_ind)), int(math.floor(end_ind))
                    blk = data[:, start_ind: end_ind]
                    start_ind += shift_size
                    end_ind += shift_size

                    S = mcs.multichannel_spectrogram(
                        blk,
                        hop_length=hop_length,
                        resolution=resolution, win_length=resolution,
                    )
                    fname = pjoin(sub_sd, f"spectrogram-{i}.png")
                    files.append(fname)
                    isfemale, doing_math, age, label = self.get_label(sub_file)
                    genders.append(isfemale)
                    maths.append(doing_math)
                    ages. append(age)
                    labels.append(label)
                    S.save(pjoin(outdir, fname))
                    i += 1

        assert total_specs == len(files), (
                    "{} spectrograms where generated, {} should've been made".format(
                        len(files), total_specs
                    )
                )

        return files , genders , maths, ages, labels

    def make_tvt_splits(self, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=None):
        if seed is not None:
            np.random.seed(seed)
        # Get shuffled metadata
        metadata = pd.read_csv(pjoin(self.datadir, "stfts", "metadata.csv"))
        # Empty per-split metadata dfs
        trainmeta = metadata.iloc[0:0]
        valmeta = metadata.iloc[0:0]
        testmeta = metadata.iloc[0:0]
        # Group by subject
        N = len(metadata)
        n_train = int(np.ceil(train_frac * N))
        n_val = int(np.floor(val_frac * N))
        n_test = N - (n_train + n_val)
        subjects = metadata.loc[:,"file_name"].apply(lambda s: s.split("/")[0])
        subjects = subjects.unique()
        np.random.shuffle(subjects)

        for sub in subjects:
            addition = metadata.loc[metadata["file_name"].apply(lambda s: s.split("/")[0] == sub)]

            # print("SUBS: ", sub)
            # print(addition_ind)
            # print('TRUTH: ', True in addition_ind)
            # print(metadata.loc[addition_ind])
            # print(len(metadata.loc[addition_ind]))

            if len(trainmeta) < n_train:
                trainmeta = pd.concat([trainmeta, addition])
            elif len(valmeta) < n_val:
                valmeta = pd.concat([valmeta, addition])
            else:
                testmeta = pd.concat([testmeta, addition])

        trainmeta.to_csv(pjoin(self.datadir, "stfts", "train-metadata.csv"), index=False)
        valmeta.to_csv(pjoin(self.datadir, "stfts", "val-metadata.csv"), index=False)
        testmeta.to_csv(pjoin(self.datadir, "stfts", "test-metadata.csv"), index=False)

    def decimate(self, data):
        if self.decimation > 1:
            data = decimate(data, self.decimation, axis=1, zero_phase=True)
        return data

    def get_subject(self, index):
        if index < len(self.ind_math_files):
            file = self.ind_math_files[index]
        else:
            file = self.ind_bkgnd_files[index - len(self.ind_math_files)]
        subject = file.split("_")[0]
        return subject

    def get_subject_info(self, subject):
        age = self.subject_info[subject][0]
        gender = self.subject_info[subject][1]
        subtractions = self.subject_info[subject][3]
        return {"age": age, "gender": gender, "subtractions": subtractions}


class MathDataset(torch.utils.data.Dataset):
    def __init__(self, datadir, split="train", transform=None):
        self.dataname = 'math'
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
        doing_math = self.metadata.iloc[index]["doingmath"]
        gender = self.metadata.iloc[index]["gender"]
        isfemale = gender == "F"
        y = int(2 * doing_math + 1 * isfemale)

        return im, y % 2

    def caption(self, index):
        return self.metadata.iloc[index]["text"]
