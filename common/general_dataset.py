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
import math

# import support scripts: pull_data
import common.ear_eeg_support_scripts.read_in_ear_eeg as read_in_ear_eeg
import common.ear_eeg_support_scripts.read_in_labels as read_in_labels
import common.ear_eeg_support_scripts.eeg_filter as eeg_filter
import common.multichannel_spectrograms as mcs
from common.channel_map import parkinsons_channels, seed_channels

mne.set_log_level("WARNING")

ear_class_labels = bidict({0: "awake", 1: "sleepy"})

math_class_labels = bidict({
    0: "bkgnd_male",
    1: "bkgnd_female",
    2: "math_male",
    3: "math_female",
})

parkinsons_class_labels = bidict({
    0: "healthy_male",
    1: "healthy_female",
    2: "parkinsons_male",
    3: "parkinsons_female",
})

emotion_map = bidict({
    "disgust": 0,
    "fear": 1,
    "sad": 2,
    "neutral": 3,
    "happy": 4
})
seed_class_labels = bidict({
    0: f"{emotion_map.inverse[0]}_male",
    1: f"{emotion_map.inverse[0]}_female",
    2: f"{emotion_map.inverse[1]}_male",
    3: f"{emotion_map.inverse[1]}_female",
    4: f"{emotion_map.inverse[2]}_male",
    5: f"{emotion_map.inverse[2]}_female",
    6: f"{emotion_map.inverse[3]}_male",
    7: f"{emotion_map.inverse[3]}_female",
    8: f"{emotion_map.inverse[4]}_male",
    9: f"{emotion_map.inverse[4]}_female",
})

general_class_labels = bidict({
    0: "male",
    1: "femal",
})


class CacheDict(OrderedDict):
    """Dict with a limited length, ejecting LRUs as needed."""

    def __init__(self, *args, cache_len: int = 10, **kwargs):
        assert cache_len > 0
        self.cache_len = cache_len
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().move_to_end(key)

        while len(self) > self.cache_len:
            oldkey = next(iter(self))
            super().__delitem__(oldkey)

    def __getitem__(self, key):
        val = super().__getitem__(key)
        super().move_to_end(key)

        return val

class RawMathDataset(torch.utils.data.Dataset):
    """Dataset loader for Math EEG dataset"""
    def __init__(self, data_path, n_samples, n_context, fs=250, math_only=False, background_only=False):
        """
        Construct a parameterized dataloader for math eeg data

        # Parameters
        - data_path: path to directory containing math eeg data EDF files and subject-info.csv metadata
        - n_samples: total number of time samples in one data point
        - n_context: number of time samples in the context window per-channel for a single token
        - math_only: if True, only load recordings when subjects were doing math
        - background_only: if True, only load recordings when subjects were not doing math

        # Returns
        For index i, returns a tuple (X, y) where
        - X is a tensor of shape (n_tokens x n_channels*n_context), for n_tokens = n_samples // n_context
        - y is a tensor of size 1, where y = 2 * (doing_math) + 1 * (gender_is_male), i.e.
        | y | math? | gender |
        |---|-------|--------|
        | 0 |   0   |   0    |
        | 1 |   0   |   1    |
        | 2 |   1   |   0    |
        | 3 |   1   |   1    |
        """
        assert not (math_only and background_only), "Can't have both math_only and background_only"
        assert n_samples >= n_context and n_samples % n_context == 0, "n_samples must be divisible by n_context"
        self.data_path = data_path
        self.n_samples = n_samples
        self.n_context = n_context
        self.fs = fs
        # TODO: implement decimation or preprocessing
        self.decimation = 500 // fs  # Raw samples are 500 Hz
        self.math_only = math_only
        self.background_only = background_only

        # Get dataset descriptions
        self.files = list(filter(lambda f: f.endswith(".edf"), os.listdir(self.data_path)))
        assert len(self.files) == 72, "Expected 72 files, found {}".format(len(self.files))
        self.math_files = list(filter(lambda f: f.endswith("_2.edf"), self.files))
        self.bkgnd_files = list(filter(lambda f: f.endswith("_1.edf"), self.files))

        self.subject_info = {}
        with open(os.path.join(self.data_path, "subject-info.csv"), "r") as fcsv:
            reader = csv.reader(fcsv)
            _ = next(reader)  # skip header
            for row in reader:
                self.subject_info[row[0]] = int(row[1]), row[2], int(row[3]), float(row[4]), bool(row[5])

        # Get file content info
        self.math_file_lens = [mne.io.read_raw_edf(os.path.join(self.data_path, f), preload=False).n_times
                               for f in self.math_files]
        self.bkgnd_file_lens = [mne.io.read_raw_edf(os.path.join(self.data_path, f), preload=False).n_times
                                for f in self.bkgnd_files]
        chans = mne.io.read_raw_edf(os.path.join(self.data_path, self.files[0]), preload=False).ch_names
        self.n_channels = len(chans) - 1  # exclude EKG channel
        # Define index to file & offset mapping
        self.ind_math_files = []
        self.ind_math_offsets = []
        for f, L in zip(self.math_files, self.math_file_lens):
            nchunks = L // n_samples
            self.ind_math_files.extend([f] * nchunks)
            self.ind_math_offsets.extend([n_samples * i for i in range(nchunks)])
        self.ind_bkgnd_files = []
        self.ind_bkgnd_offsets = []
        for f, L in zip(self.bkgnd_files, self.bkgnd_file_lens):
            nchunks = L // n_samples
            self.ind_bkgnd_files.extend([f] * nchunks)
            self.ind_bkgnd_offsets.extend([n_samples * i for i in range(nchunks)])

        # Caching
        self.cache = CacheDict(cache_len=1000)

    def __len__(self):
        if self.math_only:
            return len(self.ind_math_files)
        elif self.background_only:
            return len(self.ind_bkgnd_files)
        else:
            return len(self.ind_math_files) + len(self.ind_bkgnd_files)

    def format_data(self, data):
        # Data comes in as n_channels x n_samples
        data = torch.tensor(data).float().reshape(self.n_channels, -1, self.n_context)
        data = data.permute(1, 0, 2).reshape(-1, self.n_channels * self.n_context)
        return data.contiguous()

    def get_label(self, file):
        info, _ = os.path.splitext(file)
        subject, m_or_bk = info.split("_")
        # age = self.subject_info[subject][0]
        gender = self.subject_info[subject][1]
        isfemale = gender == "F"
        # subtractions = self.subject_info[subject][3]
        doing_math = m_or_bk == "2"
        return torch.tensor(2 * doing_math + 1 * isfemale).long()

    def load_data(self, index):
        if self.math_only:
            file = self.ind_math_files[index]
            offset = self.ind_math_offsets[index]
        elif self.background_only:
            file = self.ind_math_files[index]
            offset = self.ind_math_offsets[index]
        else:
            if index < len(self.ind_math_files):
                file = self.ind_math_files[index]
                offset = self.ind_math_offsets[index]
            else:
                file = self.ind_bkgnd_files[index - len(self.ind_math_files)]
                offset = self.ind_bkgnd_offsets[index - len(self.ind_math_files)]
        edf = mne.io.read_raw_edf(os.path.join(self.data_path, file), preload=True)
        # Ignore last channel, which is EKG data
        data, _ = edf[:-1, offset:offset + self.n_samples]
        X = self.format_data(data)
        y = self.get_label(file)
        return X, y

    def __getitem__(self, index):
        val = self.cache.get(index, None)
        if val is None:
            val = self.load_data(index)
            self.cache[index] = val
        X, y = val
        return X, y

class MathPreprocessor:
    def __init__(self, data_path, nsamps, ovr_perc=0, fs=250):
        self.data_path = data_path
        self.nsamps = nsamps
        self.fs = fs
        orig_fs = 500
        self.decimation = orig_fs // fs
        print("Decimation factor: {}".format(self.decimation))

        # Get dataset descriptions
        self.files = list(filter(lambda f: f.endswith(".edf"), os.listdir(self.data_path)))
        assert len(self.files) == 72, "Expected 72 files, found {}".format(len(self.files))
        self.math_files = list(filter(lambda f: f.endswith("_2.edf"), self.files))
        self.bkgnd_files = list(filter(lambda f: f.endswith("_1.edf"), self.files))

        self.subject_info = {}
        with open(os.path.join(self.data_path, "subject-info.csv"), "r") as fcsv:
            reader = csv.reader(fcsv)
            _ = next(reader)  # skip header
            for row in reader:
                self.subject_info[row[0]] = int(row[1]), row[2], int(row[3]), float(row[4]), bool(row[5])

        # Get file content info
        self.math_file_lens = [mne.io.read_raw_edf(os.path.join(self.data_path, f), preload=False).n_times
                               for f in self.math_files]
        self.bkgnd_file_lens = [mne.io.read_raw_edf(os.path.join(self.data_path, f), preload=False).n_times
                                for f in self.bkgnd_files]
        chans = mne.io.read_raw_edf(os.path.join(self.data_path, self.files[0]), preload=False).ch_names
        self.n_channels = len(chans) - 1  # exclude EKG channel
        # Define index to file & offset mapping
        self.ind_math_files = []
        self.ind_math_offsets = []
        for f, L in zip(self.math_files, self.math_file_lens):
            nchunks = L // (self.decimation * self.nsamps)
            assert nchunks > 1, (
                "File {} (T={}) is too short to be used with nsamps {} and decimation {}".format(
                    f, L, self.nsamps, self.decimation
                )
            )
            self.ind_math_files.extend([f] * nchunks)
            self.ind_math_offsets.extend([self.nsamps * self.decimation * i for i in range(nchunks)])
        self.ind_bkgnd_files = []
        self.ind_bkgnd_offsets = []
        for f, L in zip(self.bkgnd_files, self.bkgnd_file_lens):
            nchunks = L // (self.decimation * self.nsamps)
            assert nchunks > 1, (
                "File {} (T={}) is too short to be used with nsamps {} and decimation {}".format(
                    f, L, self.nsamps, self.decimation
                )
            )
            self.ind_bkgnd_files.extend([f] * nchunks)
            self.ind_bkgnd_offsets.extend([self.nsamps * self.decimation * i for i in range(nchunks)])

    def __len__(self):
        return len(self.ind_math_files) + len(self.ind_bkgnd_files)

    def get_label(self, file):
        """
        | y | math? | gender |
        |---|-------|--------|
        | 0 |   0   |   0    |
        | 1 |   0   |   1    |
        | 2 |   1   |   0    |
        | 3 |   1   |   1    |
        """
        info, _ = os.path.splitext(file)
        subject, m_or_bk = info.split("_")
        # age = self.subject_info[subject][0]
        gender = self.subject_info[subject][1]
        isfemale = gender == "F"
        # subtractions = self.subject_info[subject][3]
        doing_math = m_or_bk == "2"
        return torch.tensor(2 * doing_math + 1 * isfemale).long()

    def load_data(self, index):
        if index < len(self.ind_math_files):
            file = self.ind_math_files[index]
            offset = self.ind_math_offsets[index]
        else:
            file = self.ind_bkgnd_files[index - len(self.ind_math_files)]
            offset = self.ind_bkgnd_offsets[index - len(self.ind_math_files)]
        edf = mne.io.read_raw_edf(os.path.join(self.data_path, file), preload=True)
        # Ignore last channel, which is EKG data
        data, _ = edf[:-1, offset:offset + self.nsamps * self.decimation]
        if self.decimation > 1:
            data = decimate(data, self.decimation, axis=1).copy()
        X = torch.tensor(data).float()
        y = self.get_label(file)
        return X, y

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

    def preprocess(self, inds_per_file=20, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=42):
        output_dir = {'train': os.path.join(self.data_path, "train"),
                      'val': os.path.join(self.data_path, "val"),
                      'test': os.path.join(self.data_path, "test")}
        for d in output_dir.values():
            os.makedirs(d, exist_ok=True)
        assert train_frac + val_frac + test_frac == 1, "train_frac + val_frac + test_frac must equal 1"
        N = len(self)
        n_train = int(np.ceil(train_frac * N))
        n_val = int(np.floor(val_frac * N))
        # n_test = int(np.floor(test_frac * N))
        shuffled_inds = np.arange(N)
        np.random.shuffle(shuffled_inds)
        train_inds = shuffled_inds[:n_train]
        val_inds = shuffled_inds[n_train:n_train + n_val]
        test_inds = shuffled_inds[n_train + n_val:]

        for split, inds in zip(['train', 'val', 'test'], [train_inds, val_inds, test_inds]):
            subindex = 0
            file_ind = 0
            filenames, offsets, nsamples, subjects = [], [], [], []
            while subindex < len(inds):
                offsets.append(subindex)
                X, y = [], []
                for i in range(inds_per_file):
                    subjects.append(self.get_subject(inds[subindex]))
                    iX, iy = self.load_data(inds[subindex])
                    subindex += 1
                    X.append(iX)
                    y.append(iy)
                    if subindex == len(inds):
                        break
                X = torch.stack(X)
                y = torch.stack(y)
                torch.save(X, os.path.join(output_dir[split], "X_{}.pt".format(file_ind)))
                torch.save(y, os.path.join(output_dir[split], "y_{}.pt".format(file_ind)))
                filenames.append("X_{}.pt,y_{}.pt".format(file_ind, file_ind))
                nsamples.append(X.shape[0])
                file_ind += 1
            with open(os.path.join(self.data_path, split, "metadata.csv"), "w") as f:
                writer = csv.writer(f)
                writer.writerow(["X_filename", "y_filename", "start_sample_offset", "n_samples"])
                for fn, off, ns in zip(filenames, offsets, nsamples):
                    writer.writerow(fn.split(',') + [off, ns])
            with open(os.path.join(self.data_path, split, "subjects.csv"), "w") as f:
                writer = csv.writer(f)
                writer.writerow(["sample_index", "subject"])
                for i, s in enumerate(subjects):
                    writer.writerow([i, s])

        return output_dir


class MathDataset(torch.utils.data.Dataset):
    """Dataset for preprocessed math eeg dataset"""
    def __init__(self, data_path, ncontext):
        self.data_path = data_path
        self.ncontext = ncontext
        self.metadata = pd.read_csv(os.path.join(data_path, "metadata.csv"))
        self.subjects = pd.read_csv(os.path.join(data_path, "subjects.csv"))
        self.cache = CacheDict(cache_len=1000)
        # Get shape info
        assert len(self.metadata) > 0, "No data found in {}".format(data_path)
        X = self._get_file(self.metadata.iloc[0]["X_filename"])
        self.n_channels = X.shape[1]
        self.nsamps = X.shape[2]
        assert self.nsamps >= self.ncontext and self.nsamps % self.ncontext == 0, (
            f"Incompatible nsamps {self.nsamps} and ncontext {self.ncontext}"
        )

    def __len__(self):
        return self.metadata.iloc[-1]["start_sample_offset"] + self.metadata.iloc[-1]["n_samples"]

    def _get_file(self, filename):
        if filename in self.cache:
            return self.cache[filename]
        else:
            X = torch.load(os.path.join(self.data_path, filename))
            self.cache[filename] = X
            return X

    def __getitem__(self, index):
        # Get X and y
        metaind = self.metadata["start_sample_offset"].searchsorted(index, side="right") - 1
        Xfile = self.metadata.iloc[metaind]["X_filename"]
        yfile = self.metadata.iloc[metaind]["y_filename"]
        multiX = self._get_file(Xfile)
        multiy = self._get_file(yfile)
        offset = index - self.metadata.iloc[metaind]["start_sample_offset"]
        X = multiX[offset, :, :]
        y = multiy[offset]
        # Format X
        X = X.reshape(-1, self.nsamps // self.ncontext, self.ncontext)
        X = X.permute(1, 2, 0).reshape(self.nsamps // self.ncontext, -1).contiguous()
        return X, y

    def get_subject(self, index):
        return self.subjects.iloc[index].subject

class MathSpectrumDataset(MathDataset):
    def __init__(self, data_path, resolution=256, hop_length=192, transform=None):
        super().__init__(data_path, 1)
        self.resolution = resolution
        self.hop_length = hop_length
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        self.transform = transform

    def __getitem__(self, index):
        # Get X and y
        metaind = self.metadata["start_sample_offset"].searchsorted(index, side="right") - 1
        Xfile = self.metadata.iloc[metaind]["X_filename"]
        yfile = self.metadata.iloc[metaind]["y_filename"]
        multiX = self._get_file(Xfile)
        multiy = self._get_file(yfile)
        offset = index - self.metadata.iloc[metaind]["start_sample_offset"]
        X = multiX[offset, :, :]
        y = multiy[offset]
        # Make spectrogram
        S = mcs.multichannel_spectrogram(X.numpy(), self.resolution, self.hop_length, self.resolution)
        S = self.transform(S)
        return S, int(y) % 2 # Turn this to an int so it matches Parkinsons dataset

class GeneratedSpectrumDataset(torch.utils.data.Dataset):
    """Dataset class for Diffusion generated spectrograms"""
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        self.transform = transform
        self.cache = CacheDict(cache_len=10)
        self.metadata = pd.read_csv(os.path.join(data_path, "metadata.csv"))

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        file = self.metadata.iloc[index]["file"]
        y = self.metadata.iloc[index]["y"]
        if file in self.cache:
            S = self.cache[file]
        else:
            im = Image.open(os.path.join(self.data_path, file))
            S = self.transform(im)
            self.cache[file] = S
        return S, y 


class ParkinsonsPreprocessor():
    # Originally 250, changed to 125
    def __init__(self, datadir, nsamps, ovr_perc=0, fs=250):
        self.datadir = datadir
        self.nsamps = nsamps
        self.ovr_perc = ovr_perc
        self.noverlaps = nsamps * ovr_perc

        orig_fs = 500
        self.decimation = orig_fs // fs
        self.fs = orig_fs // self.decimation
        print("Decimation factor {} new fs {}".format(self.decimation, orig_fs / self.decimation))
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
        for sd in subject_dirs:
            setfile = glob.glob(pjoin(self.datadir, sd, "eeg", "*eeg.set"))[0]
            data = mne.io.read_raw_eeglab(setfile)
            data = data.get_data()
            # Get subset of channels we want
            chan_inds = [ch[1] for ch in parkinsons_channels]
            data = data[chan_inds, :]
            data = self.decimate(data)

            # decimate data
            data = decimate(data, int(self.decimation), axis=1, zero_phase=True)
            N = data.shape[1]
            nblocks = N // self.nsamps
            nblocks = 2
            shift_size = self.nsamps - self.noverlaps
        
            # Break data into chunks and save
            os.makedirs(pjoin(outdir, sd), exist_ok=True)

            start_ind = 0
            end_ind = self.nsamps
            i = 0
            while end_ind < N:
                start_ind = int(math.floor(start_ind))
                end_ind = int(math.floor(end_ind))
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
        self.datadir = datadir
        self.split = split
        assert os.path.isfile(pjoin(datadir, f"{split}-metadata.csv")), "No metadata file found for split {}".format(split)
        self.metadata = pd.read_csv(pjoin(datadir, f"{split}-metadata.csv"))
        print(self.split, " metadata len: ", len(self.metadata))
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


class SEEDPreprocessor():
    start_second = {
        0: [30, 132, 287, 555, 773, 982, 1271, 1628, 1730, 2025, 2227, 2435, 2667, 2932, 3204],
        1: [30, 299, 548, 646, 836, 1000, 1091, 1392, 1657, 1809, 1966, 2186, 2333, 2490, 2741],
        2: [30, 353, 478, 674, 825, 908, 1200, 1346, 1451, 1711, 2055, 2307, 2457, 2726, 2888]
    }
    end_second = {
        0: [102, 228, 524, 742, 920, 1240, 1568, 1697, 1994, 2166, 2401, 2607, 2901, 3172, 3359],
        1: [267, 488, 614, 773, 967, 1059, 1331, 1622, 1777, 1908, 2153, 2302, 2428, 2709, 2817],
        2: [321, 418, 643, 764, 877, 1147, 1284, 1418, 1679, 1996, 2275, 2425, 2664, 2857, 3066]
    }
    emotion_map = bidict({
        "disgust": 0,
        "fear": 1,
        "sad": 2,
        "neutral": 3,
        "happy": 4
    })
    trial_emotion = {
        0: [4, 1, 3, 2, 0, 4, 1, 3, 2, 0, 4, 1, 3, 2, 0],
        1: [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0],
        2: [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0],
    }

    def __init__(self, datadir, nsamps, fs=250):
        self.datadir = datadir
        self.nsamps = nsamps
        self.orig_fs = 1000
        self.decimation = self.orig_fs // fs
        self.fs = self.orig_fs // self.decimation
        print("Decimation factor {} new fs {}".format(self.decimation, self.orig_fs / self.decimation))
        self.subjects = pd.read_csv(os.path.join(datadir, "participants_info.csv"))
        self.n_channels = len(seed_channels)
        self.n_sessions = 3
        self.n_trials = len(self.start_second[0])

    def get_gender(self, subject):
        return self.subjects.iloc[subject].Sex

    def get_age(self, subject):
        return self.subjects.iloc[subject].Age

    def get_emotion(self, session, trial, text=False):
        emotion = self.trial_emotion[session][trial]
        if text:
            return self.emotion_map.inverse[emotion]
        else:
            return emotion

    def get_caption(self, subject, session, trial):
        caption = "an EEG spectrogram of a {} year old, {} subject feeling {}".format(
            self.get_age(subject), self.get_gender(subject),
            self.get_emotion(session, trial, text=True)
        )
        return caption

    def decimate(self, data):
        if self.decimation > 1:
            data = decimate(data, self.decimation, axis=1, zero_phase=True)
        return data

    def _generate_spectrograms(self, outdir, data_file, subject, session, resolution, hop_length):
        # Containers for sample metadata
        files = []
        genders = []
        ages = []
        emotions = []
        captions = []
        # Load data file for each subject
        raw = mne.io.read_raw_cnt(pjoin(self.datadir, "EEG_raw", data_file), preload=True)
        raw_np = raw.get_data()
        for trial in range(self.n_trials):
            start_samp = self.start_second[session][trial] * self.orig_fs
            end_samp = self.end_second[session][trial] * self.orig_fs
            chan_inds = [ch[1] for ch in seed_channels]
            data = raw_np[chan_inds, start_samp:end_samp]
            data = self.decimate(data)
            N = data.shape[1]
            nblocks = N // self.nsamps
            # Break data into chunks and save
            os.makedirs(pjoin(outdir, f"sub-{subject}"), exist_ok=True)
            for i in range(nblocks):
                blk = data[:, i * self.nsamps: (i + 1) * self.nsamps]
                S = mcs.multichannel_spectrogram(
                    blk,
                    hop_length=hop_length,
                    resolution=resolution, win_length=resolution,
                )
                fname = pjoin(f"sub-{subject}", f"spectrogram-s{session}-t{trial}-{i}.png")
                files.append(fname)
                genders.append(self.get_gender(subject))
                ages.append(self.get_age(subject))
                emotions.append(self.get_emotion(session, trial))
                captions.append(self.get_caption(subject, session, trial))
                S.save(pjoin(outdir, fname))
        return files, genders, ages, emotions, captions

    def make_tvt_splits(self, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=None):
        if seed is not None:
            np.random.seed(seed)
        # Get shuffled metadata
        metadata = pd.read_csv(pjoin(self.datadir, "stfts", "metadata.csv"))
        # Empty per-split metadata dfs
        trainmeta = metadata.iloc[0:0]
        valmeta = metadata.iloc[0:0]
        testmeta = metadata.iloc[0:0]
        # Group by class
        gmeta = metadata.groupby(["gender", "emotion"])
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
        os.makedirs(outdir, exist_ok=True)
        # Spectrogram parameters
        max_bins = resolution / self.n_channels
        hop_length = 8  # number of samples per time-step in spectrogram
        while self.nsamps / hop_length > max_bins:
            hop_length += 8
        # Get list of data directories
        data_files = os.listdir(os.path.join(self.datadir, "EEG_raw"))
        data_files = sorted(list(filter(lambda f: f.endswith(".cnt"), data_files)))
        subjects_and_sessions = [(f.split('_')[0], f.split('_')[1]) for f in data_files]
        # Extract trials and generate spectrograms
        files, genders, ages, emotions, captions = [], [], [], [], []
        for (i, (sub, sess)) in enumerate(subjects_and_sessions):
            sub = int(sub) - 1
            sess = int(sess) - 1
            metadata = self._generate_spectrograms(outdir, data_files[i], sub, sess, resolution, hop_length)
            files.extend(metadata[0])
            genders.extend(metadata[1])
            ages.extend(metadata[2])
            emotions.extend(metadata[3])
            captions.extend(metadata[4])
        # Write metadata to file
        with open(pjoin(outdir, "metadata.csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerow(["file_name", "text", "gender", "age", "emotion"])
            for fn, g, a, e, c in zip(files, genders, ages, emotions, captions):
                writer.writerow([fn, c, g, a, e])
        # Make splits
        self.make_tvt_splits(train_frac, val_frac, test_frac, seed)


class SEEDDataset(torch.utils.data.Dataset):
    def __init__(self, datadir, split="train", transform=None):
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
        emotion = self.metadata.iloc[index]["emotion"]
        gender = self.metadata.iloc[index]["gender"]
        y = (emotion) * 2 + (gender == "F")
        return im, y

    def caption(self, index):
        return self.metadata.iloc[index]["text"]

##### General Dataset #####

class GeneralPreprocessor():

    ####  0 = Male, 1 = Female

    # Originally 250, changed to 125
    def __init__(self, datadirs, nsamps, ovr_perc=0, fs=250):
        self.datadirs = datadirs
        self.nsamps = nsamps
        self.ovr_perc = ovr_perc

        # Init each processor individually
        # Math Init
        self.math_datadir = datadirs["math"]
        self.math_pre = MathPreprocessor(self.math_datadir, nsamps, ovr_perc=ovr_perc, fs=fs)

        # Parkinsons init
        self.park_datadir = datadirs["parkinsons"]
        self.park_pre = ParkinsonsPreprocessor(self.park_datadir, nsamps, ovr_perc=ovr_perc, fs=fs)

        # self.subjects = pd.read_csv(os.path.join(datadir, "participants.tsv"), sep="\t")
        # self.n_channels = len(parkinsons_channels)

    def preprocess(self, resolution=256, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # Preprocess Math data
        samps_per_file = 100
        self.math_pre.preprocess(samps_per_file)

        # Preprocess Parkinsons data
        self.park_pre.preprocess(resolution=resolution)
        self.park_pre.make_tvt_splits()

class GeneralDataset(torch.utils.data.ConcatDataset):
    def __init__(self, datasets, resolution=256, hop_length=192, split="train"):
        super().__init__(datasets)

        self.split = split
        self.datasets = datasets
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore[arg-type]
        for d in self.datasets:
            assert not isinstance(d, torch.utils.data.IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def caption(self, index):
        return self.metadata.iloc[index]["text"]


# class GeneralDataset(torch.utils.data.ConcatDataset):
#     def __init__(self, datadirs, datasets, resolution=256, hop_length=192, split="train", transform=None):
#         super().__init__()

#         self.datadir = datadir
#         self.split = split
#         assert os.path.isfile(pjoin(datadir, f"{split}-metadata.csv")), "No metadata file found for split {}".format(split)
#         self.metadata = pd.read_csv(pjoin(datadir, f"{split}-metadata.csv"))
#         print(self.split, " metadata len: ", len(self.metadata))
#         if transform is None:
#             transform = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.5], [0.5])
#             ])
#         self.transform = transform

#         # Data directories
#         self.park_stft_datadir = datadirs["parkinsons-stft"]
#         self.math_datadir = datadirs["math"]
#         self.math_datadir = self.math_datadir + "/" + split

#         # Init individual datasets
#         self.math_dataset = MathSpectrumDataset(self.math_datadir, resolution, hop_length)
#         self.park_dataset = ParkinsonsDataset(self.park_stft_datadir, split=test)
        
#         # Create cumulative dataset
#         self.datasets = [self.math_dataset, self.park_dataset]

#         assert len(self.datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore[arg-type]
#         for d in self.datasets:
#             assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
#         self.cumulative_sizes = self.cumsum(self.datasets)


#     def __len__(self):
#         return self.cumulative_sizes[-1]

#     def __getitem__(self, idx):
#         if idx < 0:
#             if -idx > len(self):
#                 raise ValueError("absolute value of index should not exceed dataset length")
#             idx = len(self) + idx
#         dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
#         if dataset_idx == 0:
#             sample_idx = idx
#         else:
#             sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
#         return self.datasets[dataset_idx][sample_idx]

#     @property
#     def cummulative_sizes(self):
#         warnings.warn("cummulative_sizes attribute is renamed to "
#                       "cumulative_sizes", DeprecationWarning, stacklevel=2)
#         return self.cumulative_sizes

#     @staticmethod
#     def cumsum(sequence):
#         r, s = [], 0
#         for e in sequence:
#             l = len(e)
#             r.append(l + s)
#             s += l
#         return r

#     def caption(self, index):
#         return self.metadata.iloc[index]["text"]