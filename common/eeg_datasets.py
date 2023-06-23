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


class EarEEGPreprocessor:
    def __init__(self, base_path, raw_fs=1000, fs_out=250, input_users='all'):
        self.base_path = base_path
        ##################
        # READ-IN EAR EEG
        ##################
        # NOTE, this takes a long time to run.
        # (It could be parallelized to reduce runtime)

        # name of spreadsheet with experiment details
        # details_spreadsheet = 'gdrive/My Drive/Muller Group Drive/Ear EEG/Drowsiness_Detection/classifier_TBME/classification_scripts/trial_details_spreadsheet_basic.csv'
        self.details_spreadsheet = base_path + 'eeg_classification_data/ear_eeg_data/trial_details_spreadsheet_good.csv'

        # file path to ear eeg data (must be formated r'filepath\\')
        # data_filepath = r'C:\Users\Carolyn\OneDrive\Documents\school\berkeley\research\ear_eeg_classification_framework\experimental_recordings\drowsiness_studies\ear_eeg\\'
        self.data_filepath = base_path + 'eeg_classification_data/ear_eeg_data/ear_eeg_clean/'

        # user number or all users('all', 'ryan', 'justin', 'carolyn', 'ashwin', 'connor')
        self.input_users = input_users

        # channels of eeg to read in for each trial (must include 5 and 11 if re-refernecing is enabled in the next block)
        self.data_chs = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]

        # sampling frequency of system (fs=1000 for wandmini)
        self.raw_fs = raw_fs
        self.fs_out = fs_out

        #################
        # READ-IN LABELS
        #################

        # Note: label read in will match Ear EEG read in
        # (same trials will be read in, and the experiment lengths will be the same)

        # file path to labels(must be formated r'filepath\\')
        # label_filepath = r'C:\Users\Carolyn\OneDrive\Documents\school\berkeley\research\ear_eeg_classification_framework\experimental_recordings\drowsiness_studies\labels\\'
        self.label_filepath = base_path + 'eeg_classification_data/ear_eeg_data/labels/'

        # call read in labels
        # all}|_labels

    def format_data(self, data_set, seq_len):
        # Data needs to be input as (samples, channels), for ex: (2,400,000, 10)
        formatted_datasets = []
        for i in range(len(data_set)):
            data = data_set[i]
            data = decimate(data, int(self.raw_fs / self.fs_out), zero_phase=True, axis=0)
            data_length = data.shape[0]

            # 0-mean, unit variance per channel
            data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

            num_seqs = int(np.floor(data_length / seq_len))
            formatted_data = np.array(np.split(data[:num_seqs * seq_len], num_seqs))
            formatted_datasets.append(formatted_data)
        return formatted_datasets

    @staticmethod
    def one_hot_encode(input):
        b = np.zeros((int(input.size), int(input.max() + 1)))
        b[np.arange(input.size), input] = 1
        one_hot_labels = np.array(b)

        return one_hot_labels

    def format_labels(self, labels_set, seq_len):
        formatted_labels = []
        decimation = int(self.raw_fs / self.fs_out)
        for i in range(len(labels_set)):
            labels = labels_set[i][::decimation]
            labels_length = labels.shape[0]
            # No need to one-hot encode for nn.CrossEntropyLoss

            num_seqs = int(np.floor(labels_length / seq_len))
            new_labels = np.array(np.split(labels[:num_seqs * seq_len], num_seqs)).astype('int')
            formatted_labels.append(new_labels)
        return formatted_labels

    def write_proc_data_to_disk(self, proc_train_X, proc_train_y, proc_val_X, proc_val_y,
                                proc_test_X, proc_test_y):
        for X, y, split in [(proc_train_X, proc_train_y, "train"),
                            (proc_val_X, proc_val_y, "val"),
                            (proc_test_X, proc_test_y, "test")]:
            print(f"Writing {split} samples...")
            preproc_path = os.path.join(self.base_path, f"ear_eeg_{split}")
            os.makedirs(preproc_path, exist_ok=True)
            counter = 0
            sample_map = []
            for index in range(len(X)):
                print(f"Splitting and saving recording {index}...")
                dir_name = preproc_path + '/recording_' + str(index)
                os.makedirs(dir_name, exist_ok=True)
                N_sample_per_file = 100
                cur_samples = X[index]
                cur_labels = y[index]
                cur_num_samples = cur_samples.shape[0]
                cnt = 0
                while cnt < cur_num_samples:
                    file_samples = cur_samples[cnt:cnt + N_sample_per_file, :, :]
                    file_labels = cur_labels[cnt:cnt + N_sample_per_file, :]
                    nsamps = file_samples.shape[0]  # May not be N_sample_per_file for final split
                    filename = f"{counter}-{counter + nsamps - 1}.npz"
                    np.savez(f"{dir_name}/{filename}", X=file_samples, y=file_labels)
                    sample_map.append((f"recording_{index}/{filename}", counter, counter + nsamps - 1))
                    cnt += nsamps
                    counter += nsamps

            print(f"Writing metadata to {preproc_path + '/metadata.csv'}...")
            with open(preproc_path + "/metadata.csv", "w") as fcsv:
                writer = csv.writer(fcsv)
                writer.writerow(["filename", "start_sample", "final_sample"])
                writer.writerows(sample_map)

    def preprocess(self, seq_len=256):
        # call read in ear eeg
        print("Reading in raw data...")
        all_raw_data, filenames, data_lengths, file_users, refs = read_in_ear_eeg.read_in_clean_data(
            self.details_spreadsheet, self.data_filepath,
            self.input_users, self.data_chs, self.raw_fs, False
        )
        all_labels = read_in_labels.read_in_labels(
            filenames, data_lengths, self.label_filepath, False
        )
        filtered_data = eeg_filter.filter_studies(all_raw_data)
        # print(len(all_raw_data))
        # print(all_raw_data[0].shape)
        # print(len(filtered_data))
        # print(filtered_data[0].shape)
        del all_raw_data

        # Data constants
        # carolyn_indices = [0, 1, 2, 3, 4]
        # ryan_indices = [5, 6, 7, 8, 9]
        # justin_indices = [10, 11, 12, 13, 14]
        # conor_indices = [15, 16, 17, 18, 19]
        # avi_indices = [20, 21]
        # train_perc, val_perc, test_perc = 0.55, 0.30, .15
        train_ind = [2, 3, 4, 8, 9, 12, 13, 14, 15, 17, 18, 19, 21]
        val_ind = [1, 6, 11, 16, 20, 7]
        # test_ind = [0, 5, 10]

        # Split up into train, val, and test datasets
        train_data, val_data, test_data = [], [], []
        train_labels, val_labels, test_labels = [], [], []
        for i in range(len(filtered_data)):
            if i in train_ind:
                train_data.append(filtered_data[i])
                train_labels.append(all_labels[i])
            elif i in val_ind:
                val_data.append(filtered_data[i])
                val_labels.append(all_labels[i])
            else:
                test_data.append(filtered_data[i])
                test_labels.append(all_labels[i])

        # Format data
        print("Downsampling signals...")
        proc_train_X = self.format_data(train_data, seq_len)
        proc_val_X = self.format_data(val_data, seq_len)
        proc_test_X = self.format_data(test_data, seq_len)
        filtered_data, train_data, val_data, test_data = [], [], [], []

        # Format labels
        proc_train_y = self.format_labels(train_labels, seq_len)
        proc_val_y = self.format_labels(val_labels, seq_len)
        proc_test_y = self.format_labels(test_labels, seq_len)
        train_labels, val_labels, test_labels = [], [], []

        # Save to disk
        self.write_proc_data_to_disk(
            proc_train_X, proc_train_y,
            proc_val_X, proc_val_y,
            proc_test_X, proc_test_y)

        return (os.path.join(self.base_path, "ear_eeg_train"),
                os.path.join(self.base_path, "ear_eeg_val"),
                os.path.join(self.base_path, "ear_eeg_test"))


class EarDataset(torch.utils.data.Dataset):
    'Dataset loader for preprocessed ear EEG data'
    def __init__(self, data_path, vector_samps=10, cache_mb=500):
        'Initialization'
        self.data_path = data_path
        self.vector_samps = vector_samps
        files, starts, stops = [], [], []
        with open(data_path + "/metadata.csv", "r") as fcsv:
            reader = csv.reader(fcsv)
            _ = next(reader)  # skip header
            for (f, s0, s1) in reader:
                files.append(f)
                starts.append(int(s0))
                stops.append(int(s1))
        self.start_inds = np.array(starts)
        self.stop_inds = np.array(stops)
        self.files = np.array(files)
        # LRU cache for loaded data to try to reduce disk access
        # Be careful, the memory foot print will grow per dataloader worker
        cache_len = int(np.ceil(cache_mb / 1.2))  # 1.2 MB per loaded file
        self.cache = CacheDict(cache_len=cache_len)

    def __len__(self):
        'Denotes the total number of samples'
        return self.stop_inds[-1]

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        file_idx = np.searchsorted(self.start_inds, index, side='right') - 1
        filename = self.files[file_idx]
        val = self.cache.get(filename, None)
        if val is None:
            data = np.load(os.path.join(self.data_path, filename))
            X = torch.tensor(data['X']).float()
            y = torch.tensor(data['y']).long()
            self.cache[filename] = (X, y)
        else:
            X, y = val
        offset = index - self.start_inds[file_idx]
        X, y = X[offset, :, :], y[offset, :]

        # Group vector_samps x nchannel chunks into input vectors
        shape = X.shape
        # samples x channels --> channels x samples --> channels x sample chunks x samples
        nchunks = shape[0] // self.vector_samps
        X = X.permute(1, 0).reshape(shape[1], nchunks, self.vector_samps)
        # channels x sample chunks x samples --> chunks x channels x samples --> chunks x channels+samples
        X = X.permute(1, 0, 2).reshape(nchunks, -1).contiguous()
        # Use plurality voting to determine new y values
        shape = y.shape
        y = y.reshape(-1, self.vector_samps)
        y = y.median(dim=1).values
        return X, y


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
    def __init__(self, data_path, nsamps, fs=250):
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
        return S, y


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

<<<<<<< HEAD
    def decimate(self, data):
        if self.decimation > 1:
            data = decimate(data, self.decimation, axis=1, zero_phase=True)
        return data
=======
    # def mydecimate(data, decimation_factor):
    #     if decimation_factor > 1:
    #         data = 
    #     return data
>>>>>>> 149d4d7 (added sampler, not working)

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
<<<<<<< HEAD
            data = self.decimate(data)
=======

            # decimate data
            data = decimate(data, int(self.decimation), axis=1, zero_phase=True)
>>>>>>> 149d4d7 (added sampler, not working)
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
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        print("index: ", index, "metdata len: ", len(self.metadata), "split: ", self.split)
        print('\n')
        fn = self.metadata.iloc[index]["file_name"]
        im = Image.open(pjoin(self.datadir, fn))
        im = self.transform(im)
        health = self.metadata.iloc[index]["health"]
        gender = self.metadata.iloc[index]["gender"]
        y = (health == "PD") * 2 + (gender == "F")
        return im, y

    def caption(self, index):
        return self.metadata.iloc[index]["text"]


<<<<<<< HEAD
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
=======
# class WeightedRandomSampler(Sampler):

#     def __iter__(self) -> Iterator[int]:
#         rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=self.generator)
#         yield from iter(rand_tensor.tolist())

#     def __len__(self) -> int:
#         return 


class ParkinsonsSampler(torch.utils.data.WeightedRandomSampler):
    def __init__(self, datadir, num_samples, split="train", replacement=True, generator=None):
        assert os.path.isfile(pjoin(datadir, f"{split}-metadata.csv")), "No metadata file found for split {}".format(split)
        self.metadata = pd.read_csv(pjoin(datadir, f"{split}-metadata.csv"))
        self.weights = torch.as_tensor(self.generate_weights(self.metadata), dtype=torch.double)
        self.num_samples = num_samples# Number of samples to draw not total
        self.split = split
        self.replacement = replacement
        self.generator = generator
>>>>>>> 149d4d7 (added sampler, not working)

    def __len__(self):
        return len(self.metadata)

<<<<<<< HEAD
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
=======
    def __iter__(self):
        rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=self.generator)
        print("iter length: ", rand_tensor.size())
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
        print("metadata len: ", len(self.metadata))
        print(" weights len; ", len(output_weights))

        return output_weights








        
>>>>>>> 149d4d7 (added sampler, not working)
