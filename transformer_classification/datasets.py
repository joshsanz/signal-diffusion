# Datasets for EEG classification
import csv
import numpy as np
import os
import torch
from collections import OrderedDict
from scipy.signal import decimate

# import support scripts: pull_data
import support_scripts.read_in_ear_eeg as read_in_ear_eeg
import support_scripts.read_in_labels as read_in_labels
import support_scripts.eeg_filter as eeg_filter


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
