# Datasets for EEG classification
import mne
import numpy as np
import os
import pandas as pd
import torch
from bidict import bidict
from collections import OrderedDict
from os.path import join as pjoin
import bisect
import warnings

# Import support scripts: pull_data
from data_processing.math import MathPreprocessor
from data_processing.parkinsons import ParkinsonsPreprocessor
from data_processing.seed import SEEDPreprocessor

mne.set_log_level("WARNING")

ear_class_labels = bidict({0: "awake", 1: "sleepy"})


emotion_map = bidict({"disgust": 0, "fear": 1, "sad": 2, "neutral": 3, "happy": 4})
seed_class_labels = bidict(
    {
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
    }
)

general_class_labels = bidict(
    {
        0: "male",
        1: "female",
    }
)

general_dataset_map = bidict({0: "math", 1: "parkinsons", 2: "SEED"})


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


#  Class to generalize all the preprocessors, WORK ON LATER

# class EEGPreprocessor:
#     def __init__(self, datadir, nsampls, orig_fs, ovr_perc=0, fs=250):
#         self.datadir = datadir
#         self.eegdir = os.path.join(self.datadir, "raw_eeg")
#         self.stfttdir = os.path.join(self.datadir, "stfts")

#         # Establish sampling constants
#         self.orig_fs = orig_fs
#         self.fs = fs
#         self.decimation = orig_fs // fs
#         print("Decimation factor {} new fs {}".format(self.decimation, orig_fs / self.decimation))
#         self.nsamps = int(nsamps * (1/self.decimation))
#         print("Decimation factor {} new number of samples {}".format(self.decimation, self.nsamps))
#         self.ovr_perc = ovr_perc
#         self.noverlaps = int(nsamps * ovr_perc * (1/self.decimation))

#         self.subjects = pd.read_csv(os.path.join(datadir, "subject-info.csv"))


#################
# General Dataset
class GeneralPreprocessor:
    # 0 = Male, 1 = Female
    # Originally 250, changed to 125
    def __init__(self, datadirs, nsamps, ovr_perc=0, fs=125):
        self.datadirs = datadirs
        self.nsamps = nsamps
        self.ovr_perc = ovr_perc

        # Init each processor individually
        # Math init
        self.math_datadir = datadirs["math"]
        self.math_pre = MathPreprocessor(
            self.math_datadir, nsamps, ovr_perc=ovr_perc, fs=fs
        )

        # Parkinsons init
        self.park_datadir = datadirs["parkinsons"]
        self.park_pre = ParkinsonsPreprocessor(
            self.park_datadir, nsamps, ovr_perc=ovr_perc, fs=fs
        )

        # # SEED init
        self.seed_datadir = datadirs["seed"]
        self.seed_pre = SEEDPreprocessor(
            self.seed_datadir, nsamps, ovr_perc=ovr_perc, fs=fs
        )

    def make_tvt_splits(self, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=None):
        self.math_pre.make_tvt_splits(train_frac, val_frac, test_frac, seed)
        self.park_pre.make_tvt_splits(train_frac, val_frac, test_frac, seed)
        self.seed_pre.make_tvt_splits(train_frac, val_frac, test_frac, seed)

    def preprocess(self, resolution=256, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # Preprocess Math data
        self.math_pre.preprocess(resolution=resolution)

        # Preprocess Parkinsons data
        self.park_pre.preprocess(resolution=resolution)

        # Preprocess SEED data
        self.seed_pre.preprocess(resolution=resolution)


class GeneralDataset(torch.utils.data.ConcatDataset):
    def __init__(self, datasets, resolution=256, hop_length=192, split="train"):
        super().__init__(datasets)

        self.split = split
        self.datasets = datasets
        assert len(self.datasets) > 0, "datasets should not be an empty iterable"  # type: ignore[arg-type]
        for d in self.datasets:
            assert not isinstance(
                d, torch.utils.data.IterableDataset
            ), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)
        self.dataset_calls = []

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        self.dataset_calls.append(dataset_idx)
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn(
            "cummulative_sizes attribute is renamed to " "cumulative_sizes",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.cumulative_sizes

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            elen = len(e)
            r.append(elen + s)
            s += elen
        return r

    def caption(self, index):
        return self.metadata.iloc[index]["text"]


class GeneralSampler(torch.utils.data.WeightedRandomSampler):
    def __init__(
        self, datasets, num_samples, split="train", replacement=True, generator=None
    ):
        self.datasets = datasets
        self.metadatas = []
        for dataset in datasets:
            meta_datadir = pjoin(dataset.datadir, f"{split}-metadata.csv")
            if not os.path.isfile(meta_datadir):
                meta_datadir = pjoin(dataset.datadir, "metadata.csv")

            assert os.path.isfile(
                meta_datadir
            ), "No metadata file found for split {}".format(split)
            metadata = pd.read_csv(meta_datadir)
            self.metadatas.append(metadata)

        self.weights = torch.as_tensor(
            self.generate_weights(self.metadatas), dtype=torch.double
        )
        self.num_samples = num_samples  # Number of samples to draw not total
        self.split = split
        self.replacement = replacement
        self.generator = generator

    def __len__(self):
        output = 0
        for dataset in self.datasets:
            output += len(dataset)
        return output

    def __iter__(self):
        rand_tensor = torch.multinomial(
            self.weights, len(self), self.replacement, generator=self.generator
        )
        yield from iter(rand_tensor.tolist())

    def generate_weights(self, metadatas):
        metadatas = self.metadatas
        # Find male to female ratio
        totals, male, female = [], 0, 0

        for metadata in metadatas:
            genders = metadata.loc[:, "gender"]
            totals.append(len(metadata))
            female += list(genders).count('F')
            male += list(genders).count('M')

            print("totals: ", totals)
            print("female: ", female)
            print("male: ", male)

        total_samps = sum(totals)
        gend_weights = [male / total_samps, (female / total_samps)]
        print(gend_weights)
        dataset_weights = [total / total_samps for total in totals]

        summed_weights = []
        weights = []
        for dw in dataset_weights:
            data_gender_w = []
            for gw in gend_weights:
                data_gender_w.append(gw + dw)
            summed_weights.append(sum(data_gender_w))
            weights.append(data_gender_w)

        # Class rankings
        rankings = {}
        for label in range(len(metadatas)):
            label_weight = summed_weights[label]
            rank = 0
            for weight in summed_weights:
                if label_weight > weight:
                    rank += 1
            rankings[label] = rank

        # Flip weights so smaller classes are more prominent
        new_label_weights = [weights[(len(metadatas) - 1) - rankings[i]] for i in range(len(metadatas))]
        print("weights: ", weights)
        print("new_label_weights: ", new_label_weights)

        output_weights = []

        for i in range(len(metadatas)):
            metadata = metadatas[i]
            genders = np.array(metadata.loc[:, "gender"])
            genders[genders == 'F'] = new_label_weights[i][0]
            genders[genders == 'M'] = new_label_weights[i][1]
            len(genders)
            output_weights += list(genders)

        # Normalize output weights
        norm_fact = sum(output_weights)
        output_weights = [weights / norm_fact for weights in output_weights]

        return output_weights
