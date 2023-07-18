# Datasets for EEG classification
import csv
import mne
import numpy as np
import os
import pandas as pd
import torch
from bidict import bidict
from PIL import Image
from scipy.signal import decimate
from torchvision import transforms
from os.path import join as pjoin
import math
from tqdm.auto import tqdm


# import support scripts: pull_data
import common.multichannel_spectrograms as mcs
from data_processing.channel_map import seed_channels

mne.set_log_level("WARNING")

ear_class_labels = bidict({0: "awake", 1: "sleepy"})


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

    def __init__(self, datadir, nsamps, ovr_perc=0, fs=250):
        self.datadir = datadir

        # Establish sampling constants
        self.orig_fs = 1000
        self.decimation = self.orig_fs // fs
        print("Decimation factor {} new fs {}".format(self.decimation, self.orig_fs / self.decimation))
        self.fs = self.orig_fs / self.decimation
        self.nsamps = nsamps
        print("Decimation factor {} new number of samples {}".format(self.decimation, self.nsamps))
        self.ovr_perc = ovr_perc
        self.noverlap = int(np.floor(nsamps * ovr_perc))

        # Get subject info
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
        sub_file = pjoin(self.datadir, "EEG_raw", data_file)
        raw = mne.io.read_raw_cnt(sub_file, preload=True)
        raw_np = raw.get_data()
        for trial in range(self.n_trials):
            start_samp = self.start_second[session][trial] * self.orig_fs
            end_samp = self.end_second[session][trial] * self.orig_fs
            chan_inds = [ch[1] for ch in seed_channels]
            data = raw_np[chan_inds, start_samp:end_samp]
            # Decimate data
            data = self.decimate(data)

            # Break data into chunks and save
            os.makedirs(pjoin(outdir, f"sub-{subject}"), exist_ok=True)
            N = data.shape[1]
            shift_size = self.nsamps - self.noverlap
            if self.noverlap != 0:
                nblocks = math.floor((N - self.nsamps) / shift_size) + 1
            else:
                nblocks = math.floor(N / self.nsamps)
            assert nblocks > 1, (
                "File {} (T={}) is too short to be used with nsamps {} and decimation {}".format(
                    sub_file, N, self.nsamps, self.decimation
                )
            )

            start_ind = 0
            end_ind = self.nsamps
            i = 0
            while end_ind <= N:
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
        print("LEN META: ", len(metadata))

        N = len(metadata)
        n_train = int(np.ceil(train_frac * N))
        n_val = int(np.floor(val_frac * N))
        # n_test = N - (n_train + n_val)
        subjects = metadata.loc[:, "file_name"].apply(lambda s: s.split("/")[0])
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

        print("trainmeta: ", len(trainmeta.loc[:, 'file_name'].apply(lambda s: s.split("/")[0]).unique()))

        trainmeta.to_csv(pjoin(self.datadir, "stfts", "train-metadata.csv"), index=False)
        valmeta.to_csv(pjoin(self.datadir, "stfts", "val-metadata.csv"), index=False)
        testmeta.to_csv(pjoin(self.datadir, "stfts", "test-metadata.csv"), index=False)

    def preprocess(self, resolution=512, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=None):
        # Make output dir
        outdir = pjoin(self.datadir, "stfts")
        # Delete any premade stfts
        # if os.path.isdir(outdir):
        #     shutil.rmtree(outdir)
        # Create new stft output directory
        os.makedirs(outdir, exist_ok=True)
        # Spectrogram parameters
        max_bins = math.floor(resolution / self.n_channels)
        hop_length = 8  # number of samples per time-step in spectrogram
        while self.nsamps / hop_length > max_bins:
            hop_length += 8
        # Get list of data directories
        data_files = os.listdir(os.path.join(self.datadir, "EEG_raw"))
        data_files = sorted(list(filter(lambda f: f.endswith(".cnt"), data_files)))
        subjects_and_sessions = [(f.split('_')[0], f.split('_')[1]) for f in data_files]
        # Extract trials and generate spectrograms
        files, genders, ages, emotions, captions = [], [], [], [], []
        for (i, (sub, sess)) in tqdm(enumerate(subjects_and_sessions)):
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
    name = "SEED V"

    def __init__(self, datadir, split="train", transform=None):
        self.dataname = 'seed'
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
        return im, (y % 2)

    def caption(self, index):
        return self.metadata.iloc[index]["text"]
