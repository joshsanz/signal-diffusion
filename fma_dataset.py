""" Utilities for working with FMA dataset """
import os
import numpy as np
import sys
import torch
from datetime import datetime
from diffusers import Mel
from threading import Lock
from tqdm.auto import tqdm

import fma_utils


class FMADataset(torch.utils.data.Dataset):
    def __init__(self, fma_dir='~/data2', subset='small', fs=22050, audio_subdir='fma_large', resolution=512):
        assert subset in ['small', 'medium', 'large']
        self.fma_dir = os.path.expanduser(os.path.expandvars((fma_dir)))
        self.audio_subdir = audio_subdir
        self.subset = subset
        self.fs = fs
        tracks = fma_utils.load(os.path.join(fma_dir, 'fma_metadata', 'tracks.csv'))
        self.tracks = tracks[tracks['set', 'subset'] <= self.subset]
        self.genres = fma_utils.load(os.path.join(fma_dir, 'fma_metadata', 'genres.csv'))
        # self.features = fma_utils.load(os.path.join(fma_dir, 'fma_metadata', 'features.csv'))
        self.echonest = fma_utils.load(os.path.join(fma_dir, 'fma_metadata', 'echonest.csv'))
        self.mel = Mel(x_res=resolution, y_res=resolution, sample_rate=self.fs, n_fft=2048,
                       hop_length=resolution, top_db=80, n_iter=32,)
        # Find top 200 tags for captioning
        tagseries = self.tracks.loc[:, 'track'].tags
        alltags = []
        for tag in tagseries:
            alltags.extend(tag)
        tags, counts = np.unique(alltags, return_counts=True)
        isort = list(reversed(np.argsort(counts)))
        tags = tags[isort]
        counts = counts[isort]
        self.top_tags = tags[:200]
        # Force getitem to be atomic, just in case
        self.lock = Lock()

    def get_echonest_description(self, track_id):
        description = ""
        try:
            en = self.echonest.loc[track_id, ('echonest', 'audio_features',)]
            if en.acousticness > 0.7:
                description += "acoustic "
            if en.danceability > 0.6:
                description += "danceable "
            if en.energy > 0.6:
                description += "energetic "
            if en.instrumentalness > 0.8:
                description += "instrumental "
            if en.liveness > 0.5:
                description += "live "
            if en.speechiness > 0.4:
                description += "spoken "
            if en.tempo < 81:
                description += "down-tempo "
            elif en.tempo > 150:
                description += "up-tempo "
        except KeyError:
            pass
        return description.strip()

    def get_genre_description(self, track_id):
        description = ""
        genres = self.tracks.loc[track_id, ('track', 'genres_all')]
        subgenres = set()
        topgenres = set()
        for genre in genres:
            info = self.genres.loc[genre]
            if info.parent == 0:
                topgenres.add(info.name)
            else:
                subgenres.add(info.name)
                topgenres.add(info.parent)
        subnames = [self.genres.loc[g].title for g in subgenres]
        topnames = [self.genres.loc[g].title for g in topgenres]
        description += " ".join(subnames) + " " + " and ".join(topnames)
        return description

    def get_caption(self, track_id):
        caption = "a "
        # Add echonest features
        caption += self.get_echonest_description(track_id)
        # Add genre
        caption += self.get_genre_description(track_id)
        caption += " song"
        # Add tags
        tags = self.tracks.loc[track_id, ('track', 'tags')]
        to_tag = []
        for t in tags:
            if t in self.top_tags:
                to_tag.append(t)
        if to_tag:
            caption += ", tagged " + ", ".join(to_tag)
        return caption

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        self.lock.acquire()
        track_id = self.tracks.iloc[idx].name
        filepath = fma_utils.get_audio_path(os.path.join(self.fma_dir, self.audio_subdir), track_id)
        self.mel.load_audio(filepath)
        nslices = self.mel.get_number_of_slices()
        islice = np.random.choice(nslices)
        Im = self.mel.audio_slice_to_image(islice)
        caption = self.get_caption(track_id)
        self.lock.release()
        return Im, caption

    def get_track_id(self, idx):
        return self.tracks.iloc[idx].name

    def get_audio_plus_stft(self, idx):
        track_id = self.tracks.iloc[idx].name
        filepath = fma_utils.get_audio_path(os.path.join(self.fma_dir, self.audio_subdir), track_id)
        self.mel.load_audio(filepath)
        nslices = self.mel.get_number_of_slices()
        islice = np.random.choice(nslices)
        Im = self.mel.audio_slice_to_image(islice)
        return Im, self.mel.get_audio_slice(islice)

    def get_unique_genres(self):
        genres = set()
        for i in range(len(self)):
            tid = self.get_track_id(i)
            genres.add(self.get_genre_description(tid))
        return genres


# Convert audio files to captioned STFT images
def preprocess_file(dataset, idx, output_dir, verbose):
    if verbose:
        t = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
        print(f"{t} >> idx {idx}", flush=True)
    Im, caption = dataset[idx]
    track_id = dataset.get_track_id(idx)
    subdir = f"{track_id:06}"[:3]
    os.makedirs(f"{output_dir}/{subdir}", exist_ok=True)
    fname = f"{output_dir}/{subdir}/{track_id:06}.png"
    Im.save(fname)
    return f"{subdir}/{track_id:06}.png", caption


def preprocess_fma_audio(data_dir, fma_subset, output_dir, resolution=512, sample_rate=22050, skip_idxs=[],
                         start_from=0, verbose=False):
    import pandas as pd
    import concurrent.futures

    ds = FMADataset(fma_dir=data_dir, subset=fma_subset, resolution=resolution, fs=sample_rate)
    N = len(ds)
    metadata = []
    os.makedirs(output_dir, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        # Start the preprocessing operations and mark each future with its index
        future_to_idx = {executor.submit(preprocess_file, ds, i, output_dir, verbose): i for i in range(start_from, N) if i not in skip_idxs}
        progress = tqdm(total=N) if not verbose else False
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                data = future.result()
            except Exception as e:
                print(f"Index {idx} generated exception: {e}", file=sys.stderr)
                raise e
            metadata.append(data)
            if progress:
                progress.update(1)
    if progress:
        progress.close()
    metadata = pd.DataFrame(metadata, columns=["file_name", "text"])
    metadata.to_csv(f"{output_dir}/metadata.csv", index=False)
    return metadata
