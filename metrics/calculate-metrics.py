# Calculate KDD, CMMD (Maximal Mean Distance with DINO or CLIP features) and other(?) metrics for generated datasets
# Don't do FID - everyone agrees it's bad even though everyone still reports it
# AND it's even worse for small (<50k) sample sizes

import argparse
import json
import numpy as np
import os
import shutil
import torch_fidelity as tfid
from torch.utils.data import Dataset
from datasets import load_dataset, Features, Image


class DatasetWrapper(Dataset):
    def __init__(self, dataset, truncate: int = -1):
        self.dataset = dataset
        self.truncate = truncate
        if truncate > 0:
            self.idx_map = np.random.choice(len(dataset), truncate, replace=False)

    def __len__(self):
        if self.truncate > 0:
            return self.truncate
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.truncate > 0:
            idx = self.idx_map[idx]
        return self.dataset[idx]["image"]


def calculate_metrics(generated, real, kid_subset_size=1000, batch_size=64,
                      feature_extractor='dinov2-vit-b-14',
                      real_cache=None, generated_cache=None):
    kwargs = dict(
        isc=False, fid=False, kid=True, prc=True, ppl=False,
        kid_kernel='rbf', kid_kernel_rbf_sigma=10.0,
        # Paper recommends 3 with 50k images, but we need a larger neighborhood
        # to have any overlap between manifolds of real and generated data
        prc_neighborhood=10,
        kid_subset_size=kid_subset_size,
        feature_extractor=feature_extractor, batch_size=batch_size,
        input1=real, input2=generated,
        input1_cache_name=real_cache, input2_cache_name=generated_cache,
    )
    metrics = tfid.calculate_metrics(**kwargs)
    # TODO: PRC always gives 0s, need to investigate
    return metrics


def clear_cache(dataset_name):
    shutil.rmtree(f"{os.environ['ENV_TORCH_HOME']}/fidelity_datasets/{dataset_name}", ignore_errors=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Calculate KDD, CMMD and other metrics for generated datasets')
    parser.add_argument('-g', '--generated', type=str, help='Path to generated dataset')
    parser.add_argument('-r', '--real', type=str, help='Path to real dataset')
    parser.add_argument("--kid-subset-size", type=int, default=1000, help="Number of samples to use per subset for KID calculation (default %(default)s)")
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="Batch size for feature calculation (default %(default)s)")
    parser.add_argument("-o", '--output', type=str, default="metrics.json", help='File to save metrics (default %(default)s)')
    parser.add_argument("--cache", action="store_true", help="Cache feature extraction results")
    parser.add_argument("--clear-cache", action="store_true", help="Clear feature extraction cache (e.g. if datasets have changed)")
    return parser.parse_args()


def main(args):
    real_name = os.path.basename(args.real.rstrip("/"))
    generated_name = os.path.basename(args.generated.rstrip("/"))

    real_cache = None
    generated_cache = None
    if args.cache:
        real_cache = real_name
        generated_cache = generated_name
    if args.clear_cache:
        clear_cache(real_name)
        clear_cache(generated_name)

    print(f"Loading generateds {generated_name}")
    generated = load_dataset(
        "imagefolder", data_dir=args.generated,
        features=Features({"image": Image(mode="RGB")}),
    ).with_format("torch")["train"]
    print(f"Loading reals {real_name}")
    real = load_dataset(
        "imagefolder", data_dir=args.real,
        features=Features({"image": Image(mode="RGB")}),
    ).with_format("torch")["train"]
    generated = DatasetWrapper(generated)
    real = DatasetWrapper(real)
    feature_extractors = ["dinov2-vit-l-14", 'clip-vit-l-14']
    out = {}
    for fe in feature_extractors:
        metrics = calculate_metrics(
            generated, real,
            args.kid_subset_size, args.batch_size, fe,
            real_cache, generated_cache,)
        print(f"Metrics for {fe}: {metrics}")
        out[fe] = metrics
    # TODO: per-class metrics
    # TODO: memorization metrics

    # Save and report metrics
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)


if __name__ == '__main__':
    args = parse_args()
    main(args)
