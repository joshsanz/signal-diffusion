# Train the best classifier for each data source to test on the rest

import argparse
from datasets import load_dataset
import numpy as np
from omegaconf import OmegaConf
import os
from os.path import join as pjoin
import shutil
import sys
import time
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import torchvision.transforms.v2 as transforms
import warnings

sys.path.append("../")
from cnn_models import CNNClassifierLight, LabelSmoothingCrossEntropy
# from data_processing.math import MathDataset
from data_processing.parkinsons import ParkinsonsDataset
from data_processing.seed import SEEDDataset
from data_processing.general_dataset import GeneralDataset, GeneralSampler
from training import train_class, TrainingConfig


opt_dict = {
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
}


class DummyLogger:
    def __init__(self):
        pass

    def add_scalar(self, *args, **kwargs):
        pass


def timefmt(t):
    t = int(t)
    h = t // 3600
    m = (t % 3600) // 60
    s = t % 60
    fmtd = f"{h:02d}:" if h > 0 else ""
    fmtd += f"{m:02d}:" if m > 0 else "00:"
    fmtd += f"{s:02d}"
    return fmtd


def print_run_header(t0, run, num_runs):
    t1 = time.time()
    t_run = timefmt(t1 - t0)
    t_est = "??" if run == 0 else timefmt((t1 - t0) / run * num_runs)
    print(f"Training run {run + 1}/{num_runs} [{t_run}/{t_est}]")


def load_datasets(cfg):
    all_datasets = []
    if cfg.data.from_eeg_data:
        # Load real datasets
        datadirs = {}
        datahome = '/data/shared/signal-diffusion'
        # datahome = '/mnt/d/data/signal-diffusion'
        # Math dataset
        datadirs['math'] = f'{datahome}/eeg_math'
        datadirs['math-stft'] = os.path.join(datadirs['math'], 'stfts')
        # Parkinsons dataset
        datadirs['parkinsons'] = f'{datahome}/parkinsons/'
        datadirs['parkinsons-stft'] = os.path.join(datadirs['parkinsons'], 'stfts')
        # SEED dataset
        datadirs['seed'] = f'{datahome}/seed/'
        datadirs['seed-stft'] = os.path.join(datadirs['seed'], "stfts")

        if cfg.data.transform == "trivial_augment_wide":
            transform_fn = transforms.Compose([
                transforms.TrivialAugmentWide(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
        else:
            transform_fn = None
        # Validation datasets
        # math_val_dataset = MathDataset(datadirs['math-stft'], split="val")
        parkinsons_val_dataset = ParkinsonsDataset(datadirs['parkinsons-stft'], split="val")
        seed_val_dataset = SEEDDataset(datadirs['seed-stft'], split="val")
        val_datasets = [parkinsons_val_dataset, seed_val_dataset]
        # Train datasets
        # math_real_train_dataset = MathDataset(datadirs['math-stft'], split="train",
        #                                       transform=transform_fn)
        parkinsons_train_dataset = ParkinsonsDataset(datadirs['parkinsons-stft'], split="train",
                                                     transform=transform_fn)
        seed_train_dataset = SEEDDataset(datadirs['seed-stft'], split="train",
                                         transform=transform_fn)
        train_datasets = [parkinsons_train_dataset, seed_train_dataset]

        val_set = GeneralDataset(val_datasets, split='val')
        train_set = GeneralDataset(train_datasets, split='train')

        # Sampler for balanced training data
        train_samp = GeneralSampler(train_datasets, cfg.training.batch_size, split='train')

        # Dataloaders
        num_workers = 4
        persistent = num_workers > 0
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=cfg.training.batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, persistent_workers=True
        )
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=cfg.training.batch_size, num_workers=num_workers,
            pin_memory=True, persistent_workers=persistent, sampler=train_samp
        )
        all_datasets.append(("real_eeg", train_loader, val_loader))

    # Synthetic datasets
    if cfg.data.from_synth_dis:
        dis_data = load_dataset("imagefolder", data_dir=cfg.data.dis_data_dir)
        # TODO: check for unsplit data and split it here
        train_loader = torch.utils.data.DataLoader(
            dis_data["train"], batch_size=cfg.training.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            dis_data["validation"], batch_size=cfg.training.batch_size, shuffle=False)
        all_datasets.append(("dis", train_loader, val_loader))

    if cfg.data.from_synth_sd:
        sd_data = load_dataset("imagefolder", data_dir=cfg.data.sd_data_dir)
        # TODO: check for unsplit data and split it here
        train_loader = torch.utils.data.DataLoader(
            sd_data["train"], batch_size=cfg.training.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            sd_data["validation"], batch_size=cfg.training.batch_size, shuffle=False)
        all_datasets.append(("sd", train_loader, val_loader))

    return all_datasets


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        warnings.warn("Running on CPU. This will be slow.")

    cfg = OmegaConf.load(args.config)
    with open(pjoin(args.output_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)

    all_datasets = load_datasets(cfg)
    criterion = LabelSmoothingCrossEntropy(cfg.training.label_smoothing)
    tconf = TrainingConfig(cfg.training.epochs, val_every_epochs=1)
    dlogger = DummyLogger()

    for datasource, train_loader, val_loader in all_datasets:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        model = CNNClassifierLight(cfg.architecture.in_channels, cfg.architecture.out_dim,
                                   dropout=cfg.training.dropout, pooling=cfg.architecture.pooling)
        model.to(device)

        optimizer = opt_dict[cfg.training.optimizer.lower()](
            model.parameters(), lr=cfg.training.lr)

        ema_model = None
        if cfg.training.ema:
            ema_model = AveragedModel(model,
                                      multi_avg_fn=get_ema_multi_avg_fn(cfg.training.ema_decay))

        run_name = f"{datasource}_best"
        print("#" * 40)
        print(f"Training on {datasource}\n")
        _, accs, val_accs, ema_val_accs = train_class(
            tconf, model, ema_model, train_loader, val_loader, optimizer, None,
            criterion, device, dlogger, run_name, save_model=True
        )

        print(f"Training accuracy: {np.max(accs) * 100:.2f}%")

        shutil.move(f"models/best_model-{run_name}.pt",
                    pjoin(args.output_dir, f"best_model-{datasource}.pt"))
        shutil.move(f"models/last_model-{run_name}.pt",
                    pjoin(args.output_dir, f"last_model-{datasource}.pt"))
        print(f"Validation accuracy: {np.max(val_accs) * 100:.2f}%")

        if cfg.training.ema:
            shutil.move(f"models/best_ema_model-{run_name}.pt",
                        pjoin(args.output_dir, f"best_ema_model-{datasource}.pt"))
            shutil.move(f"models/last_ema_model-{run_name}.pt",
                        pjoin(args.output_dir, f"last_ema_model-{datasource}.pt"))
            print(f"EMA validation accuracy: {np.max(ema_val_accs) * 100:.2f}%")

        print()


def parse_args():
    parser = argparse.ArgumentParser(
        "Train the best classifier for each data source to test on the rest")
    parser.add_argument("-c", "--config", type=str, default="best_classifier.yaml")
    parser.add_argument("-o", "--output-dir", type=str, default="bestmodels")
    parser.add_argument("-s", "--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    t0 = time.time()
    main(args)
    t1 = time.time()
    print(f"Total time: {timefmt(t1 - t0)}")
