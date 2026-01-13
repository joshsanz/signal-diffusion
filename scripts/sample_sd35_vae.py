#!/usr/bin/env python
"""Run a single dataset sample through the SD3.5 VAE encoder/decoder.

Loads the parquet dataset with `datasets.load_dataset`, grabs one sample by
index, passes it through the Stable Diffusion 3.5 AutoencoderKL (subfolder=vae),
prints latent shape/mean/std, and saves the original plus reconstruction.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tomllib
from typing import Any, cast

import torch
from datasets import load_dataset
from diffusers import AutoencoderKL
from torchvision import transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode/decode a dataset sample with the SD3.5 VAE"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/data/data/signal-diffusion/processed-iq/reweighted_meta_dataset_log_n2048_fs125"),
        help="Directory containing train/val/test parquet files",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Sample index to visualize",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Shuffle seed for random sampling",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("config/default.toml"),
        help="TOML config path that holds `hf_models.stable_diffusion_model_id`",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/vae_sample"),
        help="Where to save the original and reconstructed images",
    )
    return parser.parse_args()


def load_model_id(config_path: Path) -> str:
    with config_path.open("rb") as f:
        cfg = tomllib.load(f)
    return cfg["hf_models"]["stable_diffusion_model_id"]


def load_dataset_split(dataset_dir: Path):
    data_files = {"train": str(dataset_dir / "train.parquet")}
    return load_dataset("parquet", data_files=data_files, split="train")


def ensure_rgb(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def preprocess_image(image) -> torch.Tensor:
    tensor = transforms.ToTensor()(image)  # in [0, 1]
    tensor = tensor * 2 - 1  # scale to [-1, 1] for the VAE
    return tensor.unsqueeze(0)


def tensor_to_pil(tensor: torch.Tensor):
    tensor = tensor.detach().cpu().clamp(-1, 1)
    tensor = (tensor + 1) / 2  # back to [0, 1]
    return transforms.ToPILImage()(tensor.squeeze(0))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_id = load_model_id(args.config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"Loading VAE from {model_id} (subfolder='vae') on {device} ({dtype})")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
    vae.to(device)
    vae.eval()

    ds = load_dataset_split(args.dataset_dir)
    sample = ds[args.index]
    image = ensure_rgb(sample["image"])
    print(f"Loaded sample {args.index} from {args.dataset_dir}")
    print(f"Metadata: file_name={sample.get('file_name')}, split={sample.get('split')}")

    pixel_values = preprocess_image(image).to(device=device, dtype=dtype)

    vae_config = cast(Any, vae.config)
    with torch.no_grad():
        latent_dist = vae.encode(pixel_values).latent_dist
        latents = latent_dist.sample() * vae_config.scaling_factor

    print(f"Latent shape: {tuple(latents.shape)}")
    print(f"Latent mean: {latents.mean().item():.4f}, std: {latents.std().item():.4f}")

    with torch.no_grad():
        decoded = vae.decode(latents / vae_config.scaling_factor).sample

    orig_path = args.output_dir / f"sample_{args.index}_original.png"
    recon_path = args.output_dir / f"sample_{args.index}_reconstruction.png"
    image.save(orig_path)
    tensor_to_pil(decoded).save(recon_path)

    print(f"Saved original to {orig_path}")
    print(f"Saved reconstruction to {recon_path}")

    # Estimate aggregated latent stats over 100 random samples for stability.
    shuffled = ds.shuffle(seed=args.seed)
    num_samples = min(1000, len(shuffled))
    total_sum = 0.0
    total_sq = 0.0
    total_count = 0

    for i in range(num_samples):
        sample_img = ensure_rgb(shuffled[i]["image"])
        pixels = preprocess_image(sample_img).to(device=device, dtype=dtype)
        with torch.no_grad():
            latent_dist = vae.encode(pixels).latent_dist
            sample_latents = latent_dist.sample() * vae_config.scaling_factor
        # Accumulate in float64 to reduce rounding error across the batch of samples.
        sample_latents64 = sample_latents.double()
        total_sum += sample_latents64.sum().item()
        total_sq += (sample_latents64 ** 2).sum().item()
        total_count += sample_latents.numel()

    aggregated_mean = total_sum / total_count
    aggregated_var = total_sq / total_count - aggregated_mean**2
    aggregated_std = aggregated_var**0.5 if aggregated_var > 0 else 0.0
    print(f"Aggregated over {num_samples} random samples -> mean: {aggregated_mean:.4f}, std: {aggregated_std:.4f}")


if __name__ == "__main__":
    main()
