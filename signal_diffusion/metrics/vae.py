"""Utilities for reconstructing datasets with diffusion VAEs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, cast

import numpy as np
import torch
import torchvision.transforms.v2 as v2
from diffusers import AutoencoderKL
from datasets import Dataset as HFDataset, Image as HFImage
from PIL import Image
import tqdm.auto as tqdm

from .data import ParquetDatasetConfig, load_parquet_dataset

# Enable TF32 where available to accelerate inference
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@dataclass(frozen=True)
class VAEGenerationConfig:
    """Arguments controlling VAE reconstruction export."""

    dataset: ParquetDatasetConfig
    model: str
    output_dir: Path
    batch_size: int = 16
    image_size: int = 256
    num_workers: int = 4
    dtype: torch.dtype = torch.bfloat16


def generate_vae_dataset(config: VAEGenerationConfig) -> int:
    """Run a VAE across a dataset and save the reconstructions."""

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device()
    transform = _build_transform(config.image_size)
    dataset = load_parquet_dataset(config.dataset, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=config.num_workers,
        collate_fn=lambda examples: _collate_examples(examples, config.dataset.image_key),
    )

    vae = _load_vae(config.model, device, config.dtype)
    vae_dtype = next(vae.parameters()).dtype
    reconstructed: list[dict[str, Any]] = []
    count = 0
    for batch in tqdm.tqdm(dataloader):
        images = batch["images"].to(device, dtype=vae_dtype)
        with torch.no_grad():
            recon = vae(images)
        recon = recon.sample.permute(0, 2, 3, 1).cpu().float().numpy().clip(-1, 1)
        for meta, image in zip(batch["metadata"], recon):
            updated = dict(meta)
            updated[config.dataset.image_key] = _to_pil(image)
            reconstructed.append(updated)
            count += 1

    output_path = output_dir / f"{config.dataset.split}.parquet"
    _write_parquet_dataset(reconstructed, output_path, config.dataset.image_key)
    return count


def _build_transform(image_size: int) -> v2.Compose:
    return v2.Compose(
        [
            v2.Resize((image_size, image_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5], [0.5]),
        ]
    )


def _collate_examples(examples: Iterable[dict], image_key: str) -> dict[str, Any]:
    images = torch.stack([sample[image_key] for sample in examples])
    metadata = []
    for sample in examples:
        entry = dict(sample)
        entry.pop(image_key, None)
        metadata.append(entry)
    return {"images": images, "metadata": metadata}


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_vae(model_path: str, device: torch.device, dtype: torch.dtype) -> AutoencoderKL:
    if "vae" in model_path:
        vae = AutoencoderKL.from_pretrained(model_path, torch_dtype=dtype)
    else:
        vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype)
    return vae.to(device).eval()


def _to_pil(image: np.ndarray) -> Image.Image:
    image = ((image * 0.5 + 0.5) * 255).astype(np.uint8)
    return Image.fromarray(image)

def _write_parquet_dataset(
    rows: list[dict[str, Any]],
    output_path: Path,
    image_key: str,
) -> None:
    if not rows:
        raise ValueError("No reconstructed samples were generated.")
    dataset = HFDataset.from_list(rows)
    dataset = dataset.cast_column(image_key, HFImage())
    dataset.to_parquet(str(output_path))
