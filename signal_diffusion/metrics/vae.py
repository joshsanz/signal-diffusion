"""Utilities for reconstructing datasets with diffusion VAEs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torchvision.transforms.v2 as v2
from diffusers import AutoencoderKL
from PIL import Image as PilImage
import tqdm.auto as tqdm

from .data import ImageFolderConfig, load_imagefolder_dataset

# Enable TF32 where available to accelerate inference
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@dataclass(frozen=True)
class VAEGenerationConfig:
    """Arguments controlling VAE reconstruction export."""

    dataset: ImageFolderConfig
    model: str
    output_dir: Path
    batch_size: int = 16
    image_size: int = 256
    num_workers: int = 4


def generate_vae_dataset(config: VAEGenerationConfig) -> int:
    """Run a VAE across a dataset and save the reconstructions."""

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device()
    transform = _build_transform(config.image_size)
    dataset = load_imagefolder_dataset(config.dataset, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=config.num_workers,
        collate_fn=_collate_images,
    )

    vae = _load_vae(config.model, device)
    count = 0
    for batch in tqdm.tqdm(dataloader):
        images = batch.to(device)
        with torch.no_grad():
            recon = vae(images)
        recon = recon.sample.permute(0, 2, 3, 1).cpu().float().numpy().clip(-1, 1)
        for image in recon:
            pil_image = _to_pil(image)
            pil_image.save(output_dir / f"recon_{count}.jpg")
            count += 1

    _write_metadata(output_dir, count)
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


def _collate_images(examples: Iterable[dict]) -> torch.Tensor:
    return torch.stack([sample["image"] for sample in examples])


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_vae(model_path: str, device: torch.device) -> AutoencoderKL:
    if "vae" in model_path:
        vae = AutoencoderKL.from_pretrained(model_path)
    else:
        vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
    return vae.to(device).eval()


def _to_pil(image: np.ndarray) -> PilImage:
    image = ((image * 0.5 + 0.5) * 255).astype(np.uint8)
    return PilImage.fromarray(image)


def _write_metadata(output_dir: Path, count: int) -> None:
    metadata_path = output_dir / "metadata.csv"
    with metadata_path.open("w", encoding="utf-8") as handle:
        handle.write("file_name\n")
        for idx in range(count):
            handle.write(f"recon_{idx}.jpg\n")
