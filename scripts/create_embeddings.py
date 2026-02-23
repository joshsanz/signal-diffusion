#!/usr/bin/env python3
"""Embed images from parquet datasets using embeddings models.

This script reads images from two parquet files (synthetic + real),
extracts embeddings from selected models, and writes a
combined parquet with:
  - age, gender, health
  - label: 'F' for synthetic, 'T' for real
  - subject: subject id for labeled/real rows (empty for synthetic rows), derived from file path
  - embedding columns: cnn_embedding, siglipv2_embedding, sscd_embedding (fixed-size float32 vectors)
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm.auto import tqdm
from datasets import Image as HFImage
from datasets import load_dataset
from torchvision import transforms
from transformers import AutoModel, AutoProcessor

from signal_diffusion.classification.config import load_classification_config
from signal_diffusion.classification.datasets import default_transform
from signal_diffusion.classification.factory import ClassifierConfig, build_classifier, tasks_from_registry
from signal_diffusion.data.meta import META_LABELS
from signal_diffusion.data.metadata_utils import (
    build_caption,
    normalize_age,
    normalize_gender,
    normalize_health,
)


BASELINE_TOML = Path(__file__).resolve().parent.parent / "config" / "classification" / "baseline.toml"
DEFAULT_SIGLIP_MODEL = "google/siglip2-so400m-patch14-384"
DEFAULT_SSCD_MODEL = Path(__file__).resolve().parent.parent / "models" / "sscd_disc_mixup.torchscript.pt"
SSCD_MODEL_URL = "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt"
SSCD_EMBEDDING_DIM = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract CNN embeddings from two parquet datasets and write a combined parquet."
    )
    parser.add_argument("--synthetic-parquet", type=Path, required=True, help="Path to synthetic parquet file")
    parser.add_argument("--label-parquet", type=Path, required=True, help="Path to labeled/real parquet file")
    parser.add_argument("--output", type=Path, required=True, help="Output parquet path")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained classifier checkpoint (.pt)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for embedding extraction")
    parser.add_argument(
        "--siglip-model",
        type=str,
        default=DEFAULT_SIGLIP_MODEL,
        help=f"SigLIP v2 checkpoint id (default: {DEFAULT_SIGLIP_MODEL})",
    )
    parser.add_argument(
        "--siglip-batch-size",
        type=int,
        default=4,
        help="SigLIP v2 sub-batch size for image feature extraction",
    )
    parser.add_argument(
        "--sscd-model",
        type=Path,
        default=DEFAULT_SSCD_MODEL,
        help=f"Path to SSCD TorchScript model (default: {DEFAULT_SSCD_MODEL}). "
        "If missing, downloads from Facebook Research.",
    )
    parser.add_argument(
        "--sscd-batch-size",
        type=int,
        default=16,
        help="SSCD sub-batch size for image feature extraction",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on rows per parquet (useful for quick tests).",
    )
    return parser.parse_args()


def _load_backbone_and_transform(
    *,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, int, str]:
    cfg = load_classification_config(BASELINE_TOML)

    tasks = tasks_from_registry(META_LABELS, cfg.dataset.tasks)
    model = build_classifier(
        ClassifierConfig(
            backbone=cfg.model.backbone,
            input_channels=cfg.model.input_channels,
            tasks=tasks,
            embedding_dim=cfg.model.embedding_dim,
            dropout=cfg.model.dropout,
            activation=cfg.model.activation,
            depth=cfg.model.depth,
            layer_repeats=cfg.model.layer_repeats,
            extras=dict(cfg.model.extras),
        )
    )

    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(device)

    backbone = model.backbone
    backbone.eval()

    output_type = str(cfg.data_overrides.get("output_type", "db-only"))
    transform = default_transform(output_type)

    embedding_dim = int(getattr(backbone, "embedding_dim"))
    return backbone, transform, embedding_dim, output_type


def _load_siglip_model_and_processor(
    *,
    model_id: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.nn.Module, Any, int]:
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    model = AutoModel.from_pretrained(model_id, dtype=dtype)
    model.eval()
    model.to(device)

    # get_image_features returns pooled visual representation.
    output_dim = int(model.config.vision_config.hidden_size)
    return model, processor, output_dim


def _ensure_sscd_model(path: Path) -> Path:
    """Ensure SSCD model exists; download from URL if missing."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading SSCD model from {SSCD_MODEL_URL} to {path}...")
    urllib.request.urlretrieve(SSCD_MODEL_URL, path)
    return path


def _load_sscd_model(*, model_path: Path, device: torch.device) -> tuple[torch.nn.Module, Any]:
    """Load SSCD TorchScript model and return (model, transform).

    SSCD expects RGB images resized to 320x320 with ImageNet normalization.
    """
    path = _ensure_sscd_model(model_path)
    model = torch.jit.load(str(path))
    model.eval()
    model.to(device)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    transform = transforms.Compose([
        transforms.Resize([320, 320]),
        transforms.ToTensor(),
        normalize,
    ])
    return model, transform


def _images_to_sscd_embeddings(
    images: list[Any],
    *,
    sscd_model: torch.nn.Module,
    sscd_transform: Any,
    device: torch.device,
    sscd_batch_size: int,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for start in range(0, len(images), sscd_batch_size):
        sub = images[start : start + sscd_batch_size]
        sub_rgb = [img.convert("RGB") for img in sub]
        tensors = [sscd_transform(img) for img in sub_rgb]
        batch = torch.stack(tensors, dim=0).to(device)
        with torch.no_grad():
            feats = sscd_model(batch)
        outputs.append(feats.to(torch.float32))
    return torch.cat(outputs, dim=0)


def _iter_dataset_batches(ds, *, batch_size: int, max_rows: int | None) -> Iterable[dict[str, list[Any]]]:
    limit = len(ds) if max_rows is None else min(len(ds), max_rows)
    for start in range(0, limit, batch_size):
        batch = ds[start : min(start + batch_size, limit)]
        # `datasets` slicing returns dict[str, list]
        yield batch


def _derive_subject_from_file_name(file_name: str | None) -> str:
    """Extract subject identifier from file_name path.

    Expected:
      train/sub-324/spectrogram-0.png -> sub-324
    """

    text = str(file_name).replace("\\", "/").strip("/")
    parts = [p for p in PurePosixPath(text).parts if p]

    if parts[0].lower() in {"train", "val", "validation", "test"} and len(parts) >= 2:
        return parts[1]
    if len(parts) >= 2:
        return parts[-2]
    return parts[0]


def _as_string(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return text


def _prepare_metadata_row(batch: dict[str, list[Any]], idx: int) -> dict[str, Any]:
    age = normalize_age(batch["age"][idx])
    gender = normalize_gender(batch["gender"][idx])
    health = normalize_health(batch["health"][idx])

    file_name_val = batch.get("file_name", [None])[idx] if "file_name" in batch else None
    file_name_str = _as_string(file_name_val).strip() or None
    subject = _derive_subject_from_file_name(file_name_str) if file_name_str else ""

    return {
        "age": age,
        "gender": gender,
        "health": health,
        "subject": subject,
    }


def _images_to_tensor(
    images: list[Any],
    *,
    transform: Any,
    output_type: str,
) -> torch.Tensor:
    mode = "L" if output_type == "db-only" else "RGB"
    processed = [transform(img.convert(mode)) for img in images]
    return torch.stack(processed, dim=0)


def _images_to_siglip_embeddings(
    images: list[Any],
    *,
    siglip_model: torch.nn.Module,
    siglip_processor: Any,
    device: torch.device,
    siglip_batch_size: int,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []

    for start in range(0, len(images), siglip_batch_size):
        sub = images[start : start + siglip_batch_size]
        # SigLIP expects RGB images.
        sub_rgb = [img.convert("RGB") for img in sub]
        inputs = siglip_processor(images=sub_rgb, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        try:
            with torch.no_grad():
                feats = siglip_model.get_image_features(**inputs)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise RuntimeError(
                    "SigLIP OOM during get_image_features. Reduce --siglip-batch-size "
                    "(e.g. 4 -> 2 -> 1) or switch --device cpu."
                ) from exc
            raise
        outputs.append(feats.to(torch.float32))

    return torch.cat(outputs, dim=0)


def _ensure_parquet_file(path: Path, *, arg_name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{arg_name} not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"{arg_name} must be a parquet file, got directory: {path}")


def _load_parquet_dataset(path: Path, *, output_type: str):
    ds = load_dataset("parquet", data_files={"data": str(path)}, split="data")
    if "image" not in ds.column_names:
        raise KeyError(f"Parquet is missing required 'image' column: {path}")
    mode = "L" if output_type == "db-only" else "RGB"
    # Ensure HF decodes the image column to PIL on access.
    ds = ds.cast_column("image", HFImage(decode=True, mode=mode))
    return ds


def _make_schema(
    *,
    cnn_embedding_dim: int,
    siglip_embedding_dim: int,
    sscd_embedding_dim: int = SSCD_EMBEDDING_DIM,
) -> pa.Schema:
    cnn_embedding_type = pa.list_(pa.float32(), list_size=cnn_embedding_dim)
    siglip_embedding_type = pa.list_(pa.float32(), list_size=siglip_embedding_dim)
    sscd_embedding_type = pa.list_(pa.float32(), list_size=sscd_embedding_dim)
    return pa.schema(
        [
            ("age", pa.int32()),
            ("gender", pa.string()),
            ("health", pa.string()),
            ("subject", pa.string()),
            ("label", pa.string()),
            ("cnn_embedding", cnn_embedding_type),
            ("siglipv2_embedding", siglip_embedding_type),
            ("sscd_embedding", sscd_embedding_type),
        ]
    )


def _write_chunk(
    writer: pq.ParquetWriter,
    *,
    schema: pa.Schema,
    metadata_rows: list[dict[str, Any]],
    label_value: str,
    cnn_embeddings: np.ndarray,
    siglip_embeddings: np.ndarray,
    sscd_embeddings: np.ndarray,
) -> None:
    if cnn_embeddings.ndim != 2:
        raise ValueError(f"Expected cnn embeddings with shape (B, D), got {cnn_embeddings.shape}")
    if siglip_embeddings.ndim != 2:
        raise ValueError(f"Expected siglip embeddings with shape (B, D), got {siglip_embeddings.shape}")
    if sscd_embeddings.ndim != 2:
        raise ValueError(f"Expected sscd embeddings with shape (B, D), got {sscd_embeddings.shape}")
    n = cnn_embeddings.shape[0]
    if siglip_embeddings.shape[0] != n or sscd_embeddings.shape[0] != n:
        raise ValueError(
            f"Embedding batch sizes must match: cnn={n}, siglip={siglip_embeddings.shape[0]}, "
            f"sscd={sscd_embeddings.shape[0]}"
        )
    if len(metadata_rows) != n:
        raise ValueError(
            f"Metadata rows ({len(metadata_rows)}) must match embedding rows ({n})"
        )

    ages = [row["age"] for row in metadata_rows]
    genders = [row["gender"] for row in metadata_rows]
    healths = [row["health"] for row in metadata_rows]
    subjects = [row["subject"] for row in metadata_rows]
    labels = [label_value] * len(metadata_rows)

    # Fixed-size list arrays for embeddings.
    cnn_values = pa.array(cnn_embeddings.reshape(-1), type=pa.float32())
    cnn_emb = pa.FixedSizeListArray.from_arrays(cnn_values, cnn_embeddings.shape[1])
    siglip_values = pa.array(siglip_embeddings.reshape(-1), type=pa.float32())
    siglip_emb = pa.FixedSizeListArray.from_arrays(siglip_values, siglip_embeddings.shape[1])
    sscd_values = pa.array(sscd_embeddings.reshape(-1), type=pa.float32())
    sscd_emb = pa.FixedSizeListArray.from_arrays(sscd_values, sscd_embeddings.shape[1])

    table = pa.table(
        {
            "age": pa.array(ages, type=pa.int32()),
            "gender": pa.array(genders, type=pa.string()),
            "health": pa.array(healths, type=pa.string()),
            "subject": pa.array(subjects, type=pa.string()),
            "label": pa.array(labels, type=pa.string()),
            "cnn_embedding": cnn_emb,
            "siglipv2_embedding": siglip_emb,
            "sscd_embedding": sscd_emb,
        },
        schema=schema,
    )
    writer.write_table(table)


def _embed_and_write(
    *,
    ds,
    dataset_label: str,
    backbone: torch.nn.Module,
    siglip_model: torch.nn.Module,
    siglip_processor: Any,
    sscd_model: torch.nn.Module,
    sscd_transform: Any,
    transform: Any,
    output_type: str,
    device: torch.device,
    batch_size: int,
    siglip_batch_size: int,
    sscd_batch_size: int,
    max_rows: int | None,
    writer: pq.ParquetWriter,
    schema: pa.Schema,
) -> int:
    written = 0
    total = len(ds) if max_rows is None else min(len(ds), max_rows)
    desc = "Embedding synthetic" if dataset_label == "F" else "Embedding labeled"
    progress = tqdm(total=total, desc=desc, unit="img")

    for batch in _iter_dataset_batches(ds, batch_size=batch_size, max_rows=max_rows):
        images = batch["image"]
        pixel_values = _images_to_tensor(images, transform=transform, output_type=output_type)
        pixel_values = pixel_values.to(device=device, dtype=torch.float32, non_blocking=True)

        with torch.no_grad():
            cnn_features = backbone(pixel_values)
        cnn_features_np = cnn_features.detach().to("cpu").to(torch.float32).numpy()

        siglip_features = _images_to_siglip_embeddings(
            images,
            siglip_model=siglip_model,
            siglip_processor=siglip_processor,
            device=device,
            siglip_batch_size=siglip_batch_size,
        )
        siglip_features_np = siglip_features.detach().to("cpu").to(torch.float32).numpy()

        sscd_features = _images_to_sscd_embeddings(
            images,
            sscd_model=sscd_model,
            sscd_transform=sscd_transform,
            device=device,
            sscd_batch_size=sscd_batch_size,
        )
        sscd_features_np = sscd_features.detach().to("cpu").to(torch.float32).numpy()

        meta_rows = [_prepare_metadata_row(batch, i) for i in range(len(images))]
        _write_chunk(
            writer,
            schema=schema,
            metadata_rows=meta_rows,
            label_value=dataset_label,
            cnn_embeddings=cnn_features_np,
            siglip_embeddings=siglip_features_np,
            sscd_embeddings=sscd_features_np,
        )
        written += len(images)
        progress.update(len(images))

    progress.close()
    return written


def main() -> None:
    args = parse_args()

    _ensure_parquet_file(args.synthetic_parquet, arg_name="--synthetic-parquet")
    _ensure_parquet_file(args.label_parquet, arg_name="--label-parquet")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.siglip_batch_size <= 0:
        raise ValueError("--siglip-batch-size must be positive")
    if args.sscd_batch_size <= 0:
        raise ValueError("--sscd-batch-size must be positive")
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be positive when provided")

    device = torch.device("cuda")

    backbone, transform, cnn_embedding_dim, output_type = _load_backbone_and_transform(
        checkpoint_path=args.checkpoint,
        device=device,
    )
    siglip_model, siglip_processor, siglip_embedding_dim = _load_siglip_model_and_processor(
        model_id=args.siglip_model,
        device=device,
        dtype=torch.float16,
    )
    sscd_model, sscd_transform = _load_sscd_model(
        model_path=args.sscd_model,
        device=device,
    )

    synthetic = _load_parquet_dataset(args.synthetic_parquet, output_type=output_type)
    labeled = _load_parquet_dataset(args.label_parquet, output_type=output_type)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    schema = _make_schema(
        cnn_embedding_dim=cnn_embedding_dim,
        siglip_embedding_dim=siglip_embedding_dim,
    )

    print(
        "Resolved encoders: "
        f"cnn_dim={cnn_embedding_dim}, "
        f"siglip_model={args.siglip_model}, "
        f"siglip_dim={siglip_embedding_dim}, "
        f"sscd_model={args.sscd_model}, "
        f"sscd_dim={SSCD_EMBEDDING_DIM}, "
        f"batch_size={args.batch_size}, "
        f"siglip_batch_size={args.siglip_batch_size}, "
        f"sscd_batch_size={args.sscd_batch_size}, "
        f"siglip_dtype=float16, "
        f"device={device}"
    )

    with pq.ParquetWriter(where=str(args.output), schema=schema) as writer:
        synthetic_written = _embed_and_write(
            ds=synthetic,
            dataset_label="F",
            backbone=backbone,
            siglip_model=siglip_model,
            siglip_processor=siglip_processor,
            sscd_model=sscd_model,
            sscd_transform=sscd_transform,
            transform=transform,
            output_type=output_type,
            device=device,
            batch_size=args.batch_size,
            siglip_batch_size=args.siglip_batch_size,
            sscd_batch_size=args.sscd_batch_size,
            max_rows=args.max_rows,
            writer=writer,
            schema=schema,
        )
        labeled_written = _embed_and_write(
            ds=labeled,
            dataset_label="T",
            backbone=backbone,
            siglip_model=siglip_model,
            siglip_processor=siglip_processor,
            sscd_model=sscd_model,
            sscd_transform=sscd_transform,
            transform=transform,
            output_type=output_type,
            device=device,
            batch_size=args.batch_size,
            siglip_batch_size=args.siglip_batch_size,
            sscd_batch_size=args.sscd_batch_size,
            max_rows=args.max_rows,
            writer=writer,
            schema=schema,
        )

    total_written = synthetic_written + labeled_written
    print(
        f"Wrote embeddings to {args.output} "
        f"(cnn_embedding_dim={cnn_embedding_dim}, siglipv2_embedding_dim={siglip_embedding_dim}, "
        f"sscd_embedding_dim={SSCD_EMBEDDING_DIM}, output_type={output_type}, "
        f"embeddings_generated={total_written})"
    )


if __name__ == "__main__":
    main()

