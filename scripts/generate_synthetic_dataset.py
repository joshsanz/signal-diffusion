"""Generate synthetic EEG spectrograms using a pretrained diffusion model.

Creates a parquet dataset with controlled attribute distributions (gender, health, age)
using the specified diffusion model for generation.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import typer
from accelerate import Accelerator
from datasets import Dataset, Image as DatasetImage, load_dataset
from PIL import Image
from tqdm.auto import tqdm

from signal_diffusion.diffusion.config import load_diffusion_config
from signal_diffusion.diffusion.image_utils import tensor_to_pil
from signal_diffusion.diffusion.models import registry
from signal_diffusion.diffusion.models.base import load_pretrained_weights
from signal_diffusion.log_setup import get_logger
from signal_diffusion.utils import set_random_seeds
from weighted_dataset_utils import build_caption, prepare_output_dir

logger = get_logger(__name__)

app = typer.Typer()

MIN_AGE = 15
MAX_AGE = 100


def resolve_conditioning_mode(cfg) -> str:
    """Extract conditioning mode from config."""
    conditioning_value = getattr(cfg.model, "conditioning", None)
    if conditioning_value is None:
        conditioning_value = cfg.model.extras.get("conditioning", "none")
    return str(conditioning_value or "none").strip().lower()


def load_unique_combos_from_parquet(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load unique (gender, health, age) combos from a parquet dataset.

    Expected schema:
      - gender: "M" or "F"
      - health: "H" or "PD"
      - age: integer in [MIN_AGE, MAX_AGE]

    Returns numeric arrays aligned to the existing conditioning pipeline:
      - gender: 0=male, 1=female
      - health: 0=healthy, 1=parkinsons
      - age: integer age
    """
    if not path.exists():
        raise typer.BadParameter(f"Input parquet not found: {path}")

    ds = load_dataset("parquet", data_files=str(path), split="train")
    required_columns = {"gender", "health", "age"}
    missing = required_columns.difference(set(ds.column_names))
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise typer.BadParameter(
            f"Input parquet missing required columns: {missing_str}. "
            "Expected columns: gender, health, age."
        )

    # Read columns once; the number of unique combos is small (<= 2*2*(MAX_AGE-MIN_AGE+1)).
    genders = ds["gender"]
    healths = ds["health"]
    ages = ds["age"]

    gender_map = {"M": 0, "F": 1}
    health_map = {"H": 0, "PD": 1}

    combos: set[tuple[int, int, int]] = set()
    for g, h, a in zip(genders, healths, ages):
        if g not in gender_map:
            raise typer.BadParameter(
                f"Invalid gender value {g!r} in input parquet; expected 'M' or 'F'."
            )
        if h not in health_map:
            raise typer.BadParameter(
                f"Invalid health value {h!r} in input parquet; expected 'H' or 'PD'."
            )

        # Cast age to int; parquet sources can store it as string/float depending on upstream.
        # We validate strictly so conditioning is well-defined.
        try:
            age_int = int(a)
        except (TypeError, ValueError) as exc:
            raise typer.BadParameter(
                f"Invalid age value {a!r} in input parquet; expected an integer."
            ) from exc

        if not (MIN_AGE <= age_int <= MAX_AGE):
            raise typer.BadParameter(
                f"Age out of range in input parquet: {age_int} (expected {MIN_AGE}-{MAX_AGE})."
            )

        combos.add((gender_map[g], health_map[h], age_int))

    if not combos:
        raise typer.BadParameter("No rows found in input parquet to derive attribute combos.")

    # Deterministic ordering for reproducibility.
    combos_sorted = sorted(combos, key=lambda t: (t[0], t[1], t[2]))
    gender_arr = np.array([t[0] for t in combos_sorted], dtype=np.int64)
    health_arr = np.array([t[1] for t in combos_sorted], dtype=np.int64)
    age_arr = np.array([t[2] for t in combos_sorted], dtype=np.int64)
    return gender_arr, health_arr, age_arr


def generate_attributes(
    n: int,
    rng: np.random.Generator,
    parkinsons_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate random attributes for n samples.

    Returns:
        Tuple of (gender, health, age) arrays where:
        - gender: 0 (male) or 1 (female), 50/50 split
        - health: 0 (healthy) or 1 (parkinsons), based on parkinsons_ratio
        - age: integers from MIN_AGE to MAX_AGE (inclusive)
    """
    gender = rng.integers(0, 2, size=n)  # 0=male, 1=female
    health = (rng.random(n) < parkinsons_ratio).astype(np.int64)  # 0=healthy, 1=parkinsons
    age = rng.integers(MIN_AGE, MAX_AGE + 1, size=n)  # 18-80 inclusive
    return gender, health, age


def normalize_age(age: int | np.ndarray) -> float | np.ndarray:
    """Normalize age to [0, 1] range for model input."""
    return (age - MIN_AGE) / (MAX_AGE - MIN_AGE)


def prepare_conditioning(
    conditioning_mode: str,
    gender: np.ndarray,
    health: np.ndarray,
    age: np.ndarray,
    device: torch.device,
    num_classes: int,
    rng: np.random.Generator,
) -> dict[str, torch.Tensor] | list[str] | torch.Tensor | None:
    """Prepare conditioning input based on mode."""
    batch_size = len(gender)

    if conditioning_mode == "gend_hlth_age":
        return {
            "gender": torch.tensor(gender, device=device, dtype=torch.long),
            "health": torch.tensor(health, device=device, dtype=torch.long),
            "age": torch.tensor(normalize_age(age), device=device, dtype=torch.float32),
        }
    elif conditioning_mode == "caption":
        captions = []
        for g, h, a in zip(gender, health, age):
            metadata = {
                "gender": "F" if g == 1 else "M",
                "health": "PD" if h == 1 else "H",
                "age": int(a),
            }
            captions.append(build_caption(metadata))
        return captions
    elif conditioning_mode == "classes":
        if num_classes <= 0:
            raise ValueError("num_classes must be > 0 for class conditioning")
        return torch.tensor(
            rng.integers(0, num_classes, size=batch_size),
            device=device,
            dtype=torch.long,
        )
    elif conditioning_mode == "none":
        return None
    else:
        raise ValueError(f"Unsupported conditioning mode: {conditioning_mode}")


@app.command()
def main(
    config: Path = typer.Option(..., "-c", "--config", help="Path to diffusion config TOML file"),
    model_path: Path = typer.Option(
        ..., "-m", "--model-path", help="Path to pretrained model checkpoint (.pt)"
    ),
    output_dir: Path = typer.Option(
        ..., "-o", "--output-dir", help="Output directory for parquet dataset"
    ),
    n: int | None = typer.Option(
        None, "-n", help="Number of samples to generate (ignored with --input-parquet)"
    ),
    input_parquet: Path | None = typer.Option(
        None,
        "--input-parquet",
        help="Parquet file to derive unique gender/health/age combos (one sample per combo)",
    ),
    batch_size: int = typer.Option(4, "-b", "--batch-size", help="Batch size for generation"),
    seed: int = typer.Option(42, help="Random seed"),
    parkinsons_ratio: float = typer.Option(0.25, help="Fraction with Parkinsons"),
    overwrite: bool = typer.Option(False, help="Overwrite existing output"),
):
    """Generate synthetic EEG spectrograms with controlled attribute distributions."""
    # Validate inputs
    if not config.exists():
        raise typer.BadParameter(f"Config file not found: {config}")
    if not model_path.exists():
        raise typer.BadParameter(f"Model checkpoint not found: {model_path}")
    if batch_size <= 0:
        raise typer.BadParameter("batch_size must be > 0")
    if not 0 <= parkinsons_ratio <= 1:
        raise typer.BadParameter("parkinsons_ratio must be between 0 and 1")

    if input_parquet is not None:
        if n is not None:
            raise typer.BadParameter("Do not pass -n when using --input-parquet; n is derived.")
    else:
        if n is None:
            raise typer.BadParameter("Missing required option: -n/--n (or pass --input-parquet).")
        if n <= 0:
            raise typer.BadParameter("n must be > 0")

    # Setup
    set_random_seeds(seed)
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir).expanduser().resolve()
    prepare_output_dir(output_dir, overwrite=overwrite, force=True)

    # Load config
    cfg = load_diffusion_config(config)
    conditioning_mode = resolve_conditioning_mode(cfg)
    logger.info("Conditioning mode: %s", conditioning_mode)

    # Initialize accelerator with mixed precision from config
    accelerator = Accelerator(
        mixed_precision=cfg.training.mixed_precision,
    )
    generator = torch.Generator(device=accelerator.device).manual_seed(seed)

    # Enable TF32 if configured and CUDA is available
    if cfg.training.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TF32 enabled for CUDA matmul and cuDNN operations")

    # Load model
    adapter = registry.get(cfg.model.name)
    tokenizer = adapter.create_tokenizer(cfg)
    modules = adapter.build_modules(accelerator, cfg, tokenizer=tokenizer)

    load_pretrained_weights(
        modules.denoiser,
        model_path,
        logger,
        model_name=cfg.model.name,
    )
    modules.denoiser.to(accelerator.device)
    modules.denoiser.eval()
    logger.info("Loaded model from %s", model_path)

    # Pre-generate all attributes
    if input_parquet is not None:
        all_gender, all_health, all_age = load_unique_combos_from_parquet(input_parquet)
        n = int(len(all_age))
        logger.info("Derived %d unique (gender, health, age) combos from %s", n, input_parquet)
        logger.info(
            "Combo breakdown: %d male / %d female, %d healthy / %d parkinsons",
            (all_gender == 0).sum(),
            (all_gender == 1).sum(),
            (all_health == 0).sum(),
            (all_health == 1).sum(),
        )
        if conditioning_mode in {"classes", "none"}:
            logger.warning(
                "Conditioning mode %r ignores gender/health/age; attributes will be stored "
                "in the output parquet but may not affect generation.",
                conditioning_mode,
            )
    else:
        assert n is not None  # validated above
        all_gender, all_health, all_age = generate_attributes(n, rng, parkinsons_ratio)
        logger.info(
            "Generated attributes: %d male / %d female, %d healthy / %d parkinsons",
            (all_gender == 0).sum(),
            (all_gender == 1).sum(),
            (all_health == 0).sum(),
            (all_health == 1).sum(),
        )

    # Generate samples in batches
    assert n is not None
    typer.echo(f"Will generate {n} samples")
    examples: list[dict[str, Any]] = []
    num_batches = (n + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Generating samples"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n)
        current_batch_size = end_idx - start_idx

        # Extract batch attributes
        batch_gender = all_gender[start_idx:end_idx]
        batch_health = all_health[start_idx:end_idx]
        batch_age = all_age[start_idx:end_idx]

        # Prepare conditioning
        conditioning = prepare_conditioning(
            conditioning_mode,
            batch_gender,
            batch_health,
            batch_age,
            accelerator.device,
            cfg.dataset.num_classes,
            rng,
        )

        # Generate samples
        batch_samples = adapter.generate_conditional_samples(
            accelerator,
            cfg,
            modules,
            current_batch_size,
            denoising_steps=cfg.inference.denoising_steps,
            cfg_scale=cfg.inference.cfg_scale,
            conditioning=conditioning,
            generator=generator,
        )

        # Convert to PIL images and build examples
        for i, tensor in enumerate(batch_samples.detach().cpu()):
            pil_image = tensor_to_pil(tensor)

            gender_str = "F" if batch_gender[i] == 1 else "M"
            health_str = "PD" if batch_health[i] == 1 else "H"
            age_int = int(batch_age[i])

            metadata = {
                "gender": gender_str,
                "health": health_str,
                "age": age_int,
            }
            caption = build_caption(metadata)

            examples.append({
                "image": pil_image,
                "gender": gender_str,
                "health": health_str,
                "age": age_int,
                "caption": caption,
            })

    # Save dataset
    logger.info("Saving dataset with %d samples", len(examples))

    # Create HuggingFace dataset
    dataset_dict = {
        "image": [ex["image"] for ex in examples],
        "gender": [ex["gender"] for ex in examples],
        "health": [ex["health"] for ex in examples],
        "age": [ex["age"] for ex in examples],
        "caption": [ex["caption"] for ex in examples],
    }

    hf_dataset = Dataset.from_dict(dataset_dict)
    hf_dataset = hf_dataset.cast_column("image", DatasetImage())

    output_path = output_dir / "synthetic_samples.parquet"
    hf_dataset.to_parquet(str(output_path))
    logger.info("Saved dataset to %s", output_path)

    # Log summary statistics
    male_count = sum(1 for ex in examples if ex["gender"] == "M")
    female_count = sum(1 for ex in examples if ex["gender"] == "F")
    healthy_count = sum(1 for ex in examples if ex["health"] == "H")
    pd_count = sum(1 for ex in examples if ex["health"] == "PD")

    logger.info("Summary:")
    logger.info("  Total samples: %d", len(examples))
    logger.info("  Gender: %d male (%.1f%%), %d female (%.1f%%)",
                male_count, 100 * male_count / len(examples),
                female_count, 100 * female_count / len(examples))
    logger.info("  Health: %d healthy (%.1f%%), %d parkinsons (%.1f%%)",
                healthy_count, 100 * healthy_count / len(examples),
                pd_count, 100 * pd_count / len(examples))


if __name__ == "__main__":
    app()
