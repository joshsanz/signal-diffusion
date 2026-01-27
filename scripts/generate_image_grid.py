from __future__ import annotations

from collections.abc import Iterable, Mapping, Sized
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import typer
from accelerate import Accelerator
from PIL import Image
from torchvision.transforms import v2 as transforms
from torchvision.utils import make_grid

from signal_diffusion.diffusion.config import load_diffusion_config
from signal_diffusion.diffusion.data import build_dataloaders
from signal_diffusion.diffusion.image_utils import tensor_to_pil
from signal_diffusion.diffusion.models import registry

app = typer.Typer()


def get_class_labels(dataset: torch.utils.data.Dataset) -> Optional[list[str]]:
    """Extract class labels from datasets produced by torchvision or Hugging Face."""
    classes = getattr(dataset, "classes", None)
    if isinstance(classes, Iterable):
        return [str(value) for value in classes]
    features = getattr(dataset, "features", None)
    if isinstance(features, Mapping):
        label_feature = features.get("label")
    elif hasattr(features, "get"):
        label_feature = features.get("label")
    else:
        label_feature = None
        if label_feature is not None and hasattr(label_feature, "names"):
            return [str(value) for value in label_feature.names]
    return None


def resolve_conditioning_mode(cfg) -> str:
    conditioning_value = getattr(cfg.model, "conditioning", None)
    if conditioning_value is None:
        conditioning_value = cfg.model.extras.get("conditioning", "none")
    return str(conditioning_value or "none").strip().lower()


@app.command()
def main(
    model_path: Path = typer.Option(..., help="Path to the pretrained diffusion model checkpoint (.pt file)"),
    dataset_config: Path = typer.Option(..., help="Path to the dataset configuration TOML file"),
    n: int = typer.Option(4, help="Number of examples to generate per class"),
    c: int = typer.Option(4, help="Number of classes to generate for"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
    batch_size: int = typer.Option(4, help="Batch size for generation"),
    unconditioned: bool = typer.Option(False, help="Add an unconditional column to the grid"),
    output_filename: str = typer.Option("generated_grid.jpg", help="Output filename for the image grid"),
):
    """
    Generate images from a pretrained diffusion model and arrange them in a grid.
    """
    accelerator = Accelerator()
    generator = torch.Generator(device=accelerator.device).manual_seed(seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    cfg = load_diffusion_config(dataset_config)
    d_config = cfg.dataset

    train_loader, _ = build_dataloaders(d_config, tokenizer=None, settings_path=None, data_type=cfg.settings.data_type)
    dataset = train_loader.dataset

    conditioning_mode = resolve_conditioning_mode(cfg)

    class_names = get_class_labels(dataset) if conditioning_mode == "classes" else None
    if conditioning_mode == "classes" and not class_names:
        raise ValueError("Class conditioning requested but dataset does not expose class names.")
    if conditioning_mode == "classes":
        assert class_names is not None
        class_names_list = class_names

    condition_indices: list[tuple[Optional[int], str]] = []
    if conditioning_mode == "classes":
        num_classes = len(class_names_list)
        num_to_select = min(c, num_classes)
        selected_ids = np.random.choice(range(num_classes), num_to_select, replace=False)
        condition_indices = [(int(idx), str(class_names_list[idx])) for idx in selected_ids]
        print(f"Selected classes: {[name for _, name in condition_indices]}")
    elif conditioning_mode == "caption":
        raise NotImplementedError("Caption-based sampling is not supported by this script yet.")
    elif conditioning_mode == "gend_hlth_age":
        raise NotImplementedError("Multi-attribute (gend_hlth_age) sampling is not supported by this script yet.")
    elif conditioning_mode == "none":
        condition_indices = []
    else:
        raise ValueError(
            f"Unsupported conditioning mode '{conditioning_mode}'. "
            f"Must be one of: 'none', 'classes', 'caption', 'gend_hlth_age'"
        )

    adapter = registry.get(cfg.model.name)
    tokenizer = adapter.create_tokenizer(cfg)
    modules = adapter.build_modules(accelerator, cfg, tokenizer=tokenizer)

    state_dict = torch.load(model_path, map_location="cpu")
    modules.denoiser.load_state_dict(state_dict)
    modules.denoiser.to(accelerator.device)
    modules.denoiser.eval()

    conditions: list[tuple[Optional[int], str]] = condition_indices.copy()
    if unconditioned:
        conditions.append((None, "unconditional"))
        print("Including unconditional samples.")

    if not conditions:
        conditions.append((None, "generated"))

    image_height = int(d_config.resolution)
    if not isinstance(dataset, Sized):
        raise TypeError("Dataset must be sized to sample reference images.")
    indices = np.arange(len(dataset))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    reference_images: list[torch.Tensor] = []
    for idx, _ in conditions:
        if idx is None:
            reference_images.append(torch.full((3, image_height, image_height), -1.0, dtype=torch.float32))
            continue
        reference_tensor: torch.Tensor | None = None
        for dataset_index in indices:
            example = dataset[dataset_index]
            class_value = example.get("class_labels") if isinstance(example, dict) else example[1]
            if int(class_value) == idx:
                pixel_values = example["pixel_values"] if isinstance(example, dict) else example[0]
                reference_tensor = (
                    pixel_values.detach().cpu()
                    if isinstance(pixel_values, torch.Tensor)
                    else transforms.ToImage()(pixel_values) * 2.0 - 1.0
                )
                break
        if reference_tensor is None:
            raise ValueError(f"No samples found for class index {idx}")
        reference_images.append(reference_tensor)

    generated_by_condition: dict[Optional[int], list[torch.Tensor]] = {idx: [] for idx, _ in conditions}

    for idx, name in conditions:
        remaining = n
        print(f"Generating {n} samples for '{name}'")
        while remaining > 0:
            current_batch = min(batch_size, remaining)
            if idx is None:
                conditioning_input: torch.Tensor | None = None
            else:
                conditioning_input = torch.full(
                    (current_batch,),
                    idx,
                    device=accelerator.device,
                    dtype=torch.long,
                )

            batch_samples = adapter.generate_conditional_samples(
                accelerator,
                cfg,
                modules,
                current_batch,
                denoising_steps=cfg.inference.denoising_steps,
                cfg_scale=cfg.inference.cfg_scale,
                conditioning=conditioning_input,
                generator=generator,
            )
            for tensor in batch_samples.detach().cpu():
                generated_by_condition[idx].append(tensor)
            remaining -= current_batch

        generated_by_condition[idx] = generated_by_condition[idx][:n]

    grid_rows: list[list[torch.Tensor]] = [reference_images]
    for row_idx in range(n):
        row: list[torch.Tensor] = []
        for idx, _ in conditions:
            row.append(generated_by_condition[idx][row_idx])
        grid_rows.append(row)

    flat = [tensor for row in grid_rows for tensor in row]
    if not flat:
        raise RuntimeError("No images available to assemble grid.")

    stacked = torch.stack(flat, dim=0)
    nrow = len(conditions)
    grid = make_grid(stacked, nrow=nrow)
    grid_image = tensor_to_pil(grid)
    grid_image.save(output_filename)
    print(f"Saved image grid to {output_filename}")

if __name__ == "__main__":
    app()
