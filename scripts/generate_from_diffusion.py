
import torch
import typer
import numpy as np
from pathlib import Path
from accelerate import Accelerator
from accelerate.utils import set_seed
from torchvision.utils import make_grid
from PIL import Image
from typing import List, Optional
from torchvision.transforms import v2 as transforms
from tqdm import tqdm

from signal_diffusion.diffusion.data import build_dataloaders
from signal_diffusion.diffusion.config import load_diffusion_config
from signal_diffusion.diffusion.eval_utils import save_image_grid, _to_uint8
from signal_diffusion.diffusion.models import registry
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from signal_diffusion.diffusion.train_utils import get_sigmas_from_timesteps

app = typer.Typer()


def get_class_labels(dataset: torch.utils.data.Dataset) -> Optional[List[str]]:
    """Extracts class labels from a dataset, supporting both standard and Hugging Face formats."""
    if hasattr(dataset, "classes"):
        return dataset.classes
    if hasattr(dataset, "features"):
        if "label" in dataset.features and hasattr(dataset.features["label"], "names"):
            return dataset.features["label"].names
    return None

def generate_conditional_samples(
    accelerator: Accelerator,
    cfg,
    modules,
    num_images: int,
    class_labels: torch.Tensor,
    *,
    denoising_steps: int,
    generator: torch.Generator,
) -> torch.Tensor:
    scheduler: FlowMatchEulerDiscreteScheduler = FlowMatchEulerDiscreteScheduler.from_config(modules.noise_scheduler.config)
    device = accelerator.device
    dtype = modules.weight_dtype
    scheduler.set_timesteps(denoising_steps, device=device)

    model_config = getattr(modules.denoiser, "config", None)
    channels = getattr(model_config, "in_channels", 3) if model_config is not None else 3
    sample_size = getattr(model_config, "sample_size", None)
    if sample_size is None:
        sample_size = int(cfg.model.sample_size or cfg.dataset.resolution)

    sample = torch.randn((num_images, channels, sample_size, sample_size), generator=generator, device=device, dtype=dtype)

    with torch.no_grad():
        for timestep in tqdm(scheduler.timesteps, desc="Denoising", leave=False):
            model_input = sample
            if hasattr(scheduler, "scale_model_input"):
                model_input = scheduler.scale_model_input(model_input, timestep)
            timesteps = torch.ones(sample.size(0)).to(device) * timestep
            sigmas = get_sigmas_from_timesteps(scheduler, timesteps, device=device)
            model_output = modules.denoiser(
                model_input, sigma=sigmas, class_cond=class_labels
            )

            step_output = scheduler.step(model_output, timestep, sample, return_dict=True)
            sample = step_output.prev_sample

    return sample.to(dtype=torch.float32).detach()


@app.command()
def main(
    model_path: Path = typer.Option(..., help="Path to the pretrained diffusion model checkpoint (.pt file)"),
    dataset_config: Path = typer.Option(..., help="Path to the dataset configuration TOML file"),
    n: int = typer.Option(4, help="Number of examples to generate per class"),
    c: int = typer.Option(4, help="Number of classes to generate for"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
    batch_size: int = typer.Option(4, help="Batch size for generation"),
    unconditioned: bool = typer.Option(False, help="Generate column without class conditioning"),
    output_filename: str = typer.Option("generated_grid.jpg", help="Output filename for the image grid"),
):
    """
    Generate images from a pretrained diffusion model and arrange them in a grid.
    """
    accelerator = Accelerator()
    generator = torch.Generator(device=accelerator.device).manual_seed(seed)

    cfg = load_diffusion_config(dataset_config)
    d_config = cfg.dataset

    train_loader, _ = build_dataloaders(d_config, tokenizer=None, settings_path=None)
    dataset = train_loader.dataset

    class_names = get_class_labels(dataset)
    if not class_names:
        raise ValueError("Could not extract class names from the dataset.")

    num_classes = len(class_names)

    num_to_select = min(c, num_classes)
    selected_class_indices = np.random.choice(range(num_classes), num_to_select, replace=False)
    selected_classes = [(i, class_names[i]) for i in selected_class_indices]
    print(f"Selected classes: {[name for _, name in selected_classes]}")
    if unconditioned:
        selected_classes.append(num_classes)
        print("Also generating with no class conditioning.")

    # Get original images
    original_images = []
    for class_idx, _ in selected_classes:
        class_dataset = dataset.filter(lambda example: example["class_labels"] == class_idx)
        if len(class_dataset) > 0:
            random_index = np.random.randint(0, len(class_dataset))
            image = class_dataset[random_index]["pixel_values"]
            original_images.append(transforms.functional.to_pil_image(_to_uint8(image)))
    if unconditioned:
        image_size = (d_config.resolution, d_config.resolution)
        black_image = Image.new("RGB", image_size, (0, 0, 0))
        original_images.append(black_image)

    # Load model
    adapter = registry.get(cfg.model.name)
    modules = adapter.build_modules(accelerator, cfg)
    state_dict = torch.load(model_path, map_location="cpu")
    modules.denoiser.load_state_dict(state_dict)
    modules.denoiser.to(accelerator.device)
    modules.denoiser.eval()

    # Generate images
    generated_images_rows = []
    for i in range(n):
        print(f"Generating row {i+1}/{n}")
        row_images = []
        for class_idx, _ in selected_classes:
            for _ in range((n + batch_size - 1) // batch_size):
                labels = torch.ones(batch_size, device=accelerator.device, dtype=torch.long) * class_idx
                generated_batch = generate_conditional_samples(
                    accelerator, cfg, modules, batch_size, labels,
                    denoising_steps=cfg.inference.denoising_steps,
                    generator=generator,
                )
                row_images.append(generated_batch[0])
        generated_images_rows.append(row_images)
    if unconditioned:
        row_images = []
        for _ in range((n + batch_size - 1) // batch_size):
            labels = torch.ones(batch_size, device=accelerator.device, dtype=torch.long) * num_classes
            generated_batch = generate_conditional_samples(
                    accelerator, cfg, modules, batch_size, labels,
                    denoising_steps=cfg.inference.denoising_steps,
                    generator=generator,
                )
            row_images.append(generated_batch[0])

    # Create grid
    grid_rows = [original_images] + generated_images_rows

    flat_image_list = [img for row in grid_rows for img in row]

    pil_images = []
    for img in flat_image_list:
        if not isinstance(img, Image.Image):
            # Assuming tensor, convert to PIL
            img = transforms.ToPILImage()(img.cpu())
        pil_images.append(img)

    if pil_images:
        grid = make_grid(
            [transforms.ToImage()(img) for img in pil_images],
            nrow=len(selected_classes) if not no_class_conditioning else c,
        )

        to_pil = transforms.ToPILImage()
        grid_image = to_pil(grid)
        grid_image.save(output_filename)
        print(f"Saved image grid to {output_filename}")
    else:
        print("No images were generated or found to create a grid.")

if __name__ == "__main__":
    app()
