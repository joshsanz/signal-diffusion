import torch
import typer
import numpy as np
from pathlib import Path
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import DiffusionPipeline
from torchvision.utils import make_grid
from PIL import Image
import tomllib
from typing import List, Optional
from torchvision.transforms import v2 as transforms

from signal_diffusion.diffusion.data import build_dataloaders, DatasetConfig
from signal_diffusion.diffusion.eval_utils import save_image_grid

app = typer.Typer()

def get_class_labels(dataset: torch.utils.data.Dataset) -> Optional[List[str]]:
    """Extracts class labels from a dataset, supporting both standard and Hugging Face formats."""
    if hasattr(dataset, "classes"):
        return dataset.classes
    if hasattr(dataset, "features"):
        if "label" in dataset.features and hasattr(dataset.features["label"], "names"):
            return dataset.features["label"].names
    return None

@app.command()
def main(
    model_path: Path = typer.Option(..., help="Path to the pretrained diffusion model pipeline"),
    dataset_config: Path = typer.Option(..., help="Path to the dataset configuration TOML file"),
    n: int = typer.Option(4, help="Number of examples to generate per class"),
    c: int = typer.Option(4, help="Number of classes to generate for"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
    batch_size: int = typer.Option(4, help="Batch size for generation"),
    no_class_conditioning: bool = typer.Option(False, help="Generate without class conditioning"),
    output_filename: str = typer.Option("generated_grid.jpg", help="Output filename for the image grid"),
):
    """
    Generate images from a pretrained diffusion model and arrange them in a grid.
    """
    set_seed(seed)
    accelerator = Accelerator()

    with dataset_config.open("rb") as f:
        dataset_cfg_data = tomllib.load(f)
    
    d_config = DatasetConfig(**dataset_cfg_data['dataset'])

    train_loader, _ = build_dataloaders(d_config, tokenizer=None, settings_path=None)
    dataset = train_loader.dataset

    class_names = get_class_labels(dataset.dataset)
    if not class_names:
        raise ValueError("Could not extract class names from the dataset.")
    
    num_classes = len(class_names)

    if no_class_conditioning:
        selected_classes = []
        print("Generating with no class conditioning.")
    else:
        num_to_select = min(c, num_classes)
        selected_class_indices = np.random.choice(range(num_classes), num_to_select, replace=False)
        selected_classes = [(i, class_names[i]) for i in selected_class_indices]
        print(f"Selected classes: {[name for _, name in selected_classes]}")

    # Get original images
    original_images = []
    if no_class_conditioning:
        # Use black images as placeholders
        image_size = (d_config.resolution, d_config.resolution)
        black_image = Image.new("RGB", image_size, (0, 0, 0))
        original_images = [black_image] * min(c, 1)
    else:
        for class_idx, _ in selected_classes:
            class_dataset = dataset.dataset.filter(lambda example: example[d_config.class_column] == class_idx)
            if len(class_dataset) > 0:
                random_index = np.random.randint(0, len(class_dataset))
                image = class_dataset[random_index]["image"]
                original_images.append(image.resize((d_config.resolution, d_config.resolution)))

    # Load pipeline
    pipeline = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipeline = pipeline.to(accelerator.device)

    # Generate images
    generated_images_rows = []
    for i in range(n):
        print(f"Generating row {i+1}/{n}")
        row_images = []
        if no_class_conditioning:
            generated_batch = pipeline(
                batch_size=c,
                generator=torch.Generator(device=accelerator.device).manual_seed(seed + i),
            ).images
            row_images.extend(generated_batch)
        else:
            for class_idx, _ in selected_classes:
                generated_batch = pipeline(
                    class_labels=[class_idx] * batch_size,
                    batch_size=batch_size,
                    generator=torch.Generator(device=accelerator.device).manual_seed(seed + i * len(selected_classes) + class_idx),
                ).images
                row_images.extend(generated_batch[:1]) # Take one image per class per row
        generated_images_rows.append(row_images)

    # Create grid
    grid_rows = [original_images] + generated_images_rows
    
    # Flatten the list of lists and create the grid
    flat_image_list = [img for row in grid_rows for img in row]
    
    # Ensure all images are PIL Images
    pil_images = []
    for img in flat_image_list:
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        pil_images.append(img)

    # Create a grid
    if pil_images:
        grid = make_grid(
            [transforms.ToImage()(img) for img in pil_images],
            nrow=len(selected_classes) if not no_class_conditioning else c,
        )
        
        # Convert tensor to PIL Image and save
        to_pil = transforms.ToPILImage()
        grid_image = to_pil(grid)
        grid_image.save(output_filename)
        print(f"Saved image grid to {output_filename}")
    else:
        print("No images were generated or found to create a grid.")

if __name__ == "__main__":
    app()