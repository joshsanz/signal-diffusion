"""CLI for generating VAE reconstruction datasets."""

import argparse
from pathlib import Path

from signal_diffusion.metrics import (
    ImageFolderConfig,
    VAEGenerationConfig,
    generate_vae_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate VAE reconstructions for KID baselines")
    parser.add_argument("-d", "--dataset", required=True, help="Path to source dataset")
    parser.add_argument("-m", "--model", required=True, help="Diffusers model or checkpoint path")
    parser.add_argument("-o", "--output", required=True, help="Directory to store reconstructed images")
    parser.add_argument("-b", "--batch-size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("-s", "--image-size", type=int, default=256, help="Resize edge length before encoding")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader worker processes")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    config = VAEGenerationConfig(
        dataset=ImageFolderConfig(args.dataset),
        model=args.model,
        output_dir=output_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )
    total = generate_vae_dataset(config)
    print(f"Saved {total} reconstructions to {output_dir}")


if __name__ == "__main__":
    main(parse_args())
