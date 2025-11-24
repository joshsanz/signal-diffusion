"""
Preprocess all datasets to generate spectrograms and metadata.
"""
import argparse
from pathlib import Path

from signal_diffusion.config import load_settings
from signal_diffusion.data.meta import MetaPreprocessor
from signal_diffusion.log_setup import get_logger

logger = get_logger(__name__)


def main():
    """
    Runs the preprocessing for all datasets.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None, help="Path to the config file.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing data.")
    parser.add_argument("--all", action="store_true", help="Preprocess all datasets defined in the config.")
    parser.add_argument("datasets", nargs="*", default=["math", "parkinsons", "seed", "longitudinal"], help="Datasets to preprocess.")
    args = parser.parse_args()

    settings = load_settings(args.config)
    dataset_names = list(settings.datasets.keys()) if args.all else args.datasets
    logger.info(f"Loading datasets from          {settings.data_root}")
    logger.info(f"Saving preprocessed outputs to {settings.output_root}")

    preprocessor = MetaPreprocessor(
        settings=settings,
        dataset_names=dataset_names,
        nsamps=2000,
        ovr_perc=0.5,
        fs=125,
        bin_spacing="log",
    )
    preprocessor.preprocess(
        splits={"train": 0.8, "val": 0.1, "test": 0.1},
        resolution=256,
        overwrite=args.overwrite,
    )
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
