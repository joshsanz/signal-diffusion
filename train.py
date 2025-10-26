
import argparse
import os
from pathlib import Path
from rich.traceback import install
import sys
import tomllib

# Useful for debugging
install(show_locals=False)

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))


def main():
    """
    Main entry point for training.
    """
    parser = argparse.ArgumentParser(description="Run training for classification or diffusion models.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration TOML file.")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional: Path to the output directory.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Optional: Path to a checkpoint directory to resume training from.")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    output_dir = Path(args.output_dir) if args.output_dir else None

    with config_path.open("rb") as f:
        config_data = tomllib.load(f)

    settings = config_data.get("settings", {})
    trainer_type = settings.get("trainer")

    if not trainer_type:
        raise ValueError("The configuration file must specify a 'trainer' under the [settings] table.")

    print(f"Starting training for '{trainer_type}'...")

    if trainer_type == "classification":
        from signal_diffusion.training.classification import load_experiment_config, train_from_config

        experiment_config = load_experiment_config(config_path)
        if output_dir:
            experiment_config.training.output_dir = output_dir
        train_from_config(experiment_config)

    elif trainer_type == "diffusion":
        from signal_diffusion.training.diffusion import train as train_diffusion

        train_diffusion(config_path=config_path, output_dir=output_dir, resume_from_checkpoint=args.resume_from_checkpoint)

    else:
        raise ValueError(f"Unknown trainer type: '{trainer_type}'. Must be 'classification' or 'diffusion'.")

    print("Training finished.")

if __name__ == "__main__":
    main()
