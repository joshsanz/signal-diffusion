"""Shared helpers for diffusion-based training pipelines."""
from __future__ import annotations

import gc
import math
import os
import random
from collections import deque
from argparse import Namespace
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

import datasets
import numpy as np
import torch
from accelerate import Accelerator
from datasets import DatasetDict, load_dataset
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTokenizer
from torchvision.transforms import v2 as transforms

from signal_diffusion.diffusion.eval_utils import _to_uint8, compute_kid_score, save_image_grid

__all__ = [
    "resolve_output_dir",
    "build_optimizer",
    "log_generated_samples",
    "run_evaluation",
    "flatten_and_sanitize_hparams",
    "should_evaluate",
    "DATASET_NAME_MAPPING",
    "GITIGNORE_ENTRIES",
    "get_full_repo_name",
    "setup_repository",
    "build_image_caption_dataloader",
]

# ---------------------------------------------------------------------------
# Diffusion training run utilities
# ---------------------------------------------------------------------------


def resolve_output_dir(cfg: Any, config_path: Path, override: Optional[Path]) -> Path:
    """Determine the output directory for a diffusion run."""
    if override is not None:
        return override
    configured = getattr(cfg.training, "output_dir", None)
    if configured:
        return Path(configured)
    runs_root = Path("runs") / "diffusion"
    runs_root.mkdir(parents=True, exist_ok=True)
    return runs_root / config_path.stem


def build_optimizer(parameters: Iterable[torch.nn.Parameter], cfg: Any) -> torch.optim.Optimizer:
    """Construct the optimizer used during diffusion training."""
    params = list(parameters)
    if not params:
        raise ValueError("No trainable parameters found for optimizer")

    optimizer_name = cfg.optimizer.name.lower()
    lr = cfg.optimizer.learning_rate
    betas = cfg.optimizer.betas
    eps = cfg.optimizer.eps
    weight_decay = cfg.optimizer.weight_decay

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "adamw_8bit":
        try:
            import bitsandbytes as bnb  # type: ignore[unresolved-import]
        except ImportError as exc:
            raise ImportError(
                "bitsandbytes is required for adamw_8bit optimizer. "
                "Install with: pip install bitsandbytes"
            ) from exc
        return bnb.optim.AdamW8bit(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer '{cfg.optimizer.name}'")



def _compute_per_channel_stats(tensor: torch.Tensor) -> dict[str, dict[str, float]]:
    """Computes per-channel min, max, mean, and std for a given tensor."""
    stats: dict[str, dict[str, float]] = {}
    if tensor.ndim != 4:
        return stats
    for i in range(tensor.shape[1]):
        channel_data = tensor[:, i, :, :]
        stats[f"ch{i}"] = {
            "min": channel_data.min().item(),
            "max": channel_data.max().item(),
            "mean": channel_data.mean().item(),
            "std": channel_data.std().item(),
        }
    return stats


def _build_caption(age: int, gender: str, health: str, modality: str) -> str:
    # Inline caption building (avoiding import of build_caption for simplicity)
    gender_word = "female" if gender == "F" else "male"
    if health == "PD":
        health_primary = None
        health_clause = "with Parkinson's disease"
    else:
        health_primary = "healthy"
        health_clause = None

    primary_bits = [f"{age} year old"]
    if health_primary:
        primary_bits.append(health_primary)
    primary_bits.append(gender_word)

    if modality == "spectrogram":
        modality_desc = "spectrogram image"
    elif modality == "timeseries":
        modality_desc = "timeseries"
    else:
        raise ValueError(f"Unknown modality '{modality}' for caption building")

    caption = f"a {modality_desc} of a {', '.join(primary_bits)} subject"
    if health_clause:
        caption += f" {health_clause}"
    return caption


def log_generated_samples(
    accelerator: Accelerator,
    samples: torch.Tensor,
    run_dir: Path,
    global_step: int,
    *,
    num_images: int,
    grid_cols: int = 4,
    log_prefix: str = "eval",
) -> None:
    """Log generated samples to disk and experiment trackers.

    Mirrors the image logging behavior used in run_evaluation so it can be reused
    by debug scripts that pre-generate samples.
    """
    if num_images <= 0:
        return
    if samples.ndim != 4:
        raise ValueError("Expected generated samples tensor shaped (N, C, H, W)")

    grid_images = samples[:num_images].detach().cpu()
    grid_path = run_dir / f"{global_step:06d}.jpg"
    save_image_grid(grid_images, grid_path, cols=grid_cols)

    images_uint8 = _to_uint8(grid_images)
    images_np = images_uint8.permute(0, 2, 3, 1).numpy()

    # Log to wandb using log_images for proper image display
    try:
        wandb_tracker = accelerator.get_tracker("wandb")
    except ValueError:
        wandb_tracker = None
    if wandb_tracker:
        import wandb
        images_list = [wandb.Image(img) for img in images_np]
        wandb_tracker.log({f"{log_prefix}/generated_samples": images_list}, step=global_step)
        wandb_tracker.log({f"{log_prefix}/generated_samples_hist": grid_images.flatten()}, step=global_step)


    # Log to tensorboard using log_images
    try:
        tb_tracker = accelerator.get_tracker("tensorboard")
    except ValueError:
        tb_tracker = None
    if tb_tracker:
        tb_tracker.log({f"{log_prefix}/generated_samples": grid_images}, step=global_step)  # type: ignore[arg-type]

    # Log to MLflow using mlflow.log_image API
    try:
        mlflow_tracker = accelerator.get_tracker("mlflow")
    except ValueError:
        mlflow_tracker = None
    if mlflow_tracker and accelerator.is_main_process:
        import mlflow

        # Log each image individually using uint8 numpy arrays
        for idx, img_np in enumerate(images_np):
            mlflow.log_image(
                img_np,
                key=f"{log_prefix}/generated_samples/image_{idx}",
                step=global_step,
            )


def run_evaluation(
    accelerator: Accelerator,
    adapter: Any,
    cfg: Any,
    modules: Any,
    train_loader: Any,
    val_loader: Any,
    run_dir: Path,
    global_step: int,
) -> dict[str, float]:
    """Generate samples and compute evaluation metrics for the current step."""
    eval_examples = getattr(cfg.training, "eval_num_examples", 0) or 0
    eval_mmd_samples = getattr(cfg.training, "eval_mmd_samples", 0) or 0
    if eval_examples <= 0 and eval_mmd_samples <= 0:
        return {}

    num_generate = max(eval_examples, eval_mmd_samples, 1)

    eval_gen_seed = getattr(cfg.training, "eval_gen_seed", None)
    generator = None
    if eval_gen_seed is not None:
        generator = torch.Generator(device=accelerator.device).manual_seed(eval_gen_seed)

    # Extract conditioning mode from config
    default_conditioning = "caption" if cfg.model.name.startswith("stable-diffusion") else "none"
    conditioning_mode = cfg.model.conditioning
    if conditioning_mode is None:
        conditioning_mode = default_conditioning

    eval_batch_size = cfg.training.eval_batch_size or cfg.dataset.batch_size
    was_training = modules.denoiser.training
    modules.denoiser.eval()
    generated_samples = []
    try:
        with torch.no_grad():
            num_classes = int(getattr(cfg.dataset, "num_classes", 0) or 0)
            unconditional_eval = 0
            if eval_examples > 0:
                unconditional_eval = max(1, math.ceil(eval_examples * 0.25))
                unconditional_eval = min(unconditional_eval, eval_examples)
            class_eval = max(eval_examples - unconditional_eval, 0)

            # Determine if we should use conditional generation
            use_conditioning = (
                (conditioning_mode == "classes" and num_classes > 0)
                or conditioning_mode == "gend_hlth_age"
                or conditioning_mode == "caption"
            )

            conditioning_plan: list[tuple[str, int]] = []
            if unconditional_eval > 0:
                conditioning_plan.append(("uncond", unconditional_eval))
            if class_eval > 0:
                conditioning_plan.append((conditioning_mode if use_conditioning else "uncond", class_eval))

            remaining = num_generate - eval_examples
            if remaining > 0:
                conditioning_plan.append((conditioning_mode if use_conditioning else "uncond", remaining))

            if not conditioning_plan:
                conditioning_plan.append(("uncond", num_generate))

            task_queue = deque(conditioning_plan)
            total_remaining = num_generate
            pbar = tqdm(
                total=num_generate,
                desc="Generating samples",
                disable=not accelerator.is_main_process,
                dynamic_ncols=True,
                leave=True,
            )
            while total_remaining > 0 and task_queue:
                task_type, task_count = task_queue[0]
                batch_size = min(eval_batch_size, total_remaining, task_count)
                if batch_size <= 0:
                    task_queue.popleft()
                    continue

                conditioning_input: torch.Tensor | dict[str, torch.Tensor] | str | list[str] | None
                if task_type == "classes" and num_classes > 0:
                    conditioning_input = torch.randint(
                        low=0,
                        high=num_classes,
                        size=(batch_size,),
                        device=accelerator.device,
                        generator=generator,
                    )
                elif task_type == "gend_hlth_age":
                    # Generate random gender (0 or 1), health (0 or 1), and age (20-80)
                    gender = torch.randint(0, 2, (batch_size,), device=accelerator.device, generator=generator)
                    health = torch.randint(0, 2, (batch_size,), device=accelerator.device, generator=generator)
                    # Sample age uniformly from 20-80 years
                    age = torch.rand(batch_size, device=accelerator.device, generator=generator) * 60 + 20
                    conditioning_input = {
                        "gender": gender,
                        "health": health,
                        "age": age,
                    }
                elif task_type == "caption":
                    # Generate random captions using the dataset caption format
                    captions = []
                    for _ in range(batch_size):
                        # Generate random attributes
                        age_val = int(random.randint(20, 80))
                        gender_code = random.choice(["M", "F"])
                        health_code = random.choice(["H", "PD"])

                        # Build caption using the weighted dataset format
                        age_val_i = int(age_val)
                        gender_code_s = str(gender_code)
                        health_code_s = str(health_code)
                        modality_s = str(cfg.settings.data_type)
                        caption = _build_caption(
                            age=age_val_i,
                            gender=gender_code_s,
                            health=health_code_s,
                            modality=modality_s,
                        )
                        captions.append(caption)

                    conditioning_input = captions
                else:
                    conditioning_input = None

                generated_batch = adapter.generate_conditional_samples(
                    accelerator,
                    cfg,
                    modules,
                    batch_size,
                    denoising_steps=cfg.inference.denoising_steps,
                    cfg_scale=cfg.inference.cfg_scale,
                    conditioning=conditioning_input,
                    generator=generator,
                )
                generated_samples.append(generated_batch.cpu())

                total_remaining -= batch_size
                task_count -= batch_size
                if task_count <= 0:
                    task_queue.popleft()
                else:
                    task_queue[0] = (task_type, task_count)
                pbar.update(batch_size)
        generated = torch.cat(generated_samples, dim=0)
    finally:
        if was_training:
            modules.denoiser.train()
        else:
            modules.denoiser.eval()

    if generated.ndim != 4:
        raise ValueError("Adapter.generate_conditional_samples must return a tensor shaped (N, C, H, W)")

    metrics: dict[str, float] = {}

    if eval_examples > 0:
        log_generated_samples(
            accelerator,
            generated,
            run_dir,
            global_step,
            num_images=eval_examples,
            grid_cols=4,
            log_prefix="eval",
        )

    if eval_mmd_samples > 0:
        gen_for_kid = generated[:eval_mmd_samples]
        if gen_for_kid.shape[0] == 0:
            raise ValueError("Not enough generated samples to compute KID score")

        if val_loader is not None:
            ref_dataset = val_loader.dataset
        else:
            ref_dataset = train_loader.dataset

        kid_mean, kid_std = compute_kid_score(gen_for_kid, ref_dataset)
        metrics["eval/kid_mean"] = kid_mean
        metrics["eval/kid_std"] = kid_std

    # Free intermediate buffers before returning
    del generated_samples
    del generated
    if eval_examples > 0:
        del grid_images
    if eval_mmd_samples > 0:
        del gen_for_kid

    return metrics


def flatten_and_sanitize_hparams(config_dict: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested config structures for logging-friendly hyperparameters."""
    hparams: dict[str, Any] = {}
    if not isinstance(config_dict, dict):
        return hparams

    for key, value in config_dict.items():
        composed_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            hparams.update(flatten_and_sanitize_hparams(value, prefix=composed_key))
        elif isinstance(value, (str, int, float, bool)):
            hparams[composed_key] = value
        elif isinstance(value, (list, tuple)) and all(
            isinstance(item, (str, int, float, bool)) for item in value
        ):
            hparams[composed_key] = str(value)
    return hparams


def should_evaluate(global_step: int, steps_per_epoch: int, training_cfg: Any) -> bool:
    """Decide whether evaluation should run at the current training step."""
    strategy = (getattr(training_cfg, "eval_strategy", "epoch") or "epoch").strip().lower()
    if strategy == "epoch":
        step_in_epoch = global_step % steps_per_epoch
        return step_in_epoch == (steps_per_epoch - 1)
    if strategy == "steps":
        interval = getattr(training_cfg, "eval_num_steps", 0) or 0
        return interval > 0 and global_step > 0 and global_step % interval == 0
    return False


# ---------------------------------------------------------------------------
# Dataset preparation utilities used by the legacy fine-tuning scripts
# ---------------------------------------------------------------------------

# Mapping of known dataset identifiers to their (image_column, caption_column) pair.
DATASET_NAME_MAPPING: Mapping[str, Tuple[str, str]] = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}

# Entries that should be ignored when cloning/pushing training artifacts.
GITIGNORE_ENTRIES: Tuple[str, ...] = ("step_*", "epoch_*")


def get_full_repo_name(
    model_id: str,
    organization: Optional[str] = None,
    token: Optional[str] = None,
) -> str:
    """Resolve the full repository path for Hugging Face Hub uploads."""
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    return f"{organization}/{model_id}"


def _ensure_gitignore_entries(output_dir: Path, entries: Sequence[str]) -> None:
    """Write gitignore entries for artifacts produced during training."""
    gitignore_path = output_dir / ".gitignore"
    existing: Sequence[str]
    if gitignore_path.exists():
        existing = gitignore_path.read_text(encoding="utf-8").splitlines()
    else:
        existing = []

    missing = [entry for entry in entries if entry not in existing]
    if not missing:
        return

    all_entries = list(existing) + missing
    with gitignore_path.open("w", encoding="utf-8") as handle:
        for entry in all_entries:
            handle.write(f"{entry}\n")


def setup_repository(
    accelerator: Accelerator,
    args: Namespace,
    *,
    gitignore_entries: Sequence[str] = GITIGNORE_ENTRIES,
) -> Optional[Repository]:
    """Prepare the output directory and optional Hugging Face Hub repository.

    Parameters
    ----------
    accelerator:
        The active :class:`accelerate.Accelerator` instance.
    args:
        Namespace holding script arguments. Must provide ``output_dir`` and the
        Hugging Face hub-related flags used in the existing scripts.
    gitignore_entries:
        Patterns to append to ``.gitignore`` when pushing to the hub.

    Returns
    -------
    Repository or None
        The configured :class:`huggingface_hub.Repository` for the main process,
        otherwise ``None``.
    """
    output_dir = getattr(args, "output_dir", None)
    if output_dir is None:
        return None

    repo: Optional[Repository] = None
    output_path = Path(output_dir)

    if accelerator.is_main_process:
        if getattr(args, "push_to_hub", False):
            repo_name = getattr(args, "hub_model_id", None) or get_full_repo_name(
                output_path.name, token=getattr(args, "hub_token", None)
            )
            repo_id = create_repo(repo_name, exist_ok=True, token=getattr(args, "hub_token", None))
            repo = Repository(str(output_path), clone_from=repo_id, token=getattr(args, "hub_token", None))
            _ensure_gitignore_entries(output_path, gitignore_entries)
        else:
            output_path.mkdir(parents=True, exist_ok=True)

    return repo


def build_image_caption_dataloader(
    accelerator: Accelerator,
    args: Namespace,
    tokenizer: CLIPTokenizer,
    batch_size: int,
    *,
    dataset_name_mapping: Optional[Mapping[str, Tuple[str, str]]] = None,
) -> Tuple[datasets.Dataset, DataLoader]:
    """Construct the training dataset and dataloader for image-caption fine-tuning."""
    dataset_mapping = dataset_name_mapping or DATASET_NAME_MAPPING

    if getattr(args, "dataset_name", None):
        dataset = load_dataset(
            args.dataset_name,
            getattr(args, "dataset_config_name", None),
            cache_dir=getattr(args, "cache_dir", None),
        )
    else:
        data_files: MutableMapping[str, str] = {}
        train_dir = getattr(args, "train_data_dir", None)
        if train_dir:
            data_files["train"] = os.path.join(train_dir, "**", "*")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files or None,
            cache_dir=getattr(args, "cache_dir", None),
        )

    if not isinstance(dataset, DatasetDict) or "train" not in dataset:
        raise ValueError("Expected a dataset with a 'train' split for diffusion training.")

    column_names = dataset["train"].column_names
    dataset_name = getattr(args, "dataset_name", None)
    dataset_columns = dataset_mapping.get(dataset_name) if isinstance(dataset_name, str) else None

    image_column = getattr(args, "image_column", None)
    if image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    elif image_column not in column_names:
        raise ValueError(f"image column '{image_column}' not found in dataset columns: {column_names}")

    caption_column = getattr(args, "caption_column", None)
    if caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    elif caption_column not in column_names:
        raise ValueError(f"caption column '{caption_column}' not found in dataset columns: {column_names}")

    def tokenize_captions(examples: Mapping[str, Sequence[str]], is_train: bool = True) -> torch.Tensor:
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, tuple, np.ndarray)):
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain strings or lists of strings."
                )
        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids

    train_transforms = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Resize(getattr(args, "resolution", 512), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(getattr(args, "resolution", 512))
            if getattr(args, "center_crop", False)
            else transforms.RandomCrop(getattr(args, "resolution", 512)),
            transforms.RandomHorizontalFlip() if getattr(args, "random_flip", False) else transforms.Lambda(lambda x: x),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    def preprocess_train(examples: Mapping[str, Sequence]) -> Mapping[str, Sequence]:
        images = [image.convert("RGB") for image in examples[image_column]]
        examples = dict(examples)
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    with accelerator.main_process_first():
        max_train_samples = getattr(args, "max_train_samples", None)
        seed = getattr(args, "seed", None)
        if max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=seed).select(range(max_train_samples))
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples: Sequence[Mapping[str, torch.Tensor]]) -> Mapping[str, torch.Tensor]:
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=getattr(args, "dataloader_num_workers", 0),
    )

    return train_dataset, train_dataloader
