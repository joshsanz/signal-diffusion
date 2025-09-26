"""Shared helpers for diffusion-based training pipelines."""
from __future__ import annotations

import os
import random
from argparse import Namespace
from pathlib import Path
from typing import Mapping, MutableMapping, Optional, Sequence, Tuple

import datasets
import numpy as np
import torch
from accelerate import Accelerator
from datasets import DatasetDict, load_dataset
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer
from torchvision import transforms

__all__ = [
    "DATASET_NAME_MAPPING",
    "GITIGNORE_ENTRIES",
    "get_full_repo_name",
    "setup_repository",
    "build_image_caption_dataloader",
]

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
    """Append missing gitignore entries for artifacts produced during training."""
    gitignore_path = output_dir / ".gitignore"
    existing: Sequence[str]
    if gitignore_path.exists():
        existing = gitignore_path.read_text(encoding="utf-8").splitlines()
    else:
        existing = []

    missing = [entry for entry in entries if entry not in existing]
    if not missing:
        return

    with gitignore_path.open("a", encoding="utf-8") as handle:
        for entry in missing:
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
    dataset_columns = dataset_mapping.get(getattr(args, "dataset_name", None))

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
            transforms.Resize(getattr(args, "resolution", 512), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(getattr(args, "resolution", 512))
            if getattr(args, "center_crop", False)
            else transforms.RandomCrop(getattr(args, "resolution", 512)),
            transforms.RandomHorizontalFlip() if getattr(args, "random_flip", False) else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
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
