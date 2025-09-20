"""Meta-dataset for combining multiple EEG spectrogram datasets."""
from __future__ import annotations

import importlib
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, WeightedRandomSampler

from signal_diffusion.config import Settings


class MetaPreprocessor:
    """Orchestrates preprocessing for multiple datasets."""

    def __init__(self, settings: Settings, dataset_names: Sequence[str], **kwargs: Any) -> None:
        self.settings = settings
        self.dataset_names = dataset_names
        self.preprocessor_kwargs = kwargs

    def preprocess(self, **kwargs: Any) -> None:
        for name in self.dataset_names:
            print(f"Preprocessing {name} dataset...")
            module = importlib.import_module(f"signal_diffusion.data.{name}")
            if name in ("mit", "seed"):
                preprocessor_class_name = f"{name.upper()}Preprocessor"
            else:
                preprocessor_class_name = f"{name.capitalize()}Preprocessor"
            preprocessor_class = getattr(module, preprocessor_class_name)
            preprocessor = preprocessor_class(self.settings, **self.preprocessor_kwargs)
            preprocessor.preprocess(**kwargs)


class MetaDataset(ConcatDataset):
    """Concatenates multiple datasets."""

    def __init__(self, settings: Settings, dataset_names: Sequence[str], split: str, tasks: Sequence[str] = ("gender",), **kwargs: Any) -> None:
        self.settings = settings
        self.dataset_names = dataset_names
        self.split = split
        self.tasks = tasks

        datasets = []
        for name in self.dataset_names:
            module = importlib.import_module(f"signal_diffusion.data.{name}")
            if name in ("mit", "seed"):
                dataset_class_name = f"{name.upper()}Dataset"
            else:
                dataset_class_name = f"{name.capitalize()}Dataset"
            dataset_class = getattr(module, dataset_class_name)
            datasets.append(dataset_class(settings, split=split, tasks=tasks, **kwargs))

        super().__init__(datasets)


class MetaSampler(WeightedRandomSampler):
    """Weighted sampler for meta-dataset."""

    def __init__(self, dataset: MetaDataset, num_samples: int, replacement: bool = True, generator=None) -> None:
        self.dataset = dataset
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator

        weights = self._generate_weights()
        super().__init__(weights, num_samples, replacement=replacement, generator=generator)

    def _generate_weights(self) -> torch.Tensor:
        metadatas = []
        for ds in self.dataset.datasets:
            metadata_path = ds.dataset_settings.output / f"{self.dataset.split}-metadata.csv"
            if not metadata_path.exists():
                metadata_path = ds.dataset_settings.output / "metadata.csv"
            metadatas.append(pd.read_csv(metadata_path))

        totals, male, female = [], 0, 0

        for metadata in metadatas:
            genders = metadata["gender"].astype(str).str.upper()
            totals.append(len(metadata))
            female += genders.str.startswith("F").sum()
            male += genders.str.startswith("M").sum()

        total_samps = sum(totals)
        if total_samps == 0:
            return torch.empty(0)
            
        gend_weights = [(male / total_samps) if total_samps > 0 else 0, (female / total_samps) if total_samps > 0 else 0]
        dataset_weights = [total / total_samps for total in totals]

        summed_weights = []
        weights = []
        for dw in dataset_weights:
            data_gender_w = []
            for gw in gend_weights:
                data_gender_w.append(gw + dw)
            summed_weights.append(sum(data_gender_w))
            weights.append(data_gender_w)

        rankings = {}
        for label in range(len(metadatas)):
            label_weight = summed_weights[label]
            rank = 0
            for weight in summed_weights:
                if label_weight > weight:
                    rank += 1
            rankings[label] = rank

        new_label_weights = [weights[(len(metadatas) - 1) - rankings[i]] for i in range(len(metadatas))]
        output_weights = []

        for i, metadata in enumerate(metadatas):
            genders = np.array(metadata["gender"].astype(str).str.upper())
            # Default to 0 weight if new_label_weights is not long enough
            female_weight = new_label_weights[i][0] if i < len(new_label_weights) and len(new_label_weights[i]) > 0 else 0
            male_weight = new_label_weights[i][1] if i < len(new_label_weights) and len(new_label_weights[i]) > 1 else 0
            
            gender_weights = np.where(np.char.startswith(genders, 'F'), female_weight, male_weight)
            output_weights.extend(gender_weights.tolist())

        if not output_weights:
            return torch.empty(0)

        norm_fact = sum(output_weights)
        if norm_fact == 0:
            return torch.ones(len(output_weights)) / len(output_weights)
            
        final_weights = [w / norm_fact for w in output_weights]

        return torch.as_tensor(final_weights, dtype=torch.double)