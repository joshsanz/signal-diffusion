#!/usr/bin/env python3
"""Inspect model parameters defined by classification or diffusion configs."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

import tomllib
import torch
import typer


app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_show_locals=False)


def _detect_config_kind(config_path: Path) -> str:
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)
    model_section = data.get("model") or {}
    if "name" in model_section:
        return "diffusion"
    if "backbone" in model_section:
        return "classification"
    raise ValueError(
        "Unable to determine config kind. Expected [model] section to contain either "
        "'name' (diffusion) or 'backbone' (classification)."
    )


def _format_int(value: int) -> str:
    return f"{value:,}"


def _summarize_named_parameters(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    include_frozen: bool,
) -> tuple[list[tuple[str, int, tuple[int, ...]]], Counter[tuple[int, ...]], int]:
    entries: list[tuple[str, int, tuple[int, ...]]] = []
    shape_counts: Counter[tuple[int, ...]] = Counter()
    total_params = 0
    for name, param in named_parameters:
        if not include_frozen and not param.requires_grad:
            continue
        count = param.numel()
        shape = tuple(param.shape)
        entries.append((name, count, shape))
        shape_counts[shape] += 1
        total_params += count
    entries.sort(key=lambda item: item[0])
    return entries, shape_counts, total_params


def _iter_diffusion_parameters(modules) -> Iterator[tuple[str, torch.nn.Parameter]]:
    components = [
        ("denoiser", getattr(modules, "denoiser", None)),
        ("text_encoder", getattr(modules, "text_encoder", None)),
        ("vae", getattr(modules, "vae", None)),
    ]
    for prefix, module in components:
        if module is None:
            continue
        for name, param in module.named_parameters():
            yield f"{prefix}.{name}", param


def _load_classification_model(config_path: Path):
    from signal_diffusion.classification import ClassifierConfig, build_classifier, build_task_specs
    from signal_diffusion.classification.config import load_classification_config

    cfg = load_classification_config(config_path)
    task_specs = build_task_specs(cfg.dataset.name, cfg.dataset.tasks)
    classifier_config = ClassifierConfig(
        backbone=cfg.model.backbone,
        input_channels=cfg.model.input_channels,
        tasks=task_specs,
        embedding_dim=cfg.model.embedding_dim,
        dropout=cfg.model.dropout,
        activation=cfg.model.activation,
        depth=cfg.model.depth,
        layer_repeats=cfg.model.layer_repeats,
        extras=dict(cfg.model.extras),
    )
    model = build_classifier(classifier_config)
    return model


def _load_diffusion_modules(config_path: Path, *, device: str):
    from accelerate import Accelerator

    from signal_diffusion.diffusion import load_diffusion_config
    try:
        from signal_diffusion.diffusion.models import registry
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            f"Failed to import diffusion adapters because '{exc.name}' is missing. "
            "Install the required dependency (e.g., `uv pip install mamba-ssm`)."
        ) from exc

    cfg = load_diffusion_config(config_path)

    default_conditioning = "caption" if cfg.model.name.startswith("stable-diffusion") else "none"
    conditioning_value = cfg.model.conditioning
    if conditioning_value is None:
        conditioning_value = cfg.model.extras.get("conditioning", default_conditioning)
    conditioning = str(conditioning_value or "").strip().lower() or default_conditioning
    allowed_conditioning = {"none", "caption", "classes", "gend_hlth_age"}
    if conditioning not in allowed_conditioning:
        raise ValueError(f"Unsupported conditioning type '{conditioning}'. Expected one of {sorted(allowed_conditioning)}.")

    tokenizer = None
    adapter = registry.get(cfg.model.name)
    if conditioning == "caption":
        tokenizer = adapter.create_tokenizer(cfg)

    # Limit everything to CPU unless user explicitly asks for cuda.
    cpu_requested = device.lower() == "cpu"
    accelerator = Accelerator(cpu=cpu_requested, mixed_precision="no")
    modules = adapter.build_modules(accelerator, cfg, tokenizer=tokenizer)
    return modules


def _print_parameter_tree(
    entries: list[tuple[str, int, tuple[int, ...]]],
    shape_counts: Counter[tuple[int, ...]],
    total_params: int,
) -> None:
    typer.echo("Full parameter tree:")
    for name, count, shape in entries:
        typer.echo(f"  {name}: {_format_int(count)} {list(shape)}")
    typer.echo(f"\nTotal parameters: {_format_int(total_params)} across {len(entries)} tensors")
    typer.echo("\nParameter tensors by shape:")
    for shape, count in shape_counts.most_common():
        typer.echo(f"  {list(shape)}: {count}")


@app.command()
def inspect(
    config_path: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, resolve_path=True, help="Path to a classification or diffusion config."),
    device: str = typer.Option("cpu", "--device", help="Device to place the instantiated model on ('cpu' or 'cuda')."),
    include_frozen: bool = typer.Option(False, "--include-frozen", help="Include parameters with requires_grad=False."),
) -> None:
    """Load a model from config and display parameter statistics."""
    config_kind = _detect_config_kind(config_path)
    typer.echo(f"Detected {config_kind} config: {config_path}")

    if config_kind == "classification":
        model = _load_classification_model(config_path)
        if device != "cpu":
            model.to(device)
        named_params = list(model.named_parameters())
        entries, shape_counts, total_params = _summarize_named_parameters(named_params, include_frozen=include_frozen)
        _print_parameter_tree(entries, shape_counts, total_params)
        return

    modules = _load_diffusion_modules(config_path, device=device)
    named_params = list(_iter_diffusion_parameters(modules))
    entries, shape_counts, total_params = _summarize_named_parameters(named_params, include_frozen=include_frozen)
    _print_parameter_tree(entries, shape_counts, total_params)


if __name__ == "__main__":
    app()
