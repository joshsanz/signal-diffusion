# Diffusion Training Workflow

Use `signal_diffusion.training.diffusion` to launch text-to-image or flow-matching runs with a model registry that currently includes Stable Diffusion v1.5 (`stable-diffusion-v1-5`) and DiT (`dit`).

## Quickstart

```bash
# Default smoke config for DiT
scripts/run_diffusion_training.sh

# Explicit config + output override
uv run python -m signal_diffusion.training.diffusion config/diffusion/flowers.toml --output-dir runs/diffusion/demo
```

Configurations live under `config/diffusion/` and support:

- Dataset parameters (`[dataset]`) with optional project-wide settings integration.
- Model selection (`[model]`) plus registry-specific extras and LoRA toggles (`[model.lora]`).
- Objective choice (`[objective]`) between noise prediction (`epsilon`) and flow matching (`vector_field`).
- Optimizer and scheduler controls, logging destinations, and runtime knobs (mixed precision, EMA, etc.).

For Stable Diffusion LoRA finetuning, set `model.name = "stable-diffusion-v1-5"` and enable `[model.lora]`. Ensure a compatible diffusers version that exposes `LoRACrossAttnProcessor` is installed.

## Validation

Run a short-step sanity check by lowering `training.epochs` or setting `training.max_train_steps`. Monitor metrics via TensorBoard when `logging.tensorboard = true` or Weights & Biases if configured.
