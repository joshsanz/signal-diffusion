# Memory Profiling Guide for OOM Debugging

This guide shows how to use the MemoryProfiler to debug OOM errors in diffusion training.

## Problem Description

OOM errors occur when switching from train to eval mode, even though the initial eval (before training) succeeds. This suggests memory is accumulating during training and not being released.

## Integration Steps

### 1. Import the MemoryProfiler

Add to `signal_diffusion/training/diffusion.py`:

```python
from signal_diffusion.training.memory_profiler import MemoryProfiler
```

### 2. Initialize the profiler

After creating the accelerator (around line 346), add:

```python
# Initialize memory profiler
mem_profiler = MemoryProfiler(enabled=accelerator.is_main_process)
```

### 3. Profile after model initialization

After line 360 (after EMA initialization), add:

```python
if accelerator.is_main_process:
    mem_profiler.set_baseline("after_model_init")
    mem_profiler.comprehensive_profile(
        "after_model_init",
        model=modules.denoiser,
        ema_model=modules.ema,
        optimizer=None,  # Optimizer not created yet
        vae=modules.vae if hasattr(modules, 'vae') else None,
        text_encoder=modules.text_encoder if hasattr(modules, 'text_encoder') else None,
    )
```

### 4. Profile after optimizer initialization

After line 473 (after `modules.parameters = ...`), add:

```python
if accelerator.is_main_process:
    mem_profiler.comprehensive_profile(
        "after_optimizer_init",
        model=modules.denoiser,
        ema_model=modules.ema,
        optimizer=optimizer,
        vae=modules.vae if hasattr(modules, 'vae') else None,
        text_encoder=modules.text_encoder if hasattr(modules, 'text_encoder') else None,
    )
```

### 5. Profile during training loop

Add these profiling points in the training loop:

#### a. After training step (inside `if accelerator.sync_gradients:` block, around line 614)

```python
if accelerator.sync_gradients:
    global_step += 1

    # Profile first few steps and periodically
    if accelerator.is_main_process and (global_step <= 3 or global_step % 100 == 0):
        mem_profiler.log_memory(f"train_step_{global_step}/after_step")
```

#### b. After EMA update (after line 612)

```python
if modules.ema is not None and accelerator.sync_gradients:
    modules.ema.step(accelerator.unwrap_model(modules.denoiser).parameters())

    # Profile EMA memory periodically
    if accelerator.is_main_process and global_step % 100 == 0:
        mem_profiler.profile_ema_model(modules.ema, f"train_step_{global_step}/after_ema_update")
```

#### c. Before evaluation (replace lines 673-687)

```python
eval_metrics: dict[str, float] = {}
if run_eval_now:
    if accelerator.is_main_process:
        # COMPREHENSIVE PROFILING BEFORE EVAL
        mem_profiler.comprehensive_profile(
            f"eval_step_{global_step}/before_cache_clear",
            model=modules.denoiser,
            ema_model=modules.ema,
            optimizer=optimizer,
            vae=modules.vae if hasattr(modules, 'vae') else None,
            text_encoder=modules.text_encoder if hasattr(modules, 'text_encoder') else None,
        )

        # Clear cache and profile again
        mem_profiler.garbage_collect(f"eval_step_{global_step}/gc_before_eval")
        torch.cuda.empty_cache()

        mem_profiler.log_memory(f"eval_step_{global_step}/after_cache_clear")

        # Profile before entering EMA context
        mem_profiler.log_memory(f"eval_step_{global_step}/before_ema_context")

        with ema_weights_context(accelerator, modules):
            # Profile inside EMA context
            mem_profiler.log_memory(f"eval_step_{global_step}/inside_ema_context")

            eval_metrics = run_evaluation(
                accelerator=accelerator,
                adapter=adapter,
                cfg=cfg,
                modules=modules,
                train_loader=train_loader,
                val_loader=val_loader,
                run_dir=run_dir,
                global_step=global_step,
            )

        # Profile after exiting EMA context
        mem_profiler.log_memory(f"eval_step_{global_step}/after_ema_context")

        torch.cuda.empty_cache()
        mem_profiler.log_memory(f"eval_step_{global_step}/after_final_cache_clear")
```

### 6. Add profiling to run_evaluation function

Modify `signal_diffusion/training/diffusion_utils.py`, in the `run_evaluation` function (around line 170):

```python
def run_evaluation(
    accelerator: Accelerator,
    adapter: Any,
    cfg: Any,
    modules: Any,
    train_loader: Any,
    val_loader: Any,
    run_dir: Path,
    global_step: int,
    mem_profiler: Optional[Any] = None,  # Add this parameter
) -> dict[str, float]:
    """Generate samples and compute evaluation metrics for the current step."""
    eval_examples = getattr(cfg.training, "eval_num_examples", 0) or 0
    eval_mmd_samples = getattr(cfg.training, "eval_mmd_samples", 0) or 0
    if eval_examples <= 0 and eval_mmd_samples <= 0:
        return {}

    if mem_profiler is not None:
        mem_profiler.log_memory(f"run_evaluation_{global_step}/start")

    # ... rest of function ...

    # Before the generation loop (around line 210)
    if mem_profiler is not None:
        mem_profiler.log_memory(f"run_evaluation_{global_step}/before_generation")

    while total_remaining > 0 and task_queue:
        # ... generation code ...

        # After each batch (around line 273)
        generated_samples.append(generated_batch.cpu())

        if mem_profiler is not None and len(generated_samples) % 5 == 0:
            mem_profiler.log_memory(f"run_evaluation_{global_step}/after_batch_{len(generated_samples)}")

        # ... rest of loop ...
```

## Expected Output

The profiler will log detailed memory usage information like:

```
================================================================================
COMPREHENSIVE MEMORY PROFILE: train_step_100/before_eval
================================================================================
[train_step_100/before_eval/overall] Memory: allocated=12234.5 MB (Δ+234.2), reserved=13000.0 MB (Δ+300.0), max_allocated=12500.0 MB, max_reserved=13500.0 MB
[train_step_100/before_eval/denoiser] Model: params=890,123,456 (trainable=890,123,456), param_mem=3397.4 MB, buffer_mem=12.3 MB, total=3409.7 MB
[train_step_100/before_eval/vae] Model: params=83,653,863 (trainable=83,653,863), param_mem=319.1 MB, buffer_mem=2.1 MB, total=321.2 MB
[train_step_100/before_eval/text_encoder] Model: params=123,060,480 (trainable=0), param_mem=469.3 MB, buffer_mem=0.5 MB, total=469.8 MB
[train_step_100/before_eval/ema] EMA: shadow_params=890, shadow_mem=3397.4 MB
[train_step_100/before_eval/optimizer] Optimizer: params_with_state=890, state_mem=6794.8 MB
================================================================================
```

## Analysis

Based on the output, look for:

1. **Growing delta values**: If `delta_allocated_mb` keeps growing between evals, memory is leaking
2. **Denoiser (main model)**: Your diffusion model parameters (~3.4 GB in example above)
3. **VAE memory**: If using Stable Diffusion, VAE encodes/decodes images (~320 MB in example)
4. **Text encoder memory**: If using caption conditioning, CLIP text encoder (~470 MB in example)
5. **EMA memory doubling**: EMA stores a full copy of denoiser parameters (~3.4 GB = same as denoiser)
6. **Optimizer memory**: AdamW stores 2 states per parameter (~2x denoiser size = 6.8 GB)
7. **Total before eval**: Denoiser + VAE + Text Encoder + EMA + Optimizer ≈ 3.4 + 0.3 + 0.5 + 3.4 + 6.8 = 14.4 GB
8. **Eval memory spike**: Look at `inside_ema_context` - the EMA weights are copied into the model temporarily

**Important**: The VAE and text encoder are typically frozen (not trained), so they don't have optimizer states. Only the denoiser has optimizer states and EMA.

## Common Issues & Fixes

### Issue 1: EMA context doubles memory temporarily

**Symptom**: Memory spike when entering `ema_weights_context`

**Explanation**: The EMA context does:
1. `store()` - saves current model params (allocates temp storage)
2. `copy_to()` - copies EMA params to model
3. `restore()` - copies saved params back

**Fix**: The EMA implementation may be keeping extra copies. Consider:

```python
# Option 1: Disable EMA during problematic evals
if global_step > 0 and should_skip_ema_for_memory:
    # Skip EMA and just eval with current weights
    eval_metrics = run_evaluation(...)
else:
    with ema_weights_context(accelerator, modules):
        eval_metrics = run_evaluation(...)
```

### Issue 2: Generated samples accumulating

**Symptom**: Memory grows during `run_evaluation` generation loop

**Fix**: Move samples to CPU immediately and clear cache:

```python
generated_batch = adapter.generate_conditional_samples(...)
generated_samples.append(generated_batch.cpu())  # Move to CPU immediately
del generated_batch  # Explicitly delete GPU reference
if len(generated_samples) % 5 == 0:
    torch.cuda.empty_cache()
```

### Issue 3: Optimizer states too large

**Symptom**: Optimizer taking >60% of total memory

**Fix**: Use 8-bit optimizer:

```toml
[optimizer]
name = "adamw_8bit"  # Reduces optimizer memory by ~50%
```

### Issue 4: Gradient accumulation not releasing memory

**Symptom**: Memory grows with each training step

**Fix**: Ensure gradients are zeroed with `set_to_none=True`:

```python
optimizer.zero_grad(set_to_none=True)  # More aggressive memory release
```

### Issue 5: VAE encoding/decoding during eval

**Symptom**: OOM during eval, especially when generating many samples. VAE appears in memory profile.

**Explanation**: If your adapter uses a VAE (e.g., Stable Diffusion):
- Training: VAE encodes images to latents (happens once per batch)
- Eval: VAE decodes latents to images (happens for every generated sample)
- Decoding is memory-intensive, especially with large batch sizes

**Fix**: Reduce eval batch size or decode in smaller chunks:

```python
# In your config
[training]
eval_batch_size = 4  # Much smaller than training batch_size

# Or in adapter's generate_samples, decode in chunks:
latents_chunks = torch.chunk(latents, chunks=4, dim=0)
decoded = []
for chunk in latents_chunks:
    decoded.append(vae.decode(chunk).sample)
    torch.cuda.empty_cache()
decoded = torch.cat(decoded, dim=0)
```

### Issue 6: Text encoder in memory during eval

**Symptom**: Text encoder appears in profile, taking ~500 MB even during unconditioned generation

**Fix**: Move text encoder to CPU when not needed:

```python
# After encoding prompts in training/eval
if modules.text_encoder is not None and not needs_text_encoder:
    modules.text_encoder.to('cpu')
    torch.cuda.empty_cache()

# Move back when needed
if modules.text_encoder is not None and needs_text_encoder:
    modules.text_encoder.to(accelerator.device)
```

## Quick Debug Script

To quickly test memory profiling without modifying the main training script:

```python
# debug_memory.py
from signal_diffusion.training.memory_profiler import MemoryProfiler
import torch

profiler = MemoryProfiler(enabled=True)

# Simulate your training setup
model = ...  # Your model
ema = ...    # Your EMA
optimizer = ...  # Your optimizer

profiler.set_baseline("start")
profiler.comprehensive_profile("initial", model=model, ema=ema, optimizer=optimizer)

# Simulate training step
for i in range(10):
    # ... training code ...
    profiler.log_memory(f"step_{i}")

# Simulate eval
profiler.log_memory("before_eval")
torch.cuda.empty_cache()
profiler.log_memory("after_cache_clear")
```
