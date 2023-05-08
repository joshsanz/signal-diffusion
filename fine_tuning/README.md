# Riffusion Replication

## System Requirements

### Minimum

* Recent linux
  * Windows and WSL may be possible with correct drivers and a lot of fiddling with versions
  * Running on CPU may be possible on intel macbooks, seems broken on m1/m2
* Nvidia GPU
  * At least 6GB VRAM, the more the better
  * CUDA > ? 10.0 ? installed, can check with `nvidia-smi`
* 10+ GB of disk space for FMA-small, preprocessed images, model AttentionProcessor checkpoints

### Desired

* 10+ GB VRAM for LoRA fine-tuning
* 30+ GB VRAM for full fine-tuning
* 1+ TB for FMA-large, full model checkpoints

## Setup

Create and activate a new conda environent/virtualenv with python 3.10+

`pip install -r requirements.txt`

-------------------

## LoRA fine-tuning

Open `train_text_to_stft_lora.ipynb` in Jupyter.

Run imports and FMA dataset utility sections. Optionally, play with the example datapoint
visualization cell.

### Preprocessing

On the first runthrough, you should _either_ set `preprocess_audio` to `True` in the configuration
_or_ uncomment and run the cell with the `preprocess_fma_audio` call. Setting `verbose` to `True`
will print the current dataset index being preprocessed, which can be used to determine which audio
files are corrupted and should be skipped.

### Configuration

Finally, adjust any preferred defaults (such as `fma_path`) in the `LoRATrainConfig` definition to
match your environment, or set them in the `LoRATrainConfig(...)` constructor in the next cell.

### Training

At this point, you are ready to run

```py
main(config)
```

and train your model.

-------------------

## Full fine-tuning

_Requires at least 24 GB GPU memory._

Use the `train_text_to_image.py` script for full fine tuning. An example invocation is

```sh
export RUN_ID=0
export MODEL_NAME='runwayml/stable-diffusion-v1-5'
export dataset_name='/data'
export output_dir="tmp/stft-full.$RUN_ID"
accelerate launch --mixed_precision=no train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
  --use_ema \
  --resolution=512 --center_crop \
  --train_batch_size=1 \
  --gradient_accumulation_steps=8 \
  --gradient_checkpointing \
  --max_train_steps=10000 \
  --learning_rate=1e-4 \
  --max_grad_norm=1 \
  --allow_tf32 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --checkpointing_steps=1000 --checkpoints_total_limit=3 \
  --resume_from_checkpoint="latest" \
  --output_dir="$output_dir" \
  --validation_prompt="a international instrumental song" --validation_epochs=1
```

### SkyPilot

[SkyPilot](https://skypilot.readthedocs.io) provides setup and control of instances and training runs on cloud infrastructure, all managed from your laptop command line.

The primary configuration for cloud runs is `sky-aws-stft.yaml`, which demonstrates most of the useful features. Start a training run by editing `RUN_ID` then calling

```sh
sky launch -c <cluster name> sky-aws-stft.yaml
```

You may want to schedule cluster shutdown with `sky autostop <cluster name>` to prevent charges while idle.

-------------------


## Validation and inference

Validation images and losses are logged for `tensorboard` in the `sd-stft-lora/logs` directory.
Simply run

```sh
tensorboard --logdir sd-stft-lora/logs
```

and open your browser to <http://localhost:6006>.

For LoRA trained inference, pass your trained LoRA weights to the `infer_text_to_image_lora.py` script. For
example,

```sh
infer_text_to_image_lora.py --path sd-stft-lora/pytorch_lora_weights.bin \
    --prompt "a soft jazz song" \
    -n 8 --seed 42 -x 0.5
```

For fully tuned weights, pass the path to the saved results to `infer_text_to_image.py`. For example,

```sh
infer_text_to_image.py --path tmp/stft-full.0/ \
  --prompt "a soft jazz song" \
  -n 8 --seed 42
```

Call either script with `--help` for more information.

## Notes

### Pricing

| Hardware |   |   | \| Price / hr |   |   |
| -------- | - | - | ------------- | - | - |
| **GPU**  | **Relative Perf.** | **Memory GB** | **\| AWS** | **AWS Spot** | **Lambda** |
| A10      | 2 (?) | 24 | \|  $1.01 |      - | $0.60 |
| 4x A10   | 2 (?) | 24 | \|  $5.67 |      - | $2.40 |
| A100     | 3.57  | 40 | \|      - |      - | $1.10 |
| 8x A100  |       |    | \| $32.77 | $15.22 | $8.80 |
| V100     | 1     | 16 | \|  $3.06 |  $1.07 |     - |
| 8x V100  |       |    | \| $24.48 |  $4.45 | $4.40 |
| 8x K80   | < 1   | 12 | \|      - |  $2.39 |     - |
| A6000    | 2.15  | 48 | \|      - |      - | $0.80 |
| RTX 3080 | 0.86  | 10 | \|      - |      - |     - |

### Bugs

* `bf16` mixed precision uses more memory than fp16, causing OutOfMemory errors on RTX3080
* `fp16` for `small-stable-diffusion-v0` and `stable-diffusion-v1-5` causes NaN gradients in specific layers, breaking training
* PyTorch==2.0.0 breaks on A10/A6000 GPUs somewhere in attention layers
* StableDiffusionPipeline requires `torch.autocast('cuda')` iff using mixed precision

## TODOs

* Diagnose weird checkpoint load crash
* Add support for interruptible training
  * tensorboard writer needs `s3://` location
  * `output_dir` should be the s3 bucket to detect previous checkpoints
* WandB support
* Test learning rate values, batch size, num noise steps for training and inference quality
