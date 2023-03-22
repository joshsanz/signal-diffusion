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

## Validation and inference

Validation images and losses are logged for `tensorboard` in the `sd-stft-lora/logs` directory.
Simply run

```sh
tensorboard --logdir sd-stft-lora/logs
```

and open your browser to <http://localhost:6006>.

For inference, pass your trained LoRA weights to the `infer_text_to_image_lora.py` script. For
example,

```sh
infer_text_to_image_lora.py --path sd-stft-lora/pytorch_lora_weights.bin \
    --prompt "a soft jazz song" \
    -n 8 --seed 42 -x 0.5
```

Call the script with `--help` for more information.

## TODOs

* Automatically convert STFT images back to audio
* Test learning rate values, batch size, num noise steps for training and inference quality
* Make full fine-tune version of notebook/script and run on BRC or LambdaCloud
