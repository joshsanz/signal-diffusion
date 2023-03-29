#!/bin/bash

export MODEL_NAME='OFA-Sys/small-stable-diffusion-v0'
export dataset_name='lambdalabs/pokemon-blip-captions'

accelerate launch --mixed_precision=fp16 train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=40000 \
  --learning_rate=2e-5 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-pokemon-model.2" \
  --enable_xformers_memory_efficient_attention \
  # --validation_prompt="a cute blue dragon"

