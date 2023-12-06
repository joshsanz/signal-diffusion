#!/bin/bash

export RUN_ID=meta_set.0
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export dataset_name="/data/shared/signal-diffusion/reweighted_meta_dataset"
export output_dir="./stft-full.$RUN_ID"
echo "Model $MODEL_NAME"
echo "Dataset $dataset_name"
echo "Output location $output_dir"
#   --use_ema \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 accelerate launch --mixed_precision=no train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir="$dataset_name" \
  --resolution=256 --center_crop \
  --train_batch_size=1 \
  --gradient_accumulation_steps=16 \
  --gradient_checkpointing \
  --max_train_steps=1000 \
  --learning_rate=1e-5 \
  --optimizer=adamw \
  --adam_weight_decay=3e-6 \
  --max_grad_norm=1 \
  --allow_tf32 \
  --lr_scheduler="constant" --lr_warmup_steps=100 \
  --checkpointing_steps=501 --checkpoints_total_limit=2 \
  --output_dir="$output_dir" \
  --report_to=wandb \
  --validation_prompt="an EEG spectrogram for a 70 year old, healthy, female subject" \
  --validation_epochs=1
