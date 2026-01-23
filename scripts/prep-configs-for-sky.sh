# Update location of default config file
uv run python scripts/edit_config.py -c config/diffusion/*.toml \
    -s settings.config="~/sky_workdir/config/default.toml" \
    -s logging.wandb_project="signal-diffusion"

# Update data paths to /data, with appropriate subfolders
uv run python scripts/edit_config.py -c config/diffusion/*-db-only.toml \
    -s dataset.name="/data/processed/reweighted_meta_dataset_log_n2048_fs125"

uv run python scripts/edit_config.py -c config/diffusion/*-db-iq.toml \
    -s dataset.name="/data/processed-iq/reweighted_meta_dataset_log_n2048_fs125"

uv run python scripts/edit_config.py -c config/diffusion/*-db-polar.toml \
    -s dataset.name="/data/processed-polar/reweighted_meta_dataset_log_n2048_fs125"

uv run python scripts/edit_config.py -c config/diffusion/*-timeseries.toml \
    -s dataset.name="/data/processed/reweighted_timeseries_meta_dataset_n2048_fs125"

# Set latent space parameters for models
uv run python scripts/edit_config.py -c config/diffusion/localmamba-*.toml \
    -s model.vae_tiling=true \
    -s model.latent_space=true \
    -s model.extras.in_channels=16 \
    -s model.extras.out_channels=16

uv run python scripts/edit_config.py -c config/diffusion/sd35-*.toml \
    -s model.latent_space=true \
    -s model.vae_tiling=true

# Set hourglass and localmamba model sizes to maximum that fits in GPU memory with batch size 8
uv run python scripts/edit_config.py -c config/diffusion/hourglass-*.toml config/diffusion/localmamba-*.toml \
    -s model.extras.depths="[4, 4, 4]" \
    -s model.extras.widths="[128, 256, 512]"

uv run python scripts/edit_config.py -c config/diffusion/localmamba-*.toml \
    -s model.extras.depths="[2, 2, 7, 2]" \
    -s model.extras.dims="[96, 192, 384, 768]" \
    -s model.extras.mlp_ratio=4.0

# Set batch sizes to what fits in memory
uv run python scripts/edit_config.py -c config/diffusion/localmamba-*.toml \
    -s dataset.batch_size=128 \
    -s training.eval_batch_size=12 \
    -s training.gradient_accumulation_steps=1

uv run python scripts/edit_config.py -c config/diffusion/hourglass-*.toml \
    -s dataset.batch_size=128 \
    -s training.eval_batch_size=128 \
    -s training.gradient_accumulation_steps=1

uv run python scripts/edit_config.py -c config/diffusion/sd35-*.toml \
    -s model.vae_tiling=true \
    -s dataset.batch_size=96 \
    -s training.eval_batch_size=96 \
    -s training.gradient_accumulation_steps=1
