# Update location of default config file
uv run python scripts/edit_config.py -c config/diffusion/*.toml \
    -s settings.config="~/sky_workdir/config/default.toml"

# Update data paths to /data, with appropriate subfolders
uv run python scripts/edit_config.py -c config/diffusion/*-db-only.toml \
    -s dataset.name="/data/processed/reweighted_meta_dataset_log_n2048_fs125"

uv run python scripts/edit_config.py -c config/diffusion/*-db-iq.toml \
    -s dataset.name="/data/processed-iq/reweighted_meta_dataset_n2048_fs125"

uv run python scripts/edit_config.py -c config/diffusion/*-db-polar.toml \
    -s dataset.name="/data/processed-polar/reweighted_meta_dataset_n2048_fs125"

uv run python scripts/edit_config.py -c config/diffusion/*-timeseries.toml \
    -s dataset.name="/data/processed/reweighted_timeseries_meta_dataset_n2048_fs125"
