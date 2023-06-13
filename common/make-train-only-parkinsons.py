# Make train-only version of dataset
import pandas as pd
import shutil
import os

datadir = "/data/shared/signal-diffusion/parkinsons/stfts"
outdir = "/data/shared/signal-diffusion/parkinsons/stfts-train-only"
# Reset the output directory
shutil.rmtree(outdir, ignore_errors=True)
os.makedirs(outdir, exist_ok=True)

df = pd.read_csv(os.path.join(datadir, "train-metadata.csv"))
subjects = df["file_name"].str.split("/").str[0].unique()

for subject in subjects:
    shutil.copytree(
        os.path.join(datadir, subject),
        os.path.join(outdir, subject),
    )

shutil.copyfile(
    os.path.join(datadir, "train-metadata.csv"),
    os.path.join(outdir, "metadata.csv"),
)
