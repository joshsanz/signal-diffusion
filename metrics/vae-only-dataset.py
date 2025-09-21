# Generate a just-the-VAE dataset for KDD baseline

import argparse
from datasets import load_dataset, Features, Image
from diffusers import AutoencoderKL
import numpy as np
import os
from PIL import Image as pImage
import torch
import torchvision.transforms.v2 as v2
import tqdm.auto as tqdm

# Enable TF32 on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main(args):
    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = v2.Compose([
        v2.Resize((args.image_size, args.image_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.5], [0.5]),
    ])

    def transform_fn(x):
        x['image'] = transform(x['image'])
        return x

    def collate_fn(examples):
        return torch.stack([x['image'] for x in examples])

    dataset = load_dataset("imagefolder", data_dir=args.dataset,
                           features=Features({"image": Image(mode="RGB")})
                           ).with_format("torch")
    dataset = dataset.with_transform(transform_fn)
    use_pin_memory = torch.cuda.is_available()
    dataloader = torch.utils.data.DataLoader(
        dataset['train'],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=use_pin_memory,
        num_workers=4,
    )

    if "vae" in args.model:
        vae = AutoencoderKL.from_pretrained(args.model).to(device)
    else:
        vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae").to(device)
    vae.eval()

    count = 0
    for batch in tqdm.tqdm(dataloader):
        images = batch.to(device)
        with torch.no_grad():
            recon = vae(images)
        recon = recon.sample.permute(0, 2, 3, 1).cpu().float().numpy().clip(-1, 1)
        # Save the reconstructions
        for im in recon:
            im = pImage.fromarray(((im * 0.5 + 0.5) * 255).astype(np.uint8))
            im.save(f"{args.output}/recon_{count}.jpg")
            count += 1

    with open(f"{args.output}/metadata.csv", "w") as f:
        f.write("file_name\n")
        for i in range(count):
            f.write(f"recon_{i}.jpg\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a just-the-VAE dataset for KDD baseline')
    parser.add_argument('-d', '--dataset', type=str, help='Path to dataset')
    parser.add_argument('-m', '--model', type=str, help='Path to model')
    parser.add_argument('-o', '--output', type=str, help='Path to save output')
    parser.add_argument('-b', '--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('-s', '--image-size', type=int, default=256, help='Image size')
    args = parser.parse_args()
    main(args)
