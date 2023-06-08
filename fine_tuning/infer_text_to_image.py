from diffusers import StableDiffusionPipeline
from diffusers import Mel
from PIL import Image
import torch
import argparse
import numpy as np
import os
import soundfile as sf


def image_grid(array, ncols=4):
    index, height, width, channels = array.shape
    nrows = index // ncols
    img_grid = (array.reshape(nrows, ncols, height, width, channels)
                .swapaxes(1, 2)
                .reshape(height * nrows, width * ncols, channels))
    return img_grid


# Parse command line arguments
model_path = "./sd-pokemon-model-lora"
prompt = "A pokemon with blue eyes."
steps = 30

parser = argparse.ArgumentParser("An inference pipeline for finetuned SDv1.5")
parser.add_argument("--path", default=model_path, help="Path to fine-tuned weights")
parser.add_argument("--comparison", default=None, help="Path to original weights for comparison")
parser.add_argument("-p", "--prompt", default=prompt, help="Text prompt for generation")
parser.add_argument("-n", "--num_images", type=int, default=1, help="Number of images to generate from prompt")
parser.add_argument("-o", "--output", default="out.png", help="Output filename")
parser.add_argument("-s", "--steps", type=int, default=steps, help="Number of diffusion steps")
parser.add_argument("-r", "--seed", type=int, default=42, help="Random seed")
parser.add_argument("--audio", action="store_true", help="Generate audio from image output, interpreted as an STFT")
parser.add_argument("--audio_fs", type=int, default=22050, help="Audio sampling rate, Hz")
args = parser.parse_args()

# Seed
torch.manual_seed(args.seed)

# Run the pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    os.path.expanduser(args.path),
    safety_checker=lambda images, **kwargs: (images, False),  # Disable safety checker - spectrograms won't be NSFW
    torch_dtype=torch.float16
)
pipe.to("cuda")

prompts = [args.prompt] * args.num_images
images = [pipe(prompts[i], num_inference_steps=args.steps, guidance_scale=7.5,).images[0]
          for i in range(args.num_images)]

# Generate comparison images
if args.comparison is not None:
    torch.manual_seed(args.seed)
    pipe = StableDiffusionPipeline.from_pretrained(os.path.expanduser(args.comparison), torch_dtype=torch.float16)
    pipe.to("cuda")
    comparisons = [pipe(prompts[i], num_inference_steps=args.steps, guidance_scale=7.5,).images[0]
                   for i in range(args.num_images)]
    images.extend(comparisons)

# Generate audio
if args.audio:
    name_stem = os.path.splitext(args.output)[0]
    resolution = 512
    mel = Mel(x_res=resolution, y_res=resolution, sample_rate=args.audio_fs, n_fft=2048,
              hop_length=resolution, top_db=80, n_iter=32,)
    audios = [mel.image_to_audio(im.convert('L')) for im in images]
    for i, a in enumerate(audios):
        sf.write(f"{name_stem}_{i}.wav", a, args.audio_fs)

# Plot
ncol = int(np.ceil(np.sqrt(args.num_images))) if args.comparison is None else args.num_images
nrow = int(np.ceil(args.num_images / ncol))

images = [np.asarray(im) for im in images]
images.extend([np.zeros_like(images[0]) for _ in range(nrow * ncol - args.num_images)])

grid = image_grid(np.array(images), ncol)
image = Image.fromarray(grid)
image.save(args.output)
