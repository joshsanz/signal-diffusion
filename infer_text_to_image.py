from diffusers import StableDiffusionPipeline
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os


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
parser.add_argument("-n", "--num-images", type=int, default=1, help="Number of images to generate from prompt")
parser.add_argument("-o", "--output", default="out.png", help="Output filename")
parser.add_argument("-s", "--steps", type=int, default=steps, help="Number of diffusion steps")
parser.add_argument("-r", "--seed", type=int, default=42, help="Random seed")
args = parser.parse_args()

# Seed
torch.manual_seed(args.seed)

# Run the pipeline
pipe = StableDiffusionPipeline.from_pretrained(os.path.expanduser(args.path), torch_dtype=torch.float16)
pipe.to("cuda")

prompts = [args.prompt] * args.num_images
images = [pipe(prompts[i], num_inference_steps=args.steps, guidance_scale=7.5,).images[0]
          for i in range(args.num_images)]
images = [np.asarray(im) for im in images]

# Generate comparison images
if args.comparison is not None:
    torch.manual_seed(args.seed)
    pipe = StableDiffusionPipeline.from_pretrained(os.path.expanduser(args.comparison), torch_dtype=torch.float16)
    pipe.to("cuda")
    comparisons = [pipe(prompts[i], num_inference_steps=args.steps, guidance_scale=7.5,).images[0]
                   for i in range(args.num_images)]
    comparisons = [np.asarray(comp) for comp in comparisons]
    images.extend(comparisons)

# Plot
ncol = int(np.ceil(np.sqrt(args.num_images))) if args.comparison is None else args.num_images
nrow = int(np.ceil(args.num_images / ncol))

images.extend([np.zeros_like(images[0]) for _ in range(nrow * ncol - args.num_images)])

grid = image_grid(np.array(images), ncol)
fig = plt.figure(figsize=(nrow * 3, ncol * 3))
plt.imshow(grid)
plt.axis('off')
plt.savefig(args.output, bbox_inches='tight')
