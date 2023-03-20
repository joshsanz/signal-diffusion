from diffusers import StableDiffusionPipeline
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np


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

parser = argparse.ArgumentParser("An inference pipeline for LoRA-finetuned SDv1.5")
parser.add_argument("--path", default=model_path, help="Path to LoRA weights")
parser.add_argument("-p", "--prompt", default=prompt, help="Text prompt for generation")
parser.add_argument("-n", "--num-images", type=int, default=1, help="Number of images to generate from prompt")
parser.add_argument("-o", "--output", default="out.png", help="Output filename")
parser.add_argument("-s", "--steps", type=int, default=steps, help="Number of diffusion steps")
parser.add_argument("-x", "--cross-attention", type=float, default=0.5, help="LoRA weight cross-attention scale; 0=base model, 1=full LoRA")
parser.add_argument("-r", "--seed", type=int, default=42, help="Random seed")
parser.add_argument("--sweep-cross-attention", action="store_true", default=False, help="Sweep cross-attention from 0 to 1")
args = parser.parse_args()

# Seed
torch.manual_seed(args.seed)

# Run the pipeline
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.unet.load_attn_procs(args.path)
pipe.to("cuda")

prompts = [args.prompt] * args.num_images
if args.sweep_cross_attention:
    images = []
    scales = np.linspace(0., 1., args.num_images, endpoint=True)
    for i in range(args.num_images):
        torch.manual_seed(args.seed)
        images.append(pipe(prompts[i], num_inference_steps=args.steps, guidance_scale=7.5,
                           cross_attention_kwargs={"scale": scales[i]}).images[0])
else:
    images = [pipe(prompts[i], num_inference_steps=args.steps, guidance_scale=7.5,
                   cross_attention_kwargs={"scale": args.cross_attention}).images[0]
              for i in range(args.num_images)]
images = [np.asarray(im) for im in images]

# Plot
nrow = int(np.floor(np.sqrt(args.num_images)))
ncol = int(np.ceil(args.num_images / nrow))
images.extend([np.zeros_like(images[0]) for _ in range(nrow * ncol - args.num_images)])

grid = image_grid(np.array(images), ncol)
fig = plt.figure(figsize=(nrow * 3, ncol * 3))
plt.imshow(grid)
plt.axis('off')
plt.savefig(args.output, bbox_inches='tight')
