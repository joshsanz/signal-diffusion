import argparse
import csv
import random
import os
import sys
import time
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler

sys.path.append("../")
from data_processing.general_dataset import general_class_labels


def class_to_text(class_id, num_classes):
    age = random.choice([19, 20, 21, 22, 23, 24, 52, 53, 54, 56, 57, 58, 60, 61, 62, 64, 65,
                         66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
                         83, 84, 85, 86])
    mf = general_class_labels[class_id % 2]
    if num_classes == 2:
        text = f"an EEG spectrogram of a {age} year old, {mf} subject"
    elif num_classes == 4:
        health_text = "healthy" if class_id < 2 else "parkinsons disease diagnosed"
        text = f"an EEG spectrogram of a {age} year old, {health_text}, {mf} subject"
    else:
        raise ValueError(f"Unsupported number of classes: {num_classes}")
    return text


def timefmt(t):
    t = int(t)
    h = t // 3600
    m = (t % 3600) // 60
    s = t % 60
    fmtd = f"{h:02d}:" if h > 0 else ""
    fmtd += f"{m:02d}:" if m > 0 else "00:"
    fmtd += f"{s:02d}"
    return fmtd


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected: received {v}.')


def print_epoch(t0, b, B):
    t1 = time.time()
    t_run = timefmt(t1 - t0)
    t_est = "??" if b == 0 else timefmt((t1 - t0) / b * B)
    print(f"Generating batch {b + 1}/{B} [{t_run}/{t_est}]")


def disabled_safety_checker(images, clip_input):
    if len(images.shape) == 4:
        num_images = images.shape[0]
        return images, [False] * num_images
    else:
        return images, False


def main(args):
    print("Sample images from a fine-tuned stable diffusion.")
    with open(os.path.join(args.output_dir, "run_args.txt"), "w") as f:
        f.write(" ".join(sys.argv))
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: running on CPU. This will be slow.")

    N = args.num_images
    B = args.batch_size
    assert N % B == 0

    # Load model
    ddim = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        os.path.expanduser(args.ckpt),
        torch_dtype=torch.float16,
        scheduler=ddim,
    )
    pipe.to(device)
    pipe.safety_checker = disabled_safety_checker  # Must be done after pipe configuration, e.g. attention slicing or enabling xformers

    # Generate synthetic dataset
    filenames = []
    classes = []
    t0 = time.time()
    for b in range(N // B):
        print_epoch(t0, b, N // B)
        if args.num_classes > 0:
            texts = []
            for i in range(B):
                class_id = random.randint(0, args.num_classes - 1)
                texts.append(class_to_text(class_id, args.num_classes))
                classes.append(class_id)
        else:
            texts = [""] * B

        images = pipe(texts, num_inference_steps=args.num_sampling_steps, guidance_scale=args.cfg_scale,
                      height=args.image_size, width=args.image_size).images

        # Save images
        for i, im in enumerate(images):
            im.convert("L").save(f"{args.output_dir}/sample_{b * B + i}.jpg")
            filenames.append(f"sample_{b * B + i}.jpg")

    # Write metadata file
    with open(f"{args.output_dir}/metadata.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "class", "text", "gender", "health", "age"])
        for i in range(N):
            age = texts[i].split("year")[0].split()[-1]
            gender = "M" if "male" in texts[i] else "F"
            health = "H" if "healthy" in texts[i] else "PD"
            writer.writerow([filenames[i], classes[i], texts[i], gender, health, age])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--image-size", type=int, choices=[32, 64, 256, 512], default=256)
    parser.add_argument("-c", "--num-classes", type=int, default=0)
    parser.add_argument("-C", "--cfg-scale", type=float, default=1.5)
    parser.add_argument("-S", "--num-sampling-steps", type=int, default=50)
    parser.add_argument("-n", "--num-images", type=int, default=8)
    parser.add_argument("-b", "--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt", type=str, default="/data/shared/signal-diffusion/stft-full.meta_set.3")
    parser.add_argument("-o", "--output-dir", type=str, default="/data/shared/signal-diffusion/generated_images_sd")
    args = parser.parse_args()
    assert args.num_classes in [0, 2, 4], "Only 0, 2, or 4 classes supported for now"
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
