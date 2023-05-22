import logging
import math
import numpy as np
import os
import time
from dataclasses import dataclass
from packaging import version
from PIL import Image
# from pprint import pprint
from tqdm.auto import tqdm
from typing import Optional, Union

import datasets
from datasets import load_dataset

import torch
import torch.utils.checkpoint
from torchvision import transforms

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, GradScalerKwargs, set_seed

# from huggingface_hub import HfFolder, Repository, create_repo, whoami

import transformers
import diffusers
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import is_wandb_available  # , check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers import Mel

from contperceptual_loss import LPIPSWithDiscriminator


logger = get_logger(__name__, log_level="INFO")


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def decode_latents(vae, latents):
    decoded = vae.decode(latents).sample
    shifted = (decoded / 2 + 0.5).clamp(0, 1)
    image = shifted.cpu().permute(0, 2, 3, 1).float().numpy()
    return image


def encode_sample(vae, sample):
    img_tensor = sample["pixel_values"]
    if "image" in sample.keys():
        images = sample["image"]
    else:
        images = ((img_tensor + 1.) / 2).permute(0, 3, 2, 1).numpy()
        images = numpy_to_pil(images)
    latents = vae.encode(img_tensor.to('cuda')).latent_dist.mode()
    return images, latents


def get_concat_h(images, padding=0):
    width = sum([im.width + padding for im in images]) - padding
    dx = images[0].width + padding
    dst = Image.new("RGB", (width, images[0].height), (255, 255, 0))
    for i in range(len(images)):
        dst.paste(images[i], (dx * i, 0))
    return dst


def get_concat_v(images, padding=0):
    height = sum([im.height + padding for im in images]) - padding
    dy = images[0].height + padding
    dst = Image.new("RGB", (images[0].width, height), (255, 255, 0))
    for i in range(len(images)):
        dst.paste(images[i], (0, dy * i))
    return dst


def image_grid(images, nrow=1, padding=2):
    N = len(images)
    ncol = (N + nrow - 1) // nrow
    rows = [images[i:i + ncol] for i in range(nrow)]
    rows = [get_concat_h(r, padding) for r in rows]
    grid = get_concat_v(rows, padding)
    return grid


def get_model_size(model, verbose=False):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    if verbose:
        print("model size: {:.3f}MB".format(size_all_mb))
    return size_all_mb


def prepare_checkpointing(accelerator, args, ema_vae):
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_vae.save_pretrained(os.path.join(output_dir, "vae_ema"))

            for i, model in enumerate(models):
                if isinstance(model, AutoencoderKL):
                    model.save_pretrained(os.path.join(output_dir, "vae"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()
                elif isinstance(model, LPIPSWithDiscriminator):
                    os.makedirs(os.path.join(output_dir, "discriminator"), exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(output_dir, "discriminator", "discriminator.pt"))
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                logger.info("Loading ema weights...")
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "vae_ema"), AutoencoderKL
                )
                ema_vae.load_state_dict(load_model.state_dict())
                ema_vae.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                logger.info("Loading model {}/{}...".format(i + 1, len(models)))
                # pop models so that they are not loaded again
                model = models.pop()

                if isinstance(model, AutoencoderKL):
                    # load diffusers style into model
                    load_model = AutoencoderKL.from_pretrained(input_dir, subfolder="vae")
                    model.register_to_config(**load_model.config)

                    model.load_state_dict(load_model.state_dict())
                    del load_model
                elif isinstance(model, LPIPSWithDiscriminator):
                    model.load_state_dict(torch.load(os.path.join(input_dir, "discriminator", "discriminator.pt")))

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    existing_ckpts = os.listdir(ckpt_dir)
    if not args.resume_from_checkpoint:
        if existing_ckpts:
            logger.error("Not resuming from checkpoint, but found existing checkpoints: {}. Remove or start from checkpoints to continue.".format(existing_ckpts))
            raise RuntimeError("Bad checkpoint setup: attempted overwrite")
    elif args.resume_from_checkpoint == "latest":
        if not existing_ckpts:
            logger.error("Requested training from 'latest' checkpoint, but no checkpoints found.")
            raise RuntimeError("Bad checkpoint setup: none exist")
    else:
        if os.path.basename(args.resume_from_checkpoint) not in existing_ckpts:
            logger.error("Requested training from checkpoint {}, but checkpoint not found.".format(args.resume_from_checkpoint))
            raise RuntimeError("Bad checkpoint setup: requested checkpoint does not exist")


def get_dataloader(accelerator, args):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    dataset_name_mapping = {
        "lambdalabs/pokemon-blip-captions": ("image", "text"),
    }
    dataset_columns = dataset_name_mapping.get(args.dataset_name, None)
    if args.image_column is None:
        image_column = (
            dataset_columns[0] if dataset_columns is not None else column_names[0]
        )
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.caption_column is None:
        caption_column = (
            dataset_columns[1] if dataset_columns is not None else column_names[1]
        )
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(
                args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(args.resolution)
            if args.center_crop
            else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip()
            if args.random_flip
            else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = (
                dataset["train"]
                .shuffle(seed=args.seed)
                .select(range(args.max_train_samples))
            )
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        return {
            "pixel_values": pixel_values,
        }

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    return train_dataset, train_dataloader


def train(args):
    t0 = time.time()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )
        import wandb

    accelerator_project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit, automatic_checkpoint_naming=True
    )
    accelerator_grad_kwargs = GradScalerKwargs(init_scale=1)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=args.output_dir,
        logging_dir=logging_dir,
        project_config=accelerator_project_config,
        kwargs_handlers=[
            accelerator_grad_kwargs
        ],  # Set GradScaler initial scale to 1.0 to avoid overflow.
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        if args.seed is None:
            args.seed = np.random.randint(1, 100000)
            logger.info(f"Using random seed {args.seed}")
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load the VAE.
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
    )
    # We are only training the decoder.
    vae.requires_grad_(False)
    vae.decoder.requires_grad_(True)
    vae.encoder.eval()
    vae.decoder.train()

    # Create EMA for the vae.
    if args.use_ema:
        ema_vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
        )
        ema_vae = EMAModel(ema_vae.parameters(), model_cls=AutoencoderKL, model_config=ema_vae.config)
    else:
        ema_vae = None

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            vae.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    # Setup for training checkpoints
    prepare_checkpointing(accelerator, args, ema_vae)

    if args.gradient_checkpointing:
        vae.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        vae.decoder.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # LPIPS + Discriminator loss
    disc_start = args.discriminator_start
    perceptual_loss = LPIPSWithDiscriminator(
        disc_start, logvar_init=0.0, kl_weight=0.000001, pixelloss_weight=1.0,
        disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=0.5,
        perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
        disc_loss="hinge"
    )
    perceptual_loss.train()
    opt_disc = optimizer_cls(
        perceptual_loss.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get dataloader
    train_dataset, train_dataloader = get_dataloader(accelerator, args)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,  # * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps,  # * args.gradient_accumulation_steps,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Prepare everything with our `accelerator`.
    vae, optimizer, perceptual_loss, opt_disc, train_dataloader, lr_scheduler = accelerator.prepare(
        vae, optimizer, perceptual_loss, opt_disc, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast inference-only model weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    # weight_dtype = torch.float32
    # if accelerator.mixed_precision == "fp16":
    #     logger.info("Using torch.float16")
    #     weight_dtype = torch.float16
    # elif accelerator.mixed_precision == "bf16":
    #     logger.info("Using torch.bfloat16")
    #     weight_dtype = torch.bfloat16

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("signal-diffusion")
        for tracker in accelerator.trackers:
            tracker.store_init_configuration(args)

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.relpath(args.resume_from_checkpoint, args.output_dir)
        else:
            # Get the most recent checkpoint
            os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
            dirs = os.listdir(os.path.join(args.output_dir, "checkpoints"))
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("_")[1]))
            path = os.path.join("checkpoints", dirs[-1]) if len(dirs) > 0 else None

        if path is None:
            logger.info(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            logger.info(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            print("Finished load_state")
            with open(os.path.join(args.output_dir, path, "global_step.txt"), "r") as f:
                global_step = int(f.read())

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps
            )

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    # Validate same images each epoch
    validation_idxs = [
        np.random.randint(0, len(train_dataset))
        for _ in range(args.num_validation_images)
    ]

    for epoch in range(first_epoch, args.num_train_epochs):
        vae.decoder.train()
        train_loss = 0.0
        grad_norm = 0.0
        torch.cuda.reset_peak_memory_stats()
        # Optionally skip first N batches to reach epoch/validation quickly
        for step, batch in enumerate(
            accelerator.skip_first_batches(train_dataloader, num_batches=0)
        ):
            # Skip steps until we reach the resumed step
            if (args.resume_from_checkpoint and epoch == first_epoch and step < resume_step):
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(vae):
                target = batch["pixel_values"]

                # Convert images to latent space
                latents = vae.encode(target).latent_dist
                sampled_latents = latents.sample()

                # Convert latents back to image space
                model_pred = vae.decode(sampled_latents).sample

                # Compute loss
                loss, logs = perceptual_loss(target, model_pred, latents, 0, global_step,
                                             last_layer=vae.decoder.conv_out.weight, split="train",
                                             weights=None)

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        vae.decoder.parameters(), args.max_grad_norm
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Discriminator update
                with torch.no_grad():
                    fakes = vae.decode(sampled_latents).sample
                    # Quantize to prevent cheating
                    fakes_int = ((fakes.clip(-1, 1) + 1) / 2 * 255).to(torch.uint8)
                    fakes = (fakes_int.float() / 255 - 0.5) * 2
                disc_loss, disc_logs = perceptual_loss(target, fakes, latents, 1, global_step,
                                                       last_layer=vae.decoder.conv_out.weight, split="train",
                                                       weights=None)
                avg_disc_loss = accelerator.gather(disc_loss.repeat(args.train_batch_size)).mean()
                train_disc_loss = avg_disc_loss.item() / args.gradient_accumulation_steps
                accelerator.backward(disc_loss)
                opt_disc.step()
                opt_disc.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                # Batch accumulation complete, update global steps
                progress_bar.update(1)
                global_step += 1
                log_dict = {
                    "train_loss": train_loss,
                    "train_disc_loss": train_disc_loss,
                    "grad_norm": grad_norm.item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                log_dict.update(logs)
                log_dict.update(disc_logs)
                accelerator.log(log_dict, step=global_step,)
                train_loss = 0.0
                train_disc_loss = 0.0
                grad_norm = 0.0

                # Checkpointing
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = accelerator.save_state()
                        with open(os.path.join(save_path, "global_step.txt",), "w") as f:
                            f.write(str(global_step))
                        logger.info(f"Saved state to {save_path}")

                # Validation
                if global_step % args.validation_steps == 0:
                    examples = [train_dataset[i] for i in validation_idxs]
                    with torch.no_grad():
                        image_latent_tuples = [
                            encode_sample(vae, example) for example in examples
                        ]
                        images, latents = list(zip(*image_latent_tuples))
                        decoded = [decode_latents(vae, latent) for latent in latents]
                    images = list(images)
                    recon_images = [numpy_to_pil(dec)[0] for dec in decoded]
                    all_images = images + recon_images
                    # Intersperse target and reconstructed images
                    all_images = all_images[0::2] + all_images[1::2]

                    if args.audio:
                        mel = Mel(
                            x_res=args.resolution,
                            y_res=args.resolution,
                            sample_rate=args.audio_fs,
                            n_fft=2048,
                            hop_length=args.resolution,
                            top_db=80,
                            n_iter=32,
                        )
                        audios = [
                            mel.image_to_audio(im.convert("L")) for im in all_images
                        ]

                    for tracker in accelerator.trackers:
                        if tracker.name == "tensorboard":
                            np_images = np.stack([np.asarray(img) for img in images])
                            tracker.writer.add_images(
                                "validation", np_images, epoch, dataformats="NHWC"
                            )
                            if args.audio:
                                tracker.writer.add_audio(
                                    "validation_audio",
                                    torch.tensor(np.hstack(audios)),
                                    epoch,
                                    sample_rate=args.audio_fs,
                                )
                        if tracker.name == "wandb":
                            tracker.log(
                                {
                                    "validation": [
                                        wandb.Image(
                                            image,
                                            caption=f"{i // 2}: original={i % 2 == 0}",
                                        )
                                        for i, image in enumerate(all_images)
                                    ]
                                },
                                step=global_step,
                            )
                            if args.audio:
                                tracker.log(
                                    {
                                        "validation_audio": [
                                            wandb.Audio(
                                                audio,
                                                caption=f"{i // 2}: original={i % 2 == 0}",
                                                sample_rate=args.audio_fs,
                                            )
                                            for i, audio in enumerate(audios)
                                        ]
                                    },
                                    step=global_step,
                                )

            # Sub-batch progress bar update
            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        train_peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
        accelerator.log({"train_peak_mem": train_peak_mem}, step=epoch)

    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        vae.save_pretrained(args.output_dir)

    dt = time.time() - t0
    logger.info(
        f"Training finished in {dt // 3600:.0f}hr {dt % 3600 // 60:.0f}min {dt % 60:.0f}s"
    )
    accelerator.end_training()


@dataclass
class VAETrainConfig:
    seed: Optional[int] = 42
    # Model and data locations
    output_dir: str = "vae-stft-fma"
    pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
    revision: Optional[str] = None
    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    train_data_dir: str = "/data/shared/signal-diffusion/fma_preprocessed"
    image_column: str = "image"
    caption_column: str = "text"
    cache_dir: Optional[str] = None
    use_ema: bool = False
    # Audio output
    audio: bool = True
    audio_fs: int = 22050
    # Preprocessing
    center_crop: bool = True
    random_flip: bool = False
    # Validation
    num_validation_images: int = 4
    validation_steps: int = 100
    # Training duration and batches
    max_train_samples: Optional[int] = None
    resolution: int = 512
    train_batch_size: int = 1
    num_train_epochs: int = 3
    max_train_steps: Optional[int] = None
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    discriminator_start: int = 10
    # Learning rate
    learning_rate: float = 1e-5
    scale_lr: bool = False
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 50
    # Optimizer
    use_8bit_adam: bool = False
    adam_beta1: float = 0.5  # 0.9
    adam_beta2: float = 0.9  # 0.999
    adam_weight_decay: float = 0.0
    adam_epsilon: float = 1e-08
    max_grad_norm: float = 1.0
    # Accelerate
    allow_tf32: bool = True
    dataloader_num_workers: int = 2
    logging_dir: str = "logs"
    mixed_precision: Optional[str] = "fp16"
    report_to: str = "wandb"
    local_rank: int = -1
    enable_xformers_memory_efficient_attention: bool = True
    # Checkpointing
    checkpointing_steps: int = 1000
    checkpoints_total_limit: Optional[int] = None
    resume_from_checkpoint: Optional[Union[int, str]] = "latest"
    # Huggingface
    push_to_hub: bool = False
    hub_token: Optional[str] = None
    hub_model_id: Optional[str] = None

    def __repr__(self):
        header = "VAETrainConfig("
        lines = [header]
        cur_line = "\t"
        for k, v in self.__dict__.items():
            if len(cur_line) < 50:
                cur_line += f"{k}={v}, "
            else:
                lines.append(cur_line)
                cur_line = f"\t{k}={v}, "
        lines.append(cur_line)
        lines.append(")")
        return "\n".join(lines)


if __name__ == "__main__":
    train(VAETrainConfig())
