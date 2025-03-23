from timeit import default_timer as timer
from datetime import timedelta
from PIL import Image
import os
import numpy as np
from einops import rearrange
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from packaging import version
from PIL import Image
from tqdm import tqdm

from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import AttnProcsLayers, LoraLoaderMixin
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    SlicedAttnAddedKVProcessor,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from natsort import natsorted

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.17.0")


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")

def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs

def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=False):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds


class LoRADataset(Dataset):
    def __init__(self, data_dir, use_controlnet=False, transform=None, control_image_processor=None, do_classifier_free_guidance=True):
        self.data_dir = data_dir
        self.use_controlnet = use_controlnet
        self.images = [os.path.join(data_dir, 'image', img) for img in os.listdir(os.path.join(data_dir, "image")) if img.endswith(".jpg")]
        if self.use_controlnet:
            self.control_images = [os.path.join(data_dir, 'dwpose', img) for img in os.listdir(os.path.join(data_dir, "dwpose")) if img.endswith(".jpg")]
        self.transform = transform
        self.control_image_processor = control_image_processor
        self.do_classifier_free_guidance = do_classifier_free_guidance

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        if self.use_controlnet:
            condition_path = self.control_images[idx]

        image = Image.open(image_path).convert("RGB")
        if self.use_controlnet:
            condition = Image.open(condition_path).convert("RGB")
            condition = self.convert_to_control_image(condition)
        else:
            condition = None
    
        if self.transform:
            image = self.transform(image)
        
        sample = {'image': image, 'condition': condition}
        return idx, sample
    
    def convert_to_control_image(self, image, height=512, width=512):
        image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        if self.do_classifier_free_guidance:
            image = torch.cat([image] * 2)
        return image
    
    def collate_fn(self, batch_list):
        # this function is borrowed from https://github.com/lioryariv/idr/blob/main/code/datasets/scene_dataset.py
        # get list of dictionaries and returns sample, ground_truth as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    try:
                        ret[k] = torch.stack([obj[k] for obj in entry])
                    except:
                        ret[k] = [obj[k] for obj in entry]
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)


def train_lora_from_images(data_dir, 
                           prompt, 
                           save_lora_dir, 
                           model_path=None, 
                           tokenizer=None, 
                           text_encoder=None, 
                           vae=None, 
                           unet=None, 
                           noise_scheduler=None, 
                           lora_epochs=5, 
                           lora_lr=2e-4, 
                           lora_rank=16, 
                           safe_serialization=False, 
                           batch_size=4,
                           use_controlnet=False,
                           control_image_processor=None,
                           controlnet_weight=0.8,
                           controlnets=None):
    
    # initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        # mixed_precision='fp16'
    )
    set_seed(0)

    # Load the tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            subfolder="tokenizer",
            revision=None,
            use_fast=False,
        )
    # initialize the model
    if noise_scheduler is None:
        noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
    if text_encoder is None:
        text_encoder_cls = import_model_class_from_model_name_or_path(model_path, revision=None)
        text_encoder = text_encoder_cls.from_pretrained(
            model_path, subfolder="text_encoder", revision=None
        )
    if vae is None:
        vae = AutoencoderKL.from_pretrained(
            model_path, subfolder="vae", revision=None
        )
    if unet is None:
        unet = UNet2DConditionModel.from_pretrained(
            model_path, subfolder="unet", revision=None
        )

    # set device and dtype
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    unet.to(device)
    vae.to(device)
    text_encoder.to(device)

    # initialize UNet LoRA
    unet_lora_attn_procs = {}
    for name, attn_processor in unet.attn_processors.items():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            raise NotImplementedError("name must start with up_blocks, mid_blocks, or down_blocks")

        if isinstance(attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
            lora_attn_processor_class = LoRAAttnAddedKVProcessor
        else:
            lora_attn_processor_class = (
                LoRAAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else LoRAAttnProcessor
            )
        unet_lora_attn_procs[name] = lora_attn_processor_class(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank
        )
    unet.set_attn_processor(unet_lora_attn_procs)
    unet_lora_layers = AttnProcsLayers(unet.attn_processors)

    # if type(image) == np.ndarray:
    #     image = Image.fromarray(image)
        
    # initialize latent distribution
    image_transforms = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.RandomCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    dataset = LoRADataset(data_dir, 
                          use_controlnet=False, 
                          transform=image_transforms, 
                          control_image_processor=control_image_processor, 
                          do_classifier_free_guidance=False)
    
    # Create DataLoader
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=True,
                            collate_fn=dataset.collate_fn,
                            num_workers=batch_size*2,
                            )
    # data_iterator = iter(dataloader)

    # Optimizer creation
    params_to_optimize = (unet_lora_layers.parameters())
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=lora_lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=lora_epochs * len(dataloader),
        num_cycles=1,
        power=1.0,
    )

    # prepare accelerator
    unet_lora_layers = accelerator.prepare_model(unet_lora_layers)
    optimizer = accelerator.prepare_optimizer(optimizer)
    lr_scheduler = accelerator.prepare_scheduler(lr_scheduler)

    # initialize text embeddings
    with torch.no_grad():
        text_inputs = tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None)
        text_embedding = encode_prompt(
            text_encoder,
            text_inputs.input_ids,
            text_inputs.attention_mask,
            text_encoder_use_attention_mask=False
        )

    unet.train()
    
    # latents_dist = vae.encode(image).latent_dist
    # for _ in progress.tqdm(range(lora_steps), desc="Training LoRA..."):
    for epoch in range(lora_epochs):
        for data_idx, (sample_idx, sample) in tqdm(enumerate(dataloader), desc="Training LoRA...", total=len(dataloader)):
            images, condition_images = sample['image'], sample['condition']
            images = images.to(device)
            if use_controlnet:
                condition_images = condition_images.to(device)

            latents_dist = vae.encode(images).latent_dist
            model_input = latents_dist.sample() * vae.config.scaling_factor         # NOTE model_input: [1, 4, 64, 64]
            # Sample noise that we'll add to the latents
            noise = torch.randn_like(model_input)
            bsz, channels, height, width = model_input.shape
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
            )
            timesteps = timesteps.long()
            
            # Add noise to the model input according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
            _text_embedding = text_embedding.repeat(bsz, 1, 1)

            if use_controlnet:
                with torch.no_grad():
                    # In case using controlnet
                    guess_mode = False
                    control_model_input = noisy_model_input        # NOT ASSUMING GUESS MODE
                    
                    cond_scale = controlnet_weight # Repeated define (for cap of multi-step version)
                    control_images = control_images.to(device)
                    if len(control_images.shape) == 5:
                        control_images = control_images.reshape(-1, *control_images.shape[2:])
                    elif len(control_images.shape) == 3:
                        control_images = control_images[NotImplemented]

                    down_block_res_samples, mid_block_res_sample = controlnets(
                        control_model_input,
                        timesteps,
                        encoder_hidden_states=_text_embedding,
                        controlnet_cond=control_images,
                        conditioning_scale=cond_scale,
                        guess_mode=guess_mode,
                        return_dict=False,
                    )
            else:
                down_block_res_samples, mid_block_res_sample = None, None

            # Predict the noise residual
            model_pred = unet(noisy_model_input, 
                              timesteps,
                              _text_embedding,
                              down_block_additional_residuals=down_block_res_samples,
                              mid_block_additional_residual=mid_block_res_sample,
                              ).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(model_input, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            if data_idx % 20 == 0:
                tqdm.write(f"Epoch: {epoch}, Batch: {data_idx}, Loss: {loss}")

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    # save the trained lora
    # unet = unet.to(torch.float32)
    # vae = vae.to(torch.float32)
    # text_encoder = text_encoder.to(torch.float32)

    weight_name = f"lora_{lora_epochs}.ckpt"
    # unwrap_model is used to remove all special modules added when doing distributed training
    # so here, there is no need to call unwrap_model
    # unet_lora_layers = accelerator.unwrap_model(unet_lora_layers)
    LoraLoaderMixin.save_lora_weights(
        save_directory=save_lora_dir,
        unet_lora_layers=unet_lora_layers,
        text_encoder_lora_layers=None,
        weight_name=weight_name,
        safe_serialization=safe_serialization
    )


# model_path: path of the model
# image: input image, have not been pre-processed
# save_lora_dir: the path to save the lora
# prompt: the user input prompt
# lora_steps: number of lora training step
# lora_lr: learning rate of lora training
# lora_rank: the rank of lora
def train_lora(image, prompt, save_lora_dir, model_path=None, tokenizer=None, text_encoder=None, vae=None, unet=None, noise_scheduler=None, lora_steps=200, lora_lr=2e-4, lora_rank=16, weight_name=None, safe_serialization=False, progress=tqdm):
    # initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        # mixed_precision='fp16'
    )
    set_seed(0)

    # Load the tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            subfolder="tokenizer",
            revision=None,
            use_fast=False,
        )
    # initialize the model
    if noise_scheduler is None:
        noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
    if text_encoder is None:
        text_encoder_cls = import_model_class_from_model_name_or_path(model_path, revision=None)
        text_encoder = text_encoder_cls.from_pretrained(
            model_path, subfolder="text_encoder", revision=None
        )
    if vae is None:
        vae = AutoencoderKL.from_pretrained(
            model_path, subfolder="vae", revision=None
        )
    if unet is None:
        unet = UNet2DConditionModel.from_pretrained(
            model_path, subfolder="unet", revision=None
        )

    # set device and dtype
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    unet.to(device)
    vae.to(device)
    text_encoder.to(device)

    # initialize UNet LoRA
    unet_lora_attn_procs = {}
    for name, attn_processor in unet.attn_processors.items():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            raise NotImplementedError("name must start with up_blocks, mid_blocks, or down_blocks")

        if isinstance(attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
            lora_attn_processor_class = LoRAAttnAddedKVProcessor
        else:
            lora_attn_processor_class = (
                LoRAAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else LoRAAttnProcessor
            )
        unet_lora_attn_procs[name] = lora_attn_processor_class(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank
        )
    unet.set_attn_processor(unet_lora_attn_procs)
    unet_lora_layers = AttnProcsLayers(unet.attn_processors)

    # Optimizer creation
    params_to_optimize = (unet_lora_layers.parameters())
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=lora_lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=lora_steps,
        num_cycles=1,
        power=1.0,
    )

    # prepare accelerator
    unet_lora_layers = accelerator.prepare_model(unet_lora_layers)
    optimizer = accelerator.prepare_optimizer(optimizer)
    lr_scheduler = accelerator.prepare_scheduler(lr_scheduler)

    # initialize text embeddings
    with torch.no_grad():
        text_inputs = tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None)
        text_embedding = encode_prompt(
            text_encoder,
            text_inputs.input_ids,
            text_inputs.attention_mask,
            text_encoder_use_attention_mask=False
        )

    if type(image) == np.ndarray:
        image = Image.fromarray(image)
        
    # initialize latent distribution
    image_transforms = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.RandomCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    image = image_transforms(image).to(device)
    image = image.unsqueeze(dim=0)
    
    latents_dist = vae.encode(image).latent_dist
    for _ in progress.tqdm(range(lora_steps), desc="Training LoRA..."):
        unet.train()
        model_input = latents_dist.sample() * vae.config.scaling_factor         # NOTE model_input: [1, 4, 64, 64]
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(model_input)
        bsz, channels, height, width = model_input.shape
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
        )
        timesteps = timesteps.long()

        # Add noise to the model input according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

        # Predict the noise residual
        model_pred = unet(noisy_model_input, timesteps, text_embedding).sample

        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(model_input, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    # save the trained lora
    # unet = unet.to(torch.float32)
    # vae = vae.to(torch.float32)
    # text_encoder = text_encoder.to(torch.float32)

    # unwrap_model is used to remove all special modules added when doing distributed training
    # so here, there is no need to call unwrap_model
    # unet_lora_layers = accelerator.unwrap_model(unet_lora_layers)
    LoraLoaderMixin.save_lora_weights(
        save_directory=save_lora_dir,
        unet_lora_layers=unet_lora_layers,
        text_encoder_lora_layers=None,
        weight_name=weight_name,
        safe_serialization=safe_serialization
    )
    
def load_lora(unet, lora_0, lora_1, alpha):
    lora = {}
    for key in lora_0:
        lora[key] = (1 - alpha) * lora_0[key] + alpha * lora_1[key]
    unet.load_attn_procs(lora)
    return unet
