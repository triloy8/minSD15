# Copyright 2025 HuggingFace Inc.
# Modifications Copyright 2025 triloy8
#
# SPDX-License-Identifier: Apache-2.0
#
# Provenance:
#   • Original file:
#       https://github.com/huggingface/diffusers
#
# Changes by triloy8:
#   • Deconstructed pipeline

from diffusers import AutoencoderKL
from min_eulerd import EulerDiscreteScheduler
from min_unet import UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextConfig
import torch
from safetensors.torch import load_file
from tqdm.auto import tqdm
from PIL import Image
import json

#####################################################################
######################## LOADING BASE MODELS ########################
#####################################################################

clip_tokenizer_config = json.load(open(str("./config/tokenizer/tokenizer_config.json")))
clip_config = json.load(open("./config/clip_config.json"))
vae_config = json.load(open("./config/vae_config.json"))

scheduler = EulerDiscreteScheduler()

clip_tokenizer_vocab_path = "./config/tokenizer/vocab.json"
clip_tokenizer_merges_path = "./config/tokenizer/merges.txt"
clip_tokenizer = CLIPTokenizer(clip_tokenizer_vocab_path, clip_tokenizer_merges_path, model_max_length=77)

clip_model_path = "./weights/clip.fp16.safetensors"
clip_state_dict = load_file(clip_model_path)
clip_state_dict.pop("text_model.embeddings.position_ids")
clip_text_config = CLIPTextConfig(**clip_config)
clip = CLIPTextModel(clip_text_config)
clip.load_state_dict(clip_state_dict)

unet_model_path = "./weights/unet.fp16.safetensors"
unet_state_dict = load_file(unet_model_path)
unet = UNet2DConditionModel()
unet.load_state_dict(unet_state_dict)

vae_model_path = "./weights/vae.fp16.safetensors"
vae_state_dict = load_file(vae_model_path)
vae = AutoencoderKL(**vae_config)
vae.load_state_dict(vae_state_dict)

#####################################################################
######################## MODELS TO DEVICE ###########################
#####################################################################

torch_device = "cuda"

clip.to(torch_device, dtype=torch.float16)
unet.to(torch_device, dtype=torch.float16)
vae.to(torch_device, dtype=torch.float16)

#####################################################################
########################## PIPE PARAMS ##############################
#####################################################################

prompt = ["a pink dog"]
height = 512
width = 512
num_inference_steps = 25
guidance_scale = 7.5
seed = 3
generator = torch.cuda.manual_seed(seed)
batch_size = len(prompt)

#####################################################################
######################### CONDITIONING ##############################
#####################################################################

text_input = clip_tokenizer(
    prompt, padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt"
)

with torch.no_grad():
    text_embeddings = clip(text_input.input_ids.to(torch_device))[0]

max_length = text_input.input_ids.shape[-1]
uncond_input = clip_tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
uncond_embeddings = clip(uncond_input.input_ids.to(torch_device))[0]

text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

#####################################################################
############################# SAMPLING ##############################
#####################################################################

latents = torch.randn(
    (batch_size, unet.in_channels, height // 8, width // 8),
    generator=generator,
    device=torch_device,
    dtype=torch.float16
)

latents = latents * scheduler.init_noise_sigma

scheduler.set_timesteps(num_inference_steps)

for t in tqdm(scheduler.timesteps):
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = scheduler.scale_model_input(latent_model_input)

    with torch.no_grad():
        noise_pred = unet(latent_model_input, t.to(torch_device, dtype=torch.float16), encoder_hidden_states=text_embeddings)[0]

    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    latents = scheduler.step(model_output=noise_pred, sample=latents)

#####################################################################
############################# DECODING ##############################
#####################################################################

latents = 1 / 0.18215 * latents
with torch.no_grad():
    image = vae.decode(latents).sample

image = (image / 2 + 0.5).clamp(0, 1).squeeze()
image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
image = Image.fromarray(image)

image.save("output/generated.png")
