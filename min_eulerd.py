# Copyright 2026 HuggingFace Inc.
# Modifications Copyright 2026 triloy8
#
# SPDX-License-Identifier: Apache-2.0
#
# Provenance:
#   • Original file:
#       https://github.com/huggingface/diffusers
#
# Changes by triloy8:
#   • Lean version 

from utils import interp

import torch

class EulerDiscreteScheduler():
    def __init__(self):
        self.beta_start = 0.00085
        self.beta_end = 0.012
        self.num_train_timesteps = 1000

        self.betas = torch.linspace(self.beta_start**0.5, self.beta_end**0.5, self.num_train_timesteps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        sigmas = (((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5).flip(0)
        self.sigmas = torch.cat([sigmas, torch.zeros(1)])

        self.init_noise_sigma = self.sigmas.max()

        self.timesteps = None
        
        self.step_index = 0

    def scale_model_input(self, sample):
        sigma = self.sigmas[self.step_index]
        sample = sample / ((sigma**2 + 1) ** 0.5)

        return sample

    def set_timesteps(self, num_inference_steps = None):
        timesteps = torch.linspace(0, self.num_train_timesteps - 1, num_inference_steps, dtype=torch.float32)
        
        sigmas = interp(x=timesteps, xp=torch.arange(0, len(self.sigmas)), fp=self.sigmas, dim=0, extrapolate="linear")
        self.sigmas = torch.cat([sigmas, torch.zeros(1)])

        self.timesteps = timesteps.flip(0)

    def step(self, model_output, sample):
        prev_sample = sample + (self.sigmas[self.step_index + 1]- self.sigmas[self.step_index]) * model_output

        self.step_index += 1

        return prev_sample