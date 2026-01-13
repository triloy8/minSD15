# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Tuple, Union, List
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

logger = logging.getLogger(__name__)

VAE_IN_CHANNELS = 3
VAE_OUT_CHANNELS = 3
VAE_LATENT_CHANNELS = 4
VAE_BLOCK_OUT_CHANNELS = (128, 256, 512, 512)
VAE_LAYERS_PER_BLOCK = 2
VAE_NORM_NUM_GROUPS = 32
VAE_ACT_FN = "silu"
VAE_SAMPLE_SIZE = 512
VAE_DOWN_BLOCK_TYPES = ("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D")
VAE_UP_BLOCK_TYPES = ("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D")


def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional[Union[str, "torch.device"]] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    if isinstance(device, str):
        device = torch.device(device)
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                print(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slightly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents


class Attention(nn.Module):
    def __init__(
        self,
        channels: int,
        heads: int,
        dim_head: int,
        rescale_output_factor: float = 1.0,
        eps: float = 1e-6,
        norm_num_groups: Optional[int] = None,
        bias: bool = True,
    ):
        super().__init__()
        norm_num_groups = norm_num_groups or 32
        self.rescale_output_factor = rescale_output_factor
        self.group_norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=channels, eps=eps, affine=True)

        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.to_q = nn.Linear(channels, inner_dim, bias=bias)
        self.to_k = nn.Linear(channels, inner_dim, bias=bias)
        self.to_v = nn.Linear(channels, inner_dim, bias=bias)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, channels, bias=bias), nn.Dropout(0.0)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.group_norm(hidden_states)

        b, c, h, w = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(b, h * w, c)

        q = self.to_q(hidden_states)
        k = self.to_k(hidden_states)
        v = self.to_v(hidden_states)

        q = q.view(b, h * w, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(b, h * w, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(b, h * w, self.heads, self.dim_head).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_probs = attn_scores.float().softmax(dim=-1).to(q.dtype)

        hidden_states = torch.matmul(attn_probs, v)
        hidden_states = hidden_states.transpose(1, 2).reshape(b, h * w, self.heads * self.dim_head)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        hidden_states = hidden_states.reshape(b, h, w, c).permute(0, 3, 1, 2)

        hidden_states = (hidden_states + residual) / self.rescale_output_factor

        return hidden_states


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        # make sure sample is on the same device as the parameters and has same dtype
        sample = randn_tensor(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        x = self.mean + self.std * sample
        return x

    def kl(self, other: "DiagonalGaussianDistribution" = None) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample: torch.Tensor, dims: Tuple[int, ...] = [1, 2, 3]) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self) -> torch.Tensor:
        return self.mean


class ResnetBlock2D(nn.Module):
    r"""
    A Resnet block.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer. If None, same as `in_channels`.
        dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
        temb_channels (`int`, *optional*, default to `512`): the number of channels in timestep embedding.
        groups (`int`, *optional*, default to `32`): The number of groups to use for the first normalization layer.
        groups_out (`int`, *optional*, default to None):
            The number of groups to use for the second normalization layer. if set to None, same as `groups`.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
        non_linearity (`str`, *optional*, default to `"swish"`): the activation function to use.
        time_embedding_norm (`str`, *optional*, default to `"default"` ): Time scale shift config.
            By default, apply timestep embedding conditioning with a simple shift mechanism. Choose "scale_shift" for a
            stronger conditioning with scale and shift.
        kernel (`torch.Tensor`, optional, default to None): FIR filter, see
            [`~models.resnet.FirUpsample2D`] and [`~models.resnet.FirDownsample2D`].
        output_scale_factor (`float`, *optional*, default to be `1.0`): the scale factor to use for the output.
        use_in_shortcut (`bool`, *optional*, default to `True`):
            If `True`, add a 1x1 nn.conv2d layer for skip-connection.
        up (`bool`, *optional*, default to `False`): If `True`, add an upsample layer.
        down (`bool`, *optional*, default to `False`): If `True`, add a downsample layer.
        conv_shortcut_bias (`bool`, *optional*, default to `True`):  If `True`, adds a learnable bias to the
            `conv_shortcut` output.
        conv_2d_out_channels (`int`, *optional*, default to `None`): the number of channels in the output.
            If None, same as `out_channels`.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        groups: int = 32,
        eps: float = 1e-6,
        output_scale_factor: float = 1.0,
        conv_shortcut_bias: bool = True,
        conv_2d_out_channels: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_scale_factor = output_scale_factor

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = torch.nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)

        self.dropout = torch.nn.Dropout(dropout)
        conv_2d_out_channels = conv_2d_out_channels or out_channels
        self.conv2 = nn.Conv2d(out_channels, conv_2d_out_channels, kernel_size=3, stride=1, padding=1)

        self.nonlinearity = nn.SiLU()

        self.use_in_shortcut = self.in_channels != conv_2d_out_channels

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels,
                conv_2d_out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=conv_shortcut_bias,
            )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor.contiguous())

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


class Downsample2D(nn.Module):
    """A 2D downsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        padding (`int`, default `1`):
            padding for the convolution.
        name (`str`, default `conv`):
            name of the downsampling 2D layer.
    """

    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        kernel_size=3,
        bias=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        stride = 2
        self.conv = nn.Conv2d(
            self.channels, self.out_channels, kernel_size=kernel_size, stride=stride, padding=0, bias=bias
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.shape[1] == self.channels

        hidden_states = F.pad(hidden_states, (0, 1, 0, 1), mode="constant", value=0)

        assert hidden_states.shape[1] == self.channels

        hidden_states = self.conv(hidden_states)

        return hidden_states


class DownEncoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    output_scale_factor=output_scale_factor,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(out_channels, out_channels=out_channels)
                ]
            )
        else:
            self.downsamplers = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        return hidden_states


class UNetMidBlock2D(nn.Module):
    """
    A 2D UNet mid-block [`UNetMidBlock2D`] with multiple residual blocks and optional attention blocks.

    Args:
        in_channels (`int`): The number of input channels.
        temb_channels (`int`): The number of temporal embedding channels.
        dropout (`float`, *optional*, defaults to 0.0): The dropout rate.
        num_layers (`int`, *optional*, defaults to 1): The number of residual blocks.
        resnet_eps (`float`, *optional*, 1e-6 ): The epsilon value for the resnet blocks.
        resnet_time_scale_shift (`str`, *optional*, defaults to `default`):
            The type of normalization to apply to the time embeddings. This can help to improve the performance of the
            model on tasks with long-range temporal dependencies.
        resnet_act_fn (`str`, *optional*, defaults to `swish`): The activation function for the resnet blocks.
        resnet_groups (`int`, *optional*, defaults to 32):
            The number of groups to use in the group normalization layers of the resnet blocks.
        attn_groups (`Optional[int]`, *optional*, defaults to None): The number of groups for the attention blocks.
        resnet_pre_norm (`bool`, *optional*, defaults to `True`):
            Whether to use pre-normalization for the resnet blocks.
        add_attention (`bool`, *optional*, defaults to `True`): Whether to add attention blocks.
        attention_head_dim (`int`, *optional*, defaults to 1):
            Dimension of a single attention head. The number of attention heads is determined based on this value and
            the number of input channels.
        output_scale_factor (`float`, *optional*, defaults to 1.0): The output scale factor.

    Returns:
        `torch.Tensor`: The output of the last residual block, which is a tensor of shape `(batch_size, in_channels,
        height, width)`.

    """

    def __init__(
        self,
        in_channels: int,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
    ):
        super().__init__()
        attn_groups = resnet_groups

        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=0.0,
                output_scale_factor=output_scale_factor,
            )
        ]
        attentions = []

        for _ in range(num_layers):
            attentions.append(
                Attention(
                    in_channels,
                    heads=in_channels // attention_head_dim,
                    dim_head=attention_head_dim,
                    rescale_output_factor=output_scale_factor,
                    eps=resnet_eps,
                    norm_num_groups=attn_groups,
                    bias=True,
                )
            )

            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=0.0,
                    output_scale_factor=output_scale_factor,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states)
            hidden_states = resnet(hidden_states)

        return hidden_states


class Upsample2D(nn.Module):
    """A 2D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        name (`str`, default `conv`):
            name of the upsampling 2D layer.
    """

    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        kernel_size: int = 3,
        padding=1,
        bias=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.conv = nn.Conv2d(self.channels, self.out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.shape[1] == self.channels

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        hidden_states = self.conv(hidden_states)

        return hidden_states


class UpDecoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=0.0,
                    output_scale_factor=output_scale_factor,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, out_channels=out_channels)])
        else:
            self.upsamplers = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class Encoder(nn.Module):
    r"""
    The `Encoder` layer of a variational autoencoder that encodes its input into a latent representation.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        down_block_types (`Tuple[str, ...]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            The types of down blocks to use. See `~diffusers.models.unet_2d_blocks.get_down_block` for available
            options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
        double_z (`bool`, *optional*, defaults to `True`):
            Whether to double the number of output channels for the last block.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = ("DownEncoderBlock2D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = DownEncoderBlock2D(
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                dropout=0.0,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_groups=norm_num_groups,
            )

            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            output_scale_factor=1,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels
        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        r"""The forward method of the `Encoder` class."""

        sample = self.conv_in(sample)

        for down_block in self.down_blocks:
            sample = down_block(sample)

        sample = self.mid_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class Decoder(nn.Module):
    r"""
    The `Decoder` layer of a variational autoencoder that decodes its latent representation into an output sample.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        up_block_types (`Tuple[str, ...]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            The types of up blocks to use. See `~diffusers.models.unet_2d_blocks.get_up_block` for available options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
        norm_type (`str`, *optional*, defaults to `"group"`):
            The normalization type to use. Can be either `"group"` or `"spatial"`.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.up_blocks = nn.ModuleList([])

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            output_scale_factor=1,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            num_layers=1,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block = UpDecoderBlock2D(
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_groups=norm_num_groups,
            )

            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(
        self,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        r"""The forward method of the `Decoder` class."""

        sample = self.conv_in(sample)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        sample = self.mid_block(sample)
        sample = sample.to(upscale_dtype)

        for up_block in self.up_blocks:
            sample = up_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class AutoencoderKL(nn.Module):
    r"""
    A VAE model with KL loss for encoding images into latents and decoding latent representations into images.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to 4): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): Sample input size.
        scaling_factor (`float`, *optional*, defaults to 0.18215):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://huggingface.co/papers/2112.10752) paper.
        force_upcast (`bool`, *optional*, default to `True`):
            If enabled it will force the VAE to run in float32 for high image resolution pipelines, such as SD-XL. VAE
            can be fine-tuned / trained to a lower range without losing too much precision in which case `force_upcast`
            can be set to `False` - see: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
        mid_block_add_attention (`bool`, *optional*, default to `True`):
            If enabled, the mid_block of the Encoder and Decoder will have attention blocks. If set to false, the
            mid_block will only have resnet blocks
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["BasicTransformerBlock", "ResnetBlock2D"]

    def __init__(self):
        super().__init__()

        # pass init params to Encoder
        self.encoder = Encoder(
            in_channels=VAE_IN_CHANNELS,
            out_channels=VAE_LATENT_CHANNELS,
            down_block_types=VAE_DOWN_BLOCK_TYPES,
            block_out_channels=VAE_BLOCK_OUT_CHANNELS,
            layers_per_block=VAE_LAYERS_PER_BLOCK,
            norm_num_groups=VAE_NORM_NUM_GROUPS,
        )

        # pass init params to Decoder
        self.decoder = Decoder(
            in_channels=VAE_LATENT_CHANNELS,
            out_channels=VAE_OUT_CHANNELS,
            up_block_types=VAE_UP_BLOCK_TYPES,
            block_out_channels=VAE_BLOCK_OUT_CHANNELS,
            layers_per_block=VAE_LAYERS_PER_BLOCK,
            norm_num_groups=VAE_NORM_NUM_GROUPS,
        )

        self.quant_conv = nn.Conv2d(2 * VAE_LATENT_CHANNELS, 2 * VAE_LATENT_CHANNELS, 1)
        self.post_quant_conv = nn.Conv2d(VAE_LATENT_CHANNELS, VAE_LATENT_CHANNELS, 1)

        self.scaling_factor = 0.18215

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(x)
        enc = self.quant_conv(enc)

        return enc

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._encode(x)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def decode(self, z: torch.FloatTensor) -> torch.FloatTensor:
        return self._decode(z)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        r"""Forward pass returning decoded pixels."""
        x = sample
        moments = self.encode(x)
        posterior = DiagonalGaussianDistribution(moments)
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z)

        return dec
