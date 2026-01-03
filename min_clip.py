# Copyright 2025 HuggingFace Inc.
# Further Modifications Copyright 2025 triloy8
#
# SPDX-License-Identifier: Apache-2.0
#
# Provenance:
#   • Original file:
#       https://github.com/huggingface/transformers
#
# Changes by triloy8:
#   • Left only CLIPTextModel w/ SD1.5 config

from typing import Optional, Tuple

import torch
from torch import nn


def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
) -> tuple[torch.Tensor, None]:

    # Instead of relying on the value set in the module directly, we use the is_causal passed in kwargs if it is presented
    is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
    # print("is_causal")
    # print(is_causal)

    # SDPA's Flash Attention (and cuDNN) kernels rely on the `is_causal` flag. However, there are certain conditions:
    # - Not in decoding phase (otherwise we want full attention on the single query token)
    # - Attention mask is not to be provided (even if it is a causal pattern)
    # - Internally, we marked this as compatible with causal, i.e. it is a decoder attention type
    #
    # Quirks on the conditionals:
    # - We avoid inline passing this to the SDPA function directly to support both torch.compile's dynamic shapes and
    #   full graph options. Otherwise, dynamic shapes are prevented from compiling.
    # - It is important to check first for the shape, otherwise compile will fail with
    #   `argument 'is_causal' must be bool, not SymBool`.
    is_causal = query.shape[2] > 1 and attention_mask is None and is_causal

    # print("is_causal")
    # print(is_causal)

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
        # **sdpa_kwargs,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None


class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * torch.sigmoid(1.702 * input)


class CLIPTextEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        embed_dim = 768

        self.token_embedding = nn.Embedding(49408, embed_dim)
        self.position_embedding = nn.Embedding(77, embed_dim)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(77).expand((1, -1)), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        max_position_embedding = self.position_embedding.weight.shape[0]

        if seq_length > max_position_embedding:
            raise ValueError(
                f"Sequence length must be less than max_position_embeddings (got `sequence length`: "
                f"{seq_length} and max_position_embeddings: {max_position_embedding}"
            )

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings


class CLIPAttention(nn.Module):
    def __init__(self,):
        super().__init__()
        self.embed_dim = 768
        self.num_heads = 12
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.is_causal = False

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool | None = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

        batch_size, seq_length, embed_dim = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(batch_size, seq_length, -1, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, -1, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, -1, self.head_dim).transpose(1, 2)

        attn_output, attn_weights = sdpa_attention_forward(
            self,
            queries,
            keys,
            values,
            attention_mask,
            scaling=self.scale,
            is_causal=is_causal,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class CLIPMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation_fn = QuickGELUActivation()
        self.fc1 = nn.Linear(768, 3072)
        self.fc2 = nn.Linear(3072, 768)

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        
        return hidden_states


class CLIPEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 768
        self.self_attn = CLIPAttention()
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=1e-05)
        self.mlp = CLIPMLP()
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=1e-05)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        is_causal: bool | None = None,
    ) -> torch.FloatTensor:
        
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            is_causal=is_causal,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class CLIPEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([CLIPEncoderLayer() for _ in range(12)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool | None = None,
    ) -> torch.Tensor:
        
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask,
                is_causal,
            )

        last_hidden_state=hidden_states

        return last_hidden_state


class CLIPTextTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        embed_dim = 768
        self.embeddings = CLIPTextEmbeddings()
        self.encoder = CLIPEncoder()
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=1e-05)

        # For `pooled_output` computation
        self.eos_token_id = 2

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        # attn mask must be -> (batch, heads or 1, q_len, kv_len)
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        encoder_last_hidden_state = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            is_causal=True,
        )

        last_hidden_state = self.final_layer_norm(encoder_last_hidden_state)

        # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
        # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
        # ------------------------------------------------------------
        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
        ]

        last_hidden_state=last_hidden_state
        pooler_output=pooled_output

        return last_hidden_state, pooler_output


class CLIPTextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_model = CLIPTextTransformer()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # last_hidden_state, pooler_output
        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
