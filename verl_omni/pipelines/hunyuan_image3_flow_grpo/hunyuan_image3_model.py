# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""HunyuanImage3ForTraining — FSDP2-compatible HunyuanImage3 module for flow-GRPO training.

Ported from Tencent HunyuanImage-3.0 (``modeling_hunyuan_image_3.py``, Tencent Hunyuan
Community License).  Only the generation backbone is ported: the unified MoE transformer
(text + image share one 64-expert / topk-8 backbone), the UNet latent patchifier /
unpatchifier, timestep embedding, and 2-D RoPE.  The VAE, ViT, tokenizer, and generation
cache are left to the rollout side (vllm-omni).

The training ``forward`` reproduces the vllm-omni ``forward_call`` gen-image path:
embed text tokens, scatter noisy latent patches and the ``<timestep>`` token into the
embedding sequence, run the decoder stack with a causal-text / full-image attention
mask, then unpatch the image tokens back to a latent-space velocity.
"""

from __future__ import annotations

import json
import logging
import math
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from verl_omni.pipelines.non_diffusers_model_base import NonDiffusersModelBase

logger = logging.getLogger(__name__)

# ===================================================================
#  Config
# ===================================================================


@dataclass
class HunyuanImage3TrainingConfig:
    """Training-side subset of ``HunyuanImage3Config`` (generation backbone only).

    Values default to the HunyuanImage-3.0-Instruct checkpoint and are overridden
    from the checkpoint ``config.json`` when loaded through ``from_model_path``.
    """

    vocab_size: int = 133120
    hidden_size: int = 4096
    intermediate_size: int = 3072
    moe_intermediate_size: int = 3072
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    attention_head_dim: int = 128
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-5
    norm_type: str = "rms"
    max_position_embeddings: int = 22800
    rope_theta: float = 10000.0
    use_qk_norm: bool = True
    use_rotary_pos_emb: bool = True
    attention_bias: bool = False
    mlp_bias: bool = False
    num_experts: int = 64
    use_mixed_mlp_moe: bool = True
    num_shared_expert: int = 1
    moe_topk: int = 8
    moe_layer_num_skipped: int = 0
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.0
    group_limited_greedy: bool = False
    n_group: Optional[int] = None
    topk_group: Optional[int] = None
    # image generation
    img_proj_type: str = "unet"
    patch_size: int = 1
    patch_embed_hidden_dim: int = 1024
    latent_channels: int = 32
    vae_downsample_factor: tuple[int, int] = (16, 16)
    # token ids
    pad_token_id: int = 128009
    bos_token_id: int = 127958
    eos_token_id: int = 127957
    im_start_id: int = 128000
    im_end_id: int = 128001
    image_token_id: int = 128006

    def save_pretrained(self, save_directory: str) -> None:
        """Save config as JSON beside the weights."""
        from dataclasses import asdict

        os.makedirs(save_directory, exist_ok=True)
        output_path = os.path.join(save_directory, "config.json")
        with open(output_path, "w") as f:
            json.dump(asdict(self), f, indent=4, sort_keys=True)

    @classmethod
    def from_model_path(cls, model_path: str) -> HunyuanImage3TrainingConfig:
        """Parse the training config from the checkpoint ``config.json``."""
        cfg_path = os.path.join(model_path, "config.json")
        with open(cfg_path) as f:
            raw = json.load(f)

        # moe_topk / num_experts / num_shared_expert may be per-layer lists; the
        # checkpoint uses uniform values so take the first element.
        def _scalar(value, default):
            if value is None:
                return default
            if isinstance(value, list):
                return value[0] if value else default
            return value

        vae = raw.get("vae") or {}
        return cls(
            vocab_size=int(raw.get("vocab_size", 133120)),
            hidden_size=int(raw.get("hidden_size", 4096)),
            intermediate_size=int(raw.get("intermediate_size", 3072)),
            moe_intermediate_size=int(_scalar(raw.get("moe_intermediate_size"), 3072)),
            num_hidden_layers=int(raw.get("num_hidden_layers", 32)),
            num_attention_heads=int(raw.get("num_attention_heads", 32)),
            num_key_value_heads=int(raw.get("num_key_value_heads", 8) or 8),
            attention_head_dim=int(raw.get("attention_head_dim", raw.get("head_dim", 128))),
            hidden_act=str(raw.get("hidden_act", "silu")),
            rms_norm_eps=float(raw.get("rms_norm_eps", 1e-5)),
            norm_type=str(raw.get("norm_type", "rms")),
            max_position_embeddings=int(raw.get("max_position_embeddings", 22800)),
            rope_theta=float(raw.get("rope_theta", 10000.0)),
            use_qk_norm=bool(raw.get("use_qk_norm", True)),
            use_rotary_pos_emb=bool(raw.get("use_rotary_pos_emb", True)),
            attention_bias=bool(raw.get("attention_bias", False)),
            mlp_bias=bool(raw.get("mlp_bias", False)),
            num_experts=int(_scalar(raw.get("num_experts"), 64)),
            use_mixed_mlp_moe=bool(raw.get("use_mixed_mlp_moe", True)),
            num_shared_expert=int(_scalar(raw.get("num_shared_expert"), 1)),
            moe_topk=int(_scalar(raw.get("moe_topk"), 8)),
            moe_layer_num_skipped=int(raw.get("moe_layer_num_skipped", 0)),
            norm_topk_prob=bool(raw.get("norm_topk_prob", True)),
            routed_scaling_factor=float(raw.get("routed_scaling_factor", 1.0)),
            group_limited_greedy=bool(raw.get("group_limited_greedy", False)),
            n_group=raw.get("n_group"),
            topk_group=raw.get("topk_group"),
            img_proj_type=str(raw.get("img_proj_type", "unet")),
            patch_size=int(raw.get("patch_size", 1)),
            patch_embed_hidden_dim=int(raw.get("patch_embed_hidden_dim", 1024)),
            latent_channels=int(vae.get("latent_channels", 32)),
            vae_downsample_factor=tuple(raw.get("vae_downsample_factor", (16, 16))),
            pad_token_id=int(raw.get("pad_token_id", 128009)),
            bos_token_id=int(raw.get("bos_token_id", 127958)),
            eos_token_id=int(raw.get("eos_token_id", 127957)),
            im_start_id=int(raw.get("im_start_id", 128000)),
            im_end_id=int(raw.get("im_end_id", 128001)),
            image_token_id=int(raw.get("image_token_id", 128006)),
        )


# ===================================================================
#  Helper functions (ported from modeling_hunyuan_image_3.py)
# ===================================================================


def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
    """Create GLIDE-style sinusoidal timestep embeddings (``(N, dim)``)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def conv_nd(dims: int, *args, **kwargs) -> nn.Module:
    """Create a 1D/2D/3D convolution module."""
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    if dims == 2:
        return nn.Conv2d(*args, **kwargs)
    if dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def zero_module(module: nn.Module) -> nn.Module:
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def normalization(channels: int, **kwargs) -> nn.Module:
    """Standard GroupNorm (32 groups) used by the ResBlock."""
    return nn.GroupNorm(32, channels, **kwargs)


def _to_tuple(x, dim: int = 2):
    if isinstance(x, int):
        return (x,) * dim
    if len(x) == dim:
        return x
    raise ValueError(f"Expected length {dim} or int, but got {x}")


def get_meshgrid_nd(start, *args, dim: int = 2, device: str = "cpu") -> Tensor:
    """Get n-D meshgrid with start/stop/num (``[dim, ...]``) for 2-D RoPE."""
    if len(args) == 0:
        num = _to_tuple(start, dim=dim)
        start = (0,) * dim
        stop = num
    elif len(args) == 1:
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = [int(stop[i] - start[i]) for i in range(dim)]
    elif len(args) == 2:
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = _to_tuple(args[1], dim=dim)
    else:
        raise ValueError("len(args) should be 0, 1 or 2")

    axis_grid = []
    for i in range(dim):
        a, b, n = start[i], stop[i], num[i]
        g = torch.linspace(a, b, n + 1, dtype=torch.float32, device=device)[:n]
        axis_grid.append(g)
    return torch.stack(torch.meshgrid(*axis_grid, indexing="ij"), dim=0)  # [dim, H, W]


def build_2d_rope(
    seq_len: int,
    n_elem: int,
    image_infos: Optional[list[tuple[slice, tuple[int, int]]]] = None,
    device: Optional[torch.device] = None,
    base: int = 10000,
) -> tuple[Tensor, Tensor]:
    """Build 2-D RoPE cos/sin of shape ``(seq_len, n_elem)``.

    Image tokens get 2-D positions ``(beta_y, beta_x)`` derived from their grid
    coordinates; text tokens get linear positions.
    """
    assert n_elem % 4 == 0, f"n_elem must be divisible by 4, but got {n_elem}."

    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))
    theta = theta.reshape(1, n_elem // 4, 2)  # [1, half_d, 2]

    image_infos_list = [image_infos or []]
    x_sections = []
    y_sections = []
    for sample_image_infos in image_infos_list:
        last_pos = 0
        for sec_slice, (h, w) in sample_image_infos:
            ll = sec_slice.start
            if last_pos < ll:
                y_sections.append(torch.arange(last_pos, ll, device=device))
                x_sections.append(torch.arange(last_pos, ll, device=device))
            elif h is None:
                # Special boundary tokens (<boi>/<size>/<ratio>/<timestep>/<eoi>) share
                # overlapped positions; use linear ids for them.
                y_sections.append(torch.arange(sec_slice.start, sec_slice.stop, device=device))
                x_sections.append(torch.arange(sec_slice.start, sec_slice.stop, device=device))
                continue
            beta_y = ll + (w * h - h) / 2
            beta_x = ll + (w * h - w) / 2
            grid = get_meshgrid_nd((beta_y, beta_x), (beta_y + h, beta_x + w), device=device)  # [2, h, w]
            grid = grid.reshape(2, -1)
            y_sections.append(grid[0])
            x_sections.append(grid[1])
            last_pos = ll + w * h
        y_sections.append(torch.arange(last_pos, seq_len, device=device))
        x_sections.append(torch.arange(last_pos, seq_len, device=device))

    x_pos = torch.cat(x_sections).long()
    y_pos = torch.cat(y_sections).long()
    x_pos = x_pos[:seq_len]
    y_pos = y_pos[:seq_len]
    all_pos = torch.stack((y_pos, x_pos), dim=1).unsqueeze(1)  # [seq_len, 1, 2]

    idx_theta = (all_pos * theta).reshape(all_pos.shape[0], n_elem // 2).repeat(1, 2)
    return torch.cos(idx_theta), torch.sin(idx_theta)


def build_batch_2d_rope(
    seq_len: int,
    n_elem: int,
    image_infos: Optional[list[list[tuple[slice, tuple[int, int]]]]] = None,
    device: Optional[torch.device] = None,
    base: int = 10000,
) -> tuple[Tensor, Tensor]:
    """Batch 2-D RoPE: returns ``(cos, sin)`` of shape ``(B, seq_len, n_elem)``."""
    if image_infos is None:
        image_infos = [None]
    cos_list, sin_list = [], []
    for image_info in image_infos:
        cos, sin = build_2d_rope(seq_len, n_elem, image_infos=image_info, device=device, base=base)
        cos_list.append(cos)
        sin_list.append(sin)
    return torch.stack(cos_list, dim=0), torch.stack(sin_list, dim=0)


def rotate_half(x: Tensor) -> Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Apply 2-D rotary embeddings to q/k of shape ``(B, H, L, D)``."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    """Repeat KV heads ``n_rep`` times for GQA."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# ===================================================================
#  Building blocks
# ===================================================================


class HunyuanRMSNorm(nn.Module):
    """Hunyuan RMSNorm (T5-style, fp32 variance)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class HunyuanMLP(nn.Module):
    """SwiGLU MLP, shared by experts and the shared expert."""

    def __init__(self, config: HunyuanImage3TrainingConfig, is_shared_mlp: bool = False, is_moe: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hidden_act = config.hidden_act

        intermediate_size = config.intermediate_size
        if is_shared_mlp or is_moe:
            intermediate_size = config.moe_intermediate_size
        if is_shared_mlp:
            intermediate_size *= config.num_shared_expert
        if config.hidden_act == "silu":
            intermediate_size *= 2  # SwiGLU
            self.gate_and_up_proj = nn.Linear(self.hidden_size, intermediate_size, bias=config.mlp_bias)
            self.down_proj = nn.Linear(intermediate_size // 2, self.hidden_size, bias=config.mlp_bias)
        else:
            raise ValueError(f"Unsupported hidden_act: {config.hidden_act}")

    def forward(self, x: Tensor) -> Tensor:
        gate_and_up_proj = self.gate_and_up_proj(x)
        x1, x2 = gate_and_up_proj.chunk(2, dim=-1)
        return self.down_proj(x1 * F.silu(x2))


class HunyuanTopKGate(nn.Module):
    """Router gate with the simple top-k path used by the eager MoE."""

    def __init__(self, config: HunyuanImage3TrainingConfig):
        super().__init__()
        self.moe_topk = config.moe_topk
        self.wg = nn.Linear(config.hidden_size, config.num_experts, bias=False, dtype=torch.float32)

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        bsz, seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_size)
        if self.wg.weight.dtype == torch.float32:
            hidden_states = hidden_states.float()
        logits = self.wg(hidden_states)
        gates = F.softmax(logits, dim=1)
        # Stable sort so a checkpoint recompute can't flip a tied-score token to
        # a different expert (torch.topk is not stable for ties, breaking grad).
        sorted_gates, sorted_idx = torch.sort(gates, dim=-1, descending=True, stable=True)
        # Densify the top-k slice so MoE dispatch's ``view(-1)`` sees contiguous memory.
        topk_weight = sorted_gates[:, : self.moe_topk].contiguous()
        expert_index = sorted_idx[:, : self.moe_topk].contiguous()
        weight_sums = topk_weight.sum(dim=1, keepdim=True)
        weight_sums = torch.clamp(weight_sums, min=1e-8)
        topk_weight = topk_weight / weight_sums
        return topk_weight, expert_index


class HunyuanMoE(nn.Module):
    """DeepSeek-style MoE with an optional shared expert (mixed MLP MoE)."""

    def __init__(self, config: HunyuanImage3TrainingConfig):
        super().__init__()
        self.moe_topk = config.moe_topk
        self.num_experts = config.num_experts
        self.use_mixed_mlp_moe = config.use_mixed_mlp_moe
        if self.use_mixed_mlp_moe:
            self.shared_mlp = HunyuanMLP(config, is_shared_mlp=True)
        self.gate = HunyuanTopKGate(config)
        self.experts = nn.ModuleList([HunyuanMLP(config, is_moe=True) for _ in range(self.num_experts)])

    def forward(self, hidden_states: Tensor) -> Tensor:
        bsz, seq_len, hidden_size = hidden_states.shape
        input_hidden_states = hidden_states

        shared_out = self.shared_mlp(hidden_states) if self.use_mixed_mlp_moe else None

        # Gate weights are fp32; keep them out of autocast so routing logits stay numerically stable.
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            topk_weights, topk_idx = self.gate(hidden_states)
        topk_weights = topk_weights.to(hidden_states.dtype)

        flat_topk_idx = topk_idx.view(-1)
        hidden_states_flat = input_hidden_states.view(-1, hidden_size)
        hidden_states_repeated = hidden_states_flat.repeat_interleave(self.moe_topk, dim=0)

        expert_outputs = torch.zeros_like(hidden_states_repeated)
        for i in range(self.num_experts):
            expert_mask = flat_topk_idx == i
            if not expert_mask.any():
                continue
            expert_outputs[expert_mask] = self.experts[i](hidden_states_repeated[expert_mask])

        combined_output = (
            expert_outputs.view(bsz * seq_len, self.moe_topk, hidden_size) * topk_weights.unsqueeze(-1)
        ).sum(dim=1)
        combined_output = combined_output.to(hidden_states.dtype).view(bsz, seq_len, hidden_size)

        if shared_out is not None:
            combined_output = shared_out + combined_output
        return combined_output


class HunyuanImage3SDPAAttention(nn.Module):
    """SDPA self-attention with fused QKV, GQA, QK-norm, and 2-D RoPE."""

    def __init__(self, config: HunyuanImage3TrainingConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.attention_head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.hidden_size_q = self.head_dim * self.num_heads
        self.hidden_size_kv = self.head_dim * self.num_key_value_heads
        self.use_qk_norm = config.use_qk_norm
        self.use_rotary_pos_emb = config.use_rotary_pos_emb

        self.qkv_proj = nn.Linear(
            self.hidden_size, self.hidden_size_q + 2 * self.hidden_size_kv, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(self.hidden_size_q, self.hidden_size, bias=config.attention_bias)

        if self.use_qk_norm:
            self.query_layernorm = HunyuanRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_layernorm = HunyuanRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        custom_pos_emb: Optional[tuple[Tensor, Tensor]] = None,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> Tensor:
        bsz, q_len, _ = hidden_states.size()

        qkv_states = self.qkv_proj(hidden_states)
        qkv_states = qkv_states.reshape(
            bsz, q_len, self.num_key_value_heads, self.num_key_value_groups + 2, self.head_dim
        )
        query_states, key_states, value_states = torch.split(qkv_states, [self.num_key_value_groups, 1, 1], dim=3)

        query_states = query_states.reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if self.use_rotary_pos_emb:
            cos, sin = custom_pos_emb
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.use_qk_norm:
            query_states = self.query_layernorm(query_states)
            key_states = self.key_layernorm(key_states)

        query_states = query_states.to(value_states.dtype)
        key_states = key_states.to(value_states.dtype)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if cu_seqlens is not None:
            # rmpad path (AR text): bsz == 1, cu_seqlens marks sample boundaries.
            attn_output = _varlen_attention(
                query_states, key_states, value_states, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
            )
        else:
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=attention_mask, dropout_p=0.0
            )
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        return self.o_proj(attn_output)


def _varlen_attention(
    query_states: Tensor,
    key_states: Tensor,
    value_states: Tensor,
    cu_seqlens: Tensor,
    max_seqlen: Optional[int],
) -> Tensor:
    """Varlen causal self-attention over concatenated variable-length samples.

    Prefers ``flash_attn.flash_attn_varlen_func``; falls back to SDPA with a
    per-sample block-diagonal causal mask on CPU / without flash-attn. Inputs
    and output are ``(1, H, total_nnz, D)``.
    """
    is_cuda = query_states.is_cuda
    if is_cuda:
        try:
            from flash_attn import flash_attn_varlen_func
        except ImportError:
            flash_attn_varlen_func = None
    else:
        flash_attn_varlen_func = None

    if flash_attn_varlen_func is not None:
        # flash_attn_varlen_func expects (total_nnz, H, D).
        q = query_states.squeeze(0).transpose(0, 1).contiguous()
        k = key_states.squeeze(0).transpose(0, 1).contiguous()
        v = value_states.squeeze(0).transpose(0, 1).contiguous()
        cu = cu_seqlens.to(torch.int32)
        max_s = int(max_seqlen) if max_seqlen is not None else int((cu[1:] - cu[:-1]).max().item())
        out = flash_attn_varlen_func(
            q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=max_s, max_seqlen_k=max_s, causal=True
        )
        return out.transpose(0, 1).unsqueeze(0)

    total_nnz = query_states.shape[2]
    device = query_states.device
    mask = torch.zeros(total_nnz, total_nnz, dtype=torch.bool, device=device)
    cu_list = cu_seqlens.tolist()
    for start, end in zip(cu_list[:-1], cu_list[1:], strict=False):
        block = torch.ones(end - start, end - start, dtype=torch.bool, device=device).tril()
        mask[start:end, start:end] = block
    mask = mask.unsqueeze(0).unsqueeze(0)
    return F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=mask, dropout_p=0.0)


class HunyuanImage3DecoderLayer(nn.Module):
    """Single decoder layer: pre-norm attention + pre-norm MoE MLP."""

    def __init__(self, config: HunyuanImage3TrainingConfig):
        super().__init__()
        self.self_attn = HunyuanImage3SDPAAttention(config)
        self.mlp = HunyuanMoE(config) if config.num_experts > 1 else HunyuanMLP(config, is_moe=True)
        self.input_layernorm = HunyuanRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = HunyuanRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        custom_pos_emb: Optional[tuple[Tensor, Tensor]] = None,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            custom_pos_emb=custom_pos_emb,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class TimestepEmbedder(nn.Module):
    """Embeds scalar (or per-image) timesteps into vectors."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256, max_period: int = 10000):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, t: Tensor) -> Tensor:
        t_freq = timestep_embedding(t, self.frequency_embedding_size, self.max_period).type(self.mlp[0].weight.dtype)
        return self.mlp(t_freq)


class Upsample(nn.Module):
    """Forward module: nearest upsample with optional conv."""

    def __init__(self, channels: int, use_conv: bool, dims: int = 2, out_channels: Optional[int] = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """Downsample module: strided conv or average pool."""

    def __init__(self, channels: int, use_conv: bool, dims: int = 2, out_channels: Optional[int] = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            self.op = nn.AvgPool2d(kernel_size=stride, stride=stride)

    def forward(self, x: Tensor) -> Tensor:
        return self.op(x)


class ResBlock(nn.Module):
    """Residual block with adaptive group normalization (timestep-conditioned)."""

    def __init__(
        self,
        in_channels: int,
        emb_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        use_conv: bool = False,
        up: bool = False,
        down: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels

        self.in_layers = nn.Sequential(
            normalization(in_channels),
            nn.SiLU(),
            conv_nd(2, in_channels, self.out_channels, 3, padding=1),
        )
        self.updown = up or down
        if up:
            self.h_upd = Upsample(in_channels, False)
            self.x_upd = Upsample(in_channels, False)
        elif down:
            self.h_upd = Downsample(in_channels, False)
            self.x_upd = Downsample(in_channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, 2 * self.out_channels))
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(2, self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == in_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(2, in_channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(2, in_channels, self.out_channels, 1)

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = out_norm(h) * (1.0 + scale) + shift
        h = out_rest(h)
        return self.skip_connection(x) + h


class UNetDown(nn.Module):
    """Latent patchifier: VAE latent ``(B, C, H, W)`` -> ``(B, num_patches, D)``."""

    def __init__(self, patch_size: int, in_channels: int, emb_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.patch_size = patch_size
        assert patch_size in [1, 2, 4, 8]

        self.model: nn.ModuleList = nn.ModuleList([conv_nd(2, in_channels, hidden_channels, 3, padding=1)])
        if patch_size == 1:
            self.model.append(ResBlock(hidden_channels, emb_channels, out_channels=out_channels))
        else:
            for i in range(patch_size // 2):
                self.model.append(
                    ResBlock(
                        hidden_channels,
                        emb_channels,
                        out_channels=hidden_channels if (i + 1) * 2 != patch_size else out_channels,
                        down=True,
                    )
                )

    def forward(self, x: Tensor, t: Tensor) -> tuple[Tensor, int, int]:
        for module in self.model:
            x = module(x, t) if isinstance(module, ResBlock) else module(x)
        _, _, token_h, token_w = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1).transpose(1, 2)
        return x, token_h, token_w


class UNetUp(nn.Module):
    """Latent unpatchifier: ``(B, num_patches, D)`` -> VAE latent ``(B, C, H, W)``."""

    def __init__(
        self,
        patch_size: int,
        in_channels: int,
        emb_channels: int,
        hidden_channels: int,
        out_channels: int,
        out_norm: bool = False,
    ):
        super().__init__()
        self.patch_size = patch_size
        assert patch_size in [1, 2, 4, 8]

        self.model: nn.ModuleList = nn.ModuleList()
        if patch_size == 1:
            self.model.append(ResBlock(in_channels, emb_channels, out_channels=hidden_channels))
        else:
            for i in range(patch_size // 2):
                self.model.append(
                    ResBlock(
                        in_channels if i == 0 else hidden_channels,
                        emb_channels,
                        out_channels=hidden_channels,
                        up=True,
                    )
                )

        if out_norm:
            self.model.append(
                nn.Sequential(
                    normalization(hidden_channels),
                    nn.SiLU(),
                    conv_nd(2, hidden_channels, out_channels, 3, padding=1),
                )
            )
        else:
            self.model.append(conv_nd(2, hidden_channels, out_channels, 3, padding=1))

    def forward(self, x: Tensor, t: Tensor, token_h: int, token_w: int) -> Tensor:
        x = x.transpose(1, 2)  # (B, D, token_h * token_w)
        x = x.reshape(x.shape[0], x.shape[1], token_h, token_w)  # (B, D, token_h, token_w)
        for module in self.model:
            x = module(x, t) if isinstance(module, ResBlock) else module(x)
        return x


# ===================================================================
#  Main module: HunyuanImage3ForTraining
# ===================================================================


class HunyuanImage3ForTraining(NonDiffusersModelBase):
    """Training-side generation backbone for HunyuanImage3 (flow-matching).

    ``_no_split_modules`` enables layer-level FSDP2 sharding; the encoder-side
    VAE/ViT are rollout-only, but the non-tied ``lm_head`` is kept so the unified
    AR+DiT loss can compute AR token log-probs over the recaption segment.
    """

    _no_split_modules = ["HunyuanImage3DecoderLayer"]
    _supports_gradient_checkpointing = True

    def __init__(self, config: HunyuanImage3TrainingConfig):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([HunyuanImage3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = HunyuanRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Image generation pathway (img_proj_type == "unet").
        self.timestep_emb = TimestepEmbedder(config.hidden_size)
        self.time_embed = TimestepEmbedder(config.hidden_size)
        self.time_embed_2 = TimestepEmbedder(config.hidden_size)
        self.patch_embed = UNetDown(
            patch_size=config.patch_size,
            in_channels=config.latent_channels,
            emb_channels=config.hidden_size,
            hidden_channels=config.patch_embed_hidden_dim,
            out_channels=config.hidden_size,
        )
        self.final_layer = UNetUp(
            patch_size=config.patch_size,
            in_channels=config.hidden_size,
            emb_channels=config.hidden_size,
            hidden_channels=config.patch_embed_hidden_dim,
            out_channels=config.latent_channels,
            out_norm=True,
        )

    def forward(self, input_ids: Tensor, *, mode: str = "gen_image", **kwargs):
        """Dispatch entry: both paths must reach the transformer body through this
        method so FSDP2's root forward hook fires and all-gathers root-owned
        DTensor params (e.g. ``ln_f.weight``).

        Args:
            input_ids: gen-image ``(B, L)`` or text rmpad ``(1, total_nnz)``.
            mode: ``"gen_image"`` (default) or ``"text"``. Text requires the
                rmpad kwargs ``cu_seqlens`` (``(B+1,)`` int32) and ``max_seqlen``.
        """
        if mode == "text":
            if "cu_seqlens" not in kwargs or "max_seqlen" not in kwargs:
                raise ValueError("mode='text' requires 'cu_seqlens' and 'max_seqlen' kwargs (rmpad + varlen FA path)")
            return self._forward_text(input_ids, cu_seqlens=kwargs["cu_seqlens"], max_seqlen=kwargs["max_seqlen"])
        if mode == "gen_image":
            return self._forward_gen_image(input_ids, **kwargs)
        raise ValueError(f"Unknown forward mode: {mode!r}")

    def _forward_gen_image(
        self,
        input_ids: Tensor,
        images: Tensor,
        timesteps: Tensor,
        image_mask: Tensor,
        gen_timestep_scatter_index: Tensor,
        rope_image_info: list[list[tuple[slice, tuple[int, int]]]],
        attention_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> tuple[Tensor]:
        """Gen-image forward pass reproducing the vllm-omni ``forward_call`` path.

        Args:
            input_ids: ``(B, L)`` token sequence with ``<img>`` placeholders.
            images: ``(B, C, H, W)`` noisy latent at the current timestep.
            timesteps: ``(B,)`` flow-matching timestep.
            image_mask: ``(B, L)`` bool — True at image-token positions.
            gen_timestep_scatter_index: ``(B, k)`` positions of the ``<timestep>`` token.
            rope_image_info: per-sample list of ``(slice, (h, w))`` for 2-D RoPE.
            attention_mask: ``(B, 1, L, L)`` bool (built from rope_image_info if None).

        Returns:
            ``(velocity,)`` of shape ``(B, C, H, W)`` in latent space.
        """
        device = input_ids.device
        bsz, seqlen = input_ids.shape
        n_embd = self.config.hidden_size

        hidden_states = self.wte(input_ids)  # (B, L, D)

        # Pin RoPE cos/sin to activation dtype so ``use_reentrant=False`` recompute
        # sees the same saved dtypes as the initial forward. TODO: align with
        # vllm-omni which runs RoPE fully in fp32.
        cos, sin = build_batch_2d_rope(seqlen, self.config.attention_head_dim, rope_image_info, device)
        cos = cos.to(hidden_states.dtype)
        sin = sin.to(hidden_states.dtype)

        # Scatter noisy latent patches into the sequence.
        t_emb = self.time_embed(timesteps)
        image_seq, token_h, token_w = self.patch_embed(images, t_emb)  # (B, num_patches, D)
        index = torch.arange(seqlen, device=device).unsqueeze(0).repeat(bsz, 1)
        image_scatter_index = index.masked_select(image_mask.bool()).reshape(bsz, -1)
        hidden_states.scatter_(
            dim=1,
            index=image_scatter_index.unsqueeze(-1).repeat(1, 1, n_embd),
            src=image_seq,
        )

        # Scatter the <timestep> token.
        timestep_emb = self.timestep_emb(timesteps)  # (B, D)
        hidden_states.scatter_(
            dim=1,
            index=gen_timestep_scatter_index.unsqueeze(-1).repeat(1, 1, n_embd),
            src=timestep_emb.unsqueeze(1),
        )

        if attention_mask is None:
            attention_mask = self._build_attention_mask(rope_image_info, seqlen, device)

        for layer in self.layers:

            def _layer_fn(hs, *, _layer=layer, _mask=attention_mask, _cos=cos, _sin=sin):
                return _layer(hs, attention_mask=_mask, custom_pos_emb=(_cos, _sin))

            hidden_states = self._checkpointed_call(_layer_fn, hidden_states)

        # Unpatch the image-token outputs back to latent space.
        image_output = hidden_states.masked_select(image_mask.unsqueeze(-1).bool()).reshape(
            bsz, token_h * token_w, n_embd
        )
        velocity = self.final_layer(image_output, self.time_embed_2(timesteps), token_h, token_w)
        return (velocity,)

    def _forward_text(self, input_ids: Tensor, cu_seqlens: Tensor, max_seqlen: int) -> Tensor:
        """AR text forward with rmpad + varlen FA (upstream verl LM layout).

        Args:
            input_ids: ``(1, total_nnz)`` flat token ids, samples concatenated.
            cu_seqlens: ``(B+1,)`` int32 sample offsets into the flat sequence.
            max_seqlen: max sample length (drives RoPE precompute).

        Returns:
            ``logits`` of shape ``(1, total_nnz, vocab_size)``.
        """
        device = input_ids.device

        hidden_states = self.wte(input_ids)  # (1, total_nnz, D)

        # Per-token position ids from cu_seqlens: each sample starts fresh at 0.
        seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int64)
        position_ids = torch.cat(
            [torch.arange(int(L), device=device, dtype=torch.long) for L in seq_lens.tolist()]
        )  # (total_nnz,)

        # Text-only 1-D RoPE (image_infos=None): build once for max_seqlen, gather by position id.
        cos_full, sin_full = build_2d_rope(
            int(max_seqlen), self.config.attention_head_dim, image_infos=None, device=device
        )
        cos = cos_full[position_ids].unsqueeze(0).to(hidden_states.dtype)
        sin = sin_full[position_ids].unsqueeze(0).to(hidden_states.dtype)

        for layer in self.layers:

            def _layer_fn(hs, *, _layer=layer, _cos=cos, _sin=sin, _cu=cu_seqlens, _max=int(max_seqlen)):
                return _layer(hs, custom_pos_emb=(_cos, _sin), cu_seqlens=_cu, max_seqlen=_max)

            hidden_states = self._checkpointed_call(_layer_fn, hidden_states)

        hidden_states = self.ln_f(hidden_states)
        return self.lm_head(hidden_states)

    @staticmethod
    def _build_attention_mask(
        rope_image_info: list[list[tuple[slice, tuple[int, int]]]], seqlen: int, device: torch.device
    ) -> Tensor:
        """Build the causal-text / full-image 4-D attention mask from image spans."""
        bsz = len(rope_image_info)
        causal = torch.ones(seqlen, seqlen, dtype=torch.bool, device=device).tril()
        mask = causal.unsqueeze(0).unsqueeze(0).repeat(bsz, 1, 1, 1)
        for i, image_infos in enumerate(rope_image_info):
            for sec_slice, (h, w) in image_infos:
                start, stop = sec_slice.start, sec_slice.start + w * h
                mask[i, :, start:stop, start:stop] = True
        return mask

    @classmethod
    def from_pretrained(cls, model_path: str, torch_dtype: torch.dtype = torch.bfloat16) -> HunyuanImage3ForTraining:
        """Load the generation backbone from the sharded checkpoint.

        Args:
            model_path: Directory containing ``config.json`` and sharded ``.safetensors``.
            torch_dtype: Target dtype.

        Returns:
            ``HunyuanImage3ForTraining`` with the generation backbone loaded.

        The 32-shard backbone is ~150 GB in bf16, so only the FSDP source rank
        (global rank 0) materializes real weights and streams the shards in;
        every other rank builds on ``meta`` and receives weights during the
        FSDP2 wrap via ``fsdp2_load_full_state_dict``'s broadcast-from-rank-0.
        """
        from safetensors.torch import load_file
        from verl.utils.fsdp_utils import get_init_weight_context_manager

        config = HunyuanImage3TrainingConfig.from_model_path(model_path)
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))

        # Build in the target dtype (MoE gate self-promotes to fp32) so we never
        # transiently hold an fp32 copy of the ~80B-param MoE. Non-source ranks
        # build on ``meta``; the source rank builds on CPU.
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch_dtype)
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                init_context = get_init_weight_context_manager(use_meta_tensor=True)
            else:
                init_context = nullcontext
            with init_context():
                model = cls(config)
        finally:
            torch.set_default_dtype(default_dtype)

        # Non-source ranks build on ``meta``; weights arrive via the FSDP2 wrap broadcast.
        if next(model.parameters()).is_meta:
            return model

        missing: set[str] = set()
        unexpected: set[str] = set()
        with torch.no_grad():
            for shard in shard_files:
                shard_state = {
                    mapped: tensor
                    for key, tensor in load_file(os.path.join(model_path, shard)).items()
                    if (mapped := _map_checkpoint_key(key)) is not None
                }
                result = model.load_state_dict(shard_state, strict=False)
                missing.update(result.missing_keys)
                unexpected.update(result.unexpected_keys)
                del shard_state, result

        if missing:
            logger.warning("Missing keys when loading HunyuanImage3ForTraining: %d keys", len(missing))
        if unexpected:
            logger.warning("Unexpected keys when loading HunyuanImage3ForTraining: %d keys", len(unexpected))
        return model


_SKIP_PREFIXES = ("vae.", "vision_model.", "vision_aligner.")


def _map_checkpoint_key(key: str) -> Optional[str]:
    """Map a checkpoint key to a ``HunyuanImage3ForTraining`` parameter name.

    Returns ``None`` for rollout-only modules (VAE, vision encoder).
    """
    if any(key.startswith(prefix) for prefix in _SKIP_PREFIXES):
        return None
    if key.startswith("model.wte."):
        return key[len("model.") :]
    if key.startswith("model.layers."):
        return key[len("model.") :]
    if key.startswith("model.ln_f."):
        return key[len("model.") :]
    return key
