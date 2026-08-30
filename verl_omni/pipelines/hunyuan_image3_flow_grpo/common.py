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

"""Shared utilities for HunyuanImage3 FlowGRPO adapters.

Training and rollout must construct the identical scheduler and call the same
``set_timesteps`` so their sigma schedules match to floating-point precision --
otherwise the importance-sampling ratio is biased. Values below mirror
vllm-omni's ``HunyuanImage3Pipeline``.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

HUNYUAN_IMAGE3_TIMESTEP_SHIFT = 3.0

# Official HunyuanImage-3.0-Instruct AR prompt structure (vllm-omni
# ``prompt_utils`` builder): ``t2i_think`` -> ``en_unified`` system prompt +
# ``Assistant:  thinking`` trigger.  UniRL drives the same task/sys pair.
HUNYUAN_IMAGE3_AR_TASK = "t2i_think"
HUNYUAN_IMAGE3_AR_SYS_TYPE = "en_unified"

HUNYUAN_IMAGE3_SCHEDULER_KWARGS: dict[str, float | int | bool] = {
    "num_train_timesteps": 1000,
    "shift": HUNYUAN_IMAGE3_TIMESTEP_SHIFT,
    "use_dynamic_shifting": False,
    "base_shift": 0.5,
    "max_shift": 1.15,
    "time_shift_type": "exponential",
    "stochastic_sampling": False,
}

# ``guidance_scale`` mirrors HunyuanImage3's ``diff_guidance_scale``; override in
# the launch script if the rollout uses a different value.
HUNYUAN_IMAGE3_FLOWGRPO_CFG_DEFAULTS = {
    "guidance_scale": 2.5,
    "guidance_rescale": 0.0,
}


def maybe_to_cpu(value):
    """Move a single value to CPU if it is a ``torch.Tensor``; else return unchanged."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def messages_to_text(messages: Any) -> str:
    """Extract plain text content without applying a chat template."""
    if isinstance(messages, str):
        return messages
    if isinstance(messages, dict):
        messages = [messages]

    parts = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            parts.append(content)
            continue
        for item in content or []:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
    return "\n".join(part for part in parts if part).strip()


def normalize_ar_cot_text(cot: Optional[str]) -> Optional[str]:
    """Prepend the trigger tag the AR generation omitted (mirrors the engine's
    ``HunyuanImage3Pipeline._normalize_cot_text`` applied before the DiT template).
    """
    if not cot:
        return cot
    if " response" in cot and not cot.startswith(" thinking"):
        return " thinking" + cot
    if "</recaption>" in cot and not cot.startswith("<recaption>"):
        return "<recaption>" + cot
    return cot


def build_hunyuan_image3_scheduler() -> FlowMatchSDEDiscreteScheduler:
    """Construct the SDE scheduler with vllm-omni's exact HunyuanImage3 config."""
    return FlowMatchSDEDiscreteScheduler(**HUNYUAN_IMAGE3_SCHEDULER_KWARGS)


def setup_hunyuan_image3_sigmas(
    scheduler: FlowMatchSDEDiscreteScheduler,
    num_steps: int,
    device: str | None = None,
) -> list[float]:
    """Configure the scheduler with vllm-omni's sigmas and return them.

    ``scheduler`` must be built by :func:`build_hunyuan_image3_scheduler` first
    (``shift`` is a construction-time knob). The returned list is the N denoise
    sigmas, terminal 0 excluded.
    """
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")

    if device is not None:
        scheduler.set_timesteps(num_inference_steps=num_steps, device=device)
    else:
        scheduler.set_timesteps(num_inference_steps=num_steps)
    scheduler.set_begin_index(0)
    return scheduler.timesteps.tolist()


def apply_hunyuan_cfg(
    noise_pred: torch.Tensor,
    negative_noise_pred: torch.Tensor,
    guidance_scale: float,
    guidance_rescale: float = 0.0,
) -> torch.Tensor:
    """Standard CFG combine: ``v_uncond + scale * (v_cond - v_uncond)`` with optional rescale."""
    noise_pred_text = noise_pred
    noise_pred_uncond = negative_noise_pred
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    if guidance_rescale > 0.0:
        norm_dims = tuple(range(1, noise_pred.ndim))
        noise_pred = noise_pred * (
            noise_pred_uncond.norm(dim=norm_dims, keepdim=True) / noise_pred.norm(dim=norm_dims, keepdim=True)
        )
        noise_pred = guidance_rescale * noise_pred + (1 - guidance_rescale) * noise_pred_uncond
    return noise_pred
