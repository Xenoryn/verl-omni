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

"""HunyuanImage3 rollout-side adapter for FlowGRPO.

HunyuanImage3 runs vllm-omni's step-execution loop (``prepare_encode`` ->
``denoise_step`` -> ``step_scheduler`` -> ``post_decode``). This adapter swaps
in the SDE scheduler and records the per-step trajectory at the
``step_scheduler`` hook, like ``qwen_image_flow_grpo``'s rollout adapter;
``denoise_step`` already applies CFG via ``state.extra[_STEP_GUIDANCE_SCALE]``.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

import torch
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
    _STEP_GENERATOR,
    HunyuanImage3Pipeline,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from verl_omni.pipelines.model_base import VllmOmniPipelineBase

from .common import HUNYUAN_IMAGE3_FLOWGRPO_CFG_DEFAULTS, build_hunyuan_image3_scheduler

logger = logging.getLogger(__name__)

__all__ = ["HunyuanImage3PipelineWithLogProb"]


@VllmOmniPipelineBase.register("HunyuanImage3ForCausalMM", algorithm="flow_grpo")
class HunyuanImage3PipelineWithLogProb(HunyuanImage3Pipeline):
    """HunyuanImage3 pipeline variant that records per-step SDE log-probs."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        del prefix  # HunyuanImage3Pipeline.__init__ takes only od_config
        super().__init__(od_config=od_config)
        self.scheduler = build_hunyuan_image3_scheduler()

        # Pre-build the inner diffusers pipeline with our SDE scheduler so the base
        # ``pipeline`` property's lazy path never overwrites ``self.scheduler``.
        from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
            HunyuanImage3Text2ImagePipeline,
        )

        self._pipeline = HunyuanImage3Text2ImagePipeline(model=self, scheduler=self.scheduler, vae=self.vae)
        logger.info("HunyuanImage3PipelineWithLogProb: SDE scheduler enabled for RL rollouts")

    def map_lora_update_to_engine(
        self, tensors: dict[str, torch.Tensor], peft_config: dict
    ) -> tuple[dict[str, torch.Tensor], dict]:
        """Split fused ``self_attn.qkv_proj`` LoRA into per-slice sub-loras.

        Actor-side PEFT injects LoRA on the fused ``qkv_proj`` linear, so the
        pushed ``lora_B`` is ``[q+k+v, r]`` = ``[6144, r]`` for the reference
        HunyuanImage-3.0 (Q=4096, K=V=1024). vllm-omni's LoRA manager splits
        that tensor by the per-TP-shard ``output_slices`` sum
        (``manager.py:610``), which is ``1536`` at ``tensor_parallel_size=4``;
        the shape check rejects it. Emit three sub-loras keyed on
        ``q_proj``/``k_proj``/``v_proj`` with a shared ``lora_A`` and a
        per-slice ``lora_B``; the manager's sub-lora fallback path TP-shards
        each slice correctly.

        TODO: drop once vllm-omni's ``DiffusionLoRAManager`` splits packed
        LoRA-B by ``base_layer.output_sizes`` instead of ``output_slices``.
        """
        transformer = getattr(self, "transformer", None)
        first_qkv = None
        if transformer is not None:
            for module in transformer.modules():
                if type(module).__name__ == "QKVParallelLinear" and hasattr(module, "output_sizes"):
                    first_qkv = module
                    break
        if first_qkv is None:
            return tensors, peft_config

        q_full, k_full, v_full = first_qkv.output_sizes
        expected_full = q_full + k_full + v_full

        mapped: dict[str, torch.Tensor] = {}
        for name, tensor in tensors.items():
            is_lora_a = name.endswith(".lora_A.weight")
            is_lora_b = name.endswith(".lora_B.weight")
            if not (is_lora_a or is_lora_b):
                mapped[name] = tensor
                continue
            suffix = ".lora_A.weight" if is_lora_a else ".lora_B.weight"
            module = name[: -len(suffix)]
            if not module.endswith(".qkv_proj"):
                mapped[name] = tensor
                continue

            prefix = module[: -len(".qkv_proj")]
            if is_lora_a:
                for sub in ("q_proj", "k_proj", "v_proj"):
                    mapped[f"{prefix}.{sub}{suffix}"] = tensor
                continue

            # lora_B: split ``[Q+K+V, r]`` into three per-slice tensors.
            if tensor.shape[0] != expected_full:
                logger.warning(
                    "map_lora_update_to_engine: unexpected fused qkv_proj lora_B "
                    "shape %s for %s (expected first-dim=%d); passing through unchanged",
                    tuple(tensor.shape),
                    name,
                    expected_full,
                )
                mapped[name] = tensor
                continue
            q_b, k_b, v_b = torch.split(tensor, [q_full, k_full, v_full], dim=0)
            mapped[f"{prefix}.q_proj{suffix}"] = q_b.contiguous()
            mapped[f"{prefix}.k_proj{suffix}"] = k_b.contiguous()
            mapped[f"{prefix}.v_proj{suffix}"] = v_b.contiguous()

        new_config = dict(peft_config) if peft_config is not None else {}
        target_modules = new_config.get("target_modules")
        if isinstance(target_modules, list | tuple) and any("qkv_proj" in t for t in target_modules):
            expanded = []
            for t in target_modules:
                if isinstance(t, str) and "qkv_proj" in t:
                    for sub in ("q_proj", "k_proj", "v_proj"):
                        expanded.append(t.replace("qkv_proj", sub))
                else:
                    expanded.append(t)
            seen = set()
            deduped = []
            for t in expanded:
                if t not in seen:
                    seen.add(t)
                    deduped.append(t)
            new_config["target_modules"] = deduped
        return mapped, new_config

    def prepare_encode(self, state, **kwargs: Any):
        """Populate rollout SDE state after the base ``prepare_encode``.

        The base method sets ``state.timesteps`` via ``retrieve_timesteps(self.scheduler, N)``
        (already SD3-shifted because ``self.scheduler`` carries HunyuanImage3's
        ``shift=3.0`` config) plus ``state.latents`` / ``state.scheduler`` / the
        ``_STEP_*`` payloads.  Here we resolve the SDE knobs from the sampling
        ``extra_args`` and initialise the trajectory buffers consumed by
        :meth:`step_scheduler` and :meth:`post_decode`.
        """
        sampling = state.sampling
        # HunyuanImage3's diff_guidance_scale is 2.5; the vllm-omni base hardcodes 5.0.
        if not getattr(sampling, "guidance_scale_provided", False):
            sampling.guidance_scale = HUNYUAN_IMAGE3_FLOWGRPO_CFG_DEFAULTS["guidance_scale"]
            sampling.guidance_scale_provided = True

        state = super().prepare_encode(state, **kwargs)

        # Base ``prepare_encode`` seeds latents in bf16 but ``step_scheduler`` keeps
        # them in fp32; promote up front so mixed-request batches share one dtype.
        state.latents = state.latents.to(torch.float32)

        num_timesteps = state.total_steps
        extra = sampling.extra_args or {}
        noise_level = float(extra.get("noise_level", 1.0))
        sde_type = extra.get("sde_type", "sde")
        logprobs = bool(extra.get("logprobs", True))
        sde_window_size = extra.get("sde_window_size", None)

        if sde_window_size is not None:
            size = int(sde_window_size)
            window_range = extra.get("sde_window_range", (0, num_timesteps))
            low, high = int(window_range[0]), int(window_range[1])
            if high - low < size:
                # Window does not fit; clamp to the lowest valid begin.
                begin = low
            else:
                generator = state.extra.get(_STEP_GENERATOR)
                begin = int(
                    torch.randint(
                        low,
                        high - size + 1,
                        (1,),
                        generator=generator,
                        device=state.latents.device,
                    ).item()
                )
            sde_window = (begin, begin + size)
        else:
            # Full trajectory except the final deterministic (ODE) step.
            sde_window = (0, max(num_timesteps - 1, 0))

        state.sde_window = sde_window
        state.noise_level = noise_level
        state.sde_type = sde_type
        state.logprobs = logprobs
        state.all_latents = []
        state.all_log_probs = []
        state.all_timesteps = []
        return state

    def step_scheduler(self, state, noise_pred: torch.Tensor, **kwargs: Any) -> None:
        """One SDE denoise step, recording the trajectory inside the SDE window."""
        del kwargs
        if getattr(self, "interrupt", False):
            return

        i = state.step_index
        timestep_value = state.current_timestep
        begin, end = state.sde_window
        in_window = begin <= i < end
        cur_noise_level = state.noise_level if in_window else 0.0
        cur_logprobs = state.logprobs and in_window

        if i == begin:
            state.all_latents.append(state.latents.to(torch.float32))

        new_latents, log_prob, _, _ = state.scheduler.step(
            noise_pred.to(torch.float32),
            timestep_value,
            state.latents.to(torch.float32),
            generator=state.extra.get(_STEP_GENERATOR),
            noise_level=cur_noise_level,
            sde_type=state.sde_type,
            return_logprobs=cur_logprobs,
            return_dict=False,
        )

        if in_window:
            state.all_latents.append(new_latents.to(torch.float32))
            state.all_log_probs.append(None if log_prob is None else log_prob)
            state.all_timesteps.append(timestep_value)

        # Keep live state in fp32 throughout so freshly-added requests and
        # stepped requests share one dtype in the engine batch.
        state.latents = new_latents.to(torch.float32)
        state.step_index += 1

    def post_decode(self, state, **kwargs: Any) -> DiffusionOutput:
        """Decode final latents and package the recorded rollout trajectory."""
        output = super().post_decode(state, **kwargs)
        if not isinstance(output, DiffusionOutput):
            return output

        all_latents = state.all_latents
        all_log_probs = state.all_log_probs
        all_timesteps = state.all_timesteps

        # A complete trajectory needs at least one recorded denoise step; the
        # engine's dummy/warmup run can leave the SDE window empty.
        if not all_latents or not all_timesteps:
            return output

        stacked_latents = torch.stack(all_latents, dim=1)
        stacked_log_probs = (
            torch.stack(all_log_probs, dim=1) if all_log_probs and all_log_probs[0] is not None else None
        )
        stacked_timesteps = torch.stack(all_timesteps).unsqueeze(0).expand(state.latents.shape[0], -1)

        # HunyuanImage3 has no ``get_<model>_post_process_func`` in vllm-omni, so
        # attach the trajectory fields directly on ``output`` (whose payload is
        # a bare PIL image reaching ``_process_output`` unchanged).
        return dataclasses.replace(
            output,
            trajectory_latents=stacked_latents,
            trajectory_log_probs=stacked_log_probs,
            trajectory_timesteps=stacked_timesteps,
            to_cpu=True,
        )

    def forward(self, req: OmniDiffusionRequest | DiffusionRequestBatch) -> DiffusionOutput:
        """Monolithic fallback (warmup/dummy runs); step-execution is the RL path."""
        return super().forward(req)
