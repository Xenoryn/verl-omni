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

"""HunyuanImage3 training-side adapter for FlowGRPO.

Registered as ``HunyuanImage3ForCausalMM`` in the DiffusionModelBase registry.
HunyuanImage3 is a unified flow-matching MoE transformer: the noisy VAE latent
is patched into the sequence next to the text tokens, the ``<timestep>`` is a
scalar token, and the velocity is unpatchified back to latent space.  CFG is
performed by batch doubling (conditional vs. empty prompt), matching vllm-omni.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import torch
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorStack
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_name

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.workers.config import DiffusionModelConfig

from .common import (
    HUNYUAN_IMAGE3_AR_SYS_TYPE,
    HUNYUAN_IMAGE3_FLOWGRPO_CFG_DEFAULTS,
    apply_hunyuan_cfg,
    build_hunyuan_image3_scheduler,
    messages_to_text,
    normalize_ar_cot_text,
    setup_hunyuan_image3_sigmas,
)
from .hunyuan_image3_model import HunyuanImage3ForTraining

logger = logging.getLogger(__name__)


def _token_ids_to_batch(token_ids_field, device: torch.device, pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a ragged ``prompt_token_ids`` field into ``(ids, mask)`` of shape ``(B, L)``.

    Accepts a ``NonTensorStack`` (the dataset column), a ``torch.Tensor`` (1-D or
    2-D), or a plain list of token-id sequences.
    """
    if isinstance(token_ids_field, NonTensorStack):
        token_ids_field = [tu.unwrap_non_tensor_data(token_ids_field[i]) for i in range(token_ids_field.shape[0])]
    elif isinstance(token_ids_field, torch.Tensor):
        if token_ids_field.ndim == 1:
            token_ids_field = [token_ids_field.tolist()]
        else:
            token_ids_field = token_ids_field.tolist()
    else:
        token_ids_field = list(token_ids_field)

    batch = len(token_ids_field)
    max_len = max(len(ids) for ids in token_ids_field)
    ids = torch.full((batch, max_len), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros(batch, max_len, dtype=torch.bool, device=device)
    for i, seq in enumerate(token_ids_field):
        n = len(seq)
        ids[i, :n] = torch.as_tensor(seq, dtype=torch.long, device=device)
        mask[i, :n] = True
    return ids, mask


def _extract_image_mask(token_ids: torch.Tensor, image_token_id: int) -> torch.Tensor:
    """Derive the image-token mask from ``<img>`` placeholder positions."""
    return token_ids == image_token_id


def _scatter_index_to_tensor(field, device: torch.device) -> torch.Tensor:
    """Batch a per-sample ``<timestep>`` position into a ``(B, 1)`` tensor."""
    if isinstance(field, torch.Tensor):
        value = field.to(device)
    elif isinstance(field, NonTensorStack):
        rows = [tu.unwrap_non_tensor_data(field[i]) for i in range(field.shape[0])]
        value = torch.as_tensor(rows, dtype=torch.long, device=device).reshape(-1, 1)
    else:
        value = torch.as_tensor(list(field), dtype=torch.long, device=device).reshape(-1, 1)
    return value.reshape(-1, 1)


def _rope_info_to_slices(field) -> list[list[tuple[slice, tuple[int, int]]]]:
    """Rebuild ``list[list[(slice, (h, w))]]`` from serialized ``[start, stop, h, w]`` rows.

    Accepts a tensor ``(B, 4)`` (or a ``NonTensorStack`` / nested Python list) of
    single-span rows produced by the data preprocessor; the actor reconstructs the
    ``(slice, (h, w))`` the rollout records.
    """
    if isinstance(field, torch.Tensor):
        rows = field.detach().cpu().tolist()
    elif isinstance(field, NonTensorStack):
        rows = []
        for i in range(field.shape[0]):
            element = tu.unwrap_non_tensor_data(field[i])
            rows.append(element.tolist() if isinstance(element, torch.Tensor) else list(element))
    else:
        rows = list(field)

    rope_info: list[list[tuple[slice, tuple[int, int]]]] = []
    for sample in rows:
        start, stop, h, w = int(sample[0]), int(sample[1]), int(sample[2]), int(sample[3])
        rope_info.append([(slice(start, stop), (h, w))])
    return rope_info


_FUSED_TOKENIZER_CTX_CACHE: dict[str, dict[str, Any]] = {}


def _get_hunyuan_image3_tokenizer_ctx(model_path: str) -> dict[str, Any]:
    """Load the vllm-omni tokenizer context once per worker.

    The DiT train forward rebuilds the rollout text prefix through the same
    ``TokenizerWrapper`` the engine uses, so template decisions (sequence
    template, system prompt body, cot re-parse) match the rollout by construction.
    """
    ctx = _FUSED_TOKENIZER_CTX_CACHE.get(model_path)
    if ctx is not None:
        return ctx

    import transformers
    from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_tokenizer import TokenizerWrapper
    from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
        HunyuanImage3ImageProcessor,
    )
    from vllm_omni.diffusion.models.hunyuan_image3.system_prompt import get_system_prompt

    config = transformers.AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    gen_cfg = transformers.GenerationConfig.from_pretrained(model_path)
    system_prompt = get_system_prompt(HUNYUAN_IMAGE3_AR_SYS_TYPE, "image")
    ctx = {
        "tkw": TokenizerWrapper(transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)),
        "image_processor": HunyuanImage3ImageProcessor(config),
        "drop_think": bool(getattr(gen_cfg, "drop_think", False)),
        "image_base_size": getattr(config, "image_base_size", None),
        "sequence_template": getattr(gen_cfg, "sequence_template", "pretrain"),
        "system_prompt": system_prompt.strip() if system_prompt else None,
    }
    _FUSED_TOKENIZER_CTX_CACHE[model_path] = ctx
    return ctx


def propagate_image_rewards_to_ar(
    image_rewards: torch.Tensor,
    ar_traj_group_id: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool per-image rewards up to their parent AR trajectory.

    The unified AR + DiT rollout expands one prompt into N AR trajectories and
    each AR trajectory into M images. The AR reward is the mean of its
    downstream image rewards; this helper implements that group-by-mean.
    Consumed by :meth:`HunyuanImage3FlowGRPO.postprocess_advantage`.

    Args:
        image_rewards: ``(N_img,)`` scalar outcome reward per image.
        ar_traj_group_id: ``(N_img,)`` integer id of the AR trajectory each
            image descends from. Ids need not be contiguous or sorted.

    Returns:
        ``(N_ar,)`` mean image reward per AR trajectory, indexed by the sorted
        unique values of ``ar_traj_group_id``.
    """
    if image_rewards.numel() == 0:
        return torch.empty(0, dtype=image_rewards.dtype, device=image_rewards.device)

    ids, inverse = torch.unique(ar_traj_group_id.to(image_rewards.device), sorted=True, return_inverse=True)
    sums = torch.zeros(ids.numel(), dtype=image_rewards.dtype, device=image_rewards.device).scatter_add(
        0, inverse, image_rewards
    )
    counts = torch.zeros(ids.numel(), dtype=image_rewards.dtype, device=image_rewards.device).scatter_add(
        0, inverse, torch.ones(inverse.numel(), dtype=image_rewards.dtype, device=image_rewards.device)
    )
    return sums / counts.clamp_min(1)


@DiffusionModelBase.register("HunyuanImage3ForCausalMM", algorithm="flow_grpo")
class HunyuanImage3FlowGRPO(DiffusionModelBase):
    """DiffusionModelBase wrapper for ``HunyuanImage3ForTraining``."""

    @classmethod
    def build_module(cls, model_config: DiffusionModelConfig, torch_dtype: torch.dtype):
        logger.info("Loading HunyuanImage3ForTraining from %s", model_config.local_path)
        return HunyuanImage3ForTraining.from_pretrained(model_config.local_path, torch_dtype=torch_dtype)

    @classmethod
    def postprocess_advantage(cls, data, *, adv_kwargs: dict, norm_adv_by_std_in_grpo: bool):
        """Attach ``ar_advantages`` for the recaption segment (unified AR + DiT).

        The scalar image reward is propagated up to its parent AR trajectory
        (mean-pool) and then GRPO-normalized per original prompt. In the
        current flat 1:1 rollout (one image per AR trajectory) the propagation
        is identity; a two-level rollout would key on the real
        ``ar_traj_group_id`` column instead.
        """
        if "ar_prompt_token_ids" not in data.non_tensor_batch or "sample_level_scores" not in data.batch:
            return data

        bsz = data.batch["sample_level_scores"].shape[0]
        prompt_group_id = adv_kwargs.get("index", np.arange(bsz))
        ar_traj_group_id = torch.arange(bsz)
        ar_rewards = propagate_image_rewards_to_ar(data.batch["sample_level_scores"].squeeze(-1), ar_traj_group_id)
        ar_adv, _ = compute_grpo_outcome_advantage(
            token_level_rewards=ar_rewards.unsqueeze(-1),
            response_mask=torch.ones_like(ar_rewards).unsqueeze(-1),
            index=prompt_group_id,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["ar_advantages"] = ar_adv.squeeze(-1)
        return data

    @classmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig):
        scheduler = build_hunyuan_image3_scheduler()
        cls.set_timesteps(scheduler, model_config, get_device_name())
        return scheduler

    @classmethod
    def set_timesteps(cls, scheduler: FlowMatchSDEDiscreteScheduler, model_config: DiffusionModelConfig, device: str):
        setup_hunyuan_image3_sigmas(
            scheduler,
            model_config.pipeline.num_inference_steps,
            device=device,
        )

    @staticmethod
    def _build_fused_dit_inputs(
        model_config: DiffusionModelConfig,
        config: Any,
        micro_batch: TensorDict,
        latents: torch.Tensor,
        step: int,
        device: torch.device,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]]:
        """Rebuild the rollout DiT text prefix from the AR rollout fields.

        The engine DiT conditions on the official instruct template with the AR
        recaption spliced in (``TokenizeWrapper.apply_chat_template``); replaying
        the cot-free preprocessor layout would compute the flow log-prob under a
        different text condition than the rollout sampled.  Returns ``None`` when
        the AR fields are absent, letting the caller use the precomputed columns.
        """
        ar_prompt_field = micro_batch.get("ar_prompt_token_ids")
        cot_field = micro_batch.get("ar_generated_text")
        raw_prompt_field = micro_batch.get("raw_prompt")
        if ar_prompt_field is None or cot_field is None or raw_prompt_field is None:
            return None

        from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
            HunyuanImage3Pipeline,
        )

        ctx = _get_hunyuan_image3_tokenizer_ctx(model_config.local_path)
        batch_size = int(cot_field.shape[0])
        captions = [messages_to_text(tu.unwrap_non_tensor_data(raw_prompt_field[i])) for i in range(batch_size)]
        cot_texts = [normalize_ar_cot_text(tu.unwrap_non_tensor_data(cot_field[i])) for i in range(batch_size)]
        if any(text is None for text in cot_texts):
            return None

        ffactor_h, ffactor_w = tuple(config.vae_downsample_factor)
        image_latents = latents[:, step]
        height = image_latents.shape[-2] * ffactor_h
        width = image_latents.shape[-1] * ffactor_w
        image_info = ctx["image_processor"].build_image_info((height, width))

        out = ctx["tkw"].apply_chat_template(
            batch_prompt=captions,
            batch_cot_text=cot_texts,
            batch_system_prompt=[ctx["system_prompt"]] * batch_size,
            mode="gen_image",
            batch_gen_image_info=[image_info] * batch_size,
            bot_task="auto",
            image_base_size=ctx["image_base_size"],
            sequence_template=ctx["sequence_template"],
            cfg_factor=1,
            drop_think=ctx["drop_think"],
        )
        output, sections = out["output"], out["sections"]

        positive_ids = output.tokens.to(device)
        image_mask = output.gen_image_mask.to(device)
        gen_timestep_scatter_index = output.gen_timestep_scatter_index.to(device)
        rope_image_info = HunyuanImage3Pipeline.build_batch_rope_image_info(output, sections)
        return positive_ids, image_mask, gen_timestep_scatter_index, rope_image_info

    @classmethod
    def prepare_model_inputs(
        cls,
        module: HunyuanImage3ForTraining,
        model_config: DiffusionModelConfig,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        negative_prompt_embeds_mask: torch.Tensor,
        micro_batch: TensorDict,
        step: int,
    ) -> tuple[dict, Optional[dict]]:
        """Build positive/negative model-input dicts.

        When the batch carries the AR rollout fields, the positive branch is
        rebuilt from the rollout text prefix (see :meth:`_build_fused_dit_inputs`);
        otherwise it falls back to the precomputed ``[text, <boi>, <size>,
        <ratio>, <timestep>, <eoi>, <img>*]`` columns.  The negative branch always
        uses the precomputed columns.
        """
        device = latents.device
        config = module.config

        fused = cls._build_fused_dit_inputs(model_config, config, micro_batch, latents, step, device)
        if fused is not None:
            positive_ids, image_mask, gen_timestep_scatter_index, rope_image_info = fused
        else:
            positive_ids, _ = _token_ids_to_batch(micro_batch["prompt_token_ids"], device, config.pad_token_id)
            image_mask = _extract_image_mask(positive_ids, config.image_token_id)
            gen_timestep_scatter_index = _scatter_index_to_tensor(micro_batch.get("gen_timestep_scatter_index"), device)
            rope_image_info = _rope_info_to_slices(micro_batch.get("rope_image_info"))

        negative_ids, _ = _token_ids_to_batch(micro_batch["negative_prompt_token_ids"], device, config.pad_token_id)
        negative_image_mask = _extract_image_mask(negative_ids, config.image_token_id)

        # The <timestep> token position and 2-D RoPE spans differ between the
        # conditional and empty-prompt branches (their text lengths differ), so
        # the data preprocessor emits them separately.
        negative_gen_timestep_scatter_index = _scatter_index_to_tensor(
            micro_batch.get("negative_gen_timestep_scatter_index"), device
        )
        negative_rope_image_info = _rope_info_to_slices(micro_batch.get("negative_rope_image_info"))

        image = latents[:, step]
        timestep = timesteps[:, step]

        model_inputs = {
            "input_ids": positive_ids,
            "images": image,
            "timesteps": timestep,
            "image_mask": image_mask,
            "gen_timestep_scatter_index": gen_timestep_scatter_index,
            "rope_image_info": rope_image_info,
        }
        negative_model_inputs = {
            "input_ids": negative_ids,
            "images": image,
            "timesteps": timestep,
            "image_mask": negative_image_mask,
            "gen_timestep_scatter_index": negative_gen_timestep_scatter_index,
            "rope_image_info": negative_rope_image_info,
        }
        return model_inputs, negative_model_inputs

    @classmethod
    def forward_and_sample_previous_step(
        cls,
        module: HunyuanImage3ForTraining,
        scheduler: FlowMatchSDEDiscreteScheduler,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]],
        scheduler_inputs: Optional[TensorDict | dict[str, torch.Tensor]],
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert scheduler_inputs is not None
        latents = scheduler_inputs["all_latents"]
        timesteps = scheduler_inputs["all_timesteps"]

        noise_pred = cls.forward(module, model_config, model_inputs)
        guidance_scale = float(
            getattr(model_config.pipeline, "guidance_scale", HUNYUAN_IMAGE3_FLOWGRPO_CFG_DEFAULTS["guidance_scale"])
        )
        guidance_rescale = float(
            getattr(
                model_config.pipeline,
                "guidance_rescale",
                HUNYUAN_IMAGE3_FLOWGRPO_CFG_DEFAULTS["guidance_rescale"],
            )
        )
        if guidance_scale > 1.0:
            assert negative_model_inputs is not None, "HunyuanImage3 CFG requires the empty-prompt branch"
            negative_noise_pred = cls.forward(module, model_config, negative_model_inputs)
            noise_pred = apply_hunyuan_cfg(noise_pred, negative_noise_pred, guidance_scale, guidance_rescale)

        _, log_prob, prev_sample_mean, std_dev_t, sqrt_dt = scheduler.sample_previous_step(
            sample=latents[:, step].float(),
            model_output=noise_pred.float(),
            timestep=timesteps[:, step],
            noise_level=model_config.algo.noise_level,
            prev_sample=latents[:, step + 1].float(),
            sde_type=model_config.algo.sde_type,
            return_logprobs=True,
            return_sqrt_dt=True,
            include_logprob_normalizer=False,
        )
        return log_prob, prev_sample_mean, std_dev_t, sqrt_dt
