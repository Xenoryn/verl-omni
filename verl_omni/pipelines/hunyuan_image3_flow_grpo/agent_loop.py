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

"""HunyuanImage3 agent loop that builds the official AR think/recaption prompt.

The HF HunyuanImage-3.0-Instruct checkpoint ships no Jinja chat template, so a
generic ``apply_chat_template`` cannot render its official prompt structure
(``<|startoftext|>`` + ``en_unified`` system prompt + ``User: <caption>`` +
``Assistant:  thinking`` trigger).  This loop builds it through vllm-omni's
``prompt_utils`` builder and forwards the raw caption plus ``use_system_prompt``
to the engine so the DiT stage rebuilds the identical prefix.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import register
from verl.utils.profiler import simple_timer
from verl.utils.ray_utils import get_event_loop
from verl.utils.tokenizer import normalize_token_ids

from verl_omni.agent_loop.diffusion_agent_loop import DiffusionAgentLoopOutput
from verl_omni.agent_loop.single_turn_agent_loop import DiffusionSingleTurnAgentLoop

from .common import HUNYUAN_IMAGE3_AR_SYS_TYPE, HUNYUAN_IMAGE3_AR_TASK, messages_to_text


@register("hunyuan_image3_diffusion_single_turn_agent")
class HunyuanImage3DiffusionSingleTurnAgentLoop(DiffusionSingleTurnAgentLoop):
    """hi3 two-stage loop: official AR prompt ids (client-built) -> engine."""

    def __init__(
        self,
        trainer_config,
        server_manager,
        tokenizer,
        processor,
        dataset_cls,
        data_config,
        extra_tokenizer_map: dict[str, dict[str, Any]] | None = None,
        **kwargs,
    ) -> None:
        # hi3 has no chat template to probe; skip AgentLoopBase.__init__ the same
        # way the ltx2 / minimax pipelines do.
        del kwargs
        self.config = trainer_config.config
        self.rollout_config = self.config.actor_rollout_ref.rollout
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.processor = processor
        self.dataset_cls = dataset_cls
        self.data_config = data_config.config
        self.apply_chat_template_kwargs = self.data_config.get("apply_chat_template_kwargs", {})
        self.mm_processor_kwargs = self.data_config.get("mm_processor_kwargs", {})
        self.extra_tokenizer_map = extra_tokenizer_map or {}
        self.system_prompt = []
        self.loop = get_event_loop()

    async def _official_prompt_ids(self, text: str) -> list[int]:
        """Tokenize ``text`` into the official AR prompt sequence."""

        def _build() -> list[int]:
            from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import build_prompt_tokens

            return build_prompt_tokens(
                text,
                self.tokenizer,
                task=HUNYUAN_IMAGE3_AR_TASK,
                sys_type=HUNYUAN_IMAGE3_AR_SYS_TYPE,
            ).token_ids

        prompt_length = self.rollout_config.prompt_length
        token_ids = await self.loop.run_in_executor(None, _build)
        if len(token_ids) > prompt_length:
            raise ValueError(
                f"HunyuanImage3 AR prompt length ({len(token_ids)}) exceeds "
                f"rollout.prompt_length ({prompt_length}); raise prompt_length."
            )
        return normalize_token_ids(token_ids)

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> DiffusionAgentLoopOutput:
        """One hi3 turn: render the official AR prompt, then generate the image."""
        raw_prompt = kwargs["raw_prompt"]
        raw_negative_prompt = kwargs.get("raw_negative_prompt")

        multi_modal_data = await self.process_vision_info(raw_prompt)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        caption = messages_to_text(raw_prompt)
        prompt_ids = await self._official_prompt_ids(caption)

        negative_prompt_ids = None
        if raw_negative_prompt is not None:
            negative_prompt_ids = await self._official_prompt_ids(messages_to_text(raw_negative_prompt))

        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
                negative_prompt_ids=negative_prompt_ids,
                prompt_text=caption,
                use_system_prompt=HUNYUAN_IMAGE3_AR_SYS_TYPE,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1

        return DiffusionAgentLoopOutput(
            prompt_ids=prompt_ids,
            response_diffusion_output=output.diffusion_output,
            response_logprobs=output.log_probs,
            num_turns=2,
            metrics=metrics,
            extra_fields=output.extra_fields,
        )
