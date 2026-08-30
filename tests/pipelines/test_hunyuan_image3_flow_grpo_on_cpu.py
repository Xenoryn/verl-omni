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

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from verl_omni.pipelines.hunyuan_image3_flow_grpo.common import (
    HUNYUAN_IMAGE3_SCHEDULER_KWARGS,
    HUNYUAN_IMAGE3_TIMESTEP_SHIFT,
    apply_hunyuan_cfg,
    build_hunyuan_image3_scheduler,
    setup_hunyuan_image3_sigmas,
)
from verl_omni.pipelines.hunyuan_image3_flow_grpo.diffusers_training_adapter import HunyuanImage3FlowGRPO
from verl_omni.pipelines.hunyuan_image3_flow_grpo.hunyuan_image3_model import (
    HunyuanImage3ForTraining,
    HunyuanImage3TrainingConfig,
    HunyuanTopKGate,
)
from verl_omni.pipelines.hunyuan_image3_flow_grpo.vllm_omni_rollout_adapter import HunyuanImage3PipelineWithLogProb
from verl_omni.pipelines.model_base import DiffusionModelBase, VllmOmniPipelineBase


def _tiny_config() -> HunyuanImage3TrainingConfig:
    return HunyuanImage3TrainingConfig(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=16,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_head_dim=8,
        num_experts=2,
        moe_topk=1,
        num_shared_expert=1,
        use_mixed_mlp_moe=True,
        patch_size=1,
        patch_embed_hidden_dim=32,
        latent_channels=4,
        pad_token_id=0,
        image_token_id=7,
    )


def test_architecture_registers_both_adapters() -> None:
    assert DiffusionModelBase.get_class_by_name("HunyuanImage3ForCausalMM", "flow_grpo") is HunyuanImage3FlowGRPO
    assert VllmOmniPipelineBase.get_class("HunyuanImage3ForCausalMM", "flow_grpo") is HunyuanImage3PipelineWithLogProb


def test_scheduler_config_matches_hunyuan() -> None:
    assert HUNYUAN_IMAGE3_TIMESTEP_SHIFT == 3.0
    assert HUNYUAN_IMAGE3_SCHEDULER_KWARGS["shift"] == 3.0
    assert HUNYUAN_IMAGE3_SCHEDULER_KWARGS["time_shift_type"] == "exponential"
    assert HUNYUAN_IMAGE3_SCHEDULER_KWARGS["use_dynamic_shifting"] is False


def test_scheduler_matches_vllm_omni_constructor() -> None:
    # The SDE subclass must share the exact sigma schedule with the diffusers
    # scheduler vllm-omni builds (same constructor kwargs, same set_timesteps).
    from diffusers import FlowMatchEulerDiscreteScheduler

    reference = FlowMatchEulerDiscreteScheduler(**HUNYUAN_IMAGE3_SCHEDULER_KWARGS)
    ours = build_hunyuan_image3_scheduler()
    reference.set_timesteps(num_inference_steps=8)
    ours.set_timesteps(num_inference_steps=8)
    torch.testing.assert_close(ours.sigmas, reference.sigmas)
    torch.testing.assert_close(ours.timesteps, reference.timesteps)


def test_cfg_combine_standard_guidance() -> None:
    cond = torch.tensor([[[2.0]]])
    uncond = torch.tensor([[[1.0]]])
    combined = apply_hunyuan_cfg(cond, uncond, guidance_scale=2.5)
    assert combined.item() == pytest.approx(1.0 + 2.5 * (2.0 - 1.0))


def test_sigma_schedule_configures_scheduler() -> None:
    scheduler = build_hunyuan_image3_scheduler()
    sigmas = setup_hunyuan_image3_sigmas(scheduler, num_steps=4, device="cpu")
    # vllm-omni runs set_timesteps(num_inference_steps=4): timesteps holds N
    # entries (0-1000 denoise scale), sigmas holds N+1 with the terminal 0.
    assert scheduler.timesteps.numel() == 4
    assert scheduler.sigmas.numel() == 5
    assert scheduler.sigmas[0].item() == pytest.approx(1.0)
    assert scheduler.sigmas[-1].item() == pytest.approx(0.0)
    # shift=3.0 keeps the first non-trivial sigma high (vs ~0.75 at shift=1.0).
    assert scheduler.sigmas[1].item() > 0.85
    # The wrapper returns the denoise timesteps unchanged.
    assert sigmas == scheduler.timesteps.tolist()
    assert len(sigmas) == 4


def test_training_adapter_builds_model_inputs() -> None:
    config = _tiny_config()
    module = HunyuanImage3ForTraining(config)

    batch_size, num_steps = 2, 3
    latents = torch.randn(batch_size, num_steps, config.latent_channels, 2, 2)
    timesteps = torch.tensor([[900.0, 700.0, 500.0]]).expand(batch_size, -1)

    positive_ids = [[3, 5, 6, 8, 7, 7, 7, 7], [3, 5, 6, 8, 7, 7, 7, 7]]
    negative_ids = [[6, 8, 7, 7, 7, 7], [6, 8, 7, 7, 7, 7]]

    micro_batch = MagicMock()
    fields = {
        "prompt_token_ids": positive_ids,
        "negative_prompt_token_ids": negative_ids,
        "gen_timestep_scatter_index": [[3], [3]],
        "negative_gen_timestep_scatter_index": [[1], [1]],
        "rope_image_info": [[4, 8, 2, 2], [4, 8, 2, 2]],
        "negative_rope_image_info": [[2, 6, 2, 2], [2, 6, 2, 2]],
    }
    micro_batch.__getitem__.side_effect = lambda key: fields[key]
    micro_batch.get.side_effect = lambda key, default=None: fields.get(key, default)

    model_config = SimpleNamespace(
        pipeline=SimpleNamespace(num_inference_steps=4, guidance_scale=2.5, guidance_rescale=0.0),
    )

    positive, negative = HunyuanImage3FlowGRPO.prepare_model_inputs(
        module=module,
        model_config=model_config,
        latents=latents,
        timesteps=timesteps,
        prompt_embeds=None,
        prompt_embeds_mask=None,
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        micro_batch=micro_batch,
        step=1,
    )

    assert positive["input_ids"].shape == (batch_size, 8)
    assert positive["images"].shape == (batch_size, config.latent_channels, 2, 2)
    assert positive["timesteps"].tolist() == [700.0, 700.0]
    assert positive["image_mask"].sum(dim=1).tolist() == [4, 4]
    assert negative["input_ids"].shape == (batch_size, 6)
    # Positive and negative branches carry distinct <timestep> scatter indices and
    # RoPE spans (their text lengths differ).
    assert positive["gen_timestep_scatter_index"].tolist() == [[3], [3]]
    assert negative["gen_timestep_scatter_index"].tolist() == [[1], [1]]
    assert positive["rope_image_info"] == [[(slice(4, 8), (2, 2))], [(slice(4, 8), (2, 2))]]
    assert negative["rope_image_info"] == [[(slice(2, 6), (2, 2))], [(slice(2, 6), (2, 2))]]


def test_tiny_model_forward_returns_latent_velocity() -> None:
    config = _tiny_config()
    model = HunyuanImage3ForTraining(config)

    batch_size, seq_len = 2, 8
    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    images = torch.randn(batch_size, config.latent_channels, 2, 2)
    timesteps = torch.tensor([0.8, 0.6])
    image_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    image_mask[:, 4:8] = True
    gen_timestep_scatter_index = torch.tensor([[3], [3]])
    rope_image_info = [[(slice(4, 8), (2, 2))], [(slice(4, 8), (2, 2))]]

    velocity = model(
        input_ids=input_ids,
        images=images,
        timesteps=timesteps,
        image_mask=image_mask,
        gen_timestep_scatter_index=gen_timestep_scatter_index,
        rope_image_info=rope_image_info,
    )

    assert isinstance(velocity, tuple)
    assert velocity[0].shape == (batch_size, config.latent_channels, 2, 2)


def test_from_model_path_parses_checkpoint_config(tmp_path) -> None:
    import json

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "vocab_size": 133120,
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "attention_head_dim": 8,
                "num_experts": 8,
                "moe_topk": [8, 8],
                "num_shared_expert": [1, 1],
                "moe_intermediate_size": [16, 16],
                "vae": {"latent_channels": 16},
            }
        )
    )
    config = HunyuanImage3TrainingConfig.from_model_path(str(tmp_path))
    assert config.hidden_size == 64
    assert config.num_experts == 8
    assert config.moe_topk == 8
    assert config.latent_channels == 16


def test_moe_gate_routing_is_deterministic_under_ties() -> None:
    """Tied gate scores must route to the same experts on every invocation.

    ``torch.topk`` leaves tied indices unstable, so a checkpoint recompute can
    flip a token's expert and produce wrong gradients.  The gate uses a stable
    sort, which breaks ties by ascending expert id; the test locks the
    tie-break and normalization semantics.
    """
    config = HunyuanImage3TrainingConfig(hidden_size=8, num_experts=8, moe_topk=4)
    gate = HunyuanTopKGate(config)
    with torch.no_grad():
        # Zero router weights -> uniform softmax -> every expert is tied.
        gate.wg.weight.zero_()
        hidden = torch.zeros(2, 1, config.hidden_size)
        weights, indices = gate(hidden)
        weights_again, indices_again = gate(hidden)

    # Ties break by ascending expert id, so the first moe_topk experts are chosen.
    assert indices.tolist() == [[0, 1, 2, 3], [0, 1, 2, 3]]
    # Re-invoking the same gate yields identical routing.
    assert indices.tolist() == indices_again.tolist()
    assert weights.tolist() == weights_again.tolist()
    # Normalization is preserved: top-k weights sum to 1 per token.
    assert torch.allclose(weights.sum(dim=1), torch.ones(weights.shape[0], dtype=weights.dtype))
    # The sort-based top-k slice must stay dense: the MoE dispatch flattens the
    # index with ``view(-1)``, which rejects a non-contiguous row slice.
    assert indices.is_contiguous()
    assert weights.is_contiguous()
    assert indices.view(-1).tolist() == [0, 1, 2, 3, 0, 1, 2, 3]


def test_text_forward_returns_lm_head_logits_rmpad() -> None:
    """AR text forward is rmpad (upstream verl LM layout): input_ids is
    ``(1, total_nnz)`` with ``cu_seqlens`` marking sample boundaries."""
    config = _tiny_config()
    model = HunyuanImage3ForTraining(config)

    # Two samples of lengths 4 and 6 concatenated end-to-end.
    seq_lens = [4, 6]
    total_nnz = sum(seq_lens)
    input_ids = torch.randint(0, config.vocab_size - 1, (1, total_nnz))
    cu_seqlens = torch.tensor([0, seq_lens[0], sum(seq_lens)], dtype=torch.int32)

    logits = model(input_ids, mode="text", cu_seqlens=cu_seqlens, max_seqlen=max(seq_lens))

    assert logits.shape == (1, total_nnz, config.vocab_size)
    # lm_head is a real (non-tied) parameter, distinct from wte.
    assert model.lm_head.weight is not model.wte.weight
    assert model.lm_head.weight.shape == (config.vocab_size, config.hidden_size)


def test_text_forward_rmpad_matches_padded_sample_by_sample() -> None:
    """Equivalence check: running each sample through the rmpad path in
    isolation must yield the same logits at each token as running the whole
    batch concatenated. Locks the varlen attention's block-diagonal masking
    (samples must not attend across boundaries) via the SDPA fallback path."""
    config = _tiny_config()
    model = HunyuanImage3ForTraining(config).eval()

    seq_lens = [3, 5]
    samples = [torch.randint(0, config.vocab_size - 1, (L,)) for L in seq_lens]
    total_nnz = sum(seq_lens)

    # Rmpad batched call.
    input_ids_flat = torch.cat(samples).unsqueeze(0)  # (1, total_nnz)
    cu = torch.tensor([0, seq_lens[0], sum(seq_lens)], dtype=torch.int32)
    with torch.no_grad():
        logits_batched = model(input_ids_flat, mode="text", cu_seqlens=cu, max_seqlen=max(seq_lens))

    # Per-sample isolated calls (each its own rmpad batch of one).
    logits_isolated = []
    for L, s in zip(seq_lens, samples):
        cu_solo = torch.tensor([0, L], dtype=torch.int32)
        with torch.no_grad():
            logits_isolated.append(model(s.unsqueeze(0), mode="text", cu_seqlens=cu_solo, max_seqlen=L))
    logits_isolated_cat = torch.cat([x.squeeze(0) for x in logits_isolated], dim=0).unsqueeze(0)

    # If sample 0 could attend to sample 1 (or vice versa) in the batched call,
    # its logits at every position would drift from the isolated case.
    assert torch.allclose(logits_batched, logits_isolated_cat, atol=1e-5, rtol=1e-4)


def test_forward_mode_dispatch() -> None:
    """Routing both paths through ``forward`` is what lets the root FSDP hook fire.

    Locks four invariants: (a) ``mode`` is the sole switch — no separate public
    entry point; (b) unknown modes fail loudly; (c) ``HunyuanFinalRMSNorm`` is
    not in ``_no_split_modules``; (d) ``mode="text"`` requires the rmpad
    kwargs (varlen FA layout).
    """
    config = _tiny_config()
    model = HunyuanImage3ForTraining(config)

    # Only forward() is a public entry: both paths must go through the root
    # forward hook for FSDP2 param all-gather.
    assert not hasattr(model, "forward_text")
    assert "HunyuanFinalRMSNorm" not in HunyuanImage3ForTraining._no_split_modules

    input_ids = torch.randint(0, config.vocab_size - 1, (1, 4))
    with pytest.raises(ValueError, match="Unknown forward mode"):
        model(input_ids, mode="not_a_mode")

    # mode='text' without rmpad kwargs is a caller-side error (varlen FA layout
    # is not optional).
    with pytest.raises(ValueError, match="rmpad"):
        model(input_ids, mode="text")


def test_from_pretrained_loads_sharded_checkpoint(tmp_path) -> None:
    """The source-rank materialize path round-trips a sharded checkpoint exactly.

    On a single CPU process ``from_pretrained`` takes the source-rank branch (no
    ``meta`` device, ``nullcontext`` fallback) and streams the shards in.  Covers
    the checkpoint-key remap (``model.`` prefix on the transformer body, none on
    the image-pathway + ``lm_head`` modules) and the strict-free load.
    """
    import json
    import os

    from safetensors.torch import save_file

    # from_model_path reads latent_channels from a nested "vae" block (the vllm-omni
    # checkpoint layout), so write the config in that shape rather than the flat
    # dataclass `asdict` used by save_pretrained.
    config_json = {
        "vocab_size": 256,
        "hidden_size": 32,
        "intermediate_size": 16,
        "moe_intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "attention_head_dim": 8,
        "num_experts": 2,
        "moe_topk": 1,
        "num_shared_expert": 1,
        "use_mixed_mlp_moe": True,
        "patch_size": 1,
        "patch_embed_hidden_dim": 32,
        "vae": {"latent_channels": 4},
        "pad_token_id": 0,
        "image_token_id": 7,
    }
    with open(os.path.join(tmp_path, "config.json"), "w") as f:
        json.dump(config_json, f)

    source = HunyuanImage3ForTraining(_tiny_config())

    def _checkpoint_key(name: str) -> str:
        return f"model.{name}" if name.startswith(("wte.", "layers.", "ln_f.")) else name

    weights = {_checkpoint_key(k): v for k, v in source.state_dict().items() if isinstance(v, torch.Tensor)}
    save_file(weights, os.path.join(tmp_path, "model.safetensors"))
    with open(os.path.join(tmp_path, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": {k: "model.safetensors" for k in weights}}, f)

    loaded = HunyuanImage3ForTraining.from_pretrained(str(tmp_path), torch_dtype=torch.float32)

    assert not next(loaded.parameters()).is_meta
    for (name, weight), (loaded_name, loaded_weight) in zip(source.state_dict().items(), loaded.state_dict().items()):
        assert name == loaded_name
        torch.testing.assert_close(loaded_weight, weight)

def test_hi3_common_message_helpers() -> None:
    from verl_omni.pipelines.hunyuan_image3_flow_grpo.common import messages_to_text, normalize_ar_cot_text

    assert messages_to_text("plain text") == "plain text"
    assert messages_to_text([{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]) == "a\nb"
    assert messages_to_text([{"role": "user", "content": [{"type": "text", "text": "x"}]}]) == "x"

    # The AR generation omits the trigger tag; the helper restores it exactly
    # like vllm-omni's HunyuanImage3Pipeline._normalize_cot_text.
    assert normalize_ar_cot_text(None) is None
    assert normalize_ar_cot_text("") == ""
    assert normalize_ar_cot_text("hello  responsex") == " thinkinghello  responsex"
    assert normalize_ar_cot_text("x</recaption>y") == "<recaption>x</recaption>y"
    assert normalize_ar_cot_text(" thinkinga  responseb") == " thinkinga  responseb"


def test_hi3_agent_loop_registered() -> None:
    import verl_omni.agent_loop  # noqa: F401
    from verl.experimental.agent_loop.agent_loop import _agent_loop_registry

    assert "hunyuan_image3_diffusion_single_turn_agent" in _agent_loop_registry


class _FakeColumn:
    def __init__(self, rows):
        self._rows = list(rows)
        self.shape = (len(rows),)

    def __getitem__(self, i):
        return self._rows[i]


def test_fused_dit_inputs_rebuilds_rollout_prefix(monkeypatch) -> None:
    from verl_omni.pipelines.hunyuan_image3_flow_grpo import diffusers_training_adapter as adapter

    from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import HunyuanImage3Pipeline

    captured = {}

    class _FakeTokenizerWrapper:
        def apply_chat_template(self, **kwargs):
            captured.update(kwargs)
            batch_size = len(kwargs["batch_prompt"])
            seq_len = 6
            output = SimpleNamespace(
                tokens=torch.zeros(batch_size, seq_len, dtype=torch.long),
                gen_image_mask=torch.zeros(batch_size, seq_len, dtype=torch.bool),
                gen_timestep_scatter_index=torch.arange(batch_size).unsqueeze(-1),
            )
            return {"output": output, "sections": None}

    ctx = {
        "tkw": _FakeTokenizerWrapper(),
        "image_processor": SimpleNamespace(build_image_info=lambda size: "image_info"),
        "drop_think": False,
        "image_base_size": None,
        "sequence_template": "instruct",
        "system_prompt": "SYS",
    }
    monkeypatch.setattr(adapter, "_get_hunyuan_image3_tokenizer_ctx", lambda path: ctx)
    monkeypatch.setattr(
        HunyuanImage3Pipeline,
        "build_batch_rope_image_info",
        lambda output, sections: [["rope"]] * output.tokens.shape[0],
    )

    micro_batch = MagicMock()
    fields = {
        "ar_prompt_token_ids": _FakeColumn([[1, 2], [3, 4]]),
        "ar_generated_text": _FakeColumn(["x  responsey", "</recaption>z"]),
        "raw_prompt": _FakeColumn([[{"role": "user", "content": "cap0"}], [{"role": "user", "content": "cap1"}]]),
    }
    micro_batch.get.side_effect = lambda key, default=None: fields.get(key, default)

    latents = torch.randn(2, 3, 4, 1, 1)
    config = SimpleNamespace(vae_downsample_factor=(16, 16))
    model_config = SimpleNamespace(local_path="/fake/model")

    result = adapter.HunyuanImage3FlowGRPO._build_fused_dit_inputs(
        model_config, config, micro_batch, latents, step=1, device=torch.device("cpu")
    )
    assert result is not None
    positive_ids, image_mask, scatter, rope = result
    assert positive_ids.shape == (2, 6)
    assert image_mask.shape == (2, 6)
    assert scatter.tolist() == [[0], [1]]
    assert rope == [["rope"], ["rope"]]

    # Rollout-parity knobs: cot re-parse affects the fused ids, so the rebuild
    # must apply the same normalization as the engine template.
    assert captured["batch_prompt"] == ["cap0", "cap1"]
    assert captured["batch_cot_text"] == [" thinkingx  responsey", "<recaption></recaption>z"]
    assert captured["batch_system_prompt"] == ["SYS", "SYS"]
    assert captured["mode"] == "gen_image"
    assert captured["bot_task"] == "auto"
    assert captured["sequence_template"] == "instruct"
    assert captured["cfg_factor"] == 1
    assert captured["drop_think"] is False


def test_fused_dit_inputs_falls_back_when_ar_fields_absent() -> None:
    from verl_omni.pipelines.hunyuan_image3_flow_grpo import diffusers_training_adapter as adapter

    micro_batch = MagicMock()
    micro_batch.get.side_effect = lambda key, default=None: None
    config = SimpleNamespace(vae_downsample_factor=(16, 16))
    model_config = SimpleNamespace(local_path="/fake/model")
    latents = torch.randn(2, 3, 4, 1, 1)

    result = adapter.HunyuanImage3FlowGRPO._build_fused_dit_inputs(
        model_config, config, micro_batch, latents, step=0, device=torch.device("cpu")
    )
    assert result is None
