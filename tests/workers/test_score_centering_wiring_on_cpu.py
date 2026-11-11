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

"""CPU coverage for score centering's FSDP and Megatron engine wiring."""

import socket
import sys
import types
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensordict import TensorDict

import verl.utils.torch_functional as verl_F
from verl.models.transformers.dense_common import forward_with_torch_backend, forward_with_triton_backend
from verl.trainer.ppo.score_centering import pad_rollout_topk, score_centering_ppo_loss
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.workers.config import ActorConfig, FSDPEngineConfig, PolicyLossConfig
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

try:
    from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead
except ModuleNotFoundError:
    MegatronEngineWithLMHead = None


@pytest.mark.parametrize("use_remove_padding", [True, False])
@pytest.mark.parametrize("score_centering", [True, False])
def test_prepare_model_outputs_keeps_logits_intact_for_score_centering(use_remove_padding, score_centering):
    """The score centering hook saves the logits and re-reads them in its backward, so the engine
    must not run the flash-attn cross-entropy backward that writes the gradient into the logits."""
    seq_lengths = torch.tensor([3, 2])
    total_nnz, vocab_size = int(seq_lengths.sum()), 8
    cu_seqlens = torch.cat([torch.tensor([0]), seq_lengths.cumsum(0)])
    input_ids = torch.nested.nested_tensor_from_jagged(torch.randint(0, vocab_size, (total_nnz,)), offsets=cu_seqlens)
    output = type("Output", (), {})()
    output_args = {"input_ids_rmpad_rolled": torch.randint(0, vocab_size, (total_nnz,))}
    if use_remove_padding:
        output.logits = torch.randn(1, total_nnz, vocab_size)
        output_args.update(temperature_rmpad=torch.ones(total_nnz), pad_size=0)
    else:
        output.logits = torch.randn(2, 3, vocab_size)
        output_args["temperature"] = torch.ones(2)

    micro_batch = TensorDict({"input_ids": input_ids}, batch_size=[])
    tu.assign_non_tensor(
        micro_batch,
        use_remove_padding=use_remove_padding,
        pad_mode=DatasetPadMode.NO_PADDING,
        score_centering=score_centering,
    )
    engine = object.__new__(FSDPEngineWithLMHead)
    engine.use_ulysses_sp = False
    engine.engine_config = SimpleNamespace(batch_invariant=False)

    def logits_processor(student_logits, data):
        return {"sc_correction": torch.zeros(student_logits.shape[:2])}

    with patch(
        "verl.workers.engine.fsdp.transformer_impl.logprobs_from_logits", return_value=torch.zeros(total_nnz)
    ) as mock_logprobs:
        FSDPEngineWithLMHead.prepare_model_outputs(engine, output, output_args, micro_batch, logits_processor)

    assert mock_logprobs.call_args.kwargs.get("inplace_backward", True) is not score_centering


@pytest.mark.skipif(MegatronEngineWithLMHead is None, reason="megatron-core is not installed")
@pytest.mark.parametrize("score_centering", [True, False])
def test_megatron_logits_processor_runs_score_centering_hook_and_log_probs(score_centering):
    total_nnz, vocab_size = 5, 8
    engine = object.__new__(MegatronEngineWithLMHead)
    engine.engine_config = SimpleNamespace(entropy_from_logits_with_chunking=False, entropy_from_logits_chunk_size=1024)
    engine.tf_config = SimpleNamespace(fp8=None)
    calls = []

    def logits_processor(student_logits, data, data_format, **packing):
        calls.append(data_format)
        return {"sc_correction": torch.zeros(student_logits.shape[:2])}

    with patch(
        "verl.workers.engine.megatron.transformer_impl.vocab_parallel_log_probs_from_logits",
        return_value=torch.zeros(1, total_nnz),
    ) as mock_log_probs:
        ret = engine._lm_head_logits_processor(
            torch.randn(1, total_nnz, vocab_size),
            torch.randint(0, vocab_size, (1, total_nnz)),
            torch.ones(1, total_nnz),
            calculate_sum_pi_squared=False,
            calculate_entropy=False,
            distillation_use_topk=False,
            distillation_only=False,
            score_centering=score_centering,
            logits_processor_func=logits_processor,
            batch=TensorDict({}, batch_size=[]),
            data_format="thd",
        )

    mock_log_probs.assert_called_once()
    assert "log_probs" in ret
    assert calls == (["thd"] if score_centering else [])
    assert ("sc_correction" in ret) is score_centering


@pytest.mark.skipif(MegatronEngineWithLMHead is None, reason="megatron-core is not installed")
def test_megatron_log_probs_do_not_touch_the_logits_score_centering_saved():
    """Megatron's vocab-parallel cross entropy shifts fp32 logits in place; the score centering hook
    re-reads the logits it was given in its backward, so the engine must hand the cross entropy a copy."""
    total_nnz, vocab_size = 5, 8
    engine = object.__new__(MegatronEngineWithLMHead)
    engine.engine_config = SimpleNamespace(entropy_from_logits_with_chunking=False, entropy_from_logits_chunk_size=1024)
    engine.tf_config = SimpleNamespace(fp8=None)
    seen = {}

    def logits_processor(student_logits, data, data_format, **packing):
        seen["logits"] = student_logits
        seen["snapshot"] = student_logits.clone()
        return {"sc_correction": torch.zeros(student_logits.shape[:2])}

    def shifting_log_probs(logits, label):
        logits.sub_(logits.max(dim=-1, keepdim=True).values)
        return torch.zeros(logits.shape[:2])

    with patch(
        "verl.workers.engine.megatron.transformer_impl.vocab_parallel_log_probs_from_logits",
        side_effect=shifting_log_probs,
    ):
        engine._lm_head_logits_processor(
            torch.randn(1, total_nnz, vocab_size, dtype=torch.float32),
            torch.randint(0, vocab_size, (1, total_nnz)),
            torch.ones(1, total_nnz),
            calculate_sum_pi_squared=False,
            calculate_entropy=False,
            distillation_use_topk=False,
            distillation_only=False,
            score_centering=True,
            logits_processor_func=logits_processor,
            batch=TensorDict({}, batch_size=[]),
            data_format="thd",
        )

    torch.testing.assert_close(seen["logits"], seen["snapshot"])


def _build_ragged_micro_batch(vocab_size=19, k=4, support_complete=False):
    torch.manual_seed(11)
    prompts = [torch.tensor([1, 2]), torch.tensor([3, 4, 5])]
    responses = [torch.tensor([6, 7, 8]), torch.tensor([9, 10])]
    input_seqs = [torch.cat([p, r]) for p, r in zip(prompts, responses, strict=True)]
    pos_seqs = [torch.arange(len(s)) for s in input_seqs]

    sampler_logits = [torch.randn(len(r), vocab_size) for r in responses]
    topk_ids_list, topk_lp_list, complete_list, rollout_lp_list = [], [], [], []
    for idx, (r, slogits) in enumerate(zip(responses, sampler_logits, strict=True)):
        prompt_len = len(input_seqs[idx]) - len(r)
        if support_complete:
            vals, order = torch.topk(slogits, k=k, dim=-1)
            vals[:, -1] = float("-inf")
            order[:, 0] = r
            vals[:, 0] = slogits.gather(-1, r.unsqueeze(-1)).squeeze(-1)
            slp = torch.log_softmax(vals, dim=-1)
            sampled_lp = slp[:, 0]
            full_ids, full_lp = pad_rollout_topk(order, slp, prompt_len)
            comp = torch.zeros(len(full_ids), dtype=torch.bool)
            comp[prompt_len - 1 : prompt_len - 1 + len(r)] = True
        else:
            full_slp = torch.log_softmax(slogits, dim=-1)
            sampled_lp = full_slp.gather(-1, r.unsqueeze(-1)).squeeze(-1)
            ids = torch.topk(full_slp, k=k, dim=-1).indices
            full_ids, full_lp = pad_rollout_topk(ids, full_slp.gather(-1, ids), prompt_len)
            comp = torch.zeros(len(full_ids), dtype=torch.bool)
        topk_ids_list.append(full_ids)
        topk_lp_list.append(full_lp)
        complete_list.append(comp)
        rollout_lp_list.append(sampled_lp)

    batch = TensorDict(
        {
            "prompts": torch.nested.as_nested_tensor(prompts, layout=torch.jagged),
            "input_ids": torch.nested.as_nested_tensor(input_seqs, layout=torch.jagged),
            "position_ids": torch.nested.as_nested_tensor(pos_seqs, layout=torch.jagged),
            "responses": torch.nested.as_nested_tensor(responses, layout=torch.jagged),
            "response_mask": torch.nested.as_nested_tensor(
                [torch.ones_like(r) for r in responses], layout=torch.jagged
            ),
            "advantages": torch.nested.as_nested_tensor(
                [torch.tensor([1.2, -0.5, 0.8]), torch.tensor([-0.9, 0.4])], layout=torch.jagged
            ),
            "old_log_probs": torch.nested.as_nested_tensor(rollout_lp_list, layout=torch.jagged),
            "rollout_log_probs": torch.nested.as_nested_tensor(rollout_lp_list, layout=torch.jagged),
            "rollout_topk_ids": torch.nested.as_nested_tensor(topk_ids_list, layout=torch.jagged),
            "rollout_topk_log_probs": torch.nested.as_nested_tensor(topk_lp_list, layout=torch.jagged),
            "rollout_support_complete": torch.nested.as_nested_tensor(complete_list, layout=torch.jagged),
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(
        batch,
        temperature=0.8,
        use_remove_padding=True,
        pad_mode=DatasetPadMode.NO_PADDING,
        score_centering=True,
        calculate_entropy=True,
        dp_size=1,
        batch_num_tokens=5,
        global_batch_size=2,
    )
    return batch


class _TinyFSDPCausalLM(torch.nn.Module):
    """Minimal causal LM exercising ``dense_common.forward_with_{torch,triton}_backend`` and eager logits."""

    def __init__(self, vocab_size: int, hidden_dim: int, fused_backend: str | None = None):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3", output_attentions=False, output_hidden_states=False)
        self.embed = torch.nn.Embedding(vocab_size, hidden_dim)
        self.lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False)
        self._verl_fused_kernels_backend = fused_backend
        self._fused_backend = fused_backend

    def model(self, input_ids=None, **kwargs):
        from transformers.modeling_outputs import BaseModelOutputWithPast

        return BaseModelOutputWithPast(last_hidden_state=self.embed(input_ids))

    def forward(self, input_ids=None, **kwargs):
        if self._fused_backend in ("torch", "liger"):
            return forward_with_torch_backend(self, input_ids=input_ids, **kwargs)
        if self._fused_backend == "triton":
            return forward_with_triton_backend(self, input_ids=input_ids, **kwargs)
        return SimpleNamespace(logits=self.lm_head(self.embed(input_ids)))


@pytest.mark.parametrize("fused_backend", ["torch", "liger", "triton"])
@pytest.mark.parametrize(
    "rollout_is, threshold",
    [
        (None, 2.0),  # SC
        ("token", 2.0),  # SC + TIS
        ("token", "0.5_5.0"),  # SC + MIS
    ],
)
@pytest.mark.parametrize("support_complete", [False, True])
def test_fsdp_fused_score_centering_matches_eager_across_backends(
    fused_backend, rollout_is, threshold, support_complete
):
    vocab_size, hidden_dim = 19, 12
    torch.manual_seed(29)
    model_fused = _TinyFSDPCausalLM(vocab_size, hidden_dim, fused_backend=fused_backend)
    model_eager = _TinyFSDPCausalLM(vocab_size, hidden_dim, fused_backend=None)
    model_eager.load_state_dict(model_fused.state_dict())

    rc = {
        "bypass_mode": True,
        "loss_type": "reinforce",
        "rollout_is": rollout_is,
        "rollout_is_threshold": threshold,
        "score_centering": True,
    }
    actor_cfg = ActorConfig(
        strategy="fsdp2",
        rollout_n=1,
        ppo_mini_batch_size=2,
        ppo_micro_batch_size_per_gpu=2,
        use_dynamic_bsz=False,
        policy_loss=PolicyLossConfig(loss_mode="bypass_mode", rollout_correction=rc),
    )
    loss_fn = partial(score_centering_ppo_loss, config=actor_cfg)

    def _run(model, use_fused):
        engine = object.__new__(FSDPEngineWithLMHead)
        engine.module = model
        engine.engine_config = FSDPEngineConfig(batch_invariant=False)
        engine.use_ulysses_sp = False
        engine.ulysses_sequence_parallel_size = 1
        engine.pad_to_length = False
        engine.pad_to_length_bucket = 0
        engine._autocast_dtype = torch.float32
        engine.compute_entropy_from_logits = verl_F.entropy_from_logits
        engine.get_data_parallel_group = lambda: None

        batch = _build_ragged_micro_batch(vocab_size=vocab_size, k=4, support_complete=support_complete)
        tu.assign_non_tensor(batch, use_fused_kernels=use_fused)
        with (
            patch("verl.workers.engine.fsdp.transformer_impl.get_device_name", return_value="cpu"),
            patch("verl.workers.engine.fsdp.transformer_impl.get_device_id", return_value="cpu"),
        ):
            loss, out = engine.forward_step(batch, loss_function=loss_fn, forward_only=False)
        loss.backward()
        return loss.detach(), out["metrics"]

    with patch.dict("os.environ", {"VERL_DISABLE_FLASH_ATTN_CE": "1"}):
        loss_fused, metrics_fused = _run(model_fused, use_fused=True)
        loss_eager, metrics_eager = _run(model_eager, use_fused=False)

    torch.testing.assert_close(loss_fused, loss_eager, atol=1e-5, rtol=1e-5)
    for key in (
        "actor/sc_correction",
        "actor/sc_sampler_head_mass",
        "actor/sc_train_head_mass",
        "actor/sc_support_aligned",
        "actor/sc_sampler_tail_mass",
    ):
        v_fused = metrics_fused[key].aggregate() if hasattr(metrics_fused[key], "aggregate") else metrics_fused[key]
        v_eager = metrics_eager[key].aggregate() if hasattr(metrics_eager[key], "aggregate") else metrics_eager[key]
        torch.testing.assert_close(torch.as_tensor(v_fused), torch.as_tensor(v_eager), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(model_fused.embed.weight.grad, model_eager.embed.weight.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(model_fused.lm_head.weight.grad, model_eager.lm_head.weight.grad, atol=1e-5, rtol=1e-5)


def _install_megatron_stubs():
    """Install lightweight CPU stubs when megatron-core is not installed in the unit-test environment."""
    if "megatron.core" in sys.modules:
        return

    @dataclass
    class _PackedSeqParams:
        qkv_format: str = "thd"
        cu_seqlens_q: torch.Tensor = None
        max_seqlen_q: int = 0
        cu_seqlens_kv: torch.Tensor = None
        max_seqlen_kv: int = 0
        cu_seqlens_q_padded: torch.Tensor = None
        cu_seqlens_kv_padded: torch.Tensor = None
        cp_partition_mode: str = "zigzag"
        local_cp_size: int = None
        cp_group: object = None

    class _GPTModel(torch.nn.Module):
        pass

    megatron_pkg = types.ModuleType("megatron")
    mcore = types.ModuleType("megatron.core")
    mcore.__version__ = "0.18.0"

    parallel_state = types.ModuleType("megatron.core.parallel_state")
    parallel_state.get_tensor_model_parallel_group = lambda: None
    parallel_state.get_tensor_model_parallel_world_size = lambda: 1
    parallel_state.get_tensor_model_parallel_rank = lambda: 0
    parallel_state.get_context_parallel_world_size = lambda: 1
    parallel_state.get_context_parallel_rank = lambda: 0
    parallel_state.get_context_parallel_group = lambda: None
    parallel_state.get_expert_tensor_and_model_parallel_group = lambda: None

    config_logger = types.ModuleType("megatron.core.config_logger")
    config_logger.has_config_logger_enabled = lambda cfg: False
    config_logger.log_config_to_disk = lambda *a, **kw: None

    inference_pkg = types.ModuleType("megatron.core.inference")
    inference_ctx = types.ModuleType("megatron.core.inference.contexts")
    inference_ctx.BaseInferenceContext = object

    models_pkg = types.ModuleType("megatron.core.models")
    gpt_pkg = types.ModuleType("megatron.core.models.gpt")
    gpt_model_mod = types.ModuleType("megatron.core.models.gpt.gpt_model")
    gpt_model_mod.GPTModel = _GPTModel

    packed_mod = types.ModuleType("megatron.core.packed_seq_params")
    packed_mod.PackedSeqParams = _PackedSeqParams

    tp_pkg = types.ModuleType("megatron.core.tensor_parallel")
    tp_mappings = types.ModuleType("megatron.core.tensor_parallel.mappings")
    tp_mappings.gather_from_sequence_parallel_region = lambda x, tensor_parallel_output_grad=True: x

    utils_mod = types.ModuleType("megatron.core.utils")
    utils_mod.deprecate_inference_params = lambda ctx, params: ctx if ctx is not None else params

    mcore.parallel_state = parallel_state
    megatron_pkg.core = mcore

    for name, mod in {
        "megatron": megatron_pkg,
        "megatron.core": mcore,
        "megatron.core.parallel_state": parallel_state,
        "megatron.core.config_logger": config_logger,
        "megatron.core.inference": inference_pkg,
        "megatron.core.inference.contexts": inference_ctx,
        "megatron.core.models": models_pkg,
        "megatron.core.models.gpt": gpt_pkg,
        "megatron.core.models.gpt.gpt_model": gpt_model_mod,
        "megatron.core.packed_seq_params": packed_mod,
        "megatron.core.tensor_parallel": tp_pkg,
        "megatron.core.tensor_parallel.mappings": tp_mappings,
        "megatron.core.utils": utils_mod,
    }.items():
        sys.modules.setdefault(name, mod)

    sys.modules.setdefault("verl.utils.megatron_utils", SimpleNamespace(unwrap_model=lambda m: m))
    patch_mod = types.ModuleType("verl.models.mcore.patch")
    patch_mod.apply_patch_megatron_v012_with_torch_v28_v29 = lambda: None
    patch_mod.neutralize_broken_flash_attn_cute = lambda: None
    sys.modules.setdefault("verl.models.mcore.patch", patch_mod)


def _import_mcore_modules():
    _install_megatron_stubs()
    from verl.models.mcore import model_forward as mf
    from verl.models.mcore import model_forward_fused as mff

    return mf, mff


def _megatron_actor_config(rollout_is, rollout_is_threshold=2.0):
    rc = {
        "bypass_mode": True,
        "loss_type": "reinforce",
        "rollout_is": rollout_is,
        "rollout_is_threshold": rollout_is_threshold,
        "score_centering": True,
    }
    return ActorConfig(
        strategy="megatron",
        rollout_n=1,
        ppo_mini_batch_size=2,
        ppo_micro_batch_size_per_gpu=2,
        use_dynamic_bsz=False,
        policy_loss=PolicyLossConfig(loss_mode="bypass_mode", rollout_correction=rc),
    )


class _TinyMegatronHookModel(torch.nn.Module):
    """Minimal Megatron GPTModel supporting both native output_processor hook and eager logits."""

    def __init__(self, vocab_size: int, hidden_dim: int, weight_shard: torch.Tensor = None):
        super().__init__()
        self.pre_process = True
        self.post_process = True
        self.config = SimpleNamespace(sequence_parallel=False, fp8=None, fp4=None)
        self.embed = torch.nn.Embedding(vocab_size, hidden_dim)
        if weight_shard is None:
            self.output_layer = torch.nn.Linear(hidden_dim, vocab_size, bias=False)
        else:
            self.output_layer = SimpleNamespace(weight=weight_shard)
        self._verl_fused_forward_mode = "hook"
        self._verl_fused_impl_backend = "triton"

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        packed_seq_params=None,
        labels=None,
        output_processor=None,
        output_processor_context=None,
        **kwargs,
    ):
        hidden = self.embed(input_ids)
        if output_processor is not None:
            return output_processor(
                hidden_states=hidden,
                output_layer=self.output_layer,
                output_weight=None,
                labels=labels,
                context=output_processor_context,
                config=self.config,
            )
        return hidden @ self.output_layer.weight.t()


@pytest.mark.parametrize("fused_mode", ["hook", "legacy"])
@pytest.mark.parametrize(
    "rollout_is, threshold",
    [
        (None, 2.0),  # SC
        ("token", 2.0),  # SC + TIS
        ("token", "0.5_5.0"),  # SC + MIS
    ],
)
@pytest.mark.parametrize("support_complete", [False, True])
def test_megatron_fused_score_centering_matches_eager(fused_mode, rollout_is, threshold, support_complete):
    mf, mff = _import_mcore_modules()
    vocab_size, hidden_dim = 24, 12

    torch.manual_seed(41)
    model_fused = _TinyMegatronHookModel(vocab_size, hidden_dim)
    model_eager = _TinyMegatronHookModel(vocab_size, hidden_dim)
    model_eager.load_state_dict(model_fused.state_dict())
    if fused_mode == "legacy":
        model_fused._verl_fused_forward_mode = "legacy"
        model_fused._preprocess = lambda input_ids, **kw: (model_fused.embed(input_ids), None, None, None, None)
        model_fused.decoder = lambda hidden_states, **kw: hidden_states
        model_fused.forward = mff._fused_GPTModel_forward.__get__(model_fused, model_fused.__class__)

    actor_cfg = _megatron_actor_config(rollout_is, threshold)
    loss_fn = partial(score_centering_ppo_loss, config=actor_cfg)
    batch = _build_ragged_micro_batch(vocab_size=vocab_size, k=4, support_complete=support_complete)

    out_fused = mff.fused_forward_model_engine(vision_model=False)(
        model=model_fused,
        input_ids=batch["input_ids"],
        labels=batch["input_ids"].clone(),
        multi_modal_inputs={},
        temperature=0.8,
        calculate_entropy=True,
        pad_token_id=0,
        score_centering=True,
        logits_processor_func=loss_fn,
        batch=batch,
    )
    loss_fused, metrics_fused = loss_fn(model_output=out_fused, data=batch, dp_group=None)
    loss_fused.backward()

    def eager_logits_processor(logits, label, temperature):
        scaled = logits / temperature.unsqueeze(-1)
        probs = torch.softmax(scaled.float(), dim=-1)
        logp = torch.log_softmax(scaled.float(), dim=-1)
        ret = {
            "entropy": -(probs * logp).sum(dim=-1),
            "log_probs": logp.gather(-1, label.unsqueeze(-1)).squeeze(-1),
        }
        ret.update(loss_fn(student_logits=scaled, data=batch, data_format="thd"))
        return ret

    temp_nested = torch.nested.as_nested_tensor(
        [torch.full((len(s),), 0.8) for s in batch["input_ids"].unbind()], layout=torch.jagged
    )
    out_eager = mf.gptmodel_forward_model_engine(
        model=model_eager,
        input_ids=batch["input_ids"],
        multi_modal_inputs={},
        logits_processor=eager_logits_processor,
        logits_processor_args={"label": batch["input_ids"].clone(), "temperature": temp_nested},
        data_format="thd",
    )
    loss_eager, metrics_eager = loss_fn(model_output=out_eager, data=batch, dp_group=None)
    loss_eager.backward()

    torch.testing.assert_close(loss_fused.detach(), loss_eager.detach(), atol=1e-5, rtol=1e-5)
    for key in (
        "actor/sc_correction",
        "actor/sc_sampler_head_mass",
        "actor/sc_train_head_mass",
        "actor/sc_support_aligned",
        "actor/sc_sampler_tail_mass",
    ):
        v_f = metrics_fused[key].aggregate() if hasattr(metrics_fused[key], "aggregate") else metrics_fused[key]
        v_e = metrics_eager[key].aggregate() if hasattr(metrics_eager[key], "aggregate") else metrics_eager[key]
        torch.testing.assert_close(torch.as_tensor(v_f), torch.as_tensor(v_e), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(model_fused.embed.weight.grad, model_eager.embed.weight.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        model_fused.output_layer.weight.grad, model_eager.output_layer.weight.grad, atol=1e-5, rtol=1e-5
    )


def _tp2_megatron_worker(rank: int, world_size: int, port: int, state_dict: dict, queue):
    _, mff = _import_mcore_modules()
    from verl.models.mcore import util as mcore_util

    dist.init_process_group(backend="gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size)
    try:
        mff.parallel_state.get_tensor_model_parallel_group = lambda: dist.group.WORLD
        mff.parallel_state.get_tensor_model_parallel_world_size = lambda: world_size
        mff.parallel_state.get_tensor_model_parallel_rank = lambda: rank
        mcore_util.mpu.get_tensor_model_parallel_world_size = lambda: world_size
        sys.modules["megatron.core.parallel_state"].get_tensor_model_parallel_group = lambda: dist.group.WORLD
        sys.modules["megatron.core.parallel_state"].get_tensor_model_parallel_world_size = lambda: world_size

        vocab_size, hidden_dim = 24, 12
        local_vocab = vocab_size // world_size
        full_out_weight = state_dict["output_layer.weight"]
        shard_weight = torch.nn.Parameter(full_out_weight[rank * local_vocab : (rank + 1) * local_vocab].clone())

        model = _TinyMegatronHookModel(vocab_size, hidden_dim, weight_shard=shard_weight)
        model.embed.weight.data.copy_(state_dict["embed.weight"])

        batch = _build_ragged_micro_batch(vocab_size=vocab_size, k=4, support_complete=False)
        loss_fn = partial(score_centering_ppo_loss, config=_megatron_actor_config("token", 2.0))
        out = mff.fused_forward_model_engine(vision_model=False)(
            model=model,
            input_ids=batch["input_ids"],
            labels=batch["input_ids"].clone(),
            multi_modal_inputs={},
            temperature=0.8,
            calculate_entropy=True,
            pad_token_id=0,
            score_centering=True,
            logits_processor_func=loss_fn,
            batch=batch,
        )
        loss, _ = loss_fn(model_output=out, data=batch, dp_group=None)
        loss.backward()
        queue.put((rank, float(loss.detach()), model.embed.weight.grad.tolist(), shard_weight.grad.tolist()))
    finally:
        dist.destroy_process_group()


def test_megatron_fused_score_centering_tp2_vocab_parallel_matches_tp1():
    _, mff = _import_mcore_modules()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    vocab_size, hidden_dim = 24, 12
    torch.manual_seed(53)
    ref_model = _TinyMegatronHookModel(vocab_size, hidden_dim)
    state_dict = {k: v.detach().clone() for k, v in ref_model.state_dict().items()}

    batch = _build_ragged_micro_batch(vocab_size=vocab_size, k=4, support_complete=False)
    loss_fn = partial(score_centering_ppo_loss, config=_megatron_actor_config("token", 2.0))
    out_tp1 = mff.fused_forward_model_engine(vision_model=False)(
        model=ref_model,
        input_ids=batch["input_ids"],
        labels=batch["input_ids"].clone(),
        multi_modal_inputs={},
        temperature=0.8,
        calculate_entropy=True,
        pad_token_id=0,
        score_centering=True,
        logits_processor_func=loss_fn,
        batch=batch,
    )
    loss_tp1, _ = loss_fn(model_output=out_tp1, data=batch, dp_group=None)
    loss_tp1.backward()

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    mp.spawn(_tp2_megatron_worker, args=(2, port, state_dict, queue), nprocs=2, join=True)
    items = sorted((queue.get(timeout=5) for _ in range(2)), key=lambda x: x[0])

    for _, loss_tp2, embed_grad_tp2, _ in items:
        torch.testing.assert_close(torch.tensor(loss_tp2), loss_tp1.detach(), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(torch.tensor(embed_grad_tp2), ref_model.embed.weight.grad, atol=1e-5, rtol=1e-5)
    out_grad_tp2 = torch.cat([torch.tensor(item[3]) for item in items], dim=0)
    torch.testing.assert_close(out_grad_tp2, ref_model.output_layer.weight.grad, atol=1e-5, rtol=1e-5)


def test_gather_fused_hidden_states_disables_tp_grad_reduce_when_has_topk(monkeypatch):
    _, mff = _import_mcore_modules()
    seen = []

    def fake_gather(hidden, tensor_parallel_output_grad=True):
        seen.append(tensor_parallel_output_grad)
        return hidden

    monkeypatch.setattr(mff, "gather_from_sequence_parallel_region", fake_gather)
    hidden = torch.randn(4, 8)
    mff._gather_fused_hidden_states(hidden, sequence_parallel=True, impl_backend="triton", has_topk=False)
    mff._gather_fused_hidden_states(hidden, sequence_parallel=True, impl_backend="triton", has_topk=True)
    assert seen == [True, False]
