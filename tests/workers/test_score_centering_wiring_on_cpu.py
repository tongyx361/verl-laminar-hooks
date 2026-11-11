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

from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
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
