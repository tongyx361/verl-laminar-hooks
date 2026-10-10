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
"""Recover one standalone TP2 vLLM replica using normal verl weight synchronization.

Requires three CUDA GPUs on one node and a prepared vLLM/NCCL environment.
Creates a local Ray cluster and a tiny model without downloading a checkpoint.
"""

import argparse
import asyncio
import hashlib
import json
import math
import os
import tempfile
import time
import uuid
from pathlib import Path


def create_tiny_model(path):
    """Write a seeded Qwen2 and tokenizer small enough for this demonstration."""
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(42)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=128,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=128,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=False,
        )
    ).to(torch.bfloat16)
    model.save_pretrained(path)
    vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3}
    vocab.update({f"token{i}": i for i in range(4, 128)})
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]"
    )
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] + ' ' }}{% endfor %}"
    tokenizer.save_pretrained(path)


def fingerprints(model):
    """Compare exact parameter values; these CPU copies are never recovery inputs."""
    import torch

    result = {}
    for name, parameter in model.named_parameters():
        value = parameter.detach().cpu().contiguous()
        assert torch.isfinite(value).all(), f"non-finite parameter: {name}"
        digest = hashlib.sha256(value.view(torch.uint8).numpy()).hexdigest()
        result[name] = (list(value.shape), str(value.dtype), digest)
    return result


def mp_fingerprints(worker):
    """Run on each real vLLM MP worker to inspect its loaded model shard."""
    return worker.rank, fingerprints(worker.model_runner.model)


def finalize_if_initialized(worker):
    """Avoid repeating NCCL finalize after the normal sync already finalized it."""
    from ray.util import collective

    engine = worker.checkpoint_engine
    if collective.is_group_initialized(engine.group_name):
        engine.finalize()


def make_actor_classes():
    """Keep the toy trainer and fault inspection separate from production classes."""
    import psutil
    import ray
    import torch
    from transformers import AutoModelForCausalLM

    from verl.checkpoint_engine import CheckpointEngineRegistry
    from verl.single_controller.base import Worker
    from verl.single_controller.base.decorator import Dispatch, register
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica

    class Trainer(Worker):
        """One real AdamW update and the unchanged verl NCCL weight sender."""

        def __init__(self, model_path, config):
            super().__init__()
            self.device = torch.device("cuda", torch.cuda.current_device())
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, dtype=torch.float32, attn_implementation="eager", local_files_only=True
            ).to(self.device)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
            self.checkpoint_engine = CheckpointEngineRegistry.new(
                "nccl",
                bucket_size=config.update_weights_bucket_megabytes << 20,
                is_master=True,
                **config.engine_kwargs["nccl"],
            )

        @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
        def execute_checkpoint_engine(self, method, **kwargs):
            return getattr(self.checkpoint_engine, method)(**kwargs)

        @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
        async def update_weights(self, global_steps=None, mode="auto"):
            weights = ((name, p.detach().to(torch.bfloat16).contiguous()) for name, p in self.model.named_parameters())
            await self.checkpoint_engine.send_weights(weights, global_steps=global_steps)
            return {}

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def optimizer_step(self):
            before = fingerprints(self.model)
            self.model.train()
            inputs = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]], device=self.device)
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.model(input_ids=inputs, labels=inputs).loss
            assert torch.isfinite(loss), "non-finite loss"
            loss.backward()
            norm = torch.linalg.vector_norm(
                torch.stack(
                    [torch.linalg.vector_norm(p.grad.float()) for p in self.model.parameters() if p.grad is not None]
                )
            )
            assert torch.isfinite(norm) and norm > 0, "gradient must be finite and nonzero"
            self.optimizer.step()
            assert before != fingerprints(self.model), "AdamW did not change weights"
            return {"loss": float(loss.detach()), "grad_norm": float(norm)}

    class FaultServer(vLLMHttpServer):
        """Test-owned inspection and one precisely targeted EngineCore failure."""

        def __init__(self, *args, **kwargs):
            # Only this local demo accepts the trusted MP inspection callable.
            os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
            super().__init__(*args, **kwargs)

        async def inspect(self, weights=True):
            result = {"paused": self._submission_paused, "rejecting": self._rejecting, "version": self.global_steps}
            if weights:
                result["weights"] = dict(await self.engine.collective_rpc(mp_fingerprints, timeout=60))
            return result

        def inject_core_failure(self):
            # vLLM's private process list is isolated here, only for fault injection.
            cores = self.engine.engine_core.resources.engine_manager.processes
            if len(cores) != 1 or not cores[0].is_alive():
                raise RuntimeError("expected one live local EngineCore")
            process = psutil.Process(cores[0].pid)
            if process not in self._engine_cleanup.record():
                raise RuntimeError("EngineCore is not owned by this server")
            identity = process.pid, process.create_time()
            if psutil.Process(identity[0]).create_time() != identity[1]:
                raise RuntimeError("EngineCore identity changed")
            process.kill()  # psutil also checks creation time before signaling a reused PID.
            return identity

    class FaultReplica(vLLMReplica):
        """Use the inspection server with the unchanged native replica lifecycle."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.server_class = ray.remote(FaultServer)

    return Trainer, FaultReplica


async def generate(server, version, name):
    """Check generation only after the expected weight version is ready."""
    output = await server.generate.remote(
        prompt_ids=[4, 5, 6],
        sampling_params={"temperature": 0, "max_tokens": 8, "ignore_eos": True, "logprobs": True},
        request_id=name,
    )
    assert len(output.token_ids) == len(output.log_probs) == 8
    assert all(math.isfinite(value) for value in output.log_probs)
    assert output.extra_fields["global_steps"] == version and output.stop_reason != "aborted"


async def cleanup(replica, trainer, pool):
    """Finalize every owned rank before killing actors or releasing their GPUs."""
    import ray
    from ray.util.placement_group import remove_placement_group

    workers = ([] if trainer is None else trainer.workers) + ([] if replica is None else replica.workers)
    errors = []
    try:
        # Dispatch every rank before waiting: NCCL destroy is collective.
        results = await asyncio.gather(
            *[asyncio.wait_for(worker.__ray_call__.remote(finalize_if_initialized), timeout=30) for worker in workers],
            return_exceptions=True,
        )
        errors.extend(result for result in results if isinstance(result, BaseException))
        for server in [] if replica is None else replica.servers:
            try:
                # Native stop verifies engine children, Ray death and HTTP OS exit.
                # The retained CE process is the witness, so it must stay alive.
                await asyncio.wait_for(replica._stop_server_actor(server), timeout=60)
            except Exception as exc:
                errors.append(exc)
            finally:
                try:
                    ray.kill(server, no_restart=True)
                except Exception as exc:
                    errors.append(exc)
    finally:
        for worker in workers:
            try:
                ray.kill(worker, no_restart=True)
            except Exception as exc:
                errors.append(exc)
        for pg in pool.pgs or []:
            try:
                remove_placement_group(pg)
            except Exception as exc:
                errors.append(exc)
    if errors:
        raise RuntimeError(f"owned resource cleanup failed: {errors}") from errors[0]


async def run(model_path):
    """Demonstrate optimizer publication, Core failure, restart and normal resync."""
    import ray
    from vllm.v1.engine.exceptions import EngineDeadError

    from verl.checkpoint_engine import CheckpointEngineManager
    from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
    from verl.single_controller.ray.base import split_resource_pool
    from verl.workers.config import CheckpointEngineConfig, HFModelConfig, RolloutConfig

    name = f"recovery-{uuid.uuid4().hex[:12]}"
    checkpoint = CheckpointEngineConfig(
        backend="nccl",
        update_weights_bucket_megabytes=8,
        engine_kwargs={"nccl": {"group_name": name, "rebuild_group": True, "multi_sender": True}},
    )
    rollout = RolloutConfig(
        name="vllm",
        checkpoint_engine=checkpoint,
        tensor_model_parallel_size=2,
        prompt_length=32,
        response_length=8,
        max_model_len=128,
        max_num_seqs=4,
        max_num_batched_tokens=128,
        gpu_memory_utilization=0.2,
        standalone_gpu_memory_utilization=0.2,
        enforce_eager=True,
        load_format="auto",
        engine_kwargs={"vllm": {"attention_backend": "TRITON_ATTN", "disable_custom_all_reduce": True}},
    )
    pool = RayResourcePool(process_on_nodes=[3], max_colocate_count=2, name_prefix=name)
    trainer = replica = None
    trainer_class, replica_class = make_actor_classes()
    try:
        trainer_pool, rollout_pool = split_resource_pool(pool, [1, 2])
        trainer = RayWorkerGroup(
            resource_pool=trainer_pool,
            ray_cls_with_init=RayClassWithInitArgs(
                cls=ray.remote(trainer_class), model_path=str(model_path), config=checkpoint
            ),
            name_prefix=f"{name}-trainer-",
            device_name="cuda",
        )
        replica = replica_class(
            replica_rank=0,
            config=rollout,
            model_config=HFModelConfig(path=str(model_path)),
            gpus_per_node=2,
            name_suffix=name,
        )
        await replica.init_standalone(resource_pool=rollout_pool)
        await asyncio.gather(*[worker.bind_server_handle.remote(replica.server_handle) for worker in replica.workers])
        manager = CheckpointEngineManager(checkpoint, trainer, [replica])
        await manager.update_weights(global_steps=0)
        baseline = await replica.server_handle.inspect.remote()
        await generate(replica.server_handle, 0, f"{name}-v0")
        optimizer = trainer.optimizer_step()[0]
        await manager.update_weights(global_steps=1)
        before = await replica.server_handle.inspect.remote()
        assert set(before["weights"]) == {0, 1}
        assert all(before["weights"][rank] != baseline["weights"][rank] for rank in (0, 1))
        await generate(replica.server_handle, 1, f"{name}-v1")
        print(json.dumps({"stage": "optimizer_and_full_sync", "version": 1, **optimizer}), flush=True)

        old_server, workers, resources = replica.server_handle, replica.workers, replica.resource_pool
        identity = await old_server.inject_core_failure.remote()
        deadline = time.monotonic() + 30
        while True:
            try:
                await replica.snapshot()
            except ray.exceptions.RayTaskError as exc:
                if not isinstance(exc.cause, EngineDeadError):
                    raise
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("EngineCore failure did not reach native health check")
            await asyncio.sleep(0.1)
        await manager.restart_replica(replica)
        staged = await replica.server_handle.inspect.remote(weights=False)
        assert staged["paused"] and staged["rejecting"]
        assert replica.server_handle._actor_id != old_server._actor_id
        assert replica.workers is workers and replica.resource_pool is resources
        adapters = await asyncio.gather(
            *[
                worker.__ray_call__.remote(lambda worker: worker.server_adapter.server_handle._actor_id)
                for worker in workers
            ]
        )
        assert all(handle == replica.server_handle._actor_id for handle in adapters)

        # No backup or new channel: the trainer reuses the normal full sync.
        await manager.update_weights(global_steps=1)
        restored = await replica.server_handle.inspect.remote()
        assert restored["weights"] == before["weights"] and restored["version"] == 1
        assert not restored["paused"] and not restored["rejecting"]
        await generate(replica.server_handle, 1, f"{name}-recovered")
        print(json.dumps({"stage": "recovered", "killed_core": identity, "version": 1}), flush=True)
    finally:
        await cleanup(replica, trainer, pool)


def main():
    """Parse help without importing GPU libraries; own only a fresh local Ray cluster."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="New directory for the tiny model and result.json")
    args = parser.parse_args()
    import ray
    import torch

    assert torch.cuda.is_available() and torch.cuda.device_count() >= 3, "three local CUDA GPUs are required"
    output = args.output_dir.resolve() if args.output_dir else Path(tempfile.mkdtemp(prefix="verl-recovery-"))
    if args.output_dir:
        output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {"status": "FAIL", "output_dir": str(output)}
    print(json.dumps({"stage": "setup", "output_dir": str(output)}), flush=True)
    try:
        create_tiny_model(output / "tiny-qwen2")
        ray.init(address="local", num_gpus=3, include_dashboard=False, object_store_memory=256 * 1024**2)
        asyncio.run(run(output / "tiny-qwen2"))
        result["status"] = "PASS"
    finally:
        ray.shutdown()  # This driver started the local cluster; never invoke global ray stop.
        result["seconds"] = time.monotonic() - started
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
