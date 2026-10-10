# Recover a standalone vLLM replica

This example runs one real optimizer update, kills its rollout replica's
EngineCore, and restores serving through verl's normal weight synchronization.
It creates a tiny Qwen2 model locally and needs no dataset or model download.

Run from the verl repository root on one node with three available CUDA GPUs:

```bash
uv run --extra vllm --extra cupy-cu130 python examples/fault_tolerance/restart_vllm_replica.py
```

The `cupy-cu130` extra supplies the NCCL transport dependency; the example's
trainer is a small HF/AdamW worker. Prepare this environment once to avoid repeating cold
dependency installation. The selected vLLM version and its CUDA/driver
requirements come from verl's dependency configuration.

## What happens

One GPU holds the trainer. Two GPUs hold a standalone vLLM MP replica with TP=2;
its checkpoint-engine receivers share those two rollout GPUs.

1. Create the seeded model and publish version 0 through `update_weights(0)`.
2. Perform one real AdamW update, publish version 1, and generate eight tokens.
3. Signal only this server's owned EngineCore with `SIGKILL`. Its native health
   check must raise `EngineDeadError`.
4. Call `restart_replica(replica)`. The replacement stays paused while the
   original checkpoint receivers and resource pool are retained and rebound.
5. Call the normal `update_weights(1)` again. Compare the restored MP weights
   with their pre-fault fingerprints, then generate with version 1.
6. Finalize all owned NCCL ranks together, stop the server, and release this
   example's actors and placement group.

The essential recovery calls are:

```python
await checkpoint_manager.restart_replica(replica)
await checkpoint_manager.update_weights(global_steps=1)
```

The script makes sequential requests to its own replica, so it has no concurrent
traffic to fence. An application must fence routing before recovery, choose a
trainer-safe GPU/optimizer boundary for synchronization, and publish the new
server handle and HTTP address afterward. Recovery does not require another
optimizer update. This example invokes recovery explicitly after its injected
fault; it does not implement a failure detector or a training controller.

## Scope and validation

This demonstrates a single-node standalone MP replica, full named-tensor NCCL
updates, and a small synthetic workload. Colocated restart, PD, unmerged LoRA,
multi-node recovery and requests interrupted mid-generation are outside its
scope. Successful generation is a mechanism check, not a PPO or model-quality
result.

Engine health checks also work with the default `disable_log_stats=True`;
scheduler metric snapshots are separate from recovery.

NCCL uses `rebuild_group=True`: each normal synchronization finalizes its
transport group through the native lifecycle. The checkpoint receiver actors
and their GPU reservations remain in place across serving-engine replacement.
If initialization or synchronization fails, final cleanup first checks which
groups are initialized and dispatches their finalizers concurrently. It only
releases resources created by this example.

The fault-injection and fingerprint helpers use vLLM's EngineManager and MP
inspection interfaces. These implementation details are confined to the
example; application recovery uses the public checkpoint-manager methods.

The script writes `result.json` to a new local output directory and prints the
optimizer, injected failure and recovery stages. A successful result includes
weight equality and generation checks, followed by successful owned-resource
cleanup. Use `--output-dir /path/to/new-directory` to keep these artifacts.
