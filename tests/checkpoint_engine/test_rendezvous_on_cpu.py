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

"""Exercise the actual worker-owned TCPStore with separate CPU Ray actors."""

import asyncio
import os
import socket
from datetime import timedelta
from types import SimpleNamespace

import pytest
import ray
import torch

from verl.checkpoint_engine.base import CheckpointEngineWorker
from verl.utils.distributed import initialize_global_process_group_ray
from verl.workers.rollout import replica as replica_module
from verl.workers.rollout.replica import RolloutReplica


class _CPUWorker(CheckpointEngineWorker):
    def __init__(self, rank, master_port, **kwargs):
        os.environ.update(
            RANK=str(rank),
            WORLD_SIZE="2",
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(master_port),
            LOCAL_RANK="0",
            LOCAL_WORLD_SIZE="2",
        )
        # The real constructor and rendezvous code run unchanged. Only the GPU
        # model adapter is replaced, since this test concerns its CPU group.
        super().__init__(server_adapter=object(), **kwargs)

    def check_collective(self):
        value = torch.tensor([self._rank + 1])
        torch.distributed.all_reduce(value)
        return {
            "sum": value.item(),
            "rank": torch.distributed.get_rank(),
            "world_size": torch.distributed.get_world_size(),
            "store_port": self._process_group_store.port,
            "store_timeout": self._process_group_store.timeout,
        }


class _CPUReplica(RolloutReplica):
    async def launch_servers(self):
        # Successful collectives here prove every rank was initialized before
        # the existing server-launch stage is entered.
        self.results = await asyncio.gather(*[worker.check_collective.remote() for worker in self.workers])


@pytest.fixture(scope="module")
def local_ray(tmp_path_factory):
    # Explicit local address must ignore an operator's RAY_ADDRESS and must
    # never attach the CPU test to an existing training cluster.
    ray.init(
        address="local",
        num_cpus=4,
        include_dashboard=False,
        object_store_memory=128 << 20,
        _temp_dir=str(tmp_path_factory.mktemp("ray")),
        _node_ip_address="127.0.0.1",
    )
    yield
    ray.shutdown()


def _config():
    return SimpleNamespace(
        name="cpu-test",
        checkpoint_engine=SimpleNamespace(
            backend="naive", update_weights_bucket_megabytes=1, engine_kwargs={}, custom_backend_module=None
        ),
    )


def _replica(rank):
    replica = _CPUReplica.__new__(_CPUReplica)
    replica.replica_rank = rank
    replica.config = _config()
    replica.model_config = None
    replica.is_reward_model = False
    replica.is_teacher_model = False
    replica.name_suffix = ""
    return replica


def test_parallel_replicas_ignore_occupied_old_master_port(local_ray, monkeypatch):
    """A competing bind after free-port selection cannot break the new path."""
    actors = []
    with socket.socket() as occupied:
        occupied.bind(("", 0))
        occupied.listen()
        old_port = occupied.getsockname()[1]
        # Negative control: this is the actual old env:// rendezvous failure.
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
        monkeypatch.setenv("MASTER_PORT", str(old_port))
        with pytest.raises(torch.distributed.DistNetworkError, match="EADDRINUSE|address already in use"):
            initialize_global_process_group_ray(timeout_second=2, backend="cpu:gloo")

        def create_workers(**kwargs):
            init = kwargs["ray_cls_with_init"]
            assert init.kwargs["defer_process_group_init"] is True
            workers = [
                ray.remote(_CPUWorker).options(num_cpus=0.5).remote(rank, old_port, **init.kwargs) for rank in range(2)
            ]
            actors.extend(workers)
            return SimpleNamespace(workers=workers)

        monkeypatch.setattr(replica_module, "RayWorkerGroup", create_workers)
        standalone, colocated = _replica(0), _replica(1)

        async def initialize():
            await asyncio.gather(standalone.init_standalone(object()), colocated.init_colocated(object()))

        try:
            asyncio.run(asyncio.wait_for(initialize(), timeout=90))
            ports = []
            for replica in (standalone, colocated):
                assert [result["rank"] for result in replica.results] == [0, 1]
                assert all(result["sum"] == 3 and result["world_size"] == 2 for result in replica.results)
                assert all(result["store_timeout"] == timedelta(minutes=30) for result in replica.results)
                port = replica.results[0]["store_port"]
                assert all(result["store_port"] == port for result in replica.results)
                assert port != old_port
                # Store remains bound after bootstrap and the first collective.
                with socket.socket() as competitor:
                    with pytest.raises(OSError, match="Address already in use"):
                        competitor.bind(("", port))
                ports.append(port)
            assert len(set(ports)) == 2
            # A later collective verifies rank zero did not drop its store at
            # the return from the prepare RPC or replica initialization.
            later = ray.get([worker.check_collective.remote() for worker in actors], timeout=20)
            assert all(result["sum"] == 3 for result in later)
        finally:
            for actor in actors:
                ray.kill(actor)


def test_direct_constructor_keeps_process_group_initialization(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "0")
    try:
        worker = CheckpointEngineWorker(_config(), None, server_adapter=object())
        assert torch.distributed.is_initialized()
        assert torch.distributed.get_world_size() == 1
        assert worker._process_group_store is None
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
