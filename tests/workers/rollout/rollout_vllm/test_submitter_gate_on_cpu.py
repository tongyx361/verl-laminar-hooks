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
"""GPU-free tests for the vLLMHttpServer submission gate.

vLLM's pause stops requests being scheduled but still accepts them. A request
admitted between abort_all_requests() and resume_generation() is parked in the
scheduler's waiting queue and masked out of the drain's liveness check, so
wait_for_requests_to_drain() cannot return. These tests pin the ordering that
makes such an admission impossible.

They also pin abort_all_requests(reject_request=True), which fails late arrivals
instead of parking them when the server is leaving the load balancer and no
resume_generation() is coming soon.

Finally, they pin where generate() runs the request hooks relative to the gate:
_preprocess_sampling_params before admission, _postprocess_output after release.
"""

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("ray")
pytest.importorskip("vllm")

from vllm.logprobs import Logprob

from verl.workers.config import RolloutConfig
from verl.workers.rollout.vllm_rollout import vllm_async_server


class _FakeEngine:
    """Records the state of the gate at the moment the engine is paused."""

    def __init__(self):
        self.output_processor = SimpleNamespace(request_states={}, parent_requests={})
        self.server = None
        self.pause_calls = 0
        self.resume_calls = 0
        self.admitting_at_pause = None
        self.abort_calls = []
        self.drain_calls = 0
        self.reset_prefix_calls = 0
        self.outputs = []
        self.sampling_params = []
        self.health_failure = None

    async def generate(self, prompt, sampling_params, request_id, lora_request=None, priority=0):
        self.sampling_params.append(sampling_params)
        for output in self.outputs:
            yield output

    async def pause_generation(self, **kwargs):
        self.pause_calls += 1
        self.admitting_at_pause = self.server._admitting

    async def resume_generation(self):
        self.resume_calls += 1

    async def abort(self, request_ids, internal=True):
        self.abort_calls.append(list(request_ids))

    async def wait_for_requests_to_drain(self):
        self.drain_calls += 1

    async def reset_prefix_cache(self, reset_connector=True):
        self.reset_prefix_calls += 1

    async def check_health(self):
        if self.health_failure is not None:
            raise self.health_failure


def _make_server(node_rank: int = 0, cls=vllm_async_server.vLLMHttpServer):
    server = object.__new__(cls)
    server.node_rank = node_rank
    server.global_steps = 7
    server.engine = _FakeEngine()
    server.engine.server = server
    server._submission_paused = False
    server._admitting = 0
    server._resume_event = asyncio.Event()
    server._resume_event.set()
    server._rejecting = False
    server._disaggregation_role = "null"
    server._engine_cleanup = vllm_async_server._LegacyMPEngineCleanup()
    return server


def test_abort_does_not_pause_until_inflight_admissions_land():
    async def main():
        server = _make_server()
        server._admitting = 1  # a turn is past the gate but not yet in the engine

        abort = asyncio.create_task(server.abort_all_requests())
        await asyncio.sleep(0.05)

        assert server._submission_paused is True, "gate must close before the barrier runs"
        assert not abort.done(), "abort must not proceed while an admission is in flight"
        assert server.engine.pause_calls == 0, "engine paused while an admission was in flight"

        server._admitting = 0  # the in-flight admission reaches the engine
        await asyncio.wait_for(abort, timeout=5)

        assert server.engine.pause_calls == 1
        assert server.engine.admitting_at_pause == 0

    asyncio.run(main())


def test_submission_parks_while_gate_closed_and_wakes_on_resume():
    async def main():
        server = _make_server()
        await server.abort_all_requests()
        assert server._submission_paused is True

        task = asyncio.create_task(server._park_until_admitted("r1"))
        await asyncio.sleep(0.05)
        assert not task.done(), "submission must park while the gate is closed"
        assert server._admitting == 0

        await server.resume_generation()
        assert await asyncio.wait_for(task, timeout=5) is None
        assert server._admitting == 1

    asyncio.run(main())


def test_reject_request_fails_late_arrivals_instead_of_parking():
    async def main():
        server = _make_server()
        await server.abort_all_requests(reject_request=True)

        output = await asyncio.wait_for(server._park_until_admitted("late"), timeout=5)

        assert output.stop_reason == "aborted", "a rejecting gate must fail over, not park"
        assert output.token_ids == []
        assert output.extra_fields["global_steps"] == 7
        assert server._admitting == 0, "rejected requests never count as admissions"
        assert server._submission_paused is True, "the gate stays closed until resume_generation"

    asyncio.run(main())


def test_weight_sync_abort_restores_parking_after_a_rejecting_abort():
    # switch_to_trainer aborts with reject_request=True; the weight sync inside the following
    # switch_to_rollout aborts again with the default, and by then a resume is imminent, so
    # requests must go back to parking rather than being failed over.
    async def main():
        server = _make_server()
        await server.abort_all_requests(reject_request=True)
        assert server._rejecting is True

        await server.abort_all_requests()
        assert server._rejecting is False

        task = asyncio.create_task(server._park_until_admitted("r1"))
        await asyncio.sleep(0.05)
        assert not task.done(), "a plain abort must restore parking"

        await server.resume_generation()
        assert await asyncio.wait_for(task, timeout=5) is None

    asyncio.run(main())


def test_resume_clears_rejection():
    async def main():
        server = _make_server()
        await server.abort_all_requests(reject_request=True)

        await server.resume_generation()

        assert server._rejecting is False
        assert server._submission_paused is False
        assert await server._park_until_admitted("r1") is None
        assert server._admitting == 1

    asyncio.run(main())


def test_resume_reopens_gate_on_non_head_server():
    async def main():
        server = _make_server(node_rank=1)
        server._submission_paused = True
        server._resume_event.clear()

        await server.resume_generation()

        assert server._submission_paused is False, "non-head server stays gated forever"
        assert server._resume_event.is_set()
        assert server.engine.resume_calls == 0, "only node rank 0 drives the engine"

    asyncio.run(main())


def test_resume_on_head_server_also_resumes_engine():
    async def main():
        server = _make_server(node_rank=0)
        await server.abort_all_requests()

        await server.resume_generation()

        assert server._submission_paused is False
        assert server._resume_event.is_set()
        assert server.engine.resume_calls == 1

    asyncio.run(main())


def test_barrier_times_out_instead_of_hanging(monkeypatch):
    # raising=False: this asserts the barrier cannot deadlock, not that the constant exists.
    monkeypatch.setattr(vllm_async_server, "_GATE_BARRIER_TIMEOUT_S", 0.05, raising=False)

    async def main():
        server = _make_server()
        server._admitting = 1  # never clears

        await asyncio.wait_for(server.abort_all_requests(), timeout=5)

        assert server.engine.pause_calls == 1, "barrier must proceed rather than deadlock"

    asyncio.run(main())


def test_abort_all_requests_abort_only_leaves_admission_open():
    async def main():
        server = _make_server()
        server.engine.output_processor.request_states = {"r1": object(), "r2": object()}

        # Default reset_prefix_cache=True must not clear caches on the abort-only path.
        result = await server.abort_all_requests(abort_only=True)

        assert server._submission_paused is False
        assert server.engine.pause_calls == 0
        assert server.engine.abort_calls == [["r1", "r2"]]
        assert server.engine.drain_calls == 0
        assert server.engine.reset_prefix_calls == 0
        assert result["aborted_count"] == 2
        assert result["request_ids"] == ["r1", "r2"]

    asyncio.run(main())


def test_abort_all_requests_abort_only_releases_parallel_sampling_parents():
    """n>1 parents live outside request_states and must be aborted after children."""

    async def main():
        server = _make_server()
        server.engine.output_processor.request_states = {"0_p": object(), "1_p": object()}
        server.engine.output_processor.parent_requests = {"p": object()}

        result = await server.abort_all_requests(abort_only=True)

        assert server.engine.abort_calls == [["0_p", "1_p", "p"]]
        assert result["aborted_count"] == 2
        assert result["request_ids"] == ["0_p", "1_p"]
        assert server._submission_paused is False
        assert server.engine.pause_calls == 0

    asyncio.run(main())


def test_snapshot_rejects_pd_disaggregation():
    async def main():
        server = _make_server()
        server._disaggregation_role = "prefill"
        with pytest.raises(NotImplementedError, match="does not support PD disaggregation"):
            await server.snapshot()

    asyncio.run(main())


def test_snapshot_rejects_headless_node_without_touching_engine():
    async def main():
        server = _make_server(node_rank=1)
        del server.engine
        with pytest.raises(RuntimeError, match="requires the node-rank-0 AsyncLLM"):
            await server.snapshot()

    asyncio.run(main())


def test_snapshot_propagates_engine_failure_before_reading_stale_gauges():
    server = _make_server()
    failure = RuntimeError("engine core failed")
    server.engine.health_failure = failure
    # No logger is installed: a stale observation must never be read.
    with pytest.raises(RuntimeError, match="engine core failed") as caught:
        asyncio.run(server.snapshot())
    assert caught.value is failure


class _Process:
    def __init__(self, name, events, children=()):
        self.name = name
        self.events = events
        self.descendants = list(children)
        self.pid = 1234

    def create_time(self):
        return 1.0

    def children(self, recursive=True):
        assert recursive
        return self.descendants

    def terminate(self):
        self.events.append(("terminate", self.name))

    def kill(self):
        self.events.append(("kill", self.name))


def test_healthy_snapshot_retains_engine_descendants_for_orphan_cleanup(monkeypatch):
    events = []
    engine_child = _Process("engine", events)
    actor = _Process("actor", events)
    monkeypatch.setattr(vllm_async_server.psutil, "Process", lambda: actor)
    server = _make_server()
    actor.descendants = [engine_child]

    def gauge(name, value):
        sample = SimpleNamespace(name=name, value=value)
        return SimpleNamespace(collect=lambda: [SimpleNamespace(samples=[sample])])

    logger = object.__new__(vllm_async_server.PrometheusStatLogger)
    logger.gauge_kv_cache_usage = {0: gauge("vllm:kv_cache_usage_perc", 0.25)}
    logger.gauge_scheduler_waiting = {0: gauge("vllm:num_requests_waiting", 1)}
    logger.gauge_scheduler_running = {0: gauge("vllm:num_requests_running", 2)}
    server.engine.logger_manager = SimpleNamespace(prometheus_logger=logger)
    assert asyncio.run(server.snapshot()) == {
        "kv_cache_usage": 0.25,
        "num_waiting_requests": 1,
        "num_running_requests": 2,
    }
    # The core died and its worker was reparented before shutdown enumerated children.
    actor.descendants = []
    assert server._engine_cleanup.record() == [engine_child]


def _make_shutdown_server():
    server = _make_server()
    server.rollout_mode = vllm_async_server.RolloutMode.STANDALONE
    server.nnodes = 1
    server.engine.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(distributed_executor_backend="mp"))
    return server


@pytest.mark.parametrize("remaining", [False, True])
def test_shutdown_reaps_only_owned_processes_and_refuses_survivors(monkeypatch, remaining):
    events = []
    orphan = _Process("orphan", events)
    unrelated_child = _Process("unrelated_child", events)
    unrelated = _Process("unrelated", events)
    actor = _Process("actor", events, children=[unrelated])
    monkeypatch.setattr(vllm_async_server.psutil, "Process", lambda: actor)
    server = _make_shutdown_server()
    unrelated.descendants = [unrelated_child]
    actor.descendants = [unrelated, unrelated_child, orphan]
    server._engine_cleanup.record()
    actor.descendants = [unrelated, unrelated_child]
    server.engine.shutdown = lambda: events.append("shutdown")
    waits = 0

    def wait_procs(processes, timeout):
        nonlocal waits
        assert processes == [orphan], "a pre-existing child or its descendants must not be signaled"
        assert timeout == 5
        waits += 1
        return ([], [orphan]) if waits < 3 or remaining else ([orphan], [])

    monkeypatch.setattr(vllm_async_server.psutil, "wait_procs", wait_procs)
    if remaining:
        with pytest.raises(RuntimeError, match="owned engine processes alive"):
            asyncio.run(server.shutdown())
    else:
        asyncio.run(server.shutdown())
    assert events == ["shutdown", ("terminate", "orphan"), ("kill", "orphan")]
    assert server._rejecting and server._submission_paused and server._resume_event.is_set()


def test_shutdown_reaps_orphans_before_propagating_engine_shutdown_failure(monkeypatch):
    events = []
    orphan = _Process("orphan", events)
    actor = _Process("actor", events)
    monkeypatch.setattr(vllm_async_server.psutil, "Process", lambda: actor)
    server = _make_shutdown_server()
    actor.descendants = [orphan]
    server._engine_cleanup.record()
    actor.descendants = []
    failure = RuntimeError("engine cleanup failed")

    def shutdown():
        events.append("shutdown")
        raise failure

    server.engine.shutdown = shutdown
    waits = 0

    def wait_procs(processes, timeout):
        nonlocal waits
        assert processes == [orphan]
        waits += 1
        return ([], [orphan]) if waits == 1 else ([orphan], [])

    monkeypatch.setattr(vllm_async_server.psutil, "wait_procs", wait_procs)
    with pytest.raises(RuntimeError, match="engine cleanup failed") as caught:
        asyncio.run(server.shutdown())
    assert caught.value is failure
    assert events == ["shutdown", ("terminate", "orphan")]


@pytest.mark.parametrize(
    "field,value",
    [
        ("rollout_mode", vllm_async_server.RolloutMode.COLOCATED),
        ("nnodes", 2),
        ("node_rank", 1),
        ("_disaggregation_role", "prefill"),
        ("executor", "ray"),
    ],
)
def test_shutdown_rejects_unsupported_engine_before_closing_admission(field, value):
    server = _make_shutdown_server()
    if field == "executor":
        server.engine.vllm_config.parallel_config.distributed_executor_backend = value
    else:
        setattr(server, field, value)
    with pytest.raises(NotImplementedError):
        asyncio.run(server.shutdown())
    assert not server._submission_paused and not server._rejecting


def _make_restart_replica(monkeypatch, shutdown, probe, events, worker_probe=None, process_identity=(1234, 1.0)):
    async def shutdown_with_identity():
        await shutdown()
        return process_identity

    old = SimpleNamespace(
        shutdown=SimpleNamespace(remote=shutdown_with_identity),
        get_server_address=SimpleNamespace(remote=probe),
    )
    replica = object.__new__(vllm_async_server.vLLMReplica)
    replica.rollout_mode = vllm_async_server.RolloutMode.STANDALONE
    replica.nnodes = 1
    replica.config = SimpleNamespace(disaggregation=SimpleNamespace(enabled=False))
    replica.servers = [old]
    replica._server_handle = old
    replica._server_address = "old-address"

    async def probe_process(fn, pid, create_time):
        events.append("identity")
        if worker_probe is not None:
            return await worker_probe(fn, pid, create_time)
        return False

    workers = [SimpleNamespace(__ray_call__=SimpleNamespace(remote=probe_process))]
    resource_pool = object()
    replica.workers, replica.resource_pool = workers, resource_pool

    async def launch_servers():
        assert replica.workers is workers and replica.resource_pool is resource_pool
        assert replica.servers == [] and replica._server_handle is None and replica._server_address is None
        events.append("launch")

    def kill(actor, no_restart):
        assert actor is old and no_restart is True
        events.append("kill")

    replica.launch_servers = launch_servers
    monkeypatch.setattr(vllm_async_server.ray, "kill", kill)
    return replica, old


def test_restart_retains_reservations_until_old_actor_is_confirmed_dead(monkeypatch):
    events = []

    async def shutdown():
        events.append("shutdown")

    async def probe():
        events.append("probe")
        count = events.count("probe")
        if count == 2:
            raise vllm_async_server.ray.exceptions.ActorUnavailableError("temporarily unreachable", None)
        if count == 3:
            raise vllm_async_server.ray.exceptions.ActorDiedError()
        return "old-address"

    replica, old = _make_restart_replica(monkeypatch, shutdown, probe, events)
    asyncio.run(replica.restart())
    assert events == ["shutdown", "kill", "probe", "probe", "probe", "identity", "launch"]


@pytest.mark.parametrize("actor_already_dead", [False, True])
def test_restart_refuses_failed_shutdown_without_replacing_server(monkeypatch, actor_already_dead):
    events = []
    failure = vllm_async_server.ray.exceptions.ActorDiedError() if actor_already_dead else RuntimeError("not released")

    async def shutdown():
        events.append("shutdown")
        raise failure

    async def probe():
        raise AssertionError("failed shutdown must not kill or probe the actor")

    replica, old = _make_restart_replica(monkeypatch, shutdown, probe, events)
    with pytest.raises(type(failure)) as caught:
        asyncio.run(replica.restart())
    assert caught.value is failure
    assert events == ["shutdown"]
    assert replica.servers == [old] and replica._server_handle is old


@pytest.mark.parametrize("identity", [None, (True, 1.0), (1234, float("nan"))])
def test_restart_refuses_invalid_process_identity_before_killing_actor(monkeypatch, identity):
    events = []

    async def shutdown():
        events.append("shutdown")

    async def probe():
        raise AssertionError("invalid identity must not kill or probe the actor")

    replica, old = _make_restart_replica(monkeypatch, shutdown, probe, events, process_identity=identity)
    with pytest.raises(RuntimeError, match="invalid process identity"):
        asyncio.run(replica.restart())
    assert events == ["shutdown"]
    assert replica.servers == [old] and replica._server_handle is old


@pytest.mark.parametrize("behavior", ["alive", "unavailable", "hung"])
def test_restart_bounds_actor_exit_confirmation_and_refuses_replacement(monkeypatch, behavior):
    events = []

    async def shutdown():
        events.append("shutdown")

    async def probe():
        events.append("probe")
        if behavior == "unavailable":
            raise vllm_async_server.ray.exceptions.ActorUnavailableError("temporarily unreachable", None)
        if behavior == "hung":
            await asyncio.Event().wait()
        return "old-address"

    replica, old = _make_restart_replica(monkeypatch, shutdown, probe, events)
    with pytest.raises(TimeoutError, match="actor exit timed out"):
        asyncio.run(replica.restart(timeout=0.03))
    assert "launch" not in events
    assert replica.servers == [old] and replica._server_handle is old


def test_restart_bounds_shutdown_rpc_before_killing_actor(monkeypatch):
    events = []

    async def shutdown():
        events.append("shutdown")
        await asyncio.Event().wait()

    async def probe():
        raise AssertionError("timed-out shutdown must not probe the actor")

    replica, old = _make_restart_replica(monkeypatch, shutdown, probe, events)
    with pytest.raises(TimeoutError, match="shutdown timed out"):
        asyncio.run(replica.restart(timeout=0.03))
    assert events == ["shutdown"]
    assert replica.servers == [old]


def test_restart_refuses_overlapping_calls_and_releases_guard_after_cancellation(monkeypatch):
    events = []

    async def main():
        shutting_down = asyncio.Event()

        async def shutdown():
            events.append("shutdown")
            shutting_down.set()
            await asyncio.Event().wait()

        async def probe():
            raise AssertionError("unfinished shutdown must not probe the actor")

        replica, old = _make_restart_replica(monkeypatch, shutdown, probe, events)
        first = asyncio.create_task(replica.restart())
        await shutting_down.wait()
        with pytest.raises(RuntimeError, match="already in progress"):
            await replica.restart()
        assert events == ["shutdown"] and replica.servers == [old]
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert not replica._restart_in_progress
        with pytest.raises(TimeoutError, match="shutdown timed out"):
            await replica.restart(timeout=0.03)
        assert events == ["shutdown", "shutdown"] and replica.servers == [old]
        assert not replica._restart_in_progress

    asyncio.run(main())


def test_restart_cancellation_does_not_replace_server(monkeypatch):
    events = []

    async def main():
        probing = asyncio.Event()

        async def shutdown():
            events.append("shutdown")

        async def probe():
            probing.set()
            await asyncio.Event().wait()

        replica, old = _make_restart_replica(monkeypatch, shutdown, probe, events)
        task = asyncio.create_task(replica.restart())
        await probing.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert replica.servers == [old] and replica._server_handle is old
        assert events == ["shutdown", "kill"]

    asyncio.run(main())


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_restart_rejects_invalid_deadline_before_mutation(monkeypatch, timeout):
    events = []

    async def untouched():
        raise AssertionError("invalid timeout must not reach the actor")

    replica, old = _make_restart_replica(monkeypatch, untouched, untouched, events)
    with pytest.raises(ValueError, match="positive and finite"):
        asyncio.run(replica.restart(timeout=timeout))
    assert events == [] and replica.servers == [old]


def test_restart_does_not_treat_generic_actor_error_as_proof_of_death(monkeypatch):
    events = []

    async def shutdown():
        events.append("shutdown")

    async def probe():
        raise vllm_async_server.ray.exceptions.RayActorError()

    replica, old = _make_restart_replica(monkeypatch, shutdown, probe, events)
    with pytest.raises(vllm_async_server.ray.exceptions.RayActorError):
        asyncio.run(replica.restart())
    assert events == ["shutdown", "kill"] and replica.servers == [old]


def test_restart_supports_legacy_ray_terminal_actor_error(monkeypatch):
    events = []
    legacy_actor_error = vllm_async_server.ray.exceptions.RayActorError
    monkeypatch.delattr(vllm_async_server.ray.exceptions, "ActorDiedError")
    monkeypatch.delattr(vllm_async_server.ray.exceptions, "ActorUnavailableError")

    async def shutdown():
        events.append("shutdown")

    async def probe():
        events.append("probe")
        raise legacy_actor_error()

    replica, old = _make_restart_replica(monkeypatch, shutdown, probe, events)
    asyncio.run(replica.restart())
    assert events == ["shutdown", "kill", "probe", "identity", "launch"]


def test_restart_waits_for_os_process_exit_after_ray_reports_actor_dead(monkeypatch):
    events = []

    async def shutdown():
        events.append("shutdown")

    async def probe():
        events.append("probe")
        raise vllm_async_server.ray.exceptions.ActorDiedError()

    async def worker_probe(fn, pid, create_time):
        assert (pid, create_time) == (1234, 1.0)
        return events.count("identity") == 1

    replica, old = _make_restart_replica(monkeypatch, shutdown, probe, events, worker_probe)
    asyncio.run(replica.restart())
    assert events == ["shutdown", "kill", "probe", "identity", "identity", "launch"]


@pytest.mark.parametrize("behavior", ["alive", "hung", "access_error", "invalid_result"])
def test_restart_fails_closed_when_old_process_exit_cannot_be_proved(monkeypatch, behavior):
    events = []

    async def shutdown():
        events.append("shutdown")

    async def probe():
        raise vllm_async_server.ray.exceptions.ActorDiedError()

    async def worker_probe(fn, pid, create_time):
        if behavior == "access_error":
            raise vllm_async_server.psutil.AccessDenied(pid)
        if behavior == "hung":
            await asyncio.Event().wait()
        if behavior == "invalid_result":
            return None
        return True

    replica, old = _make_restart_replica(monkeypatch, shutdown, probe, events, worker_probe)
    error = {
        "access_error": vllm_async_server.psutil.AccessDenied,
        "invalid_result": RuntimeError,
    }.get(behavior, TimeoutError)
    with pytest.raises(error):
        asyncio.run(replica.restart(timeout=0.03))
    assert "launch" not in events
    assert replica.servers == [old] and replica._server_handle is old


def test_restart_treats_reused_pid_as_old_actor_gone_without_signaling_it(monkeypatch):
    events = []
    replacement = SimpleNamespace(create_time=lambda: 2.0, is_running=lambda: True)
    monkeypatch.setattr(vllm_async_server.psutil, "Process", lambda pid: replacement)

    async def shutdown():
        events.append("shutdown")

    async def probe():
        raise vllm_async_server.ray.exceptions.ActorDiedError()

    async def worker_probe(fn, pid, create_time):
        return fn(None, pid, create_time)

    replica, old = _make_restart_replica(monkeypatch, shutdown, probe, events, worker_probe)
    asyncio.run(replica.restart())
    assert events == ["shutdown", "kill", "identity", "launch"]


def test_process_identity_probe_reports_live_and_gone_actor(monkeypatch):
    process = SimpleNamespace(create_time=lambda: 1.0, is_running=lambda: True)
    monkeypatch.setattr(vllm_async_server.psutil, "Process", lambda pid: process)
    assert vllm_async_server._server_process_is_alive(None, 1234, 1.0)

    def gone(pid):
        raise vllm_async_server.psutil.NoSuchProcess(pid)

    monkeypatch.setattr(vllm_async_server.psutil, "Process", gone)
    assert not vllm_async_server._server_process_is_alive(None, 1234, 1.0)


class _SelectedTokenServer(vllm_async_server.vLLMHttpServer):
    """Scores fixed token IDs at every response position without overriding generate()."""

    def _preprocess_sampling_params(self, sampling_params):
        sampling_params["logprob_token_ids"] = sampling_params.pop("selected_token_ids")

    def _postprocess_output(self, output, final_res, sampling_params):
        rows = final_res.outputs[0].logprobs if final_res.outputs else []
        ids = sampling_params.logprob_token_ids
        output.extra_fields["selected_logprobs"] = [[row[t].logprob for t in ids] for row in rows]
        return output


class _RejectingServer(vllm_async_server.vLLMHttpServer):
    def _preprocess_sampling_params(self, sampling_params):
        raise ValueError("unsupported request")


def _make_generating_server(monkeypatch, cls, outputs):
    server = _make_server(cls=cls)
    server.config = RolloutConfig(name="vllm", max_model_len=64, prompt_length=32, response_length=16)
    server.model_config = SimpleNamespace(processor=None, lora_rank=0, lora={})
    server.replica_rank = 0
    server.engine.outputs = outputs
    # Outside a Ray actor, get_runtime_context() would start a local cluster.
    monkeypatch.setattr(
        vllm_async_server.ray, "get_runtime_context", lambda: SimpleNamespace(get_actor_name=lambda: "test")
    )
    return server


def _request_output(token_ids, rows=None):
    completion = SimpleNamespace(
        token_ids=token_ids, logprobs=rows, finish_reason="stop", routed_experts=None, num_preempted=0
    )
    return SimpleNamespace(outputs=[completion], num_cached_tokens=0, prompt_logprobs=None)


def test_default_request_hooks_leave_generate_output_unchanged(monkeypatch):
    async def main():
        server = _make_generating_server(monkeypatch, vllm_async_server.vLLMHttpServer, [_request_output([5, 9])])

        output = await server.generate([1, 2, 3], {"temperature": 1.0}, "r")

        assert server.engine.sampling_params[0].logprob_token_ids is None
        assert output.token_ids == [5, 9] and output.stop_reason == "completed"
        assert output.extra_fields == {"global_steps": 7, "num_cached_tokens": 0}
        assert server._admitting == 0

    asyncio.run(main())


def test_request_hooks_add_selected_token_scores_without_overriding_generate(monkeypatch):
    async def main():
        rows = [{5: Logprob(logprob=-0.1, rank=1), 7: Logprob(logprob=-2.3, rank=2)}]
        server = _make_generating_server(monkeypatch, _SelectedTokenServer, [_request_output([5], rows)])

        output = await server.generate([1, 2, 3], {"selected_token_ids": [7, 5]}, "r")

        assert server.engine.sampling_params[0].logprob_token_ids == [7, 5]
        assert output.extra_fields["selected_logprobs"] == [[-2.3, -0.1]]
        assert output.log_probs is None
        assert server._admitting == 0, "the output hook runs after the admission is released"

    asyncio.run(main())


def test_preprocess_hook_error_fails_request_before_admission(monkeypatch):
    async def main():
        server = _make_generating_server(monkeypatch, _RejectingServer, [_request_output([5])])

        with pytest.raises(ValueError, match="unsupported request"):
            await server.generate([1, 2, 3], {}, "r")

        assert server.engine.sampling_params == [], "a rejected request must never reach the engine"
        assert server._admitting == 0

    asyncio.run(main())


def test_output_hook_runs_for_requests_aborted_with_empty_outputs(monkeypatch):
    async def main():
        aborted = SimpleNamespace(outputs=[])
        server = _make_generating_server(monkeypatch, _SelectedTokenServer, [aborted])

        output = await server.generate([1, 2, 3], {"selected_token_ids": [7]}, "r")

        assert output.stop_reason == "aborted" and output.token_ids == []
        assert output.extra_fields["selected_logprobs"] == []
        assert server._admitting == 0

    asyncio.run(main())
