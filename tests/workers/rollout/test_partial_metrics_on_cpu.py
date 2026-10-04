# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
from types import SimpleNamespace

import pytest

from verl.utils.metric.partial_rollout import partial_rollout_metrics
from verl.workers.rollout.partial_metrics import engine_prefill_timing, summarize_partial_attempts


@pytest.mark.parametrize("start,end", [(None, None), (0, 0), (5, 0), (5, 4), (float("nan"), 6)])
def test_missing_or_incomplete_prefill_is_unavailable(start, end):
    assert engine_prefill_timing(SimpleNamespace(scheduled_ts=start, first_token_ts=end)) == {
        "available": False,
        "seconds": None,
    }


def test_prefill_interval_excludes_arrival_and_decode():
    metrics = SimpleNamespace(arrival_time=1, scheduled_ts=5, first_token_ts=7, last_token_ts=100)
    assert engine_prefill_timing(metrics) == {"available": True, "seconds": 2}


def test_partial_aggregation_preserves_unavailable_and_excludes_padding():
    def attempt(retained, tokens, aborted, timing):
        return {"retained_tokens": retained, "new_tokens": tokens, "aborted": aborted, "prefill": timing}

    available = {"available": True, "seconds": 2.5}
    missing = {"available": False, "seconds": None}
    resumed = summarize_partial_attempts(
        [attempt(0, 2, True, available), attempt(2, 0, True, missing), attempt(2, 3, False, available)]
    )
    plain = summarize_partial_attempts([attempt(0, 5, False, available)])
    fields = [{"partial_rollout": resumed}, {"partial_rollout": plain}, {}, {"partial_rollout": resumed}]
    result = partial_rollout_metrics(fields, [True, True, True, False])
    p = "training/partial_rollout/"
    assert result[p + "coverage"] == pytest.approx(2 / 3)
    assert result[p + "abort_count"] == result[p + "resume_count"] == 2
    assert result[p + "empty_abort_count"] == 1
    assert result[p + "resumed_response_fraction"] == 0.5
    assert result[p + "retained_prefix_response_fraction"] == 0.5
    assert result[p + "resume_prefill_coverage"] == 0.5
    assert result[p + "resume_prefill_available"] == 0
    assert result[p + "resume_prefill_observed_seconds"] == 2.5
    assert resumed["resume_prefill_available"] is False
    complete = summarize_partial_attempts([attempt(0, 2, True, available), attempt(2, 3, False, available)])
    assert complete["resume_prefill_available"] is True
    assert complete["resume_prefill_observed_seconds"] == 2.5
