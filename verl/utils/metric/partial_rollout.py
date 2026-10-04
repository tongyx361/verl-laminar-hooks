# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Consumed-trajectory partial rollout observations."""


def partial_rollout_metrics(extra_fields, non_padding):
    """Aggregate only consumed, nonpadding trajectories with explicit evidence.

    These counts exclude aborted work on trajectories subsequently evicted.
    Prefill seconds sum per-request wall intervals and are not GPU-hours.
    """
    eligible = [extra for extra, active in zip(extra_fields, non_padding, strict=True) if active]
    records = [extra["partial_rollout"] for extra in eligible if isinstance(extra, dict) and "partial_rollout" in extra]
    prefix = "training/partial_rollout/"
    result = {prefix + "coverage": len(records) / len(eligible) if eligible else 0.0}
    if not records:
        return result
    for field in ("abort_count", "empty_abort_count", "resume_count", "retained_prefix_resume_count"):
        result[prefix + field] = sum(record[field] for record in records)
    result[prefix + "resumed_response_fraction"] = sum(r["resume_count"] > 0 for r in records) / len(records)
    result[prefix + "retained_prefix_response_fraction"] = sum(
        r["retained_prefix_resume_count"] > 0 for r in records
    ) / len(records)
    observed = sum(r["resume_prefill_observed_count"] for r in records)
    resumes = result[prefix + "resume_count"]
    result[prefix + "resume_prefill_coverage"] = observed / resumes if resumes else 0.0
    result[prefix + "resume_prefill_available"] = float(resumes > 0 and observed == resumes)
    if observed:
        result[prefix + "resume_prefill_observed_seconds"] = sum(
            r["resume_prefill_observed_seconds"] or 0 for r in records
        )
    return result
