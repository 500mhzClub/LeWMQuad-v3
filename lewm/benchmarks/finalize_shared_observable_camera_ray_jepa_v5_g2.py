"""Independent G2 finalizer for Shared JEPA V5 canonical scene results."""
from __future__ import annotations

import hashlib
from pathlib import Path
from lewm.benchmarks.shared_observable_camera_ray_jepa_v5_finalizer_core import (
    _finalize_gate_records_synthetic_for_tests,
)
from lewm.benchmarks.shared_observable_camera_ray_jepa_v5_runner_policy import (
    SyntheticRunnerBatchV6,
)


G2_METRICS = (
    "aggregate_physical_gate_pass_fraction",
    "per_family_physical_gate_pass_fraction",
    "jepa_health_gate_pass_fraction",
    "counterfactual_gate_pass_fraction",
)


def source_file_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _removed_finalize_g2(
    *,
    runner_batch: object,
    expected_model_state_sha256: str,
    expected_checkpoint_file_sha256: str,
    expected_runner_source_sha256: str,
) -> dict[str, object]:
    raise PermissionError("production library finalization was removed; use the one-shot CLI")


def _finalize_g2_synthetic_for_tests(
    *,
    runner_batch: SyntheticRunnerBatchV6,
    expected_model_state_sha256: str,
    expected_checkpoint_file_sha256: str,
    expected_runner_source_sha256: str,
) -> dict[str, object]:
    return _finalize_gate_records_synthetic_for_tests(
        gate="g2",
        metric_names=G2_METRICS,
        runner_batch=runner_batch,
        expected_model_state_sha256=expected_model_state_sha256,
        expected_checkpoint_file_sha256=expected_checkpoint_file_sha256,
        expected_runner_source_sha256=expected_runner_source_sha256,
        finalizer_source_sha256=source_file_sha256(),
    )


__all__ = ["G2_METRICS", "source_file_sha256"]
