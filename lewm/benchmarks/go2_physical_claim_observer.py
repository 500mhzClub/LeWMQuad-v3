"""Observer-only finalization for runtime Go2 physical claim traces."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from lewm.benchmarks.go2_physical_claim_evaluator import (
    evaluate_physical_claim_trace,
)
from lewm_worlds.manifest import SceneManifest


def empty_evaluator_access_ledger() -> dict[str, int]:
    return {
        "evaluator_output_reads_by_controller": 0,
        "evaluator_callbacks_into_controller": 0,
        "evaluator_derived_termination_signals": 0,
    }


def evaluate_runtime_claim_trace(
    trace: Mapping[str, Any],
    physical_manifest: SceneManifest,
    expected_task_object_ids: Sequence[str],
    expected_task_object_set_sha256: str,
) -> dict[str, Any]:
    """Evaluate once after controller execution, with no controller callback."""

    return evaluate_physical_claim_trace(
        trace,
        physical_manifest,
        expected_task_object_ids,
        expected_task_object_set_sha256,
    )


__all__ = ["empty_evaluator_access_ledger", "evaluate_runtime_claim_trace"]
