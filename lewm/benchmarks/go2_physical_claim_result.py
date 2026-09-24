"""Fail-closed extraction of canonical physical claim status from result rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from lewm.benchmarks.go2_physical_claim_finalizer import (
    finalize_physical_claim_result,
)
from lewm_worlds.manifest import SceneManifest


@dataclass(frozen=True)
class CanonicalPhysicalClaimStatus:
    valid: bool
    all_targets_claimed: bool
    task_object_ids: tuple[str, ...]
    credited_object_ids: tuple[str, ...]
    attempted_count: int
    accepted_count: int
    credited_count: int
    errors: tuple[str, ...]


def canonical_physical_claim_status(
    result: Mapping[str, Any],
    *,
    scene_manifest: SceneManifest,
    expected_task_object_ids: Sequence[str] | None = None,
    expected_task_object_set_sha256: str | None = None,
    required_task_count: int | None = None,
) -> CanonicalPhysicalClaimStatus:
    finalized = finalize_physical_claim_result(
        result,
        scene_manifest=scene_manifest,
        expected_task_object_ids=expected_task_object_ids,
        expected_task_object_set_sha256=expected_task_object_set_sha256,
    )
    errors = list(finalized.errors)
    trace = finalized.recomputed_trace
    if trace is None:
        summary: Mapping[str, Any] = {}
    else:
        summary = trace["physical_claim_summary"]
    task_ids = tuple(summary.get("task_object_ids", ()))
    credited_ids = tuple(summary.get("credited_object_ids", ()))

    def exact_count(name: str) -> int:
        value = summary.get(name)
        if type(value) is not int or value < 0:
            errors.append(f"physical_claim_{name}_invalid")
            return -1
        return value

    attempted_count = exact_count("attempted_count")
    accepted_count = exact_count("accepted_count")
    credited_count = exact_count("credited_count")
    if required_task_count is not None and (
        type(required_task_count) is not int
        or required_task_count < 1
        or len(task_ids) != required_task_count
    ):
        errors.append("physical_claim_task_count_invalid")
    derived_all = bool(
        task_ids
        and credited_ids == task_ids
        and summary.get("all_targets_claimed") is True
    )
    valid = not errors
    return CanonicalPhysicalClaimStatus(
        valid=valid,
        all_targets_claimed=bool(valid and derived_all),
        task_object_ids=task_ids,
        credited_object_ids=credited_ids,
        attempted_count=attempted_count,
        accepted_count=accepted_count,
        credited_count=credited_count,
        errors=tuple(errors),
    )


__all__ = ["CanonicalPhysicalClaimStatus", "canonical_physical_claim_status"]
