"""Independent finalization of one runtime canonical physical-claim result."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from lewm.benchmarks.go2_physical_claim_canonical import (
    canonical_content_sha256_valid,
    canonical_json_equal,
)
from lewm.benchmarks.go2_physical_claim_evaluator import (
    evaluate_physical_claim_trace,
)
from lewm.benchmarks.go2_physical_claim_trace import (
    canonical_task_object_ids,
    task_object_set_sha256,
)
from lewm_worlds.manifest import SceneManifest


@dataclass(frozen=True)
class PhysicalClaimFinalization:
    passed: bool
    errors: tuple[str, ...]
    recomputed_trace: Mapping[str, Any] | None


def finalize_physical_claim_result(
    result: Mapping[str, Any],
    *,
    scene_manifest: SceneManifest,
    expected_task_object_ids: Sequence[str] | None = None,
    expected_task_object_set_sha256: str | None = None,
) -> PhysicalClaimFinalization:
    """Recompute the full trace and top-level physical success fail-closed."""

    errors: list[str] = []
    task_ids = canonical_task_object_ids(scene_manifest, expected_task_object_ids)
    task_hash = (
        task_object_set_sha256(scene_manifest, task_ids)
        if expected_task_object_set_sha256 is None
        else expected_task_object_set_sha256
    )
    stored = result.get("canonical_physical_claim_trace")
    recomputed: Mapping[str, Any] | None = None
    if not isinstance(stored, Mapping):
        errors.append("canonical_physical_claim_trace_missing")
    else:
        raw = {
            "schema": "lewm_go2_claim_trace_v1",
            "trace_id": stored.get("trace_id"),
            "episode_id": stored.get("episode_id"),
            "scene_id": stored.get("scene_id"),
            "physical_manifest_sha256": stored.get("physical_manifest_sha256"),
            "task_object_ids": stored.get("task_object_ids"),
            "task_object_set_sha256": stored.get("task_object_set_sha256"),
            "controller_claim_attempts": stored.get("controller_claim_attempts"),
            "evaluator_feedback_to_controller": stored.get(
                "evaluator_feedback_to_controller"
            ),
        }
        try:
            recomputed = evaluate_physical_claim_trace(
                raw,
                scene_manifest,
                task_ids,
                task_hash,
            )
        except (TypeError, ValueError) as exc:
            errors.append(f"canonical_physical_claim_trace_invalid:{exc}")
        if recomputed is not None and not canonical_json_equal(recomputed, stored):
            errors.append("canonical_physical_claim_trace_recomputation_mismatch")
        if not canonical_content_sha256_valid(
            stored, hash_field="trace_content_sha256"
        ):
            errors.append("canonical_physical_claim_trace_content_hash_invalid")
        stored_events = stored.get("physical_claim_evaluations")
        if not isinstance(stored_events, list) or any(
            not canonical_content_sha256_valid(event, hash_field="content_sha256")
            for event in stored_events
        ):
            errors.append("canonical_physical_claim_event_content_hash_invalid")
        if not canonical_content_sha256_valid(
            stored.get("physical_claim_summary"), hash_field="content_sha256"
        ):
            errors.append("canonical_physical_claim_summary_content_hash_invalid")

    ledger = result.get("runtime_evaluator_access_ledger")
    if not canonical_json_equal(
        ledger,
        {
            "evaluator_output_reads_by_controller": 0,
            "evaluator_callbacks_into_controller": 0,
            "evaluator_derived_termination_signals": 0,
        },
    ):
        errors.append("runtime_evaluator_access_ledger_invalid")

    if recomputed is not None:
        summary = recomputed["physical_claim_summary"]
        all_claimed = bool(summary["all_targets_claimed"])
        credited_ids = set(summary["credited_object_ids"])
        colors = []
        for landmark in scene_manifest.landmarks:
            if landmark.object_id not in credited_ids:
                continue
            material = landmark.material_id.casefold()
            colors.append(
                material.removeprefix("landmark_")
                if material.startswith("landmark_")
                else landmark.object_id
            )
        if result.get("claimed") is not all_claimed:
            errors.append("top_level_claimed_not_physical_summary")
        top_level_success = result.get("success")
        if type(top_level_success) is not bool:
            errors.append("top_level_success_not_boolean")
        elif top_level_success and not all_claimed:
            errors.append("top_level_success_without_physical_completion")
        if not canonical_json_equal(result.get("claimed_colors"), sorted(colors)):
            errors.append("top_level_claimed_colors_not_physical_credit")
        if summary["credited_count"] != len(credited_ids):
            errors.append("physical_summary_duplicate_credit")
        if any(
            str(item.get("pose_provenance", "")).startswith("legacy_")
            and item.get("accepted") is True
            for item in recomputed["physical_claim_evaluations"]
        ):
            errors.append("legacy_event_counted_as_physical_acceptance")
    return PhysicalClaimFinalization(
        passed=not errors,
        errors=tuple(errors),
        recomputed_trace=recomputed,
    )


__all__ = ["PhysicalClaimFinalization", "finalize_physical_claim_result"]
