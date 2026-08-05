#!/usr/bin/env python3
"""Build, but never execute, replacement V2 one-shot authority.

The reviewed V1 replacement authority builder remains the field/cap/custody
implementation.  This thin wrapper supplies the fresh V2 documents and source
closure and additionally requires explicit independent audits of the
per-scene process evidence and Genesis-reset scientific equivalence.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
import threading
from typing import Any, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_go2_scene_diversity_recurrent_replication_integrity_replacement_v1_authority as predecessor_authority  # noqa: E402
from scripts import build_go2_scene_diversity_recurrent_replication_integrity_replacement_v2_plan as plan_builder  # noqa: E402
from scripts import collect_go2_scene_diversity_recurrent_replication_integrity_replacement_v2 as collector  # noqa: E402
from scripts import run_go2_scene_diversity_recurrent_replication_integrity_replacement_v2 as runner  # noqa: E402


SOURCE_REVIEW_SCHEMA = runner.SOURCE_REVIEW_SCHEMA
SOURCE_REVIEW_STATUS = runner.SOURCE_REVIEW_STATUS

PREREGISTRATION = runner.PREREGISTRATION
SCENE_PANEL = runner.SCENE_PANEL
EXACT_PLAN = plan_builder.DEFAULT_PLAN_OUTPUT
SOURCE_REVIEW = runner.SOURCE_REVIEW
AUTHORITY_OUTPUT = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "integrity_replacement_v2_execution_authority_2026-08-04.json"
)
ATTEMPT_ROOT = runner.DEFAULT_ATTEMPT_ROOT
COLLECTION_ROOT = runner.DEFAULT_COLLECTION_ROOT

REQUIRED_RNG_EQUIVALENCE_AUDIT = collector.PROCESS_RESET_EQUIVALENCE_AUDIT_V2

REQUIRED_PROCESS_EVIDENCE_AUDIT = {
    "audit_passed": True,
    "exactly_64_sequential_one_scene_processes": True,
    "fresh_pre_launch_vram_baseline_for_each_scene": True,
    "release_barrier_baseline_equals_matching_worker_prelaunch_baseline": True,
    "release_barrier_after_each_scene_including_final": True,
    "worker_wait_receipt_precedes_mandatory_matching_release_barrier": True,
    "nonzero_exit_rejected_only_after_matching_release_barrier": True,
    "scene_result_loaded_only_after_zero_exit_and_release_barrier": True,
    "live_parent_and_child_process_group_ids_measured": True,
    "child_process_group_equals_parent": True,
    "full_plan_first_genesis_seed_in_every_worker": True,
    "no_partial_reuse_adaptive_batching_or_fallback": True,
    "pure_validator_requires_exact_workers_seeds_barriers_and_join": True,
    "pre_dino_all_scene_result_receipt_mesh_bindings_rehashed": True,
    "all_plan_input_bindings_rehashed_pre_dino": True,
    "bound_scene_inputs_absolute_and_non_generated_pre_dino": True,
    "wall_feasibility_verified": False,
    "unbound_operator_timing_used_as_authority": False,
}

SceneDiversityReplacementAuthorityError = (
    predecessor_authority.SceneDiversityReplacementAuthorityError
)

_CONFIGURATION_LOCK = threading.RLock()


def _configuration_overrides_v2() -> dict[str, object]:
    return {
        "runner": runner,
        "plan_builder": plan_builder,
        "collector": collector,
        "SOURCE_REVIEW_SCHEMA": SOURCE_REVIEW_SCHEMA,
        "SOURCE_REVIEW_STATUS": SOURCE_REVIEW_STATUS,
        "PREREGISTRATION": PREREGISTRATION,
        "SCENE_PANEL": SCENE_PANEL,
        "EXACT_PLAN": EXACT_PLAN,
        "SOURCE_REVIEW": SOURCE_REVIEW,
        "AUTHORITY_OUTPUT": AUTHORITY_OUTPUT,
        "ATTEMPT_ROOT": ATTEMPT_ROOT,
        "COLLECTION_ROOT": COLLECTION_ROOT,
    }


@contextmanager
def _configured_predecessor_authority_v2() -> Iterator[None]:
    with _CONFIGURATION_LOCK:
        overrides = _configuration_overrides_v2()
        originals = {
            name: getattr(predecessor_authority, name) for name in overrides
        }
        try:
            for name, value in overrides.items():
                setattr(predecessor_authority, name, value)
            yield
        finally:
            for name, value in originals.items():
                setattr(predecessor_authority, name, value)


def file_binding_v2(path: Path) -> dict[str, object]:
    with _configured_predecessor_authority_v2():
        return predecessor_authority.file_binding_v1(path)


file_binding_v1 = file_binding_v2


def source_bindings_v2() -> dict[str, dict[str, object]]:
    with _configured_predecessor_authority_v2():
        return predecessor_authority.source_bindings_v1()


source_bindings_v1 = source_bindings_v2


def dino_declaration_v2() -> dict[str, object]:
    with _configured_predecessor_authority_v2():
        return predecessor_authority.dino_declaration_v1()


dino_declaration_v1 = dino_declaration_v2


def _validate_science_identical_plan_v2(
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    with _configured_predecessor_authority_v2():
        return predecessor_authority._validate_science_identical_plan_v1(  # noqa: SLF001
            plan
        )


def _require_explicit_review_audits_v2(
    source_review: Mapping[str, Any],
) -> None:
    if (
        source_review.get("process_reset_equivalence_audit")
        != REQUIRED_RNG_EQUIVALENCE_AUDIT
    ):
        raise SceneDiversityReplacementAuthorityError(
            "independent RNG-equivalence audit did not pass exactly"
        )
    if (
        source_review.get("per_scene_process_evidence_audit")
        != REQUIRED_PROCESS_EVIDENCE_AUDIT
    ):
        raise SceneDiversityReplacementAuthorityError(
            "independent per-scene process-evidence audit did not pass exactly"
        )


def build_authority_v2(
    *,
    preregistration_binding: Mapping[str, Any],
    scene_panel_binding: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    source_review: Mapping[str, Any],
    source_review_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a validated authority document without writing or executing it."""

    _require_explicit_review_audits_v2(source_review)
    with _configured_predecessor_authority_v2():
        return predecessor_authority.build_authority_v1(
            preregistration_binding=preregistration_binding,
            scene_panel_binding=scene_panel_binding,
            plan=plan,
            plan_binding=plan_binding,
            source_review=source_review,
            source_review_binding=source_review_binding,
        )


build_authority_v1 = build_authority_v2


def build_parser():
    with _configured_predecessor_authority_v2():
        return predecessor_authority.build_parser()


def main(argv: Sequence[str] | None = None) -> int:
    with _configured_predecessor_authority_v2():
        return predecessor_authority.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORITY_OUTPUT",
    "COLLECTION_ROOT",
    "REQUIRED_PROCESS_EVIDENCE_AUDIT",
    "REQUIRED_RNG_EQUIVALENCE_AUDIT",
    "SOURCE_REVIEW_SCHEMA",
    "SOURCE_REVIEW_STATUS",
    "SceneDiversityReplacementAuthorityError",
    "build_authority_v1",
    "build_authority_v2",
    "dino_declaration_v1",
    "dino_declaration_v2",
    "file_binding_v1",
    "file_binding_v2",
    "source_bindings_v1",
    "source_bindings_v2",
]
