#!/usr/bin/env python3
"""Run the one-scene-per-process scene-diversity replacement V2.

The original frozen V1 scientific runner remains the execution and custody
implementation.  This module applies a scoped identity/collector overlay to
the already reviewed replacement wrapper, adds exact evidence for both
consumed predecessor failures, and validates the 64-worker evidence before the
scientific route can begin.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sys
import threading
from typing import Any, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_go2_scene_diversity_recurrent_replication_integrity_replacement_v2_plan as plan_builder  # noqa: E402
from scripts import collect_go2_scene_diversity_recurrent_replication_integrity_replacement_v2 as collector  # noqa: E402
from scripts import run_go2_scene_diversity_recurrent_replication_integrity_replacement_v1 as predecessor_runner  # noqa: E402


frozen_runner = predecessor_runner.frozen_runner

AUTHORITY_SCHEMA = collector.AUTHORITY_SCHEMA
AUTHORITY_STATUS = collector.AUTHORITY_STATUS
SOURCE_REVIEW_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_integrity_replacement_v2_"
    "source_review_v1"
)
SOURCE_REVIEW_STATUS = "PASS_INDEPENDENT_SOURCE_REVIEW"
RESULT_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_integrity_replacement_v2_"
    "result_v1"
)
TERMINAL_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_integrity_replacement_v2_"
    "terminal_v1"
)
RESERVATION_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_integrity_replacement_v2_"
    "attempt_reservation_v1"
)

PREREGISTRATION = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "integrity_replacement_v2_preregistration_2026-08-04.md"
)
SCENE_PANEL = predecessor_runner.SCENE_PANEL
SCENE_PANEL_SHA256 = predecessor_runner.SCENE_PANEL_SHA256
SCENE_PANEL_BYTE_COUNT = predecessor_runner.SCENE_PANEL_BYTE_COUNT
SOURCE_REVIEW = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "integrity_replacement_v2_source_review_2026-08-04.json"
)
DEFAULT_ATTEMPT_ROOT = plan_builder.DEFAULT_ATTEMPT_ROOT
DEFAULT_COLLECTION_ROOT = plan_builder.DEFAULT_OUTPUT_ROOT

PREDECESSOR_REPLACEMENT_TERMINAL = REPO_ROOT / (
    ".generated/dev/"
    "go2_scene_diversity_recurrent_replication_integrity_replacement_v1/"
    "attempt_v1/terminal.json"
)
PREDECESSOR_REPLACEMENT_TERMINAL_SHA256 = (
    "c2650529ff6b1aa1322738e7d4f748fbab03ea9ef33a623ff3e182a98e1cb77d"
)
PREDECESSOR_REPLACEMENT_TERMINAL_BYTE_COUNT = 473
PREDECESSOR_REPLACEMENT_TERMINAL_REVIEW = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "integrity_replacement_v1_terminal_review_2026-08-04.json"
)
PREDECESSOR_REPLACEMENT_TERMINAL_REVIEW_SHA256 = (
    "e25f9281449147b6937edfafa022b5fbeb18ca3e844438c70a5babb84c2eb0cb"
)
PREDECESSOR_REPLACEMENT_TERMINAL_REVIEW_BYTE_COUNT = 21_621

DINO_REPOSITORY = predecessor_runner.DINO_REPOSITORY
DINO_CHECKPOINT = predecessor_runner.DINO_CHECKPOINT
DINO_REPOSITORY_COMMIT = predecessor_runner.DINO_REPOSITORY_COMMIT
DINO_CHECKPOINT_SHA256 = predecessor_runner.DINO_CHECKPOINT_SHA256
DINO_CHECKPOINT_BYTE_COUNT = predecessor_runner.DINO_CHECKPOINT_BYTE_COUNT

GENESIS_DEPENDENCY_ROOT = REPO_ROOT / (
    ".generated/venvs/genesis_render_vulkan/lib/python3.12/site-packages"
)
PROCESS_RESET_DEPENDENCY_PATHS = {
    "replacement_v2_dependency_genesis_init": GENESIS_DEPENDENCY_ROOT
    / "genesis/__init__.py",
    "replacement_v2_dependency_genesis_misc": GENESIS_DEPENDENCY_ROOT
    / "genesis/utils/misc.py",
    "replacement_v2_dependency_genesis_scene": GENESIS_DEPENDENCY_ROOT
    / "genesis/engine/scene.py",
    "replacement_v2_dependency_genesis_rigid_solver": GENESIS_DEPENDENCY_ROOT
    / "genesis/engine/solvers/rigid/rigid_solver.py",
    "replacement_v2_dependency_genesis_rigid_entity": GENESIS_DEPENDENCY_ROOT
    / "genesis/engine/entities/rigid_entity/rigid_entity.py",
    "replacement_v2_dependency_genesis_engine_mesh": GENESIS_DEPENDENCY_ROOT
    / "genesis/engine/mesh.py",
    "replacement_v2_dependency_genesis_mesh": GENESIS_DEPENDENCY_ROOT
    / "genesis/utils/mesh.py",
    "replacement_v2_dependency_genesis_options_misc": GENESIS_DEPENDENCY_ROOT
    / "genesis/options/misc.py",
    "replacement_v2_dependency_genesis_rasterizer_context": GENESIS_DEPENDENCY_ROOT
    / "genesis/vis/rasterizer_context.py",
    "replacement_v2_dependency_rsl_on_policy_runner": GENESIS_DEPENDENCY_ROOT
    / "rsl_rl/runners/on_policy_runner.py",
    "replacement_v2_dependency_rsl_ppo": GENESIS_DEPENDENCY_ROOT
    / "rsl_rl/algorithms/ppo.py",
    "replacement_v2_dependency_rsl_mlp_model": GENESIS_DEPENDENCY_ROOT
    / "rsl_rl/models/mlp_model.py",
}

SOURCE_PATHS = {
    **predecessor_runner.SOURCE_PATHS,
    **PROCESS_RESET_DEPENDENCY_PATHS,
    "replacement_v2_plan_builder": REPO_ROOT
    / "scripts/build_go2_scene_diversity_recurrent_replication_integrity_replacement_v2_plan.py",
    "replacement_v2_collector": REPO_ROOT
    / "scripts/collect_go2_scene_diversity_recurrent_replication_integrity_replacement_v2.py",
    "replacement_v2_runner": Path(__file__).resolve(),
    "replacement_v2_authority_builder": REPO_ROOT
    / "scripts/build_go2_scene_diversity_recurrent_replication_integrity_replacement_v2_authority.py",
    "replacement_v2_plan_test": REPO_ROOT
    / "lewm/tests/test_build_go2_scene_diversity_recurrent_replication_integrity_replacement_v2_plan.py",
    "replacement_v2_collector_test": REPO_ROOT
    / "lewm/tests/test_collect_go2_scene_diversity_recurrent_replication_integrity_replacement_v2.py",
    "replacement_v2_runner_test": REPO_ROOT
    / "lewm/tests/test_run_go2_scene_diversity_recurrent_replication_integrity_replacement_v2.py",
    "replacement_v2_authority_test": REPO_ROOT
    / "lewm/tests/test_build_go2_scene_diversity_recurrent_replication_integrity_replacement_v2_authority.py",
    "predecessor_replacement_v1_failure_terminal": PREDECESSOR_REPLACEMENT_TERMINAL,
    "predecessor_replacement_v1_terminal_review": PREDECESSOR_REPLACEMENT_TERMINAL_REVIEW,
}

SceneDiversityRunnerError = predecessor_runner.SceneDiversityRunnerError
ContextOnlyLedgerV1 = predecessor_runner.ContextOnlyLedgerV1
RoleRuntimeDataV1 = predecessor_runner.RoleRuntimeDataV1
benchmark = predecessor_runner.benchmark
torch = predecessor_runner.torch

canonical_bytes_v1 = predecessor_runner.canonical_bytes_v1
file_binding_v1 = predecessor_runner.file_binding_v1
expected_dino_v1 = predecessor_runner.expected_dino_v1
assert_role_disjointness_v1 = predecessor_runner.assert_role_disjointness_v1
_read_context_rgb_v1 = predecessor_runner._read_context_rgb_v1  # noqa: SLF001
_save_checkpoint_exclusive = predecessor_runner._save_checkpoint_exclusive  # noqa: SLF001
_write_json_exclusive = predecessor_runner._write_json_exclusive  # noqa: SLF001
_FROZEN_LOAD_PHYSICS_INDEX_V1 = predecessor_runner._FROZEN_LOAD_PHYSICS_INDEX_V1  # noqa: SLF001
_V1_PREDECESSOR_FAILURE_BINDINGS = predecessor_runner.predecessor_failure_bindings_v1

_CONFIGURATION_LOCK = threading.RLock()


def _expected_binding_v2(
    *, path: Path, sha256: str, byte_count: int
) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256,
        "byte_count": byte_count,
    }


def predecessor_failure_bindings_v2() -> dict[str, dict[str, object]]:
    """Rehash and require both consumed fail-closed predecessor attempts."""

    evidence = dict(_V1_PREDECESSOR_FAILURE_BINDINGS())
    expected_terminal = _expected_binding_v2(
        path=PREDECESSOR_REPLACEMENT_TERMINAL,
        sha256=PREDECESSOR_REPLACEMENT_TERMINAL_SHA256,
        byte_count=PREDECESSOR_REPLACEMENT_TERMINAL_BYTE_COUNT,
    )
    expected_review = _expected_binding_v2(
        path=PREDECESSOR_REPLACEMENT_TERMINAL_REVIEW,
        sha256=PREDECESSOR_REPLACEMENT_TERMINAL_REVIEW_SHA256,
        byte_count=PREDECESSOR_REPLACEMENT_TERMINAL_REVIEW_BYTE_COUNT,
    )
    if file_binding_v1(PREDECESSOR_REPLACEMENT_TERMINAL) != expected_terminal:
        raise SceneDiversityRunnerError(
            "predecessor replacement V1 failure terminal changed"
        )
    if file_binding_v1(PREDECESSOR_REPLACEMENT_TERMINAL_REVIEW) != expected_review:
        raise SceneDiversityRunnerError(
            "predecessor replacement V1 terminal review changed"
        )
    try:
        terminal = json.loads(PREDECESSOR_REPLACEMENT_TERMINAL.read_bytes())
        review = json.loads(PREDECESSOR_REPLACEMENT_TERMINAL_REVIEW.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SceneDiversityRunnerError(
            "predecessor replacement V1 evidence is not strict JSON"
        ) from exc
    if (
        not isinstance(terminal, Mapping)
        or terminal.get("schema")
        != "lewm_go2_scene_diversity_recurrent_replication_"
        "integrity_replacement_v1_terminal_v1"
        or terminal.get("status")
        != "FAIL_INFRASTRUCTURE_NO_SCIENTIFIC_DECISION"
        or terminal.get("result_binding") is not None
        or terminal.get("authorizes_retry_or_resume") is not False
        or terminal.get("failure")
        != {
            "type": "BoundedBranchSupervisionError",
            "message": (
                "active selected-device VRAM ceiling exceeded "
                "(17259581440 > 16977405952)"
            ),
        }
    ):
        raise SceneDiversityRunnerError(
            "predecessor replacement V1 terminal contract changed"
        )
    permission = review.get("permission_audit", {}) if isinstance(review, Mapping) else {}
    checks = review.get("checks", {}) if isinstance(review, Mapping) else {}
    if (
        not isinstance(review, Mapping)
        or review.get("schema")
        != "lewm_go2_scene_diversity_recurrent_replication_"
        "integrity_replacement_v1_terminal_review_v1"
        or review.get("status")
        != "PASS_FAIL_CLOSED_INFRASTRUCTURE_TERMINAL_REVIEW"
        or review.get("audit_passed") is not True
        or review.get("bindings", {}).get("terminal") != expected_terminal
        or permission.get("retry_authorized") is not False
        or permission.get("resume_authorized") is not False
        or permission.get("partial_attempt_artifact_reuse_authorized") is not False
        or permission.get("successor_attempt_authorized_by_this_review") is not False
        or checks.get(
            "zero_role_results_physics_result_checkpoint_result_metrics_and_gates"
        )
        is not True
        or checks.get("no_scientific_metric_or_verdict_admitted") is not True
    ):
        raise SceneDiversityRunnerError(
            "predecessor replacement V1 terminal review contract changed"
        )
    evidence.update(
        {
            "predecessor_replacement_v1_failure_terminal": expected_terminal,
            "predecessor_replacement_v1_terminal_review": expected_review,
        }
    )
    return evidence


def _load_replacement_physics_index_v2(
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, object],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Require exact 64-worker evidence before any scientific execution."""

    physics_index = _FROZEN_LOAD_PHYSICS_INDEX_V1(
        authority, authority_binding, plan
    )
    validator = getattr(collector, "validate_scene_process_evidence_v2", None)
    if not callable(validator):
        raise SceneDiversityRunnerError(
            "replacement V2 scene-process validator is absent"
        )
    try:
        evidence = validator(
            physics_index,
            authority_binding=authority_binding,
            plan_binding=physics_index.get("plan_binding"),
            plan=plan,
        )
    except Exception as exc:
        raise SceneDiversityRunnerError(
            f"replacement V2 scene-process evidence changed: {exc}"
        ) from exc
    process_evidence = physics_index.get("scene_process_evidence")
    if (
        not isinstance(evidence, Mapping)
        or not isinstance(process_evidence, Mapping)
        or process_evidence.get("process_reset_equivalence_audit")
        != collector.PROCESS_RESET_EQUIVALENCE_AUDIT_V2
        or evidence.get("validated") is not True
        or evidence.get("workers_exact") is not True
        or evidence.get("fixed_seed_exact") is not True
        or evidence.get("release_barriers_exact") is not True
        or evidence.get("join_exact") is not True
    ):
        raise SceneDiversityRunnerError(
            "replacement V2 scene-process evidence did not pass exactly"
        )
    try:
        collector.pilot.require_plan_bindings(plan)
        collector.bounded._validate_bound_scenes(plan)  # noqa: SLF001
    except Exception as exc:
        raise SceneDiversityRunnerError(
            f"replacement V2 pre-science plan input closure changed: {exc}"
        ) from exc
    closure_validator = getattr(
        collector, "validate_scene_process_closure_v2", None
    )
    if not callable(closure_validator):
        raise SceneDiversityRunnerError(
            "replacement V2 filesystem-closure validator is absent"
        )
    try:
        collection_root = Path(str(authority["collection_root"])).resolve(
            strict=True
        )
        closure = closure_validator(
            physics_index,
            collection_root=collection_root,
            authority_binding=authority_binding,
            plan_binding=physics_index["plan_binding"],
            plan=plan,
        )
    except Exception as exc:
        raise SceneDiversityRunnerError(
            f"replacement V2 pre-science filesystem closure changed: {exc}"
        ) from exc
    required_closure_fields = {
        "validated",
        "evidence_validated",
        "closure_rehashed",
        "scene_results_rehashed",
        "state_receipts_rehashed",
        "render_receipts_rehashed",
        "derived_meshes_rehashed",
        "plan_scene_input_bindings_rehashed",
    }
    if (
        not isinstance(closure, Mapping)
        or set(closure) != required_closure_fields
        or any(closure.get(field) is not True for field in required_closure_fields)
    ):
        raise SceneDiversityRunnerError(
            "replacement V2 pre-science filesystem closure did not pass exactly"
        )
    physics_index["_replacement_v2_scene_process_validation"] = dict(evidence)
    physics_index["_replacement_v2_plan_input_closure"] = {
        "all_plan_input_bindings_rehashed": True,
        "bound_scene_inputs_absolute_and_non_generated": True,
    }
    physics_index["_replacement_v2_generated_output_closure"] = {
        **dict(closure),
        "all_scene_result_receipt_mesh_bindings_rehashed": True,
    }
    return physics_index


def _configuration_overrides_v2() -> dict[str, object]:
    """Fields temporarily supplied to the reviewed V1 replacement wrapper."""

    return {
        "collector": collector,
        "plan_builder": plan_builder,
        "AUTHORITY_SCHEMA": AUTHORITY_SCHEMA,
        "AUTHORITY_STATUS": AUTHORITY_STATUS,
        "SOURCE_REVIEW_SCHEMA": SOURCE_REVIEW_SCHEMA,
        "SOURCE_REVIEW_STATUS": SOURCE_REVIEW_STATUS,
        "RESULT_SCHEMA": RESULT_SCHEMA,
        "TERMINAL_SCHEMA": TERMINAL_SCHEMA,
        "RESERVATION_SCHEMA": RESERVATION_SCHEMA,
        "PREREGISTRATION": PREREGISTRATION,
        "SCENE_PANEL": SCENE_PANEL,
        "SCENE_PANEL_SHA256": SCENE_PANEL_SHA256,
        "SCENE_PANEL_BYTE_COUNT": SCENE_PANEL_BYTE_COUNT,
        "SOURCE_REVIEW": SOURCE_REVIEW,
        "DEFAULT_ATTEMPT_ROOT": DEFAULT_ATTEMPT_ROOT,
        "DEFAULT_COLLECTION_ROOT": DEFAULT_COLLECTION_ROOT,
        "SOURCE_PATHS": SOURCE_PATHS,
        "predecessor_failure_bindings_v1": predecessor_failure_bindings_v2,
        "_load_replacement_physics_index_v1": _load_replacement_physics_index_v2,
    }


@contextmanager
def _configured_predecessor_runner_v2() -> Iterator[None]:
    """Apply and then restore the narrow V2 wrapper overlay."""

    with _CONFIGURATION_LOCK:
        overrides = _configuration_overrides_v2()
        originals = {
            name: getattr(predecessor_runner, name) for name in overrides
        }
        try:
            for name, value in overrides.items():
                setattr(predecessor_runner, name, value)
            yield
        finally:
            for name, value in originals.items():
                setattr(predecessor_runner, name, value)


def _validate_plan_v2(
    plan: Mapping[str, Any], authority: Mapping[str, Any]
) -> None:
    with _configured_predecessor_runner_v2():
        predecessor_runner._validate_plan_v1(plan, authority)  # noqa: SLF001


def _validate_authority_v2(
    authority_path: Path, *, expected_sha256: str, expected_byte_count: int
) -> tuple[dict[str, Any], dict[str, object], dict[str, Any]]:
    predecessor_failure_bindings_v2()
    with _configured_predecessor_runner_v2():
        validated = predecessor_runner._validate_authority_v1(  # noqa: SLF001
            authority_path,
            expected_sha256=expected_sha256,
            expected_byte_count=expected_byte_count,
        )
    _validate_plan_v2(validated[2], validated[0])
    return validated


def execute_v2(
    authority: Mapping[str, Any],
    *,
    authority_binding: Mapping[str, object],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute the frozen science only under the exact V2 overlay."""

    _validate_plan_v2(plan, authority)
    predecessor_failure_bindings_v2()
    with _configured_predecessor_runner_v2():
        return predecessor_runner.execute_v1(
            authority,
            authority_binding=authority_binding,
            plan=plan,
        )


execute_v1 = execute_v2
_validate_plan_v1 = _validate_plan_v2
_validate_authority_v1 = _validate_authority_v2
predecessor_failure_bindings_v1 = predecessor_failure_bindings_v2


def build_parser():
    return predecessor_runner.build_parser()


def main(argv: Sequence[str] | None = None) -> int:
    with _configured_predecessor_runner_v2():
        return predecessor_runner.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORITY_SCHEMA",
    "AUTHORITY_STATUS",
    "ContextOnlyLedgerV1",
    "DEFAULT_ATTEMPT_ROOT",
    "DEFAULT_COLLECTION_ROOT",
    "DINO_CHECKPOINT",
    "DINO_CHECKPOINT_BYTE_COUNT",
    "DINO_CHECKPOINT_SHA256",
    "DINO_REPOSITORY",
    "DINO_REPOSITORY_COMMIT",
    "PREDECESSOR_REPLACEMENT_TERMINAL",
    "PREDECESSOR_REPLACEMENT_TERMINAL_REVIEW",
    "PROCESS_RESET_DEPENDENCY_PATHS",
    "RESULT_SCHEMA",
    "RESERVATION_SCHEMA",
    "SOURCE_PATHS",
    "SceneDiversityRunnerError",
    "TERMINAL_SCHEMA",
    "assert_role_disjointness_v1",
    "execute_v1",
    "execute_v2",
    "expected_dino_v1",
    "file_binding_v1",
    "predecessor_failure_bindings_v2",
]
