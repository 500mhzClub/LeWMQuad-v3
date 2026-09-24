#!/usr/bin/env python3
"""Run the final identity-only scene-diversity replacement V3.

The reviewed V2 runner remains the complete collection, custody and scientific
implementation.  This wrapper supplies a fresh identity and exact evidence
that V2 was consumed by an audited transient GPU/native-runtime failure before
any scientific stage.  It does not reuse any V2 runtime artifact.
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

from scripts import build_go2_scene_diversity_recurrent_replication_integrity_replacement_v3_plan as plan_builder  # noqa: E402
from scripts import collect_go2_scene_diversity_recurrent_replication_integrity_replacement_v3 as collector  # noqa: E402
from scripts import run_go2_scene_diversity_recurrent_replication_integrity_replacement_v2 as predecessor_runner  # noqa: E402


frozen_runner = predecessor_runner.frozen_runner

AUTHORITY_SCHEMA = collector.AUTHORITY_SCHEMA
AUTHORITY_STATUS = collector.AUTHORITY_STATUS
SOURCE_REVIEW_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_integrity_replacement_v3_"
    "source_review_v1"
)
SOURCE_REVIEW_STATUS = "PASS_INDEPENDENT_SOURCE_REVIEW"
RESULT_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_integrity_replacement_v3_"
    "result_v1"
)
TERMINAL_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_integrity_replacement_v3_"
    "terminal_v1"
)
RESERVATION_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_integrity_replacement_v3_"
    "attempt_reservation_v1"
)

PREREGISTRATION = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "integrity_replacement_v3_preregistration_2026-08-04.md"
)
SCENE_PANEL = predecessor_runner.SCENE_PANEL
SCENE_PANEL_SHA256 = predecessor_runner.SCENE_PANEL_SHA256
SCENE_PANEL_BYTE_COUNT = predecessor_runner.SCENE_PANEL_BYTE_COUNT
SOURCE_REVIEW = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "integrity_replacement_v3_source_review_2026-08-04.json"
)
DEFAULT_ATTEMPT_ROOT = plan_builder.DEFAULT_ATTEMPT_ROOT
DEFAULT_COLLECTION_ROOT = plan_builder.DEFAULT_OUTPUT_ROOT

PREDECESSOR_V2_TERMINAL = REPO_ROOT / (
    ".generated/dev/"
    "go2_scene_diversity_recurrent_replication_integrity_replacement_v2/"
    "attempt_v1/terminal.json"
)
PREDECESSOR_V2_TERMINAL_SHA256 = (
    "ebb520c596ae69c19e3be255c0f661fe55286883e03994a43a0e506936120465"
)
PREDECESSOR_V2_TERMINAL_BYTE_COUNT = 442
PREDECESSOR_V2_TERMINAL_REVIEW = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "integrity_replacement_v2_terminal_review_2026-08-04.json"
)
PREDECESSOR_V2_TERMINAL_REVIEW_SHA256 = (
    "7878dbedf21ed3bdb13927e0404925edc376e512c00a5ba4bf56e0091e3204c6"
)
PREDECESSOR_V2_TERMINAL_REVIEW_BYTE_COUNT = 20_561

DINO_REPOSITORY = predecessor_runner.DINO_REPOSITORY
DINO_CHECKPOINT = predecessor_runner.DINO_CHECKPOINT
DINO_REPOSITORY_COMMIT = predecessor_runner.DINO_REPOSITORY_COMMIT
DINO_CHECKPOINT_SHA256 = predecessor_runner.DINO_CHECKPOINT_SHA256
DINO_CHECKPOINT_BYTE_COUNT = predecessor_runner.DINO_CHECKPOINT_BYTE_COUNT
PROCESS_RESET_DEPENDENCY_PATHS = predecessor_runner.PROCESS_RESET_DEPENDENCY_PATHS

SOURCE_PATHS = {
    **predecessor_runner.SOURCE_PATHS,
    "replacement_v3_plan_builder": REPO_ROOT
    / "scripts/build_go2_scene_diversity_recurrent_replication_integrity_replacement_v3_plan.py",
    "replacement_v3_collector": REPO_ROOT
    / "scripts/collect_go2_scene_diversity_recurrent_replication_integrity_replacement_v3.py",
    "replacement_v3_runner": Path(__file__).resolve(),
    "replacement_v3_authority_builder": REPO_ROOT
    / "scripts/build_go2_scene_diversity_recurrent_replication_integrity_replacement_v3_authority.py",
    "replacement_v3_plan_test": REPO_ROOT
    / "lewm/tests/test_build_go2_scene_diversity_recurrent_replication_integrity_replacement_v3_plan.py",
    "replacement_v3_collector_test": REPO_ROOT
    / "lewm/tests/test_collect_go2_scene_diversity_recurrent_replication_integrity_replacement_v3.py",
    "replacement_v3_runner_test": REPO_ROOT
    / "lewm/tests/test_run_go2_scene_diversity_recurrent_replication_integrity_replacement_v3.py",
    "replacement_v3_authority_test": REPO_ROOT
    / "lewm/tests/test_build_go2_scene_diversity_recurrent_replication_integrity_replacement_v3_authority.py",
    "predecessor_replacement_v2_failure_terminal": PREDECESSOR_V2_TERMINAL,
    "predecessor_replacement_v2_terminal_review": PREDECESSOR_V2_TERMINAL_REVIEW,
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
_V2_PREDECESSOR_FAILURE_BINDINGS = predecessor_runner.predecessor_failure_bindings_v2
_V2_LOAD_PHYSICS_INDEX = predecessor_runner._load_replacement_physics_index_v2  # noqa: SLF001

_CONFIGURATION_LOCK = threading.RLock()


def _expected_binding_v3(
    *, path: Path, sha256: str, byte_count: int
) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256,
        "byte_count": byte_count,
    }


def predecessor_failure_bindings_v3() -> dict[str, dict[str, object]]:
    """Require all predecessors plus exact fail-closed V2 evidence."""

    evidence = dict(_V2_PREDECESSOR_FAILURE_BINDINGS())
    expected_terminal = _expected_binding_v3(
        path=PREDECESSOR_V2_TERMINAL,
        sha256=PREDECESSOR_V2_TERMINAL_SHA256,
        byte_count=PREDECESSOR_V2_TERMINAL_BYTE_COUNT,
    )
    expected_review = _expected_binding_v3(
        path=PREDECESSOR_V2_TERMINAL_REVIEW,
        sha256=PREDECESSOR_V2_TERMINAL_REVIEW_SHA256,
        byte_count=PREDECESSOR_V2_TERMINAL_REVIEW_BYTE_COUNT,
    )
    if file_binding_v1(PREDECESSOR_V2_TERMINAL) != expected_terminal:
        raise SceneDiversityRunnerError("predecessor V2 failure terminal changed")
    if file_binding_v1(PREDECESSOR_V2_TERMINAL_REVIEW) != expected_review:
        raise SceneDiversityRunnerError("predecessor V2 terminal review changed")
    try:
        terminal = json.loads(PREDECESSOR_V2_TERMINAL.read_bytes())
        review = json.loads(PREDECESSOR_V2_TERMINAL_REVIEW.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SceneDiversityRunnerError(
            "predecessor V2 evidence is not strict JSON"
        ) from exc
    if (
        not isinstance(terminal, Mapping)
        or terminal.get("schema")
        != "lewm_go2_scene_diversity_recurrent_replication_"
        "integrity_replacement_v2_terminal_v1"
        or terminal.get("status")
        != "FAIL_INFRASTRUCTURE_NO_SCIENTIFIC_DECISION"
        or terminal.get("result_binding") is not None
        or terminal.get("authorizes_retry_or_resume") is not False
        or terminal.get("failure")
        != {
            "type": "BoundedBranchSupervisionError",
            "message": "supervised collector exited with status 1",
        }
    ):
        raise SceneDiversityRunnerError("predecessor V2 terminal contract changed")
    bindings = review.get("bindings", {}) if isinstance(review, Mapping) else {}
    permission = (
        review.get("permission_audit", {}) if isinstance(review, Mapping) else {}
    )
    checks = review.get("checks", {}) if isinstance(review, Mapping) else {}
    verdict = review.get("verdict", {}) if isinstance(review, Mapping) else {}
    cause = (
        review.get("failure_evidence", {}).get("best_supported_cause", {})
        if isinstance(review, Mapping)
        else {}
    )
    custody = review.get("custody_audit", {}) if isinstance(review, Mapping) else {}
    if (
        not isinstance(review, Mapping)
        or review.get("schema")
        != "lewm_go2_scene_diversity_recurrent_replication_"
        "integrity_replacement_v2_terminal_review_v1"
        or review.get("status")
        != "PASS_FAIL_CLOSED_INFRASTRUCTURE_TERMINAL_REVIEW"
        or review.get("audit_passed") is not True
        or bindings.get("terminal") != expected_terminal
        or verdict.get("attempt_consumed") is not True
        or verdict.get("scientific_decision") is not None
        or verdict.get("result_binding") is not None
        or verdict.get("retry_or_resume_authorized") is not False
        or verdict.get("partial_artifact_reuse_authorized") is not False
        or permission.get("successor_attempt_authorized_by_this_review") is not False
        or permission.get("scientific_conclusion_authorized") is not False
        or checks.get("combined_physics_result_is_absent") is not True
        or checks.get(
            "zero_dino_training_checkpoint_evaluation_result_metric_and_gate_stages_reached"
        )
        is not True
        or checks.get("no_scientific_metric_or_verdict_admitted") is not True
        or cause.get("category")
        != "GPU_DRIVER_OR_NATIVE_RENDER_RUNTIME_INFRASTRUCTURE_FAILURE"
        or cause.get("scientific_model_or_metric_cause") is not False
        or custody.get("protected_material_opened") is not False
        or custody.get("partial_rgb_opened") is not False
        or custody.get("scientific_output_opened") is not False
    ):
        raise SceneDiversityRunnerError(
            "predecessor V2 terminal review contract changed"
        )
    evidence.update(
        {
            "predecessor_replacement_v2_failure_terminal": expected_terminal,
            "predecessor_replacement_v2_terminal_review": expected_review,
        }
    )
    return evidence


def _load_replacement_physics_index_v3(
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, object],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Delegate unchanged to the reviewed V2 pre-science closure."""

    with _configured_predecessor_runner_v3():
        return _V2_LOAD_PHYSICS_INDEX(authority, authority_binding, plan)


def _configuration_overrides_v3() -> dict[str, object]:
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
        "predecessor_failure_bindings_v2": predecessor_failure_bindings_v3,
        "_load_replacement_physics_index_v2": _load_replacement_physics_index_v3,
    }


@contextmanager
def _configured_predecessor_runner_v3() -> Iterator[None]:
    """Apply and then restore the narrow V3 identity overlay."""

    with _CONFIGURATION_LOCK:
        overrides = _configuration_overrides_v3()
        originals = {name: getattr(predecessor_runner, name) for name in overrides}
        try:
            for name, value in overrides.items():
                setattr(predecessor_runner, name, value)
            yield
        finally:
            for name, value in originals.items():
                setattr(predecessor_runner, name, value)


def _validate_plan_v3(
    plan: Mapping[str, Any], authority: Mapping[str, Any]
) -> None:
    with _configured_predecessor_runner_v3():
        predecessor_runner._validate_plan_v2(plan, authority)  # noqa: SLF001


def _validate_authority_v3(
    authority_path: Path, *, expected_sha256: str, expected_byte_count: int
) -> tuple[dict[str, Any], dict[str, object], dict[str, Any]]:
    predecessor_failure_bindings_v3()
    with _configured_predecessor_runner_v3():
        validated = predecessor_runner._validate_authority_v2(  # noqa: SLF001
            authority_path,
            expected_sha256=expected_sha256,
            expected_byte_count=expected_byte_count,
        )
    _validate_plan_v3(validated[2], validated[0])
    return validated


def execute_v3(
    authority: Mapping[str, Any],
    *,
    authority_binding: Mapping[str, object],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute unchanged V2 custody/science under the exact V3 overlay."""

    _validate_plan_v3(plan, authority)
    predecessor_failure_bindings_v3()
    with _configured_predecessor_runner_v3():
        return predecessor_runner.execute_v2(
            authority,
            authority_binding=authority_binding,
            plan=plan,
        )


execute_v2 = execute_v3
execute_v1 = execute_v3
_validate_plan_v2 = _validate_plan_v3
_validate_plan_v1 = _validate_plan_v3
_validate_authority_v2 = _validate_authority_v3
_validate_authority_v1 = _validate_authority_v3
predecessor_failure_bindings_v2 = predecessor_failure_bindings_v3
predecessor_failure_bindings_v1 = predecessor_failure_bindings_v3


def build_parser():
    return predecessor_runner.build_parser()


def main(argv: Sequence[str] | None = None) -> int:
    with _configured_predecessor_runner_v3():
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
    "PREDECESSOR_V2_TERMINAL",
    "PREDECESSOR_V2_TERMINAL_REVIEW",
    "PROCESS_RESET_DEPENDENCY_PATHS",
    "RESULT_SCHEMA",
    "RESERVATION_SCHEMA",
    "SOURCE_PATHS",
    "SceneDiversityRunnerError",
    "TERMINAL_SCHEMA",
    "assert_role_disjointness_v1",
    "execute_v1",
    "execute_v2",
    "execute_v3",
    "expected_dino_v1",
    "file_binding_v1",
    "predecessor_failure_bindings_v3",
]
