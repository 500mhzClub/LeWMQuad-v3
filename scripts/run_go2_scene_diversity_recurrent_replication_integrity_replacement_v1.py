#!/usr/bin/env python3
"""Run the one-shot scene-diversity infrastructure replacement.

The scientific runner is the frozen V1 implementation.  This module applies a
small, scoped configuration overlay for the fresh replacement authority,
schemas, roots, source closure, and split-process collector.  All checkpoint,
train/evaluation access-order, context-only, model, metric, gate, and result
construction behavior continues to execute in the frozen V1 runner.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import threading
from typing import Any, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import collect_go2_scene_diversity_recurrent_replication_integrity_replacement_v1 as collector  # noqa: E402
from scripts import build_go2_scene_diversity_recurrent_replication_integrity_replacement_v1_plan as plan_builder  # noqa: E402
from scripts import run_go2_scene_diversity_recurrent_replication_v1 as frozen_runner  # noqa: E402


AUTHORITY_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_integrity_replacement_v1_"
    "execution_authority_v1"
)
AUTHORITY_STATUS = (
    "AUTHORIZED_ONE_SCENE_DIVERSITY_RECURRENT_REPLICATION_"
    "INTEGRITY_REPLACEMENT_V1"
)
SOURCE_REVIEW_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_integrity_replacement_v1_"
    "source_review_v1"
)
SOURCE_REVIEW_STATUS = "PASS_INDEPENDENT_SOURCE_REVIEW"
RESULT_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_integrity_replacement_v1_"
    "result_v1"
)
TERMINAL_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_integrity_replacement_v1_"
    "terminal_v1"
)
RESERVATION_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_integrity_replacement_v1_"
    "attempt_reservation_v1"
)

PREREGISTRATION = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "integrity_replacement_v1_preregistration_2026-08-04.md"
)
SCENE_PANEL = frozen_runner.SCENE_PANEL
SCENE_PANEL_SHA256 = frozen_runner.SCENE_PANEL_SHA256
SCENE_PANEL_BYTE_COUNT = frozen_runner.SCENE_PANEL_BYTE_COUNT
SOURCE_REVIEW = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "integrity_replacement_v1_source_review_2026-08-04.json"
)
DEFAULT_ATTEMPT_ROOT = REPO_ROOT / (
    ".generated/dev/"
    "go2_scene_diversity_recurrent_replication_integrity_replacement_v1/"
    "attempt_v1"
)
DEFAULT_COLLECTION_ROOT = DEFAULT_ATTEMPT_ROOT / "collection"

PREDECESSOR_TERMINAL = REPO_ROOT / (
    ".generated/dev/go2_scene_diversity_recurrent_replication_v1/"
    "attempt_v1/terminal.json"
)
PREDECESSOR_TERMINAL_SHA256 = (
    "df4cecb5edc45f25a98f4753e82e95334a6b8c4e9e0d719bb13150b9be690bfa"
)
PREDECESSOR_TERMINAL_BYTE_COUNT = 451
PREDECESSOR_TERMINAL_REVIEW = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_v1_"
    "terminal_review_2026-08-04.json"
)
PREDECESSOR_TERMINAL_REVIEW_SHA256 = (
    "7f2ac7eb3f9fa16fd91a5311009cdbcd7d4777e8d9c3f7746666ce1afbc6da59"
)
PREDECESSOR_TERMINAL_REVIEW_BYTE_COUNT = 15_809

DINO_REPOSITORY = frozen_runner.DINO_REPOSITORY
DINO_CHECKPOINT = frozen_runner.DINO_CHECKPOINT
DINO_REPOSITORY_COMMIT = frozen_runner.DINO_REPOSITORY_COMMIT
DINO_CHECKPOINT_SHA256 = frozen_runner.DINO_CHECKPOINT_SHA256
DINO_CHECKPOINT_BYTE_COUNT = frozen_runner.DINO_CHECKPOINT_BYTE_COUNT

SOURCE_PATHS = {
    **frozen_runner.SOURCE_PATHS,
    "replacement_plan_builder": REPO_ROOT
    / "scripts/build_go2_scene_diversity_recurrent_replication_integrity_replacement_v1_plan.py",
    "replacement_collector": REPO_ROOT
    / "scripts/collect_go2_scene_diversity_recurrent_replication_integrity_replacement_v1.py",
    "replacement_runner": Path(__file__).resolve(),
    "replacement_authority_builder": REPO_ROOT
    / "scripts/build_go2_scene_diversity_recurrent_replication_integrity_replacement_v1_authority.py",
    "replacement_plan_test": REPO_ROOT
    / "lewm/tests/test_build_go2_scene_diversity_recurrent_replication_integrity_replacement_v1_plan.py",
    "replacement_collector_test": REPO_ROOT
    / "lewm/tests/test_collect_go2_scene_diversity_recurrent_replication_integrity_replacement_v1.py",
    "replacement_runner_test": REPO_ROOT
    / "lewm/tests/test_run_go2_scene_diversity_recurrent_replication_integrity_replacement_v1.py",
    "replacement_authority_test": REPO_ROOT
    / "lewm/tests/test_build_go2_scene_diversity_recurrent_replication_integrity_replacement_v1_authority.py",
    "predecessor_exact_plan": plan_builder.FROZEN_V1_EXACT_PLAN,
    "predecessor_failure_terminal": PREDECESSOR_TERMINAL,
    "predecessor_terminal_review": PREDECESSOR_TERMINAL_REVIEW,
}

SceneDiversityRunnerError = frozen_runner.SceneDiversityRunnerError
ContextOnlyLedgerV1 = frozen_runner.ContextOnlyLedgerV1
RoleRuntimeDataV1 = frozen_runner.RoleRuntimeDataV1
benchmark = frozen_runner.benchmark
torch = frozen_runner.torch

canonical_bytes_v1 = frozen_runner.canonical_bytes_v1
file_binding_v1 = frozen_runner.file_binding_v1
expected_dino_v1 = frozen_runner.expected_dino_v1
assert_role_disjointness_v1 = frozen_runner.assert_role_disjointness_v1
_read_context_rgb_v1 = frozen_runner._read_context_rgb_v1  # noqa: SLF001
_save_checkpoint_exclusive = frozen_runner._save_checkpoint_exclusive  # noqa: SLF001
_write_json_exclusive = frozen_runner._write_json_exclusive  # noqa: SLF001
_FROZEN_LOAD_PHYSICS_INDEX_V1 = frozen_runner._load_physics_index_v1  # noqa: SLF001

_CONFIGURATION_LOCK = threading.RLock()


def _expected_binding_v1(
    *, path: Path, sha256: str, byte_count: int
) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256,
        "byte_count": byte_count,
    }


def predecessor_failure_bindings_v1() -> dict[str, dict[str, object]]:
    """Rehash and semantically require the consumed V1 failure evidence."""

    expected_terminal = _expected_binding_v1(
        path=PREDECESSOR_TERMINAL,
        sha256=PREDECESSOR_TERMINAL_SHA256,
        byte_count=PREDECESSOR_TERMINAL_BYTE_COUNT,
    )
    expected_review = _expected_binding_v1(
        path=PREDECESSOR_TERMINAL_REVIEW,
        sha256=PREDECESSOR_TERMINAL_REVIEW_SHA256,
        byte_count=PREDECESSOR_TERMINAL_REVIEW_BYTE_COUNT,
    )
    if file_binding_v1(PREDECESSOR_TERMINAL) != expected_terminal:
        raise SceneDiversityRunnerError("predecessor failure terminal changed")
    if file_binding_v1(PREDECESSOR_TERMINAL_REVIEW) != expected_review:
        raise SceneDiversityRunnerError("predecessor terminal review changed")
    try:
        terminal = json.loads(PREDECESSOR_TERMINAL.read_bytes())
        review = json.loads(PREDECESSOR_TERMINAL_REVIEW.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SceneDiversityRunnerError(
            "predecessor failure evidence is not strict JSON"
        ) from exc
    if (
        not isinstance(terminal, Mapping)
        or terminal.get("schema")
        != "lewm_go2_scene_diversity_recurrent_replication_v1_terminal_v1"
        or terminal.get("status")
        != "FAIL_INFRASTRUCTURE_NO_SCIENTIFIC_DECISION"
        or terminal.get("result_binding") is not None
        or terminal.get("authorizes_retry_or_resume") is not False
        or terminal.get("failure")
        != {
            "type": "BoundedBranchSupervisionError",
            "message": (
                "active selected-device VRAM ceiling exceeded "
                "(17808412672 > 16977405952)"
            ),
        }
    ):
        raise SceneDiversityRunnerError(
            "predecessor failure terminal contract changed"
        )
    permission = review.get("permission_audit", {}) if isinstance(review, Mapping) else {}
    checks = review.get("checks", {}) if isinstance(review, Mapping) else {}
    if (
        not isinstance(review, Mapping)
        or review.get("schema")
        != "lewm_go2_scene_diversity_recurrent_replication_v1_terminal_review_v1"
        or review.get("status")
        != "PASS_FAIL_CLOSED_INFRASTRUCTURE_TERMINAL_REVIEW"
        or review.get("audit_passed") is not True
        or review.get("bindings", {}).get("terminal") != expected_terminal
        or permission.get("retry_authorized") is not False
        or permission.get("resume_authorized") is not False
        or permission.get("partial_attempt_artifact_reuse_authorized") is not False
        or permission.get("replacement_attempt_authorized_by_this_review") is not False
        or checks.get("physics_result_checkpoint_and_result_absent") is not True
        or checks.get("no_scientific_metric_or_verdict_admitted") is not True
    ):
        raise SceneDiversityRunnerError(
            "predecessor terminal review contract changed"
        )
    return {
        "predecessor_failure_terminal": expected_terminal,
        "predecessor_terminal_review": expected_review,
    }


def _load_replacement_physics_index_v1(
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, object],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Require split-process evidence after the frozen combined-result checks.

    Frozen V1 validates the standard combined physics-result and every later
    receipt binding.  The replacement collector additionally owns the pure
    infrastructure validator for worker order, fixed seed, process-exit release
    barrier, role closures, and exact join.  This hook runs before determinism,
    receipt loading, DINO execution, model fitting, or checkpoint publication.
    """

    physics_index = _FROZEN_LOAD_PHYSICS_INDEX_V1(
        authority, authority_binding, plan
    )
    validator = getattr(
        collector, "validate_split_collection_evidence_v1", None
    )
    if not callable(validator):
        raise SceneDiversityRunnerError(
            "replacement collector split-process validator is absent"
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
            f"replacement split-process evidence changed: {exc}"
        ) from exc
    if (
        not isinstance(evidence, Mapping)
        or evidence.get("validated") is not True
        or evidence.get("workers_exact") is not True
        or evidence.get("fixed_seed_exact") is not True
        or evidence.get("release_barrier_exact") is not True
        or evidence.get("join_exact") is not True
    ):
        raise SceneDiversityRunnerError(
            "replacement split-process evidence did not pass exactly"
        )
    physics_index["_replacement_split_process_validation"] = dict(evidence)
    return physics_index


def _configuration_overrides_v1() -> dict[str, object]:
    return {
        "collector": collector,
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
        "_load_physics_index_v1": _load_replacement_physics_index_v1,
    }


@contextmanager
def _configured_frozen_runner_v1() -> Iterator[None]:
    """Temporarily overlay only replacement identity and collector globals."""

    with _CONFIGURATION_LOCK:
        overrides = _configuration_overrides_v1()
        originals = {name: getattr(frozen_runner, name) for name in overrides}
        try:
            for name, value in overrides.items():
                setattr(frozen_runner, name, value)
            yield
        finally:
            for name, value in originals.items():
                setattr(frozen_runner, name, value)


def _validate_authority_v1(
    authority_path: Path, *, expected_sha256: str, expected_byte_count: int
) -> tuple[dict[str, Any], dict[str, object], dict[str, Any]]:
    predecessor_failure_bindings_v1()
    with _configured_frozen_runner_v1():
        validated = frozen_runner._validate_authority_v1(  # noqa: SLF001
            authority_path,
            expected_sha256=expected_sha256,
            expected_byte_count=expected_byte_count,
        )
    _validate_plan_v1(validated[2], validated[0])
    return validated


def _validate_plan_v1(
    plan: Mapping[str, Any], authority: Mapping[str, Any]
) -> None:
    with _configured_frozen_runner_v1():
        frozen_runner._validate_plan_v1(plan, authority)  # noqa: SLF001
    expected_frozen_binding = {
        "path": str(plan_builder.FROZEN_V1_EXACT_PLAN.resolve()),
        "sha256": plan_builder.FROZEN_V1_EXACT_PLAN_SHA256,
        "byte_count": plan_builder.FROZEN_V1_EXACT_PLAN_BYTE_COUNT,
    }
    if file_binding_v1(plan_builder.FROZEN_V1_EXACT_PLAN) != expected_frozen_binding:
        raise SceneDiversityRunnerError("frozen V1 exact plan changed")
    try:
        frozen_plan = json.loads(plan_builder.FROZEN_V1_EXACT_PLAN.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SceneDiversityRunnerError(
            "frozen V1 exact plan is not strict JSON"
        ) from exc
    if (
        not isinstance(frozen_plan, dict)
        or set(plan) != set(frozen_plan)
        or plan.get("attempt_id") != plan_builder.DEFAULT_ATTEMPT_ID
        or plan.get("output_root") != str(DEFAULT_COLLECTION_ROOT.resolve())
        or any(
            canonical_bytes_v1(plan[field])
            != canonical_bytes_v1(frozen_plan[field])
            for field in set(frozen_plan) - {"attempt_id", "output_root"}
        )
    ):
        raise SceneDiversityRunnerError(
            "replacement plan is not science-identical to frozen V1"
        )


def execute_v1(
    authority: Mapping[str, Any],
    *,
    authority_binding: Mapping[str, object],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute the unchanged scientific runner under replacement identity."""

    _validate_plan_v1(plan, authority)
    predecessor_failure_bindings_v1()
    with _configured_frozen_runner_v1():
        return frozen_runner.execute_v1(
            authority,
            authority_binding=authority_binding,
            plan=plan,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--expected-authority-sha256", required=True)
    parser.add_argument("--expected-authority-byte-count", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    authority: Mapping[str, Any] | None = None
    try:
        authority, binding, plan = _validate_authority_v1(
            args.authority,
            expected_sha256=args.expected_authority_sha256,
            expected_byte_count=args.expected_authority_byte_count,
        )
        result = execute_v1(authority, authority_binding=binding, plan=plan)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "attempt_root": authority["attempt_root"],
                }
            )
        )
        return 0
    except Exception as error:
        if authority is not None:
            attempt = Path(str(authority["attempt_root"]))
            terminal = attempt / "terminal.json"
            if attempt.is_dir() and not terminal.exists():
                try:
                    _write_json_exclusive(
                        terminal,
                        {
                            "schema": TERMINAL_SCHEMA,
                            "status": "FAIL_INFRASTRUCTURE_NO_SCIENTIFIC_DECISION",
                            "authorizes_retry_or_resume": False,
                            "authorizes_navigation_claim": False,
                            "authorizes_blind_rollout_preregistration": False,
                            "result_binding": None,
                            "failure": {
                                "type": type(error).__name__,
                                "message": str(error),
                            },
                        },
                    )
                except Exception:
                    pass
        print(f"error: {error}", file=sys.stderr)
        return 1


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
    "PREDECESSOR_TERMINAL",
    "PREDECESSOR_TERMINAL_REVIEW",
    "RESULT_SCHEMA",
    "RESERVATION_SCHEMA",
    "SOURCE_PATHS",
    "SceneDiversityRunnerError",
    "TERMINAL_SCHEMA",
    "assert_role_disjointness_v1",
    "execute_v1",
    "expected_dino_v1",
    "file_binding_v1",
    "predecessor_failure_bindings_v1",
]
