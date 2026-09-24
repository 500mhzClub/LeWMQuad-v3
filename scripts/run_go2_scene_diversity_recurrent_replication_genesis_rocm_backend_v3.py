#!/usr/bin/env python3
"""Run fresh V3 science only after exact V3 qualification PASS."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import json
from pathlib import Path
import sys
import threading
from typing import Any, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_plan as plan_builder  # noqa: E402
from scripts import collect_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3 as collector  # noqa: E402
from scripts import qualify_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3 as qualifier  # noqa: E402
from scripts import run_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v2 as predecessor  # noqa: E402


predecessor_runner = predecessor.predecessor_runner
v2_runner = predecessor.v2_runner
v1_replacement_runner = predecessor.v1_replacement_runner
frozen_runner = predecessor.frozen_runner

AUTHORITY_SCHEMA = collector.AUTHORITY_SCHEMA
AUTHORITY_STATUS = collector.AUTHORITY_STATUS
SOURCE_REVIEW_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v3_source_review_v1"
)
SOURCE_REVIEW_STATUS = "PASS_INDEPENDENT_SOURCE_REVIEW"
RESULT_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v3_result_v1"
)
TERMINAL_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v3_terminal_v1"
)
RESERVATION_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v3_attempt_reservation_v1"
)
PREREGISTRATION = qualifier.PREREGISTRATION
SOURCE_REVIEW = qualifier.SOURCE_REVIEW
SCENE_PANEL = predecessor.SCENE_PANEL
SCENE_PANEL_SHA256 = predecessor.SCENE_PANEL_SHA256
SCENE_PANEL_BYTE_COUNT = predecessor.SCENE_PANEL_BYTE_COUNT
DEFAULT_ATTEMPT_ROOT = plan_builder.DEFAULT_ATTEMPT_ROOT
DEFAULT_COLLECTION_ROOT = plan_builder.DEFAULT_OUTPUT_ROOT

CPU_TERMINAL_REVIEW = plan_builder.CPU_TERMINAL_REVIEW
CPU_TERMINAL_REVIEW_SHA256 = plan_builder.CPU_TERMINAL_REVIEW_SHA256
CPU_TERMINAL_REVIEW_BYTE_COUNT = plan_builder.CPU_TERMINAL_REVIEW_BYTE_COUNT

DINO_REPOSITORY = predecessor.DINO_REPOSITORY
DINO_CHECKPOINT = predecessor.DINO_CHECKPOINT
DINO_REPOSITORY_COMMIT = predecessor.DINO_REPOSITORY_COMMIT
DINO_CHECKPOINT_SHA256 = predecessor.DINO_CHECKPOINT_SHA256
DINO_CHECKPOINT_BYTE_COUNT = predecessor.DINO_CHECKPOINT_BYTE_COUNT
PROCESS_RESET_DEPENDENCY_PATHS = predecessor.PROCESS_RESET_DEPENDENCY_PATHS
ROCM_BACKEND_DEPENDENCY_PATHS = predecessor.ROCM_BACKEND_DEPENDENCY_PATHS

_V2_SOURCE_ONLY_PATHS = {
    "v2_rocm_plan_builder_source": Path(
        plan_builder.predecessor.__file__
    ).resolve(),
    "v2_rocm_collector_source": Path(collector.predecessor.__file__).resolve(),
    "v2_rocm_qualifier_source": Path(qualifier.predecessor.__file__).resolve(),
    "v2_rocm_runner_source": Path(predecessor.__file__).resolve(),
    "v2_rocm_qualification_authority_builder_source": REPO_ROOT
    / "scripts/build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v2_qualification_authority.py",
    "v2_rocm_scientific_authority_builder_source": REPO_ROOT
    / "scripts/build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v2_authority.py",
    "v2_rocm_plan_test_source": REPO_ROOT
    / "lewm/tests/test_build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v2_plan.py",
    "v2_rocm_collector_test_source": REPO_ROOT
    / "lewm/tests/test_collect_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v2.py",
    "v2_rocm_qualifier_test_source": REPO_ROOT
    / "lewm/tests/test_qualify_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v2.py",
    "v2_rocm_runner_test_source": REPO_ROOT
    / "lewm/tests/test_run_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v2.py",
    "v2_rocm_qualification_authority_test_source": REPO_ROOT
    / "lewm/tests/test_build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v2_qualification_authority.py",
    "v2_rocm_scientific_authority_test_source": REPO_ROOT
    / "lewm/tests/test_build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v2_authority.py",
    "v2_rocm_preregistration_source": predecessor.PREREGISTRATION,
}

SOURCE_PATHS = {
    **predecessor_runner.SOURCE_PATHS,
    **ROCM_BACKEND_DEPENDENCY_PATHS,
    **_V2_SOURCE_ONLY_PATHS,
    "rocm_backend_v3_plan_builder": Path(plan_builder.__file__).resolve(),
    "rocm_backend_v3_collector": Path(collector.__file__).resolve(),
    "rocm_backend_v3_qualifier": Path(qualifier.__file__).resolve(),
    "rocm_backend_v3_runner": Path(__file__).resolve(),
    "rocm_backend_v3_qualification_authority_builder": REPO_ROOT
    / "scripts/build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_qualification_authority.py",
    "rocm_backend_v3_scientific_authority_builder": REPO_ROOT
    / "scripts/build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_authority.py",
    "rocm_backend_v3_plan_test": REPO_ROOT
    / "lewm/tests/test_build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_plan.py",
    "rocm_backend_v3_collector_test": REPO_ROOT
    / "lewm/tests/test_collect_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3.py",
    "rocm_backend_v3_qualifier_test": REPO_ROOT
    / "lewm/tests/test_qualify_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3.py",
    "rocm_backend_v3_runner_test": REPO_ROOT
    / "lewm/tests/test_run_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3.py",
    "rocm_backend_v3_qualification_authority_test": REPO_ROOT
    / "lewm/tests/test_build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_qualification_authority.py",
    "rocm_backend_v3_scientific_authority_test": REPO_ROOT
    / "lewm/tests/test_build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_authority.py",
    "rocm_backend_v3_preregistration": PREREGISTRATION,
    "rocm_backend_v3_scientific_exact_plan": plan_builder.DEFAULT_PLAN_OUTPUT,
    "rocm_backend_v3_qualification_exact_plan": (
        plan_builder.QUALIFICATION_PLAN_OUTPUT
    ),
    "predecessor_cpu_qualification_terminal_review": CPU_TERMINAL_REVIEW,
    "predecessor_v1_qualification_terminal_review": (
        plan_builder.predecessor.V1_QUALIFICATION_TERMINAL_REVIEW
    ),
    "predecessor_v2_qualification_terminal_review": (
        plan_builder.V2_QUALIFICATION_TERMINAL_REVIEW
    ),
}

SceneDiversityRunnerError = predecessor.SceneDiversityRunnerError
ContextOnlyLedgerV1 = predecessor.ContextOnlyLedgerV1
RoleRuntimeDataV1 = predecessor.RoleRuntimeDataV1
benchmark = predecessor.benchmark
torch = predecessor.torch
canonical_bytes_v1 = predecessor.canonical_bytes_v1
file_binding_v1 = predecessor.file_binding_v1
expected_dino_v1 = predecessor.expected_dino_v1
assert_role_disjointness_v1 = predecessor.assert_role_disjointness_v1

ROCM_EGL_PREFLIGHT_FIELDS = frozenset(
    set(predecessor.ROCM_EGL_PREFLIGHT_FIELDS)
    | {
        "path_ld_lld_driver",
        "rocm_path_ld_lld_driver",
        "lld_driver_entrypoint",
        "lld_driver_link_text",
        "lld_resolved_target",
        "lld_invocation_argv",
        "lld_version_prefix_passed",
    }
)
ROCM_IDENTITY_FIELDS = frozenset(
    set(predecessor.ROCM_IDENTITY_FIELDS) | {"home"}
)
QUALIFICATION_SCENE_RESULT_FIELDS = predecessor.QUALIFICATION_SCENE_RESULT_FIELDS
QUALIFICATION_PROBE_FIELDS = predecessor.QUALIFICATION_PROBE_FIELDS
QUALIFICATION_WORKER_FIELDS = predecessor.QUALIFICATION_WORKER_FIELDS
QUALIFICATION_RELEASE_BARRIER_FIELDS = (
    predecessor.QUALIFICATION_RELEASE_BARRIER_FIELDS
)
QUALIFICATION_KERNEL_AUDIT_FIELDS = predecessor.QUALIFICATION_KERNEL_AUDIT_FIELDS
QUALIFICATION_TIMING_GATE_FIELDS = predecessor.QUALIFICATION_TIMING_GATE_FIELDS

_ORIGINAL_PREDECESSOR_FAILURE_BINDINGS = (
    predecessor.predecessor_failure_bindings_rocm
)
_ORIGINAL_VALIDATE_QUALIFICATION_RESULT = (
    predecessor.validate_qualification_result_binding
)
_ORIGINAL_VALIDATE_AUTHORITY_ROCM = predecessor._validate_authority_rocm  # noqa: SLF001
_CONFIGURATION_LOCK = threading.RLock()


class _V3QualifierReplayFacade:
    """Expose V3 qualification APIs with the inherited scene implementation.

    The inherited runner treats ``qualifier.predecessor`` as the scene
    collector implementation.  V3 preserves V2's explicit replay seam.
    """

    predecessor = qualifier.scene_predecessor

    def __getattr__(self, name: str) -> object:
        return getattr(qualifier, name)


_QUALIFIER_REPLAY_FACADE = _V3QualifierReplayFacade()


def _standard_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(value["path"]),
        "sha256": str(value.get("sha256", value.get("file_sha256"))),
        "byte_count": int(value["byte_count"]),
    }


def predecessor_failure_bindings_rocm() -> dict[str, dict[str, object]]:
    """Add only the permitted V2 terminal-review document to prior evidence."""

    evidence = dict(_ORIGINAL_PREDECESSOR_FAILURE_BINDINGS())
    expected = _standard_binding(
        plan_builder.v2_qualification_terminal_review_binding()
    )
    if file_binding_v1(plan_builder.V2_QUALIFICATION_TERMINAL_REVIEW) != expected:
        raise SceneDiversityRunnerError(
            "V2 qualification terminal-review document changed"
        )
    evidence["predecessor_v2_qualification_terminal_review"] = expected
    return evidence


def _validate_v3_preflight_evidence(result: Mapping[str, Any]) -> None:
    preflight = result.get("rocm_egl_preflight")
    if not isinstance(preflight, Mapping):
        raise SceneDiversityRunnerError("V3 linker evidence is absent")
    driver = str(plan_builder.ROCM_LD_LLD_DRIVER_ENTRYPOINT)
    target = str(
        plan_builder.ROCM_RUNTIME_PATHS["rocm_lld_executable"].resolve(
            strict=True
        )
    )
    if (
        set(preflight) != ROCM_EGL_PREFLIGHT_FIELDS
        or preflight.get("path_ld_lld") != target
        or preflight.get("rocm_path_ld_lld") != target
        or preflight.get("path_ld_lld_driver") != driver
        or preflight.get("rocm_path_ld_lld_driver") != driver
        or preflight.get("lld_driver_entrypoint") != driver
        or preflight.get("lld_driver_link_text")
        != plan_builder.ROCM_LD_LLD_DRIVER_LINK_TEXT
        or preflight.get("lld_resolved_target") != target
        or preflight.get("lld_invocation_argv") != [driver, "--version"]
        or preflight.get("lld_version_prefix_passed") is not True
        or not isinstance(preflight.get("identity"), Mapping)
        or preflight["identity"].get("home")
        != plan_builder.REQUIRED_HOST_HOME
        or preflight.get("expectation")
        != plan_builder.ROCM_GRAPHICS_PREFLIGHT_EXPECTATION
    ):
        raise SceneDiversityRunnerError(
            "V3 exact ld.lld driver evidence changed"
        )


def _configuration_overrides_v3() -> dict[str, object]:
    return {
        "plan_builder": plan_builder,
        "collector": collector,
        "qualifier": _QUALIFIER_REPLAY_FACADE,
        "AUTHORITY_SCHEMA": AUTHORITY_SCHEMA,
        "AUTHORITY_STATUS": AUTHORITY_STATUS,
        "SOURCE_REVIEW_SCHEMA": SOURCE_REVIEW_SCHEMA,
        "SOURCE_REVIEW_STATUS": SOURCE_REVIEW_STATUS,
        "RESULT_SCHEMA": RESULT_SCHEMA,
        "TERMINAL_SCHEMA": TERMINAL_SCHEMA,
        "RESERVATION_SCHEMA": RESERVATION_SCHEMA,
        "PREREGISTRATION": PREREGISTRATION,
        "SOURCE_REVIEW": SOURCE_REVIEW,
        "DEFAULT_ATTEMPT_ROOT": DEFAULT_ATTEMPT_ROOT,
        "DEFAULT_COLLECTION_ROOT": DEFAULT_COLLECTION_ROOT,
        "SOURCE_PATHS": SOURCE_PATHS,
        "ROCM_EGL_PREFLIGHT_FIELDS": ROCM_EGL_PREFLIGHT_FIELDS,
        "ROCM_IDENTITY_FIELDS": ROCM_IDENTITY_FIELDS,
        "predecessor_failure_bindings_rocm": predecessor_failure_bindings_rocm,
        "validate_qualification_result_binding": (
            validate_qualification_result_binding
        ),
        "_validate_plan_rocm": _validate_plan_rocm,
    }


@contextmanager
def _configured_predecessor_runner_rocm() -> Iterator[None]:
    with _CONFIGURATION_LOCK:
        overrides = _configuration_overrides_v3()
        originals = {
            name: getattr(predecessor, name) for name in overrides
        }
        lower = predecessor.predecessor
        original_identity_fields = lower.ROCM_IDENTITY_FIELDS
        try:
            for name, value in overrides.items():
                setattr(predecessor, name, value)
            lower.ROCM_IDENTITY_FIELDS = ROCM_IDENTITY_FIELDS
            with predecessor._configured_predecessor_runner_rocm():  # noqa: SLF001
                yield
        finally:
            lower.ROCM_IDENTITY_FIELDS = original_identity_fields
            for name, value in originals.items():
                setattr(predecessor, name, value)


def validate_qualification_result_binding(value: object):
    with _configured_predecessor_runner_rocm():
        result, binding = _ORIGINAL_VALIDATE_QUALIFICATION_RESULT(
            value
        )
    _validate_v3_preflight_evidence(result)
    try:
        authority = json.loads(qualifier.QUALIFICATION_AUTHORITY.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SceneDiversityRunnerError(
            "V3 qualification authority is not strict JSON"
        ) from exc
    collector._require_v2_review_binding(authority)  # noqa: SLF001
    return result, binding


def _validate_plan_rocm(
    plan: Mapping[str, Any], authority: Mapping[str, Any]
) -> None:
    if authority.get("attempt_id") != plan_builder.DEFAULT_ATTEMPT_ID:
        raise SceneDiversityRunnerError(
            "V3 scientific authority identity changed"
        )
    try:
        plan_builder.validate_rocm_plan(
            plan,
            expected_attempt_id=plan_builder.DEFAULT_ATTEMPT_ID,
            expected_output_root=plan_builder.DEFAULT_OUTPUT_ROOT,
            plan_role="scientific",
        )
    except plan_builder.SceneDiversityGenesisRocmV3PlanError as exc:
        raise SceneDiversityRunnerError(str(exc)) from exc


def _validate_authority_rocm(
    authority_path: Path,
    *,
    expected_sha256: str,
    expected_byte_count: int,
):
    predecessor_failure_bindings_rocm()
    with _configured_predecessor_runner_rocm():
        validated = _ORIGINAL_VALIDATE_AUTHORITY_ROCM(
            authority_path,
            expected_sha256=expected_sha256,
            expected_byte_count=expected_byte_count,
        )
    collector._require_v2_review_binding(validated[0])  # noqa: SLF001
    validate_qualification_result_binding(
        validated[0].get("qualification_result_binding")
    )
    _validate_plan_rocm(validated[2], validated[0])
    return validated


def execute_rocm(
    authority: Mapping[str, Any],
    *,
    authority_binding: Mapping[str, object],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    collector.require_exact_orchestrator_python()
    collector.require_exact_orchestrator_environment("scientific")
    _validate_plan_rocm(plan, authority)
    predecessor_failure_bindings_rocm()
    collector._require_v2_review_binding(authority)  # noqa: SLF001
    validate_qualification_result_binding(
        authority.get("qualification_result_binding")
    )
    with _configured_predecessor_runner_rocm():
        return predecessor_runner.execute_v3(
            authority,
            authority_binding=authority_binding,
            plan=plan,
        )


execute_v3 = execute_rocm
execute_v2 = execute_rocm
execute_v1 = execute_rocm
_validate_plan_v3 = _validate_plan_rocm
_validate_authority_v3 = _validate_authority_rocm


def build_parser():
    parser = predecessor_runner.build_parser()
    parser.description = (
        "Run fresh Genesis ROCm V3 science after exact V3 qualification."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    collector.require_exact_orchestrator_python()
    collector.require_exact_orchestrator_environment("scientific")
    args = build_parser().parse_args(argv)
    authority: Mapping[str, Any] | None = None
    try:
        authority, binding, plan = _validate_authority_rocm(
            args.authority,
            expected_sha256=args.expected_authority_sha256,
            expected_byte_count=args.expected_authority_byte_count,
        )
        result = execute_rocm(
            authority,
            authority_binding=binding,
            plan=plan,
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "attempt_root": authority["attempt_root"],
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        if authority is not None:
            attempt = DEFAULT_ATTEMPT_ROOT
            terminal = attempt / "terminal.json"
            if attempt.is_dir() and not terminal.exists():
                try:
                    collector.pilot.write_json_exclusive(
                        terminal,
                        {
                            "schema": TERMINAL_SCHEMA,
                            "status": (
                                "FAIL_INFRASTRUCTURE_NO_SCIENTIFIC_DECISION"
                            ),
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
    "RESULT_SCHEMA",
    "ROCM_BACKEND_DEPENDENCY_PATHS",
    "SOURCE_PATHS",
    "SceneDiversityRunnerError",
    "TERMINAL_SCHEMA",
    "execute_rocm",
    "predecessor_failure_bindings_rocm",
    "validate_qualification_result_binding",
]
