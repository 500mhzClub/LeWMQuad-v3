#!/usr/bin/env python3
"""Collect the fresh V3 host-identity Genesis ROCm successor.

This is a scoped identity adapter over the corrected V2 collector source.  It
owns the V3 plan role/root routing so a fresh child process cannot fall back
to V2 identities.  No invocation is self-authorizing.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import os
from pathlib import Path
import sys
import threading
from typing import Any, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_plan as plan_builder  # noqa: E402
from scripts import collect_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v2 as predecessor  # noqa: E402


AUTHORITY_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v3_execution_authority_v1"
)
AUTHORITY_STATUS = (
    "AUTHORIZED_SCENE_DIVERSITY_RECURRENT_REPLICATION_"
    "GENESIS_ROCM_BACKEND_V3"
)
AUTHORITY_FIELDS = frozenset(
    set(predecessor.AUTHORITY_FIELDS)
    | {"predecessor_v2_qualification_terminal_review_binding"}
)
ATTEMPT_ID = plan_builder.DEFAULT_ATTEMPT_ID
RESERVATION_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v3_collection_reservation_v1"
)
SCENE_RESULT_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v3_scene_physics_result_v1"
)
SCENE_EVIDENCE_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v3_scene_process_evidence_v1"
)

EXPECTED_CAPS = predecessor.EXPECTED_CAPS
EXPECTED_COUNTS = predecessor.EXPECTED_COUNTS
EXPECTED_HISTORY_PANEL = predecessor.EXPECTED_HISTORY_PANEL
EXPECTED_PERMISSIONS = predecessor.EXPECTED_PERMISSIONS
PLAN_FIRST_EFFECTIVE_GENESIS_SEED = predecessor.PLAN_FIRST_EFFECTIVE_GENESIS_SEED
PLAN_FIRST_PHYSICS_SEED = predecessor.PLAN_FIRST_PHYSICS_SEED
ROLE_ORDER = predecessor.ROLE_ORDER
SCENE_COUNT = predecessor.SCENE_COUNT
SCENE_EVIDENCE_STATUS = predecessor.SCENE_EVIDENCE_STATUS
SceneProcessCollectionError = predecessor.SceneProcessCollectionError
pilot = predecessor.pilot
bounded = predecessor.bounded
kernel = predecessor.kernel
ROCM_ADDITIONAL_SANITIZED_KEYS = frozenset(
    set(predecessor.ROCM_ADDITIONAL_SANITIZED_KEYS)
    | {"HOME", "USER", "LOGNAME", "LANG"}
)

CONTACT_FORCE_ROUTE_AUDIT = {
    **copy.deepcopy(predecessor.CONTACT_FORCE_ROUTE_AUDIT),
    "schema": (
        "lewm_go2_scene_diversity_recurrent_replication_"
        "genesis_rocm_backend_v3_contact_force_route_source_audit_v1"
    ),
}
PROCESS_RESET_EQUIVALENCE_AUDIT_ROCM = {
    **copy.deepcopy(predecessor.PROCESS_RESET_EQUIVALENCE_AUDIT_ROCM),
    "schema": (
        "lewm_go2_scene_diversity_recurrent_replication_"
        "genesis_rocm_backend_v3_process_reset_equivalence_source_audit_v1"
    ),
    "v2_runtime_payload_reuse_authorized": False,
    "v3_required_host_home": plan_builder.REQUIRED_HOST_HOME,
    "contact_force_route_audit": copy.deepcopy(CONTACT_FORCE_ROUTE_AUDIT),
}
PROCESS_RESET_EQUIVALENCE_AUDIT_V3 = PROCESS_RESET_EQUIVALENCE_AUDIT_ROCM

_ORIGINAL_WORKER_ARGV = predecessor._worker_argv_rocm  # noqa: SLF001
_read_collection_reservation_rocm = (
    predecessor._read_collection_reservation_rocm  # noqa: SLF001
)
_initialize_from_plan_first_scene_rocm = (
    predecessor._initialize_from_plan_first_scene_rocm  # noqa: SLF001
)
_build_rollout_runner_rocm = predecessor._build_rollout_runner_rocm  # noqa: SLF001
_selected_gpu_memory_files_rocm = (
    predecessor._selected_gpu_memory_files_rocm  # noqa: SLF001
)
_ORIGINAL_VALIDATE_AUTHORITY_V2 = predecessor._ORIGINAL_VALIDATE_AUTHORITY_V2  # noqa: SLF001
_CONFIGURATION_LOCK = threading.RLock()


def _standard_v2_review_binding() -> dict[str, Any]:
    value = plan_builder.v2_qualification_terminal_review_binding()
    return {
        "path": str(value["path"]),
        "sha256": str(value["file_sha256"]),
        "byte_count": int(value["byte_count"]),
    }


def _require_v2_review_binding(authority: Mapping[str, Any]) -> None:
    if authority.get(
        "predecessor_v2_qualification_terminal_review_binding"
    ) != _standard_v2_review_binding():
        raise SceneProcessCollectionError(
            "V3 authority V2 terminal-review binding changed"
        )


def _validate_authority_v3_review_bound(*args: Any, **kwargs: Any):
    authority = _ORIGINAL_VALIDATE_AUTHORITY_V2(*args, **kwargs)
    _require_v2_review_binding(authority)
    return authority


def _expected_output_root_for_identity() -> Path:
    if ATTEMPT_ID == plan_builder.DEFAULT_ATTEMPT_ID:
        return plan_builder.DEFAULT_OUTPUT_ROOT
    if ATTEMPT_ID == plan_builder.QUALIFICATION_ATTEMPT_ID:
        return plan_builder.QUALIFICATION_OUTPUT_ROOT
    raise SceneProcessCollectionError("V3 collector attempt identity changed")


def _plan_role_for_identity() -> str:
    if ATTEMPT_ID == plan_builder.DEFAULT_ATTEMPT_ID:
        return "scientific"
    if ATTEMPT_ID == plan_builder.QUALIFICATION_ATTEMPT_ID:
        return "qualification"
    raise SceneProcessCollectionError("V3 collector attempt role changed")


def _validate_rocm_plan_runtime(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return plan_builder.validate_rocm_plan(
            value,
            expected_attempt_id=ATTEMPT_ID,
            expected_output_root=_expected_output_root_for_identity(),
            plan_role=_plan_role_for_identity(),
        )
    except plan_builder.SceneDiversityGenesisRocmV3PlanError as exc:
        raise pilot.PilotContractError(str(exc)) from exc


def require_exact_orchestrator_python() -> str:
    """Fail before reservation unless V3 runs from its lexical ROCm venv."""

    expected_python = str(plan_builder.ROCM_PYTHON.absolute())
    if sys.executable != expected_python:
        raise SceneProcessCollectionError(
            "V3 orchestrator Python launcher is not the exact lexical ROCm venv path"
        )
    return expected_python


def require_exact_orchestrator_environment(
    plan_role: str,
) -> dict[str, str]:
    """Require the complete sanitized environment before V3 reservation."""

    expected = plan_builder.rocm_execution_environment(plan_role)
    selector_keys = (
        set(kernel._SANITIZED_SELECTOR_KEYS)  # noqa: SLF001
        | set(ROCM_ADDITIONAL_SANITIZED_KEYS)
        | set(expected)
    )
    for key in sorted(selector_keys):
        if key in expected:
            if os.environ.get(key) != expected[key]:
                raise SceneProcessCollectionError(
                    f"V3 {plan_role} orchestrator environment changed at {key}"
                )
        elif key in os.environ:
            raise SceneProcessCollectionError(
                f"V3 {plan_role} orchestrator environment retained forbidden {key}"
            )
    return dict(expected)


def _worker_argv_rocm(**kwargs: Any) -> list[str]:
    expected_python = require_exact_orchestrator_python()
    argv = _ORIGINAL_WORKER_ARGV(**kwargs)
    if (
        len(argv) < 2
        or argv[0] != expected_python
        or Path(argv[1]).resolve() != Path(predecessor.__file__).resolve()
    ):
        raise SceneProcessCollectionError(
            "V2 worker argv changed from the exact V3 ROCm launcher contract"
        )
    argv[1] = str(Path(__file__).resolve())
    return argv


def _configuration_overrides_v3() -> dict[str, object]:
    return {
        "plan_builder": plan_builder,
        "AUTHORITY_FIELDS": AUTHORITY_FIELDS,
        "AUTHORITY_SCHEMA": AUTHORITY_SCHEMA,
        "AUTHORITY_STATUS": AUTHORITY_STATUS,
        "ATTEMPT_ID": ATTEMPT_ID,
        "RESERVATION_SCHEMA": RESERVATION_SCHEMA,
        "SCENE_RESULT_SCHEMA": SCENE_RESULT_SCHEMA,
        "SCENE_EVIDENCE_SCHEMA": SCENE_EVIDENCE_SCHEMA,
        "CONTACT_FORCE_ROUTE_AUDIT": CONTACT_FORCE_ROUTE_AUDIT,
        "PROCESS_RESET_EQUIVALENCE_AUDIT_ROCM": (
            PROCESS_RESET_EQUIVALENCE_AUDIT_ROCM
        ),
        "PROCESS_RESET_EQUIVALENCE_AUDIT_V2": (
            PROCESS_RESET_EQUIVALENCE_AUDIT_ROCM
        ),
        "ROCM_ADDITIONAL_SANITIZED_KEYS": ROCM_ADDITIONAL_SANITIZED_KEYS,
        "_expected_output_root_for_identity": (
            _expected_output_root_for_identity
        ),
        "_plan_role_for_identity": _plan_role_for_identity,
        "_validate_rocm_plan_runtime": _validate_rocm_plan_runtime,
        "_read_collection_reservation_rocm": (
            _read_collection_reservation_rocm
        ),
        "_worker_argv_rocm": _worker_argv_rocm,
        "_validate_authority_v2_review_bound": (
            _validate_authority_v3_review_bound
        ),
    }


@contextmanager
def _configured_predecessor_collector_rocm() -> Iterator[None]:
    """Install V3 identity before entering the inherited runtime adapter."""

    with _CONFIGURATION_LOCK:
        overrides = _configuration_overrides_v3()
        originals = {
            name: getattr(predecessor, name) for name in overrides
        }
        lower = predecessor.predecessor
        original_sanitized_keys = lower.ROCM_ADDITIONAL_SANITIZED_KEYS
        try:
            for name, value in overrides.items():
                setattr(predecessor, name, value)
            lower.ROCM_ADDITIONAL_SANITIZED_KEYS = (
                ROCM_ADDITIONAL_SANITIZED_KEYS
            )
            with predecessor._configured_predecessor_collector_rocm():  # noqa: SLF001
                yield
        finally:
            lower.ROCM_ADDITIONAL_SANITIZED_KEYS = original_sanitized_keys
            for name, value in originals.items():
                setattr(predecessor, name, value)


def load_and_validate_rocm(*args: Any, **kwargs: Any):
    with _configured_predecessor_collector_rocm():
        return predecessor.load_and_validate_v2(*args, **kwargs)


load_and_validate_v3 = load_and_validate_rocm
load_and_validate_v2 = load_and_validate_rocm
load_and_validate_replacement_v3 = load_and_validate_rocm


def validate_scene_process_evidence_rocm(*args: Any, **kwargs: Any):
    with _configured_predecessor_collector_rocm():
        return predecessor.validate_scene_process_evidence_v2(
            *args, **kwargs
        )


validate_scene_process_evidence_v3 = validate_scene_process_evidence_rocm
validate_scene_process_evidence_v2 = validate_scene_process_evidence_rocm


def validate_scene_process_closure_rocm(*args: Any, **kwargs: Any):
    with _configured_predecessor_collector_rocm():
        return predecessor.validate_scene_process_closure_v2(
            *args, **kwargs
        )


validate_scene_process_closure_v3 = validate_scene_process_closure_rocm
validate_scene_process_closure_v2 = validate_scene_process_closure_rocm


def collect_rocm(
    *,
    plan_path: Path,
    expected_plan_byte_count: int,
    expected_plan_sha256: str,
    authority_path: Path,
    expected_authority_byte_count: int,
    expected_authority_sha256: str,
):
    require_exact_orchestrator_python()
    require_exact_orchestrator_environment(_plan_role_for_identity())
    with _configured_predecessor_collector_rocm():
        return predecessor.collect_v2(
            plan_path=plan_path,
            expected_plan_byte_count=expected_plan_byte_count,
            expected_plan_sha256=expected_plan_sha256,
            authority_path=authority_path,
            expected_authority_byte_count=expected_authority_byte_count,
            expected_authority_sha256=expected_authority_sha256,
        )


collect_v3 = collect_rocm
collect_v2 = collect_rocm
collect_v1 = collect_rocm


def build_parser():
    parser = predecessor.build_parser()
    parser.description = (
        "Collect the separately authorized fresh Genesis ROCm V3 successor."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    require_exact_orchestrator_python()
    require_exact_orchestrator_environment(_plan_role_for_identity())
    with _configured_predecessor_collector_rocm():
        return predecessor.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ATTEMPT_ID",
    "AUTHORITY_FIELDS",
    "AUTHORITY_SCHEMA",
    "AUTHORITY_STATUS",
    "CONTACT_FORCE_ROUTE_AUDIT",
    "EXPECTED_CAPS",
    "EXPECTED_COUNTS",
    "EXPECTED_HISTORY_PANEL",
    "EXPECTED_PERMISSIONS",
    "PROCESS_RESET_EQUIVALENCE_AUDIT_ROCM",
    "PROCESS_RESET_EQUIVALENCE_AUDIT_V3",
    "ROCM_ADDITIONAL_SANITIZED_KEYS",
    "ROLE_ORDER",
    "SCENE_COUNT",
    "SCENE_EVIDENCE_SCHEMA",
    "SCENE_EVIDENCE_STATUS",
    "SceneProcessCollectionError",
    "bounded",
    "collect_rocm",
    "collect_v1",
    "collect_v2",
    "collect_v3",
    "load_and_validate_rocm",
    "pilot",
    "predecessor",
    "require_exact_orchestrator_environment",
    "require_exact_orchestrator_python",
    "validate_scene_process_closure_rocm",
    "validate_scene_process_evidence_rocm",
]
