#!/usr/bin/env python3
"""Collect the science-identical V3 integrity replacement V2.

This adapter owns a closed compatibility surface for every inherited
collector layer as well as fresh identity and review evidence.  No invocation
is self-authorizing.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import os
from pathlib import Path
import sys
import threading
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v2_plan as plan_builder  # noqa: E402
from scripts import collect_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v1 as predecessor  # noqa: E402


AUTHORITY_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v3_integrity_replacement_v2_execution_authority_v1"
)
AUTHORITY_STATUS = (
    "AUTHORIZED_SCENE_DIVERSITY_RECURRENT_REPLICATION_"
    "GENESIS_ROCM_BACKEND_V3_INTEGRITY_REPLACEMENT_V2"
)
AUTHORITY_FIELDS = frozenset(
    set(predecessor.AUTHORITY_FIELDS)
    | {"predecessor_replacement_v1_qualification_terminal_review_binding"}
)
ATTEMPT_ID = plan_builder.DEFAULT_ATTEMPT_ID
RESERVATION_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v3_integrity_replacement_v2_collection_reservation_v1"
)
SCENE_RESULT_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v3_integrity_replacement_v2_scene_physics_result_v1"
)
SCENE_EVIDENCE_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v3_integrity_replacement_v2_scene_process_evidence_v1"
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
        "genesis_rocm_backend_v3_integrity_replacement_v2_contact_force_route_source_audit_v1"
    ),
}
PROCESS_RESET_EQUIVALENCE_AUDIT_ROCM = {
    **copy.deepcopy(predecessor.PROCESS_RESET_EQUIVALENCE_AUDIT_ROCM),
    "schema": (
        "lewm_go2_scene_diversity_recurrent_replication_"
        "genesis_rocm_backend_v3_integrity_replacement_v2_process_reset_equivalence_source_audit_v1"
    ),
    "v2_runtime_payload_reuse_authorized": False,
    "v3_runtime_payload_reuse_authorized": False,
    "replacement_v1_runtime_payload_reuse_authorized": False,
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
_LOWEST_IMMUTABLE_BASE_VALIDATE_AUTHORITY = (
    predecessor.predecessor.predecessor.predecessor.predecessor._validate_authority_v2  # noqa: SLF001
)
_CONFIGURATION_LOCK = threading.RLock()


def _standard_review_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(value["path"]),
        "sha256": str(value["file_sha256"]),
        "byte_count": int(value["byte_count"]),
    }


_EXACT_V1_REVIEW_BINDING = _standard_review_binding(
    plan_builder.predecessor.predecessor.predecessor.V1_QUALIFICATION_TERMINAL_REVIEW_BINDING
)
_EXACT_V2_REVIEW_BINDING = _standard_review_binding(
    plan_builder.predecessor.predecessor.v2_qualification_terminal_review_binding()
)
_EXACT_V3_REVIEW_BINDING = _standard_review_binding(
    plan_builder.predecessor.v3_qualification_terminal_review_binding()
)
_EXACT_REPLACEMENT_V1_REVIEW_BINDING = _standard_review_binding(
    plan_builder.replacement_v1_qualification_terminal_review_binding()
)


def _require_exact_review_binding(
    authority: Mapping[str, Any],
    *,
    field: str,
    expected: Mapping[str, Any],
    label: str,
) -> None:
    path = Path(str(expected["path"]))
    actual = _standard_review_binding(pilot.file_binding(path))
    if authority.get(field) != dict(expected) or actual != dict(expected):
        raise SceneProcessCollectionError(
            f"replacement authority {label} terminal-review binding changed"
        )


def _require_v1_review_binding(authority: Mapping[str, Any]) -> None:
    _require_exact_review_binding(
        authority,
        field="predecessor_v1_qualification_terminal_review_binding",
        expected=_EXACT_V1_REVIEW_BINDING,
        label="V1",
    )


def _require_v2_review_binding(authority: Mapping[str, Any]) -> None:
    _require_exact_review_binding(
        authority,
        field="predecessor_v2_qualification_terminal_review_binding",
        expected=_EXACT_V2_REVIEW_BINDING,
        label="V2",
    )


def _require_v3_review_binding(authority: Mapping[str, Any]) -> None:
    _require_exact_review_binding(
        authority,
        field="predecessor_v3_qualification_terminal_review_binding",
        expected=_EXACT_V3_REVIEW_BINDING,
        label="V3",
    )


def _standard_replacement_v1_review_binding() -> dict[str, Any]:
    return dict(_EXACT_REPLACEMENT_V1_REVIEW_BINDING)


def _require_replacement_v1_review_binding(
    authority: Mapping[str, Any]
) -> None:
    _require_exact_review_binding(
        authority,
        field="predecessor_replacement_v1_qualification_terminal_review_binding",
        expected=_EXACT_REPLACEMENT_V1_REVIEW_BINDING,
        label="replacement V1",
    )


def _validate_authority_replacement_v2_review_bound(
    *args: Any, **kwargs: Any
):
    authority = _LOWEST_IMMUTABLE_BASE_VALIDATE_AUTHORITY(*args, **kwargs)
    _require_v1_review_binding(authority)
    _require_v2_review_binding(authority)
    _require_v3_review_binding(authority)
    _require_replacement_v1_review_binding(authority)
    return authority


_BASE_COLLECTOR_OVERRIDE_KEYS = frozenset(
    {
        "ATTEMPT_ID",
        "AUTHORITY_FIELDS",
        "AUTHORITY_SCHEMA",
        "AUTHORITY_STATUS",
        "PROCESS_RESET_EQUIVALENCE_AUDIT_V2",
        "RESERVATION_SCHEMA",
        "SCENE_EVIDENCE_SCHEMA",
        "SCENE_RESULT_SCHEMA",
        "_initialize_from_plan_first_scene_v2",
        "_read_collection_reservation_v2",
        "_worker_argv_v2",
    }
)
_V2_COLLECTOR_OVERRIDE_KEYS = frozenset(
    {
        "plan_builder",
        "AUTHORITY_FIELDS",
        "AUTHORITY_SCHEMA",
        "AUTHORITY_STATUS",
        "ATTEMPT_ID",
        "RESERVATION_SCHEMA",
        "SCENE_RESULT_SCHEMA",
        "SCENE_EVIDENCE_SCHEMA",
        "CONTACT_FORCE_ROUTE_AUDIT",
        "PROCESS_RESET_EQUIVALENCE_AUDIT_ROCM",
        "PROCESS_RESET_EQUIVALENCE_AUDIT_V2",
        "_expected_output_root_for_identity",
        "_plan_role_for_identity",
        "_validate_rocm_plan_runtime",
        "_read_collection_reservation_rocm",
        "_worker_argv_rocm",
    }
)
_V3_COLLECTOR_OVERRIDE_KEYS = frozenset(
    set(_V2_COLLECTOR_OVERRIDE_KEYS)
    | {
        "ROCM_ADDITIONAL_SANITIZED_KEYS",
        "_validate_authority_v2_review_bound",
    }
)
_REPLACEMENT_V1_COLLECTOR_OVERRIDE_KEYS = frozenset(
    (set(_V3_COLLECTOR_OVERRIDE_KEYS) - {
        "PROCESS_RESET_EQUIVALENCE_AUDIT_V2",
        "_validate_authority_v2_review_bound",
    })
    | {
        "PROCESS_RESET_EQUIVALENCE_AUDIT_V3",
        "_validate_authority_v3_review_bound",
    }
)
_REPLACEMENT_V2_COLLECTOR_OVERRIDE_KEYS = frozenset(
    (set(_REPLACEMENT_V1_COLLECTOR_OVERRIDE_KEYS) - {
        "_validate_authority_v3_review_bound",
    })
    | {"_validate_authority_replacement_review_bound"}
)


COLLECTOR_COMPATIBILITY_MATRIX = MappingProxyType(
    {
        "all_inherited_qualifiers": (
            "pilot",
            "kernel",
            "predecessor",
            "__file__",
            "EXPECTED_CAPS",
            "EXPECTED_PERMISSIONS",
            "CONTACT_FORCE_ROUTE_AUDIT",
            "ROCM_ADDITIONAL_SANITIZED_KEYS",
            "_worker_argv_rocm",
            "_configured_predecessor_collector_rocm",
            "_selected_gpu_memory_files_rocm",
            "_validate_rocm_plan_runtime",
            "require_exact_orchestrator_python",
            "require_exact_orchestrator_environment",
            "build_parser",
            "main",
            "_require_v1_review_binding",
            "_require_v2_review_binding",
            "_require_v3_review_binding",
            "_require_replacement_v1_review_binding",
        ),
        "v1_qualification_collector_overlay": (
            "AUTHORITY_FIELDS",
            "AUTHORITY_SCHEMA",
            "AUTHORITY_STATUS",
            "ATTEMPT_ID",
            "RESERVATION_SCHEMA",
            "SCENE_RESULT_SCHEMA",
            "_read_collection_reservation_rocm",
            "_worker_argv_rocm",
        ),
        "replacement_v2_to_replacement_v1": tuple(
            sorted(_REPLACEMENT_V2_COLLECTOR_OVERRIDE_KEYS)
        ),
        "replacement_v1_to_v3": tuple(
            sorted(_REPLACEMENT_V1_COLLECTOR_OVERRIDE_KEYS)
        ),
        "v3_to_v2": tuple(sorted(_V3_COLLECTOR_OVERRIDE_KEYS)),
        "v2_to_v1": tuple(sorted(_V2_COLLECTOR_OVERRIDE_KEYS)),
        "v1_to_base": tuple(sorted(_BASE_COLLECTOR_OVERRIDE_KEYS)),
    }
)


def validate_collector_compatibility_facade() -> dict[str, Any]:
    missing = {
        layer: [
            name
            for name in COLLECTOR_COMPATIBILITY_MATRIX[layer]
            if name not in globals()
        ]
        for layer in (
            "all_inherited_qualifiers",
            "v1_qualification_collector_overlay",
        )
    }
    missing = {layer: names for layer, names in missing.items() if names}
    if missing:
        raise SceneProcessCollectionError(
            f"replacement V2 collector compatibility facade is incomplete: {missing}"
        )
    replacement_v1 = predecessor
    v3 = replacement_v1.predecessor
    v2 = v3.predecessor
    v1 = v2.predecessor
    observed = {
        "replacement_v2_to_replacement_v1": set(
            _configuration_overrides_v3()
        ),
        "replacement_v1_to_v3": set(
            replacement_v1._configuration_overrides_v3()  # noqa: SLF001
        ),
        "v3_to_v2": set(v3._configuration_overrides_v3()),  # noqa: SLF001
        "v2_to_v1": set(v2._configuration_overrides_v2()),  # noqa: SLF001
        "v1_to_base": set(v1._configuration_overrides_rocm()),  # noqa: SLF001
    }
    expected = {
        "replacement_v2_to_replacement_v1": (
            _REPLACEMENT_V2_COLLECTOR_OVERRIDE_KEYS
        ),
        "replacement_v1_to_v3": _REPLACEMENT_V1_COLLECTOR_OVERRIDE_KEYS,
        "v3_to_v2": _V3_COLLECTOR_OVERRIDE_KEYS,
        "v2_to_v1": _V2_COLLECTOR_OVERRIDE_KEYS,
        "v1_to_base": _BASE_COLLECTOR_OVERRIDE_KEYS,
    }
    mismatches = {
        layer: {
            "missing": sorted(set(expected[layer]) - values),
            "unexpected": sorted(values - set(expected[layer])),
        }
        for layer, values in observed.items()
        if values != set(expected[layer])
    }
    if mismatches:
        raise SceneProcessCollectionError(
            f"replacement V2 collector layered interface changed: {mismatches}"
        )
    return {
        "schema": (
            "lewm_go2_scene_diversity_recurrent_replication_genesis_rocm_"
            "backend_v3_integrity_replacement_v2_collector_compatibility_v1"
        ),
        "layers": {
            layer: list(names)
            for layer, names in COLLECTOR_COMPATIBILITY_MATRIX.items()
        },
        "passed": True,
    }


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
    except plan_builder.SceneDiversityGenesisRocmV3IntegrityReplacementV2PlanError as exc:
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
            "V3 worker argv changed from the exact replacement launcher contract"
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
        "PROCESS_RESET_EQUIVALENCE_AUDIT_V3": (
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
        "_validate_authority_replacement_review_bound": (
            _validate_authority_replacement_v2_review_bound
        ),
    }


@contextmanager
def _configured_predecessor_collector_rocm() -> Iterator[None]:
    """Install V3 identity before entering the inherited runtime adapter."""

    validate_collector_compatibility_facade()
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
    validate_collector_compatibility_facade()
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
    validate_collector_compatibility_facade()
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
