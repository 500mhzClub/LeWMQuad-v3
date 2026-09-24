#!/usr/bin/env python3
"""Run the reviewed one-scene collector with Genesis physics on CPU.

Only the material Genesis backend/environment selector and fresh successor
identity differ from V2.  EGL/R9700 rendering selectors, one-scene process
policy, resource monitoring, data, joins and all scientific inputs remain the
reviewed implementation.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
from pathlib import Path
import sys
import threading
from typing import Any, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_go2_scene_diversity_recurrent_replication_cpu_backend_v1_plan as plan_builder  # noqa: E402
from scripts import collect_go2_scene_diversity_recurrent_replication_integrity_replacement_v2 as predecessor  # noqa: E402


AUTHORITY_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_cpu_backend_v1_"
    "execution_authority_v1"
)
AUTHORITY_STATUS = "AUTHORIZED_SCENE_DIVERSITY_RECURRENT_REPLICATION_CPU_BACKEND_V1"
ATTEMPT_ID = plan_builder.DEFAULT_ATTEMPT_ID
RESERVATION_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_cpu_backend_v1_"
    "collection_reservation_v1"
)
SCENE_RESULT_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_cpu_backend_v1_"
    "scene_physics_result_v1"
)
SCENE_EVIDENCE_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_cpu_backend_v1_"
    "scene_process_evidence_v1"
)

AUTHORITY_FIELDS = frozenset(
    set(predecessor.AUTHORITY_FIELDS) | {"qualification_result_binding"}
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

PROCESS_RESET_EQUIVALENCE_AUDIT_CPU = {
    **copy.deepcopy(predecessor.PROCESS_RESET_EQUIVALENCE_AUDIT_V2),
    "schema": (
        "lewm_go2_scene_diversity_recurrent_replication_cpu_backend_v1_"
        "process_reset_equivalence_source_audit_v1"
    ),
    "execution_backend": "cpu",
    "physics_numerics_may_differ_from_vulkan": True,
}
PROCESS_RESET_EQUIVALENCE_AUDIT_V2 = PROCESS_RESET_EQUIVALENCE_AUDIT_CPU

_ORIGINAL_WORKER_ARGV = predecessor._worker_argv_v2  # noqa: SLF001
_read_collection_reservation_cpu = predecessor._read_collection_reservation_v2  # noqa: SLF001
_CONFIGURATION_LOCK = threading.RLock()


def _expected_output_root_for_identity() -> Path:
    if ATTEMPT_ID == plan_builder.DEFAULT_ATTEMPT_ID:
        return plan_builder.DEFAULT_OUTPUT_ROOT
    if ATTEMPT_ID == plan_builder.QUALIFICATION_ATTEMPT_ID:
        return plan_builder.QUALIFICATION_OUTPUT_ROOT
    raise SceneProcessCollectionError("CPU collector attempt identity changed")


def _validate_cpu_plan_runtime(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return plan_builder.validate_cpu_plan(
            value,
            expected_attempt_id=ATTEMPT_ID,
            expected_output_root=_expected_output_root_for_identity(),
        )
    except plan_builder.SceneDiversityCpuBackendPlanError as exc:
        raise pilot.PilotContractError(str(exc)) from exc


def _initialize_from_plan_first_scene_cpu(
    *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Retain V2's plan-first seed while selecting exact Genesis CPU."""

    state = plan["states"][0]
    binding = state["scene_manifest_binding"]
    manifest, actual = pilot.read_bound_json(
        Path(str(binding["path"])),
        expected_sha256=str(binding["file_sha256"]),
        expected_byte_count=int(binding["byte_count"]),
        label="plan-first Genesis seed manifest",
    )
    full_seed = manifest.get("physics_seed")
    effective = int(full_seed) & 0x7FFF_FFFF if type(full_seed) is int else None
    backend = str(plan["execution_contract"]["backend"])
    if (
        actual != binding
        or state.get("scene_id") != "large_enclosed_maze_8a6599d5327d"
        or full_seed != PLAN_FIRST_PHYSICS_SEED
        or effective != PLAN_FIRST_EFFECTIVE_GENESIS_SEED
        or backend != "cpu"
    ):
        raise SceneProcessCollectionError(
            "CPU plan-first process-global Genesis seed changed"
        )
    predecessor._initialize_genesis_v2(  # noqa: SLF001
        backend=backend, seed=int(full_seed)
    )
    return {
        "source": "full_plan_first_scene_bound_manifest",
        "state_id": str(state["state_id"]),
        "scene_id": str(state["scene_id"]),
        "manifest_binding": dict(binding),
        "backend": backend,
        "full_physics_seed": int(full_seed),
        "effective_genesis_seed": int(effective),
    }


def _worker_argv_cpu(**kwargs: Any) -> list[str]:
    argv = _ORIGINAL_WORKER_ARGV(**kwargs)
    if len(argv) < 2 or Path(argv[1]).resolve() != Path(predecessor.__file__).resolve():
        raise SceneProcessCollectionError("predecessor worker entry point changed")
    argv[1] = str(Path(__file__).resolve())
    return argv


def _configuration_overrides_cpu() -> dict[str, object]:
    return {
        "AUTHORITY_FIELDS": AUTHORITY_FIELDS,
        "AUTHORITY_SCHEMA": AUTHORITY_SCHEMA,
        "AUTHORITY_STATUS": AUTHORITY_STATUS,
        "ATTEMPT_ID": ATTEMPT_ID,
        "RESERVATION_SCHEMA": RESERVATION_SCHEMA,
        "SCENE_RESULT_SCHEMA": SCENE_RESULT_SCHEMA,
        "SCENE_EVIDENCE_SCHEMA": SCENE_EVIDENCE_SCHEMA,
        "PROCESS_RESET_EQUIVALENCE_AUDIT_V2": PROCESS_RESET_EQUIVALENCE_AUDIT_CPU,
        "_initialize_from_plan_first_scene_v2": _initialize_from_plan_first_scene_cpu,
        "_read_collection_reservation_v2": _read_collection_reservation_cpu,
        "_worker_argv_v2": _worker_argv_cpu,
    }


@contextmanager
def _configured_predecessor_collector_cpu() -> Iterator[None]:
    """Apply and restore only CPU backend/identity deltas."""

    with _CONFIGURATION_LOCK:
        overrides = _configuration_overrides_cpu()
        originals = {name: getattr(predecessor, name) for name in overrides}
        original_validate = pilot.validate_plan
        original_environment = pilot.EXECUTION_ENVIRONMENT
        try:
            for name, value in overrides.items():
                setattr(predecessor, name, value)
            pilot.validate_plan = _validate_cpu_plan_runtime
            pilot.EXECUTION_ENVIRONMENT = copy.deepcopy(
                plan_builder.CPU_EXECUTION_ENVIRONMENT
            )
            yield
        finally:
            pilot.validate_plan = original_validate
            pilot.EXECUTION_ENVIRONMENT = original_environment
            for name, value in originals.items():
                setattr(predecessor, name, value)


def load_and_validate_cpu(*args: Any, **kwargs: Any):
    with _configured_predecessor_collector_cpu():
        return predecessor.load_and_validate_v2(*args, **kwargs)


load_and_validate_v2 = load_and_validate_cpu
load_and_validate_replacement_v2 = load_and_validate_cpu


def validate_scene_process_evidence_cpu(*args: Any, **kwargs: Any):
    with _configured_predecessor_collector_cpu():
        return predecessor.validate_scene_process_evidence_v2(*args, **kwargs)


validate_scene_process_evidence_v2 = validate_scene_process_evidence_cpu


def validate_scene_process_closure_cpu(*args: Any, **kwargs: Any):
    with _configured_predecessor_collector_cpu():
        return predecessor.validate_scene_process_closure_v2(*args, **kwargs)


validate_scene_process_closure_v2 = validate_scene_process_closure_cpu


def collect_cpu(
    *,
    plan_path: Path,
    expected_plan_byte_count: int,
    expected_plan_sha256: str,
    authority_path: Path,
    expected_authority_byte_count: int,
    expected_authority_sha256: str,
):
    with _configured_predecessor_collector_cpu():
        return predecessor.collect_v2(
            plan_path=plan_path,
            expected_plan_byte_count=expected_plan_byte_count,
            expected_plan_sha256=expected_plan_sha256,
            authority_path=authority_path,
            expected_authority_byte_count=expected_authority_byte_count,
            expected_authority_sha256=expected_authority_sha256,
        )


collect_v2 = collect_cpu
collect_v1 = collect_cpu


def build_parser():
    parser = predecessor.build_parser()
    parser.description = (
        "Collect the preregistered CPU-backend successor, with one fresh "
        "Genesis process per scene."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    with _configured_predecessor_collector_cpu():
        return predecessor.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ATTEMPT_ID",
    "AUTHORITY_FIELDS",
    "AUTHORITY_SCHEMA",
    "AUTHORITY_STATUS",
    "EXPECTED_CAPS",
    "EXPECTED_COUNTS",
    "EXPECTED_HISTORY_PANEL",
    "EXPECTED_PERMISSIONS",
    "PLAN_FIRST_EFFECTIVE_GENESIS_SEED",
    "PLAN_FIRST_PHYSICS_SEED",
    "PROCESS_RESET_EQUIVALENCE_AUDIT_CPU",
    "PROCESS_RESET_EQUIVALENCE_AUDIT_V2",
    "ROLE_ORDER",
    "SCENE_COUNT",
    "SCENE_EVIDENCE_SCHEMA",
    "SCENE_EVIDENCE_STATUS",
    "SceneProcessCollectionError",
    "collect_cpu",
    "collect_v1",
    "collect_v2",
    "load_and_validate_cpu",
    "pilot",
    "bounded",
    "validate_scene_process_closure_cpu",
    "validate_scene_process_evidence_cpu",
]
