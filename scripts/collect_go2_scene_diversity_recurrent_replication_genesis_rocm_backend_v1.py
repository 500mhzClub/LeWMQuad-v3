#!/usr/bin/env python3
"""Collect the scene-diversity successor with Genesis 0.4.6 ``gs.amdgpu``.

This module is deliberately a narrow adapter around the reviewed one-scene
process collector.  It changes backend/runtime identity, selects the known
R9700-safe no-contact-read configuration, and leaves the data, trajectories,
render contract, counts, resource caps, and downstream science unchanged.

No invocation of this source is self-authorizing.  Qualification and any
later scientific collection require separately reviewed one-shot authorities.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import os
from pathlib import Path
import re
import sys
import threading
from typing import Any, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1_plan as plan_builder  # noqa: E402
from scripts import collect_go2_scene_diversity_recurrent_replication_integrity_replacement_v2 as predecessor  # noqa: E402


AUTHORITY_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1_"
    "execution_authority_v1"
)
AUTHORITY_STATUS = (
    "AUTHORIZED_SCENE_DIVERSITY_RECURRENT_REPLICATION_GENESIS_ROCM_BACKEND_V1"
)
ATTEMPT_ID = plan_builder.DEFAULT_ATTEMPT_ID
RESERVATION_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1_"
    "collection_reservation_v1"
)
SCENE_RESULT_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1_"
    "scene_physics_result_v1"
)
SCENE_EVIDENCE_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1_"
    "scene_process_evidence_v1"
)

AUTHORITY_FIELDS = frozenset(
    set(predecessor.AUTHORITY_FIELDS)
    | {"qualification_result_binding", "predecessor_cpu_terminal_review_binding"}
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

CONTACT_FORCE_ROUTE_AUDIT = {
    "schema": (
        "lewm_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1_"
        "contact_force_route_source_audit_v1"
    ),
    "known_bad_api": "robot.get_links_net_contact_force",
    "selected_entrypoint": "RolloutRunner.execute_requested_block",
    "selected_entrypoint_emits_per_tick_records": False,
    "selected_entrypoint_calls_extract_foot_contacts": False,
    "rollout_config_foot_contact_source": "zero",
    "known_bad_api_reachable": False,
}

PROCESS_RESET_EQUIVALENCE_AUDIT_ROCM = {
    **copy.deepcopy(predecessor.PROCESS_RESET_EQUIVALENCE_AUDIT_V2),
    "schema": (
        "lewm_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1_"
        "process_reset_equivalence_source_audit_v1"
    ),
    "execution_backend": "amdgpu",
    "genesis_version": "0.4.6",
    "backend_api": "gs.amdgpu",
    "physics_numerics_may_differ_from_vulkan_and_cpu": True,
    "contact_force_route_audit": copy.deepcopy(CONTACT_FORCE_ROUTE_AUDIT),
}
PROCESS_RESET_EQUIVALENCE_AUDIT_V2 = PROCESS_RESET_EQUIVALENCE_AUDIT_ROCM

_ORIGINAL_WORKER_ARGV = predecessor._worker_argv_v2  # noqa: SLF001
_ORIGINAL_BUILD_ROLLOUT_RUNNER = kernel._build_rollout_runner  # noqa: SLF001
_read_collection_reservation_rocm = predecessor._read_collection_reservation_v2  # noqa: SLF001
_CONFIGURATION_LOCK = threading.RLock()

# Remove every ambient selector consumed by the Genesis 0.4.6 / Quadrants
# runtime before applying the exact plan-bound environment.  The inherited
# collector already owns the general Python, graphics, and accelerator
# selector set; this is the focused delta found in the installed sources.
ROCM_ADDITIONAL_SANITIZED_KEYS = frozenset(
    {
        "PATH",
        "ROCM_PATH",
        "GS_CACHE_FILE_PATH",
        "GS_DISABLE_OFFSCREEN_MARKERS",
        "GS_ENABLE_FASTCACHE",
        "GS_ENABLE_NDARRAY",
        "GS_ENABLE_ZEROCOPY",
        "GS_TORCH_FORCE_CPU_DEVICE",
        "QD_ARCH",
        "QD_CI",
        "QD_DEBUG",
        "QD_DEFAULT_FP",
        "QD_DEFAULT_IP",
        "QD_DUMP_AST",
        "QD_DUMP_IR",
        "QD_ENABLE_AMDGPU",
        "QD_ENABLE_PYBUF",
        "QD_ENABLE_TORCH",
        "QD_GRAPH",
        "QD_IN_DOCKER",
        "QD_LOG_LEVEL",
        "QD_MANYLINUX2014_OK",
        "QD_NUM_THREADS",
        "QD_PERFDISPATCH_FORCE",
        "QD_PERFDISPATCH_PRINT_DEBUG",
    }
)


def _expected_output_root_for_identity() -> Path:
    if ATTEMPT_ID == plan_builder.DEFAULT_ATTEMPT_ID:
        return plan_builder.DEFAULT_OUTPUT_ROOT
    if ATTEMPT_ID == plan_builder.QUALIFICATION_ATTEMPT_ID:
        return plan_builder.QUALIFICATION_OUTPUT_ROOT
    raise SceneProcessCollectionError("ROCm collector attempt identity changed")


def _plan_role_for_identity() -> str:
    if ATTEMPT_ID == plan_builder.DEFAULT_ATTEMPT_ID:
        return "scientific"
    if ATTEMPT_ID == plan_builder.QUALIFICATION_ATTEMPT_ID:
        return "qualification"
    raise SceneProcessCollectionError("ROCm collector attempt role changed")


def _validate_rocm_plan_runtime(value: Mapping[str, Any]) -> dict[str, Any]:
    """Use the builder's import-time immutable frozen validator."""

    try:
        return plan_builder.validate_rocm_plan(
            value,
            expected_attempt_id=ATTEMPT_ID,
            expected_output_root=_expected_output_root_for_identity(),
            plan_role=_plan_role_for_identity(),
        )
    except plan_builder.SceneDiversityGenesisRocmPlanError as exc:
        raise pilot.PilotContractError(str(exc)) from exc


def _initialize_from_plan_first_scene_rocm(
    *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Retain the plan-first seed while selecting exact ``gs.amdgpu``."""

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
        or backend != "amdgpu"
    ):
        raise SceneProcessCollectionError(
            "ROCm plan-first process-global Genesis seed changed"
        )
    if "HSA_OVERRIDE_GFX_VERSION" in os.environ:
        raise SceneProcessCollectionError(
            "HSA_OVERRIDE_GFX_VERSION must remain absent on gfx1201"
        )
    predecessor._initialize_genesis_v2(  # noqa: SLF001
        backend="amdgpu", seed=int(full_seed)
    )
    return {
        "source": "full_plan_first_scene_bound_manifest",
        "state_id": str(state["state_id"]),
        "scene_id": str(state["scene_id"]),
        "manifest_binding": dict(binding),
        "backend": backend,
        "backend_api": "gs.amdgpu",
        "full_physics_seed": int(full_seed),
        "effective_genesis_seed": int(effective),
        "hsa_override_gfx_version_present": "HSA_OVERRIDE_GFX_VERSION" in os.environ,
    }


def _build_rollout_runner_rocm(**kwargs: Any) -> Any:
    """Force the R9700-safe zero-contact configuration on the selected route.

    ``execute_requested_block`` does not emit telemetry and therefore never
    asks for contacts today.  Supplying ``foot_contact_source='zero'`` also
    makes that safety property explicit if the shared constructor evolves.
    """

    runtime = dict(kwargs["runtime"])
    rollout_config = runtime["RolloutConfig"]

    def safe_rollout_config(**config_kwargs: Any) -> Any:
        if "foot_contact_source" in config_kwargs:
            raise SceneProcessCollectionError(
                "ROCm rollout contact-source ownership changed"
            )
        return rollout_config(foot_contact_source="zero", **config_kwargs)

    runtime["RolloutConfig"] = safe_rollout_config
    return _ORIGINAL_BUILD_ROLLOUT_RUNNER(
        plan=kwargs["plan"],
        runtime=runtime,
        platform=kwargs["platform"],
        build=kwargs["build"],
        registry=kwargs["registry"],
    )


def _selected_gpu_memory_files_rocm(
    plan: Mapping[str, Any], *, drm_root: Path = Path("/sys/class/drm")
) -> tuple[Path, Path, str, str]:
    """Resolve the exact R9700 VRAM counters without Vulkan terminology."""

    expectation = plan["execution_contract"]["graphics_preflight"]
    vendor_id = str(expectation["drm_vendor_id"]).lower()
    device_id = str(expectation["drm_device_id"]).lower()
    matches: list[tuple[Path, Path]] = []
    for card in sorted(drm_root.glob("card[0-9]*")):
        if re.fullmatch(r"card[0-9]+", card.name) is None:
            continue
        device = card / "device"
        try:
            observed_vendor = (device / "vendor").read_text().strip().lower()
            observed_device = (device / "device").read_text().strip().lower()
        except OSError:
            continue
        if observed_vendor == vendor_id and observed_device == device_id:
            matches.append(
                (device / "mem_info_vram_used", device / "mem_info_vram_total")
            )
    if len(matches) != 1 or any(not path.is_file() for path in matches[0]):
        raise SceneProcessCollectionError(
            "bound R9700 does not expose one exact sysfs VRAM counter pair"
        )
    return matches[0][0], matches[0][1], vendor_id, device_id


def _worker_argv_rocm(**kwargs: Any) -> list[str]:
    argv = _ORIGINAL_WORKER_ARGV(**kwargs)
    if len(argv) < 2 or Path(argv[1]).resolve() != Path(predecessor.__file__).resolve():
        raise SceneProcessCollectionError("predecessor worker entry point changed")
    argv[1] = str(Path(__file__).resolve())
    return argv


def _configuration_overrides_rocm() -> dict[str, object]:
    return {
        "AUTHORITY_FIELDS": AUTHORITY_FIELDS,
        "AUTHORITY_SCHEMA": AUTHORITY_SCHEMA,
        "AUTHORITY_STATUS": AUTHORITY_STATUS,
        "ATTEMPT_ID": ATTEMPT_ID,
        "RESERVATION_SCHEMA": RESERVATION_SCHEMA,
        "SCENE_RESULT_SCHEMA": SCENE_RESULT_SCHEMA,
        "SCENE_EVIDENCE_SCHEMA": SCENE_EVIDENCE_SCHEMA,
        "PROCESS_RESET_EQUIVALENCE_AUDIT_V2": PROCESS_RESET_EQUIVALENCE_AUDIT_ROCM,
        "_initialize_from_plan_first_scene_v2": _initialize_from_plan_first_scene_rocm,
        "_read_collection_reservation_v2": _read_collection_reservation_rocm,
        "_worker_argv_v2": _worker_argv_rocm,
    }


@contextmanager
def _configured_predecessor_collector_rocm() -> Iterator[None]:
    """Apply the small runtime adapter and restore every shared symbol.

    Plan validation itself is immutable: ``validate_rocm_plan`` normalizes
    through validator references captured before this context can alter the
    shared pilot module.  This is the regression boundary missed by CPU V1.
    """

    with _CONFIGURATION_LOCK:
        overrides = _configuration_overrides_rocm()
        originals = {name: getattr(predecessor, name) for name in overrides}
        original_validate = pilot.validate_plan
        original_environment = pilot.EXECUTION_ENVIRONMENT
        original_graphics = pilot.GRAPHICS_PREFLIGHT_EXPECTATION
        original_build_runner = kernel._build_rollout_runner  # noqa: SLF001
        original_memory_selector = (
            predecessor.calibration_supervisor._selected_gpu_memory_files  # noqa: SLF001
        )
        original_sanitized_selectors = kernel._SANITIZED_SELECTOR_KEYS  # noqa: SLF001
        try:
            for name, value in overrides.items():
                setattr(predecessor, name, value)
            pilot.validate_plan = _validate_rocm_plan_runtime
            pilot.EXECUTION_ENVIRONMENT = copy.deepcopy(
                plan_builder.rocm_execution_environment(
                    _plan_role_for_identity()
                )
            )
            pilot.GRAPHICS_PREFLIGHT_EXPECTATION = copy.deepcopy(
                plan_builder.ROCM_GRAPHICS_PREFLIGHT_EXPECTATION
            )
            kernel._build_rollout_runner = _build_rollout_runner_rocm  # noqa: SLF001
            predecessor.calibration_supervisor._selected_gpu_memory_files = (  # noqa: SLF001
                _selected_gpu_memory_files_rocm
            )
            kernel._SANITIZED_SELECTOR_KEYS = (  # noqa: SLF001
                set(original_sanitized_selectors)
                | set(ROCM_ADDITIONAL_SANITIZED_KEYS)
            )
            yield
        finally:
            kernel._SANITIZED_SELECTOR_KEYS = original_sanitized_selectors  # noqa: SLF001
            predecessor.calibration_supervisor._selected_gpu_memory_files = (  # noqa: SLF001
                original_memory_selector
            )
            kernel._build_rollout_runner = original_build_runner  # noqa: SLF001
            pilot.GRAPHICS_PREFLIGHT_EXPECTATION = original_graphics
            pilot.EXECUTION_ENVIRONMENT = original_environment
            pilot.validate_plan = original_validate
            for name, value in originals.items():
                setattr(predecessor, name, value)


def load_and_validate_rocm(*args: Any, **kwargs: Any):
    with _configured_predecessor_collector_rocm():
        return predecessor.load_and_validate_v2(*args, **kwargs)


load_and_validate_v2 = load_and_validate_rocm
load_and_validate_replacement_v2 = load_and_validate_rocm


def validate_scene_process_evidence_rocm(*args: Any, **kwargs: Any):
    with _configured_predecessor_collector_rocm():
        return predecessor.validate_scene_process_evidence_v2(*args, **kwargs)


validate_scene_process_evidence_v2 = validate_scene_process_evidence_rocm


def validate_scene_process_closure_rocm(*args: Any, **kwargs: Any):
    with _configured_predecessor_collector_rocm():
        return predecessor.validate_scene_process_closure_v2(*args, **kwargs)


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
    with _configured_predecessor_collector_rocm():
        return predecessor.collect_v2(
            plan_path=plan_path,
            expected_plan_byte_count=expected_plan_byte_count,
            expected_plan_sha256=expected_plan_sha256,
            authority_path=authority_path,
            expected_authority_byte_count=expected_authority_byte_count,
            expected_authority_sha256=expected_authority_sha256,
        )


collect_v2 = collect_rocm
collect_v1 = collect_rocm


def build_parser():
    parser = predecessor.build_parser()
    parser.description = (
        "Collect the preregistered Genesis 0.4.6 ROCm/HIP successor with "
        "one fresh gs.amdgpu process per scene."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
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
    "PLAN_FIRST_EFFECTIVE_GENESIS_SEED",
    "PLAN_FIRST_PHYSICS_SEED",
    "PROCESS_RESET_EQUIVALENCE_AUDIT_ROCM",
    "PROCESS_RESET_EQUIVALENCE_AUDIT_V2",
    "ROCM_ADDITIONAL_SANITIZED_KEYS",
    "ROLE_ORDER",
    "SCENE_COUNT",
    "SCENE_EVIDENCE_SCHEMA",
    "SCENE_EVIDENCE_STATUS",
    "SceneProcessCollectionError",
    "collect_rocm",
    "collect_v1",
    "collect_v2",
    "load_and_validate_rocm",
    "pilot",
    "bounded",
    "predecessor",
    "validate_scene_process_closure_rocm",
    "validate_scene_process_evidence_rocm",
]
