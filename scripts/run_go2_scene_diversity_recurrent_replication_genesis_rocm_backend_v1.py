#!/usr/bin/env python3
"""Run the qualified Genesis 0.4.6 ROCm/HIP scientific successor.

The reviewed integrity-replacement V3 runner remains the custody, model,
evaluation, and scientific implementation.  This wrapper changes only the
simulator/runtime identity, requires the consumed CPU qualification terminal
review, and independently revalidates an exact two-scene ROCm qualification
PASS before the fresh scientific attempt can begin.  Qualification scene/RGB
payloads are never inputs to the scientific route.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import threading
from typing import Any, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1_plan as plan_builder  # noqa: E402
from scripts import collect_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1 as collector  # noqa: E402
from scripts import qualify_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1 as qualifier  # noqa: E402
from scripts import run_go2_scene_diversity_recurrent_replication_integrity_replacement_v3 as predecessor_runner  # noqa: E402


v2_runner = predecessor_runner.predecessor_runner
v1_replacement_runner = v2_runner.predecessor_runner
frozen_runner = predecessor_runner.frozen_runner

AUTHORITY_SCHEMA = collector.AUTHORITY_SCHEMA
AUTHORITY_STATUS = collector.AUTHORITY_STATUS
SOURCE_REVIEW_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v1_source_review_v1"
)
SOURCE_REVIEW_STATUS = "PASS_INDEPENDENT_SOURCE_REVIEW"
RESULT_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v1_result_v1"
)
TERMINAL_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v1_terminal_v1"
)
RESERVATION_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v1_attempt_reservation_v1"
)
PREREGISTRATION = qualifier.PREREGISTRATION
SOURCE_REVIEW = qualifier.SOURCE_REVIEW
SCENE_PANEL = predecessor_runner.SCENE_PANEL
SCENE_PANEL_SHA256 = predecessor_runner.SCENE_PANEL_SHA256
SCENE_PANEL_BYTE_COUNT = predecessor_runner.SCENE_PANEL_BYTE_COUNT
DEFAULT_ATTEMPT_ROOT = plan_builder.DEFAULT_ATTEMPT_ROOT
DEFAULT_COLLECTION_ROOT = plan_builder.DEFAULT_OUTPUT_ROOT

CPU_TERMINAL_REVIEW = plan_builder.CPU_TERMINAL_REVIEW
CPU_TERMINAL_REVIEW_SHA256 = plan_builder.CPU_TERMINAL_REVIEW_SHA256
CPU_TERMINAL_REVIEW_BYTE_COUNT = plan_builder.CPU_TERMINAL_REVIEW_BYTE_COUNT

DINO_REPOSITORY = predecessor_runner.DINO_REPOSITORY
DINO_CHECKPOINT = predecessor_runner.DINO_CHECKPOINT
DINO_REPOSITORY_COMMIT = predecessor_runner.DINO_REPOSITORY_COMMIT
DINO_CHECKPOINT_SHA256 = predecessor_runner.DINO_CHECKPOINT_SHA256
DINO_CHECKPOINT_BYTE_COUNT = predecessor_runner.DINO_CHECKPOINT_BYTE_COUNT
PROCESS_RESET_DEPENDENCY_PATHS = predecessor_runner.PROCESS_RESET_DEPENDENCY_PATHS

GENESIS_ROCM_DEPENDENCY_ROOT = plan_builder.ROCM_SITE_PACKAGES
ROCM_BACKEND_DEPENDENCY_PATHS = {
    "rocm_backend_dependency_genesis_constants": GENESIS_ROCM_DEPENDENCY_ROOT
    / "genesis/constants.py",
    "rocm_backend_dependency_rigid_solver": GENESIS_ROCM_DEPENDENCY_ROOT
    / "genesis/engine/solvers/rigid/rigid_solver.py",
    "rocm_backend_dependency_rigid_constraint_solver": (
        GENESIS_ROCM_DEPENDENCY_ROOT
        / "genesis/engine/solvers/rigid/constraint/solver.py"
    ),
    "rocm_backend_dependency_rigid_collider": GENESIS_ROCM_DEPENDENCY_ROOT
    / "genesis/engine/solvers/rigid/collider/collider.py",
    "rocm_backend_dependency_rigid_entity": GENESIS_ROCM_DEPENDENCY_ROOT
    / "genesis/engine/entities/rigid_entity/rigid_entity.py",
    "rocm_backend_dependency_simulator": GENESIS_ROCM_DEPENDENCY_ROOT
    / "genesis/engine/simulator.py",
    "rocm_backend_dependency_engine_mesh": GENESIS_ROCM_DEPENDENCY_ROOT
    / "genesis/engine/mesh.py",
    "rocm_backend_dependency_mesh": GENESIS_ROCM_DEPENDENCY_ROOT
    / "genesis/utils/mesh.py",
    "rocm_backend_dependency_rasterizer_context": GENESIS_ROCM_DEPENDENCY_ROOT
    / "genesis/vis/rasterizer_context.py",
    "rocm_backend_dependency_rigid_narrowphase": GENESIS_ROCM_DEPENDENCY_ROOT
    / "genesis/engine/solvers/rigid/collider/narrowphase.py",
    "rocm_backend_dependency_pyrender_offscreen": GENESIS_ROCM_DEPENDENCY_ROOT
    / "genesis/ext/pyrender/offscreen.py",
    "rocm_backend_dependency_pyrender_renderer": GENESIS_ROCM_DEPENDENCY_ROOT
    / "genesis/ext/pyrender/renderer.py",
    "rocm_backend_dependency_rsl_ppo": GENESIS_ROCM_DEPENDENCY_ROOT
    / "rsl_rl/algorithms/ppo.py",
    "rocm_backend_dependency_rsl_mlp_model": GENESIS_ROCM_DEPENDENCY_ROOT
    / "rsl_rl/models/mlp_model.py",
    "rocm_backend_dependency_rsl_mlp": GENESIS_ROCM_DEPENDENCY_ROOT
    / "rsl_rl/modules/mlp.py",
    "rocm_backend_dependency_rsl_utils": GENESIS_ROCM_DEPENDENCY_ROOT
    / "rsl_rl/utils/utils.py",
}

SOURCE_PATHS = {
    **predecessor_runner.SOURCE_PATHS,
    **ROCM_BACKEND_DEPENDENCY_PATHS,
    "rocm_backend_plan_builder": REPO_ROOT
    / "scripts/build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1_plan.py",
    "rocm_backend_collector": REPO_ROOT
    / "scripts/collect_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1.py",
    "rocm_backend_qualifier": REPO_ROOT
    / "scripts/qualify_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1.py",
    "rocm_backend_runner": Path(__file__).resolve(),
    "rocm_backend_qualification_authority_builder": REPO_ROOT
    / "scripts/build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1_qualification_authority.py",
    "rocm_backend_scientific_authority_builder": REPO_ROOT
    / "scripts/build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1_authority.py",
    "rocm_backend_plan_test": REPO_ROOT
    / "lewm/tests/test_build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1_plan.py",
    "rocm_backend_collector_test": REPO_ROOT
    / "lewm/tests/test_collect_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1.py",
    "rocm_backend_qualifier_test": REPO_ROOT
    / "lewm/tests/test_qualify_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1.py",
    "rocm_backend_runner_test": REPO_ROOT
    / "lewm/tests/test_run_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1.py",
    "rocm_backend_qualification_authority_test": REPO_ROOT
    / "lewm/tests/test_build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1_qualification_authority.py",
    "rocm_backend_scientific_authority_test": REPO_ROOT
    / "lewm/tests/test_build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1_authority.py",
    "rocm_backend_preregistration": PREREGISTRATION,
    "rocm_backend_scientific_exact_plan": plan_builder.DEFAULT_PLAN_OUTPUT,
    "rocm_backend_qualification_exact_plan": (
        plan_builder.QUALIFICATION_PLAN_OUTPUT
    ),
    "predecessor_cpu_qualification_terminal_review": CPU_TERMINAL_REVIEW,
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
_V3_FAILURE_BINDINGS = predecessor_runner.predecessor_failure_bindings_v3
_V3_LOAD_PHYSICS_INDEX = predecessor_runner._load_replacement_physics_index_v3  # noqa: SLF001
_CONFIGURATION_LOCK = threading.RLock()

QUALIFICATION_SCENE_RESULT_FIELDS = frozenset(
    {
        "schema",
        "status",
        "attempt_id",
        "scene_index",
        "role",
        "scene_id",
        "worker_pid",
        "orchestrator_pid",
        "sys_executable",
        "fresh_process",
        "execution_seed",
        "genesis_initialization",
        "process_reset_equivalence_audit",
        "scene_local_mesh_cache",
        "plan_binding",
        "authority_binding",
        "collection_reservation_binding",
        "caps",
        "runtime_versions",
        "runtime_bindings",
        "source_bindings",
        "expected_counts",
        "observed_counts",
        "ordered_state_ids",
        "state_receipt_bindings",
        "render_receipt_binding",
        "scene_metric",
        "stored_rgb_bytes",
        "collection_wall_seconds",
        "failure",
        "authorizes_retry_or_resume",
        "allows_refill",
        "allows_overwrite",
        "allows_adaptive_batching",
    }
)
QUALIFICATION_PROBE_FIELDS = frozenset(
    {
        "scene_index",
        "role",
        "scene_id",
        "worker",
        "release_barrier",
        "scene_result_binding",
        "observed_counts",
        "existing_scene_validation_passed",
        "probe_output_scientific_reuse_authorized",
    }
)
QUALIFICATION_WORKER_FIELDS = frozenset(
    {
        "scene_index",
        "role",
        "scene_id",
        "pid",
        "parent_pid",
        "process_group_id",
        "fresh_process_group",
        "sys_executable",
        "prelaunch_baseline_used_bytes",
        "peak_selected_device_vram_bytes",
        "selected_device_vram_cap_breached",
        "watchdog_timeout",
        "exit_code",
        "elapsed_seconds",
    }
)
QUALIFICATION_RELEASE_BARRIER_FIELDS = frozenset(
    {
        "status",
        "read_only",
        "counter_path",
        "baseline_used_bytes",
        "release_margin_bytes",
        "release_ceiling_bytes",
        "absolute_vram_ceiling_bytes",
        "required_consecutive_samples",
        "sample_interval_seconds",
        "sample_count",
        "minimum_used_bytes",
        "maximum_used_bytes",
        "final_used_bytes",
        "final_consecutive_samples",
        "elapsed_seconds",
    }
)
ROCM_EGL_PREFLIGHT_FIELDS = frozenset(
    {
        "status",
        "environment",
        "expectation",
        "identity",
        "egl_device_index",
        "path_ld_lld",
        "rocm_path_ld_lld",
        "lld_stdout_sha256",
        "rocminfo_stdout_sha256",
        "egl_stdout_sha256",
        "egl_stderr_sha256",
        "egl_exit_code",
    }
)
ROCM_IDENTITY_FIELDS = frozenset(
    {
        "torch_version",
        "torch_hip_version",
        "visible_device_count",
        "device_name",
        "arch_name",
        "genesis_version",
        "genesis_backend_symbol",
        "hsa_override_present",
        "genesis_file",
        "torch_file",
        "numpy_file",
        "pillow_file",
    }
)
QUALIFICATION_KERNEL_AUDIT_FIELDS = frozenset(
    {
        "query_succeeded",
        "new_amdgpu_ring_timeout_or_reset_count",
        "matching_lines_sha256",
    }
)
QUALIFICATION_TIMING_GATE_FIELDS = frozenset(
    {
        "cold_scene_12_worker_elapsed_seconds",
        "warm_scene_0_worker_elapsed_seconds",
        "maximum_worker_elapsed_seconds",
        "projected_scientific_wall_seconds",
        "wall_ceiling_seconds",
        "passed",
    }
)


def _expected_binding(
    path: Path, sha256: str, byte_count: int
) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256,
        "byte_count": byte_count,
    }


def _standard_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(value["path"]),
        "sha256": str(value.get("sha256", value.get("file_sha256"))),
        "byte_count": int(value["byte_count"]),
    }


def predecessor_failure_bindings_rocm() -> dict[str, dict[str, object]]:
    """Bind all V3 failures plus the consumed CPU qualification review."""

    evidence = dict(_V3_FAILURE_BINDINGS())
    expected = _expected_binding(
        CPU_TERMINAL_REVIEW,
        CPU_TERMINAL_REVIEW_SHA256,
        CPU_TERMINAL_REVIEW_BYTE_COUNT,
    )
    if file_binding_v1(CPU_TERMINAL_REVIEW) != expected:
        raise SceneDiversityRunnerError(
            "CPU qualification terminal review changed"
        )
    try:
        review = json.loads(CPU_TERMINAL_REVIEW.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SceneDiversityRunnerError(
            "CPU qualification terminal review is not strict JSON"
        ) from exc
    decision = review.get("decision", {}) if isinstance(review, Mapping) else {}
    successor = (
        review.get("successor_eligibility", {})
        if isinstance(review, Mapping)
        else {}
    )
    permission = (
        review.get("permission_audit", {})
        if isinstance(review, Mapping)
        else {}
    )
    custody = (
        review.get("custody_audit", {})
        if isinstance(review, Mapping)
        else {}
    )
    if (
        not isinstance(review, Mapping)
        or review.get("schema")
        != "lewm_go2_scene_diversity_recurrent_replication_"
        "cpu_backend_v1_qualification_terminal_review_v1"
        or review.get("status")
        != "PASS_FAIL_CLOSED_PRE_GENESIS_QUALIFICATION_TERMINAL_REVIEW"
        or review.get("audit_passed") is not True
        or decision.get("attempt_consumed") is not True
        or decision.get("qualification_pass") is not False
        or decision.get("qualification_status")
        != "FAIL_CPU_BACKEND_QUALIFICATION_HARD_STOP"
        or decision.get("failure_is_pre_genesis_source_control_flow") is not True
        or decision.get("failure_is_backend_evidence") is not False
        or decision.get("failure_is_scientific") is not False
        or decision.get("scientific_decision") is not None
        or decision.get("scientific_metrics") is not None
        or successor.get("cpu_v1_retry_or_science_identical_replacement_eligible")
        is not False
        or successor.get("only_eligible_next_backend_direction")
        != "Genesis 0.4.6 ROCm/HIP backend under a separate fresh qualification design"
        or permission.get("partial_predecessor_or_qualification_artifact_reuse_authorized")
        is not False
        or permission.get("qualification_probe_output_reuse_authorized") is not False
        or permission.get("successor_execution_authorized_by_this_review") is not False
        or custody.get("predecessor_partial_runtime_payload_opened") is not False
        or custody.get("qualification_scene_payload_opened") is not False
        or custody.get("rgb_frame_opened") is not False
        or custody.get("protected_material_opened") is not False
    ):
        raise SceneDiversityRunnerError(
            "CPU qualification terminal review contract changed"
        )
    evidence["predecessor_cpu_qualification_terminal_review"] = expected
    return evidence


class _PidBoundOsProxy:
    """Delegate ``os`` except for the reviewed orchestrator PID."""

    def __init__(self, delegate: object, *, pid: int) -> None:
        self._delegate = delegate
        self._pid = pid

    def getpid(self) -> int:
        return self._pid

    def __getattr__(self, name: str) -> object:
        return getattr(self._delegate, name)


@contextmanager
def _bound_qualification_orchestrator_pid(pid: int) -> Iterator[None]:
    if type(pid) is not int or pid <= 1:
        raise SceneDiversityRunnerError(
            "ROCm qualification orchestrator PID changed"
        )
    module = qualifier.predecessor
    original = module.os
    module.os = _PidBoundOsProxy(original, pid=pid)
    try:
        yield
    finally:
        module.os = original


def _load_qualification_reservation(
    *, authority_binding: Mapping[str, Any], plan_binding: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    terminal_path = plan_builder.QUALIFICATION_ATTEMPT_ROOT / "terminal.json"
    if terminal_path.exists() or terminal_path.is_symlink():
        raise SceneDiversityRunnerError(
            "ROCm qualification has a failure terminal"
        )
    reservation_path = plan_builder.QUALIFICATION_OUTPUT_ROOT / "reservation.json"
    try:
        reservation = json.loads(reservation_path.read_bytes())
        absolute = qualifier.pilot.file_binding(reservation_path)
        if not isinstance(reservation, Mapping):
            raise TypeError("reservation is not an object")
        _validated, checked = qualifier._read_qualification_reservation(  # noqa: SLF001
            collection_root=plan_builder.QUALIFICATION_OUTPUT_ROOT,
            expected_sha256=str(absolute["file_sha256"]),
            expected_byte_count=int(absolute["byte_count"]),
            authority_binding=authority_binding,
            plan_binding=plan_binding,
            orchestrator_nonce=str(reservation.get("orchestrator_nonce")),
            orchestrator_pid=int(reservation.get("orchestrator_pid")),
        )
        relative = qualifier.kernel._relative_output_binding(  # noqa: SLF001
            checked,
            output_root=plan_builder.QUALIFICATION_OUTPUT_ROOT,
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as exc:
        raise SceneDiversityRunnerError(
            "ROCm qualification reservation did not validate"
        ) from exc
    return dict(reservation), dict(relative)


def _revalidate_qualification_scene_result(
    *,
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    reservation_binding: Mapping[str, Any],
    scene: Mapping[str, Any],
    worker: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay inherited metadata validation without opening RGB payloads."""

    orchestrator_pid = worker.get("parent_pid")
    try:
        with (
            qualifier._configured_qualification_collector(),  # noqa: SLF001
            collector._configured_predecessor_collector_rocm(),  # noqa: SLF001
            _bound_qualification_orchestrator_pid(int(orchestrator_pid)),
        ):
            loaded, binding = qualifier.predecessor._load_scene_result_v2(  # noqa: SLF001
                collection_root=plan_builder.QUALIFICATION_OUTPUT_ROOT,
                scene=scene,
                authority=authority,
                authority_binding=authority_binding,
                plan=plan,
                plan_binding=plan_binding,
                reservation_binding=reservation_binding,
                worker_receipt=worker,
            )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise SceneDiversityRunnerError(
            "ROCm qualification scene evidence did not validate"
        ) from exc
    metric = loaded.get("scene_metric") if isinstance(loaded, Mapping) else None
    initialization = (
        loaded.get("genesis_initialization")
        if isinstance(loaded, Mapping)
        else None
    )
    runtime_versions = (
        loaded.get("runtime_versions") if isinstance(loaded, Mapping) else None
    )
    if (
        set(loaded) != QUALIFICATION_SCENE_RESULT_FIELDS
        or loaded.get("orchestrator_pid") != orchestrator_pid
        or loaded.get("sys_executable") != worker.get("sys_executable")
        or not isinstance(initialization, Mapping)
        or initialization.get("backend") != "amdgpu"
        or initialization.get("backend_api") != "gs.amdgpu"
        or initialization.get("hsa_override_gfx_version_present") is not False
        or loaded.get("process_reset_equivalence_audit")
        != collector.PROCESS_RESET_EQUIVALENCE_AUDIT_ROCM
        or not isinstance(runtime_versions, Mapping)
        or runtime_versions.get("genesis") != "0.4.6"
        or runtime_versions.get("torch") != "2.12.0+rocm7.2"
        or not isinstance(metric, Mapping)
        or any(
            metric.get(name) != qualifier.predecessor.STORED_FRAMES_PER_SCENE
            for name in (
                "native_render_calls",
                "rgb_render_calls",
                "auxiliary_depth_render_calls",
                "stored_rgb_frames",
            )
        )
        or type(loaded.get("stored_rgb_bytes")) is not int
        or not 0
        <= int(loaded["stored_rgb_bytes"])
        <= int(collector.EXPECTED_CAPS["stored_rgb_byte_ceiling"])
        or not qualifier._all_numbers_finite(loaded)  # noqa: SLF001
    ):
        raise SceneDiversityRunnerError(
            "ROCm qualification scene evidence did not validate"
        )
    return dict(loaded), dict(binding)


def validate_qualification_result_binding(
    value: object,
) -> tuple[dict[str, Any], dict[str, object]]:
    """Require the complete exact ROCm qualification PASS metadata."""

    if (
        not isinstance(value, Mapping)
        or set(value) != {"path", "sha256", "byte_count"}
        or value.get("path") != str(qualifier.QUALIFICATION_RESULT_PATH.resolve())
    ):
        raise SceneDiversityRunnerError(
            "ROCm qualification result binding malformed"
        )
    binding = file_binding_v1(qualifier.QUALIFICATION_RESULT_PATH)
    if binding != dict(value):
        raise SceneDiversityRunnerError("ROCm qualification result changed")
    try:
        result = json.loads(qualifier.QUALIFICATION_RESULT_PATH.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SceneDiversityRunnerError(
            "ROCm qualification result is not strict JSON"
        ) from exc
    if not isinstance(result, Mapping):
        raise SceneDiversityRunnerError(
            "ROCm qualification result is not an object"
        )

    qualification_authority_binding = file_binding_v1(
        qualifier.QUALIFICATION_AUTHORITY
    )
    qualification_plan_binding = file_binding_v1(
        plan_builder.QUALIFICATION_PLAN_OUTPUT
    )
    qualification_plan_pilot_binding = qualifier.pilot.file_binding(
        plan_builder.QUALIFICATION_PLAN_OUTPUT
    )
    if (
        _standard_binding(qualification_plan_pilot_binding)
        != qualification_plan_binding
    ):
        raise SceneDiversityRunnerError(
            "ROCm qualification plan binding shapes disagree"
        )
    try:
        qualification_authority = json.loads(
            qualifier.QUALIFICATION_AUTHORITY.read_bytes()
        )
        qualification_plan = json.loads(
            plan_builder.QUALIFICATION_PLAN_OUTPUT.read_bytes()
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SceneDiversityRunnerError(
            "ROCm qualification authority/plan is not strict JSON"
        ) from exc
    if not isinstance(qualification_authority, Mapping):
        raise SceneDiversityRunnerError(
            "ROCm qualification authority is not an object"
        )
    try:
        qualification_plan = plan_builder.validate_rocm_plan(
            qualification_plan,
            expected_attempt_id=plan_builder.QUALIFICATION_ATTEMPT_ID,
            expected_output_root=plan_builder.QUALIFICATION_OUTPUT_ROOT,
            plan_role="qualification",
        )
    except plan_builder.SceneDiversityGenesisRocmPlanError as exc:
        raise SceneDiversityRunnerError(str(exc)) from exc

    probes = result.get("probes")
    timing = result.get("timing_gate", {})
    kernel = result.get("kernel_reset_audit", {})
    preflight = result.get("rocm_egl_preflight", {})
    identity = preflight.get("identity", {}) if isinstance(preflight, Mapping) else {}
    environment = (
        preflight.get("environment", {})
        if isinstance(preflight, Mapping)
        else {}
    )
    expected_environment = plan_builder.rocm_execution_environment(
        "qualification"
    )
    expected_preflight = qualification_plan["execution_contract"][
        "graphics_preflight"
    ]
    bound_lld = str(
        Path(
            qualification_plan["runtime_bindings"]["rocm_lld_executable"][
                "path"
            ]
        ).resolve(strict=True)
    )
    hash_pattern = re.compile(r"[0-9a-f]{64}")
    expected_cpu_review = _standard_binding(
        plan_builder.CPU_TERMINAL_REVIEW_BINDING
    )
    if (
        set(result) != qualifier.QUALIFICATION_RESULT_FIELDS
        or result.get("schema") != qualifier.QUALIFICATION_RESULT_SCHEMA
        or result.get("status") != qualifier.QUALIFICATION_RESULT_STATUS
        or result.get("attempt_id") != plan_builder.QUALIFICATION_ATTEMPT_ID
        or result.get("backend") != "amdgpu"
        or result.get("backend_api") != "gs.amdgpu"
        or result.get("qualification_contract") != qualifier.QUALIFICATION_CONTRACT
        or result.get("authority_binding") != qualification_authority_binding
        or result.get("plan_binding") != qualification_plan_binding
        or set(qualification_authority) != qualifier.QUALIFICATION_AUTHORITY_FIELDS
        or qualification_authority.get("schema")
        != qualifier.QUALIFICATION_AUTHORITY_SCHEMA
        or qualification_authority.get("status")
        != qualifier.QUALIFICATION_AUTHORITY_STATUS
        or qualification_authority.get("attempt_id")
        != plan_builder.QUALIFICATION_ATTEMPT_ID
        or qualification_authority.get("plan_binding")
        != qualification_plan_binding
        or qualification_authority.get("qualification_contract")
        != qualifier.QUALIFICATION_CONTRACT
        or qualification_authority.get("predecessor_cpu_terminal_review_binding")
        != expected_cpu_review
        or not isinstance(preflight, Mapping)
        or set(preflight) != ROCM_EGL_PREFLIGHT_FIELDS
        or preflight.get("status") != "PASS_EXACT_ROCM_HIP_AND_EGL_R9700"
        or preflight.get("expectation") != expected_preflight
        or not isinstance(environment, Mapping)
        or dict(environment) != expected_environment
        or "HSA_OVERRIDE_GFX_VERSION" in environment
        or "LD_LIBRARY_PATH" in environment
        or preflight.get("path_ld_lld") != bound_lld
        or preflight.get("rocm_path_ld_lld") != bound_lld
        or any(
            not isinstance(preflight.get(field), str)
            or hash_pattern.fullmatch(str(preflight[field])) is None
            for field in (
                "lld_stdout_sha256",
                "rocminfo_stdout_sha256",
                "egl_stdout_sha256",
                "egl_stderr_sha256",
            )
        )
        or preflight.get("egl_exit_code")
        != expected_preflight["eglinfo_expected_exit_code"]
        or preflight.get("egl_device_index")
        != expected_preflight["egl_device_index"]
        or not isinstance(identity, Mapping)
        or set(identity) != ROCM_IDENTITY_FIELDS
        or identity.get("torch_version") != "2.12.0+rocm7.2"
        or not isinstance(identity.get("torch_hip_version"), str)
        or not str(identity["torch_hip_version"]).startswith("7.2")
        or identity.get("visible_device_count") != 1
        or identity.get("device_name") != "AMD Radeon AI PRO R9700"
        or not str(identity.get("arch_name", "")).startswith("gfx1201")
        or identity.get("genesis_version") != "0.4.6"
        or identity.get("genesis_backend_symbol") != "gs.amdgpu"
        or identity.get("hsa_override_present") is not False
        or identity.get("genesis_file")
        != str(
            Path(
                qualification_plan["runtime_bindings"][
                    "genesis_init_source"
                ]["path"]
            ).resolve(strict=True)
        )
        or identity.get("torch_file")
        != str(
            (
                plan_builder.WORLD_MODEL_ROCM_SITE_PACKAGES
                / "torch/__init__.py"
            ).resolve(strict=True)
        )
        or identity.get("numpy_file")
        != str(
            (
                plan_builder.WORLD_MODEL_ROCM_SITE_PACKAGES
                / "numpy/__init__.py"
            ).resolve(strict=True)
        )
        or identity.get("pillow_file")
        != str(
            (
                plan_builder.WORLD_MODEL_ROCM_SITE_PACKAGES
                / "PIL/__init__.py"
            ).resolve(strict=True)
        )
        or result.get("probe_order") != list(qualifier.QUALIFICATION_PROBE_ORDER)
        or not isinstance(probes, list)
        or len(probes) != 2
        or not isinstance(kernel, Mapping)
        or set(kernel) != QUALIFICATION_KERNEL_AUDIT_FIELDS
        or kernel.get("query_succeeded") is not True
        or kernel.get("new_amdgpu_ring_timeout_or_reset_count") != 0
        or not isinstance(kernel.get("matching_lines_sha256"), str)
        or hash_pattern.fullmatch(str(kernel["matching_lines_sha256"])) is None
        or not isinstance(timing, Mapping)
        or set(timing) != QUALIFICATION_TIMING_GATE_FIELDS
        or timing.get("passed") is not True
        or result.get("contact_force_route_audit")
        != collector.CONTACT_FORCE_ROUTE_AUDIT
        or result.get("all_existing_scene_gates_passed") is not True
        or result.get("exact_v03_renderer_compatibility_passed") is not True
        or result.get("scientific_attempt_root_absent") is not True
        or plan_builder.DEFAULT_ATTEMPT_ROOT.exists()
        or plan_builder.DEFAULT_ATTEMPT_ROOT.is_symlink()
        or result.get("probe_output_scientific_reuse_authorized") is not False
        or result.get("authorizes_scientific_authority_consideration") is not True
        or result.get("authorizes_retry_or_resume") is not False
        or not qualifier._all_numbers_finite(result)  # noqa: SLF001
    ):
        raise SceneDiversityRunnerError(
            "ROCm qualification did not pass exactly"
        )

    reservation, reservation_binding = _load_qualification_reservation(
        authority_binding=qualification_authority_binding,
        plan_binding=qualification_plan_binding,
    )
    scenes = qualifier.predecessor._scene_slices_v2(qualification_plan)  # noqa: SLF001
    worker_elapsed: list[float] = []
    parent_pids: set[int] = set()
    for probe, scene_index in zip(
        probes, qualifier.QUALIFICATION_PROBE_ORDER, strict=True
    ):
        scene = scenes[scene_index]
        worker = probe.get("worker") if isinstance(probe, Mapping) else None
        barrier = (
            probe.get("release_barrier")
            if isinstance(probe, Mapping)
            else None
        )
        expected_counts = qualifier.predecessor._scene_expected_counts_v2(  # noqa: SLF001
            str(scene["role"])
        )
        if (
            not isinstance(probe, Mapping)
            or set(probe) != QUALIFICATION_PROBE_FIELDS
            or probe.get("scene_index") != scene_index
            or probe.get("role") != scene["role"]
            or probe.get("scene_id") != scene["scene_id"]
            or probe.get("observed_counts") != expected_counts
            or probe.get("existing_scene_validation_passed") is not True
            or probe.get("probe_output_scientific_reuse_authorized") is not False
            or not isinstance(worker, Mapping)
            or set(worker) != QUALIFICATION_WORKER_FIELDS
            or worker.get("scene_index") != scene_index
            or worker.get("role") != scene["role"]
            or worker.get("scene_id") != scene["scene_id"]
            or type(worker.get("pid")) is not int
            or int(worker["pid"]) <= 1
            or type(worker.get("parent_pid")) is not int
            or int(worker["parent_pid"]) <= 1
            or worker.get("process_group_id") != worker.get("pid")
            or worker.get("fresh_process_group") is not True
            or worker.get("sys_executable")
            != str(
                Path(
                    qualification_plan["execution_contract"][
                        "python_invocation_path"
                    ]
                ).resolve(strict=True)
            )
            or type(worker.get("prelaunch_baseline_used_bytes")) is not int
            or int(worker["prelaunch_baseline_used_bytes"]) < 0
            or type(worker.get("peak_selected_device_vram_bytes")) is not int
            or int(worker["peak_selected_device_vram_bytes"])
            < int(worker["prelaunch_baseline_used_bytes"])
            or int(worker["peak_selected_device_vram_bytes"])
            > int(collector.EXPECTED_CAPS["selected_device_vram_byte_ceiling"])
            or worker.get("selected_device_vram_cap_breached") is not False
            or worker.get("watchdog_timeout") is not False
            or worker.get("exit_code") != 0
            or isinstance(worker.get("elapsed_seconds"), bool)
            or not isinstance(worker.get("elapsed_seconds"), (int, float))
            or not 0.0
            <= float(worker["elapsed_seconds"])
            <= qualifier.WORKER_TIMEOUT_SECONDS
            or not isinstance(barrier, Mapping)
            or set(barrier) != QUALIFICATION_RELEASE_BARRIER_FIELDS
            or barrier.get("baseline_used_bytes")
            != worker.get("prelaunch_baseline_used_bytes")
        ):
            raise SceneDiversityRunnerError(
                "ROCm qualification did not pass exactly"
            )
        try:
            qualified_barrier = qualifier.predecessor._barrier_with_identity_v2(  # noqa: SLF001
                barrier,
                scene=scene,
                worker_pid=int(worker["pid"]),
            )
            qualifier.predecessor._validate_release_barrier_shape_v2(  # noqa: SLF001
                qualified_barrier,
                scene=scene,
                worker_pid=int(worker["pid"]),
            )
            checked_scene_binding = qualifier.predecessor._rehash_relative_binding_v2(  # noqa: SLF001
                probe.get("scene_result_binding"),
                collection_root=plan_builder.QUALIFICATION_OUTPUT_ROOT,
                label=f"ROCm qualification scene {scene_index:03d} result",
            )
        except qualifier.predecessor.SceneProcessCollectionError as exc:
            raise SceneDiversityRunnerError(
                "ROCm qualification did not pass exactly"
            ) from exc
        if (
            checked_scene_binding != probe.get("scene_result_binding")
            or PurePosixPath(str(checked_scene_binding["path"]))
            != PurePosixPath("scene_results", f"{scene_index:03d}.json")
        ):
            raise SceneDiversityRunnerError(
                "ROCm qualification did not pass exactly"
            )
        loaded_scene, fully_checked_scene_binding = (
            _revalidate_qualification_scene_result(
                authority=qualification_authority,
                authority_binding=qualification_authority_binding,
                plan=qualification_plan,
                plan_binding=qualification_plan_pilot_binding,
                reservation_binding=reservation_binding,
                scene=scene,
                worker=worker,
            )
        )
        if (
            fully_checked_scene_binding != probe.get("scene_result_binding")
            or loaded_scene.get("observed_counts")
            != probe.get("observed_counts")
        ):
            raise SceneDiversityRunnerError(
                "ROCm qualification did not pass exactly"
            )
        parent_pids.add(int(worker["parent_pid"]))
        worker_elapsed.append(float(worker["elapsed_seconds"]))

    maximum = max(worker_elapsed)
    projected = (
        64.0 * maximum + qualifier.SCIENTIFIC_STARTUP_ALLOWANCE_SECONDS
    )
    if (
        len(parent_pids) != 1
        or parent_pids != {int(reservation["orchestrator_pid"])}
        or timing.get("cold_scene_12_worker_elapsed_seconds")
        != worker_elapsed[0]
        or timing.get("warm_scene_0_worker_elapsed_seconds")
        != worker_elapsed[1]
        or timing.get("maximum_worker_elapsed_seconds") != maximum
        or timing.get("projected_scientific_wall_seconds") != projected
        or timing.get("wall_ceiling_seconds")
        != collector.EXPECTED_CAPS["wall_seconds"]
        or projected > float(collector.EXPECTED_CAPS["wall_seconds"])
        or not math.isfinite(projected)
    ):
        raise SceneDiversityRunnerError(
            "ROCm qualification did not pass exactly"
        )
    return dict(result), binding


def _validate_plan_rocm(
    plan: Mapping[str, Any], authority: Mapping[str, Any]
) -> None:
    if authority.get("attempt_id") != plan_builder.DEFAULT_ATTEMPT_ID:
        raise SceneDiversityRunnerError(
            "ROCm scientific authority identity changed"
        )
    try:
        plan_builder.validate_rocm_plan(
            plan,
            expected_attempt_id=plan_builder.DEFAULT_ATTEMPT_ID,
            expected_output_root=plan_builder.DEFAULT_OUTPUT_ROOT,
            plan_role="scientific",
        )
    except plan_builder.SceneDiversityGenesisRocmPlanError as exc:
        raise SceneDiversityRunnerError(str(exc)) from exc


def _load_replacement_physics_index_rocm(
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, object],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    with (
        _configured_predecessor_runner_rocm(),
        collector._configured_predecessor_collector_rocm(),  # noqa: SLF001
    ):
        return _V3_LOAD_PHYSICS_INDEX(authority, authority_binding, plan)


def _configuration_overrides_rocm() -> dict[str, object]:
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
        "predecessor_failure_bindings_v3": predecessor_failure_bindings_rocm,
        "_load_replacement_physics_index_v3": (
            _load_replacement_physics_index_rocm
        ),
        "_validate_plan_v3": _validate_plan_rocm,
    }


@contextmanager
def _configured_predecessor_runner_rocm() -> Iterator[None]:
    """Apply and restore the exact scientific ROCm identity overlay."""

    with _CONFIGURATION_LOCK:
        overrides = _configuration_overrides_rocm()
        originals = {
            name: getattr(predecessor_runner, name) for name in overrides
        }
        lower_originals = {
            "v2": v2_runner._validate_plan_v2,  # noqa: SLF001
            "v1": v1_replacement_runner._validate_plan_v1,  # noqa: SLF001
        }
        original_validate = collector.pilot.validate_plan
        original_environment = collector.pilot.EXECUTION_ENVIRONMENT
        original_graphics = collector.pilot.GRAPHICS_PREFLIGHT_EXPECTATION
        try:
            for name, value in overrides.items():
                setattr(predecessor_runner, name, value)
            v2_runner._validate_plan_v2 = _validate_plan_rocm  # noqa: SLF001
            v1_replacement_runner._validate_plan_v1 = _validate_plan_rocm  # noqa: SLF001
            collector.pilot.validate_plan = collector._validate_rocm_plan_runtime  # noqa: SLF001
            collector.pilot.EXECUTION_ENVIRONMENT = (
                plan_builder.rocm_execution_environment("scientific")
            )
            collector.pilot.GRAPHICS_PREFLIGHT_EXPECTATION = dict(
                plan_builder.ROCM_GRAPHICS_PREFLIGHT_EXPECTATION
            )
            yield
        finally:
            collector.pilot.validate_plan = original_validate
            collector.pilot.EXECUTION_ENVIRONMENT = original_environment
            collector.pilot.GRAPHICS_PREFLIGHT_EXPECTATION = original_graphics
            v2_runner._validate_plan_v2 = lower_originals["v2"]  # noqa: SLF001
            v1_replacement_runner._validate_plan_v1 = lower_originals["v1"]  # noqa: SLF001
            for name, value in originals.items():
                setattr(predecessor_runner, name, value)


def _validate_authority_rocm(
    authority_path: Path,
    *,
    expected_sha256: str,
    expected_byte_count: int,
) -> tuple[dict[str, Any], dict[str, object], dict[str, Any]]:
    predecessor_failure_bindings_rocm()
    with _configured_predecessor_runner_rocm():
        validated = predecessor_runner._validate_authority_v3(  # noqa: SLF001
            authority_path,
            expected_sha256=expected_sha256,
            expected_byte_count=expected_byte_count,
        )
    validate_qualification_result_binding(
        validated[0].get("qualification_result_binding")
    )
    if (
        validated[0].get("predecessor_cpu_terminal_review_binding")
        != _standard_binding(plan_builder.CPU_TERMINAL_REVIEW_BINDING)
    ):
        raise SceneDiversityRunnerError(
            "scientific authority CPU terminal binding changed"
        )
    _validate_plan_rocm(validated[2], validated[0])
    return validated


def execute_rocm(
    authority: Mapping[str, Any],
    *,
    authority_binding: Mapping[str, object],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute unchanged V3 science after exact ROCm qualification."""

    _validate_plan_rocm(plan, authority)
    predecessor_failure_bindings_rocm()
    validate_qualification_result_binding(
        authority.get("qualification_result_binding")
    )
    if (
        authority.get("predecessor_cpu_terminal_review_binding")
        != _standard_binding(plan_builder.CPU_TERMINAL_REVIEW_BINDING)
    ):
        raise SceneDiversityRunnerError(
            "scientific authority CPU terminal binding changed"
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
        "Run the qualified Genesis 0.4.6 ROCm/HIP successor under its "
        "one-shot scientific authority."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    with _configured_predecessor_runner_rocm():
        return predecessor_runner.main(argv)


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
