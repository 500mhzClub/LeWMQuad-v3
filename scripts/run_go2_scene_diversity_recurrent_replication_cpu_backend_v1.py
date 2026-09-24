#!/usr/bin/env python3
"""Run the qualified scene-diversity successor with Genesis CPU physics."""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
from pathlib import Path
from pathlib import PurePosixPath
import re
import sys
import threading
from typing import Any, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_go2_scene_diversity_recurrent_replication_cpu_backend_v1_plan as plan_builder  # noqa: E402
from scripts import collect_go2_scene_diversity_recurrent_replication_cpu_backend_v1 as collector  # noqa: E402
from scripts import qualify_go2_scene_diversity_recurrent_replication_cpu_backend_v1 as qualifier  # noqa: E402
from scripts import run_go2_scene_diversity_recurrent_replication_integrity_replacement_v3 as predecessor_runner  # noqa: E402


v2_runner = predecessor_runner.predecessor_runner
v1_replacement_runner = v2_runner.predecessor_runner
frozen_runner = predecessor_runner.frozen_runner

AUTHORITY_SCHEMA = collector.AUTHORITY_SCHEMA
AUTHORITY_STATUS = collector.AUTHORITY_STATUS
SOURCE_REVIEW_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_cpu_backend_v1_"
    "source_review_v1"
)
SOURCE_REVIEW_STATUS = "PASS_INDEPENDENT_SOURCE_REVIEW"
RESULT_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_cpu_backend_v1_result_v1"
)
TERMINAL_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_cpu_backend_v1_terminal_v1"
)
RESERVATION_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_cpu_backend_v1_"
    "attempt_reservation_v1"
)
PREREGISTRATION = qualifier.PREREGISTRATION
SOURCE_REVIEW = qualifier.SOURCE_REVIEW
SCENE_PANEL = predecessor_runner.SCENE_PANEL
SCENE_PANEL_SHA256 = predecessor_runner.SCENE_PANEL_SHA256
SCENE_PANEL_BYTE_COUNT = predecessor_runner.SCENE_PANEL_BYTE_COUNT
DEFAULT_ATTEMPT_ROOT = plan_builder.DEFAULT_ATTEMPT_ROOT
DEFAULT_COLLECTION_ROOT = plan_builder.DEFAULT_OUTPUT_ROOT

PREDECESSOR_V3_TERMINAL = REPO_ROOT / (
    ".generated/dev/"
    "go2_scene_diversity_recurrent_replication_integrity_replacement_v3/"
    "attempt_v1/terminal.json"
)
PREDECESSOR_V3_TERMINAL_SHA256 = (
    "0d54d5c733a074098bd6d740d71a3358700e5e608ec019b4cdbbd47e1012ff4c"
)
PREDECESSOR_V3_TERMINAL_BYTE_COUNT = 442
PREDECESSOR_V3_TERMINAL_REVIEW = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "integrity_replacement_v3_terminal_review_2026-08-04.json"
)
PREDECESSOR_V3_TERMINAL_REVIEW_SHA256 = (
    "73360d1db0a65a29f2a825f32899337ce7ad53894f3c153dc18c7c973d9243a9"
)
PREDECESSOR_V3_TERMINAL_REVIEW_BYTE_COUNT = 21_073

DINO_REPOSITORY = predecessor_runner.DINO_REPOSITORY
DINO_CHECKPOINT = predecessor_runner.DINO_CHECKPOINT
DINO_REPOSITORY_COMMIT = predecessor_runner.DINO_REPOSITORY_COMMIT
DINO_CHECKPOINT_SHA256 = predecessor_runner.DINO_CHECKPOINT_SHA256
DINO_CHECKPOINT_BYTE_COUNT = predecessor_runner.DINO_CHECKPOINT_BYTE_COUNT
PROCESS_RESET_DEPENDENCY_PATHS = predecessor_runner.PROCESS_RESET_DEPENDENCY_PATHS
GENESIS_DEPENDENCY_ROOT = v2_runner.GENESIS_DEPENDENCY_ROOT
CPU_BACKEND_DEPENDENCY_PATHS = {
    "cpu_backend_dependency_genesis_constants": GENESIS_DEPENDENCY_ROOT
    / "genesis/constants.py",
    "cpu_backend_dependency_rigid_constraint_solver": GENESIS_DEPENDENCY_ROOT
    / "genesis/engine/solvers/rigid/constraint/solver.py",
    "cpu_backend_dependency_rigid_collider": GENESIS_DEPENDENCY_ROOT
    / "genesis/engine/solvers/rigid/collider/collider.py",
}

SOURCE_PATHS = {
    **predecessor_runner.SOURCE_PATHS,
    **CPU_BACKEND_DEPENDENCY_PATHS,
    "cpu_backend_plan_builder": REPO_ROOT
    / "scripts/build_go2_scene_diversity_recurrent_replication_cpu_backend_v1_plan.py",
    "cpu_backend_collector": REPO_ROOT
    / "scripts/collect_go2_scene_diversity_recurrent_replication_cpu_backend_v1.py",
    "cpu_backend_qualifier": REPO_ROOT
    / "scripts/qualify_go2_scene_diversity_recurrent_replication_cpu_backend_v1.py",
    "cpu_backend_runner": Path(__file__).resolve(),
    "cpu_backend_qualification_authority_builder": REPO_ROOT
    / "scripts/build_go2_scene_diversity_recurrent_replication_cpu_backend_v1_qualification_authority.py",
    "cpu_backend_scientific_authority_builder": REPO_ROOT
    / "scripts/build_go2_scene_diversity_recurrent_replication_cpu_backend_v1_authority.py",
    "cpu_backend_plan_test": REPO_ROOT
    / "lewm/tests/test_build_go2_scene_diversity_recurrent_replication_cpu_backend_v1_plan.py",
    "cpu_backend_collector_test": REPO_ROOT
    / "lewm/tests/test_collect_go2_scene_diversity_recurrent_replication_cpu_backend_v1.py",
    "cpu_backend_qualifier_test": REPO_ROOT
    / "lewm/tests/test_qualify_go2_scene_diversity_recurrent_replication_cpu_backend_v1.py",
    "cpu_backend_runner_test": REPO_ROOT
    / "lewm/tests/test_run_go2_scene_diversity_recurrent_replication_cpu_backend_v1.py",
    "cpu_backend_qualification_authority_test": REPO_ROOT
    / "lewm/tests/test_build_go2_scene_diversity_recurrent_replication_cpu_backend_v1_qualification_authority.py",
    "cpu_backend_scientific_authority_test": REPO_ROOT
    / "lewm/tests/test_build_go2_scene_diversity_recurrent_replication_cpu_backend_v1_authority.py",
    "predecessor_v3_failure_terminal": PREDECESSOR_V3_TERMINAL,
    "predecessor_v3_terminal_review": PREDECESSOR_V3_TERMINAL_REVIEW,
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


def _expected_binding(path: Path, sha256: str, byte_count: int) -> dict[str, object]:
    return {"path": str(path.resolve()), "sha256": sha256, "byte_count": byte_count}


def predecessor_failure_bindings_cpu() -> dict[str, dict[str, object]]:
    evidence = dict(_V3_FAILURE_BINDINGS())
    terminal_binding = _expected_binding(
        PREDECESSOR_V3_TERMINAL, PREDECESSOR_V3_TERMINAL_SHA256,
        PREDECESSOR_V3_TERMINAL_BYTE_COUNT,
    )
    review_binding = _expected_binding(
        PREDECESSOR_V3_TERMINAL_REVIEW, PREDECESSOR_V3_TERMINAL_REVIEW_SHA256,
        PREDECESSOR_V3_TERMINAL_REVIEW_BYTE_COUNT,
    )
    if file_binding_v1(PREDECESSOR_V3_TERMINAL) != terminal_binding:
        raise SceneDiversityRunnerError("V3 failure terminal changed")
    if file_binding_v1(PREDECESSOR_V3_TERMINAL_REVIEW) != review_binding:
        raise SceneDiversityRunnerError("V3 terminal review changed")
    terminal = json.loads(PREDECESSOR_V3_TERMINAL.read_bytes())
    review = json.loads(PREDECESSOR_V3_TERMINAL_REVIEW.read_bytes())
    verdict = review.get("verdict", {}) if isinstance(review, Mapping) else {}
    permission = review.get("permission_audit", {}) if isinstance(review, Mapping) else {}
    checks = review.get("checks", {}) if isinstance(review, Mapping) else {}
    custody = review.get("custody_audit", {}) if isinstance(review, Mapping) else {}
    if (
        terminal.get("schema")
        != "lewm_go2_scene_diversity_recurrent_replication_"
        "integrity_replacement_v3_terminal_v1"
        or terminal.get("status") != "FAIL_INFRASTRUCTURE_NO_SCIENTIFIC_DECISION"
        or terminal.get("result_binding") is not None
        or terminal.get("authorizes_retry_or_resume") is not False
        or review.get("schema")
        != "lewm_go2_scene_diversity_recurrent_replication_"
        "integrity_replacement_v3_terminal_review_v1"
        or review.get("status") != "PASS_FAIL_CLOSED_INFRASTRUCTURE_TERMINAL_REVIEW"
        or review.get("audit_passed") is not True
        or review.get("bindings", {}).get("terminal") != terminal_binding
        or verdict.get("attempt_consumed") is not True
        or verdict.get("scientific_decision") is not None
        or verdict.get("v4_vulkan_replacement_authorized") is not False
        or permission.get("partial_attempt_artifact_reuse_authorized") is not False
        or permission.get("v4_vulkan_successor_authorized") is not False
        or checks.get("combined_physics_result_is_absent") is not True
        or checks.get(
            "zero_dino_training_checkpoint_evaluation_result_metric_and_gate_stages_reached"
        )
        is not True
        or custody.get("partial_rgb_opened") is not False
        or custody.get("protected_material_opened") is not False
    ):
        raise SceneDiversityRunnerError("V3 terminal evidence contract changed")
    evidence.update(
        {
            "predecessor_v3_failure_terminal": terminal_binding,
            "predecessor_v3_terminal_review": review_binding,
        }
    )
    return evidence


class _PidBoundOsProxy:
    """Delegate ``os`` except for the reviewed historical orchestrator PID."""

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
        raise SceneDiversityRunnerError("qualification orchestrator PID changed")
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
            "CPU qualification has a failure terminal"
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
        relative = qualifier.predecessor.kernel._relative_output_binding(  # noqa: SLF001
            checked, output_root=plan_builder.QUALIFICATION_OUTPUT_ROOT
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError, RuntimeError) as exc:
        raise SceneDiversityRunnerError(
            "CPU qualification reservation did not validate"
        ) from exc
    return dict(reservation), dict(relative)


def _revalidate_qualification_scene_result(
    *, authority: Mapping[str, Any], authority_binding: Mapping[str, Any],
    plan: Mapping[str, Any], plan_binding: Mapping[str, Any],
    reservation_binding: Mapping[str, Any], scene: Mapping[str, Any],
    worker: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay the complete inherited metadata validator without opening RGB."""

    expected_fields = {
        "schema", "status", "attempt_id", "scene_index", "role", "scene_id",
        "worker_pid", "orchestrator_pid", "sys_executable", "fresh_process",
        "execution_seed", "genesis_initialization",
        "process_reset_equivalence_audit", "scene_local_mesh_cache",
        "plan_binding", "authority_binding", "collection_reservation_binding",
        "caps", "runtime_versions", "runtime_bindings", "source_bindings",
        "expected_counts", "observed_counts", "ordered_state_ids",
        "state_receipt_bindings", "render_receipt_binding", "scene_metric",
        "stored_rgb_bytes", "collection_wall_seconds", "failure",
        "authorizes_retry_or_resume", "allows_refill", "allows_overwrite",
        "allows_adaptive_batching",
    }
    orchestrator_pid = worker.get("parent_pid")
    try:
        with (
            qualifier._configured_qualification_collector(),  # noqa: SLF001
            collector._configured_predecessor_collector_cpu(),  # noqa: SLF001
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
            "CPU qualification scene evidence did not validate"
        ) from exc
    metric = loaded.get("scene_metric") if isinstance(loaded, Mapping) else None
    initialization = (
        loaded.get("genesis_initialization") if isinstance(loaded, Mapping) else None
    )
    if (
        set(loaded) != expected_fields
        or loaded.get("orchestrator_pid") != orchestrator_pid
        or loaded.get("sys_executable") != worker.get("sys_executable")
        or not isinstance(initialization, Mapping)
        or initialization.get("backend") != "cpu"
        or not isinstance(metric, Mapping)
        or any(
            metric.get(name) != qualifier.predecessor.STORED_FRAMES_PER_SCENE
            for name in (
                "native_render_calls", "rgb_render_calls",
                "auxiliary_depth_render_calls", "stored_rgb_frames",
            )
        )
        or type(loaded.get("stored_rgb_bytes")) is not int
        or not 0 <= int(loaded["stored_rgb_bytes"]) <= int(
            collector.EXPECTED_CAPS["stored_rgb_byte_ceiling"]
        )
        or not qualifier._all_numbers_finite(loaded)  # noqa: SLF001
    ):
        raise SceneDiversityRunnerError(
            "CPU qualification scene evidence did not validate"
        )
    return dict(loaded), dict(binding)


def validate_qualification_result_binding(
    value: object,
) -> tuple[dict[str, Any], dict[str, object]]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"path", "sha256", "byte_count"}
        or value.get("path") != str(qualifier.QUALIFICATION_RESULT_PATH.resolve())
    ):
        raise SceneDiversityRunnerError("CPU qualification result binding malformed")
    binding = file_binding_v1(qualifier.QUALIFICATION_RESULT_PATH)
    if binding != dict(value):
        raise SceneDiversityRunnerError("CPU qualification result changed")
    try:
        result = json.loads(qualifier.QUALIFICATION_RESULT_PATH.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SceneDiversityRunnerError(
            "CPU qualification result is not strict JSON"
        ) from exc
    if not isinstance(result, Mapping):
        raise SceneDiversityRunnerError("CPU qualification result is not an object")
    probes = result.get("probes") if isinstance(result, Mapping) else None
    timing = result.get("timing_gate", {}) if isinstance(result, Mapping) else {}
    kernel = result.get("kernel_reset_audit", {}) if isinstance(result, Mapping) else {}
    graphics = result.get("graphics_preflight", {}) if isinstance(result, Mapping) else {}
    qualification_authority_binding = file_binding_v1(
        qualifier.QUALIFICATION_AUTHORITY
    )
    qualification_plan_binding = file_binding_v1(
        plan_builder.QUALIFICATION_PLAN_OUTPUT
    )
    qualification_plan = json.loads(
        plan_builder.QUALIFICATION_PLAN_OUTPUT.read_bytes()
    )
    try:
        qualification_authority = json.loads(
            qualifier.QUALIFICATION_AUTHORITY.read_bytes()
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SceneDiversityRunnerError(
            "CPU qualification authority is not strict JSON"
        ) from exc
    if not isinstance(qualification_authority, Mapping):
        raise SceneDiversityRunnerError(
            "CPU qualification authority is not an object"
        )
    plan_builder.validate_cpu_plan(
        qualification_plan,
        expected_attempt_id=plan_builder.QUALIFICATION_ATTEMPT_ID,
        expected_output_root=plan_builder.QUALIFICATION_OUTPUT_ROOT,
    )
    hash_pattern = re.compile(r"[0-9a-f]{64}")
    if (
        set(result) != qualifier.QUALIFICATION_RESULT_FIELDS
        or result.get("schema") != qualifier.QUALIFICATION_RESULT_SCHEMA
        or result.get("status") != qualifier.QUALIFICATION_RESULT_STATUS
        or result.get("attempt_id") != plan_builder.QUALIFICATION_ATTEMPT_ID
        or result.get("backend") != "cpu"
        or result.get("qualification_contract") != qualifier.QUALIFICATION_CONTRACT
        or result.get("authority_binding") != qualification_authority_binding
        or result.get("plan_binding") != qualification_plan_binding
        or not isinstance(graphics, Mapping)
        or set(graphics) != qualifier.QUALIFICATION_GRAPHICS_PREFLIGHT_FIELDS
        or graphics.get("phase") != "graphics_preflight"
        or graphics.get("status") != "PASS"
        or graphics.get("environment") != plan_builder.CPU_EXECUTION_ENVIRONMENT
        or graphics.get("expectation")
        != qualification_plan["execution_contract"]["graphics_preflight"]
        or any(
            not isinstance(graphics.get(field), str)
            or hash_pattern.fullmatch(str(graphics[field])) is None
            for field in (
                "vulkan_stdout_sha256", "egl_stdout_sha256", "egl_stderr_sha256"
            )
        )
        or graphics.get("egl_exit_code")
        != qualification_plan["execution_contract"]["graphics_preflight"][
            "eglinfo_expected_exit_code"
        ]
        or result.get("probe_order") != list(qualifier.QUALIFICATION_PROBE_ORDER)
        or not isinstance(probes, list)
        or len(probes) != 2
        or not isinstance(kernel, Mapping)
        or set(kernel) != qualifier.QUALIFICATION_KERNEL_AUDIT_FIELDS
        or kernel.get("query_succeeded") is not True
        or kernel.get("new_amdgpu_ring_timeout_or_reset_count") != 0
        or not isinstance(kernel.get("matching_lines_sha256"), str)
        or hash_pattern.fullmatch(str(kernel["matching_lines_sha256"])) is None
        or not isinstance(timing, Mapping)
        or set(timing) != qualifier.QUALIFICATION_TIMING_GATE_FIELDS
        or timing.get("passed") is not True
        or result.get("all_existing_scene_gates_passed") is not True
        or result.get("scientific_attempt_root_absent") is not True
        or result.get("probe_output_scientific_reuse_authorized") is not False
        or result.get("authorizes_cpu_scientific_authority_consideration") is not True
        or result.get("authorizes_retry_or_resume") is not False
        or not qualifier._all_numbers_finite(result)  # noqa: SLF001
    ):
        raise SceneDiversityRunnerError("CPU qualification did not pass exactly")

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
        barrier = probe.get("release_barrier") if isinstance(probe, Mapping) else None
        expected_counts = qualifier.predecessor._scene_expected_counts_v2(  # noqa: SLF001
            str(scene["role"])
        )
        if (
            not isinstance(probe, Mapping)
            or set(probe) != qualifier.QUALIFICATION_PROBE_FIELDS
            or probe.get("scene_index") != scene_index
            or probe.get("role") != scene["role"]
            or probe.get("scene_id") != scene["scene_id"]
            or probe.get("observed_counts") != expected_counts
            or probe.get("existing_scene_validation_passed") is not True
            or probe.get("probe_output_scientific_reuse_authorized") is not False
            or not isinstance(worker, Mapping)
            or set(worker) != qualifier.QUALIFICATION_WORKER_FIELDS
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
            or not 0.0 <= float(worker["elapsed_seconds"]) <= qualifier.WORKER_TIMEOUT_SECONDS
            or not isinstance(barrier, Mapping)
            or set(barrier) != qualifier.QUALIFICATION_RELEASE_BARRIER_FIELDS
            or barrier.get("baseline_used_bytes")
            != worker.get("prelaunch_baseline_used_bytes")
        ):
            raise SceneDiversityRunnerError("CPU qualification did not pass exactly")
        try:
            qualified_barrier = qualifier.predecessor._barrier_with_identity_v2(  # noqa: SLF001
                barrier, scene=scene, worker_pid=int(worker["pid"])
            )
            qualifier.predecessor._validate_release_barrier_shape_v2(  # noqa: SLF001
                qualified_barrier, scene=scene, worker_pid=int(worker["pid"])
            )
            checked_scene_binding = qualifier.predecessor._rehash_relative_binding_v2(  # noqa: SLF001
                probe.get("scene_result_binding"),
                collection_root=plan_builder.QUALIFICATION_OUTPUT_ROOT,
                label=f"qualification scene {scene_index:03d} result",
            )
        except qualifier.predecessor.SceneProcessCollectionError as exc:
            raise SceneDiversityRunnerError(
                "CPU qualification did not pass exactly"
            ) from exc
        if (
            checked_scene_binding != probe.get("scene_result_binding")
            or PurePosixPath(str(checked_scene_binding["path"]))
            != PurePosixPath("scene_results", f"{scene_index:03d}.json")
        ):
            raise SceneDiversityRunnerError("CPU qualification did not pass exactly")
        loaded_scene, fully_checked_scene_binding = (
            _revalidate_qualification_scene_result(
                authority=qualification_authority,
                authority_binding=qualification_authority_binding,
                plan=qualification_plan,
                plan_binding=qualification_plan_binding,
                reservation_binding=reservation_binding,
                scene=scene,
                worker=worker,
            )
        )
        if (
            fully_checked_scene_binding != probe.get("scene_result_binding")
            or loaded_scene.get("observed_counts") != probe.get("observed_counts")
        ):
            raise SceneDiversityRunnerError("CPU qualification did not pass exactly")
        parent_pids.add(int(worker["parent_pid"]))
        worker_elapsed.append(float(worker["elapsed_seconds"]))

    maximum = max(worker_elapsed)
    projected = (
        64.0 * maximum + qualifier.SCIENTIFIC_STARTUP_ALLOWANCE_SECONDS
    )
    if (
        len(parent_pids) != 1
        or parent_pids != {int(reservation["orchestrator_pid"])}
        or timing.get("maximum_worker_elapsed_seconds") != maximum
        or timing.get("projected_scientific_wall_seconds") != projected
        or timing.get("wall_ceiling_seconds") != collector.EXPECTED_CAPS["wall_seconds"]
        or projected > float(collector.EXPECTED_CAPS["wall_seconds"])
        or not math.isfinite(projected)
    ):
        raise SceneDiversityRunnerError("CPU qualification did not pass exactly")
    return dict(result), binding


def _validate_plan_cpu(plan: Mapping[str, Any], authority: Mapping[str, Any]) -> None:
    if authority.get("attempt_id") != plan_builder.DEFAULT_ATTEMPT_ID:
        raise SceneDiversityRunnerError("CPU scientific authority identity changed")
    try:
        plan_builder.validate_cpu_plan(
            plan,
            expected_attempt_id=plan_builder.DEFAULT_ATTEMPT_ID,
            expected_output_root=plan_builder.DEFAULT_OUTPUT_ROOT,
        )
    except plan_builder.SceneDiversityCpuBackendPlanError as exc:
        raise SceneDiversityRunnerError(str(exc)) from exc


def _load_replacement_physics_index_cpu(
    authority: Mapping[str, Any], authority_binding: Mapping[str, object],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    with _configured_predecessor_runner_cpu(), collector._configured_predecessor_collector_cpu():  # noqa: SLF001
        return _V3_LOAD_PHYSICS_INDEX(authority, authority_binding, plan)


def _configuration_overrides_cpu() -> dict[str, object]:
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
        "predecessor_failure_bindings_v3": predecessor_failure_bindings_cpu,
        "_load_replacement_physics_index_v3": _load_replacement_physics_index_cpu,
        "_validate_plan_v3": _validate_plan_cpu,
    }


@contextmanager
def _configured_predecessor_runner_cpu() -> Iterator[None]:
    with _CONFIGURATION_LOCK:
        overrides = _configuration_overrides_cpu()
        originals = {name: getattr(predecessor_runner, name) for name in overrides}
        lower_originals = {
            "v2": v2_runner._validate_plan_v2,  # noqa: SLF001
            "v1": v1_replacement_runner._validate_plan_v1,  # noqa: SLF001
        }
        original_validate = collector.pilot.validate_plan
        original_environment = collector.pilot.EXECUTION_ENVIRONMENT
        try:
            for name, value in overrides.items():
                setattr(predecessor_runner, name, value)
            v2_runner._validate_plan_v2 = _validate_plan_cpu  # noqa: SLF001
            v1_replacement_runner._validate_plan_v1 = _validate_plan_cpu  # noqa: SLF001
            collector.pilot.validate_plan = collector._validate_cpu_plan_runtime  # noqa: SLF001
            collector.pilot.EXECUTION_ENVIRONMENT = dict(
                plan_builder.CPU_EXECUTION_ENVIRONMENT
            )
            yield
        finally:
            collector.pilot.validate_plan = original_validate
            collector.pilot.EXECUTION_ENVIRONMENT = original_environment
            v2_runner._validate_plan_v2 = lower_originals["v2"]  # noqa: SLF001
            v1_replacement_runner._validate_plan_v1 = lower_originals["v1"]  # noqa: SLF001
            for name, value in originals.items():
                setattr(predecessor_runner, name, value)


def _validate_authority_cpu(
    authority_path: Path, *, expected_sha256: str, expected_byte_count: int
) -> tuple[dict[str, Any], dict[str, object], dict[str, Any]]:
    predecessor_failure_bindings_cpu()
    with _configured_predecessor_runner_cpu():
        validated = predecessor_runner._validate_authority_v3(  # noqa: SLF001
            authority_path,
            expected_sha256=expected_sha256,
            expected_byte_count=expected_byte_count,
        )
    validate_qualification_result_binding(
        validated[0].get("qualification_result_binding")
    )
    _validate_plan_cpu(validated[2], validated[0])
    return validated


def execute_cpu(
    authority: Mapping[str, Any], *, authority_binding: Mapping[str, object],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_plan_cpu(plan, authority)
    predecessor_failure_bindings_cpu()
    validate_qualification_result_binding(authority.get("qualification_result_binding"))
    with _configured_predecessor_runner_cpu():
        return predecessor_runner.execute_v3(
            authority, authority_binding=authority_binding, plan=plan
        )


execute_v3 = execute_cpu
execute_v2 = execute_cpu
execute_v1 = execute_cpu
_validate_plan_v3 = _validate_plan_cpu
_validate_authority_v3 = _validate_authority_cpu


def build_parser():
    parser = predecessor_runner.build_parser()
    parser.description = (
        "Run the qualified CPU-backend successor under its one-shot "
        "scientific authority."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    with _configured_predecessor_runner_cpu():
        return predecessor_runner.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORITY_SCHEMA", "AUTHORITY_STATUS", "ContextOnlyLedgerV1",
    "CPU_BACKEND_DEPENDENCY_PATHS",
    "DEFAULT_ATTEMPT_ROOT", "DEFAULT_COLLECTION_ROOT", "RESULT_SCHEMA",
    "SOURCE_PATHS", "SceneDiversityRunnerError", "TERMINAL_SCHEMA",
    "execute_cpu", "predecessor_failure_bindings_cpu",
    "validate_qualification_result_binding",
]
