#!/usr/bin/env python3
"""Run the one-shot qualified CPU-flat V3 scientific replication.

The coordinator collects all 64 frozen scenes in plan order using one fresh
CPU-physics process group per scene and the qualified Vulkan/EGL R9700 render
route.  It then invokes the already-frozen recurrent fit and evaluation
helpers directly.  Qualification runtime payloads are neither opened nor
reused; the minimal PASS decision and independent terminal review enter only
through the scientific plan builder's validated identity bindings.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import signal
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for _package_root in (
    REPO_ROOT,
    REPO_ROOT / "lewm_genesis",
    REPO_ROOT / "lewm_worlds",
):
    if str(_package_root) not in sys.path:
        sys.path.insert(0, str(_package_root))

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as pilot  # noqa: E402
from lewm.benchmarks import go2_scene_diversity_recurrent_replication_v1 as benchmark  # noqa: E402
from scripts import build_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_development_v3_scientific_plan as plan_builder  # noqa: E402
from scripts import collect_go2_scene_diversity_recurrent_replication_cpu_backend_v1 as cpu_core  # noqa: E402
from scripts import collect_go2_scene_diversity_recurrent_replication_integrity_replacement_v2 as scene_core  # noqa: E402
from scripts import collect_go2_world_model_bounded_branch_experiment_authorized_v1 as bounded  # noqa: E402
from scripts import collect_go2_world_model_counterfactual_pilot_v1 as kernel  # noqa: E402
from scripts import run_go2_scene_diversity_recurrent_replication_genesis_rocm_flat_development_v1 as flat_shared  # noqa: E402
from scripts import run_go2_scene_diversity_recurrent_replication_v1 as frozen_runner  # noqa: E402
from scripts import run_go2_grounded_dense_dino_joint_jepa_v1 as upstream  # noqa: E402
from scripts import run_go2_world_model_counterfactual_calibration_authorized_v1 as calibration  # noqa: E402


PREREGISTRATION = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_"
    "flat_development_v3_scientific_preregistration_2026-08-05.md"
)
PREREGISTRATION_SHA256 = (
    "ce99a5e1034288823f5e4cb92b65703ca5ca53d529010c0872571d2967613f80"
)
PREREGISTRATION_BYTE_COUNT = 10_671

RESULT_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_"
    "development_v3_scientific_result_v1"
)
TERMINAL_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_"
    "development_v3_scientific_terminal_v1"
)
RESERVATION_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_"
    "development_v3_scientific_reservation_v1"
)
SCENE_RESULT_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_"
    "development_v3_scientific_scene_result_v1"
)
WORKER_FAILURE_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_"
    "development_v3_scientific_worker_failure_v1"
)
SOURCE_REVIEW_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_"
    "development_v3_scientific_source_review_v1"
)
SOURCE_REVIEW_STATUS = "PASS_INDEPENDENT_SCIENTIFIC_SOURCE_REVIEW"
SOURCE_REVIEW = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_"
    "development_v3_scientific_source_review_2026-08-05.json"
)
SCIENTIFIC_PLAN_SHA256 = (
    "0ad79cc46cead469d6532cd0be04c5d7623fffe18ddafc737c32855d6c9a8f29"
)
SCIENTIFIC_PLAN_BYTE_COUNT = 359_692
SOURCE_PATHS = {
    "scientific_plan_builder": Path(plan_builder.__file__).resolve(),
    "scientific_runner": Path(__file__).resolve(),
    "scientific_runner_test": REPO_ROOT / (
        "lewm/tests/test_run_go2_scene_diversity_recurrent_replication_"
        "genesis_cpu_flat_development_v3_scientific.py"
    ),
    "qualified_worker_harness": REPO_ROOT / (
        "scripts/run_go2_scene_diversity_recurrent_replication_genesis_cpu_"
        "flat_development_v3.py"
    ),
    "frozen_replication_runner": Path(frozen_runner.__file__).resolve(),
    "frozen_replication_benchmark": Path(benchmark.__file__).resolve(),
    "flat_worker_helpers": Path(flat_shared.__file__).resolve(),
    "cpu_initialization": Path(cpu_core.__file__).resolve(),
    "scene_closure_helpers": Path(scene_core.__file__).resolve(),
    "collection_kernel": Path(kernel.__file__).resolve(),
    "bounded_render_checks": Path(bounded.__file__).resolve(),
    "runtime_supervision_helpers": Path(calibration.__file__).resolve(),
}

FAIL_STATUS = "FAIL_INFRASTRUCTURE_NO_SCIENTIFIC_DECISION"
SCENE_PASS_STATUS = "SCENE_PHYSICS_COMPLETE"
SCENE_FAIL_STATUS = "SCENE_PHYSICS_FAILED_NONREUSABLE"
SCENE_ORDER = tuple(range(scene_core.SCENE_COUNT))
WORKER_TIMEOUT_SECONDS = 300.0
WALL_CEILING_SECONDS = float(scene_core.EXPECTED_CAPS["wall_seconds"])
VRAM_CEILING_BYTES = int(
    scene_core.EXPECTED_CAPS["selected_device_vram_byte_ceiling"]
)
POLL_SECONDS = 0.02
GROUP_EXIT_TIMEOUT_SECONDS = 5.0
EXPECTED_ENVIRONMENT_KEYS = frozenset(plan_builder.CPU_EXECUTION_ENVIRONMENT)
EXPECTED_GRAPHICS_PREFLIGHT = {
    "egl_device_index": 1,
    "eglinfo_expected_exit_code": 2,
    "egl_renderer_name_contains": "AMD Radeon AI PRO R9700",
    "vulkan_device_index": 0,
    "vulkan_vendor_id": "0x1002",
    "vulkan_device_id": "0x7551",
    "vulkan_device_name": "AMD Radeon AI PRO R9700",
}

SCIENTIFIC_CONTRACT = {
    "backend": "cpu",
    "backend_api": "gs.cpu",
    "render_backend": "vulkan_egl_r9700",
    "genesis_version": "0.3.14",
    "branch_mechanism": "parallel_lockstep_envs_no_restore",
    "exact_within_state_lane_equality_required": True,
    "scene_indices_in_order": list(SCENE_ORDER),
    "fresh_worker_processes": 64,
    "states_per_worker": 4,
    "candidate_actions_per_state": 9,
    "branches_per_worker": 36,
    "context_frames_per_worker": 12,
    "target_frames_per_worker": 36,
    "stored_rgb_frames_per_worker": 48,
    "auxiliary_depth_validation_renders_per_worker": 48,
    "worker_process_group_watchdog_seconds": WORKER_TIMEOUT_SECONDS,
    "selected_device_vram_byte_ceiling": VRAM_CEILING_BYTES,
    "vram_sample_interval_seconds": POLL_SECONDS,
    "release_after_every_worker_including_final": True,
    "leaked_process_group_forbidden": True,
    "new_amdgpu_reset_wedge_or_hsa_exception_forbidden": True,
    "pilot_diagnostic_error_diagnostics_must_be_persisted": True,
    "whole_run_wall_ceiling_seconds": WALL_CEILING_SECONDS,
    "scientific_protocol": copy.deepcopy(plan_builder.SCIENTIFIC_PROTOCOL),
    "qualification_payload_reuse_authorized": False,
    "retry_resume_overwrite_authorized": False,
}


class CpuFlatScientificError(RuntimeError):
    """Fail-closed scientific-run error with optional JSON diagnostics."""

    def __init__(
        self, message: str, *, diagnostics: Mapping[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        if diagnostics is not None:
            self.diagnostics = dict(diagnostics)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _standard_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(value["path"]),
        "sha256": str(value.get("sha256", value.get("file_sha256"))),
        "byte_count": int(value["byte_count"]),
    }


def _relative_binding(
    value: Mapping[str, Any], *, output_root: Path
) -> dict[str, Any]:
    return kernel._relative_output_binding(value, output_root=output_root)  # noqa: SLF001


def _failure_receipt(error: Exception) -> dict[str, Any]:
    """Carry complete canonical PilotDiagnosticError diagnostics forward."""

    return kernel._failure_receipt(error)  # noqa: SLF001


def _preregistration_binding() -> dict[str, Any]:
    expected = {
        "path": str(PREREGISTRATION.resolve(strict=True)),
        "sha256": PREREGISTRATION_SHA256,
        "byte_count": PREREGISTRATION_BYTE_COUNT,
    }
    observed = _standard_binding(pilot.file_binding(PREREGISTRATION))
    if not PREREGISTRATION_SHA256 or PREREGISTRATION_BYTE_COUNT <= 0:
        raise CpuFlatScientificError("CPU-flat preregistration is not frozen")
    if observed != expected:
        raise CpuFlatScientificError("CPU-flat preregistration changed")
    return expected


def _source_bindings() -> dict[str, dict[str, Any]]:
    return {
        name: _standard_binding(pilot.file_binding(path))
        for name, path in SOURCE_PATHS.items()
    }


def _validate_dino_inputs() -> dict[str, Any]:
    expected = frozen_runner.expected_dino_v1()
    frozen_runner._require_binding(  # noqa: SLF001
        expected["checkpoint_binding"], label="frozen DINO checkpoint"
    )
    frozen_runner._validate_dino_source_v1()  # noqa: SLF001
    return expected


def _read_and_validate_source_review(
    path: Path,
    *,
    expected_sha256: str,
    expected_byte_count: int,
    plan_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the one-shot independent clearance before root reservation."""

    if path.resolve(strict=True) != SOURCE_REVIEW.resolve(strict=True):
        raise CpuFlatScientificError("scientific source-review path changed")
    review, binding = pilot.read_bound_json(
        path,
        expected_sha256=expected_sha256,
        expected_byte_count=expected_byte_count,
        label="CPU-flat V3 scientific source review",
    )
    standard_binding = _standard_binding(binding)
    decisions = review.get("decision")
    expected_decisions = {
        "source_review_passed": True,
        "exact_scientific_payload_identity_verified": True,
        "exact_eleven_key_environment_verified": True,
        "exact_64_fresh_scene_process_contract_verified": True,
        "frozen_three_arm_protocol_and_gates_verified": True,
        "minimal_qualification_decision_and_terminal_review_only": True,
        "qualification_payload_reuse_authorized": False,
        "fresh_scientific_root_verified": True,
        "exactly_one_scientific_invocation_eligible_under_user_authorization": True,
        "scientific_execution_authority_created_by_review": False,
        "retry_resume_refill_overwrite_repair_or_second_invocation_authorized": False,
    }
    sources = _source_bindings()
    if (
        not isinstance(review, dict)
        or review.get("schema") != SOURCE_REVIEW_SCHEMA
        or review.get("status") != SOURCE_REVIEW_STATUS
        or review.get("protected_material_opened") is not False
        or review.get("findings") != []
        or review.get("preregistration_binding") != _preregistration_binding()
        or review.get("scientific_plan_binding") != dict(plan_binding)
        or review.get("scientific_plan_builder_binding")
        != sources["scientific_plan_builder"]
        or review.get("scientific_runner_binding")
        != sources["scientific_runner"]
        or review.get("focused_test_binding")
        != sources["scientific_runner_test"]
        or review.get("source_bindings") != sources
        or review.get("qualification_pass_decision_binding")
        != plan_builder.qualification_pass_decision_binding()
        or review.get("independent_qualification_terminal_review_binding")
        != plan_builder.independent_qualification_terminal_review_binding()
        or review.get("dino") != _validate_dino_inputs()
        or decisions != expected_decisions
    ):
        raise CpuFlatScientificError("independent scientific source review changed")
    return dict(review), standard_binding


def _shared_state_snapshot() -> tuple[object, object, bytes, bytes]:
    return (
        pilot.EXECUTION_ENVIRONMENT,
        pilot.GRAPHICS_PREFLIGHT_EXPECTATION,
        _canonical(pilot.EXECUTION_ENVIRONMENT),
        _canonical(pilot.GRAPHICS_PREFLIGHT_EXPECTATION),
    )


def _require_shared_state_unchanged(
    snapshot: tuple[object, object, bytes, bytes]
) -> None:
    environment, graphics, environment_bytes, graphics_bytes = snapshot
    if (
        pilot.EXECUTION_ENVIRONMENT is not environment
        or pilot.GRAPHICS_PREFLIGHT_EXPECTATION is not graphics
        or _canonical(pilot.EXECUTION_ENVIRONMENT) != environment_bytes
        or _canonical(pilot.GRAPHICS_PREFLIGHT_EXPECTATION) != graphics_bytes
    ):
        raise CpuFlatScientificError("historical pilot shared state changed")


def _read_and_validate_plan(
    path: Path, *, expected_sha256: str, expected_byte_count: int
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Finish every static/input/parity check before root reservation."""

    snapshot = _shared_state_snapshot()
    raw, binding = pilot.read_bound_json(
        path,
        expected_sha256=expected_sha256,
        expected_byte_count=expected_byte_count,
        label="CPU-flat V3 scientific exact plan",
    )
    plan = plan_builder.validate_scientific_plan(raw)
    pilot.require_plan_bindings(plan)
    bounded._validate_plan_parity_prerequisites_v1(plan)  # noqa: SLF001
    bounded._validate_bound_scenes(plan)  # noqa: SLF001
    scenes = scene_core._scene_slices_v2(plan)  # noqa: SLF001
    invocation = calibration._validate_python_invocation(plan)  # noqa: SLF001
    preregistration = _preregistration_binding()
    decision = plan_builder.qualification_pass_decision_binding()
    terminal_review = (
        plan_builder.independent_qualification_terminal_review_binding()
    )
    dino = _validate_dino_inputs()
    _require_shared_state_unchanged(snapshot)
    contract = plan.get("successor_contract", {})
    if (
        set(plan_builder.CPU_EXECUTION_ENVIRONMENT)
        != EXPECTED_ENVIRONMENT_KEYS
        or len(plan_builder.CPU_EXECUTION_ENVIRONMENT) != 11
        or plan_builder.CPU_EXECUTION_ENVIRONMENT.get("HOME")
        != "/home/andrewknowles"
        or plan_builder.CPU_EXECUTION_ENVIRONMENT.get("PATH")
        != "/usr/bin:/bin"
        or plan["execution_contract"]["environment"]
        != plan_builder.CPU_EXECUTION_ENVIRONMENT
        or Path(sys.executable).resolve(strict=True)
        != invocation.resolve(strict=True)
        or plan["execution_contract"].get("backend") != "cpu"
        or plan["execution_contract"].get("graphics_preflight")
        != EXPECTED_GRAPHICS_PREFLIGHT
        or plan.get("branch_mechanism")
        != "parallel_lockstep_envs_no_restore"
        or contract.get("genesis_backend_symbol") != "gs.cpu"
        or contract.get("flat_harness_owned") is not True
        or contract.get("reviewed_cpu_delta_unchanged") is not True
        or contract.get("plan_role") != "scientific"
        or contract.get("scientific_attempt_id")
        != plan_builder.SCIENTIFIC_ATTEMPT_ID
        or contract.get("scientific_attempt_root")
        != str(plan_builder.SCIENTIFIC_ATTEMPT_ROOT.resolve(strict=False))
        or contract.get("qualification_pass_decision_binding") != decision
        or contract.get("independent_qualification_terminal_review_binding")
        != terminal_review
        or contract.get("qualification_pass_decision_validated") is not True
        or contract.get("qualification_payload_reuse_authorized") is not False
        or contract.get(
            "qualification_result_terminal_scene_receipt_rgb_cache_or_diagnostic_open_authorized"
        )
        is not False
        or contract.get("qualification_runtime_payload_opened_by_builder")
        is not False
        or contract.get("exact_64_scene_payload_unchanged") is not True
        or contract.get("one_fresh_process_per_scene") is not True
        or contract.get("scientific_protocol")
        != plan_builder.SCIENTIFIC_PROTOCOL
        or contract.get("scientific_source_review_required") is not True
        or contract.get("scientific_execution_authorized") is not False
        or tuple(SCENE_ORDER) != tuple(range(64))
        or len(scenes) != 64
        or plan.get("expected_counts") != scene_core.EXPECTED_COUNTS
        or plan.get("output_root")
        != str(plan_builder.SCIENTIFIC_OUTPUT_ROOT.resolve(strict=False))
        or str(path.resolve(strict=True))
        != str(plan_builder.SCIENTIFIC_PLAN_OUTPUT.resolve(strict=True))
        or _standard_binding(binding)
        != {
            "path": str(plan_builder.SCIENTIFIC_PLAN_OUTPUT.resolve(strict=True)),
            "sha256": SCIENTIFIC_PLAN_SHA256,
            "byte_count": SCIENTIFIC_PLAN_BYTE_COUNT,
        }
        or preregistration != _preregistration_binding()
        or dino != frozen_runner.expected_dino_v1()
        or plan_builder.SCIENTIFIC_ATTEMPT_ROOT.exists()
        or plan_builder.SCIENTIFIC_ATTEMPT_ROOT.is_symlink()
        or plan_builder.SCIENTIFIC_OUTPUT_ROOT.exists()
        or plan_builder.SCIENTIFIC_OUTPUT_ROOT.is_symlink()
    ):
        raise CpuFlatScientificError(
            "CPU-flat V3 scientific static contract changed"
        )
    return plan, _standard_binding(binding), scenes


def _read_worker_plan(
    path: Path, *, expected_sha256: str, expected_byte_count: int
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Authenticate the parent-validated plan without repeating parity."""

    raw, binding = pilot.read_bound_json(
        path,
        expected_sha256=expected_sha256,
        expected_byte_count=expected_byte_count,
        label="CPU-flat V3 scientific worker exact plan",
    )
    if (
        not isinstance(raw, dict)
        or raw.get("schema")
        != "lewm_go2_world_model_counterfactual_pilot_plan_v1"
        or raw.get("attempt_id") != plan_builder.SCIENTIFIC_ATTEMPT_ID
        or raw.get("output_root")
        != str(plan_builder.SCIENTIFIC_OUTPUT_ROOT.resolve(strict=True))
        or raw.get("execution_contract", {}).get("backend") != "cpu"
        or raw.get("execution_contract", {}).get("environment")
        != plan_builder.CPU_EXECUTION_ENVIRONMENT
        or raw.get("branch_mechanism")
        != "parallel_lockstep_envs_no_restore"
        or raw.get("expected_counts") != scene_core.EXPECTED_COUNTS
        or raw.get("successor_contract", {}).get("genesis_backend_symbol")
        != "gs.cpu"
        or not plan_builder.SCIENTIFIC_ATTEMPT_ROOT.is_dir()
        or plan_builder.SCIENTIFIC_ATTEMPT_ROOT.is_symlink()
        or not plan_builder.SCIENTIFIC_OUTPUT_ROOT.is_dir()
        or plan_builder.SCIENTIFIC_OUTPUT_ROOT.is_symlink()
    ):
        raise CpuFlatScientificError("worker exact-plan identity changed")
    scenes = scene_core._scene_slices_v2(raw)  # noqa: SLF001
    return raw, _standard_binding(binding), scenes


def _validate_child_environment(plan: Mapping[str, Any]) -> dict[str, str]:
    expected = dict(plan["execution_contract"]["environment"])
    if (
        set(expected) != EXPECTED_ENVIRONMENT_KEYS
        or len(expected) != 11
        or expected != plan_builder.CPU_EXECUTION_ENVIRONMENT
        or expected.get("HOME") != "/home/andrewknowles"
        or expected.get("PATH") != "/usr/bin:/bin"
        or expected.get("GS_BACKEND") != "cpu"
        or expected.get("MESA_VK_DEVICE_SELECT") != "1002:7551!"
        or "HSA_OVERRIDE_GFX_VERSION" in expected
        or "HIP_VISIBLE_DEVICES" in expected
        or "ROCR_VISIBLE_DEVICES" in expected
    ):
        raise CpuFlatScientificError("flat CPU child selectors changed")
    return expected


def _child_environment(plan: Mapping[str, Any]) -> dict[str, str]:
    """Return only the exact fixed eleven-key CPU environment."""

    return _validate_child_environment(plan)


def _require_worker_environment(plan: Mapping[str, Any]) -> None:
    expected = _validate_child_environment(plan)
    observed = dict(os.environ)
    coerced_locale = observed.pop("LC_CTYPE", None)
    if coerced_locale not in {None, "C.UTF-8"} or observed != expected:
        extra = sorted(set(observed) - set(expected))
        missing = sorted(set(expected) - set(observed))
        changed = sorted(
            key
            for key in set(expected) & set(observed)
            if observed[key] != expected[key]
        )
        raise CpuFlatScientificError(
            "worker environment is not exact: "
            f"extra={extra}, missing={missing}, changed={changed}"
        )


def _reserve_scientific(
    *,
    plan_binding: Mapping[str, Any],
    source_review_binding: Mapping[str, Any],
    nonce: str,
) -> tuple[Path, dict[str, Any]]:
    attempt = plan_builder.SCIENTIFIC_ATTEMPT_ROOT
    collection = plan_builder.SCIENTIFIC_OUTPUT_ROOT
    development = (REPO_ROOT / ".generated/dev").resolve(strict=True)
    namespace = attempt.parent
    if (
        attempt.resolve(strict=False).parent != namespace.resolve(strict=False)
        or not attempt.resolve(strict=False).is_relative_to(development)
        or namespace.is_symlink()
        or attempt.exists()
        or attempt.is_symlink()
        or collection.exists()
        or collection.is_symlink()
    ):
        raise CpuFlatScientificError(
            "CPU-flat scientific root is not exact and fresh"
        )
    if not namespace.exists():
        os.mkdir(namespace, mode=0o700)
    if not namespace.is_dir() or namespace.is_symlink():
        raise CpuFlatScientificError("scientific namespace changed")
    os.mkdir(attempt, mode=0o700)
    os.mkdir(collection, mode=0o700)
    os.mkdir(collection / "scenes", mode=0o700)
    os.mkdir(collection / "scenes" / "train", mode=0o700)
    os.mkdir(collection / "scenes" / "eval", mode=0o700)
    os.mkdir(collection / "scene_results", mode=0o700)
    reservation = {
        "schema": RESERVATION_SCHEMA,
        "status": "RESERVED_ONE_SHOT_CPU_FLAT_V3_SCIENCE_CONSUMED",
        "attempt_id": plan_builder.SCIENTIFIC_ATTEMPT_ID,
        "attempt_root": str(attempt.resolve(strict=True)),
        "collection_root": str(collection.resolve(strict=True)),
        "plan_binding": dict(plan_binding),
        "preregistration_binding": _preregistration_binding(),
        "source_review_binding": dict(source_review_binding),
        "qualification_pass_decision_binding": (
            plan_builder.qualification_pass_decision_binding()
        ),
        "independent_qualification_terminal_review_binding": (
            plan_builder.independent_qualification_terminal_review_binding()
        ),
        "orchestrator_nonce": nonce,
        "orchestrator_pid": os.getpid(),
        "scene_indices_in_order": list(SCENE_ORDER),
        "scientific_contract": copy.deepcopy(SCIENTIFIC_CONTRACT),
        "root_creation_consumes_attempt": True,
        "qualification_payload_reuse_authorized": False,
        "retry_authorized": False,
        "resume_authorized": False,
        "overwrite_authorized": False,
        "refill_authorized": False,
    }
    binding = pilot.write_json_exclusive(
        collection / "reservation.json", reservation
    )
    return collection.resolve(strict=True), _standard_binding(binding)


def _read_reservation(
    *,
    binding: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    nonce: str,
    expected_orchestrator_pid: int,
) -> dict[str, Any]:
    value, actual = pilot.read_bound_json(
        Path(str(binding["path"])),
        expected_sha256=str(binding["sha256"]),
        expected_byte_count=int(binding["byte_count"]),
        label="CPU-flat V3 scientific reservation",
    )
    if (
        _standard_binding(actual) != dict(binding)
        or value.get("schema") != RESERVATION_SCHEMA
        or value.get("status")
        != "RESERVED_ONE_SHOT_CPU_FLAT_V3_SCIENCE_CONSUMED"
        or value.get("attempt_id") != plan_builder.SCIENTIFIC_ATTEMPT_ID
        or value.get("plan_binding") != dict(plan_binding)
        or value.get("orchestrator_nonce") != nonce
        or value.get("orchestrator_pid") != expected_orchestrator_pid
        or value.get("scene_indices_in_order") != list(SCENE_ORDER)
        or not isinstance(value.get("source_review_binding"), Mapping)
        or value.get("qualification_pass_decision_binding")
        != plan_builder.qualification_pass_decision_binding()
        or value.get("independent_qualification_terminal_review_binding")
        != plan_builder.independent_qualification_terminal_review_binding()
        or value.get("scientific_contract") != SCIENTIFIC_CONTRACT
        or value.get("qualification_payload_reuse_authorized") is not False
        or any(
            value.get(key) is not False
            for key in (
                "retry_authorized",
                "resume_authorized",
                "overwrite_authorized",
                "refill_authorized",
            )
        )
    ):
        raise CpuFlatScientificError(
            "CPU-flat scientific reservation changed"
        )
    return value


def _clone_worker_runtime(
    runtime: Mapping[str, Any], *, scene_root: Path
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Use the reviewed backend-independent worker-local clone seam."""

    return flat_shared._clone_worker_runtime(runtime, scene_root=scene_root)  # noqa: SLF001


def _all_numbers_finite(value: object) -> bool:
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_all_numbers_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_numbers_finite(item) for item in value)
    return True


def _collect_scene_worker(
    *,
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    reservation_binding: Mapping[str, Any],
    scene_index: int,
    orchestrator_pid: int,
    nonce: str,
) -> dict[str, Any]:
    """Collect one exact four-state/36-lane scene in a fresh CPU process."""

    if os.getppid() != orchestrator_pid or orchestrator_pid <= 1:
        raise CpuFlatScientificError("worker parent ownership changed")
    _require_worker_environment(plan)
    scenes = scene_core._scene_slices_v2(plan)  # noqa: SLF001
    if type(scene_index) is not int or scene_index not in SCENE_ORDER:
        raise CpuFlatScientificError("worker scene index changed")
    scene = scenes[scene_index]
    role = str(scene["role"])
    scene_id = str(scene["scene_id"])
    states = list(scene["states"])
    collection_root = plan_builder.SCIENTIFIC_OUTPUT_ROOT.resolve(strict=True)
    reservation = _read_reservation(
        binding=reservation_binding,
        plan_binding=plan_binding,
        nonce=nonce,
        expected_orchestrator_pid=orchestrator_pid,
    )
    if reservation.get("orchestrator_pid") != orchestrator_pid:
        raise CpuFlatScientificError(
            "worker reservation ownership changed"
        )
    scene_root = collection_root / "scenes" / role / scene_id
    result_path = collection_root / "scene_results" / f"{scene_index:03d}.json"
    failure_path = (
        collection_root / "scene_results" / f"{scene_index:03d}.failure.json"
    )
    if (
        scene_root.exists()
        or scene_root.is_symlink()
        or result_path.exists()
        or result_path.is_symlink()
        or failure_path.exists()
        or failure_path.is_symlink()
    ):
        raise CpuFlatScientificError("worker output is not fresh")
    os.mkdir(scene_root, mode=0o700)
    os.mkdir(scene_root / "state_receipts", mode=0o700)
    started = time.perf_counter()

    kernel._validate_python_runtime(plan)  # noqa: SLF001
    versions = kernel._capture_runtime_versions()  # noqa: SLF001
    imported_runtime = kernel._runtime_imports(textured_v03=True)  # noqa: SLF001
    runtime, cache_root, local_runtime_audit = _clone_worker_runtime(
        imported_runtime, scene_root=scene_root
    )
    platform = runtime["load_platform_manifest"](
        plan["runtime_bindings"]["platform_manifest"]["path"]
    )
    resolved_urdf = runtime["resolve_go2_urdf"](dict(platform), REPO_ROOT)
    if pilot.file_binding(resolved_urdf) != plan["runtime_bindings"]["go2_urdf"]:
        raise CpuFlatScientificError(
            "platform resolved a different Go2 URDF"
        )
    registry = runtime["PrimitiveRegistry"].from_yaml(
        plan["runtime_bindings"]["primitive_registry"]["path"]
    )
    action_blocks = kernel._load_action_blocks(  # noqa: SLF001
        plan=plan,
        registry=registry,
        expand=runtime["expand_primitive_to_block"],
    )
    genesis_initialization = cpu_core._initialize_from_plan_first_scene_cpu(  # noqa: SLF001
        plan=plan
    )
    receipts, frames, quality, sentinels, metrics = kernel._collect_scene(  # noqa: SLF001
        plan=plan,
        states=states,
        runtime=runtime,
        platform=platform,
        registry=registry,
        action_blocks=action_blocks,
    )
    ordered_ids = [str(row["state_id"]) for row in states]
    if [str(row["state"]["state_id"]) for row in receipts] != ordered_ids:
        raise CpuFlatScientificError("worker changed planned state order")
    stored_rgb_bytes = sum(int(frame["byte_count"]) for frame in frames)
    if stored_rgb_bytes > int(
        scene_core.EXPECTED_CAPS["stored_rgb_byte_ceiling"]
    ):
        raise CpuFlatScientificError("worker stored RGB ceiling exceeded")
    render_receipt = {
        "schema": pilot.TEXTURED_V03_LIVE_RENDER_RECEIPT_V3_SCHEMA,
        "attempt_id": str(plan["attempt_id"]),
        "status": "RENDER_COMPLETE",
        "physics_validated": False,
        "citable_as_scientific_evidence": False,
        "scene": {
            "role": role,
            "scene_id": scene_id,
            "family": str(states[0]["family"]),
            "scene_manifest_binding": kernel._scene_receipt_binding(  # noqa: SLF001
                states[0],
                binding_name="scene_manifest_binding",
                output_root=collection_root,
            ),
            "scene_genesis_binding": kernel._scene_receipt_binding(  # noqa: SLF001
                states[0],
                binding_name="scene_genesis_binding",
                output_root=collection_root,
            ),
        },
        **bounded._validated_render_receipt_identity_v1(  # noqa: SLF001
            plan=plan, metrics=metrics
        ),
        "frame_receipts": frames,
        "quality_audits": quality,
        "render_sentinel_audits": sentinels,
    }
    render_binding = pilot.write_json_exclusive(
        scene_root / "live_render_receipt.json", render_receipt
    )
    relative_render = _relative_binding(
        render_binding, output_root=collection_root
    )
    state_bindings: list[dict[str, Any]] = []
    for receipt in receipts:
        state_id = str(receipt["state"]["state_id"])
        receipt["render_receipt_binding"] = relative_render
        binding = pilot.write_json_exclusive(
            scene_root / "state_receipts" / f"{state_id}.json", receipt
        )
        state_bindings.append(
            _relative_binding(binding, output_root=collection_root)
        )
    observed = scene_core._observed_scene_counts_v2(  # noqa: SLF001
        receipts, role=role
    )
    expected = scene_core._scene_expected_counts_v2(role)  # noqa: SLF001
    if observed != expected:
        raise CpuFlatScientificError("worker observed counts changed")
    for name in (
        "native_render_calls",
        "rgb_render_calls",
        "auxiliary_depth_render_calls",
        "stored_rgb_frames",
    ):
        if int(metrics[name]) != scene_core.STORED_FRAMES_PER_SCENE:
            raise CpuFlatScientificError(f"worker {name} changed")
    mesh_cache = scene_core._validate_scene_local_mesh_bindings_v2(  # noqa: SLF001
        metrics,
        cache_root=cache_root,
        collection_root=collection_root,
    )
    result = {
        "schema": SCENE_RESULT_SCHEMA,
        "status": SCENE_PASS_STATUS,
        "attempt_id": str(plan["attempt_id"]),
        "scene_index": scene_index,
        "role": role,
        "scene_id": scene_id,
        "worker_pid": os.getpid(),
        "orchestrator_pid": orchestrator_pid,
        "sys_executable": str(Path(sys.executable).absolute()),
        "fresh_process": True,
        "execution_seed": int(plan["execution_contract"]["seed"]),
        "branch_mechanism": str(plan["branch_mechanism"]),
        "exact_within_state_lane_equality_required": True,
        "genesis_initialization": genesis_initialization,
        "local_runtime_audit": local_runtime_audit,
        "scene_local_mesh_cache": mesh_cache,
        "plan_binding": dict(plan_binding),
        "reservation_binding": dict(reservation_binding),
        "runtime_versions": versions,
        "expected_counts": expected,
        "observed_counts": observed,
        "ordered_state_ids": ordered_ids,
        "state_receipt_bindings": state_bindings,
        "render_receipt_binding": relative_render,
        "scene_metric": metrics,
        "stored_rgb_bytes": stored_rgb_bytes,
        "collection_wall_seconds": time.perf_counter() - started,
        "failure": None,
        "physics_validated": False,
        "citable_as_scientific_evidence": False,
        "qualification_payload_reuse_authorized": False,
        "authorizes_retry_or_resume": False,
        "allows_refill": False,
        "allows_overwrite": False,
        "allows_adaptive_batching": False,
    }
    if (
        not math.isfinite(float(result["collection_wall_seconds"]))
        or result["collection_wall_seconds"] <= 0.0
        or not _all_numbers_finite(result)
    ):
        raise CpuFlatScientificError("worker emitted nonfinite evidence")
    pilot.write_json_exclusive(result_path, result)
    return result


def _write_worker_failure(
    error: Exception,
    *,
    scene_index: int,
    plan_binding: Mapping[str, Any] | None,
    reservation_binding: Mapping[str, Any] | None,
    orchestrator_pid: int | None,
) -> None:
    """Persist complete synchronization diagnostics before worker exit."""

    collection = plan_builder.SCIENTIFIC_OUTPUT_ROOT
    results = collection / "scene_results"
    result_path = results / f"{scene_index:03d}.json"
    failure_path = results / f"{scene_index:03d}.failure.json"
    if (
        type(scene_index) is not int
        or scene_index not in SCENE_ORDER
        or not collection.is_dir()
        or collection.is_symlink()
        or not results.is_dir()
        or results.is_symlink()
        or result_path.exists()
        or result_path.is_symlink()
        or failure_path.exists()
        or failure_path.is_symlink()
    ):
        return
    value = {
        "schema": WORKER_FAILURE_SCHEMA,
        "status": SCENE_FAIL_STATUS,
        "attempt_id": plan_builder.SCIENTIFIC_ATTEMPT_ID,
        "scene_index": scene_index,
        "worker_pid": os.getpid(),
        "orchestrator_pid": orchestrator_pid,
        "plan_binding": dict(plan_binding) if plan_binding is not None else None,
        "reservation_binding": (
            dict(reservation_binding)
            if reservation_binding is not None
            else None
        ),
        "failure": _failure_receipt(error),
        "diagnostics_persisted_if_present": (
            "diagnostics" in _failure_receipt(error)
        ),
        "physics_validated": False,
        "citable_as_scientific_evidence": False,
        "qualification_payload_reuse_authorized": False,
        "authorizes_scientific_plan_release": False,
        "authorizes_retry_or_resume": False,
    }
    try:
        pilot.write_json_exclusive(failure_path, value)
    except Exception:
        return


def _group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _run_worker_with_watchdog(
    argv: Sequence[str],
    *,
    scene: Mapping[str, Any],
    child_env: Mapping[str, str],
    used_path: Path,
) -> dict[str, Any]:
    baseline = scene_core._read_vram_counter_v2(used_path)  # noqa: SLF001
    started = time.monotonic()
    process = subprocess.Popen(
        list(argv),
        cwd=REPO_ROOT,
        env=dict(child_env),
        start_new_session=True,
    )
    pgid = os.getpgid(process.pid)
    peak = baseline
    samples = 1
    read_errors = 0
    timeout = False
    cap_breach = False
    forced_kill = False
    try:
        while process.poll() is None:
            elapsed = time.monotonic() - started
            try:
                used = scene_core._read_vram_counter_v2(used_path)  # noqa: SLF001
            except Exception:
                read_errors += 1
                forced_kill = True
                os.killpg(pgid, signal.SIGKILL)
                break
            samples += 1
            peak = max(peak, used)
            cap_breach = used > VRAM_CEILING_BYTES
            timeout = elapsed > WORKER_TIMEOUT_SECONDS
            if cap_breach or timeout:
                forced_kill = True
                os.killpg(pgid, signal.SIGKILL)
                break
            time.sleep(POLL_SECONDS)
    finally:
        returncode = int(process.wait())
    deadline = time.monotonic() + GROUP_EXIT_TIMEOUT_SECONDS
    while _group_exists(pgid) and time.monotonic() < deadline:
        time.sleep(0.05)
    leaked = _group_exists(pgid)
    if leaked:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            leaked = False
    return {
        "scene_index": int(scene["scene_index"]),
        "role": str(scene["role"]),
        "scene_id": str(scene["scene_id"]),
        "pid": process.pid,
        "parent_pid": os.getpid(),
        "process_group_id": pgid,
        "fresh_process_group": pgid == process.pid,
        "process_group_gone_after_exit": not leaked,
        "prelaunch_baseline_used_bytes": baseline,
        "peak_selected_device_vram_bytes": peak,
        "vram_sample_count": samples,
        "vram_read_errors": read_errors,
        "selected_device_vram_cap_breached": cap_breach,
        "watchdog_timeout": timeout,
        "forced_kill": forced_kill,
        "exit_code": returncode,
        "elapsed_seconds": time.monotonic() - started,
    }


def _validate_worker_receipt(worker: Mapping[str, Any]) -> None:
    if (
        worker.get("fresh_process_group") is not True
        or worker.get("process_group_gone_after_exit") is not True
        or worker.get("exit_code") != 0
        or worker.get("watchdog_timeout") is not False
        or worker.get("selected_device_vram_cap_breached") is not False
        or worker.get("vram_read_errors") != 0
        or int(worker.get("vram_sample_count", 0)) < 2
        or int(worker.get("peak_selected_device_vram_bytes", -1))
        > VRAM_CEILING_BYTES
    ):
        raise CpuFlatScientificError(
            f"worker {worker.get('scene_index')} process/VRAM gate failed"
        )


def _read_worker_failure(
    *,
    scene: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    reservation_binding: Mapping[str, Any],
    worker_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    path = (
        plan_builder.SCIENTIFIC_OUTPUT_ROOT
        / "scene_results"
        / f"{int(scene['scene_index']):03d}.failure.json"
    )
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise CpuFlatScientificError("worker failure path changed")
    binding = _standard_binding(pilot.file_binding(path))
    value, actual = pilot.read_bound_json(
        path,
        expected_sha256=str(binding["sha256"]),
        expected_byte_count=int(binding["byte_count"]),
        label=f"CPU-flat worker {scene['scene_index']} failure",
    )
    failure = value.get("failure")
    if (
        _standard_binding(actual) != binding
        or value.get("schema") != WORKER_FAILURE_SCHEMA
        or value.get("status") != SCENE_FAIL_STATUS
        or value.get("attempt_id") != plan_builder.SCIENTIFIC_ATTEMPT_ID
        or value.get("scene_index") != scene["scene_index"]
        or value.get("worker_pid") != worker_receipt.get("pid")
        or value.get("orchestrator_pid") != os.getpid()
        or value.get("plan_binding") != dict(plan_binding)
        or value.get("reservation_binding") != dict(reservation_binding)
        or not isinstance(failure, Mapping)
        or not isinstance(failure.get("type"), str)
        or not isinstance(failure.get("message"), str)
        or value.get("physics_validated") is not False
        or value.get("citable_as_scientific_evidence") is not False
        or value.get("qualification_payload_reuse_authorized") is not False
        or value.get("authorizes_scientific_plan_release") is not False
        or value.get("authorizes_retry_or_resume") is not False
        or not _all_numbers_finite(value)
    ):
        raise CpuFlatScientificError("worker failure receipt changed")
    if failure.get("type") == "PilotDiagnosticError":
        diagnostics = failure.get("diagnostics")
        if (
            value.get("diagnostics_persisted_if_present") is not True
            or not isinstance(diagnostics, Mapping)
            or not diagnostics
        ):
            raise CpuFlatScientificError(
                "PilotDiagnosticError diagnostics were not persisted"
            )
    relative = _relative_binding(
        actual, output_root=plan_builder.SCIENTIFIC_OUTPUT_ROOT
    )
    return value, relative


def _run_graphics_preflight(
    plan: Mapping[str, Any], *, child_env: Mapping[str, str]
) -> dict[str, Any]:
    """Verify the exact reviewed Vulkan/EGL R9700 render route."""

    runtime = plan["runtime_bindings"]
    expectation = plan["execution_contract"]["graphics_preflight"]
    if (
        dict(child_env) != plan_builder.CPU_EXECUTION_ENVIRONMENT
        or expectation != EXPECTED_GRAPHICS_PREFLIGHT
    ):
        raise CpuFlatScientificError("graphics preflight contract changed")
    try:
        eglinfo = pilot.require_binding(
            runtime["eglinfo_executable"], label="eglinfo executable"
        )
        vulkaninfo = pilot.require_binding(
            runtime["vulkaninfo_executable"], label="vulkaninfo executable"
        )
    except pilot.PilotContractError as exc:
        raise CpuFlatScientificError(str(exc)) from exc
    try:
        vulkan = subprocess.run(
            [str(vulkaninfo["path"]), "--summary"],
            cwd=REPO_ROOT,
            env=dict(child_env),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=90.0,
        )
    except subprocess.TimeoutExpired as exc:
        raise CpuFlatScientificError(
            "Vulkan preflight exceeded 90 seconds"
        ) from exc
    if vulkan.returncode != 0:
        raise CpuFlatScientificError(
            f"Vulkan preflight exited with status {vulkan.returncode}"
        )
    gpu_pattern = re.compile(
        rf"GPU{int(expectation['vulkan_device_index'])}:.*?"
        rf"vendorID\s*=\s*{re.escape(str(expectation['vulkan_vendor_id']))}.*?"
        rf"deviceID\s*=\s*{re.escape(str(expectation['vulkan_device_id']))}.*?"
        rf"deviceName\s*=\s*{re.escape(str(expectation['vulkan_device_name']))}(?:\n|$)",
        re.DOTALL,
    )
    if gpu_pattern.search(vulkan.stdout) is None:
        raise CpuFlatScientificError(
            "selected Vulkan GPU is not the exact bound R9700"
        )
    if re.search(r"^GPU1:", vulkan.stdout, re.MULTILINE) is not None:
        raise CpuFlatScientificError(
            "Vulkan selector did not reduce enumeration to GPU0"
        )
    try:
        egl = subprocess.run(
            [str(eglinfo["path"]), "-B"],
            cwd=REPO_ROOT,
            env=dict(child_env),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=90.0,
        )
    except subprocess.TimeoutExpired as exc:
        raise CpuFlatScientificError(
            "EGL preflight exceeded 90 seconds"
        ) from exc
    if egl.returncode != expectation["eglinfo_expected_exit_code"]:
        raise CpuFlatScientificError(
            f"EGL preflight exited with status {egl.returncode}"
        )
    device_index = int(expectation["egl_device_index"])
    match = re.search(
        rf"Device #{device_index}:\s*\n(?P<section>.*?)(?=\nDevice #\d+:|\Z)",
        egl.stdout,
        re.DOTALL,
    )
    if (
        match is None
        or str(expectation["egl_renderer_name_contains"])
        not in match.group("section")
    ):
        raise CpuFlatScientificError(
            "selected EGL device is not the exact bound R9700"
        )
    return {
        "phase": "cpu_physics_vulkan_egl_graphics_preflight",
        "status": "PASS",
        "environment": dict(plan["execution_contract"]["environment"]),
        "expectation": dict(expectation),
        "genesis_physics_backend": "cpu",
        "render_backend": "vulkan_egl_r9700",
        "vulkan_stdout_sha256": hashlib.sha256(
            vulkan.stdout.encode("utf-8")
        ).hexdigest(),
        "vulkan_stderr_sha256": hashlib.sha256(
            vulkan.stderr.encode("utf-8")
        ).hexdigest(),
        "egl_stdout_sha256": hashlib.sha256(
            egl.stdout.encode("utf-8")
        ).hexdigest(),
        "egl_stderr_sha256": hashlib.sha256(
            egl.stderr.encode("utf-8")
        ).hexdigest(),
        "egl_exit_code": int(egl.returncode),
    }


def _kernel_events_since(epoch_seconds: float) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "/usr/bin/journalctl",
            "-k",
            "--since",
            f"@{epoch_seconds:.6f}",
            "--no-pager",
            "-o",
            "short-unix",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=15.0,
    )
    if completed.returncode != 0:
        raise CpuFlatScientificError("kernel reset audit was unavailable")
    pattern = re.compile(
        r"(?:amdgpu.*(?:ring .* timeout|ring .* reset|device wedged|GPU reset)"
        r"|(?:hsa|kfd).*(?:exception|queue.*error|memory fault))",
        re.IGNORECASE,
    )
    events = [line for line in completed.stdout.splitlines() if pattern.search(line)]
    return {
        "query_succeeded": True,
        "new_amdgpu_reset_wedge_or_hsa_exception_count": len(events),
        "matching_lines_sha256": hashlib.sha256(
            "\n".join(events).encode("utf-8")
        ).hexdigest(),
    }


def _worker_argv(
    *,
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    reservation_binding: Mapping[str, Any],
    scene_index: int,
    nonce: str,
) -> list[str]:
    invocation = Path(str(plan["execution_contract"]["python_invocation_path"]))
    expected = REPO_ROOT / ".generated/venvs/genesis_render_vulkan/bin/python"
    target = Path(
        str(plan["runtime_bindings"]["python_executable_target"]["path"])
    )
    if (
        not invocation.is_absolute()
        or invocation != expected.absolute()
        or invocation.resolve(strict=True) != target.resolve(strict=True)
    ):
        raise CpuFlatScientificError("worker Python invocation changed")
    return [
        str(invocation.absolute()),
        str(Path(__file__).resolve()),
        "--worker-scene-index",
        str(scene_index),
        "--plan",
        str(plan_binding["path"]),
        "--expected-plan-sha256",
        str(plan_binding["sha256"]),
        "--expected-plan-byte-count",
        str(plan_binding["byte_count"]),
        "--reservation",
        str(reservation_binding["path"]),
        "--expected-reservation-sha256",
        str(reservation_binding["sha256"]),
        "--expected-reservation-byte-count",
        str(reservation_binding["byte_count"]),
        "--orchestrator-pid",
        str(os.getpid()),
        "--orchestrator-nonce",
        nonce,
    ]


def _validate_png_frame_receipt(
    frame: Mapping[str, Any],
    *,
    collection_root: Path,
    expected_prefix: PurePosixPath,
    seen_paths: set[str],
    seen_identities: set[str],
) -> dict[str, Any]:
    """Rehash and decode one exact in-root single-frame RGB PNG."""

    identity = frame.get("frame_identity")
    artifact_id = frame.get("artifact_id")
    pixel_sha256 = frame.get("pixel_sha256")
    if (
        not isinstance(identity, str)
        or not identity
        or artifact_id != identity
        or identity in seen_identities
        or frame.get("width") != 224
        or frame.get("height") != 224
        or frame.get("mode") != "RGB"
        or frame.get("format") != "PNG"
        or frame.get("camera_valid") is not True
        or not isinstance(pixel_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", pixel_sha256) is None
    ):
        raise CpuFlatScientificError("render frame identity/metadata changed")
    checked = scene_core._rehash_relative_binding_v2(  # noqa: SLF001
        {
            "path": frame.get("path"),
            "file_sha256": frame.get("file_sha256"),
            "byte_count": frame.get("byte_count"),
        },
        collection_root=collection_root,
        label=f"CPU-flat qualification PNG {identity}",
    )
    relative = PurePosixPath(str(checked["path"]))
    if (
        relative in {PurePosixPath("."), PurePosixPath("")}
        or not relative.is_relative_to(expected_prefix)
        or relative.suffix.lower() != ".png"
        or str(relative) in seen_paths
    ):
        raise CpuFlatScientificError("render frame path changed or repeated")
    image_path = collection_root.joinpath(*relative.parts)
    try:
        from PIL import Image, UnidentifiedImageError

        with Image.open(image_path) as image:
            if (
                image.format != "PNG"
                or image.mode != "RGB"
                or image.size != (224, 224)
                or getattr(image, "n_frames", 1) != 1
            ):
                raise CpuFlatScientificError(
                    "render artifact is not one 224x224 RGB PNG"
                )
            image.load()
            observed_pixels = hashlib.sha256(image.tobytes()).hexdigest()
    except CpuFlatScientificError:
        raise
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise CpuFlatScientificError("render artifact cannot be decoded") from exc
    if observed_pixels != pixel_sha256:
        raise CpuFlatScientificError("render pixel identity changed")
    seen_paths.add(str(relative))
    seen_identities.add(identity)
    return dict(frame)


def _read_scene_result_after_release(
    *,
    scene: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    reservation_binding: Mapping[str, Any],
    worker_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Deeply reopen all scene evidence only after the release barrier."""

    path = (
        plan_builder.SCIENTIFIC_OUTPUT_ROOT
        / "scene_results"
        / f"{int(scene['scene_index']):03d}.json"
    )
    binding = _standard_binding(pilot.file_binding(path))
    value, actual = pilot.read_bound_json(
        path,
        expected_sha256=str(binding["sha256"]),
        expected_byte_count=int(binding["byte_count"]),
        label=f"CPU-flat qualification scene {scene['scene_index']} result",
    )
    expected_counts = scene_core._scene_expected_counts_v2(  # noqa: SLF001
        str(scene["role"])
    )
    role = str(scene["role"])
    scene_id = str(scene["scene_id"])
    ordered_ids = [str(row["state_id"]) for row in scene["states"]]
    state_bindings = value.get("state_receipt_bindings")
    render_binding = value.get("render_receipt_binding")
    metric = value.get("scene_metric")
    mesh_cache = value.get("scene_local_mesh_cache")
    initialization = value.get("genesis_initialization")
    versions = value.get("runtime_versions")
    expected_python = str(
        Path(plan["execution_contract"]["python_invocation_path"]).absolute()
    )
    if (
        _standard_binding(actual) != binding
        or value.get("schema") != SCENE_RESULT_SCHEMA
        or value.get("status") != SCENE_PASS_STATUS
        or value.get("attempt_id") != plan_builder.SCIENTIFIC_ATTEMPT_ID
        or value.get("scene_index") != scene["scene_index"]
        or value.get("role") != role
        or value.get("scene_id") != scene_id
        or value.get("worker_pid") != worker_receipt.get("pid")
        or value.get("orchestrator_pid") != os.getpid()
        or value.get("sys_executable") != expected_python
        or value.get("fresh_process") is not True
        or value.get("execution_seed") != plan["execution_contract"]["seed"]
        or value.get("branch_mechanism")
        != "parallel_lockstep_envs_no_restore"
        or value.get("exact_within_state_lane_equality_required") is not True
        or value.get("plan_binding") != dict(plan_binding)
        or value.get("reservation_binding") != dict(reservation_binding)
        or value.get("expected_counts") != expected_counts
        or value.get("observed_counts") != expected_counts
        or value.get("ordered_state_ids") != ordered_ids
        or not isinstance(state_bindings, list)
        or len(state_bindings) != 4
        or not isinstance(render_binding, Mapping)
        or not isinstance(metric, Mapping)
        or metric.get("role") != role
        or metric.get("scene_id") != scene_id
        or metric.get("states") != 4
        or not isinstance(mesh_cache, Mapping)
        or not isinstance(initialization, Mapping)
        or initialization.get("source")
        != "full_plan_first_scene_bound_manifest"
        or initialization.get("full_physics_seed")
        != cpu_core.PLAN_FIRST_PHYSICS_SEED
        or initialization.get("effective_genesis_seed")
        != cpu_core.PLAN_FIRST_EFFECTIVE_GENESIS_SEED
        or initialization.get("backend") != "cpu"
        or not isinstance(versions, Mapping)
        or versions.get("genesis") != "0.3.14"
        or value.get("local_runtime_audit", {}).get(
            "original_renderer_globals_unchanged"
        )
        is not True
        or value.get("local_runtime_audit", {}).get(
            "renderer_code_identity_preserved"
        )
        is not True
        or value.get("local_runtime_audit", {}).get("renderer_globals_cloned")
        is not True
        or value.get("local_runtime_audit", {}).get(
            "rollout_config_worker_local"
        )
        is not True
        or value.get("local_runtime_audit", {}).get("foot_contact_source")
        != "zero"
        or value.get("qualification_payload_reuse_authorized") is not False
        or value.get("authorizes_retry_or_resume") is not False
        or value.get("allows_refill") is not False
        or value.get("allows_overwrite") is not False
        or value.get("allows_adaptive_batching") is not False
        or value.get("failure") is not None
        or not _all_numbers_finite(value)
    ):
        raise CpuFlatScientificError(
            f"scene {scene['scene_index']} result gate failed"
        )
    checked_render = scene_core._rehash_relative_binding_v2(  # noqa: SLF001
        render_binding,
        collection_root=plan_builder.SCIENTIFIC_OUTPUT_ROOT,
        label="CPU-flat qualification render receipt",
    )
    expected_render_path = PurePosixPath(
        "scenes", role, scene_id, "live_render_receipt.json"
    )
    if PurePosixPath(str(checked_render["path"])) != expected_render_path:
        raise CpuFlatScientificError("render receipt path changed")
    render, reopened_render = pilot.read_bound_json(
        plan_builder.SCIENTIFIC_OUTPUT_ROOT.joinpath(
            *expected_render_path.parts
        ),
        expected_sha256=str(checked_render["file_sha256"]),
        expected_byte_count=int(checked_render["byte_count"]),
        label="CPU-flat qualification render receipt",
    )
    if (
        reopened_render.get("file_sha256") != checked_render["file_sha256"]
        or reopened_render.get("byte_count") != checked_render["byte_count"]
        or render.get("schema")
        != pilot.TEXTURED_V03_LIVE_RENDER_RECEIPT_V3_SCHEMA
        or render.get("attempt_id") != plan_builder.SCIENTIFIC_ATTEMPT_ID
        or render.get("status") != "RENDER_COMPLETE"
        or render.get("physics_validated") is not False
        or render.get("citable_as_scientific_evidence") is not False
        or render.get("scene", {}).get("role") != role
        or render.get("scene", {}).get("scene_id") != scene_id
        or render.get("scene", {}).get("family")
        != scene["states"][0].get("family")
        or len(render.get("frame_receipts", [])) != 48
    ):
        raise CpuFlatScientificError("render receipt content changed")
    frame_prefix = PurePosixPath("scenes", role, scene_id, "rgb")
    seen_frame_paths: set[str] = set()
    seen_frame_identities: set[str] = set()
    render_frames_by_identity: dict[str, dict[str, Any]] = {}
    for raw_frame in render["frame_receipts"]:
        if not isinstance(raw_frame, Mapping):
            raise CpuFlatScientificError("render frame receipt changed")
        frame = _validate_png_frame_receipt(
            raw_frame,
            collection_root=plan_builder.SCIENTIFIC_OUTPUT_ROOT,
            expected_prefix=frame_prefix,
            seen_paths=seen_frame_paths,
            seen_identities=seen_frame_identities,
        )
        render_frames_by_identity[str(frame["frame_identity"])] = frame
    if (
        len(render_frames_by_identity) != 48
        or len(seen_frame_paths) != 48
        or len(seen_frame_identities) != 48
    ):
        raise CpuFlatScientificError("render frame closure changed")

    referenced_frame_identities: set[str] = set()
    for declared, state_id, raw_binding in zip(
        scene["states"], ordered_ids, state_bindings, strict=True
    ):
        checked = scene_core._rehash_relative_binding_v2(  # noqa: SLF001
            raw_binding,
            collection_root=plan_builder.SCIENTIFIC_OUTPUT_ROOT,
            label="CPU-flat qualification state receipt",
        )
        expected_path = PurePosixPath(
            "scenes", role, scene_id, "state_receipts", f"{state_id}.json"
        )
        if PurePosixPath(str(checked["path"])) != expected_path:
            raise CpuFlatScientificError(
                "state receipt order/path changed"
            )
        receipt_path = plan_builder.SCIENTIFIC_OUTPUT_ROOT.joinpath(
            *expected_path.parts
        )
        receipt, reopened = pilot.read_bound_json(
            receipt_path,
            expected_sha256=str(checked["file_sha256"]),
            expected_byte_count=int(checked["byte_count"]),
            label=f"CPU-flat qualification state receipt {state_id}",
        )
        state = receipt.get("state", {})
        synchronization = receipt.get("synchronization_audit", {})
        lane_hashes = synchronization.get("lane_state_sha256s")
        components = synchronization.get("components")
        branches = receipt.get("branches")
        context = receipt.get("context", {})
        context_identities = context.get("frame_identities")
        if (
            reopened.get("file_sha256") != checked["file_sha256"]
            or reopened.get("byte_count") != checked["byte_count"]
            or receipt.get("schema") != pilot.TEXTURED_V03_STATE_RECEIPT_SCHEMA
            or receipt.get("attempt_id")
            != plan_builder.SCIENTIFIC_ATTEMPT_ID
            or receipt.get("status") != "PHYSICS_COMPLETE"
            or receipt.get("physics_validated") is not False
            or receipt.get("citable_as_scientific_evidence") is not False
            or receipt.get("authorizes_retry_or_resume") is not False
            or state.get("state_id") != state_id
            or state.get("role") != role
            or state.get("scene_id") != scene_id
            or state.get("family") != declared.get("family")
            or state.get("group_index") != declared.get("group_index")
            or state.get("state_index_in_scene")
            != declared.get("state_index_in_scene")
            or state.get("lane_count") != 9
            or receipt.get("render_receipt_binding") != render_binding
            or synchronization.get("state_id") != state_id
            or synchronization.get("group_index")
            != declared.get("group_index")
            or synchronization.get("lane_start") != state.get("lane_start")
            or synchronization.get("lane_count") != 9
            or synchronization.get("exact_equality_required") is not True
            or synchronization.get("passed") is not True
            or not isinstance(lane_hashes, list)
            or len(lane_hashes) != 9
            or len(set(lane_hashes)) != 1
            or synchronization.get("prebranch_state_sha256") != lane_hashes[0]
            or not isinstance(components, Mapping)
            or set(components) != set(pilot.SYNC_COMPONENTS)
            or any(
                not isinstance(component, Mapping)
                or component.get("exact_equal") is not True
                or float(component.get("max_abs_difference", -1.0)) != 0.0
                or float(component.get("rms_difference", -1.0)) != 0.0
                or component.get("per_lane_max_abs_difference")
                != [0.0] * 9
                for component in components.values()
            )
            or not isinstance(branches, list)
            or len(branches) != 9
            or not isinstance(context_identities, list)
            or len(context_identities) != 3
        ):
            raise CpuFlatScientificError("state receipt content changed")
        for identity in context_identities:
            if (
                not isinstance(identity, str)
                or identity not in render_frames_by_identity
                or identity in referenced_frame_identities
            ):
                raise CpuFlatScientificError(
                    "state context render-frame closure changed"
                )
            referenced_frame_identities.add(identity)
        for expected_action_id, branch in enumerate(branches):
            if not isinstance(branch, Mapping):
                raise CpuFlatScientificError("candidate branch changed")
            frame = branch.get("frame_receipt")
            identity = branch.get("render_frame_identity")
            if (
                branch.get("lane_offset") != expected_action_id
                or branch.get("lane_index")
                != int(state["lane_start"]) + expected_action_id
                or branch.get("kind") != "candidate"
                or branch.get("action_id") != expected_action_id
                or branch.get("action_name")
                != plan["action_catalog"][expected_action_id]["name"]
                or branch.get("duplicates_candidate_action_id") is not None
                or not isinstance(identity, str)
                or not isinstance(frame, Mapping)
                or frame.get("frame_identity") != identity
                or render_frames_by_identity.get(identity) != frame
                or identity in referenced_frame_identities
            ):
                raise CpuFlatScientificError(
                    "candidate lane/action/render closure changed"
                )
            referenced_frame_identities.add(identity)
    if referenced_frame_identities != set(render_frames_by_identity):
        raise CpuFlatScientificError(
            "state receipts do not exactly cover render frames"
        )
    observed_mesh = scene_core._validate_scene_local_mesh_bindings_v2(  # noqa: SLF001
        metric,
        cache_root=(
            plan_builder.SCIENTIFIC_OUTPUT_ROOT
            / "scenes"
            / role
            / scene_id
            / "derived_meshes"
        ),
        collection_root=plan_builder.SCIENTIFIC_OUTPUT_ROOT,
    )
    if observed_mesh != dict(mesh_cache):
        raise CpuFlatScientificError("scene-local mesh cache changed")
    relative_result = _relative_binding(
        actual, output_root=plan_builder.SCIENTIFIC_OUTPUT_ROOT
    )
    if PurePosixPath(str(relative_result["path"])) != PurePosixPath(
        "scene_results", f"{int(scene['scene_index']):03d}.json"
    ):
        raise CpuFlatScientificError("scene result index path changed")
    return value, relative_result


def _remaining_wall_seconds(started: float) -> float:
    remaining = WALL_CEILING_SECONDS - (time.monotonic() - started)
    if not math.isfinite(remaining) or remaining <= 0.0:
        raise CpuFlatScientificError("whole scientific run exceeded wall cap")
    return remaining


def _validate_frozen_protocol() -> dict[str, Any]:
    protocol = plan_builder.SCIENTIFIC_PROTOCOL
    config = benchmark.config_v1()
    frozen = config.get("frozen_recurrent_protocol", {})
    if (
        protocol.get("learned_arms_in_order") != list(benchmark.ARM_ORDER)
        or protocol.get("live_control_arm") != "task_action_only"
        or protocol.get("model_seeds") != list(benchmark.MODEL_SEEDS)
        or protocol.get("shared_sampler_seed") != benchmark.SAMPLER_SEED
        or protocol.get("updates") != benchmark.UPDATES
        or protocol.get("batch_states") != benchmark.BATCH_STATES
        or protocol.get("successor_observation_access") is not False
        or protocol.get("durable_checkpoint_required_before_eval_open")
        is not True
        or protocol.get("two_exact_evaluations_required") is not True
        or frozen.get("arms") != list(benchmark.ARM_ORDER)
        or frozen.get("model_seeds") != list(benchmark.MODEL_SEEDS)
        or frozen.get("updates") != 800
    ):
        raise CpuFlatScientificError("frozen three-arm protocol changed")
    return copy.deepcopy(protocol)


def _join_collection(
    *,
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    plan_receipt_binding: Mapping[str, Any],
    source_review_binding: Mapping[str, Any],
    reservation_binding: Mapping[str, Any],
    scenes: Sequence[Mapping[str, Any]],
    scene_results: Sequence[Mapping[str, Any]],
    scene_result_bindings: Sequence[Mapping[str, Any]],
    worker_receipts: Sequence[Mapping[str, Any]],
    release_barriers: Sequence[Mapping[str, Any]],
    collection_wall_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Final rehash barrier and frozen-role-loader compatible index."""

    collection_root = plan_builder.SCIENTIFIC_OUTPUT_ROOT.resolve(strict=True)
    if not (
        len(scenes)
        == len(scene_results)
        == len(scene_result_bindings)
        == len(worker_receipts)
        == len(release_barriers)
        == scene_core.SCENE_COUNT
    ):
        raise CpuFlatScientificError("64-scene collection closure changed")
    for expected_index, (scene, result, result_binding, worker, release) in enumerate(
        zip(
            scenes,
            scene_results,
            scene_result_bindings,
            worker_receipts,
            release_barriers,
            strict=True,
        )
    ):
        scene_core._rehash_relative_binding_v2(  # noqa: SLF001
            result_binding,
            collection_root=collection_root,
            label="final scientific scene-result join",
        )
        for binding in result["state_receipt_bindings"]:
            scene_core._rehash_relative_binding_v2(  # noqa: SLF001
                binding,
                collection_root=collection_root,
                label="final scientific state-receipt join",
            )
        scene_core._rehash_relative_binding_v2(  # noqa: SLF001
            result["render_receipt_binding"],
            collection_root=collection_root,
            label="final scientific render-receipt join",
        )
        observed_mesh = scene_core._validate_scene_local_mesh_bindings_v2(  # noqa: SLF001
            result["scene_metric"],
            cache_root=(
                collection_root
                / "scenes"
                / str(scene["role"])
                / str(scene["scene_id"])
                / "derived_meshes"
            ),
            collection_root=collection_root,
        )
        if (
            expected_index != int(scene["scene_index"])
            or result.get("scene_index") != expected_index
            or result.get("role") != scene["role"]
            or result.get("scene_id") != scene["scene_id"]
            or observed_mesh != result.get("scene_local_mesh_cache")
            or worker.get("scene_index") != expected_index
            or release.get("status") != "PASSED"
        ):
            raise CpuFlatScientificError("final scene/process order changed")

    state_bindings = [
        dict(binding)
        for result in scene_results
        for binding in result["state_receipt_bindings"]
    ]
    render_bindings = [
        dict(result["render_receipt_binding"]) for result in scene_results
    ]
    metrics = [dict(result["scene_metric"]) for result in scene_results]
    all_paths = [
        str(binding["path"])
        for binding in (
            *scene_result_bindings,
            *state_bindings,
            *render_bindings,
        )
    ]
    stored_rgb_bytes = sum(
        int(result["stored_rgb_bytes"]) for result in scene_results
    )
    if (
        len(state_bindings) != scene_core.EXPECTED_COUNTS["states"]
        or len(render_bindings) != scene_core.EXPECTED_COUNTS["scenes"]
        or len(metrics) != scene_core.EXPECTED_COUNTS["scenes"]
        or len(all_paths) != len(set(all_paths))
        or stored_rgb_bytes
        > int(scene_core.EXPECTED_CAPS["stored_rgb_byte_ceiling"])
        or any(
            result["runtime_versions"] != scene_results[0]["runtime_versions"]
            for result in scene_results[1:]
        )
        or len({int(worker["pid"]) for worker in worker_receipts}) != 64
    ):
        raise CpuFlatScientificError("combined scientific scene closure changed")
    for name in (
        "native_render_calls",
        "rgb_render_calls",
        "auxiliary_depth_render_calls",
        "stored_rgb_frames",
    ):
        if sum(int(row[name]) for row in metrics) != int(
            scene_core.EXPECTED_CAPS[name]
        ):
            raise CpuFlatScientificError(f"combined {name} changed")

    source_bindings = _source_bindings()
    authority = {
        "attempt_id": plan_builder.SCIENTIFIC_ATTEMPT_ID,
        "attempt_root": str(plan_builder.SCIENTIFIC_ATTEMPT_ROOT.resolve()),
        "collection_root": str(collection_root),
        "caps": copy.deepcopy(scene_core.EXPECTED_CAPS),
        "source_bindings": source_bindings,
        "plan_binding": dict(plan_binding),
        "preregistration_binding": _preregistration_binding(),
        "source_review_binding": dict(source_review_binding),
        "dino": _validate_dino_inputs(),
    }
    historical_plan_binding = {
        "path": str(plan_binding["path"]),
        "file_sha256": str(plan_binding["sha256"]),
        "byte_count": int(plan_binding["byte_count"]),
    }
    physics_result = {
        "schema": pilot.PHYSICS_RESULT_SCHEMA,
        "attempt_id": plan_builder.SCIENTIFIC_ATTEMPT_ID,
        "purpose": str(plan["purpose"]),
        "status": "PHYSICS_COMPLETE",
        "physics_validated": False,
        "citable_as_scientific_evidence": False,
        "development_only": True,
        "authorizes_retry_or_resume": False,
        "allows_refill": False,
        "allows_overwrite": False,
        "branch_mechanism": "parallel_lockstep_envs_no_restore",
        "plan_binding": historical_plan_binding,
        "plan_receipt_binding": dict(plan_receipt_binding),
        "authority_binding": dict(source_review_binding),
        "reservation_binding": dict(reservation_binding),
        "caps": copy.deepcopy(scene_core.EXPECTED_CAPS),
        "execution_contract": copy.deepcopy(plan["execution_contract"]),
        "runtime_versions": copy.deepcopy(scene_results[0]["runtime_versions"]),
        "runtime_bindings": copy.deepcopy(plan["runtime_bindings"]),
        "source_bindings": source_bindings,
        "expected_counts": copy.deepcopy(scene_core.EXPECTED_COUNTS),
        "observed_counts": copy.deepcopy(scene_core.EXPECTED_COUNTS),
        "scene_materialization": None,
        "state_receipt_bindings": state_bindings,
        "render_receipt_bindings": render_bindings,
        "scene_metrics": metrics,
        "visual_domain_limitation": (
            "textured-v03 historical RGB plus transient depth validation; "
            "no scene screening, refill, or qualification payload reuse"
        ),
        "stored_rgb_bytes": stored_rgb_bytes,
        "collection_wall_seconds": collection_wall_seconds,
        "scene_result_bindings": [dict(row) for row in scene_result_bindings],
        "scene_process_evidence": {
            "schema": (
                "lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_"
                "flat_development_v3_scientific_process_evidence_v1"
            ),
            "scene_indices_in_order": list(SCENE_ORDER),
            "worker_receipts": [dict(row) for row in worker_receipts],
            "release_barriers": [dict(row) for row in release_barriers],
            "all_workers_fresh": True,
            "release_before_result_open_and_next_worker": True,
        },
        "qualification_payload_reuse_authorized": False,
        "failure": None,
    }
    if (
        not math.isfinite(collection_wall_seconds)
        or collection_wall_seconds <= 0.0
        or not _all_numbers_finite(physics_result)
    ):
        raise CpuFlatScientificError("combined physics result is nonfinite")
    pilot.write_json_exclusive(
        collection_root / "physics_result.json", physics_result
    )
    return authority, physics_result


def _run_frozen_model_stage(
    *,
    authority: Mapping[str, Any],
    source_review_binding: Mapping[str, Any],
    plan: Mapping[str, Any],
    wall_started: float,
) -> dict[str, Any]:
    """Run the frozen fit/checkpoint/evaluate sequence on explicit CPU."""

    protocol = _validate_frozen_protocol()
    if (
        torch.cuda.is_available() is not False
        or torch.cuda.device_count() != 0
        or torch.version.hip is not None
    ):
        raise CpuFlatScientificError(
            "bound scientific interpreter is not CPU-only for model compute"
        )
    device = torch.device("cpu")
    compute_runtime = {
        "compute_device": "cpu",
        "torch_cuda_available": False,
        "torch_cuda_device_count": 0,
        "torch_hip_version": None,
        "render_device": "AMD Radeon AI PRO R9700 via Vulkan/EGL only",
        "render_and_compute_devices_separated": True,
    }
    physics_index = frozen_runner._load_physics_index_v1(  # noqa: SLF001
        authority, source_review_binding, plan
    )
    determinism = upstream.configure_determinism_v1()
    ledger = frozen_runner.ContextOnlyLedgerV1()
    ledger.load_receipts("train")
    train = frozen_runner._load_role_runtime_data_v1(  # noqa: SLF001
        authority, plan, physics_index, role="train", ledger=ledger
    )
    _remaining_wall_seconds(wall_started)
    dino = upstream.load_dino_trunk_v1(
        Path(str(authority["dino"]["repository_path"])),
        Path(str(authority["dino"]["checkpoint_binding"]["path"])),
        device=device,
    )
    train_context = frozen_runner._full_dino_context_tokens_v1(  # noqa: SLF001
        train, ledger=ledger, dino=dino
    )
    _remaining_wall_seconds(wall_started)
    checkpoint = benchmark.fit_checkpoint_v1(
        train, train_context, device=device
    )
    checkpoint_path = plan_builder.SCIENTIFIC_ATTEMPT_ROOT / "checkpoint.pt"
    checkpoint = frozen_runner._save_checkpoint_exclusive(  # noqa: SLF001
        checkpoint_path, checkpoint
    )
    checkpoint_binding = frozen_runner.file_binding_v1(checkpoint_path)
    ledger.checkpoint()
    del train_context
    _remaining_wall_seconds(wall_started)

    ledger.load_receipts("eval")
    evaluation = frozen_runner._load_role_runtime_data_v1(  # noqa: SLF001
        authority, plan, physics_index, role="eval", ledger=ledger
    )
    disjointness = frozen_runner.assert_role_disjointness_v1(
        train.plan, evaluation.plan
    )
    eval_context = frozen_runner._full_dino_context_tokens_v1(  # noqa: SLF001
        evaluation, ledger=ledger, dino=dino
    )
    first = benchmark.evaluate_checkpoint_v1(
        checkpoint,
        train,
        evaluation,
        eval_context,
        device=device,
        integrity_passed=True,
    )
    second = benchmark.evaluate_checkpoint_v1(
        checkpoint,
        train,
        evaluation,
        eval_context,
        device=device,
        integrity_passed=True,
    )
    if frozen_runner.canonical_bytes_v1(first) != (
        frozen_runner.canonical_bytes_v1(second)
    ):
        raise CpuFlatScientificError("repeat evaluation was not bitwise exact")
    if (
        set(first.get("reports", {}))
        != {
            "no_vision_recurrent_direct",
            "visual_recurrent_direct",
            "task_action_only",
            "privileged_physical_oracle",
            "random_expected",
        }
        or first.get("successor_observations_opened") != 0
        or set(first.get("gate", {}).get("gates", {}))
        != {
            "1_integrity_and_oracle",
            "2_absolute_regret",
            "3_visual_beats_task_action_only",
            "4_visual_beats_no_vision",
            "5_visual_beats_random",
        }
    ):
        raise CpuFlatScientificError("frozen evaluation report/gates changed")
    access_audit = ledger.finalized()
    _remaining_wall_seconds(wall_started)
    return {
        "protocol": protocol,
        "compute_runtime": compute_runtime,
        "determinism": determinism,
        "physics_result_binding": dict(physics_index["_binding"]),
        "checkpoint": checkpoint,
        "checkpoint_binding": checkpoint_binding,
        "train": train,
        "evaluation_role": evaluation,
        "evaluation": first,
        "repeat_evaluation_exact": True,
        "role_disjointness": disjointness,
        "access_audit": access_audit,
    }


def execute_scientific(
    *,
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    scenes: Sequence[Mapping[str, Any]],
    source_review: Mapping[str, Any],
    source_review_binding: Mapping[str, Any],
) -> dict[str, Any]:
    child_env = _child_environment(plan)
    nonce = secrets.token_hex(32)
    wall_started = time.monotonic()
    started_epoch = time.time()
    collection_root, reservation_binding = _reserve_scientific(
        plan_binding=plan_binding,
        source_review_binding=source_review_binding,
        nonce=nonce,
    )
    historical_plan_binding = {
        "path": str(plan_binding["path"]),
        "file_sha256": str(plan_binding["sha256"]),
        "byte_count": int(plan_binding["byte_count"]),
    }
    plan_receipt_binding = kernel._copy_exact_plan_receipt(  # noqa: SLF001
        historical_plan_binding, output_root=collection_root
    )
    _read_reservation(
        binding=reservation_binding,
        plan_binding=plan_binding,
        nonce=nonce,
        expected_orchestrator_pid=os.getpid(),
    )
    used_path, total_path, vendor, device_id = (
        calibration._selected_gpu_memory_files(plan)  # noqa: SLF001
    )
    gpu_sampler = calibration._GlobalVramSampler(  # noqa: SLF001
        used_path,
        total_path,
        vendor_id=vendor,
        device_id=device_id,
        interval_seconds=POLL_SECONDS,
    )
    gpu_sampler.start()
    gpu_measurement: dict[str, Any] | None = None
    try:
        graphics_preflight = _run_graphics_preflight(
            plan, child_env=child_env
        )
        _remaining_wall_seconds(wall_started)
        scene_results: list[dict[str, Any]] = []
        scene_result_bindings: list[dict[str, Any]] = []
        workers: list[dict[str, Any]] = []
        releases: list[dict[str, Any]] = []
        for scene_index in SCENE_ORDER:
            scene = scenes[scene_index]
            worker = _run_worker_with_watchdog(
                _worker_argv(
                    plan=plan,
                    plan_binding=plan_binding,
                    reservation_binding=reservation_binding,
                    scene_index=scene_index,
                    nonce=nonce,
                ),
                scene=scene,
                child_env=child_env,
                used_path=used_path,
            )
            release = scene_core._wait_for_vram_release_v2(  # noqa: SLF001
                used_path,
                baseline_used_bytes=int(
                    worker["prelaunch_baseline_used_bytes"]
                ),
                ceiling_bytes=VRAM_CEILING_BYTES,
            )
            worker_failure = _read_worker_failure(
                scene=scene,
                plan_binding=plan_binding,
                reservation_binding=reservation_binding,
                worker_receipt=worker,
            )
            try:
                _validate_worker_receipt(worker)
            except Exception as exc:
                raise CpuFlatScientificError(
                    f"scene {scene_index} worker failed",
                    diagnostics={
                        "phase": "scene_worker_process",
                        "scene_index": scene_index,
                        "worker": dict(worker),
                        "release_barrier": dict(release),
                        "worker_failure_receipt": (
                            worker_failure[0]
                            if worker_failure is not None
                            else None
                        ),
                        "worker_failure_binding": (
                            worker_failure[1]
                            if worker_failure is not None
                            else None
                        ),
                    },
                ) from exc
            if worker_failure is not None:
                raise CpuFlatScientificError(
                    f"scene {scene_index} emitted a failure receipt",
                    diagnostics={
                        "phase": "scene_worker_failure_receipt",
                        "scene_index": scene_index,
                        "worker": dict(worker),
                        "release_barrier": dict(release),
                        "worker_failure_receipt": worker_failure[0],
                        "worker_failure_binding": worker_failure[1],
                    },
                )
            result, result_binding = _read_scene_result_after_release(
                scene=scene,
                plan=plan,
                plan_binding=plan_binding,
                reservation_binding=reservation_binding,
                worker_receipt=worker,
            )
            scene_results.append(result)
            scene_result_bindings.append(result_binding)
            workers.append(worker)
            releases.append(release)
            _remaining_wall_seconds(wall_started)

        collection_wall_seconds = time.monotonic() - wall_started
        authority, physics_result = _join_collection(
            plan=plan,
            plan_binding=plan_binding,
            plan_receipt_binding=plan_receipt_binding,
            source_review_binding=source_review_binding,
            reservation_binding=reservation_binding,
            scenes=scenes,
            scene_results=scene_results,
            scene_result_bindings=scene_result_bindings,
            worker_receipts=workers,
            release_barriers=releases,
            collection_wall_seconds=collection_wall_seconds,
        )
        kernel_audit = _kernel_events_since(started_epoch)
        if (
            kernel_audit[
                "new_amdgpu_reset_wedge_or_hsa_exception_count"
            ]
            != 0
        ):
            raise CpuFlatScientificError("render-phase kernel gate failed")
        gpu_measurement = gpu_sampler.stop()
        if (
            gpu_measurement.get("read_errors") != 0
            or int(gpu_measurement.get("peak_used_bytes", -1))
            > VRAM_CEILING_BYTES
        ):
            raise CpuFlatScientificError(
                "render-phase selected-device VRAM gate failed"
            )
        _remaining_wall_seconds(wall_started)
        model = _run_frozen_model_stage(
            authority=authority,
            source_review_binding=source_review_binding,
            plan=plan,
            wall_started=wall_started,
        )
        whole_wall_seconds = time.monotonic() - wall_started
        _remaining_wall_seconds(wall_started)
        evaluation = model["evaluation"]
        checkpoint = model["checkpoint"]
        result: dict[str, Any] = {
            "schema": RESULT_SCHEMA,
            "status": evaluation["status"],
            "attempt_id": plan_builder.SCIENTIFIC_ATTEMPT_ID,
            "citable_as_scientific_evidence": False,
            "development_only": True,
            "preregistration_binding": _preregistration_binding(),
            "source_review_binding": dict(source_review_binding),
            "source_review_decision": copy.deepcopy(source_review["decision"]),
            "source_bindings": _source_bindings(),
            "plan_binding": dict(plan_binding),
            "qualification_pass_decision_binding": (
                plan_builder.qualification_pass_decision_binding()
            ),
            "independent_qualification_terminal_review_binding": (
                plan_builder.independent_qualification_terminal_review_binding()
            ),
            "qualification_payload_reuse_authorized": False,
            "reservation_binding": dict(reservation_binding),
            "physics_result_binding": model["physics_result_binding"],
            "checkpoint_binding": model["checkpoint_binding"],
            "checkpoint_summary": {
                "identity_sha256": checkpoint["identity_sha256"],
                "frozen_config": checkpoint["config"],
                "arms": {
                    arm: [
                        {
                            "seed": member["seed"],
                            "initial_state_identity_sha256": member[
                                "initial_state_identity_sha256"
                            ],
                            "state_identity_sha256": member[
                                "state_identity_sha256"
                            ],
                            "updates": member["updates"],
                            "trace": member["trace"],
                            "training_seconds": member["training_seconds"],
                        }
                        for member in checkpoint["arms"][arm]
                    ]
                    for arm in benchmark.ARM_ORDER
                },
            },
            "evaluation": evaluation,
            "repeat_evaluation_exact": model["repeat_evaluation_exact"],
            "role_disjointness": model["role_disjointness"],
            "access_audit": model["access_audit"],
            "collection_summary": {
                "collection_wall_seconds": physics_result[
                    "collection_wall_seconds"
                ],
                "stored_rgb_bytes": physics_result["stored_rgb_bytes"],
                "stored_rgb_frames": scene_core.EXPECTED_CAPS[
                    "stored_rgb_frames"
                ],
                "observed_counts": physics_result["observed_counts"],
                "scenes": [
                    {
                        name: row[name]
                        for name in (
                            "role",
                            "family",
                            "scene_id",
                            "states",
                            "stored_rgb_frames",
                            "scene_total_wall_seconds",
                            "physics_simulation_wall_seconds",
                            "native_render_wall_seconds",
                        )
                    }
                    for row in physics_result["scene_metrics"]
                ],
            },
            "runtime": {
                **model["compute_runtime"],
                "determinism": model["determinism"],
                "torch": torch.__version__,
                "graphics_preflight": graphics_preflight,
                "selected_render_device": {
                    "vendor_id": vendor,
                    "device_id": device_id,
                },
                "selected_device_vram": gpu_measurement,
                "kernel_reset_audit": kernel_audit,
                "whole_run_wall_seconds": whole_wall_seconds,
                "whole_run_wall_ceiling_seconds": WALL_CEILING_SECONDS,
            },
            "successor_observations_opened": 0,
            "authorizes_navigation_claim": False,
            "authorizes_blind_rollout_preregistration": evaluation[
                "authorizes_blind_rollout_preregistration"
            ],
            "authorizes_retry_or_resume": False,
        }
        result["result_identity_sha256"] = (
            frozen_runner._result_identity_v1(result)  # noqa: SLF001
        )
        frozen_runner._write_json_exclusive(  # noqa: SLF001
            plan_builder.SCIENTIFIC_ATTEMPT_ROOT / "result.json", result
        )
        result_binding = frozen_runner.file_binding_v1(
            plan_builder.SCIENTIFIC_ATTEMPT_ROOT / "result.json"
        )
        frozen_runner._write_json_exclusive(  # noqa: SLF001
            plan_builder.SCIENTIFIC_ATTEMPT_ROOT / "terminal.json",
            {
                "schema": TERMINAL_SCHEMA,
                "status": result["status"],
                "authorizes_retry_or_resume": False,
                "authorizes_navigation_claim": False,
                "authorizes_blind_rollout_preregistration": result[
                    "authorizes_blind_rollout_preregistration"
                ],
                "result_binding": result_binding,
                "failure": None,
            },
        )
        return {
            **result,
            "result_binding": result_binding,
            "collection_root": str(collection_root),
        }
    except Exception as error:
        if gpu_measurement is None:
            try:
                gpu_measurement = gpu_sampler.stop()
            except Exception as stop_error:
                gpu_measurement = {
                    "stop_failure": _failure_receipt(stop_error)
                }
        diagnostics = dict(getattr(error, "diagnostics", {}))
        diagnostics.setdefault("selected_device_vram", gpu_measurement)
        diagnostics.setdefault(
            "whole_run_wall_seconds", time.monotonic() - wall_started
        )
        raise CpuFlatScientificError(
            str(error), diagnostics=diagnostics
        ) from error


def _write_failure_terminal(error: Exception) -> None:
    attempt = plan_builder.SCIENTIFIC_ATTEMPT_ROOT
    terminal = attempt / "terminal.json"
    if (
        not attempt.is_dir()
        or attempt.is_symlink()
        or terminal.exists()
        or terminal.is_symlink()
    ):
        return
    try:
        pilot.write_json_exclusive(
            terminal,
            {
                "schema": TERMINAL_SCHEMA,
                "status": FAIL_STATUS,
                "result_binding": None,
                "authorizes_retry_or_resume": False,
                "authorizes_navigation_claim": False,
                "authorizes_blind_rollout_preregistration": False,
                "qualification_payload_reuse_authorized": False,
                "failure": _failure_receipt(error),
            },
        )
    except Exception:
        return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--expected-plan-byte-count", type=int, required=True)
    parser.add_argument("--source-review", type=Path)
    parser.add_argument("--expected-source-review-sha256")
    parser.add_argument("--expected-source-review-byte-count", type=int)
    parser.add_argument("--worker-scene-index", type=int)
    parser.add_argument("--reservation", type=Path)
    parser.add_argument("--expected-reservation-sha256")
    parser.add_argument("--expected-reservation-byte-count", type=int)
    parser.add_argument("--orchestrator-pid", type=int)
    parser.add_argument("--orchestrator-nonce")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker_scene_index is not None:
        plan_binding: dict[str, Any] | None = None
        reservation_binding: dict[str, Any] | None = None
        try:
            if (
                args.reservation is None
                or args.expected_reservation_sha256 is None
                or args.expected_reservation_byte_count is None
                or args.orchestrator_pid is None
                or args.orchestrator_nonce is None
            ):
                raise CpuFlatScientificError(
                    "worker invocation is incomplete"
                )
            plan, plan_binding, _scenes = _read_worker_plan(
                args.plan,
                expected_sha256=args.expected_plan_sha256,
                expected_byte_count=args.expected_plan_byte_count,
            )
            reservation_binding = {
                "path": str(args.reservation.resolve(strict=True)),
                "sha256": args.expected_reservation_sha256,
                "byte_count": args.expected_reservation_byte_count,
            }
            _collect_scene_worker(
                plan=plan,
                plan_binding=plan_binding,
                reservation_binding=reservation_binding,
                scene_index=args.worker_scene_index,
                orchestrator_pid=args.orchestrator_pid,
                nonce=args.orchestrator_nonce,
            )
            return 0
        except Exception as error:
            _write_worker_failure(
                error,
                scene_index=args.worker_scene_index,
                plan_binding=plan_binding,
                reservation_binding=reservation_binding,
                orchestrator_pid=args.orchestrator_pid,
            )
            print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
            return 1
    try:
        plan, plan_binding, scenes = _read_and_validate_plan(
            args.plan,
            expected_sha256=args.expected_plan_sha256,
            expected_byte_count=args.expected_plan_byte_count,
        )
        if any(
            value is not None
            for value in (
                args.reservation,
                args.expected_reservation_sha256,
                args.expected_reservation_byte_count,
                args.orchestrator_pid,
                args.orchestrator_nonce,
            )
        ):
            raise CpuFlatScientificError(
                "parent received worker-only arguments"
            )
        if (
            args.source_review is None
            or args.expected_source_review_sha256 is None
            or args.expected_source_review_byte_count is None
        ):
            raise CpuFlatScientificError(
                "parent scientific source-review binding is incomplete"
            )
        source_review, source_review_binding = (
            _read_and_validate_source_review(
                args.source_review,
                expected_sha256=args.expected_source_review_sha256,
                expected_byte_count=args.expected_source_review_byte_count,
                plan_binding=plan_binding,
            )
        )
        result = execute_scientific(
            plan=plan,
            plan_binding=plan_binding,
            scenes=scenes,
            source_review=source_review,
            source_review_binding=source_review_binding,
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "result_binding": result["result_binding"],
                    "attempt_root": str(
                        plan_builder.SCIENTIFIC_ATTEMPT_ROOT.resolve()
                    ),
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        _write_failure_terminal(error)
        print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CpuFlatScientificError",
    "RESULT_SCHEMA",
    "SCENE_ORDER",
    "SOURCE_PATHS",
    "SOURCE_REVIEW_SCHEMA",
    "SOURCE_REVIEW_STATUS",
    "TERMINAL_SCHEMA",
    "execute_scientific",
]
