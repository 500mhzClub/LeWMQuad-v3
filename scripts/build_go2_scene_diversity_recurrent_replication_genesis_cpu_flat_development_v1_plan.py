#!/usr/bin/env python3
"""Build the fresh flat Genesis/CPU V1 qualification plan only.

The reviewed CPU V1 plans are immutable scientific witnesses.  This builder
copies the exact qualification payload, changes only its fresh identity/root,
and adds an explicit flat-harness qualification contract.  It deliberately
has no science-plan output: a successor science plan may be released only
after an independently validated qualification PASS.

This source emits metadata only and grants no execution authority.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as pilot  # noqa: E402


FROZEN_CPU_SCIENTIFIC_PLAN = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "cpu_backend_v1_exact_plan_2026-08-04.json"
)
FROZEN_CPU_SCIENTIFIC_PLAN_SHA256 = (
    "258d6bf004fa3618d492b583c56ea7fbc15b127ade36299fcba11295b147745e"
)
FROZEN_CPU_SCIENTIFIC_PLAN_BYTE_COUNT = 346_045
FROZEN_CPU_SCIENTIFIC_ATTEMPT_ID = (
    "go2-scene-diversity-recurrent-replication-cpu-backend-v1"
)
FROZEN_CPU_SCIENTIFIC_OUTPUT_ROOT = REPO_ROOT / (
    ".generated/dev/go2_scene_diversity_recurrent_replication_"
    "cpu_backend_v1/attempt_v1/collection"
)

FROZEN_CPU_QUALIFICATION_PLAN = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "cpu_backend_v1_qualification_exact_plan_2026-08-04.json"
)
FROZEN_CPU_QUALIFICATION_PLAN_SHA256 = (
    "8345612973b70cffda80395ae835bc2738a8aa5c64b8cce9f0becbea31c3fe5d"
)
FROZEN_CPU_QUALIFICATION_PLAN_BYTE_COUNT = 346_073
FROZEN_CPU_QUALIFICATION_ATTEMPT_ID = (
    "go2-scene-diversity-recurrent-replication-cpu-backend-v1-qualification"
)
FROZEN_CPU_QUALIFICATION_OUTPUT_ROOT = REPO_ROOT / (
    ".generated/dev/go2_scene_diversity_recurrent_replication_"
    "cpu_backend_v1_qualification/attempt_v1/collection"
)

FROZEN_CPU_SOURCE_REVIEW = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "cpu_backend_v1_source_review_2026-08-04.json"
)
FROZEN_CPU_SOURCE_REVIEW_SHA256 = (
    "1262934840b6b1077fb53d99ff7c4ca74e717ebe893e75e898aa19d56b8fc82c"
)
FROZEN_CPU_SOURCE_REVIEW_BYTE_COUNT = 78_392

FROZEN_CPU_TERMINAL_REVIEW = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "cpu_backend_v1_qualification_terminal_review_2026-08-04.json"
)
FROZEN_CPU_TERMINAL_REVIEW_SHA256 = (
    "9b0c31c05b4fb6064c67116a456d34a6f7e49cfe85ec55ed081599acb18502f0"
)
FROZEN_CPU_TERMINAL_REVIEW_BYTE_COUNT = 20_536

FLAT_ROCM_QUALIFICATION_PLAN = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_genesis_rocm_"
    "flat_development_v1_qualification_exact_plan_2026-08-05.json"
)
FLAT_ROCM_QUALIFICATION_PLAN_SHA256 = (
    "87a400d8d2688ed58c1a0dd61e4121dfa35374381d10d00b44738d544e3853b2"
)
FLAT_ROCM_QUALIFICATION_PLAN_BYTE_COUNT = 367_782

FLAT_ROCM_SOURCE_REVIEW = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_genesis_rocm_"
    "flat_development_v1_qualification_source_review_2026-08-05.json"
)
FLAT_ROCM_SOURCE_REVIEW_SHA256 = (
    "91ce59e88f464e38470fde85731c8b04043752d57fd6de24bdb48b155c8ea84f"
)
FLAT_ROCM_SOURCE_REVIEW_BYTE_COUNT = 8_837

FLAT_ROCM_TERMINAL_REVIEW = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_genesis_rocm_"
    "flat_development_v1_qualification_terminal_review_2026-08-05.json"
)
FLAT_ROCM_TERMINAL_REVIEW_SHA256 = (
    "dffcc637258d5b17b1239b603b72e9a376d421a9783c2b89e1e8691db11dc206"
)
FLAT_ROCM_TERMINAL_REVIEW_BYTE_COUNT = 8_519

QUALIFICATION_ATTEMPT_ID = (
    "go2-scene-diversity-recurrent-replication-genesis-cpu-flat-"
    "development-v1-qualification"
)
QUALIFICATION_ATTEMPT_ROOT = REPO_ROOT / (
    ".generated/dev/go2_scene_diversity_recurrent_replication_"
    "genesis_cpu_flat_development_v1_qualification/attempt_v1"
)
QUALIFICATION_OUTPUT_ROOT = QUALIFICATION_ATTEMPT_ROOT / "collection"
QUALIFICATION_PLAN_OUTPUT = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_"
    "flat_development_v1_qualification_exact_plan_2026-08-05.json"
)

DEFERRED_SCIENTIFIC_ATTEMPT_ID = (
    "go2-scene-diversity-recurrent-replication-genesis-cpu-flat-development-v1"
)
DEFERRED_SCIENTIFIC_ATTEMPT_ROOT = REPO_ROOT / (
    ".generated/dev/go2_scene_diversity_recurrent_replication_"
    "genesis_cpu_flat_development_v1/attempt_v1"
)

CPU_EXECUTION_ENVIRONMENT = {
    "EGL_DEVICE_ID": "1",
    "GS_BACKEND": "cpu",
    "GS_PARA_LEVEL": "0",
    "MESA_VK_DEVICE_SELECT": "1002:7551!",
    "PYOPENGL_PLATFORM": "egl",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "PYTHONSAFEPATH": "1",
}
QUALIFICATION_SCENE_INDICES = (12, 0)
QUALIFICATION_WORKER_WATCHDOG_SECONDS = 300
QUALIFICATION_FIXED_NONCOLLECTION_RESERVE_SECONDS = 900
SCIENTIFIC_SCENE_COUNT = 64
SCIENTIFIC_WALL_CAP_SECONDS = 7_200
SELECTED_DEVICE_VRAM_CEILING_BYTES = 16_977_405_952
SUCCESSOR_CONTRACT_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_"
    "development_v1_successor_contract_v1"
)


class CpuFlatDevelopmentPlanError(RuntimeError):
    """Raised before a changed or non-fresh qualification plan is emitted."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _standard_binding(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
    }


def _read_bound_json(
    path: Path, *, sha256: str, byte_count: int, label: str
) -> dict[str, Any]:
    expected = {
        "path": str(path.resolve(strict=True)),
        "sha256": sha256,
        "byte_count": byte_count,
    }
    if _standard_binding(path) != expected:
        raise CpuFlatDevelopmentPlanError(f"{label} binding changed")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CpuFlatDevelopmentPlanError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise CpuFlatDevelopmentPlanError(f"{label} must be an object")
    return value


def _binding(
    path: Path, *, sha256: str, byte_count: int, label: str
) -> dict[str, object]:
    _read_bound_json(path, sha256=sha256, byte_count=byte_count, label=label)
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": sha256,
        "byte_count": byte_count,
    }


def validate_reviewed_cpu_delta() -> dict[str, Any]:
    """Prove that both bound CPU plans carry one unchanged 64-scene design."""

    science = _read_bound_json(
        FROZEN_CPU_SCIENTIFIC_PLAN,
        sha256=FROZEN_CPU_SCIENTIFIC_PLAN_SHA256,
        byte_count=FROZEN_CPU_SCIENTIFIC_PLAN_BYTE_COUNT,
        label="frozen CPU scientific plan",
    )
    qualification = _read_bound_json(
        FROZEN_CPU_QUALIFICATION_PLAN,
        sha256=FROZEN_CPU_QUALIFICATION_PLAN_SHA256,
        byte_count=FROZEN_CPU_QUALIFICATION_PLAN_BYTE_COUNT,
        label="frozen CPU qualification plan",
    )
    if (
        science.get("attempt_id") != FROZEN_CPU_SCIENTIFIC_ATTEMPT_ID
        or science.get("output_root")
        != str(FROZEN_CPU_SCIENTIFIC_OUTPUT_ROOT.resolve(strict=False))
        or qualification.get("attempt_id")
        != FROZEN_CPU_QUALIFICATION_ATTEMPT_ID
        or qualification.get("output_root")
        != str(FROZEN_CPU_QUALIFICATION_OUTPUT_ROOT.resolve(strict=False))
    ):
        raise CpuFlatDevelopmentPlanError("frozen CPU plan identity changed")

    for label, plan in (("scientific", science), ("qualification", qualification)):
        execution = plan.get("execution_contract")
        if (
            not isinstance(execution, dict)
            or execution.get("backend") != "cpu"
            or execution.get("environment") != CPU_EXECUTION_ENVIRONMENT
            or plan.get("branch_mechanism")
            != "parallel_lockstep_envs_no_restore"
            or plan.get("successor_contract") is not None
        ):
            raise CpuFlatDevelopmentPlanError(
                f"frozen CPU {label} backend or branch mechanism changed"
            )

    normalized = copy.deepcopy(qualification)
    normalized["attempt_id"] = FROZEN_CPU_SCIENTIFIC_ATTEMPT_ID
    normalized["output_root"] = str(
        FROZEN_CPU_SCIENTIFIC_OUTPUT_ROOT.resolve(strict=False)
    )
    if _canonical(normalized) != _canonical(science):
        raise CpuFlatDevelopmentPlanError(
            "frozen CPU qualification changed beyond its identity/root"
        )

    review = _read_bound_json(
        FROZEN_CPU_SOURCE_REVIEW,
        sha256=FROZEN_CPU_SOURCE_REVIEW_SHA256,
        byte_count=FROZEN_CPU_SOURCE_REVIEW_BYTE_COUNT,
        label="frozen CPU source review",
    )
    audit = review.get("cpu_backend_source_audit", {})
    expected_cpu_review_schema = (
        "lewm_go2_scene_diversity_recurrent_replication_cpu_backend_v1_"
        "source_review_v1"
    )
    if (
        review.get("schema") != expected_cpu_review_schema
        or review.get("status") != "PASS_INDEPENDENT_SOURCE_REVIEW"
        or review.get("audit_passed") is not True
        or audit.get(
            "scientific_plan_differences_exactly_attempt_id_output_root_backend_and_gs_backend"
        )
        is not True
        or audit.get(
            "data_panel_model_arms_seeds_updates_evaluation_and_gates_unchanged"
        )
        is not True
        or audit.get("one_scene_fresh_process_policy_unchanged_for_science")
        is not True
    ):
        raise CpuFlatDevelopmentPlanError(
            "frozen CPU review does not establish the exact CPU delta"
        )

    flat_review = _read_bound_json(
        FLAT_ROCM_SOURCE_REVIEW,
        sha256=FLAT_ROCM_SOURCE_REVIEW_SHA256,
        byte_count=FLAT_ROCM_SOURCE_REVIEW_BYTE_COUNT,
        label="flat ROCm source review",
    )
    source_ast = flat_review.get("independent_verification", {}).get(
        "source_ast", {}
    )
    if (
        flat_review.get("status")
        != "PASS_INDEPENDENT_QUALIFICATION_SOURCE_REVIEW"
        or source_ast.get("configured_adapter_context_calls") != 0
        or source_ast.get("imported_project_module_attribute_writes") != 0
        or source_ast.get("shared_pilot_backend_or_environment_mutations") != 0
    ):
        raise CpuFlatDevelopmentPlanError(
            "flat harness review no longer establishes non-monkeypatched custody"
        )

    terminal_review = _read_bound_json(
        FLAT_ROCM_TERMINAL_REVIEW,
        sha256=FLAT_ROCM_TERMINAL_REVIEW_SHA256,
        byte_count=FLAT_ROCM_TERMINAL_REVIEW_BYTE_COUNT,
        label="flat ROCm terminal review",
    )
    disposition = terminal_review.get("scientific_disposition", {})
    next_experiment = terminal_review.get("next_experiment", {})
    if (
        terminal_review.get("schema")
        != "lewm_go2_scene_diversity_recurrent_replication_genesis_rocm_"
        "flat_development_v1_qualification_terminal_review_v1"
        or terminal_review.get("status")
        != "PASS_FAIL_CLOSED_POST_GENESIS_FIRST_STEP_DIVERGENCE_TERMINAL_REVIEW"
        or disposition.get("qualification_passed") is not False
        or disposition.get("flat_rocm_v1_retired") is not True
        or disposition.get("no_tolerance_relaxation") is not True
        or next_experiment.get("fixed_probe_order")
        != list(QUALIFICATION_SCENE_INDICES)
        or next_experiment.get("authorizes_cpu_execution_by_itself") is not False
        or "fresh flat Genesis CPU qualification"
        not in str(next_experiment.get("recommendation", ""))
    ):
        raise CpuFlatDevelopmentPlanError(
            "flat ROCm terminal review no longer supports this narrow study"
        )
    return qualification


def _successor_contract() -> dict[str, Any]:
    return {
        "schema": SUCCESSOR_CONTRACT_SCHEMA,
        "plan_role": "qualification",
        "development_only": True,
        "promotable": False,
        "flat_harness_owned": True,
        "module_global_overlay_contexts_authorized": False,
        "imported_project_module_attribute_mutation_authorized": False,
        "backend_contract_passed_explicitly": True,
        "genesis_backend_symbol": "gs.cpu",
        "reviewed_cpu_delta_unchanged": True,
        "cpu_physics_with_bound_egl_r9700_rendering": True,
        "frozen_cpu_scientific_plan_binding": _binding(
            FROZEN_CPU_SCIENTIFIC_PLAN,
            sha256=FROZEN_CPU_SCIENTIFIC_PLAN_SHA256,
            byte_count=FROZEN_CPU_SCIENTIFIC_PLAN_BYTE_COUNT,
            label="frozen CPU scientific plan",
        ),
        "frozen_cpu_qualification_plan_binding": _binding(
            FROZEN_CPU_QUALIFICATION_PLAN,
            sha256=FROZEN_CPU_QUALIFICATION_PLAN_SHA256,
            byte_count=FROZEN_CPU_QUALIFICATION_PLAN_BYTE_COUNT,
            label="frozen CPU qualification plan",
        ),
        "frozen_cpu_source_review_binding": _binding(
            FROZEN_CPU_SOURCE_REVIEW,
            sha256=FROZEN_CPU_SOURCE_REVIEW_SHA256,
            byte_count=FROZEN_CPU_SOURCE_REVIEW_BYTE_COUNT,
            label="frozen CPU source review",
        ),
        "consumed_cpu_v1_terminal_review_binding": _binding(
            FROZEN_CPU_TERMINAL_REVIEW,
            sha256=FROZEN_CPU_TERMINAL_REVIEW_SHA256,
            byte_count=FROZEN_CPU_TERMINAL_REVIEW_BYTE_COUNT,
            label="consumed CPU V1 terminal review",
        ),
        "flat_rocm_qualification_plan_binding": _binding(
            FLAT_ROCM_QUALIFICATION_PLAN,
            sha256=FLAT_ROCM_QUALIFICATION_PLAN_SHA256,
            byte_count=FLAT_ROCM_QUALIFICATION_PLAN_BYTE_COUNT,
            label="flat ROCm qualification plan",
        ),
        "flat_rocm_source_review_binding": _binding(
            FLAT_ROCM_SOURCE_REVIEW,
            sha256=FLAT_ROCM_SOURCE_REVIEW_SHA256,
            byte_count=FLAT_ROCM_SOURCE_REVIEW_BYTE_COUNT,
            label="flat ROCm source review",
        ),
        "flat_rocm_terminal_review_binding": _binding(
            FLAT_ROCM_TERMINAL_REVIEW,
            sha256=FLAT_ROCM_TERMINAL_REVIEW_SHA256,
            byte_count=FLAT_ROCM_TERMINAL_REVIEW_BYTE_COUNT,
            label="flat ROCm terminal review",
        ),
        "historical_authority_reservation_or_runtime_reuse_authorized": False,
        "qualification_scene_indices_in_order": list(
            QUALIFICATION_SCENE_INDICES
        ),
        "qualification_fresh_process_groups": 2,
        "states_per_worker": 4,
        "candidate_actions_per_state": 9,
        "branches_per_worker": 36,
        "branch_mechanism": "parallel_lockstep_envs_no_restore",
        "qualification_worker_watchdog_seconds": (
            QUALIFICATION_WORKER_WATCHDOG_SECONDS
        ),
        "selected_device_vram_ceiling_bytes": (
            SELECTED_DEVICE_VRAM_CEILING_BYTES
        ),
        "matching_prelaunch_vram_release_barrier_required": True,
        "release_barrier_before_result_open_or_next_worker": True,
        "amdgpu_kernel_journal_gate_required": True,
        "exact_nine_lane_state_group_equality_required": True,
        "numerical_tolerance_relaxation_authorized": False,
        "complete_first_step_and_checkpoint_sync_diagnostics_required": True,
        "qualification_timing_gate": {
            "formula": "64 * max(worker_elapsed_seconds) + 900 <= 7200",
            "scene_count": SCIENTIFIC_SCENE_COUNT,
            "fixed_noncollection_reserve_seconds": (
                QUALIFICATION_FIXED_NONCOLLECTION_RESERVE_SECONDS
            ),
            "scientific_wall_cap_seconds": SCIENTIFIC_WALL_CAP_SECONDS,
        },
        "probe_output_reuse_authorized": False,
        "qualification_result_is_scientific_evidence": False,
        "qualification_execution_authorized": False,
        "scientific_plan_created": False,
        "scientific_plan_release_requires_exact_pass_decision": True,
        "scientific_execution_authorized": False,
        "deferred_scientific_attempt_id": DEFERRED_SCIENTIFIC_ATTEMPT_ID,
        "deferred_scientific_attempt_root": str(
            DEFERRED_SCIENTIFIC_ATTEMPT_ROOT.resolve(strict=False)
        ),
        "three_arm_science_data_model_seeds_updates_and_gates_unchanged": True,
    }


def _expected_qualification_plan() -> dict[str, Any]:
    candidate = copy.deepcopy(validate_reviewed_cpu_delta())
    candidate["attempt_id"] = QUALIFICATION_ATTEMPT_ID
    candidate["output_root"] = str(
        QUALIFICATION_OUTPUT_ROOT.resolve(strict=False)
    )
    candidate["successor_contract"] = _successor_contract()
    return candidate


def validate_qualification_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(plan, Mapping):
        raise CpuFlatDevelopmentPlanError("qualification plan must be an object")
    candidate = copy.deepcopy(dict(plan))
    expected = _expected_qualification_plan()
    if _canonical(candidate) != _canonical(expected):
        raise CpuFlatDevelopmentPlanError(
            "CPU-flat qualification changed beyond identity/root/flat metadata"
        )
    if candidate.get("branch_mechanism") != "parallel_lockstep_envs_no_restore":
        raise CpuFlatDevelopmentPlanError("36-lane branch mechanism changed")
    return candidate


def build_qualification_plan() -> dict[str, Any]:
    if (
        QUALIFICATION_ATTEMPT_ROOT.exists()
        or QUALIFICATION_ATTEMPT_ROOT.is_symlink()
        or QUALIFICATION_OUTPUT_ROOT.exists()
        or QUALIFICATION_OUTPUT_ROOT.is_symlink()
        or DEFERRED_SCIENTIFIC_ATTEMPT_ROOT.exists()
        or DEFERRED_SCIENTIFIC_ATTEMPT_ROOT.is_symlink()
    ):
        raise CpuFlatDevelopmentPlanError(
            "fresh CPU-flat qualification/scientific roots changed"
        )
    return validate_qualification_plan(_expected_qualification_plan())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qualification-plan-output",
        type=Path,
        default=QUALIFICATION_PLAN_OUTPUT,
    )
    args = parser.parse_args(argv)
    if (
        args.qualification_plan_output.exists()
        or args.qualification_plan_output.is_symlink()
    ):
        raise CpuFlatDevelopmentPlanError(
            "CPU-flat qualification plan output must be fresh"
        )
    qualification = build_qualification_plan()
    qualification_binding = pilot.write_json_exclusive(
        args.qualification_plan_output, qualification
    )
    print(
        json.dumps(
            {
                "qualification_plan": qualification_binding,
                "scientific_plan": None,
                "scientific_plan_release_deferred_until_qualification_pass": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CPU_EXECUTION_ENVIRONMENT",
    "CpuFlatDevelopmentPlanError",
    "DEFERRED_SCIENTIFIC_ATTEMPT_ID",
    "DEFERRED_SCIENTIFIC_ATTEMPT_ROOT",
    "QUALIFICATION_ATTEMPT_ID",
    "QUALIFICATION_ATTEMPT_ROOT",
    "QUALIFICATION_OUTPUT_ROOT",
    "QUALIFICATION_PLAN_OUTPUT",
    "QUALIFICATION_SCENE_INDICES",
    "QUALIFICATION_WORKER_WATCHDOG_SECONDS",
    "SELECTED_DEVICE_VRAM_CEILING_BYTES",
    "build_qualification_plan",
    "validate_qualification_plan",
    "validate_reviewed_cpu_delta",
]
