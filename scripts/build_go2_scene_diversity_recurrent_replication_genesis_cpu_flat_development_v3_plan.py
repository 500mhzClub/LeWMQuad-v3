#!/usr/bin/env python3
"""Build the fresh CPU-flat V3 qualification plan only.

V3 preserves the exact consumed V2 qualification payload and changes only
fresh identity/root, successor-contract metadata, and the addition of fixed
PATH=/usr/bin:/bin to the otherwise unchanged ten-key child environment.  It
has no science-plan output and grants no execution authority.
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
from scripts import build_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_development_v2_plan as v2  # noqa: E402


V2_QUALIFICATION_PLAN = v2.QUALIFICATION_PLAN_OUTPUT
V2_QUALIFICATION_PLAN_SHA256 = (
    "df4516631a646b46ee89ca9be7aa55b47bf5927a9440a26c4c4fee19e0147ce0"
)
V2_QUALIFICATION_PLAN_BYTE_COUNT = 353_031

V2_PREREGISTRATION = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_"
    "development_v2_preregistration_2026-08-05.md"
)
V2_PREREGISTRATION_SHA256 = (
    "ff50441e8eca4e1b05391dfd22137532e42a8c0c6d41626a7332323325350f51"
)
V2_PREREGISTRATION_BYTE_COUNT = 9_087

V2_SOURCE_REVIEW = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_"
    "development_v2_qualification_source_review_2026-08-05.json"
)
V2_SOURCE_REVIEW_SHA256 = (
    "4d69718a82f9a4a87087e76c6353eda8fabaedb91303c0b498d85ac76e136aef"
)
V2_SOURCE_REVIEW_BYTE_COUNT = 14_170

V2_PLAN_BUILDER = REPO_ROOT / (
    "scripts/build_go2_scene_diversity_recurrent_replication_genesis_cpu_"
    "flat_development_v2_plan.py"
)
V2_PLAN_BUILDER_SHA256 = (
    "dcc3fc794001ff70b894bc62fb83a6958d2bbefc2b5fed04629338a822e41cc0"
)
V2_PLAN_BUILDER_BYTE_COUNT = 16_225

V2_QUALIFICATION_HARNESS = REPO_ROOT / (
    "scripts/run_go2_scene_diversity_recurrent_replication_genesis_cpu_"
    "flat_development_v2.py"
)
V2_QUALIFICATION_HARNESS_SHA256 = (
    "8dc0606af0ef4f0df0dcd04160afbf5aed845031e4e36cfddd8589c2a0141703"
)
V2_QUALIFICATION_HARNESS_BYTE_COUNT = 71_736

V2_TERMINAL_REVIEW = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_"
    "development_v2_qualification_terminal_review_2026-08-05.json"
)
V2_TERMINAL_REVIEW_SHA256 = (
    "60f18cad6ced0f18fd6e4884439fd688d79bef3deef0a62d41ac66f258dead1d"
)
V2_TERMINAL_REVIEW_BYTE_COUNT = 12_883

QUALIFICATION_ATTEMPT_ID = (
    "go2-scene-diversity-recurrent-replication-genesis-cpu-flat-"
    "development-v3-qualification"
)
QUALIFICATION_ATTEMPT_ROOT = REPO_ROOT / (
    ".generated/dev/go2_scene_diversity_recurrent_replication_genesis_cpu_"
    "flat_development_v3_qualification/attempt_v1"
)
QUALIFICATION_OUTPUT_ROOT = QUALIFICATION_ATTEMPT_ROOT / "collection"
QUALIFICATION_PLAN_OUTPUT = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_"
    "development_v3_qualification_exact_plan_2026-08-05.json"
)

DEFERRED_SCIENTIFIC_ATTEMPT_ID = (
    "go2-scene-diversity-recurrent-replication-genesis-cpu-flat-development-v3"
)
DEFERRED_SCIENTIFIC_ATTEMPT_ROOT = REPO_ROOT / (
    ".generated/dev/go2_scene_diversity_recurrent_replication_genesis_cpu_"
    "flat_development_v3/attempt_v1"
)

FIXED_HOME = v2.FIXED_HOME
FIXED_PATH = "/usr/bin:/bin"
CPU_EXECUTION_ENVIRONMENT = {
    **v2.CPU_EXECUTION_ENVIRONMENT,
    "PATH": FIXED_PATH,
}
QUALIFICATION_SCENE_INDICES = tuple(v2.QUALIFICATION_SCENE_INDICES)
QUALIFICATION_WORKER_WATCHDOG_SECONDS = (
    v2.QUALIFICATION_WORKER_WATCHDOG_SECONDS
)
QUALIFICATION_FIXED_NONCOLLECTION_RESERVE_SECONDS = (
    v2.QUALIFICATION_FIXED_NONCOLLECTION_RESERVE_SECONDS
)
SELECTED_DEVICE_VRAM_CEILING_BYTES = v2.SELECTED_DEVICE_VRAM_CEILING_BYTES
SUCCESSOR_CONTRACT_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_"
    "development_v3_successor_contract_v1"
)


class CpuFlatDevelopmentV3PlanError(RuntimeError):
    """Raised before a changed or non-fresh V3 plan can be emitted."""


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
    try:
        observed = _standard_binding(path)
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CpuFlatDevelopmentV3PlanError(
            f"{label} is absent or not strict JSON"
        ) from exc
    if observed != expected:
        raise CpuFlatDevelopmentV3PlanError(f"{label} binding changed")
    if not isinstance(value, dict):
        raise CpuFlatDevelopmentV3PlanError(f"{label} must be an object")
    return value


def _binding(
    path: Path, *, sha256: str, byte_count: int, label: str
) -> dict[str, object]:
    expected = {
        "path": str(path.resolve(strict=True)),
        "sha256": sha256,
        "byte_count": byte_count,
    }
    try:
        observed = _standard_binding(path)
    except OSError as exc:
        raise CpuFlatDevelopmentV3PlanError(f"{label} is absent") from exc
    if observed != expected:
        raise CpuFlatDevelopmentV3PlanError(f"{label} binding changed")
    return expected


def predecessor_terminal_review_binding() -> dict[str, object]:
    review = _read_bound_json(
        V2_TERMINAL_REVIEW,
        sha256=V2_TERMINAL_REVIEW_SHA256,
        byte_count=V2_TERMINAL_REVIEW_BYTE_COUNT,
        label="V2 terminal review",
    )
    disposition = review.get("scientific_disposition", {})
    successor = review.get("next_experiment", {})
    expected_schema = (
        "lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_"
        "development_v2_qualification_terminal_review_v1"
    )
    if (
        review.get("schema") != expected_schema
        or review.get("status")
        != "PASS_FAIL_CLOSED_PRE_PHYSICS_PATH_ENVIRONMENT_STOP_ITERATION_TERMINAL_REVIEW"
        or disposition.get("qualification_passed") is not False
        or disposition.get("v2_consumed") is not True
        or disposition.get("v2_retired") is not True
        or disposition.get("v2_output_reuse_authorized") is not False
        or successor.get("sole_material_environment_delta")
        != {"PATH": FIXED_PATH}
        or successor.get("required_retained_environment_entry")
        != {"HOME": FIXED_HOME}
        or successor.get("expected_child_environment_key_count") != 11
        or successor.get("ambient_or_inherited_path_authorized") is not False
        or successor.get("path_value_other_than_fixed_usr_bin_bin_authorized")
        is not False
        or successor.get("preserve_cpu_backend") is not True
        or successor.get("preserve_vulkan_egl_r9700_render_route") is not True
        or successor.get("preserve_probe_order")
        != list(QUALIFICATION_SCENE_INDICES)
        or successor.get("preserve_four_states_and_36_lanes_per_worker")
        is not True
        or successor.get(
            "preserve_histories_actions_horizons_render_contract_model_seeds_gates_and_thresholds"
        )
        is not True
        or successor.get("fresh_attempt_identity_and_root_required") is not True
        or successor.get(
            "separate_preregistration_plan_builder_harness_tests_and_independent_source_review_required"
        )
        is not True
        or successor.get("v2_runtime_output_or_cache_reuse_authorized") is not False
        or successor.get("authorizes_v3_execution_by_itself") is not False
        or successor.get(
            "alternate_environment_architecture_tolerance_or_batching_change_recommended"
        )
        is not False
    ):
        raise CpuFlatDevelopmentV3PlanError(
            "V2 terminal review does not close V2 and scope the PATH successor"
        )
    return {
        "path": str(V2_TERMINAL_REVIEW.resolve(strict=True)),
        "sha256": V2_TERMINAL_REVIEW_SHA256,
        "byte_count": V2_TERMINAL_REVIEW_BYTE_COUNT,
    }


def _read_v2_plan() -> dict[str, Any]:
    value = _read_bound_json(
        V2_QUALIFICATION_PLAN,
        sha256=V2_QUALIFICATION_PLAN_SHA256,
        byte_count=V2_QUALIFICATION_PLAN_BYTE_COUNT,
        label="V2 qualification plan",
    )
    try:
        validated = v2.validate_qualification_plan(value)
    except v2.CpuFlatDevelopmentV2PlanError as exc:
        raise CpuFlatDevelopmentV3PlanError(str(exc)) from exc
    if (
        validated["execution_contract"]["environment"]
        != v2.CPU_EXECUTION_ENVIRONMENT
    ):
        raise CpuFlatDevelopmentV3PlanError("V2 child environment changed")
    if "PATH" in validated["execution_contract"]["environment"]:
        raise CpuFlatDevelopmentV3PlanError("V2 unexpectedly contains PATH")
    return validated


def _successor_contract(v2_plan: Mapping[str, Any]) -> dict[str, Any]:
    contract = copy.deepcopy(dict(v2_plan["successor_contract"]))
    contract.update(
        {
            "schema": SUCCESSOR_CONTRACT_SCHEMA,
            "plan_role": "qualification",
            "flat_harness_owned": True,
            "module_global_overlay_contexts_authorized": False,
            "imported_project_module_attribute_mutation_authorized": False,
            "v2_qualification_plan_binding": _binding(
                V2_QUALIFICATION_PLAN,
                sha256=V2_QUALIFICATION_PLAN_SHA256,
                byte_count=V2_QUALIFICATION_PLAN_BYTE_COUNT,
                label="V2 qualification plan",
            ),
            "v2_preregistration_binding": _binding(
                V2_PREREGISTRATION,
                sha256=V2_PREREGISTRATION_SHA256,
                byte_count=V2_PREREGISTRATION_BYTE_COUNT,
                label="V2 preregistration",
            ),
            "v2_source_review_binding": _binding(
                V2_SOURCE_REVIEW,
                sha256=V2_SOURCE_REVIEW_SHA256,
                byte_count=V2_SOURCE_REVIEW_BYTE_COUNT,
                label="V2 source review",
            ),
            "v2_plan_builder_binding": _binding(
                V2_PLAN_BUILDER,
                sha256=V2_PLAN_BUILDER_SHA256,
                byte_count=V2_PLAN_BUILDER_BYTE_COUNT,
                label="V2 plan builder",
            ),
            "v2_qualification_harness_binding": _binding(
                V2_QUALIFICATION_HARNESS,
                sha256=V2_QUALIFICATION_HARNESS_SHA256,
                byte_count=V2_QUALIFICATION_HARNESS_BYTE_COUNT,
                label="V2 qualification harness",
            ),
            "v2_terminal_review_binding": predecessor_terminal_review_binding(),
            "v2_authority_reservation_runtime_or_payload_reuse_authorized": False,
            "v3_material_delta": "add_exact_fixed_PATH_to_child_environment",
            "child_environment_key_count": 11,
            "fixed_child_home": FIXED_HOME,
            "fixed_child_path": FIXED_PATH,
            "all_other_child_environment_keys_unchanged": True,
            "qualification_scene_indices_in_order": list(
                QUALIFICATION_SCENE_INDICES
            ),
            "qualification_worker_watchdog_seconds": (
                QUALIFICATION_WORKER_WATCHDOG_SECONDS
            ),
            "selected_device_vram_ceiling_bytes": (
                SELECTED_DEVICE_VRAM_CEILING_BYTES
            ),
            "probe_output_reuse_authorized": False,
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
    )
    return contract


def _expected_qualification_plan() -> dict[str, Any]:
    predecessor = _read_v2_plan()
    candidate = copy.deepcopy(predecessor)
    candidate["attempt_id"] = QUALIFICATION_ATTEMPT_ID
    candidate["output_root"] = str(
        QUALIFICATION_OUTPUT_ROOT.resolve(strict=False)
    )
    candidate["execution_contract"]["environment"] = copy.deepcopy(
        CPU_EXECUTION_ENVIRONMENT
    )
    candidate["successor_contract"] = _successor_contract(predecessor)
    return candidate


def validate_qualification_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(plan, Mapping):
        raise CpuFlatDevelopmentV3PlanError(
            "V3 qualification plan must be an object"
        )
    candidate = copy.deepcopy(dict(plan))
    expected = _expected_qualification_plan()
    if _canonical(candidate) != _canonical(expected):
        raise CpuFlatDevelopmentV3PlanError(
            "V3 changed beyond fresh identity/root/contracts and fixed PATH"
        )

    predecessor = _read_v2_plan()
    normalized = copy.deepcopy(candidate)
    normalized["attempt_id"] = predecessor["attempt_id"]
    normalized["output_root"] = predecessor["output_root"]
    normalized["execution_contract"]["environment"].pop("PATH", None)
    normalized["successor_contract"] = copy.deepcopy(
        predecessor["successor_contract"]
    )
    if _canonical(normalized) != _canonical(predecessor):
        raise CpuFlatDevelopmentV3PlanError(
            "V3 normalization does not restore the exact V2 plan"
        )
    environment = candidate["execution_contract"]["environment"]
    if (
        len(environment) != 11
        or environment.get("HOME") != FIXED_HOME
        or environment.get("PATH") != FIXED_PATH
    ):
        raise CpuFlatDevelopmentV3PlanError("exact eleven-key environment changed")
    return candidate


def _require_fresh_roots() -> None:
    for path in (
        QUALIFICATION_ATTEMPT_ROOT,
        QUALIFICATION_OUTPUT_ROOT,
        DEFERRED_SCIENTIFIC_ATTEMPT_ROOT,
    ):
        if path.exists() or path.is_symlink():
            raise CpuFlatDevelopmentV3PlanError(
                "fresh V3 qualification/scientific roots changed"
            )


def build_qualification_plan() -> dict[str, Any]:
    _require_fresh_roots()
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
        raise CpuFlatDevelopmentV3PlanError(
            "V3 qualification plan output must be fresh"
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
    "CpuFlatDevelopmentV3PlanError",
    "DEFERRED_SCIENTIFIC_ATTEMPT_ID",
    "DEFERRED_SCIENTIFIC_ATTEMPT_ROOT",
    "FIXED_HOME",
    "FIXED_PATH",
    "QUALIFICATION_ATTEMPT_ID",
    "QUALIFICATION_ATTEMPT_ROOT",
    "QUALIFICATION_FIXED_NONCOLLECTION_RESERVE_SECONDS",
    "QUALIFICATION_OUTPUT_ROOT",
    "QUALIFICATION_PLAN_OUTPUT",
    "QUALIFICATION_SCENE_INDICES",
    "QUALIFICATION_WORKER_WATCHDOG_SECONDS",
    "SELECTED_DEVICE_VRAM_CEILING_BYTES",
    "SUCCESSOR_CONTRACT_SCHEMA",
    "build_qualification_plan",
    "predecessor_terminal_review_binding",
    "validate_qualification_plan",
]
