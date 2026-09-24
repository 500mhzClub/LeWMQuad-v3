#!/usr/bin/env python3
"""Build the fresh CPU-flat V2 qualification plan only.

V2 is an infrastructure-only successor to consumed CPU-flat V1.  It preserves
the exact V1 qualification payload and changes only fresh identity/root,
successor-contract metadata, and the addition of the fixed host HOME to the
otherwise unchanged child environment.  It has no science-plan output.

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
from scripts import build_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_development_v1_plan as v1  # noqa: E402


V1_QUALIFICATION_PLAN = v1.QUALIFICATION_PLAN_OUTPUT
V1_QUALIFICATION_PLAN_SHA256 = (
    "397c1932f69bcbf29239fd6b86188f177222f9e350e97775cbe2968e97e868a9"
)
V1_QUALIFICATION_PLAN_BYTE_COUNT = 350_749

V1_PREREGISTRATION = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_"
    "development_v1_preregistration_2026-08-05.md"
)
V1_PREREGISTRATION_SHA256 = (
    "95d494f4487d028014a59066400d2707be6328411f79b3de781fac8bb7a0a00f"
)
V1_PREREGISTRATION_BYTE_COUNT = 10_038

V1_SOURCE_REVIEW = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_"
    "development_v1_qualification_source_review_2026-08-05.json"
)
V1_SOURCE_REVIEW_SHA256 = (
    "da21af1a2daae879ade39260312a17ded480e94671cd1198d07816028f3a87da"
)
V1_SOURCE_REVIEW_BYTE_COUNT = 15_684

V1_PLAN_BUILDER = REPO_ROOT / (
    "scripts/build_go2_scene_diversity_recurrent_replication_genesis_cpu_"
    "flat_development_v1_plan.py"
)
V1_PLAN_BUILDER_SHA256 = (
    "a2be9e81edabb201aae98681ff6b04264dad9e62f753ebff5743d060be1ec71a"
)
V1_PLAN_BUILDER_BYTE_COUNT = 19_737

V1_QUALIFICATION_HARNESS = REPO_ROOT / (
    "scripts/run_go2_scene_diversity_recurrent_replication_genesis_cpu_"
    "flat_development_v1.py"
)
V1_QUALIFICATION_HARNESS_SHA256 = (
    "d66ed881de1cb660ae0e05e356a25dcd43e7d3faf487c792d49be389c51140fa"
)
V1_QUALIFICATION_HARNESS_BYTE_COUNT = 71_494

V1_TERMINAL_REVIEW = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_"
    "development_v1_qualification_terminal_review_2026-08-05.json"
)
V1_TERMINAL_REVIEW_SHA256 = (
    "590abd021b33781d7786fc7bdc9f6286173ecd218b53428ae751bad4cd832a9c"
)
V1_TERMINAL_REVIEW_BYTE_COUNT = 9_730

QUALIFICATION_ATTEMPT_ID = (
    "go2-scene-diversity-recurrent-replication-genesis-cpu-flat-"
    "development-v2-qualification"
)
QUALIFICATION_ATTEMPT_ROOT = REPO_ROOT / (
    ".generated/dev/go2_scene_diversity_recurrent_replication_genesis_cpu_"
    "flat_development_v2_qualification/attempt_v1"
)
QUALIFICATION_OUTPUT_ROOT = QUALIFICATION_ATTEMPT_ROOT / "collection"
QUALIFICATION_PLAN_OUTPUT = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_"
    "development_v2_qualification_exact_plan_2026-08-05.json"
)

DEFERRED_SCIENTIFIC_ATTEMPT_ID = (
    "go2-scene-diversity-recurrent-replication-genesis-cpu-flat-development-v2"
)
DEFERRED_SCIENTIFIC_ATTEMPT_ROOT = REPO_ROOT / (
    ".generated/dev/go2_scene_diversity_recurrent_replication_genesis_cpu_"
    "flat_development_v2/attempt_v1"
)

FIXED_HOME = "/home/andrewknowles"
CPU_EXECUTION_ENVIRONMENT = {
    **v1.CPU_EXECUTION_ENVIRONMENT,
    "HOME": FIXED_HOME,
}
QUALIFICATION_SCENE_INDICES = tuple(v1.QUALIFICATION_SCENE_INDICES)
QUALIFICATION_WORKER_WATCHDOG_SECONDS = (
    v1.QUALIFICATION_WORKER_WATCHDOG_SECONDS
)
QUALIFICATION_FIXED_NONCOLLECTION_RESERVE_SECONDS = (
    v1.QUALIFICATION_FIXED_NONCOLLECTION_RESERVE_SECONDS
)
SELECTED_DEVICE_VRAM_CEILING_BYTES = v1.SELECTED_DEVICE_VRAM_CEILING_BYTES
SUCCESSOR_CONTRACT_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_"
    "development_v2_successor_contract_v1"
)


class CpuFlatDevelopmentV2PlanError(RuntimeError):
    """Raised before a changed or non-fresh V2 plan can be emitted."""


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
        raise CpuFlatDevelopmentV2PlanError(
            f"{label} is absent or not strict JSON"
        ) from exc
    if observed != expected:
        raise CpuFlatDevelopmentV2PlanError(f"{label} binding changed")
    if not isinstance(value, dict):
        raise CpuFlatDevelopmentV2PlanError(f"{label} must be an object")
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
        raise CpuFlatDevelopmentV2PlanError(f"{label} is absent") from exc
    if observed != expected:
        raise CpuFlatDevelopmentV2PlanError(f"{label} binding changed")
    return expected


def predecessor_terminal_review_binding() -> dict[str, object]:
    review = _read_bound_json(
        V1_TERMINAL_REVIEW,
        sha256=V1_TERMINAL_REVIEW_SHA256,
        byte_count=V1_TERMINAL_REVIEW_BYTE_COUNT,
        label="V1 terminal review",
    )
    disposition = review.get("scientific_disposition", {})
    successor = review.get("next_experiment", {})
    expected_schema = (
        "lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_"
        "development_v1_qualification_terminal_review_v1"
    )
    if (
        review.get("schema") != expected_schema
        or review.get("status")
        != "PASS_FAIL_CLOSED_PRE_GENESIS_HOME_ENVIRONMENT_ABORT_TERMINAL_REVIEW"
        or disposition.get("qualification_passed") is not False
        or disposition.get("v1_consumed") is not True
        or disposition.get("v1_retired") is not True
        or disposition.get("v1_output_reuse_authorized") is not False
        or successor.get("required_environment_delta") != {"HOME": FIXED_HOME}
        or successor.get("expected_child_environment_key_count") != 10
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
        or successor.get("v1_runtime_output_or_cache_reuse_authorized") is not False
        or successor.get("authorizes_v2_execution_by_itself") is not False
        or successor.get(
            "alternate_environment_architecture_tolerance_or_batching_change_recommended"
        )
        is not False
    ):
        raise CpuFlatDevelopmentV2PlanError(
            "V1 terminal review does not close V1 and scope the HOME successor"
        )
    return {
        "path": str(V1_TERMINAL_REVIEW.resolve(strict=True)),
        "sha256": V1_TERMINAL_REVIEW_SHA256,
        "byte_count": V1_TERMINAL_REVIEW_BYTE_COUNT,
    }


def _read_v1_plan() -> dict[str, Any]:
    value = _read_bound_json(
        V1_QUALIFICATION_PLAN,
        sha256=V1_QUALIFICATION_PLAN_SHA256,
        byte_count=V1_QUALIFICATION_PLAN_BYTE_COUNT,
        label="V1 qualification plan",
    )
    try:
        validated = v1.validate_qualification_plan(value)
    except v1.CpuFlatDevelopmentPlanError as exc:
        raise CpuFlatDevelopmentV2PlanError(str(exc)) from exc
    if (
        validated["execution_contract"]["environment"]
        != v1.CPU_EXECUTION_ENVIRONMENT
    ):
        raise CpuFlatDevelopmentV2PlanError("V1 child environment changed")
    if "HOME" in validated["execution_contract"]["environment"]:
        raise CpuFlatDevelopmentV2PlanError("V1 unexpectedly contains HOME")
    return validated


def _successor_contract(v1_plan: Mapping[str, Any]) -> dict[str, Any]:
    contract = copy.deepcopy(dict(v1_plan["successor_contract"]))
    contract.update(
        {
            "schema": SUCCESSOR_CONTRACT_SCHEMA,
            "plan_role": "qualification",
            "flat_harness_owned": True,
            "module_global_overlay_contexts_authorized": False,
            "imported_project_module_attribute_mutation_authorized": False,
            "v1_qualification_plan_binding": _binding(
                V1_QUALIFICATION_PLAN,
                sha256=V1_QUALIFICATION_PLAN_SHA256,
                byte_count=V1_QUALIFICATION_PLAN_BYTE_COUNT,
                label="V1 qualification plan",
            ),
            "v1_preregistration_binding": _binding(
                V1_PREREGISTRATION,
                sha256=V1_PREREGISTRATION_SHA256,
                byte_count=V1_PREREGISTRATION_BYTE_COUNT,
                label="V1 preregistration",
            ),
            "v1_source_review_binding": _binding(
                V1_SOURCE_REVIEW,
                sha256=V1_SOURCE_REVIEW_SHA256,
                byte_count=V1_SOURCE_REVIEW_BYTE_COUNT,
                label="V1 source review",
            ),
            "v1_plan_builder_binding": _binding(
                V1_PLAN_BUILDER,
                sha256=V1_PLAN_BUILDER_SHA256,
                byte_count=V1_PLAN_BUILDER_BYTE_COUNT,
                label="V1 plan builder",
            ),
            "v1_qualification_harness_binding": _binding(
                V1_QUALIFICATION_HARNESS,
                sha256=V1_QUALIFICATION_HARNESS_SHA256,
                byte_count=V1_QUALIFICATION_HARNESS_BYTE_COUNT,
                label="V1 qualification harness",
            ),
            "v1_terminal_review_binding": predecessor_terminal_review_binding(),
            "v1_authority_reservation_runtime_or_payload_reuse_authorized": False,
            "v2_material_delta": "add_exact_fixed_HOME_to_child_environment",
            "child_environment_key_count": 10,
            "fixed_child_home": FIXED_HOME,
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
    predecessor = _read_v1_plan()
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
        raise CpuFlatDevelopmentV2PlanError(
            "V2 qualification plan must be an object"
        )
    candidate = copy.deepcopy(dict(plan))
    expected = _expected_qualification_plan()
    if _canonical(candidate) != _canonical(expected):
        raise CpuFlatDevelopmentV2PlanError(
            "V2 changed beyond fresh identity/root/contracts and fixed HOME"
        )

    predecessor = _read_v1_plan()
    normalized = copy.deepcopy(candidate)
    normalized["attempt_id"] = predecessor["attempt_id"]
    normalized["output_root"] = predecessor["output_root"]
    normalized["execution_contract"]["environment"].pop("HOME", None)
    normalized["successor_contract"] = copy.deepcopy(
        predecessor["successor_contract"]
    )
    if _canonical(normalized) != _canonical(predecessor):
        raise CpuFlatDevelopmentV2PlanError(
            "V2 normalization does not restore the exact V1 plan"
        )
    if (
        len(candidate["execution_contract"]["environment"]) != 10
        or candidate["execution_contract"]["environment"].get("HOME")
        != FIXED_HOME
    ):
        raise CpuFlatDevelopmentV2PlanError("exact ten-key environment changed")
    return candidate


def _require_fresh_roots() -> None:
    for path in (
        QUALIFICATION_ATTEMPT_ROOT,
        QUALIFICATION_OUTPUT_ROOT,
        DEFERRED_SCIENTIFIC_ATTEMPT_ROOT,
    ):
        if path.exists() or path.is_symlink():
            raise CpuFlatDevelopmentV2PlanError(
                "fresh V2 qualification/scientific roots changed"
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
        raise CpuFlatDevelopmentV2PlanError(
            "V2 qualification plan output must be fresh"
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
    "CpuFlatDevelopmentV2PlanError",
    "DEFERRED_SCIENTIFIC_ATTEMPT_ID",
    "DEFERRED_SCIENTIFIC_ATTEMPT_ROOT",
    "FIXED_HOME",
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
