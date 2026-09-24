#!/usr/bin/env python3
"""Release the qualified CPU-flat V3 scientific plan, without payload reuse.

Only the minimal qualification PASS decision and its independent terminal
review are admissible qualification inputs.  The rich result, terminal, scene
results, receipts, RGB, meshes, caches, and diagnostics are never opened here.
The 64-scene plan differs from the exact qualification plan only in fresh
scientific identity/root and scientific successor-contract metadata.

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
from scripts import build_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_development_v3_plan as qualification  # noqa: E402


QUALIFICATION_PLAN = qualification.QUALIFICATION_PLAN_OUTPUT
QUALIFICATION_PLAN_SHA256 = (
    "6a055839ab9bb6fe45b9cb5864e8f3c87e75f468dd7e9c26e8c950e4a6fedb78"
)
QUALIFICATION_PLAN_BYTE_COUNT = 355_206

QUALIFICATION_PREREGISTRATION = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_"
    "development_v3_preregistration_2026-08-05.md"
)
QUALIFICATION_PREREGISTRATION_SHA256 = (
    "c2c894d8bef28fdc641fa9be0706b4becea1e1311cdeb9687d01b2eda3aafe7b"
)
QUALIFICATION_PREREGISTRATION_BYTE_COUNT = 9_020

QUALIFICATION_SOURCE_REVIEW = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_"
    "development_v3_qualification_source_review_2026-08-05.json"
)
QUALIFICATION_SOURCE_REVIEW_SHA256 = (
    "e5d1e57b047f9a763e3197db1a45729e8a156f0fa6a8d2e421d2371e6d8fe991"
)
QUALIFICATION_SOURCE_REVIEW_BYTE_COUNT = 14_314

QUALIFICATION_PLAN_BUILDER = REPO_ROOT / (
    "scripts/build_go2_scene_diversity_recurrent_replication_genesis_cpu_"
    "flat_development_v3_plan.py"
)
QUALIFICATION_PLAN_BUILDER_SHA256 = (
    "d8c47e9c52301f0d2e1e7f0de5b547f49ce89b94dda0e83dfec5d651d312738b"
)
QUALIFICATION_PLAN_BUILDER_BYTE_COUNT = 16_568

QUALIFICATION_HARNESS = REPO_ROOT / (
    "scripts/run_go2_scene_diversity_recurrent_replication_genesis_cpu_"
    "flat_development_v3.py"
)
QUALIFICATION_HARNESS_SHA256 = (
    "c08653e27cb33048814539f2a8ba06be69ee917bbc0bac9a4bc5ea6206c468dd"
)
QUALIFICATION_HARNESS_BYTE_COUNT = 71_879

QUALIFICATION_PASS_DECISION = qualification.QUALIFICATION_ATTEMPT_ROOT / (
    "qualification_decision.json"
)
QUALIFICATION_PASS_DECISION_SHA256 = (
    "6eac9ff6458092d6284011934c865b45519cbd2ff14c0b2da3a515bbf4a6a299"
)
QUALIFICATION_PASS_DECISION_BYTE_COUNT = 1_789
QUALIFICATION_RESULT_METADATA_BINDING = {
    "path": str(
        qualification.QUALIFICATION_ATTEMPT_ROOT
        / "qualification_result.json"
    ),
    "sha256": "fdd238760b18658b867846fde30102576964f9ece4307417de12ac4fcc80f40d",
    "byte_count": 9_179,
}
QUALIFICATION_TERMINAL_METADATA_BINDING = {
    "path": str(qualification.QUALIFICATION_ATTEMPT_ROOT / "terminal.json"),
    "sha256": "18df9336a4b118843db77940b6a4086d0ee1d708cf94984c6418935510de32d0",
    "byte_count": 711,
}

QUALIFICATION_TERMINAL_REVIEW = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_"
    "development_v3_qualification_terminal_review_2026-08-05.json"
)
QUALIFICATION_TERMINAL_REVIEW_SHA256 = (
    "bc13e880fc348515384b906ea7dce32bc7587df089f7aa7f6b630d71e87ce31d"
)
QUALIFICATION_TERMINAL_REVIEW_BYTE_COUNT = 13_591

SCIENTIFIC_ATTEMPT_ID = qualification.DEFERRED_SCIENTIFIC_ATTEMPT_ID
SCIENTIFIC_ATTEMPT_ROOT = qualification.DEFERRED_SCIENTIFIC_ATTEMPT_ROOT
SCIENTIFIC_OUTPUT_ROOT = SCIENTIFIC_ATTEMPT_ROOT / "collection"
SCIENTIFIC_PLAN_OUTPUT = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_"
    "development_v3_scientific_exact_plan_2026-08-05.json"
)
SCIENTIFIC_PREREGISTRATION = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_"
    "development_v3_scientific_preregistration_2026-08-05.md"
)

DEFAULT_ATTEMPT_ID = SCIENTIFIC_ATTEMPT_ID
DEFAULT_ATTEMPT_ROOT = SCIENTIFIC_ATTEMPT_ROOT
DEFAULT_OUTPUT_ROOT = SCIENTIFIC_OUTPUT_ROOT
DEFAULT_PLAN_OUTPUT = SCIENTIFIC_PLAN_OUTPUT

CPU_EXECUTION_ENVIRONMENT = copy.deepcopy(
    qualification.CPU_EXECUTION_ENVIRONMENT
)
SUCCESSOR_CONTRACT_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_"
    "development_v3_scientific_successor_contract_v1"
)
SCIENTIFIC_PROTOCOL = {
    "learned_arms_in_order": [
        "no_vision_recurrent_direct",
        "visual_recurrent_direct",
    ],
    "live_control_arm": "task_action_only",
    "model_seeds": [2_026_080_411, 2_026_080_412, 2_026_080_413],
    "shared_sampler_seed": 2_026_080_414,
    "updates": 800,
    "batch_states": 8,
    "optimizer": {
        "name": "AdamW",
        "learning_rate": 3.0e-4,
        "weight_decay": 1.0e-4,
        "betas": [0.9, 0.999],
        "epsilon": 1.0e-8,
    },
    "gradient_clip_norm": 1.0,
    "task_ridge_lambda": 1.0e-3,
    "successor_observation_access": False,
    "fixed_gates": {
        "integrity_role_disjointness_context_only_repeatability_and_oracle": True,
        "maximum_visual_normalized_rank_regret": 0.13,
        "maximum_visual_minus_task_action_regret": -0.02,
        "maximum_visual_minus_no_vision_regret": -0.01,
        "paired_upper_95_must_be_below_zero": True,
        "visual_must_beat_uniform_random": True,
    },
    "paired_family_scene_bootstrap_resamples": 10_000,
    "durable_checkpoint_required_before_eval_open": True,
    "two_exact_evaluations_required": True,
}


class CpuFlatDevelopmentV3ScientificPlanError(RuntimeError):
    """Raised before a changed, unqualified, or non-fresh plan is emitted."""


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
        raise CpuFlatDevelopmentV3ScientificPlanError(
            f"{label} is absent or not strict JSON"
        ) from exc
    if observed != expected:
        raise CpuFlatDevelopmentV3ScientificPlanError(
            f"{label} binding changed"
        )
    if not isinstance(value, dict):
        raise CpuFlatDevelopmentV3ScientificPlanError(
            f"{label} must be an object"
        )
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
        raise CpuFlatDevelopmentV3ScientificPlanError(
            f"{label} is absent"
        ) from exc
    if observed != expected:
        raise CpuFlatDevelopmentV3ScientificPlanError(
            f"{label} binding changed"
        )
    return expected


def qualification_pass_decision_binding() -> dict[str, object]:
    value = _read_bound_json(
        QUALIFICATION_PASS_DECISION,
        sha256=QUALIFICATION_PASS_DECISION_SHA256,
        byte_count=QUALIFICATION_PASS_DECISION_BYTE_COUNT,
        label="qualification PASS decision",
    )
    expected_keys = {
        "all_scene_gates_passed",
        "attempt_id",
        "authorizes_retry_or_resume",
        "authorizes_scientific_plan_release",
        "backend",
        "branch_mechanism",
        "exact_lane_equality_gate_passed",
        "genesis_version",
        "kernel_gate_passed",
        "plan_binding",
        "probe_order",
        "qualification_payload_reuse_authorized",
        "qualification_result_binding",
        "qualification_terminal_binding",
        "schema",
        "scientific_attempt_root_absent",
        "status",
        "timing_gate_passed",
        "vram_and_release_gates_passed",
    }
    expected_plan = {
        "path": str(QUALIFICATION_PLAN.resolve(strict=True)),
        "sha256": QUALIFICATION_PLAN_SHA256,
        "byte_count": QUALIFICATION_PLAN_BYTE_COUNT,
    }
    if (
        set(value) != expected_keys
        or value.get("schema")
        != "lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_"
        "development_v3_qualification_decision_v1"
        or value.get("status")
        != "PASS_GENESIS_CPU_FLAT_DEVELOPMENT_V3_QUALIFICATION"
        or value.get("attempt_id") != qualification.QUALIFICATION_ATTEMPT_ID
        or value.get("plan_binding") != expected_plan
        or value.get("probe_order") != [12, 0]
        or value.get("backend") != "cpu"
        or value.get("branch_mechanism")
        != "parallel_lockstep_envs_no_restore"
        or value.get("genesis_version") != "0.3.14"
        or value.get("all_scene_gates_passed") is not True
        or value.get("exact_lane_equality_gate_passed") is not True
        or value.get("kernel_gate_passed") is not True
        or value.get("timing_gate_passed") is not True
        or value.get("vram_and_release_gates_passed") is not True
        or value.get("authorizes_scientific_plan_release") is not True
        or value.get("authorizes_retry_or_resume") is not False
        or value.get("qualification_payload_reuse_authorized") is not False
        or value.get("scientific_attempt_root_absent") is not True
        or value.get("qualification_result_binding")
        != QUALIFICATION_RESULT_METADATA_BINDING
        or value.get("qualification_terminal_binding")
        != QUALIFICATION_TERMINAL_METADATA_BINDING
    ):
        raise CpuFlatDevelopmentV3ScientificPlanError(
            "qualification PASS decision changed"
        )
    return {
        "path": str(QUALIFICATION_PASS_DECISION.resolve(strict=True)),
        "sha256": QUALIFICATION_PASS_DECISION_SHA256,
        "byte_count": QUALIFICATION_PASS_DECISION_BYTE_COUNT,
    }


def independent_qualification_terminal_review_binding() -> dict[str, object]:
    review = _read_bound_json(
        QUALIFICATION_TERMINAL_REVIEW,
        sha256=QUALIFICATION_TERMINAL_REVIEW_SHA256,
        byte_count=QUALIFICATION_TERMINAL_REVIEW_BYTE_COUNT,
        label="independent qualification terminal review",
    )
    release = review.get("science_plan_release", {})
    decision = review.get("decision", {})
    disposition = review.get("scientific_disposition", {})
    expected_schema = (
        "lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_"
        "development_v3_qualification_terminal_review_v1"
    )
    if (
        review.get("schema") != expected_schema
        or review.get("status")
        != "PASS_INDEPENDENT_QUALIFICATION_TERMINAL_REVIEW"
        or release.get("authorizing_artifact") != "qualification_decision"
        or release.get("authorizing_decision_binding")
        != qualification_pass_decision_binding()
        or release.get(
            "decision_authorizes_only_a_separately_preregistered_and_independently_reviewed_science_plan"
        )
        is not True
        or release.get("scientific_attempt_root")
        != str(SCIENTIFIC_ATTEMPT_ROOT.resolve(strict=False))
        or release.get(
            "scientific_attempt_root_absent_and_not_symlink_at_terminal_review"
        )
        is not True
        or release.get("qualification_payload_reuse_authorized") is not False
        or release.get(
            "qualification_result_terminal_scene_state_render_rgb_mesh_or_diagnostic_payload_may_enter_science"
        )
        is not False
        or release.get("science_plan_created_by_this_review") is not False
        or release.get("scientific_execution_authorized_by_this_review")
        is not False
        or decision.get("terminal_review_passed") is not True
        or decision.get(
            "only_minimal_decision_plus_this_review_may_gate_a_separately_reviewed_science_plan"
        )
        is not True
        or decision.get("qualification_payload_reuse_authorized") is not False
        or disposition.get("qualification_passed") is not True
        or disposition.get("separate_science_plan_release_condition_satisfied")
        is not True
        or disposition.get("qualification_runtime_payload_reuse_authorized")
        is not False
        or disposition.get("scientific_execution_authorized") is not False
    ):
        raise CpuFlatDevelopmentV3ScientificPlanError(
            "independent qualification terminal review changed"
        )
    return {
        "path": str(QUALIFICATION_TERMINAL_REVIEW.resolve(strict=True)),
        "sha256": QUALIFICATION_TERMINAL_REVIEW_SHA256,
        "byte_count": QUALIFICATION_TERMINAL_REVIEW_BYTE_COUNT,
    }


def _read_qualification_plan() -> dict[str, Any]:
    value = _read_bound_json(
        QUALIFICATION_PLAN,
        sha256=QUALIFICATION_PLAN_SHA256,
        byte_count=QUALIFICATION_PLAN_BYTE_COUNT,
        label="V3 qualification plan",
    )
    try:
        return qualification.validate_qualification_plan(value)
    except qualification.CpuFlatDevelopmentV3PlanError as exc:
        raise CpuFlatDevelopmentV3ScientificPlanError(str(exc)) from exc


def _scientific_successor_contract(
    qualification_plan: Mapping[str, Any],
) -> dict[str, Any]:
    contract = copy.deepcopy(dict(qualification_plan["successor_contract"]))
    contract.update(
        {
            "schema": SUCCESSOR_CONTRACT_SCHEMA,
            "plan_role": "scientific",
            "scientific_attempt_id": SCIENTIFIC_ATTEMPT_ID,
            "scientific_attempt_root": str(
                SCIENTIFIC_ATTEMPT_ROOT.resolve(strict=False)
            ),
            "qualification_plan_binding": _binding(
                QUALIFICATION_PLAN,
                sha256=QUALIFICATION_PLAN_SHA256,
                byte_count=QUALIFICATION_PLAN_BYTE_COUNT,
                label="V3 qualification plan",
            ),
            "qualification_preregistration_binding": _binding(
                QUALIFICATION_PREREGISTRATION,
                sha256=QUALIFICATION_PREREGISTRATION_SHA256,
                byte_count=QUALIFICATION_PREREGISTRATION_BYTE_COUNT,
                label="V3 qualification preregistration",
            ),
            "qualification_source_review_binding": _binding(
                QUALIFICATION_SOURCE_REVIEW,
                sha256=QUALIFICATION_SOURCE_REVIEW_SHA256,
                byte_count=QUALIFICATION_SOURCE_REVIEW_BYTE_COUNT,
                label="V3 qualification source review",
            ),
            "qualification_plan_builder_binding": _binding(
                QUALIFICATION_PLAN_BUILDER,
                sha256=QUALIFICATION_PLAN_BUILDER_SHA256,
                byte_count=QUALIFICATION_PLAN_BUILDER_BYTE_COUNT,
                label="V3 qualification plan builder",
            ),
            "qualification_harness_binding": _binding(
                QUALIFICATION_HARNESS,
                sha256=QUALIFICATION_HARNESS_SHA256,
                byte_count=QUALIFICATION_HARNESS_BYTE_COUNT,
                label="V3 qualification harness",
            ),
            "qualification_pass_decision_binding": (
                qualification_pass_decision_binding()
            ),
            "independent_qualification_terminal_review_binding": (
                independent_qualification_terminal_review_binding()
            ),
            "qualification_pass_decision_validated": True,
            "qualification_payload_reuse_authorized": False,
            "qualification_result_terminal_scene_receipt_rgb_cache_or_diagnostic_open_authorized": False,
            "qualification_runtime_payload_opened_by_builder": False,
            "exact_64_scene_payload_unchanged": True,
            "expected_counts": {
                "scenes": 64,
                "states": 256,
                "candidate_branches": 2_304,
                "context_frames": 768,
                "target_frames": 2_304,
            },
            "one_fresh_process_per_scene": True,
            "branch_mechanism": "parallel_lockstep_envs_no_restore",
            "child_environment_key_count": 11,
            "scientific_protocol": copy.deepcopy(SCIENTIFIC_PROTOCOL),
            "scientific_plan_created": True,
            "scientific_plan_released_after_exact_pass": True,
            "scientific_source_review_required": True,
            "scientific_execution_authorized": False,
            "development_only": True,
            "promotable": False,
        }
    )
    return contract


def _expected_scientific_plan() -> dict[str, Any]:
    predecessor = _read_qualification_plan()
    candidate = copy.deepcopy(predecessor)
    candidate["attempt_id"] = SCIENTIFIC_ATTEMPT_ID
    candidate["output_root"] = str(SCIENTIFIC_OUTPUT_ROOT.resolve(strict=False))
    candidate["successor_contract"] = _scientific_successor_contract(
        predecessor
    )
    return candidate


def validate_scientific_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(plan, Mapping):
        raise CpuFlatDevelopmentV3ScientificPlanError(
            "scientific plan must be an object"
        )
    candidate = copy.deepcopy(dict(plan))
    expected = _expected_scientific_plan()
    if _canonical(candidate) != _canonical(expected):
        raise CpuFlatDevelopmentV3ScientificPlanError(
            "scientific plan changed beyond identity/root/release metadata"
        )
    predecessor = _read_qualification_plan()
    normalized = copy.deepcopy(candidate)
    normalized["attempt_id"] = predecessor["attempt_id"]
    normalized["output_root"] = predecessor["output_root"]
    normalized["successor_contract"] = copy.deepcopy(
        predecessor["successor_contract"]
    )
    if _canonical(normalized) != _canonical(predecessor):
        raise CpuFlatDevelopmentV3ScientificPlanError(
            "scientific normalization does not restore qualification plan"
        )
    if (
        candidate.get("expected_counts", {}).get("scenes") != 64
        or candidate.get("expected_counts", {}).get("states") != 256
        or candidate.get("expected_counts", {}).get("candidate_branches")
        != 2_304
        or candidate.get("branch_mechanism")
        != "parallel_lockstep_envs_no_restore"
        or candidate.get("execution_contract", {}).get("environment")
        != CPU_EXECUTION_ENVIRONMENT
    ):
        raise CpuFlatDevelopmentV3ScientificPlanError(
            "scientific data or runtime contract changed"
        )
    return candidate


def _require_fresh_scientific_root() -> None:
    for path in (SCIENTIFIC_ATTEMPT_ROOT, SCIENTIFIC_OUTPUT_ROOT):
        if path.exists() or path.is_symlink():
            raise CpuFlatDevelopmentV3ScientificPlanError(
                "fresh scientific root changed"
            )


def build_scientific_plan() -> dict[str, Any]:
    _require_fresh_scientific_root()
    qualification_pass_decision_binding()
    independent_qualification_terminal_review_binding()
    return validate_scientific_plan(_expected_scientific_plan())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scientific-plan-output",
        type=Path,
        default=SCIENTIFIC_PLAN_OUTPUT,
    )
    args = parser.parse_args(argv)
    if args.scientific_plan_output.exists() or args.scientific_plan_output.is_symlink():
        raise CpuFlatDevelopmentV3ScientificPlanError(
            "scientific plan output must be fresh"
        )
    plan = build_scientific_plan()
    binding = pilot.write_json_exclusive(args.scientific_plan_output, plan)
    print(
        json.dumps(
            {
                "scientific_plan": binding,
                "qualification_pass_decision": (
                    qualification_pass_decision_binding()
                ),
                "qualification_payload_reused": False,
                "scientific_execution_authorized_by_plan_builder": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CPU_EXECUTION_ENVIRONMENT",
    "CpuFlatDevelopmentV3ScientificPlanError",
    "DEFAULT_ATTEMPT_ID",
    "DEFAULT_ATTEMPT_ROOT",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_PLAN_OUTPUT",
    "QUALIFICATION_PASS_DECISION",
    "QUALIFICATION_PASS_DECISION_BYTE_COUNT",
    "QUALIFICATION_PASS_DECISION_SHA256",
    "QUALIFICATION_TERMINAL_REVIEW",
    "SCIENTIFIC_ATTEMPT_ID",
    "SCIENTIFIC_ATTEMPT_ROOT",
    "SCIENTIFIC_OUTPUT_ROOT",
    "SCIENTIFIC_PLAN_OUTPUT",
    "SCIENTIFIC_PREREGISTRATION",
    "SCIENTIFIC_PROTOCOL",
    "SUCCESSOR_CONTRACT_SCHEMA",
    "build_scientific_plan",
    "independent_qualification_terminal_review_binding",
    "qualification_pass_decision_binding",
    "validate_scientific_plan",
]
