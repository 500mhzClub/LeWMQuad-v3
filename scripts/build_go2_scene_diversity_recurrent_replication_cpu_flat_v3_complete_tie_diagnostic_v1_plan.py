#!/usr/bin/env python3
"""Build the exact metadata-only CPU-flat V3 complete-tie diagnostic plan."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as pilot  # noqa: E402
from lewm.benchmarks import go2_scene_diversity_recurrent_replication_v1 as benchmark  # noqa: E402
from scripts import run_go2_scene_diversity_recurrent_replication_v1 as frozen_runner  # noqa: E402


PLAN_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_cpu_flat_v3_"
    "complete_tie_diagnostic_v1_exact_plan_v1"
)
PLAN_STATUS = "FROZEN_POST_HOC_DEVELOPMENT_DIAGNOSTIC_PLAN"
ATTEMPT_ID = (
    "go2_scene_diversity_recurrent_replication_cpu_flat_v3_complete_tie_"
    "diagnostic_v1_attempt_v1"
)
ATTEMPT_ROOT = REPO_ROOT / (
    ".generated/dev/go2_scene_diversity_recurrent_replication_cpu_flat_v3_"
    "complete_tie_diagnostic_v1/attempt_v1"
)
RESULT_PATH = ATTEMPT_ROOT / "diagnostic_result.json"
TERMINAL_PATH = ATTEMPT_ROOT / "terminal.json"
PLAN_OUTPUT = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_cpu_flat_v3_"
    "complete_tie_diagnostic_v1_exact_plan_2026-08-05.json"
)
PLAN_SHA256 = (
    "6563deddc4532368a0ea158437c01bb0e49ac3874eae77643cfefa088cc3f918"
)
PLAN_BYTE_COUNT = 9_429
PREREGISTRATION = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_cpu_flat_v3_"
    "complete_tie_diagnostic_v1_preregistration_2026-08-05.md"
)
PREREGISTRATION_SHA256 = (
    "7ebb4c7867a9e27d77419aeabdf3fab2826897fb7f7772ffcb7247bedb778c45"
)
PREREGISTRATION_BYTE_COUNT = 6_460

PREDECESSOR_ATTEMPT_ROOT = REPO_ROOT / (
    ".generated/dev/go2_scene_diversity_recurrent_replication_genesis_cpu_"
    "flat_development_v3/attempt_v1"
)
PREDECESSOR_PLAN = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_"
    "development_v3_scientific_exact_plan_2026-08-05.json"
)
PREDECESSOR_PLAN_SHA256 = (
    "0ad79cc46cead469d6532cd0be04c5d7623fffe18ddafc737c32855d6c9a8f29"
)
PREDECESSOR_PLAN_BYTE_COUNT = 359_692
PREDECESSOR_TERMINAL = PREDECESSOR_ATTEMPT_ROOT / "terminal.json"
PREDECESSOR_TERMINAL_SHA256 = (
    "a4da81177d77372923b72775f69cfe58b596a651017ef6ebc5988df05d390327"
)
PREDECESSOR_TERMINAL_BYTE_COUNT = 1_273
PREDECESSOR_PHYSICS_RESULT = (
    PREDECESSOR_ATTEMPT_ROOT / "collection/physics_result.json"
)
PREDECESSOR_PHYSICS_RESULT_SHA256 = (
    "711b8722c11dbae663ad1b004268b77c64ff3d2e818f2c895851c547240e3ed0"
)
PREDECESSOR_PHYSICS_RESULT_BYTE_COUNT = 369_067
PREDECESSOR_CHECKPOINT = PREDECESSOR_ATTEMPT_ROOT / "checkpoint.pt"
PREDECESSOR_CHECKPOINT_SHA256 = (
    "6c16f97ae5748e1d230244b4588f3efc11330a2673bd15e2ff83aa2f2392844e"
)
PREDECESSOR_CHECKPOINT_BYTE_COUNT = 167_423
PREDECESSOR_RESULT = PREDECESSOR_ATTEMPT_ROOT / "result.json"
PREDECESSOR_TERMINAL_REVIEW = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_"
    "development_v3_scientific_terminal_review_2026-08-05.json"
)
PREDECESSOR_TERMINAL_REVIEW_SHA256 = (
    "7218c78387871e82280f96fe746acb047f46d1a2836b7638b12ce9c1514a81dd"
)
PREDECESSOR_TERMINAL_REVIEW_BYTE_COUNT = 17_379

SOURCE_REVIEW_PATH = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_cpu_flat_v3_"
    "complete_tie_diagnostic_v1_source_review_2026-08-05.json"
)
SOURCE_REVIEW_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_cpu_flat_v3_"
    "complete_tie_diagnostic_v1_source_review_v1"
)
SOURCE_REVIEW_STATUS = (
    "PASS_INDEPENDENT_COMPLETE_TIE_DIAGNOSTIC_SOURCE_REVIEW"
)

EXPECTED_COUNTS = {
    "scenes": 64,
    "states": 256,
    "roles": {"eval": 128, "train": 128},
    "actions": 9,
    "candidate_branches": 2_304,
    "sentinel_branches": 0,
    "total_branches": 2_304,
    "context_frames": 768,
    "target_frames": 2_304,
}
EXPECTED_LEARNED_ARMS = [
    "no_vision_recurrent_direct",
    "visual_recurrent_direct",
]
EXPECTED_MODEL_SEEDS = [2_026_080_411, 2_026_080_412, 2_026_080_413]
EXPECTED_FIXED_GATES = {
    "integrity_role_disjointness_context_only_repeatability_and_oracle": True,
    "maximum_visual_normalized_rank_regret": 0.13,
    "maximum_visual_minus_task_action_regret": -0.02,
    "maximum_visual_minus_no_vision_regret": -0.01,
    "paired_upper_95_must_be_below_zero": True,
    "visual_must_beat_uniform_random": True,
}


class CompleteTieDiagnosticPlanError(RuntimeError):
    """Raised when exact predecessor evidence or diagnostic scope changes."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _file_binding(path: Path) -> dict[str, object]:
    selected = path.resolve(strict=True)
    if not selected.is_file() or selected.is_symlink():
        raise CompleteTieDiagnosticPlanError("bound input is not a regular file")
    digest = hashlib.sha256()
    byte_count = 0
    with selected.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            byte_count += len(chunk)
    return {
        "path": str(selected),
        "sha256": digest.hexdigest(),
        "byte_count": byte_count,
    }


def _exact_binding(
    path: Path, *, sha256: str, byte_count: int, label: str
) -> dict[str, object]:
    expected = {
        "path": str(path.resolve(strict=True)),
        "sha256": sha256,
        "byte_count": byte_count,
    }
    if _file_binding(path) != expected:
        raise CompleteTieDiagnosticPlanError(f"{label} binding changed")
    return expected


def _read_exact_json(
    path: Path, *, sha256: str, byte_count: int, label: str
) -> tuple[dict[str, Any], dict[str, object]]:
    binding = _exact_binding(
        path, sha256=sha256, byte_count=byte_count, label=label
    )
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompleteTieDiagnosticPlanError(
            f"{label} is not strict JSON"
        ) from exc
    if not isinstance(value, dict):
        raise CompleteTieDiagnosticPlanError(f"{label} must be an object")
    return value, binding


def _preregistration_binding() -> dict[str, object]:
    return _exact_binding(
        PREREGISTRATION,
        sha256=PREREGISTRATION_SHA256,
        byte_count=PREREGISTRATION_BYTE_COUNT,
        label="diagnostic preregistration",
    )


def _require_fresh_root() -> None:
    namespace = ATTEMPT_ROOT.parent
    development = (REPO_ROOT / ".generated/dev").resolve(strict=True)
    if (
        not ATTEMPT_ROOT.resolve(strict=False).is_relative_to(development)
        or namespace.is_symlink()
        or ATTEMPT_ROOT.exists()
        or ATTEMPT_ROOT.is_symlink()
        or RESULT_PATH.exists()
        or RESULT_PATH.is_symlink()
        or TERMINAL_PATH.exists()
        or TERMINAL_PATH.is_symlink()
    ):
        raise CompleteTieDiagnosticPlanError(
            "diagnostic attempt root is not exact and fresh"
        )


def _predecessor_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    plan, plan_binding = _read_exact_json(
        PREDECESSOR_PLAN,
        sha256=PREDECESSOR_PLAN_SHA256,
        byte_count=PREDECESSOR_PLAN_BYTE_COUNT,
        label="predecessor scientific plan",
    )
    terminal, terminal_binding = _read_exact_json(
        PREDECESSOR_TERMINAL,
        sha256=PREDECESSOR_TERMINAL_SHA256,
        byte_count=PREDECESSOR_TERMINAL_BYTE_COUNT,
        label="predecessor terminal",
    )
    physics, physics_binding = _read_exact_json(
        PREDECESSOR_PHYSICS_RESULT,
        sha256=PREDECESSOR_PHYSICS_RESULT_SHA256,
        byte_count=PREDECESSOR_PHYSICS_RESULT_BYTE_COUNT,
        label="predecessor physics result",
    )
    checkpoint_binding = _exact_binding(
        PREDECESSOR_CHECKPOINT,
        sha256=PREDECESSOR_CHECKPOINT_SHA256,
        byte_count=PREDECESSOR_CHECKPOINT_BYTE_COUNT,
        label="predecessor checkpoint",
    )
    review, review_binding = _read_exact_json(
        PREDECESSOR_TERMINAL_REVIEW,
        sha256=PREDECESSOR_TERMINAL_REVIEW_SHA256,
        byte_count=PREDECESSOR_TERMINAL_REVIEW_BYTE_COUNT,
        label="independent predecessor terminal review",
    )
    protocol = plan.get("successor_contract", {}).get("scientific_protocol")
    failure_localization = review.get("failure_localization", {})
    disposition = review.get("scientific_disposition", {})
    recommendation = review.get("successor_recommendation", {})
    review_decision = review.get("decision", {})
    if (
        plan.get("schema")
        != "lewm_go2_world_model_counterfactual_pilot_plan_v1"
        or plan.get("attempt_id")
        != "go2-scene-diversity-recurrent-replication-genesis-cpu-flat-development-v3"
        or plan.get("expected_counts") != EXPECTED_COUNTS
        or not isinstance(plan.get("states"), list)
        or len(plan["states"]) != 256
        or not isinstance(protocol, Mapping)
        or protocol.get("learned_arms_in_order") != EXPECTED_LEARNED_ARMS
        or protocol.get("live_control_arm") != "task_action_only"
        or protocol.get("model_seeds") != EXPECTED_MODEL_SEEDS
        or protocol.get("updates") != 800
        or protocol.get("paired_family_scene_bootstrap_resamples") != 10_000
        or protocol.get("fixed_gates") != EXPECTED_FIXED_GATES
        or terminal.get("schema")
        != "lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_development_v3_scientific_terminal_v1"
        or terminal.get("status") != "FAIL_INFRASTRUCTURE_NO_SCIENTIFIC_DECISION"
        or terminal.get("result_binding") is not None
        or terminal.get("failure", {}).get("message") != "dense ranks are invalid"
        or terminal.get("authorizes_retry_or_resume") is not False
        or terminal.get("authorizes_navigation_claim") is not False
        or physics.get("schema") != pilot.PHYSICS_RESULT_SCHEMA
        or physics.get("status") != "PHYSICS_COMPLETE"
        or physics.get("failure") is not None
        or physics.get("expected_counts") != EXPECTED_COUNTS
        or physics.get("observed_counts") != EXPECTED_COUNTS
        or len(physics.get("state_receipt_bindings", [])) != 256
        or len(physics.get("render_receipt_bindings", [])) != 64
        or len(physics.get("scene_metrics", [])) != 64
        or physics.get("plan_binding")
        != {
            "path": str(PREDECESSOR_PLAN.resolve(strict=True)),
            "file_sha256": PREDECESSOR_PLAN_SHA256,
            "byte_count": PREDECESSOR_PLAN_BYTE_COUNT,
        }
        or review.get("schema")
        != "lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_development_v3_scientific_terminal_review_v1"
        or review.get("status") != "FAIL_CLOSED_NO_SCIENTIFIC_DECISION"
        or review.get("protected_material_opened") is not False
        or review.get("bindings", {}).get("terminal") != terminal_binding
        or review.get("bindings", {}).get("physics_result") != physics_binding
        or review.get("bindings", {}).get("checkpoint") != checkpoint_binding
        or review.get("bindings", {}).get("scientific_exact_plan")
        != plan_binding
        or review.get("bindings", {}).get("scientific_result") is not None
        or failure_localization.get("rank_invalid_all_tie_states") != 4
        or failure_localization.get("rank_invalid_by_role")
        != {"train": 0, "eval": 4}
        or review_decision.get("attempt_consumed_and_fail_closed") is not True
        or review_decision.get("scientific_decision") is not False
        or review_decision.get("successor_authority_created") is not False
        or disposition.get("scientific_conclusion_available") is not False
        or disposition.get(
            "retry_resume_refill_overwrite_or_repair_authorized"
        )
        is not False
        or recommendation.get("authority_created_by_this_review") is not False
        or recommendation.get("successor_execution_authorized") is not False
        or recommendation.get(
            "same_consumed_artifacts_could_only_support_post_hoc_diagnostic_not_preregistered_v3_decision"
        )
        is not True
        or PREDECESSOR_RESULT.exists()
        or PREDECESSOR_RESULT.is_symlink()
    ):
        raise CompleteTieDiagnosticPlanError(
            "predecessor evidence or disposition changed"
        )
    bindings = {
        "scientific_plan_binding": plan_binding,
        "terminal_binding": terminal_binding,
        "physics_result_binding": physics_binding,
        "checkpoint_binding": checkpoint_binding,
        "terminal_review_binding": review_binding,
    }
    evidence = {
        "predecessor_attempt_consumed": True,
        "predecessor_scientific_result_absent": True,
        "predecessor_scientific_decision_available": False,
        "physics_collection_complete": True,
        "checkpoint_integrity_review_passed": True,
        "complete_tie_eval_state_count": 4,
        "complete_tie_train_state_count": 0,
        "predecessor_artifact_reuse_authorized_by_review": False,
        "separate_preregistration_review_and_authority_required": True,
    }
    return {
        **bindings,
        "frozen_scientific_protocol": copy.deepcopy(dict(protocol)),
    }, evidence


def _expected_plan() -> dict[str, Any]:
    predecessor, evidence = _predecessor_inputs()
    return {
        "schema": PLAN_SCHEMA,
        "status": PLAN_STATUS,
        "attempt_id": ATTEMPT_ID,
        "attempt_root": str(ATTEMPT_ROOT.resolve(strict=False)),
        "result_path": str(RESULT_PATH.resolve(strict=False)),
        "terminal_path": str(TERMINAL_PATH.resolve(strict=False)),
        "development_only": True,
        "post_hoc_nonconfirmatory": True,
        "citable_as_scientific_evidence": False,
        "fresh_root_required": True,
        "preregistration_binding": _preregistration_binding(),
        "predecessor": {
            "attempt_root": str(PREDECESSOR_ATTEMPT_ROOT.resolve(strict=True)),
            "collection_root": str(
                (PREDECESSOR_ATTEMPT_ROOT / "collection").resolve(strict=True)
            ),
            "result_path": str(PREDECESSOR_RESULT.resolve(strict=False)),
            "result_must_be_absent": True,
            "scientific_plan_binding": predecessor[
                "scientific_plan_binding"
            ],
            "terminal_binding": predecessor["terminal_binding"],
            "physics_result_binding": predecessor[
                "physics_result_binding"
            ],
            "checkpoint_binding": predecessor["checkpoint_binding"],
            "terminal_review_binding": predecessor[
                "terminal_review_binding"
            ],
            "evidence_disposition": evidence,
        },
        "evaluation_contract": {
            "evaluation_only": True,
            "training_authorized": False,
            "rendering_authorized": False,
            "collection_authorized": False,
            "checkpoint_reuse_mode": "read_only_exact_rehash",
            "collection_reuse_mode": "read_only_exact_rehash",
            "roles_reconstructed": ["train", "eval"],
            "train_role_use": "live_task_action_only_control_metadata_only",
            "train_context_rgb_open_count": 0,
            "eval_context_rgb_open_count": 384,
            "successor_rgb_open_count": 0,
            "eval_state_count": 128,
            "expected_eval_complete_tie_state_count": 4,
            "eval_state_exclusion_authorized": False,
            "complete_tie_rule": "all_actions_oracle_equivalent",
            "random_expected_denominator": "max(1,max_dense_rank)",
            "rank_tolerance_m": 0.01,
            "evaluation_repetitions": 2,
            "repeat_evaluation_exact_required": True,
            "compute_device": "cpu",
            "frozen_recurrent_config": copy.deepcopy(benchmark.config_v1()),
            "model_seeds": EXPECTED_MODEL_SEEDS,
            "sampler_seed": 2_026_080_414,
            "bootstrap_resamples": 10_000,
            "bootstrap_seed": 2_026_080_407,
            "frozen_thresholds": copy.deepcopy(
                benchmark.config_v1()["frozen_recurrent_protocol"][
                    "frozen_h1_thresholds"
                ]
            ),
        },
        "dino": frozen_runner.expected_dino_v1(),
        "result_contract": {
            "output_kind": "POST_HOC_DEVELOPMENT_ONLY_NONCONFIRMATORY_DIAGNOSTIC",
            "mechanical_fixed_gate_values_reported": True,
            "creates_missing_v3_scientific_decision": False,
            "salvages_or_reclassifies_predecessor": False,
            "citable_as_confirmatory_scientific_evidence": False,
            "authorizes_navigation_planning_representation_world_model_or_generalization_claim": False,
            "authorizes_promotion_or_deployment": False,
        },
        "source_review_contract": {
            "required_before_root_creation": True,
            "path": str(SOURCE_REVIEW_PATH.resolve(strict=False)),
            "schema": SOURCE_REVIEW_SCHEMA,
            "status": SOURCE_REVIEW_STATUS,
            "exactly_one_diagnostic_invocation_may_be_cleared": True,
            "source_review_itself_creates_execution_authority": False,
        },
        "permissions": {
            "diagnostic_execution_authorized_by_plan": False,
            "retry_authorized": False,
            "resume_authorized": False,
            "refill_authorized": False,
            "overwrite_authorized": False,
            "repair_authorized": False,
            "second_invocation_authorized": False,
            "predecessor_mutation_authorized": False,
            "protected_material_access_authorized": False,
        },
    }


def validate_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(plan, Mapping):
        raise CompleteTieDiagnosticPlanError("diagnostic plan must be an object")
    candidate = copy.deepcopy(dict(plan))
    if _canonical(candidate) != _canonical(_expected_plan()):
        raise CompleteTieDiagnosticPlanError("diagnostic plan changed")
    return candidate


def build_plan() -> dict[str, Any]:
    _require_fresh_root()
    return validate_plan(_expected_plan())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-output", type=Path, default=PLAN_OUTPUT)
    args = parser.parse_args(argv)
    output = args.plan_output.resolve(strict=False)
    if output.exists() or output.is_symlink():
        raise CompleteTieDiagnosticPlanError("plan output must be fresh")
    plan = build_plan()
    binding = pilot.write_json_exclusive(output, plan)
    print(json.dumps({"plan": binding, "execution_authorized": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ATTEMPT_ID",
    "ATTEMPT_ROOT",
    "CompleteTieDiagnosticPlanError",
    "PLAN_OUTPUT",
    "PLAN_SHA256",
    "PLAN_BYTE_COUNT",
    "PLAN_SCHEMA",
    "PLAN_STATUS",
    "PREREGISTRATION",
    "RESULT_PATH",
    "SOURCE_REVIEW_PATH",
    "SOURCE_REVIEW_SCHEMA",
    "SOURCE_REVIEW_STATUS",
    "TERMINAL_PATH",
    "build_plan",
    "validate_plan",
]
