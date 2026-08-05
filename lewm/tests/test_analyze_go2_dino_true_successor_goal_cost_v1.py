from __future__ import annotations

import copy
import json
from pathlib import Path
import random

import pytest

from scripts.analyze_go2_dino_true_successor_goal_cost_v1 import (
    DINO_CHECKPOINT_BYTES,
    DINO_CHECKPOINT_PATH,
    DINO_CHECKPOINT_SHA256,
    DINO_COST_DEFINITION,
    DINO_ENCODER_NAME,
    DINO_IMAGENET_MEAN,
    DINO_IMAGENET_STD,
    DINO_INPUT_RGB_SHAPE,
    DINO_PATCH_OUTPUT_SHAPE,
    DINO_REPOSITORY_COMMIT,
    DINO_REPOSITORY_PATH,
    DINO_SUCCESSOR_EVALUATION,
    DINO_TOKEN_NORMALIZATION,
    EXPECTED_CANDIDATE_SEQUENCES,
    EXPECTED_POLICIES,
    EXPECTED_PRIMITIVES,
    SHUFFLE_MIX_CONSTANT,
    DinoCeilingAnalysisError,
    analyze_file,
    analyze_payload,
    main,
)


POLICY_VALUES = {
    "oracle_mpc": (1.00, True, 0.90, 0.5e-9),
    "dino_true_successor": (0.80, False, 0.70, 0.01),
    "dino_true_successor_shuffled": (0.50, False, 0.40, 0.05),
    "dino_persistence": (0.45, False, 0.35, 0.04),
    "bearing": (0.90, True, 0.80, 0.01),
    "hold": (0.00, False, 0.00, 0.08),
    "random": (0.20, False, 0.15, 0.07),
}


def _decision(policy: str, regret: float) -> dict[str, object]:
    decision: dict[str, object] = {
        "block_index": 0,
        "oracle_first_action_regret_m": regret,
        "oracle_first_action_disagreement": regret > 1.0e-9,
        "oracle_cost_tie_tolerance_m": 1.0e-9,
    }
    if policy not in {
        "dino_true_successor",
        "dino_true_successor_shuffled",
        "dino_persistence",
    }:
        return decision

    if policy == "dino_persistence":
        unshuffled = [0.6] * 9
        sources = list(range(9))
    else:
        unshuffled = [0.8, 0.7, 0.6, 0.5, 0.55, 0.65, 0.75, 0.85, 0.9]
        sources = list(range(9))
        if policy == "dino_true_successor_shuffled":
            mixed_seed = (7 & ((1 << 64) - 1)) ^ (
                SHUFFLE_MIX_CONSTANT & ((1 << 64) - 1)
            )
            random.Random(mixed_seed).shuffle(sources)
            if sources == list(range(9)):
                sources = sources[1:] + sources[:1]
    scores = [unshuffled[index] for index in sources]
    selected = min(range(len(scores)), key=scores.__getitem__)
    decision.update(
        {
            "selected_candidate_index": selected,
            "selected_policy_score": scores[selected],
            "policy_score_margin": 0.0,
            "selected_score_source_candidate_index": sources[selected],
            "policy_candidate_scores": scores,
            "unshuffled_dino_candidate_costs": unshuffled,
            "score_source_candidate_indices": sources,
            "dino_cost_definition": DINO_COST_DEFINITION,
            "dino_checkpoint_sha256": DINO_CHECKPOINT_SHA256,
        }
    )
    return decision


def _payload() -> dict[str, object]:
    results: list[dict[str, object]] = []
    initial_distance = 1.2
    for scene_index in range(24):
        for policy in EXPECTED_POLICIES:
            progress, success, efficiency, regret = POLICY_VALUES[policy]
            results.append(
                {
                    "policy": policy,
                    "scene_id": f"development-scene-{scene_index:02d}",
                    "goal_object_id": "blue_beacon",
                    "initial_distance_m": initial_distance,
                    "final_distance_m": initial_distance - progress,
                    "progress_m": progress,
                    "path_length_m": 1.0,
                    "path_efficiency": efficiency,
                    "success": success,
                    "fell": False,
                    "blocks_executed": 1,
                    "primitive_sequence": ["forward_fast"],
                    "mean_plan_cost": 0.5,
                    "decision_log": [_decision(policy, regret)],
                }
            )
    return {
        "schema": "lewm_closed_loop_mpc_benchmark_v0",
        "checkpoint": "/tmp/development-only-unused-lewm.pt",
        "model_config": {},
        "scene_corpus": (
            "/home/andrewknowles/Workspace/LeWMQuad-v3/.generated/scene_corpus/"
            "go2_generalization_v4"
        ),
        "split": "development",
        "family": "go2_deployment_medium_maze",
        "scene_count": 24,
        "trials_per_scene": 1,
        "backend": "cpu",
        "mode": "kinematic",
        "model_device": "cpu",
        "policy_device": "cpu",
        "task": "visible-beacon",
        "seed": 7,
        "horizon": 1,
        "candidate_count": 9,
        "max_candidates": None,
        "candidate_sequences": [list(value) for value in EXPECTED_CANDIDATE_SEQUENCES],
        "planning_grid": {"cell_size_m": 0.05, "inflation_m": 0.20},
        "oracle_assay": {
            "name": "privileged_kinematic_endpoint_distance",
            "version": 1,
            "mode": "kinematic",
            "tie_tolerance_m": 1.0e-9,
            "tie_break": "lowest_candidate_index_within_tolerance",
            "candidate_bank": {
                "count": 9,
                "max_candidates": None,
                "ordered_sequences": [
                    list(value) for value in EXPECTED_CANDIDATE_SEQUENCES
                ],
            },
        },
        "dino_assay": {
            "name": "frozen_dinov2_true_successor_goal_cost",
            "version": 1,
            "encoder_name": DINO_ENCODER_NAME,
            "repository_path": DINO_REPOSITORY_PATH,
            "repository_commit": DINO_REPOSITORY_COMMIT,
            "checkpoint_path": DINO_CHECKPOINT_PATH,
            "checkpoint_bytes": DINO_CHECKPOINT_BYTES,
            "checkpoint_sha256": DINO_CHECKPOINT_SHA256,
            "device": "cuda:1",
            "torch_version": "2.9.1+rocm7.2",
            "hip_version": "7.2.0",
            "device_name": "AMD Radeon AI PRO R9700",
            "frozen": True,
            "eval_mode": True,
            "no_grad": True,
            "feature_cache_written": False,
            "input_rgb_shape": list(DINO_INPUT_RGB_SHAPE),
            "imagenet_mean": list(DINO_IMAGENET_MEAN),
            "imagenet_std": list(DINO_IMAGENET_STD),
            "patch_output_shape": list(DINO_PATCH_OUTPUT_SHAPE),
            "token_normalization": DINO_TOKEN_NORMALIZATION,
            "cost_definition": DINO_COST_DEFINITION,
            "goal_view_count": 1,
            "successor_evaluation": DINO_SUCCESSOR_EVALUATION,
        },
        "primitive_names": list(EXPECTED_PRIMITIVES),
        "policies": list(EXPECTED_POLICIES),
        "max_blocks": 12,
        "goal_radius_m": 0.35,
        "goal_standoff_m": 0.85,
        "min_initial_distance_m": 1.5,
        "beacon_approach_distance_m": 1.2,
        "beacon_start_yaw_jitter_rad": 0.7,
        "aggregate": {},
        "results": results,
        "skipped": [],
    }


def test_pass_case_applies_registered_whole_scene_gate() -> None:
    report = analyze_payload(_payload())

    assert report["validation"]["complete_24_scene_seven_arm_pairing"] is True
    assert report["bootstrap"]["draws"] == 10_000
    assert report["bootstrap"]["seed"] == 2_026_080_402
    assert report["preregistered_gate"]["passes_all"] is True
    assert report["preregistered_gate"]["verdict"] == (
        "PASS_DINO_TARGET_COST_EARNS_PREDICTOR_TRAINING"
    )
    shuffled = report["paired_scene_comparisons"][
        "dino_true_successor_vs_dino_true_successor_shuffled"
    ]["metrics"]
    assert shuffled["progress_m"]["true_minus_comparator_mean"] == pytest.approx(0.30)
    assert shuffled["scene_mean_oracle_first_action_regret_m"][
        "true_minus_comparator_mean"
    ] == pytest.approx(-0.04)
    assert report["aggregate"]["oracle_mpc"][
        "scene_mean_oracle_first_action_regret_m_max"
    ] == pytest.approx(0.5e-9)


def test_action_regret_is_averaged_within_scene_before_bootstrap() -> None:
    payload = _payload()
    for row in payload["results"]:
        if row["policy"] == "dino_true_successor":
            row["decision_log"] = [
                _decision("dino_true_successor", 0.0),
                _decision("dino_true_successor", 0.02),
            ]
            row["blocks_executed"] = 2
    report = analyze_payload(payload)
    assert report["aggregate"]["dino_true_successor"][
        "scene_mean_oracle_first_action_regret_m_mean"
    ] == pytest.approx(0.01)


def test_scientific_fail_is_reported_without_relaxing_fixed_thresholds() -> None:
    payload = _payload()
    for row in payload["results"]:
        if row["policy"] == "dino_true_successor":
            row["progress_m"] = 0.55
            row["final_distance_m"] = 0.65
    report = analyze_payload(payload)
    gate = report["preregistered_gate"]
    assert gate["passes_all"] is False
    assert gate["verdict"] == "FAIL_STOP_FROZEN_DINO_SAME_PATCH_COST_ROUTE"
    assert gate["criteria"]["true_vs_shuffled_progress_mean_advantage_m"] == {
        "value": pytest.approx(0.05),
        "operator": ">=",
        "threshold": 0.10,
        "passed": False,
    }


def test_tolerance_sized_oracle_tie_is_not_a_disagreement() -> None:
    report = analyze_payload(_payload())
    assert report["preregistered_gate"]["criteria"][
        "oracle_decision_regret_max_within_tolerance"
    ]["passed"] is True

    payload = _payload()
    oracle = next(row for row in payload["results"] if row["policy"] == "oracle_mpc")
    oracle["decision_log"][0]["oracle_first_action_disagreement"] = True
    with pytest.raises(DinoCeilingAnalysisError, match="not tie-aware"):
        analyze_payload(payload)


def test_oracle_gate_uses_max_decision_not_only_scene_mean() -> None:
    payload = _payload()
    for row in payload["results"]:
        if row["policy"] == "oracle_mpc":
            row["decision_log"] = [
                _decision("oracle_mpc", 0.0),
                _decision("oracle_mpc", 1.5e-9),
            ]
            row["blocks_executed"] = 2
    report = analyze_payload(payload)
    assert report["aggregate"]["oracle_mpc"][
        "scene_mean_oracle_first_action_regret_m_max"
    ] == pytest.approx(0.75e-9)
    assert report["preregistered_gate"]["criteria"][
        "oracle_decision_regret_max_within_tolerance"
    ]["passed"] is False


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda value: value["results"].pop(), "exactly 24 complete"),
        (lambda value: value["skipped"].append({"scene": "x"}), "no skipped"),
        (
            lambda value: value["dino_assay"].update({"checkpoint_bytes": 1}),
            "checkpoint_bytes",
        ),
        (
            lambda value: value["results"][1]["decision_log"][0].update(
                {"policy_candidate_scores": [float("nan")] * 9}
            ),
            "must be finite",
        ),
    ],
)
def test_invalid_or_incomplete_panels_are_rejected(mutation, message: str) -> None:
    payload = _payload()
    mutation(payload)
    with pytest.raises(DinoCeilingAnalysisError, match=message):
        analyze_payload(payload)


def test_shuffled_scores_must_be_a_real_permutation() -> None:
    payload = _payload()
    shuffled = next(
        row
        for row in payload["results"]
        if row["policy"] == "dino_true_successor_shuffled"
    )
    decision = shuffled["decision_log"][0]
    decision["score_source_candidate_indices"] = list(range(9))
    decision["policy_candidate_scores"] = decision["unshuffled_dino_candidate_costs"]
    decision["selected_candidate_index"] = 3
    decision["selected_policy_score"] = 0.5
    decision["selected_score_source_candidate_index"] = 3
    with pytest.raises(DinoCeilingAnalysisError, match="registered deterministic"):
        analyze_payload(payload)


def test_cli_writes_bound_report_without_overwrite(tmp_path: Path) -> None:
    input_path = tmp_path / "benchmark.json"
    output_path = tmp_path / "analysis.json"
    input_path.write_text(json.dumps(_payload()), encoding="utf-8")

    assert main(["--input", str(input_path), "--output", str(output_path)]) == 0
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["input_binding"]["byte_count"] == input_path.stat().st_size
    assert written["preregistered_gate"]["passes_all"] is True
    with pytest.raises(DinoCeilingAnalysisError, match="overwrite"):
        main(["--input", str(input_path), "--output", str(output_path)])


def test_sealed_path_guards_fire_before_input_read(tmp_path: Path) -> None:
    sealed_input = tmp_path / "sealed_test.json"
    sealed_input.write_text("{}", encoding="utf-8")
    with pytest.raises(DinoCeilingAnalysisError, match="sealed input"):
        analyze_file(sealed_input)

    input_path = tmp_path / "benchmark.json"
    input_path.write_text(json.dumps(_payload()), encoding="utf-8")
    with pytest.raises(DinoCeilingAnalysisError, match="sealed output"):
        main(
            [
                "--input",
                str(input_path),
                "--output",
                str(tmp_path / "sealed_analysis" / "result.json"),
            ]
        )


def test_payload_copy_is_not_mutated() -> None:
    payload = _payload()
    before = copy.deepcopy(payload)
    analyze_payload(payload)
    assert payload == before
