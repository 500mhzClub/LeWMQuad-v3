from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.analyze_go2_planner_oracle_assay_v1 import (
    AssayAnalysisError,
    EXPECTED_POLICIES,
    analyze_file,
    analyze_payload,
    main,
)


def _decision(
    *,
    selected: int,
    source: int,
    regret: float,
    tolerance: float = 1.0e-9,
    optimal: tuple[str, ...] = ("forward_fast",),
) -> dict[str, object]:
    return {
        "selected_candidate_index": selected,
        "selected_score_source_candidate_index": source,
        "oracle_first_action_regret_m": regret,
        "oracle_first_action_disagreement": regret > tolerance,
        "oracle_cost_tie_tolerance_m": tolerance,
        "oracle_optimal_candidate_count": len(optimal),
        "oracle_optimal_first_primitives": list(optimal),
    }


def _payload(*, horizon: int = 1, scene_count: int = 2) -> dict[str, object]:
    results: list[dict[str, object]] = []
    values = {
        "oracle_mpc": (0.2, 1.0, True, 0.8),
        "oracle_shuffled": (0.6, 0.6, False, 0.4),
        "bearing": (0.1, 1.1, True, 0.9),
        "hold": (1.2, 0.0, False, 0.0),
        "random": (0.8, 0.4, False, 0.3),
    }
    for scene_index in range(scene_count):
        for policy in EXPECTED_POLICIES:
            final_distance, progress, success, efficiency = values[policy]
            if policy == "oracle_mpc":
                log = [_decision(selected=3, source=3, regret=0.0)]
            elif policy == "oracle_shuffled":
                log = [_decision(selected=8, source=3, regret=0.2)]
            else:
                log = []
            results.append(
                {
                    "policy": policy,
                    "scene_id": f"development-scene-{scene_index}",
                    "goal_object_id": "blue_beacon",
                    "initial_distance_m": 1.2,
                    "final_distance_m": final_distance + 0.1 * scene_index,
                    "progress_m": progress,
                    "success": success,
                    "path_efficiency": efficiency,
                    "blocks_executed": len(log) if policy.startswith("oracle_") else 1,
                    "decision_log": log,
                }
            )
    return {
        "schema": "lewm_closed_loop_mpc_benchmark_v0",
        "mode": "kinematic",
        "horizon": horizon,
        "scene_count": scene_count,
        "trials_per_scene": 1,
        "policies": list(EXPECTED_POLICIES),
        "skipped": [],
        "oracle_assay": {"mode": "kinematic", "tie_tolerance_m": 1.0e-9},
        "results": results,
    }


def test_analyzer_computes_aggregate_paired_bootstrap_and_tie_audit() -> None:
    report = analyze_payload(_payload(), bootstrap_draws=200, bootstrap_seed=17)

    assert report["validation"] == {
        "complete_per_scene_pairing": True,
        "no_skipped_scenes_or_trials": True,
        "oracle_mpc_tie_aware_optimality": True,
        "oracle_shuffled_nonidentity_and_disagreement": True,
    }
    assert report["aggregate"]["oracle_mpc"]["success_rate"] == 1.0
    comparison = report["paired_scene_comparisons"][
        "oracle_mpc_vs_oracle_shuffled"
    ]
    assert comparison["metrics"]["final_distance_m"]["oracle_advantage"] == pytest.approx(0.4)
    assert comparison["metrics"]["progress_m"]["oracle_advantage"] == pytest.approx(0.4)
    assert comparison["metrics"]["success"]["oracle_advantage"] == 1.0
    assert report["decision_audit"]["oracle_mpc"]["tie_aware_disagreement_count"] == 0
    assert report["decision_audit"]["oracle_shuffled"]["tie_aware_disagreement_count"] == 2
    assert report["scope"]["safety_evidence"] == "not_evaluated_in_kinematic_assay"
    assert report["paired_scene_comparisons"]["oracle_mpc_vs_bearing"][
        "comparator_role"
    ] == "saturated_reference_ceiling_not_a_superiority_gate"


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda value: value["results"].pop(), "not policy-complete"),
        (lambda value: value["skipped"].append({"scene": "x"}), "skipped"),
        (
            lambda value: value["results"][0]["decision_log"][0].update(
                {
                    "oracle_first_action_regret_m": 0.1,
                    "oracle_first_action_disagreement": True,
                }
            ),
            "oracle_mpc has positive",
        ),
    ],
)
def test_analyzer_rejects_incomplete_or_invalid_assays(mutation, message: str) -> None:
    payload = _payload()
    mutation(payload)
    with pytest.raises(AssayAnalysisError, match=message):
        analyze_payload(payload, bootstrap_draws=10)


def test_analyzer_rejects_identity_or_tie_only_shuffled_intervention() -> None:
    payload = _payload()
    for row in payload["results"]:
        if row["policy"] == "oracle_shuffled":
            row["decision_log"] = [_decision(selected=3, source=3, regret=0.0)]
    with pytest.raises(AssayAnalysisError, match="nonidentity"):
        analyze_payload(payload, bootstrap_draws=10)


def test_horizon_override_and_cli_output(tmp_path: Path) -> None:
    input_path = tmp_path / "assay.json"
    output_path = tmp_path / "analysis.json"
    input_path.write_text(json.dumps(_payload(horizon=2)), encoding="utf-8")

    with pytest.raises(AssayAnalysisError, match="expected horizon 1"):
        analyze_file(input_path, bootstrap_draws=10)

    assert main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--expected-horizon",
            "2",
            "--bootstrap-draws",
            "10",
        ]
    ) == 0
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["expected_horizon"] == 2
    assert written["bootstrap"]["seed"] == 2_026_080_401


def test_tie_aware_disagreement_uses_regret_not_primitive_name() -> None:
    payload = copy.deepcopy(_payload())
    for row in payload["results"]:
        if row["policy"] == "oracle_mpc":
            row["decision_log"] = [
                _decision(
                    selected=4,
                    source=4,
                    regret=5.0e-10,
                    optimal=("forward_fast", "arc_left"),
                )
            ]
    report = analyze_payload(payload, bootstrap_draws=10)
    audit = report["decision_audit"]["oracle_mpc"]
    assert audit["tie_aware_disagreement_count"] == 0
    assert audit["multi_first_primitive_tie_decision_count"] == 2


def test_preregistered_h1_gate_passes_only_the_fixed_criteria() -> None:
    report = analyze_payload(
        _payload(scene_count=24), bootstrap_draws=200, bootstrap_seed=17
    )
    gate = report["preregistered_h1_gate"]
    assert gate["applicable"] is True
    assert gate["passes_all"] is True
    assert all(item["passed"] for item in gate["criteria"].values())
    assert "bearing_ceiling_comparison" in gate["excluded_from_gate"]
    assert "fall_or_safety_metrics_in_kinematic_mode" in gate["excluded_from_gate"]


def test_preregistered_h1_gate_reports_a_scientific_failure() -> None:
    payload = _payload(scene_count=24)
    for row in payload["results"]:
        if row["policy"] == "oracle_shuffled":
            row["decision_log"] = [_decision(selected=8, source=3, regret=0.01)]
    report = analyze_payload(payload, bootstrap_draws=50, bootstrap_seed=17)
    gate = report["preregistered_h1_gate"]
    assert gate["passes_all"] is False
    assert gate["criteria"]["shuffled_mean_regret_m"]["passed"] is False
