#!/usr/bin/env python3
"""Validate and summarize one closed-loop planner-oracle assay result.

This analyzer reads only the supplied benchmark JSON.  It does not discover
scene inputs, load a checkpoint, or inspect any benchmark corpus.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "lewm_go2_planner_oracle_assay_analysis_v1"
INPUT_SCHEMA = "lewm_closed_loop_mpc_benchmark_v0"
EXPECTED_POLICIES = (
    "oracle_mpc",
    "oracle_shuffled",
    "bearing",
    "hold",
    "random",
)
METRIC_DIRECTIONS = {
    "final_distance_m": "lower_is_better",
    "progress_m": "higher_is_better",
    "success": "higher_is_better",
    "path_efficiency": "higher_is_better",
}
DEFAULT_BOOTSTRAP_DRAWS = 10_000
DEFAULT_BOOTSTRAP_SEED = 2_026_080_401
DEFAULT_ORACLE_TIE_TOLERANCE_M = 1.0e-9


class AssayAnalysisError(ValueError):
    """Raised when an assay result is incomplete or internally inconsistent."""


def _is_sealed_path(path: Path) -> bool:
    return path.name == "sealed_test.json" or any(
        part == "sealed" or part.startswith("sealed_") for part in path.parts
    )


def _safe_input_path(path: Path) -> Path:
    if _is_sealed_path(path):
        raise AssayAnalysisError("refusing sealed input path")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise AssayAnalysisError(f"input does not exist: {path}") from exc
    if _is_sealed_path(resolved) or not resolved.is_file():
        raise AssayAnalysisError("input must resolve to a non-sealed regular file")
    return resolved


def _safe_output_path(path: Path) -> Path:
    if _is_sealed_path(path):
        raise AssayAnalysisError("refusing sealed output path")
    resolved = path.resolve(strict=False)
    if _is_sealed_path(resolved):
        raise AssayAnalysisError("refusing sealed output path")
    return resolved


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AssayAnalysisError(f"{name} must be an object")
    return value


def _sequence(value: object, *, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise AssayAnalysisError(f"{name} must be an array")
    return value


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssayAnalysisError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise AssayAnalysisError(f"{name} must be finite")
    return result


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AssayAnalysisError(f"{name} must be an integer >= {minimum}")
    return value


def _metric(row: Mapping[str, Any], metric: str, *, name: str) -> float:
    value = row.get(metric)
    if metric == "success":
        if not isinstance(value, bool):
            raise AssayAnalysisError(f"{name}.success must be boolean")
        return float(value)
    return _finite(value, name=f"{name}.{metric}")


def _decision_tolerance(
    decision: Mapping[str, Any], *, fallback: float, name: str
) -> float:
    raw = decision.get("oracle_cost_tie_tolerance_m", fallback)
    tolerance = _finite(raw, name=f"{name}.oracle_cost_tie_tolerance_m")
    if tolerance < 0.0:
        raise AssayAnalysisError(f"{name} tie tolerance must be non-negative")
    return tolerance


def _validate_decisions(
    paired: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    fallback_tolerance: float,
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for policy in ("oracle_mpc", "oracle_shuffled"):
        decision_count = 0
        nonidentity_count = 0
        disagreement_count = 0
        regrets: list[float] = []
        tie_count = 0
        multi_primitive_tie_count = 0
        tie_metadata_count = 0
        optimal_sets: Counter[tuple[str, ...]] = Counter()
        tolerances: list[float] = []

        for scene_id in sorted(paired):
            row = paired[scene_id][policy]
            blocks = _integer(
                row.get("blocks_executed"),
                name=f"{scene_id}.{policy}.blocks_executed",
            )
            log = _sequence(
                row.get("decision_log"), name=f"{scene_id}.{policy}.decision_log"
            )
            if len(log) != blocks:
                raise AssayAnalysisError(
                    f"{scene_id}.{policy} decision-log length does not match blocks"
                )
            if not log:
                raise AssayAnalysisError(
                    f"{scene_id}.{policy} has no auditable planner decision"
                )

            for index, raw_decision in enumerate(log):
                name = f"{scene_id}.{policy}.decision_log[{index}]"
                decision = _mapping(raw_decision, name=name)
                tolerance = _decision_tolerance(
                    decision, fallback=fallback_tolerance, name=name
                )
                regret = _finite(
                    decision.get("oracle_first_action_regret_m"),
                    name=f"{name}.oracle_first_action_regret_m",
                )
                if regret < -tolerance:
                    raise AssayAnalysisError(f"{name} has negative oracle regret")
                regret = max(0.0, regret)
                derived_disagreement = regret > tolerance
                logged_disagreement = decision.get("oracle_first_action_disagreement")
                if not isinstance(logged_disagreement, bool):
                    raise AssayAnalysisError(
                        f"{name}.oracle_first_action_disagreement must be boolean"
                    )
                if logged_disagreement != derived_disagreement:
                    raise AssayAnalysisError(
                        f"{name} disagreement is not tie-aware regret > tolerance"
                    )

                selected = decision.get("selected_candidate_index")
                source = decision.get("selected_score_source_candidate_index")
                if policy == "oracle_shuffled":
                    selected_i = _integer(
                        selected, name=f"{name}.selected_candidate_index"
                    )
                    source_i = _integer(
                        source,
                        name=f"{name}.selected_score_source_candidate_index",
                    )
                    nonidentity_count += int(selected_i != source_i)

                optimal_raw = decision.get("oracle_optimal_first_primitives")
                count_raw = decision.get("oracle_optimal_candidate_count")
                if optimal_raw is not None or count_raw is not None:
                    optimal = _sequence(
                        optimal_raw, name=f"{name}.oracle_optimal_first_primitives"
                    )
                    if not optimal or any(
                        not isinstance(item, str) or not item for item in optimal
                    ):
                        raise AssayAnalysisError(
                            f"{name}.oracle_optimal_first_primitives is malformed"
                        )
                    optimal_tuple = tuple(dict.fromkeys(optimal))
                    if len(optimal_tuple) != len(optimal):
                        raise AssayAnalysisError(
                            f"{name}.oracle_optimal_first_primitives has duplicates"
                        )
                    optimal_count = _integer(
                        count_raw,
                        name=f"{name}.oracle_optimal_candidate_count",
                        minimum=1,
                    )
                    tie_metadata_count += 1
                    tie_count += int(optimal_count > 1)
                    multi_primitive_tie_count += int(len(optimal_tuple) > 1)
                    optimal_sets[optimal_tuple] += 1

                decision_count += 1
                disagreement_count += int(derived_disagreement)
                regrets.append(regret)
                tolerances.append(tolerance)

        if policy == "oracle_mpc" and disagreement_count:
            raise AssayAnalysisError(
                "oracle_mpc has positive tie-aware first-action regret"
            )
        if policy == "oracle_shuffled":
            if nonidentity_count == 0:
                raise AssayAnalysisError(
                    "oracle_shuffled never used a nonidentity score source"
                )
            if disagreement_count == 0:
                raise AssayAnalysisError(
                    "oracle_shuffled never disagreed with an oracle-optimal first action"
                )

        summaries[policy] = {
            "decision_count": decision_count,
            "first_action_regret_mean_m": float(np.mean(regrets)),
            "first_action_regret_max_m": float(np.max(regrets)),
            "tie_tolerance_m_min": float(np.min(tolerances)),
            "tie_tolerance_m_max": float(np.max(tolerances)),
            "tie_aware_disagreement_count": disagreement_count,
            "tie_aware_disagreement_rate": disagreement_count / decision_count,
            "nonidentity_selected_score_source_count": (
                nonidentity_count if policy == "oracle_shuffled" else None
            ),
            "nonidentity_selected_score_source_rate": (
                nonidentity_count / decision_count
                if policy == "oracle_shuffled"
                else None
            ),
            "tie_metadata_decision_count": tie_metadata_count,
            "candidate_tie_decision_count": tie_count,
            "multi_first_primitive_tie_decision_count": multi_primitive_tie_count,
            "optimal_first_primitive_sets": [
                {"primitives": list(primitives), "decision_count": count}
                for primitives, count in sorted(optimal_sets.items())
            ],
        }
    return summaries


def _validate_and_pair(
    payload: Mapping[str, Any], *, expected_horizon: int
) -> tuple[dict[str, dict[str, Mapping[str, Any]]], float]:
    if payload.get("schema") != INPUT_SCHEMA:
        raise AssayAnalysisError("unexpected benchmark schema")
    if payload.get("mode") != "kinematic":
        raise AssayAnalysisError(
            "this assay requires kinematic mode; physical mode is not an exact outcome oracle"
        )
    if _integer(payload.get("horizon"), name="horizon", minimum=1) != expected_horizon:
        raise AssayAnalysisError(
            f"expected horizon {expected_horizon}, got {payload.get('horizon')!r}"
        )
    if _integer(
        payload.get("trials_per_scene"), name="trials_per_scene", minimum=1
    ) != 1:
        raise AssayAnalysisError("per-scene pairing requires trials_per_scene == 1")

    policies = _sequence(payload.get("policies"), name="policies")
    if len(policies) != len(set(policies)) or set(policies) != set(EXPECTED_POLICIES):
        raise AssayAnalysisError(
            f"policies must be exactly {list(EXPECTED_POLICIES)!r}"
        )
    skipped = _sequence(payload.get("skipped"), name="skipped")
    if skipped:
        raise AssayAnalysisError("assay contains skipped scenes/trials")

    provenance_raw = payload.get("oracle_assay")
    if provenance_raw is None:
        fallback_tolerance = DEFAULT_ORACLE_TIE_TOLERANCE_M
    else:
        provenance = _mapping(provenance_raw, name="oracle_assay")
        if provenance.get("mode") != "kinematic":
            raise AssayAnalysisError("oracle provenance does not certify kinematic mode")
        fallback_tolerance = _finite(
            provenance.get("tie_tolerance_m", DEFAULT_ORACLE_TIE_TOLERANCE_M),
            name="oracle_assay.tie_tolerance_m",
        )
    if fallback_tolerance < 0.0:
        raise AssayAnalysisError("oracle assay tie tolerance must be non-negative")

    results = _sequence(payload.get("results"), name="results")
    paired: dict[str, dict[str, Mapping[str, Any]]] = {}
    for index, raw_row in enumerate(results):
        row = _mapping(raw_row, name=f"results[{index}]")
        scene_id = row.get("scene_id")
        policy = row.get("policy")
        if not isinstance(scene_id, str) or not scene_id:
            raise AssayAnalysisError(f"results[{index}].scene_id is malformed")
        if policy not in EXPECTED_POLICIES:
            raise AssayAnalysisError(f"results[{index}] has unexpected policy")
        scene_rows = paired.setdefault(scene_id, {})
        if policy in scene_rows:
            raise AssayAnalysisError(f"duplicate {policy} result for scene {scene_id}")
        for metric in METRIC_DIRECTIONS:
            _metric(row, metric, name=f"results[{index}]")
        scene_rows[str(policy)] = row

    expected_scene_count = _integer(
        payload.get("scene_count"), name="scene_count", minimum=1
    )
    if len(paired) != expected_scene_count:
        raise AssayAnalysisError("scene_count does not match complete result scenes")
    for scene_id, rows in paired.items():
        if set(rows) != set(EXPECTED_POLICIES):
            raise AssayAnalysisError(f"scene {scene_id} is not policy-complete")
        goals = {rows[policy].get("goal_object_id") for policy in EXPECTED_POLICIES}
        if len(goals) != 1 or not all(isinstance(goal, str) and goal for goal in goals):
            raise AssayAnalysisError(f"scene {scene_id} policies do not share one goal")
        starts = [
            _finite(
                rows[policy].get("initial_distance_m"),
                name=f"{scene_id}.{policy}.initial_distance_m",
            )
            for policy in EXPECTED_POLICIES
        ]
        if max(starts) - min(starts) > 1.0e-6:
            raise AssayAnalysisError(f"scene {scene_id} policies do not share one start")
    return paired, fallback_tolerance


def _aggregate(
    paired: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for policy in EXPECTED_POLICIES:
        rows = [paired[scene][policy] for scene in sorted(paired)]
        result[policy] = {
            "scene_count": len(rows),
            "final_distance_m_mean": float(
                np.mean([_metric(row, "final_distance_m", name=policy) for row in rows])
            ),
            "progress_m_mean": float(
                np.mean([_metric(row, "progress_m", name=policy) for row in rows])
            ),
            "success_count": int(sum(bool(row["success"]) for row in rows)),
            "success_rate": float(np.mean([float(row["success"]) for row in rows])),
            "path_efficiency_mean": float(
                np.mean([_metric(row, "path_efficiency", name=policy) for row in rows])
            ),
        }
    return result


def _paired_comparisons(
    paired: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    scenes = sorted(paired)
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, len(scenes), size=(bootstrap_draws, len(scenes)))
    comparisons: dict[str, Any] = {}
    for comparator in ("oracle_shuffled", "hold", "random", "bearing"):
        metric_summaries: dict[str, Any] = {}
        per_scene: list[dict[str, Any]] = [
            {"scene_id": scene_id} for scene_id in scenes
        ]
        for metric, direction in METRIC_DIRECTIONS.items():
            oracle_values = np.asarray(
                [_metric(paired[s]["oracle_mpc"], metric, name=s) for s in scenes]
            )
            comparator_values = np.asarray(
                [_metric(paired[s][comparator], metric, name=s) for s in scenes]
            )
            raw_delta = oracle_values - comparator_values
            advantage = -raw_delta if direction == "lower_is_better" else raw_delta
            boot = advantage[indices].mean(axis=1)
            lower, upper = np.percentile(boot, [2.5, 97.5])
            metric_summaries[metric] = {
                "direction": direction,
                "oracle_mean": float(np.mean(oracle_values)),
                "comparator_mean": float(np.mean(comparator_values)),
                "oracle_minus_comparator": float(np.mean(raw_delta)),
                "oracle_advantage": float(np.mean(advantage)),
                "oracle_advantage_bootstrap_lower_95": float(lower),
                "oracle_advantage_bootstrap_upper_95": float(upper),
            }
            for row, raw, favored in zip(
                per_scene, raw_delta.tolist(), advantage.tolist(), strict=True
            ):
                row[metric] = {
                    "oracle_minus_comparator": float(raw),
                    "oracle_advantage": float(favored),
                }
        comparisons[f"oracle_mpc_vs_{comparator}"] = {
            "comparator_role": (
                "saturated_reference_ceiling_not_a_superiority_gate"
                if comparator == "bearing"
                else "control"
            ),
            "metrics": metric_summaries,
            "per_scene": per_scene,
        }
    return comparisons


def _criterion(*, value: object, operator: str, threshold: object, passed: bool) -> dict[str, Any]:
    return {
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "passed": bool(passed),
    }


def _h1_gate(
    *,
    expected_horizon: int,
    scene_count: int,
    aggregate: Mapping[str, Any],
    comparisons: Mapping[str, Any],
    decisions: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply only the fixed, preregistered H1 planner-assay criteria."""
    if expected_horizon != 1:
        return {
            "applicable": False,
            "reason": "preregistered gate is defined only for horizon 1",
            "criteria": {},
            "passes_all": False,
        }

    oracle_audit = decisions["oracle_mpc"]
    shuffled_audit = decisions["oracle_shuffled"]
    progress = comparisons["oracle_mpc_vs_oracle_shuffled"]["metrics"]["progress_m"]
    oracle_progress = float(aggregate["oracle_mpc"]["progress_m_mean"])
    hold_progress = float(aggregate["hold"]["progress_m_mean"])
    random_progress = float(aggregate["random"]["progress_m_mean"])
    oracle_max_regret = float(oracle_audit["first_action_regret_max_m"])
    oracle_tolerance = float(oracle_audit["tie_tolerance_m_max"])
    shuffled_rate = float(shuffled_audit["tie_aware_disagreement_rate"])
    shuffled_mean_regret = float(shuffled_audit["first_action_regret_mean_m"])
    progress_advantage = float(progress["oracle_advantage"])
    progress_lower = float(progress["oracle_advantage_bootstrap_lower_95"])

    criteria = {
        "complete_24_scene_panel": _criterion(
            value=scene_count, operator="==", threshold=24, passed=scene_count == 24
        ),
        "oracle_max_regret_within_tolerance": _criterion(
            value=oracle_max_regret,
            operator="<=",
            threshold=oracle_tolerance,
            passed=oracle_max_regret <= oracle_tolerance,
        ),
        "shuffled_regret_positive_rate": _criterion(
            value=shuffled_rate,
            operator=">=",
            threshold=0.25,
            passed=shuffled_rate >= 0.25,
        ),
        "shuffled_mean_regret_m": _criterion(
            value=shuffled_mean_regret,
            operator=">=",
            threshold=0.02,
            passed=shuffled_mean_regret >= 0.02,
        ),
        "oracle_progress_advantage_vs_shuffled_m": _criterion(
            value=progress_advantage,
            operator=">=",
            threshold=0.15,
            passed=progress_advantage >= 0.15,
        ),
        "oracle_progress_advantage_vs_shuffled_bootstrap_lower_95": _criterion(
            value=progress_lower,
            operator=">",
            threshold=0.0,
            passed=progress_lower > 0.0,
        ),
        "oracle_mean_progress_beats_hold": _criterion(
            value=oracle_progress - hold_progress,
            operator=">",
            threshold=0.0,
            passed=oracle_progress > hold_progress,
        ),
        "oracle_mean_progress_beats_random": _criterion(
            value=oracle_progress - random_progress,
            operator=">",
            threshold=0.0,
            passed=oracle_progress > random_progress,
        ),
    }
    return {
        "applicable": True,
        "criteria": criteria,
        "passes_all": all(item["passed"] for item in criteria.values()),
        "excluded_from_gate": [
            "bearing_ceiling_comparison",
            "fall_or_safety_metrics_in_kinematic_mode",
        ],
    }


def analyze_payload(
    payload: Mapping[str, Any],
    *,
    input_path: str | None = None,
    expected_horizon: int = 1,
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    _integer(expected_horizon, name="expected_horizon", minimum=1)
    _integer(bootstrap_draws, name="bootstrap_draws", minimum=1)
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise AssayAnalysisError("bootstrap_seed must be an integer")
    paired, fallback_tolerance = _validate_and_pair(
        payload, expected_horizon=expected_horizon
    )
    decisions = _validate_decisions(
        paired, fallback_tolerance=fallback_tolerance
    )
    aggregate = _aggregate(paired)
    comparisons = _paired_comparisons(
        paired,
        bootstrap_draws=bootstrap_draws,
        bootstrap_seed=bootstrap_seed,
    )
    return {
        "schema": SCHEMA,
        "input": input_path,
        "input_schema": INPUT_SCHEMA,
        "scene_count": len(paired),
        "expected_horizon": expected_horizon,
        "expected_policies": list(EXPECTED_POLICIES),
        "validation": {
            "complete_per_scene_pairing": True,
            "no_skipped_scenes_or_trials": True,
            "oracle_mpc_tie_aware_optimality": True,
            "oracle_shuffled_nonidentity_and_disagreement": True,
        },
        "scope": {
            "mode": "kinematic",
            "safety_evidence": "not_evaluated_in_kinematic_assay",
            "bearing_role": "saturated_reference_ceiling_not_a_superiority_gate",
        },
        "bootstrap": {
            "unit": "whole_scene_paired_resampling",
            "draws": bootstrap_draws,
            "seed": bootstrap_seed,
            "percentiles": [2.5, 97.5],
        },
        "aggregate": aggregate,
        "paired_scene_comparisons": comparisons,
        "preregistered_h1_gate": _h1_gate(
            expected_horizon=expected_horizon,
            scene_count=len(paired),
            aggregate=aggregate,
            comparisons=comparisons,
            decisions=decisions,
        ),
        "decision_audit": decisions,
    }


def analyze_file(
    input_path: Path,
    *,
    expected_horizon: int = 1,
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    selected = _safe_input_path(input_path)
    try:
        payload = json.loads(selected.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssayAnalysisError(f"cannot read benchmark JSON: {exc}") from exc
    return analyze_payload(
        _mapping(payload, name="input"),
        input_path=str(selected),
        expected_horizon=expected_horizon,
        bootstrap_draws=bootstrap_draws,
        bootstrap_seed=bootstrap_seed,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-horizon", type=int, default=1)
    parser.add_argument("--bootstrap-draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args(argv)

    report = analyze_file(
        args.input,
        expected_horizon=args.expected_horizon,
        bootstrap_draws=args.bootstrap_draws,
        bootstrap_seed=args.bootstrap_seed,
    )
    output = _safe_output_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
