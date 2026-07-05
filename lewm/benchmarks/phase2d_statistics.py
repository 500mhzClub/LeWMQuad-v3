"""Cluster-aware estimands and checkpoint rules for Phase 2D."""
from __future__ import annotations

import math
import random
from collections import defaultdict
from statistics import NormalDist
from typing import Literal, Sequence

Operation = Literal["difference", "ratio"]
Alternative = Literal["greater", "less"]


def aggregate_source_state_records(
    records: Sequence[dict],
    *,
    value_keys: Sequence[str],
) -> list[dict]:
    """Average candidate-level values within each seed/scene/source experimental unit."""

    grouped: dict[tuple[int, str, int], list[dict]] = defaultdict(list)
    for record in records:
        key = (
            int(record["seed"]),
            str(record["scene_id"]),
            int(record["source_index"]),
        )
        grouped[key].append(record)
    result = []
    for (seed, scene_id, source_index), candidates in sorted(grouped.items()):
        result.append(
            {
                "seed": seed,
                "scene_id": scene_id,
                "source_index": source_index,
                "candidate_rows": len(candidates),
                **{
                    value_key: sum(float(row[value_key]) for row in candidates)
                    / len(candidates)
                    for value_key in value_keys
                },
            }
        )
    return result


def paired_cell_estimand_records(
    left: Sequence[dict],
    right: Sequence[dict],
    *,
    value_key: str,
    operation: Operation,
) -> list[dict]:
    """Construct exact matched source-state differences or ratios between cells."""

    def indexed(records: Sequence[dict]) -> dict[tuple[int, str, int], dict]:
        result = {}
        for record in records:
            key = (
                int(record["seed"]),
                str(record["scene_id"]),
                int(record["source_index"]),
            )
            if key in result:
                raise ValueError(f"duplicate source-state record: {key}")
            result[key] = record
        return result

    left_by_key = indexed(left)
    right_by_key = indexed(right)
    if left_by_key.keys() != right_by_key.keys():
        missing_left = sorted(right_by_key.keys() - left_by_key.keys())
        missing_right = sorted(left_by_key.keys() - right_by_key.keys())
        raise ValueError(
            "paired cell source-state keys differ; "
            f"missing_left={missing_left[:4]}, missing_right={missing_right[:4]}"
        )
    result = []
    for seed, scene_id, source_index in sorted(left_by_key):
        left_value = float(left_by_key[(seed, scene_id, source_index)][value_key])
        right_value = float(right_by_key[(seed, scene_id, source_index)][value_key])
        if not math.isfinite(left_value) or not math.isfinite(right_value):
            raise ValueError("paired estimands require finite values")
        if operation == "difference":
            value = left_value - right_value
        elif operation == "ratio":
            if right_value <= 0.0:
                raise ValueError("paired ratio denominator must be positive")
            value = left_value / right_value
        else:
            raise ValueError(f"unsupported operation: {operation}")
        result.append(
            {
                "seed": seed,
                "scene_id": scene_id,
                "source_index": source_index,
                "value": value,
                "left_value": left_value,
                "right_value": right_value,
                "operation": operation,
            }
        )
    return result


def _seed_mean(records: Sequence[dict], *, value_key: str) -> float:
    return sum(float(record[value_key]) for record in records) / len(records)


def _hierarchical_sample(
    records: Sequence[dict],
    *,
    value_key: str,
    rng: random.Random,
) -> float:
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_scene[str(record["scene_id"])].append(record)
    scenes = sorted(by_scene)
    sampled_values = []
    for sampled_scene in rng.choices(scenes, k=len(scenes)):
        source_states = by_scene[sampled_scene]
        sampled_values.extend(
            float(record[value_key])
            for record in rng.choices(source_states, k=len(source_states))
        )
    return sum(sampled_values) / len(sampled_values)


def paired_hierarchical_bootstrap(
    records: Sequence[dict],
    *,
    value_key: str = "value",
    samples: int = 10_000,
    seed: int = 20260614,
    confidence: float = 0.95,
    include_samples: bool = False,
) -> dict:
    """Bootstrap scenes then source states, averaging matched seeds equally."""

    if samples < 1:
        raise ValueError("samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    by_seed: dict[int, list[dict]] = defaultdict(list)
    unique_units = set()
    for record in records:
        key = (
            int(record["seed"]),
            str(record["scene_id"]),
            int(record["source_index"]),
        )
        if key in unique_units:
            raise ValueError(f"duplicate experimental unit: {key}")
        unique_units.add(key)
        value = float(record[value_key])
        if not math.isfinite(value):
            raise ValueError("bootstrap values must be finite")
        by_seed[key[0]].append(record)
    if not by_seed:
        raise ValueError("bootstrap requires at least one record")

    rng = random.Random(seed)
    estimates = []
    for _sample in range(samples):
        seed_estimates = [
            _hierarchical_sample(seed_records, value_key=value_key, rng=rng)
            for _seed, seed_records in sorted(by_seed.items())
        ]
        estimates.append(sum(seed_estimates) / len(seed_estimates))
    ordered = sorted(estimates)
    tail = (1.0 - confidence) / 2.0
    lower_index = min(samples - 1, max(0, int(math.floor(tail * samples))))
    upper_index = min(
        samples - 1,
        max(0, int(math.ceil((1.0 - tail) * samples)) - 1),
    )
    point_estimate = sum(
        _seed_mean(seed_records, value_key=value_key)
        for seed_records in by_seed.values()
    ) / len(by_seed)
    mean_estimate = sum(estimates) / samples
    standard_error = math.sqrt(
        sum((estimate - mean_estimate) ** 2 for estimate in estimates)
        / max(1, samples - 1)
    )
    result = {
        "schema": "jepa_phase2d_paired_hierarchical_bootstrap_v0",
        "point_estimate": point_estimate,
        "confidence": confidence,
        "confidence_interval": [ordered[lower_index], ordered[upper_index]],
        "bootstrap_standard_error": standard_error,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "optimization_seeds": sorted(by_seed),
        "optimization_seed_count": len(by_seed),
        "scene_count_by_seed": {
            str(run_seed): len({str(row["scene_id"]) for row in seed_records})
            for run_seed, seed_records in sorted(by_seed.items())
        },
        "source_state_count_by_seed": {
            str(run_seed): len(seed_records)
            for run_seed, seed_records in sorted(by_seed.items())
        },
        "experimental_unit": "source_state",
        "cluster_unit": "scene",
        "candidate_rows_bootstrapped_independently": False,
        "seed_aggregation": "equal_weight_mean_of_seed_level_estimands",
    }
    if include_samples:
        result["sampled_estimates"] = estimates
    return result


def cluster_aware_power_from_bootstrap(
    *,
    bootstrap_standard_error: float,
    true_effect: float,
    null_threshold: float,
    alternative: Alternative,
    confidence: float = 0.95,
) -> dict:
    """Approximate CI-based power using a cluster-bootstrap standard error."""

    if bootstrap_standard_error <= 0.0:
        raise ValueError("bootstrap_standard_error must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    normal = NormalDist()
    critical = normal.inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    if alternative == "greater":
        signal = (true_effect - null_threshold) / bootstrap_standard_error
    elif alternative == "less":
        signal = (null_threshold - true_effect) / bootstrap_standard_error
    else:
        raise ValueError(f"unsupported alternative: {alternative}")
    power = normal.cdf(signal - critical)
    return {
        "schema": "jepa_phase2d_cluster_aware_power_v0",
        "method": "normal_approximation_from_paired_hierarchical_bootstrap_se",
        "bootstrap_standard_error": bootstrap_standard_error,
        "true_effect": true_effect,
        "null_threshold": null_threshold,
        "alternative": alternative,
        "confidence": confidence,
        "estimated_power": power,
        "passes_registered_80pct_power": power >= 0.80,
    }


def select_registered_checkpoint(records: Sequence[dict]) -> dict:
    """Apply the preregistered validation-only lexicographic checkpoint rule."""

    eligible = [record for record in records if bool(record["stability_pass"])]
    if not eligible:
        raise ValueError("no checkpoint passes the registered stability gate")
    best_advantage = max(
        float(record["hard_negative_action_advantage"]) for record in eligible
    )
    advantage_ties = [
        record
        for record in eligible
        if float(record["hard_negative_action_advantage"]) >= best_advantage - 0.01
    ]
    return min(
        advantage_ties,
        key=lambda record: (
            float(record["one_step_rollout_persistence_ratio"]),
            int(record["epoch"]),
        ),
    )
