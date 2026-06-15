from __future__ import annotations

import pytest

from lewm.benchmarks.phase2d_statistics import (
    aggregate_source_state_records,
    cluster_aware_power_from_bootstrap,
    paired_cell_estimand_records,
    paired_hierarchical_bootstrap,
    select_registered_checkpoint,
)


def _source_records(offset: float = 0.0) -> list[dict]:
    return [
        {
            "seed": seed,
            "scene_id": f"scene_{scene}",
            "source_index": source,
            "metric": offset + seed * 0.01 + scene + source * 0.1,
        }
        for seed in (1, 2)
        for scene in (0, 1)
        for source in (0, 1)
    ]


def test_candidate_rows_are_aggregated_to_source_state_units() -> None:
    records = [
        {"seed": 1, "scene_id": "scene", "source_index": 2, "metric": 1.0},
        {"seed": 1, "scene_id": "scene", "source_index": 2, "metric": 3.0},
    ]

    result = aggregate_source_state_records(records, value_keys=("metric",))

    assert result == [
        {
            "seed": 1,
            "scene_id": "scene",
            "source_index": 2,
            "candidate_rows": 2,
            "metric": 2.0,
        }
    ]


def test_paired_estimands_require_exact_source_state_matching() -> None:
    left = _source_records(offset=1.0)
    right = _source_records(offset=0.5)

    differences = paired_cell_estimand_records(
        left,
        right,
        value_key="metric",
        operation="difference",
    )
    ratios = paired_cell_estimand_records(
        left,
        right,
        value_key="metric",
        operation="ratio",
    )

    assert all(record["value"] == pytest.approx(0.5) for record in differences)
    assert all(record["value"] > 1.0 for record in ratios)
    with pytest.raises(ValueError, match="keys differ"):
        paired_cell_estimand_records(
            left[:-1],
            right,
            value_key="metric",
            operation="difference",
        )


def test_hierarchical_bootstrap_is_deterministic_and_keeps_seed_weighting() -> None:
    records = [
        {
            "seed": row["seed"],
            "scene_id": row["scene_id"],
            "source_index": row["source_index"],
            "value": row["metric"],
        }
        for row in _source_records()
    ]

    left = paired_hierarchical_bootstrap(records, samples=200, seed=17)
    right = paired_hierarchical_bootstrap(records, samples=200, seed=17)

    assert left == right
    assert left["optimization_seed_count"] == 2
    assert left["scene_count_by_seed"] == {"1": 2, "2": 2}
    assert not left["candidate_rows_bootstrapped_independently"]
    assert left["confidence_interval"][0] <= left["point_estimate"]
    assert left["confidence_interval"][1] >= left["point_estimate"]


def test_cluster_aware_power_uses_registered_direction() -> None:
    strong = cluster_aware_power_from_bootstrap(
        bootstrap_standard_error=0.02,
        true_effect=0.10,
        null_threshold=0.0,
        alternative="greater",
    )
    weak = cluster_aware_power_from_bootstrap(
        bootstrap_standard_error=0.2,
        true_effect=0.90,
        null_threshold=1.0,
        alternative="less",
    )

    assert strong["passes_registered_80pct_power"]
    assert not weak["passes_registered_80pct_power"]


def test_registered_checkpoint_rule_rejects_instability_then_breaks_ties() -> None:
    records = [
        {
            "epoch": 1,
            "stability_pass": False,
            "hard_negative_action_advantage": 1.0,
            "one_step_rollout_persistence_ratio": 0.1,
        },
        {
            "epoch": 2,
            "stability_pass": True,
            "hard_negative_action_advantage": 0.20,
            "one_step_rollout_persistence_ratio": 0.8,
        },
        {
            "epoch": 3,
            "stability_pass": True,
            "hard_negative_action_advantage": 0.195,
            "one_step_rollout_persistence_ratio": 0.7,
        },
        {
            "epoch": 4,
            "stability_pass": True,
            "hard_negative_action_advantage": 0.18,
            "one_step_rollout_persistence_ratio": 0.2,
        },
    ]

    selected = select_registered_checkpoint(records)

    assert selected["epoch"] == 3
