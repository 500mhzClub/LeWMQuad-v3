from __future__ import annotations

import pytest

from lewm.benchmarks.go2_categorical_radial_n32_v2 import (
    PER_SEED_DECISION_SCHEMA,
    STAGE_NAME,
    per_seed_decision,
)


def test_failed_fit_forbids_holdouts() -> None:
    decision = per_seed_decision(
        {"terminal_fit_gate": {"passes": False}}, None
    )
    assert decision["schema"] == PER_SEED_DECISION_SCHEMA
    assert decision["classification"] == "fit_gate_failed"
    assert decision["qualifying_optimizer_stage"] is None
    assert decision["favorable"] is False
    with pytest.raises(ValueError, match="forbidden"):
        per_seed_decision(
            {"terminal_fit_gate": {"passes": False}},
            {"same_scene_holdout": {}, "cross_scene_holdout": {}},
        )


def test_fit_pass_requires_both_strict_holdout_decisions() -> None:
    stage = {"terminal_fit_gate": {"passes": True}}
    with pytest.raises(ValueError, match="mandatory"):
        per_seed_decision(stage, None)
    decision = per_seed_decision(
        stage,
        {
            "same_scene_holdout": {"passes": True},
            "cross_scene_holdout": {"passes": True},
        },
    )
    assert decision["classification"] == "favorable"
    assert decision["qualifying_optimizer_stage"] == STAGE_NAME
    assert decision["categorical_radial_full_train_candidate_licensed"] is False


def test_one_failed_holdout_is_not_favorable() -> None:
    decision = per_seed_decision(
        {"terminal_fit_gate": {"passes": True}},
        {
            "same_scene_holdout": {"passes": True},
            "cross_scene_holdout": {"passes": False},
        },
    )
    assert decision["classification"] == "fit_pass_holdout_gate_failed"
    assert decision["favorable"] is False
