from __future__ import annotations

import pytest

from lewm.benchmarks.go2_categorical_radial_n32_v3 import (
    PER_SEED_DECISION_SCHEMA,
    STAGE_NAME,
    per_seed_decision,
)


def test_fit_failure_forbids_holdouts_and_all_licenses() -> None:
    decision = per_seed_decision(
        {"terminal_fit_gate": {"passes": False}}, None
    )
    assert decision["schema"] == PER_SEED_DECISION_SCHEMA
    assert decision["classification"] == "fit_gate_failed"
    assert decision["favorable"] is False
    assert decision["shared_jepa_full_train_candidate_licensed"] is False
    assert decision["runtime_ready"] is False
    assert decision["g2_licensed"] is False
    assert decision["g3_licensed"] is False
    with pytest.raises(ValueError, match="forbidden"):
        per_seed_decision(
            {"terminal_fit_gate": {"passes": False}},
            {
                "same_scene_holdout": {"passes": True},
                "cross_scene_holdout": {"passes": True},
            },
        )


def test_fit_pass_requires_both_holdout_decisions() -> None:
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
    assert decision["shared_jepa_full_train_candidate_licensed"] is False


def test_one_failed_holdout_is_unfavorable() -> None:
    decision = per_seed_decision(
        {"terminal_fit_gate": {"passes": True}},
        {
            "same_scene_holdout": {"passes": True},
            "cross_scene_holdout": {"passes": False},
        },
    )
    assert decision["classification"] == "fit_pass_holdout_gate_failed"
    assert decision["favorable"] is False


def test_strict_boolean_is_required() -> None:
    with pytest.raises(ValueError, match="strict"):
        per_seed_decision({"terminal_fit_gate": {"passes": 1}}, None)


def test_non_authoritative_smoke_never_licenses_or_requires_holdouts() -> None:
    decision = per_seed_decision(
        {"terminal_fit_gate": {"passes": True}},
        None,
        authoritative=False,
    )

    assert decision["token_width_32_fit_passes"] is True
    assert decision["classification"] == "non_authoritative_smoke"
    assert decision["qualifying_optimizer_stage"] is None
    assert decision["holdout_passes"] is None
    assert decision["favorable"] is False
    assert decision["aggregation_eligible"] is False
    assert not any(
        decision[name]
        for name in (
            "shared_jepa_full_train_candidate_licensed",
            "runtime_ready",
            "g2_licensed",
            "g3_licensed",
            "promotion_licensed",
        )
    )

    with pytest.raises(ValueError, match="smoke must never contain holdouts"):
        per_seed_decision(
            {"terminal_fit_gate": {"passes": True}},
            {"same_scene_holdout": {"passes": True}},
            authoritative=False,
        )


def test_authoritative_flag_is_strict_boolean() -> None:
    with pytest.raises(TypeError, match="authoritative must be a bool"):
        per_seed_decision(  # type: ignore[arg-type]
            {"terminal_fit_gate": {"passes": False}},
            None,
            authoritative=1,
        )
