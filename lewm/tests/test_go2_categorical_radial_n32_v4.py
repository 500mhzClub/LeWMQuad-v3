from __future__ import annotations

import hashlib
import json

import pytest

from lewm.benchmarks.go2_categorical_radial_n32_v4 import (
    FACTOR_OUTPUT_CONTRACT,
    FACTOR_OUTPUT_CONTRACT_SHA256,
    PER_SEED_DECISION_SCHEMA,
    STAGE_NAME,
    per_seed_decision,
)


def _holdouts(*, cross_scene_passes: bool = True) -> dict[str, dict[str, bool]]:
    return {
        "same_scene_holdout": {"passes": True},
        "cross_scene_holdout": {"passes": cross_scene_passes},
    }


def test_factor_output_contract_hash_and_semantics_are_frozen() -> None:
    expected = hashlib.sha256(
        json.dumps(
            FACTOR_OUTPUT_CONTRACT,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert FACTOR_OUTPUT_CONTRACT_SHA256 == expected
    assert FACTOR_OUTPUT_CONTRACT["raw_factor_order"] == [
        "known_vs_unknown_log_odds",
        "occupied_vs_free_given_known_log_odds",
    ]
    assert FACTOR_OUTPUT_CONTRACT["class_order"] == [
        "unknown",
        "free",
        "occupied",
    ]
    assert FACTOR_OUTPUT_CONTRACT["conversion_before_cartesian_gather"] is True
    assert FACTOR_OUTPUT_CONTRACT["calibration"] == "none"


def test_fit_failure_forbids_holdouts_and_all_licenses() -> None:
    decision = per_seed_decision(
        {"terminal_fit_gate": {"passes": False}}, None
    )
    assert decision["schema"] == PER_SEED_DECISION_SCHEMA
    assert decision["classification"] == "fit_gate_failed"
    assert decision["favorable"] is False
    assert decision["explicit_hierarchical_output_fit_passes"] is False
    assert decision["categorical_radial_full_train_candidate_licensed"] is False
    assert decision["runtime_ready"] is False
    assert decision["g2_licensed"] is False
    assert decision["g3_licensed"] is False
    with pytest.raises(ValueError, match="forbidden"):
        per_seed_decision(
            {"terminal_fit_gate": {"passes": False}},
            _holdouts(),
        )


def test_fit_pass_requires_both_strict_holdout_decisions() -> None:
    stage = {"terminal_fit_gate": {"passes": True}}
    with pytest.raises(ValueError, match="mandatory"):
        per_seed_decision(stage, None)
    decision = per_seed_decision(stage, _holdouts())
    assert decision["classification"] == "favorable"
    assert decision["qualifying_optimizer_stage"] == STAGE_NAME
    assert decision["favorable"] is True
    assert decision["categorical_radial_full_train_candidate_licensed"] is False

    invalid = _holdouts()
    invalid["same_scene_holdout"]["passes"] = 1  # type: ignore[assignment]
    with pytest.raises(ValueError, match="strict"):
        per_seed_decision(stage, invalid)


def test_one_failed_holdout_is_unfavorable() -> None:
    decision = per_seed_decision(
        {"terminal_fit_gate": {"passes": True}},
        _holdouts(cross_scene_passes=False),
    )
    assert decision["classification"] == "fit_pass_holdout_gate_failed"
    assert decision["favorable"] is False


def test_strict_terminal_and_authoritative_booleans_are_required() -> None:
    with pytest.raises(ValueError, match="strict"):
        per_seed_decision({"terminal_fit_gate": {"passes": 1}}, None)
    with pytest.raises(TypeError, match="authoritative must be a bool"):
        per_seed_decision(  # type: ignore[arg-type]
            {"terminal_fit_gate": {"passes": False}},
            None,
            authoritative=1,
        )


def test_non_authoritative_smoke_never_licenses_or_opens_holdouts() -> None:
    decision = per_seed_decision(
        {"terminal_fit_gate": {"passes": True}},
        None,
        authoritative=False,
    )
    assert decision["classification"] == "non_authoritative_smoke"
    assert decision["qualifying_optimizer_stage"] is None
    assert decision["holdout_passes"] is None
    assert decision["favorable"] is False
    assert decision["aggregation_eligible"] is False
    assert not any(
        decision[name]
        for name in (
            "categorical_radial_full_train_candidate_licensed",
            "runtime_ready",
            "g2_licensed",
            "g3_licensed",
            "promotion_licensed",
        )
    )
    with pytest.raises(ValueError, match="smoke must never contain holdouts"):
        per_seed_decision(
            {"terminal_fit_gate": {"passes": True}},
            _holdouts(),
            authoritative=False,
        )
