from __future__ import annotations

from copy import deepcopy
import hashlib

import pytest

from scripts import (
    run_go2_rgb_fixed_teacher_dual_domain_trajectory_h4_jepa_v1 as runner,
)


def _observation(
    update: int,
    *,
    real: float,
    joint: float,
    action: float,
    hold: float,
    persistence: float,
    history: float,
    distribution: float,
    spread: float,
) -> dict:
    metrics = {
        "real_normalized_energy_score": [real] * 4,
        "action_gap": [action] * 4,
        "hold_gap": [hold] * 4,
        "persistence_gap": [persistence] * 4,
        "history_gap": [history] * 4,
        "distribution_value_gap": [distribution] * 4,
        "normalized_pairwise_spread": [spread] * 4,
        "best_atom_normalized_squared_error": [real] * 4,
        "centroid_normalized_squared_error": [real] * 4,
        "joint_trajectory_normalized_energy_score": joint,
        "combined_normalized_energy_score": 0.5 * (joint + real),
        "combined_distribution_value_gap": distribution,
    }
    return {
        "update": update,
        "presentations": update * 16,
        "aggregate": deepcopy(metrics),
        "family": {
            family: deepcopy(metrics) for family in runner.core.FAMILIES
        },
        "bootstrap_lower_95": {
            "action_gap_h4": action,
            "persistence_gap_h4": persistence,
            "history_gap_h4": history,
            "distribution_value_gap_h4": distribution,
            "combined_distribution_value_gap": distribution,
        },
        "noncollapse": {
            "target_effective_rank_ratio": 0.17,
            "online_effective_rank_ratio": 0.20,
            "target_near_zero_variance_fraction": 0.0,
            "online_near_zero_variance_fraction": 0.0,
        },
        "all_registered_values_finite": True,
    }


def _passing_observations() -> list[dict]:
    return [
        _observation(
            0,
            real=1.0,
            joint=1.0,
            action=0.0,
            hold=0.0,
            persistence=0.0,
            history=0.0,
            distribution=0.0,
            spread=0.0,
        ),
        _observation(
            750,
            real=0.80,
            joint=0.82,
            action=0.10,
            hold=0.08,
            persistence=0.20,
            history=0.05,
            distribution=0.06,
            spread=0.10,
        ),
    ]


def test_configuration_is_exact_dual_domain_science() -> None:
    core_names = (
        "MODEL_MODULE",
        "MODEL_SOURCE",
        "MODEL_SOURCE_SHA256",
        "MODEL_SOURCE_BYTES",
        "OUTPUT_ROOT",
        "SCHEMA",
        "PASS_DECISION",
        "STOP_DECISION",
        "PREDICTION_WEIGHT",
        "VARIANCE_WEIGHT",
        "ACTION_RANKING_WEIGHT",
        "TRAIN_WRONG_ACTION_CONTRAST",
        "UPDATE_TARGET_EMA",
        "TARGET_DESCRIPTION",
        "OBJECTIVE_DESCRIPTION",
        "ADDITIONAL_SCIENCE",
        "AUXILIARY_TRAINING_CONTROL_MULTIPLIER",
        "EXECUTION_SOURCE_BINDINGS",
    )
    inherited_names = (
        "MODEL_MODULE",
        "MODEL_SOURCE",
        "MODEL_SOURCE_SHA256",
        "MODEL_SOURCE_BYTES",
        "OUTPUT_ROOT",
        "SCHEMA",
        "PASS_DECISION",
        "STOP_DECISION",
    )
    core_original = {name: getattr(runner.core, name) for name in core_names}
    base_original = {name: getattr(runner.base, name) for name in inherited_names}
    parent_original = {
        name: getattr(runner.parent, name) for name in inherited_names
    }
    try:
        bindings = {
            "dual_domain_trajectory_h4_wrapper": {
                "path": "runner.py",
                "file_sha256": "0" * 64,
                "byte_count": 1,
            }
        }
        runner._configure_core(bindings)
        assert runner.core.PREDICTION_WEIGHT == 0.0
        assert runner.core.VARIANCE_WEIGHT == 0.0
        assert runner.core.ACTION_RANKING_WEIGHT == 0.0
        assert runner.core.TRAIN_WRONG_ACTION_CONTRAST is False
        assert runner.core.UPDATE_TARGET_EMA is False
        assert runner.core.AUXILIARY_TRAINING_CONTROL_MULTIPLIER == 3
        assert "0.5*proper_local_innovation" in runner.core.OBJECTIVE_DESCRIPTION
        science = runner.core.ADDITIONAL_SCIENCE
        assert science["proper_score"]["local_weight"] == 0.5
        assert science["proper_score"]["cumulative_weight"] == 0.5
        assert science["training_controls"]["score"].startswith("same_50_50")
        assert "all_hold_training_control" in science["absent"]
        assert runner.core.UPDATES == 1_000
        assert runner.core.BATCH_SIZE == 16
        assert runner.core.PRESENTATIONS == 16_000
        assert runner.core.OBSERVATION_UPDATES == (0, 250, 500, 750, 1_000)
        assert runner.core.SEED == 20_260_727
    finally:
        for name, value in parent_original.items():
            setattr(runner.parent, name, value)
        for name, value in base_original.items():
            setattr(runner.base, name, value)
        for name, value in core_original.items():
            setattr(runner.core, name, value)


def test_source_closure_requires_and_accepts_exact_wrapper_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sha_name = "LEWM_DUAL_DOMAIN_TRAJECTORY_H4_WRAPPER_SHA256"
    bytes_name = "LEWM_DUAL_DOMAIN_TRAJECTORY_H4_WRAPPER_BYTES"
    monkeypatch.delenv(sha_name, raising=False)
    monkeypatch.delenv(bytes_name, raising=False)
    with pytest.raises(runner.core.ContractError):
        runner._verify_source_closure()

    raw = runner.Path(runner.__file__).read_bytes()
    monkeypatch.setenv(sha_name, hashlib.sha256(raw).hexdigest())
    monkeypatch.setenv(bytes_name, str(len(raw)))
    closure = runner._verify_source_closure()
    assert set(closure) == {
        "dual_domain_trajectory_h4_wrapper",
        "local_innovation_trajectory_h4_wrapper_dependency",
        "trajectory_h4_wrapper_dependency",
        "shared_runner",
        "dual_domain_trajectory_h4_model",
        "local_innovation_trajectory_h4_model_dependency",
        "trajectory_h4_model_dependency",
        "dense_h4_model_dependency",
        "inherited_v1_model",
        "encoder_dependency",
    }


def test_hold_breadth_and_floor_are_exact_and_do_not_change_selection() -> None:
    passed = runner._dual_domain_decision(_passing_observations(), 1_000)
    assert passed["decision"] == runner.PASS_DECISION
    assert passed["failed_gates"] == []
    assert passed["diagnostics"]["selected_update"] == 750
    assert passed["diagnostics"]["hold_positive_family_count"] == 8

    six = _passing_observations()
    for family in tuple(runner.core.FAMILIES)[:2]:
        six[1]["family"][family]["hold_gap"][3] = 0.0
    assert runner._dual_domain_decision(six, 1_000)["decision"] == runner.PASS_DECISION

    five = deepcopy(six)
    five[1]["family"][tuple(runner.core.FAMILIES)[2]]["hold_gap"][3] = 0.0
    stopped = runner._dual_domain_decision(five, 1_000)
    assert "hold_positive_in_six_families" in stopped["failed_gates"]

    boundary = _passing_observations()
    first = tuple(runner.core.FAMILIES)[0]
    boundary[1]["family"][first]["hold_gap"][3] = -0.02
    assert runner._dual_domain_decision(boundary, 1_000)["decision"] == (
        runner.PASS_DECISION
    )
    boundary[1]["family"][first]["hold_gap"][3] = -0.020001
    stopped = runner._dual_domain_decision(boundary, 1_000)
    assert "no_family_hold_gap_below_minus_point02" in stopped["failed_gates"]

    selected_failure = _passing_observations()
    better = _observation(
        500,
        real=0.70,
        joint=0.72,
        action=0.10,
        hold=0.08,
        persistence=0.30,
        history=0.05,
        distribution=0.06,
        spread=0.10,
    )
    for family in tuple(runner.core.FAMILIES)[:3]:
        better["family"][family]["hold_gap"][3] = 0.0
    selected_failure.append(better)
    stopped = runner._dual_domain_decision(selected_failure, 1_000)
    assert stopped["diagnostics"]["selected_update"] == 500
    assert "hold_positive_in_six_families" in stopped["failed_gates"]


def test_runtime_adapter_keeps_parent_evaluator_and_installs_dual_handlers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner.core, "_evaluate", runner.base._CORE_EVALUATE)
    monkeypatch.setattr(runner.core, "_decision", runner.base._CORE_DECISION)
    monkeypatch.setattr(runner.core, "_run", runner.base._CORE_RUN)
    runner._install_runtime_adapters()
    assert runner.core._evaluate is runner.base._trajectory_evaluate
    assert runner.core._decision is runner._dual_domain_decision
    assert runner.core._run is runner._dual_domain_run


def test_run_adapter_replaces_local_semantics_without_new_control_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bucket = {
        "total": 1.0,
        "diagnostic_centroid_absolute_future_error": 0.7,
        "history_teacher_alignment": 0.1,
        "half_future_teacher_local_innovation_energy_score": 0.2,
        "half_future_teacher_cumulative_trajectory_energy_score": 0.3,
        "dual_domain_cyclic_wrong_action_score_ranking": 0.1,
        "dual_domain_history_counterfactual_score_ranking": 0.3,
    }
    metrics = {
        "training_losses": {
            "mean_over_completed_updates": deepcopy(bucket),
            "last_completed_update": deepcopy(bucket),
            "disabled_terms": ["x"],
            "receipt_field_semantics": {
                "future_teacher_local_innovation_energy_score": "stale"
            },
        }
    }
    artifact = {
        "local_innovation_cyclic_wrong_action_training_contrast_enabled": True,
        "reversed_and_reset_history_training_contrasts_enabled": True,
        "built_in_centroid_wrong_action_training_contrast_enabled": False,
    }

    def fake_parent_run(*_args: object, **_kwargs: object) -> tuple[dict, dict, dict]:
        return metrics, artifact, {"decision": "x"}

    monkeypatch.setattr(runner, "_PARENT_RUN", fake_parent_run)
    adapted_metrics, adapted_artifact, decision = runner._dual_domain_run()
    assert decision == {"decision": "x"}
    semantics = adapted_metrics["training_losses"]["receipt_field_semantics"]
    assert "future_teacher_local_innovation_energy_score" not in semantics
    assert semantics["half_future_teacher_local_innovation_energy_score"].endswith(
        "one_half"
    )
    assert adapted_artifact["dual_domain_prediction_score_enabled"] is True
    assert adapted_artifact["all_hold_training_contrast_enabled"] is False
    assert (
        "local_innovation_cyclic_wrong_action_training_contrast_enabled"
        not in adapted_artifact
    )
