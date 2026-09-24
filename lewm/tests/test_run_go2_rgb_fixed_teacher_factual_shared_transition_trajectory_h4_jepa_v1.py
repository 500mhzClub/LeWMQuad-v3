from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import (
    run_go2_rgb_fixed_teacher_factual_shared_transition_trajectory_h4_jepa_v1
    as runner,
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
    p0_p1_score: float,
    p0_p1_gap: float,
    future_local_score: float,
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
        "p0_p1_local_prior_combined_normalized_energy_score": p0_p1_score,
        "p0_p1_local_prior_persistence_gap": p0_p1_gap,
        "future_p2_p5_local_combined_normalized_energy_score": (
            future_local_score
        ),
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
            "p0_p1_local_prior_persistence_gap": p0_p1_gap,
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
            p0_p1_score=1.0,
            p0_p1_gap=0.0,
            future_local_score=1.0,
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
            p0_p1_score=0.75,
            p0_p1_gap=0.25,
            future_local_score=0.78,
        ),
    ]


def test_configuration_is_exact_factual_shared_transition_science() -> None:
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
    try:
        bindings = {
            "factual_shared_transition_trajectory_h4_wrapper": {
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
        assert runner.core.AUXILIARY_TRAINING_CONTROL_MULTIPLIER == 0
        assert runner.core.MODEL_MODULE == runner.MODEL_MODULE
        assert runner.core.OUTPUT_ROOT == runner.OUTPUT_ROOT
        assert runner.core.SCHEMA == runner.SCHEMA
        assert "all_six_factual_local_innovation" in (
            runner.core.OBJECTIVE_DESCRIPTION
        )
        science = runner.core.ADDITIONAL_SCIENCE
        assert science["transition"]["shared_core"].endswith("p0_through_p5")
        assert science["transition"]["target_leakage"] == "none"
        assert science["training_controls"] == {
            "enabled": False,
            "wrong_action": False,
            "all_hold": False,
            "reversed_history": False,
            "reset_history": False,
        }
        assert science["proper_score"] == {
            "all_six_factual_local_innovation_weight": 0.5,
            "open_loop_future_cumulative_trajectory_weight": 0.5,
            "each_domain": (
                "50_50_joint_plus_mean_marginal_uniform_energy_score"
            ),
            "prediction_normalization": "none",
        }
        assert runner.core.UPDATES == 1_000
        assert runner.core.BATCH_SIZE == 16
        assert runner.core.PRESENTATIONS == 16_000
        assert runner.core.OBSERVATION_UPDATES == (0, 250, 500, 750, 1_000)
        assert runner.core.SEED == 20_260_727
    finally:
        for name, value in base_original.items():
            setattr(runner.base, name, value)
        for name, value in core_original.items():
            setattr(runner.core, name, value)


def test_source_closure_requires_wrapper_binding_and_has_exact_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sha_name = "LEWM_FACTUAL_SHARED_TRANSITION_TRAJECTORY_H4_WRAPPER_SHA256"
    bytes_name = "LEWM_FACTUAL_SHARED_TRANSITION_TRAJECTORY_H4_WRAPPER_BYTES"
    monkeypatch.delenv(sha_name, raising=False)
    monkeypatch.delenv(bytes_name, raising=False)
    with pytest.raises(runner.core.ContractError):
        runner._verify_source_closure()

    calls: list[tuple[Path, str, int]] = []

    def fake_binding(path: Path, sha256: str, byte_count: int) -> dict:
        calls.append((path, sha256, byte_count))
        return {
            "path": str(path),
            "file_sha256": sha256,
            "byte_count": byte_count,
        }

    monkeypatch.setattr(runner.base, "_source_binding", fake_binding)
    monkeypatch.setenv(sha_name, "a" * 64)
    monkeypatch.setenv(bytes_name, "123")
    closure = runner._verify_source_closure()
    assert set(closure) == {
        "factual_shared_transition_trajectory_h4_wrapper",
        "trajectory_h4_wrapper_dependency",
        "shared_runner",
        "factual_shared_transition_trajectory_h4_model",
        "trajectory_h4_model_dependency",
        "local_innovation_trajectory_h4_model_dependency",
        "dense_h4_model_dependency",
        "inherited_v1_model",
        "encoder_dependency",
    }
    assert calls[0] == (Path(runner.__file__).resolve(), "a" * 64, 123)
    assert calls[3][0] == runner.MODEL_SOURCE
    assert calls[4][0] == runner.TRAJECTORY_MODEL_SOURCE
    assert calls[5][0] == runner.LOCAL_INNOVATION_MODEL_SOURCE


def test_zero_innovation_local_score_is_exact_persistence() -> None:
    runtime = SimpleNamespace(torch=torch)
    target = torch.linspace(-1.0, 1.0, steps=2 * 2 * 3 * 4).reshape(2, 2, 3, 4)
    atoms = torch.zeros(2, 4, 2, 3, 4)
    ratio, gap = runner._normalized_local_combined_score(atoms, target, runtime)
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-7, rtol=0.0)
    assert torch.allclose(gap, torch.zeros_like(gap), atol=1e-7, rtol=0.0)


def test_decision_keeps_future_28_gates_and_adds_exact_p0_p1_gates() -> None:
    passed = runner._factual_shared_transition_decision(
        _passing_observations(),
        1_000,
    )
    assert passed["decision"] == runner.PASS_DECISION
    assert passed["failed_gates"] == []
    assert len(passed["gates"]) == 32
    assert passed["diagnostics"]["selected_update"] == 750
    assert passed["diagnostics"]["hold_positive_family_count"] == 8
    assert passed["diagnostics"][
        "p0_p1_persistence_positive_family_count"
    ] == 8

    six = _passing_observations()
    for family in tuple(runner.core.FAMILIES)[:2]:
        six[1]["family"][family]["p0_p1_local_prior_persistence_gap"] = 0.0
    assert runner._factual_shared_transition_decision(six, 1_000)[
        "decision"
    ] == runner.PASS_DECISION

    five = deepcopy(six)
    five[1]["family"][tuple(runner.core.FAMILIES)[2]][
        "p0_p1_local_prior_persistence_gap"
    ] = 0.0
    stopped = runner._factual_shared_transition_decision(five, 1_000)
    assert (
        "p0_p1_local_prior_persistence_positive_in_six_families"
        in stopped["failed_gates"]
    )

    floor = _passing_observations()
    first = tuple(runner.core.FAMILIES)[0]
    floor[1]["family"][first][
        "p0_p1_local_prior_persistence_gap"
    ] = -0.020001
    stopped = runner._factual_shared_transition_decision(floor, 1_000)
    assert (
        "no_family_p0_p1_local_prior_persistence_gap_below_minus_point02"
        in stopped["failed_gates"]
    )

    score = _passing_observations()
    score[1]["aggregate"][
        "p0_p1_local_prior_combined_normalized_energy_score"
    ] = 1.0
    stopped = runner._factual_shared_transition_decision(score, 1_000)
    assert (
        "p0_p1_local_prior_combined_score_below_persistence"
        in stopped["failed_gates"]
    )

    lower = _passing_observations()
    lower[1]["bootstrap_lower_95"][
        "p0_p1_local_prior_persistence_gap"
    ] = 0.0
    stopped = runner._factual_shared_transition_decision(lower, 1_000)
    assert (
        "p0_p1_local_prior_persistence_gap_bootstrap_lower_positive"
        in stopped["failed_gates"]
    )


def test_update_zero_requires_exact_p0_p1_persistence() -> None:
    observations = _passing_observations()
    observations[0]["aggregate"][
        "p0_p1_local_prior_combined_normalized_energy_score"
    ] = 0.99
    stopped = runner._factual_shared_transition_decision(observations, 1_000)
    assert "update_zero_is_exact_persistence" in stopped["failed_gates"]

    observations = _passing_observations()
    observations[0]["aggregate"][
        "p0_p1_local_prior_persistence_gap"
    ] = 2e-5
    stopped = runner._factual_shared_transition_decision(observations, 1_000)
    assert "update_zero_is_exact_persistence" in stopped["failed_gates"]


def test_hold_breadth_and_floor_match_the_dual_domain_contract() -> None:
    six = _passing_observations()
    for family in tuple(runner.core.FAMILIES)[:2]:
        six[1]["family"][family]["hold_gap"][3] = 0.0
    assert runner._factual_shared_transition_decision(six, 1_000)[
        "decision"
    ] == runner.PASS_DECISION

    five = deepcopy(six)
    five[1]["family"][tuple(runner.core.FAMILIES)[2]]["hold_gap"][3] = 0.0
    stopped = runner._factual_shared_transition_decision(five, 1_000)
    assert "hold_positive_in_six_families" in stopped["failed_gates"]

    floor = _passing_observations()
    first = tuple(runner.core.FAMILIES)[0]
    floor[1]["family"][first]["hold_gap"][3] = -0.020001
    stopped = runner._factual_shared_transition_decision(floor, 1_000)
    assert "no_family_hold_gap_below_minus_point02" in stopped["failed_gates"]


def test_runtime_adapter_reuses_future_evaluator_and_installs_handlers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner.core, "_evaluate", runner.base._CORE_EVALUATE)
    monkeypatch.setattr(runner.core, "_decision", runner.base._CORE_DECISION)
    monkeypatch.setattr(runner.core, "_run", runner.base._CORE_RUN)
    runner._install_runtime_adapters()
    assert runner.core._evaluate is runner._factual_shared_transition_evaluate
    assert runner.core._decision is runner._factual_shared_transition_decision
    assert runner.core._run is runner._factual_shared_transition_run


def test_run_adapter_records_only_factual_objectives_and_zero_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bucket = {
        "prediction": 0.7,
        "variance": 0.0,
        "wrong_action_ranking": 0.0,
        "history_teacher_alignment": 0.1,
        "half_all_six_factual_local_innovation_energy_score": 0.2,
        "half_open_loop_future_cumulative_trajectory_energy_score": 0.3,
        "total": 0.6,
    }
    metrics = {
        "training_losses": {
            "mean_over_completed_updates": deepcopy(bucket),
            "last_completed_update": deepcopy(bucket),
            "objective": "raw",
        }
    }
    artifact = {
        "fresh_dense_history_mode_embeddings_action_path_and_shared_delta_head_"
        "initialization": True,
        "wrong_action_training_contrast_enabled": False,
    }

    def fake_base_run(*_args: object, **_kwargs: object) -> tuple[dict, dict, dict]:
        return metrics, artifact, {"decision": "x"}

    monkeypatch.setattr(runner, "_BASE_RUN", fake_base_run)
    access: Counter[str] = Counter()
    adapted_metrics, adapted_artifact, decision = (
        runner._factual_shared_transition_run(access=access)
    )
    assert decision == {"decision": "x"}
    expected = {
        "total",
        "diagnostic_centroid_absolute_future_error",
        "history_teacher_alignment",
        "half_all_six_factual_local_innovation_energy_score",
        "half_open_loop_future_cumulative_trajectory_energy_score",
    }
    for name in ("mean_over_completed_updates", "last_completed_update"):
        assert set(adapted_metrics["training_losses"][name]) == expected
    assert adapted_artifact[
        "fresh_shared_transition_mode_and_residual_head_initialization"
    ] is True
    assert adapted_artifact[
        "built_in_centroid_wrong_action_training_contrast_enabled"
    ] is False
    assert adapted_artifact[
        "cyclic_wrong_action_training_contrast_enabled"
    ] is False
    assert adapted_artifact["all_hold_training_contrast_enabled"] is False
    assert adapted_artifact[
        "reversed_and_reset_history_training_contrasts_enabled"
    ] is False
    assert access["auxiliary_training_control_sequence_count"] == 0
    assert access["wrong_action_counterfactual_sequence_count"] == 0


@pytest.mark.parametrize(
    "counter_name",
    (
        "auxiliary_training_control_sequence_count",
        "wrong_action_counterfactual_sequence_count",
    ),
)
def test_run_adapter_rejects_any_training_control_count(
    monkeypatch: pytest.MonkeyPatch,
    counter_name: str,
) -> None:
    def fake_base_run(*_args: object, **_kwargs: object) -> tuple[dict, dict, dict]:
        return {}, {}, {}

    monkeypatch.setattr(runner, "_BASE_RUN", fake_base_run)
    access: Counter[str] = Counter({counter_name: 1})
    with pytest.raises(runner.core.ContractError):
        runner._factual_shared_transition_run(access=access)
