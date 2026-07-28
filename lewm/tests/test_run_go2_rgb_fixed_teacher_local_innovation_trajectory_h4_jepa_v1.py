from __future__ import annotations

from copy import deepcopy
from collections import Counter
import hashlib

import pytest

from scripts import (
    run_go2_rgb_fixed_teacher_local_innovation_trajectory_h4_jepa_v1 as runner,
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


def test_configuration_is_exact_local_innovation_counterfactual_science() -> None:
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
    base_names = (
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
    base_original = {name: getattr(runner.base, name) for name in base_names}
    try:
        bindings = {
            "local_innovation_trajectory_h4_wrapper": {
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
        assert "local_innovation_energy" in runner.core.OBJECTIVE_DESCRIPTION
        science = runner.core.ADDITIONAL_SCIENCE
        assert science["support"].startswith("four_equal_mass")
        assert "e2_to_e3" in science["target"]
        assert science["training_controls"]["wrong_action"].endswith("0.05")
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


def test_source_closure_requires_and_accepts_exact_wrapper_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sha_name = "LEWM_LOCAL_INNOVATION_TRAJECTORY_H4_WRAPPER_SHA256"
    bytes_name = "LEWM_LOCAL_INNOVATION_TRAJECTORY_H4_WRAPPER_BYTES"
    monkeypatch.delenv(sha_name, raising=False)
    monkeypatch.delenv(bytes_name, raising=False)
    with pytest.raises(runner.core.ContractError):
        runner._verify_source_closure()

    raw = runner.Path(runner.__file__).read_bytes()
    monkeypatch.setenv(sha_name, hashlib.sha256(raw).hexdigest())
    monkeypatch.setenv(bytes_name, str(len(raw)))
    closure = runner._verify_source_closure()
    assert set(closure) == {
        "local_innovation_trajectory_h4_wrapper",
        "trajectory_h4_wrapper_dependency",
        "shared_runner",
        "local_innovation_trajectory_h4_model",
        "trajectory_h4_model_dependency",
        "dense_h4_model_dependency",
        "inherited_v1_model",
        "encoder_dependency",
    }


def test_decision_retains_full_trajectory_gate_and_updates_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner.base, "PASS_DECISION", runner.PASS_DECISION)
    monkeypatch.setattr(runner.base, "STOP_DECISION", runner.STOP_DECISION)
    passed = runner._local_innovation_decision(_passing_observations(), 1_000)
    assert passed["decision"] == runner.PASS_DECISION
    assert passed["failed_gates"] == []
    assert "local-innovation" in passed["authority"]

    failed = _passing_observations()
    failed[1]["aggregate"]["history_gap"][3] = 0.0
    stopped = runner._local_innovation_decision(failed, 1_000)
    assert stopped["decision"] == runner.STOP_DECISION
    assert "h4_history_gap_at_least_point03" in stopped["failed_gates"]


def test_runtime_adapter_reuses_trajectory_evaluator_and_installs_local_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner.core, "_evaluate", runner.base._CORE_EVALUATE)
    monkeypatch.setattr(runner.core, "_decision", runner.base._CORE_DECISION)
    monkeypatch.setattr(runner.core, "_run", runner.base._CORE_RUN)
    runner._install_runtime_adapters()
    assert runner.core._evaluate is runner.base._trajectory_evaluate
    assert runner.core._decision is runner._local_innovation_decision
    assert runner.core._run is runner._local_innovation_run


def test_run_adapter_separates_action_and_history_control_receipts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    access: Counter[str] = Counter(
        optimizer_update_count=2,
        auxiliary_training_control_sequence_count=96,
        wrong_action_counterfactual_sequence_count=0,
    )
    artifact = {
        "wrong_action_training_contrast_enabled": False,
        "unchanged": 7,
    }
    metrics_receipt = {
        "training_losses": {
            "mean_over_completed_updates": {
                "prediction": 0.7,
                "variance": 0.0,
                "wrong_action_ranking": 0.0,
            },
            "last_completed_update": {
                "prediction": 0.6,
                "variance": 0.0,
                "wrong_action_ranking": 0.0,
            },
        }
    }

    def fake_run(*_args: object, **_kwargs: object) -> tuple[dict, dict, dict]:
        return metrics_receipt, artifact, {"decision": "x"}

    monkeypatch.setattr(runner, "_BASE_RUN", fake_run)
    metrics, adapted, decision = runner._local_innovation_run(access=access)
    assert metrics is metrics_receipt
    assert decision == {"decision": "x"}
    assert access["wrong_action_counterfactual_sequence_count"] == 32
    assert access["auxiliary_training_control_sequence_count"] == 64
    assert adapted[
        "local_innovation_cyclic_wrong_action_training_contrast_enabled"
    ] is True
    assert adapted[
        "reversed_and_reset_history_training_contrasts_enabled"
    ] is True
    assert (
        adapted["built_in_centroid_wrong_action_training_contrast_enabled"]
        is False
    )
    assert adapted["unchanged"] == 7
    training = metrics["training_losses"]
    assert "prediction" not in training["last_completed_update"]
    assert (
        training["last_completed_update"][
            "diagnostic_centroid_absolute_future_error"
        ]
        == 0.6
    )
    assert training["disabled_terms"] == [
        "absolute_future_centroid_prediction_in_objective",
        "inherited_variance_regularization",
        "built_in_centroid_wrong_action_ranking",
    ]
