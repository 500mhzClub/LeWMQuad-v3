from __future__ import annotations

from copy import deepcopy

from scripts import run_go2_recurrent_h4_fixed_teacher_delta_joint_jepa_v3 as runner


def _observation(
    update: int,
    *,
    real: float,
    action: float,
    hold: float,
    persistence: float,
    history: float,
    target_rank: float = 0.17,
    online_rank: float = 0.20,
) -> dict:
    metrics = {
        "real_normalized_error": [real] * 4,
        "action_gap": [action] * 4,
        "hold_gap": [hold] * 4,
        "persistence_gap": [persistence] * 4,
        "history_gap": [history] * 4,
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
        },
        "noncollapse": {
            "target_effective_rank_ratio": target_rank,
            "online_effective_rank_ratio": online_rank,
            "target_near_zero_variance_fraction": 0.0,
            "online_near_zero_variance_fraction": 0.0,
        },
        "all_registered_values_finite": True,
    }


def test_v3_binding_disables_every_rejected_training_term() -> None:
    names = (
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
    original = {name: getattr(runner.core, name) for name in names}
    try:
        bindings = {
            "v3_wrapper": {
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
        assert runner.core.TARGET_DESCRIPTION == (
            "fixed_accepted_N320_teacher_stop_gradient_no_ema"
        )
        assert runner.core.AUXILIARY_TRAINING_CONTROL_MULTIPLIER == 0
        assert runner.core.EXECUTION_SOURCE_BINDINGS == bindings
        assert runner.core.OUTPUT_ROOT == runner.OUTPUT_ROOT
        assert runner.core.MODEL_MODULE == runner.MODEL_MODULE
    finally:
        for name, value in original.items():
            setattr(runner.core, name, value)


def test_v3_registered_gates_pass_one_fixed_teacher_candidate() -> None:
    baseline = _observation(
        0,
        real=1.0,
        action=0.0,
        hold=0.0,
        persistence=0.0,
        history=0.0,
    )
    candidate = _observation(
        750,
        real=0.80,
        action=0.10,
        hold=0.08,
        persistence=0.20,
        history=0.05,
    )
    result = runner._v3_decision([baseline, candidate], 1_000)
    assert result["decision"] == runner.PASS_DECISION
    assert result["failed_gates"] == []
    assert result["diagnostics"]["selected_update"] == 750


def test_v3_rejects_fixed_teacher_metric_drift() -> None:
    baseline = _observation(
        0,
        real=1.0,
        action=0.0,
        hold=0.0,
        persistence=0.0,
        history=0.0,
    )
    candidate = _observation(
        250,
        real=0.80,
        action=0.10,
        hold=0.08,
        persistence=0.20,
        history=0.05,
        target_rank=0.16,
    )
    result = runner._v3_decision([baseline, candidate], 1_000)
    assert result["decision"] == runner.STOP_DECISION
    assert "fixed_teacher_metric_geometry_unchanged" in result["failed_gates"]
