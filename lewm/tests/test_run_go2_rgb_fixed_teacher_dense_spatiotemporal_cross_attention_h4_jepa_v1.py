from __future__ import annotations

from copy import deepcopy
import hashlib

import pytest

from scripts import (
    run_go2_rgb_fixed_teacher_dense_spatiotemporal_cross_attention_h4_jepa_v1
    as runner,
)


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


def _passing_observations() -> list[dict]:
    return [
        _observation(
            0,
            real=1.0,
            action=0.0,
            hold=0.0,
            persistence=0.0,
            history=0.0,
        ),
        _observation(
            750,
            real=0.80,
            action=0.10,
            hold=0.08,
            persistence=0.20,
            history=0.05,
        ),
    ]


def test_dense_binding_preserves_exact_one_shot_science_and_schedule() -> None:
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
            "dense_h4_wrapper": {
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
        assert runner.core.EXECUTION_SOURCE_BINDINGS == bindings
        assert runner.core.OUTPUT_ROOT == runner.OUTPUT_ROOT
        assert runner.core.MODEL_MODULE == runner.MODEL_MODULE
        assert runner.core.ADDITIONAL_SCIENCE["recurrent_module_count"] == 0
        assert runner.core.UPDATES == 1_000
        assert runner.core.BATCH_SIZE == 16
        assert runner.core.PRESENTATIONS == 16_000
        assert runner.core.OBSERVATION_UPDATES == (0, 250, 500, 750, 1_000)
        assert runner.core.SEED == 20_260_727
    finally:
        for name, value in original.items():
            setattr(runner.core, name, value)


def test_source_closure_is_exact_and_requires_external_wrapper_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LEWM_DENSE_H4_WRAPPER_SHA256", raising=False)
    monkeypatch.delenv("LEWM_DENSE_H4_WRAPPER_BYTES", raising=False)
    with pytest.raises(runner.core.ContractError):
        runner._verify_source_closure()

    raw = runner.Path(runner.__file__).read_bytes()
    monkeypatch.setenv(
        "LEWM_DENSE_H4_WRAPPER_SHA256",
        hashlib.sha256(raw).hexdigest(),
    )
    monkeypatch.setenv("LEWM_DENSE_H4_WRAPPER_BYTES", str(len(raw)))
    closure = runner._verify_source_closure()
    assert set(closure) == {
        "dense_h4_wrapper",
        "shared_runner",
        "v3_gate_source",
        "dense_h4_model",
        "inherited_v1_model",
        "encoder_dependency",
    }
    assert closure["dense_h4_model"]["file_sha256"] == runner.MODEL_SOURCE_SHA256


def test_dense_decision_preserves_v3_gates_but_translates_branch_identity() -> None:
    observations = _passing_observations()
    inherited = runner.v3_runner._v3_decision(observations, 1_000)
    result = runner._dense_decision(observations, 1_000)
    assert result["decision"] == runner.PASS_DECISION
    assert result["gates"] == inherited["gates"]
    assert result["failed_gates"] == inherited["failed_gates"]
    assert result["diagnostics"] == inherited["diagnostics"]
    assert "dense cross-attention" in result["authority"]
    assert "recurrent-H4" not in result["authority"]


def test_dense_decision_rejects_teacher_drift_and_incomplete_cap() -> None:
    observations = _passing_observations()
    observations[1]["noncollapse"]["target_effective_rank_ratio"] = 0.16
    drift = runner._dense_decision(observations, 1_000)
    assert drift["decision"] == runner.STOP_DECISION
    assert "fixed_teacher_metric_geometry_unchanged" in drift["failed_gates"]

    incomplete = runner._dense_decision(_passing_observations(), 999)
    assert incomplete["decision"] == runner.STOP_DECISION
    assert "completed_exact_cap" in incomplete["failed_gates"]


def test_dense_run_replaces_false_recurrent_initialization_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = {
        "fresh_recurrent_and_predictor_initialization": True,
        "unchanged": 7,
    }

    def fake_run(*_args: object, **_kwargs: object) -> tuple[dict, dict, dict]:
        return {"metrics": 1}, artifact, {"decision": "x"}

    monkeypatch.setattr(runner, "_CORE_RUN", fake_run)
    metrics, dense_artifact, decision = runner._dense_run("ignored")
    assert metrics == {"metrics": 1}
    assert decision == {"decision": "x"}
    assert "fresh_recurrent_and_predictor_initialization" not in dense_artifact
    assert dense_artifact[
        "fresh_dense_attention_embeddings_action_path_and_delta_head_initialization"
    ] is True
    assert dense_artifact["unchanged"] == 7
    assert artifact["fresh_recurrent_and_predictor_initialization"] is True

    monkeypatch.setattr(
        runner,
        "_CORE_RUN",
        lambda *_args, **_kwargs: ({}, {}, {}),
    )
    with pytest.raises(runner.core.ContractError):
        runner._dense_run("ignored")


def test_main_installs_closure_stubs_configuration_and_hooks_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    bindings = {"dense_h4_wrapper": {"path": "x", "byte_count": 1}}

    monkeypatch.setattr(
        runner,
        "_verify_source_closure",
        lambda: events.append("verify") or bindings,
    )
    monkeypatch.setattr(
        runner,
        "_install_bound_model_package_stubs",
        lambda: events.append("stubs"),
    )
    monkeypatch.setattr(
        runner,
        "_configure_core",
        lambda value: events.append("configure") if value is bindings else None,
    )
    monkeypatch.setattr(
        runner,
        "_install_dense_run_adapter",
        lambda: events.append("adapter"),
    )
    monkeypatch.setattr(
        runner.core,
        "main",
        lambda argv: events.append("main") or (3 if argv == ["--x"] else 4),
    )
    assert runner.main(["--x"]) == 3
    assert events == ["verify", "stubs", "configure", "adapter", "main"]
    assert runner.core._decision is runner._dense_decision
