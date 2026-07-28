from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
from types import SimpleNamespace

import pytest
import torch

from lewm.models.encoders import VisionEncoder
from lewm.models.go2_rgb_whitened_delta_predictive_state_h4_jepa_v1 import (
    WhitenedDeltaPredictiveStateConfig,
    WhitenedDeltaPredictiveStateH4JEPA,
)
from scripts import run_go2_rgb_whitened_delta_predictive_state_h4_jepa_v1 as runner


def _observation(
    update: int,
    *,
    real: float,
    action: float,
    hold: float,
    persistence: float,
    history: float,
    state_ok: bool,
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
    rank = 0.80 if state_ok else 0.0
    minimum_std = 0.80 if state_ok else 0.0
    maximum_std = 1.20 if state_ok else 0.0
    maximum_abs_mean = 0.0
    return {
        "update": update,
        "presentations": update * 16,
        "aggregate": deepcopy(metrics),
        "family": {
            family: deepcopy(metrics) for family in runner.core.FAMILIES
        },
        "bootstrap_lower_95": {
            "action_gap_h4": action,
            "hold_gap_h4": hold,
            "persistence_gap_h4": persistence,
            "history_gap_h4": history,
        },
        "state_geometry": {
            role: {
                "participation_rank_ratio": [rank] * 4,
                "minimum_std": [minimum_std] * 4,
                "maximum_std": [maximum_std] * 4,
                "maximum_abs_mean": [maximum_abs_mean] * 4,
            }
            for role in ("predicted", "target")
        },
        "state_energy": {
            "predicted_rms": [1.0 if state_ok else 0.0] * 4,
            "target_rms": [1.0 if state_ok else 0.01] * 4,
            "predicted_mean_energy_fraction": [0.0] * 4,
            "target_mean_energy_fraction": [0.0] * 4,
            "target_mean_squared_energy": [1.0 if state_ok else 0.0001] * 4,
            "near_zero_scene_denominator_count": [0 if state_ok else 150] * 4,
        },
        "noncollapse": {
            "target_effective_rank_ratio": target_rank,
            "online_effective_rank_ratio": online_rank,
            "target_near_zero_variance_fraction": 0.0,
            "online_near_zero_variance_fraction": 0.0,
        },
        "all_registered_values_finite": True,
    }


def _passing() -> list[dict]:
    return [
        _observation(
            0,
            real=1.0,
            action=0.0,
            hold=0.0,
            persistence=0.0,
            history=0.0,
            state_ok=False,
        ),
        _observation(
            750,
            real=0.80,
            action=0.05,
            hold=0.04,
            persistence=0.20,
            history=0.05,
            state_ok=True,
        ),
    ]


def _tiny_model() -> WhitenedDeltaPredictiveStateH4JEPA:
    config = WhitenedDeltaPredictiveStateConfig(
        image_size=8,
        patch_size=4,
        feature_dim=12,
        encoder_depth=1,
        encoder_heads=3,
        recurrent_spatial_heads=3,
        cross_attention_heads=3,
    )
    encoder = VisionEncoder(
        image_size=8,
        patch_size=4,
        hidden_dim=12,
        depth=1,
        n_heads=3,
        mlp_ratio=4,
        dropout=0.0,
    )
    return WhitenedDeltaPredictiveStateH4JEPA(encoder.state_dict(), config=config)


def test_core_configuration_preserves_exact_one_shot_science() -> None:
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
        bindings = {"wdps_h4_wrapper": {"path": "x", "byte_count": 1}}
        runner._configure_core(bindings)
        assert runner.core.PREDICTION_WEIGHT == 0.0
        assert runner.core.VARIANCE_WEIGHT == 0.0
        assert runner.core.ACTION_RANKING_WEIGHT == 0.0
        assert runner.core.TRAIN_WRONG_ACTION_CONTRAST is False
        assert runner.core.UPDATE_TARGET_EMA is False
        assert runner.core.ADDITIONAL_SCIENCE["state"].startswith("four_horizons")
        assert runner.core.ADDITIONAL_SCIENCE["fixed_teacher_role"] == (
            "fixed_target_and_history_teacher"
        )
        assert "zero_mean" in runner.core.OBJECTIVE_DESCRIPTION
        assert runner.core.UPDATES == 1_000
        assert runner.core.BATCH_SIZE == 16
        assert runner.core.PRESENTATIONS == 16_000
        assert runner.core.OBSERVATION_UPDATES == (0, 250, 500, 750, 1_000)
        assert runner.core.SEED == 20_260_727
    finally:
        for name, value in original.items():
            setattr(runner.core, name, value)


def test_source_closure_requires_external_wrapper_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LEWM_WDPS_H4_WRAPPER_SHA256", raising=False)
    monkeypatch.delenv("LEWM_WDPS_H4_WRAPPER_BYTES", raising=False)
    with pytest.raises(runner.core.ContractError):
        runner._verify_source_closure()
    raw = runner.Path(runner.__file__).read_bytes()
    monkeypatch.setenv("LEWM_WDPS_H4_WRAPPER_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setenv("LEWM_WDPS_H4_WRAPPER_BYTES", str(len(raw)))
    closure = runner._verify_source_closure()
    assert set(closure) == {
        "wdps_h4_wrapper",
        "shared_runner",
        "wdps_h4_model",
        "dense_h4_model_dependency",
        "inherited_v1_model",
        "encoder_dependency",
    }


def test_decision_selects_eligible_state_and_requires_action_history() -> None:
    result = runner._decision(_passing(), 1_000)
    assert result["decision"] == runner.PASS_DECISION
    assert result["failed_gates"] == []
    assert result["diagnostics"]["selected_update"] == 750

    weak_action = _passing()
    weak_action[1]["aggregate"]["action_gap"][3] = 0.01
    stopped = runner._decision(weak_action, 1_000)
    assert stopped["decision"] == runner.STOP_DECISION
    assert "h4_action_gap_at_least_point03" in stopped["failed_gates"]

    weak_history = _passing()
    weak_history[1]["aggregate"]["history_gap"][3] = -0.01
    stopped = runner._decision(weak_history, 1_000)
    assert "h4_history_gap_at_least_point03" in stopped["failed_gates"]


def test_decision_rejects_compact_state_collapse_teacher_drift_and_short_run() -> None:
    collapsed = _passing()
    collapsed[1]["state_geometry"]["target"]["minimum_std"] = [0.1] * 4
    result = runner._decision(collapsed, 1_000)
    assert result["decision"] == runner.STOP_DECISION
    assert "eligible_noncollapsed_compact_state_exists" in result["failed_gates"]

    drift = _passing()
    drift[1]["noncollapse"]["target_effective_rank_ratio"] = 0.16
    result = runner._decision(drift, 1_000)
    assert "fixed_teacher_metric_geometry_unchanged" in result["failed_gates"]
    assert "completed_exact_cap" in runner._decision(_passing(), 999)["failed_gates"]


def test_decision_rejects_an_otherwise_healthy_high_dc_state() -> None:
    high_dc = _passing()
    for role in ("predicted", "target"):
        high_dc[1]["state_geometry"][role]["maximum_abs_mean"] = [1.0] * 4
    high_dc[1]["state_energy"]["predicted_mean_energy_fraction"] = [0.50] * 4
    high_dc[1]["state_energy"]["target_mean_energy_fraction"] = [0.50] * 4
    result = runner._decision(high_dc, 1_000)
    assert result["decision"] == runner.STOP_DECISION
    assert "eligible_noncollapsed_compact_state_exists" in result["failed_gates"]


def test_state_geometry_detects_rank_and_scale() -> None:
    generator = torch.Generator().manual_seed(73)
    healthy = torch.randn(128, 4, 8, generator=generator)
    geometry = runner._state_geometry(healthy, SimpleNamespace(torch=torch))
    assert all(value > 0.75 for value in geometry["participation_rank_ratio"])
    assert all(0.5 < value < 2.0 for value in geometry["minimum_std"])
    assert all(value < 0.35 for value in geometry["maximum_abs_mean"])
    collapsed = torch.zeros(128, 4, 8)
    geometry = runner._state_geometry(collapsed, SimpleNamespace(torch=torch))
    assert geometry["participation_rank_ratio"] == pytest.approx([0.0] * 4)


def test_synthetic_evaluator_has_exact_update_zero_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(74)
    model = _tiny_model()
    rows = [
        SimpleNamespace(family=family, scene_id=f"scene_{index}")
        for index, family in enumerate(runner.core.FAMILIES)
    ]
    rgb = torch.randn(len(rows), 7, 3, 8, 8)
    actions = torch.tensor(
        [[index % 9 for index in range(6)] for _row in rows], dtype=torch.long
    )

    def fake_load(batch_rows: object, **_kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(batch_rows) == len(rows)  # type: ignore[arg-type]
        return rgb, actions

    monkeypatch.setattr(runner.core, "_load_batch", fake_load)
    observation = runner._evaluate(
        model,
        rows,
        root_fd=-1,
        runtime=SimpleNamespace(torch=torch),
        access=Counter(),
        device=torch.device("cpu"),
        update=0,
    )
    assert observation["all_registered_values_finite"] is True
    assert observation["aggregate"]["real_normalized_error"] == pytest.approx(
        [1.0] * 4, abs=1e-6
    )
    for name in ("action_gap", "hold_gap", "persistence_gap", "history_gap"):
        assert observation["aggregate"][name] == pytest.approx([0.0] * 4, abs=1e-6)


def test_runtime_adapters_are_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner.core, "_preflight", runner._CORE_PREFLIGHT)
    monkeypatch.setattr(runner.core, "_run", runner._CORE_RUN)
    monkeypatch.setattr(runner.core, "_decision", runner._CORE_DECISION)
    runner._install_runtime_adapters()
    assert runner.core._preflight is runner._preflight
    assert runner.core._run is runner._run
    assert runner.core._decision is runner._decision
