from __future__ import annotations

from copy import deepcopy
from collections import Counter
import hashlib
from types import SimpleNamespace

import pytest
import torch

from lewm.models.go2_rgb_fixed_teacher_trajectory_distribution_h4_jepa_v1 import (
    JointRecurrentH4JEPA,
    JointRecurrentH4JEPAConfig,
)

from scripts import (
    run_go2_rgb_fixed_teacher_trajectory_distribution_h4_jepa_v1 as runner,
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
    target_rank: float = 0.17,
    online_rank: float = 0.20,
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


def test_binding_preserves_exact_capped_joint_jepa_science() -> None:
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
            "trajectory_h4_wrapper": {
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
        assert runner.core.ADDITIONAL_SCIENCE["support"].startswith("four_equal_mass")
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
    monkeypatch.delenv("LEWM_TRAJECTORY_H4_WRAPPER_SHA256", raising=False)
    monkeypatch.delenv("LEWM_TRAJECTORY_H4_WRAPPER_BYTES", raising=False)
    with pytest.raises(runner.core.ContractError):
        runner._verify_source_closure()

    raw = runner.Path(runner.__file__).read_bytes()
    monkeypatch.setenv(
        "LEWM_TRAJECTORY_H4_WRAPPER_SHA256", hashlib.sha256(raw).hexdigest()
    )
    monkeypatch.setenv("LEWM_TRAJECTORY_H4_WRAPPER_BYTES", str(len(raw)))
    closure = runner._verify_source_closure()
    assert set(closure) == {
        "trajectory_h4_wrapper",
        "shared_runner",
        "trajectory_h4_model",
        "dense_h4_model_dependency",
        "inherited_v1_model",
        "encoder_dependency",
    }


def test_trajectory_decision_passes_only_the_full_registered_pattern() -> None:
    result = runner._trajectory_decision(_passing_observations(), 1_000)
    assert result["decision"] == runner.PASS_DECISION
    assert result["failed_gates"] == []
    assert result["diagnostics"]["selected_update"] == 750

    collapsed = _passing_observations()
    collapsed[1]["aggregate"]["normalized_pairwise_spread"][3] = 0.0
    stopped = runner._trajectory_decision(collapsed, 1_000)
    assert stopped["decision"] == runner.STOP_DECISION
    assert (
        "h4_normalized_pairwise_spread_at_least_point05"
        in stopped["failed_gates"]
    )


def test_trajectory_decision_rejects_teacher_drift_and_incomplete_cap() -> None:
    observations = _passing_observations()
    observations[1]["noncollapse"]["target_effective_rank_ratio"] = 0.16
    drift = runner._trajectory_decision(observations, 1_000)
    assert drift["decision"] == runner.STOP_DECISION
    assert "fixed_teacher_metric_geometry_unchanged" in drift["failed_gates"]

    incomplete = runner._trajectory_decision(_passing_observations(), 999)
    assert incomplete["decision"] == runner.STOP_DECISION
    assert "completed_exact_cap" in incomplete["failed_gates"]


def test_run_adapter_relabels_initialization_and_selection_receipts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = {
        "fresh_recurrent_and_predictor_initialization": True,
        "unchanged": 7,
    }

    def fake_run(*_args: object, **_kwargs: object) -> tuple[dict, dict, dict]:
        return {"selection_rule": "old"}, artifact, {"decision": "x"}

    monkeypatch.setattr(runner, "_CORE_RUN", fake_run)
    metrics, adapted, decision = runner._trajectory_run("ignored")
    assert "joint-plus-marginal" in metrics["selection_rule"]
    assert decision == {"decision": "x"}
    assert "fresh_recurrent_and_predictor_initialization" not in adapted
    assert adapted[
        "fresh_dense_history_mode_embeddings_action_path_and_shared_delta_head_initialization"
    ] is True
    assert adapted["unchanged"] == 7


def test_runtime_adapter_installs_custom_evaluation_decision_and_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner.core, "_evaluate", runner._CORE_EVALUATE)
    monkeypatch.setattr(runner.core, "_decision", runner._CORE_DECISION)
    monkeypatch.setattr(runner.core, "_run", runner._CORE_RUN)
    runner._install_runtime_adapters()
    assert runner.core._evaluate is runner._trajectory_evaluate
    assert runner.core._decision is runner._trajectory_decision
    assert runner.core._run is runner._trajectory_run


def test_custom_evaluator_update_zero_is_exact_distributional_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(63)
    model = JointRecurrentH4JEPA(
        config=JointRecurrentH4JEPAConfig(
            image_size=8,
            patch_size=4,
            feature_dim=12,
            encoder_depth=1,
            encoder_heads=3,
            recurrent_spatial_heads=3,
            cross_attention_heads=3,
        )
    )
    rows = [
        SimpleNamespace(family=family, scene_id=f"scene_{index}")
        for index, family in enumerate(runner.core.FAMILIES)
    ]
    rgb = torch.randn(len(rows), 7, 3, 8, 8)
    actions = torch.tensor(
        [[index % 9 for index in range(6)] for _row in rows], dtype=torch.long
    )

    def fake_load_batch(
        batch_rows: object,
        **_kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(batch_rows) == len(rows)  # type: ignore[arg-type]
        return rgb, actions

    monkeypatch.setattr(runner.core, "_load_batch", fake_load_batch)
    observation = runner._trajectory_evaluate(
        model,
        rows,
        root_fd=-1,
        runtime=SimpleNamespace(torch=torch),
        access=Counter(),
        device=torch.device("cpu"),
        update=0,
    )
    assert observation["all_registered_values_finite"] is True
    assert observation["aggregate"][
        "joint_trajectory_normalized_energy_score"
    ] == pytest.approx(1.0, abs=1e-6)
    assert observation["aggregate"][
        "combined_normalized_energy_score"
    ] == pytest.approx(1.0, abs=1e-6)
    assert observation["aggregate"][
        "combined_distribution_value_gap"
    ] == pytest.approx(0.0, abs=1e-6)
    assert observation["aggregate"]["real_normalized_energy_score"] == pytest.approx(
        [1.0] * 4, abs=1e-6
    )
    for name in (
        "action_gap",
        "hold_gap",
        "persistence_gap",
        "history_gap",
        "distribution_value_gap",
        "normalized_pairwise_spread",
    ):
        assert observation["aggregate"][name] == pytest.approx([0.0] * 4, abs=1e-6)
