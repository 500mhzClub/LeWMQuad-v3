from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
import torch

from scripts import run_go2_rgb_full_whitened_predictive_state_h4_jepa_v1 as runner


def test_source_closure_requires_external_wrapper_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LEWM_FULL_WHITENED_H4_WRAPPER_SHA256", raising=False)
    monkeypatch.delenv("LEWM_FULL_WHITENED_H4_WRAPPER_BYTES", raising=False)
    with pytest.raises(runner.core.ContractError):
        runner._verify_source_closure()
    raw = runner.Path(runner.__file__).read_bytes()
    monkeypatch.setenv(
        "LEWM_FULL_WHITENED_H4_WRAPPER_SHA256", hashlib.sha256(raw).hexdigest()
    )
    monkeypatch.setenv("LEWM_FULL_WHITENED_H4_WRAPPER_BYTES", str(len(raw)))
    closure = runner._verify_source_closure()
    assert set(closure) == {
        "full_whitened_h4_wrapper",
        "wdps_h4_wrapper_dependency",
        "shared_runner",
        "full_whitened_h4_model",
        "wdps_h4_model_dependency",
        "dense_h4_model_dependency",
        "inherited_v1_model",
        "encoder_dependency",
    }


def test_core_configuration_changes_only_the_registered_objective() -> None:
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
    core_names = base_names + (
        "TARGET_DESCRIPTION",
        "OBJECTIVE_DESCRIPTION",
        "ADDITIONAL_SCIENCE",
        "EXECUTION_SOURCE_BINDINGS",
    )
    base_original = {name: getattr(runner.base, name) for name in base_names}
    core_original = {name: getattr(runner.core, name) for name in core_names}
    try:
        bindings = {"full_whitened_h4_wrapper": {"path": "x", "byte_count": 1}}
        runner._configure_core(bindings)
        assert runner.core.MODEL_MODULE == runner.MODEL_MODULE
        assert runner.core.OUTPUT_ROOT == runner.OUTPUT_ROOT
        assert runner.core.SCHEMA == runner.SCHEMA
        assert "cross_covariance_identity" in runner.core.OBJECTIVE_DESCRIPTION
        assert "raw_training_mse" in runner.core.OBJECTIVE_DESCRIPTION
        assert runner.core.ADDITIONAL_SCIENCE["fixed_teacher_role"] == (
            "fixed_target_and_history_teacher"
        )
        assert runner.core.UPDATES == 1_000
        assert runner.core.BATCH_SIZE == 16
        assert runner.core.PRESENTATIONS == 16_000
        assert runner.core.SEED == 20_260_727
    finally:
        for name, value in base_original.items():
            setattr(runner.base, name, value)
        for name, value in core_original.items():
            setattr(runner.core, name, value)


def test_validation_geometry_records_exact_covariance_identity() -> None:
    scale = ((16 - 1) / 2.0) ** 0.5
    samples = torch.zeros(16, 8, dtype=torch.float64)
    for index in range(8):
        samples[2 * index, index] = scale
        samples[2 * index + 1, index] = -scale
    state = samples[:, None].expand(-1, 4, -1).contiguous()
    geometry = runner._state_geometry(state, SimpleNamespace(torch=torch))
    assert geometry["covariance_identity_error"] == pytest.approx([0.0] * 4)
    assert geometry["minimum_covariance_eigenvalue"] == pytest.approx([1.0] * 4)
    assert geometry["maximum_covariance_eigenvalue"] == pytest.approx([1.0] * 4)
    assert geometry["maximum_variance_error"] == pytest.approx([0.0] * 4)
    assert geometry["maximum_offdiagonal_covariance"] == pytest.approx([0.0] * 4)
    cross = runner._cross_covariance_geometry(
        state,
        state,
        SimpleNamespace(torch=torch),
    )
    assert cross["predicted_target_cross_covariance_identity_error"] == pytest.approx(
        [0.0] * 4
    )
    assert cross["predicted_target_maximum_cross_diagonal_error"] == pytest.approx(
        [0.0] * 4
    )
    assert cross[
        "predicted_target_maximum_offdiagonal_cross_covariance"
    ] == pytest.approx([0.0] * 4)


def test_evaluator_capture_attaches_paired_cross_covariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scale = ((16 - 1) / 2.0) ** 0.5
    samples = torch.zeros(16, 8)
    for index in range(8):
        samples[2 * index, index] = scale
        samples[2 * index + 1, index] = -scale
    predicted = samples[:, None].expand(-1, 4, -1).contiguous()
    target = predicted.clone()
    runtime = SimpleNamespace(torch=torch)

    def fake_evaluate(*_args: object, **_kwargs: object) -> dict:
        return {
            "state_geometry": {
                "predicted": runner._state_geometry(predicted, runtime),
                "target": runner._state_geometry(target, runtime),
            }
        }

    monkeypatch.setattr(runner, "_BASE_EVALUATE", fake_evaluate)
    result = runner._evaluate()
    for role in ("predicted", "target"):
        assert result["state_geometry"][role][
            "predicted_target_cross_covariance_identity_error"
        ] == pytest.approx([0.0] * 4)
    assert runner._PAIR_CAPTURE_ACTIVE is False
    assert runner._PAIR_CAPTURE_PENDING is None


def _eligible_observation() -> dict:
    geometry = {
        "participation_rank_ratio": [0.90] * 4,
        "minimum_std": [0.90] * 4,
        "maximum_std": [1.10] * 4,
        "maximum_abs_mean": [0.0] * 4,
        "covariance_identity_error": [0.10] * 4,
        "predicted_target_cross_covariance_identity_error": [0.10] * 4,
    }
    return {
        "all_registered_values_finite": True,
        "noncollapse": {
            "target_effective_rank_ratio": 0.20,
            "online_effective_rank_ratio": 0.20,
            "target_near_zero_variance_fraction": 0.0,
            "online_near_zero_variance_fraction": 0.0,
        },
        "state_geometry": {
            "predicted": dict(geometry),
            "target": dict(geometry),
        },
        "state_energy": {
            "near_zero_scene_denominator_count": [0] * 4,
            "predicted_rms": [1.0] * 4,
            "target_rms": [1.0] * 4,
            "predicted_mean_energy_fraction": [0.0] * 4,
            "target_mean_energy_fraction": [0.0] * 4,
        },
    }


def test_eligibility_requires_within_and_cross_covariance_identity() -> None:
    observation = _eligible_observation()
    assert runner._state_eligible(observation)
    observation["state_geometry"]["target"]["covariance_identity_error"] = [0.51] * 4
    assert not runner._state_eligible(observation)

    observation = _eligible_observation()
    observation["state_geometry"]["predicted"][
        "predicted_target_cross_covariance_identity_error"
    ] = [0.51] * 4
    assert not runner._state_eligible(observation)


def test_run_relabels_inherited_loss_receipts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bucket = {
        "state_similarity": 1.0,
        "predicted_variance": 0.0,
        "target_variance": 0.0,
        "predicted_mean": 0.1,
        "target_mean": 0.2,
        "predicted_covariance": 0.3,
        "target_covariance": 0.4,
        "history_teacher_alignment": 0.5,
        "total": 1.5,
    }
    metrics = {
        "training_losses": {
            "mean_over_completed_updates": dict(bucket),
            "last_completed_update": dict(bucket),
        }
    }
    monkeypatch.setattr(
        runner,
        "_BASE_RUN",
        lambda *_args, **_kwargs: (metrics, {"artifact": True}, {"decision": "x"}),
    )
    adapted, artifact, decision = runner._run("ignored")
    assert artifact == {"artifact": True}
    assert decision == {"decision": "x"}
    training = adapted["training_losses"]
    for receipt in (
        training["mean_over_completed_updates"],
        training["last_completed_update"],
    ):
        assert receipt["predicted_target_cross_covariance_identity"] == 1.0
        assert receipt["predicted_within_covariance_identity"] == 0.3
        assert receipt["target_within_covariance_identity"] == 0.4
        assert "state_similarity" not in receipt
        assert "predicted_variance" not in receipt
        assert "target_variance" not in receipt
    assert "raw_state_prediction_mse" in training["disabled_terms"]


def test_runtime_adapters_install_and_reject_foreign_hooks() -> None:
    base_build = runner.base._build_model
    base_evaluate = runner.base._evaluate
    base_eligible = runner.base._state_eligible
    base_geometry = runner.base._state_geometry
    core_hooks = {
        "_preflight": runner.core._preflight,
        "_run": runner.core._run,
        "_decision": runner.core._decision,
    }
    try:
        runner.base._build_model = runner._BASE_BUILD_MODEL
        runner.base._evaluate = runner._BASE_EVALUATE
        runner.base._state_eligible = runner._BASE_STATE_ELIGIBLE
        runner.base._state_geometry = runner._BASE_STATE_GEOMETRY
        runner.core._preflight = runner.base._CORE_PREFLIGHT
        runner.core._run = runner.base._CORE_RUN
        runner.core._decision = runner.base._CORE_DECISION
        runner._install_runtime_adapters()
        assert runner.base._build_model is runner._build_model
        assert runner.base._evaluate is runner._evaluate
        assert runner.base._state_eligible is runner._state_eligible
        assert runner.base._state_geometry is runner._state_geometry
        assert runner.core._preflight is runner.base._preflight
        assert runner.core._run is runner._run
        assert runner.core._decision is runner.base._decision

        runner.base._build_model = lambda *_args, **_kwargs: None
        with pytest.raises(runner.core.ContractError):
            runner._install_runtime_adapters()
    finally:
        runner.base._build_model = base_build
        runner.base._evaluate = base_evaluate
        runner.base._state_eligible = base_eligible
        runner.base._state_geometry = base_geometry
        for name, value in core_hooks.items():
            setattr(runner.core, name, value)
