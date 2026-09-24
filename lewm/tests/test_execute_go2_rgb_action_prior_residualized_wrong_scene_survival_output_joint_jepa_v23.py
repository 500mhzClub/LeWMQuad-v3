from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
EXECUTOR_PATH = ROOT / (
    "scripts/execute_go2_rgb_action_prior_residualized_wrong_scene_survival_"
    "output_joint_jepa_v23.py"
)
TRAINING_PATH = ROOT / (
    "scripts/run_go2_rgb_action_prior_residualized_wrong_scene_survival_"
    "output_joint_jepa_v23.py"
)
V21_FIXTURE_PATH = ROOT / (
    "lewm/tests/test_run_go2_rgb_same_action_cross_scene_contrastive_"
    "innovation_joint_jepa_v21.py"
)


def _load(path: Path, name: str):
    import sys

    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _executor(name: str = "_v23_executor_test"):
    return _load(EXECUTOR_PATH, name)


def _parameters(counts: list[int]) -> tuple[torch.nn.Parameter, ...]:
    return tuple(torch.nn.Parameter(torch.zeros(count)) for count in counts)


def _model():
    encoder = _parameters([1] * 79 + [3_102_730])
    evidence = _parameters([1] * 7 + [8])
    representation_live = _parameters([1] * 5 + [3_515])
    semantic = _parameters([1] * 5 + [73_981])
    predictor = _parameters([1] * 12 + [258_996, 64, 1])
    names = (
        *(f"encoder.p{i}" for i in range(len(encoder))),
        *(f"bev_lift.evidence_head.p{i}" for i in range(len(evidence))),
        *(f"bev_lift.point_projection.p{i}" for i in range(3)),
        *(f"bev_lift.volume_block.p{i}" for i in range(3)),
        *(f"semantic_head.p{i}" for i in range(6)),
        *(f"predictor.core.p{i}" for i in range(len(predictor) - 2)),
        "predictor.swept_progress_head.output.weight",
        "predictor.swept_progress_head.output.bias",
    )
    parameters = (*encoder, *evidence, *representation_live, *semantic, *predictor)
    assert len(names) == len(parameters)
    return SimpleNamespace(named_parameters=lambda: iter(zip(names, parameters)))


def _diagnostics():
    return {
        "positive_energy_sum": 12.8,
        "positive_energy_count": 128,
        "positive_energy_mean": 0.1,
        "scene_negative_energy_sum": 20.0,
        "scene_eligible_count": 100,
        "scene_negative_energy_mean": 0.2,
        # Deliberately differs from scene-negative mean minus the global
        # positive mean: the scene arm covers only its eligible subset.
        "scene_advantage_sum": 5.0,
        "scene_advantage_mean": 0.05,
        "scene_rank_sum": 90.0,
        "prior_negative_energy_sum": 19.2,
        "prior_eligible_count": 128,
        "prior_negative_energy_mean": 0.15,
        "prior_advantage_sum": 6.4,
        "prior_advantage_mean": 0.05,
        "prior_rank_sum": 100.0,
        "non_hold_action_count_per_row": 8,
    }


def test_denied_shell_and_exact_identity(capsys) -> None:
    executor = _executor("_v23_executor_denial")
    assert executor.main([]) == 4
    assert "DENIED_SOURCE_ONLY" in capsys.readouterr().out
    assert executor.PREREGISTRATION_COMMIT == (
        "a7cf9692dd93212a82cb598d3175ff1c3598941b"
    )
    assert executor.TRAINING_REQUIRED_BATCH_KEYS_V23[-1] == (
        executor.ACTION_PRIOR_M_KEY
    )
    assert executor.STATE_RESIDUAL_SURVIVAL_PARAMETER_COUNT == 3_365_417


def test_actual_training_api_is_the_one_field_extension() -> None:
    executor = _executor("_v23_executor_api")
    training = _load(TRAINING_PATH, "_v23_training_for_executor")
    receipt = executor.validate_training_api_v23(training)
    assert receipt["new_batch_fields_over_v21"] == 1
    assert receipt["state_residual_survival_route"] == (
        executor.STATE_RESIDUAL_SURVIVAL_ROUTE_NAME
    )
    assert receipt["backward_calls_per_update"] == 12


def test_microbatch_validator_projects_prior_before_inherited_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = _executor("_v23_executor_microbatch")
    training = _load(TRAINING_PATH, "_v23_training_for_real_microbatch_validation")
    fixture = _load(V21_FIXTURE_PATH, "_v21_fixture_for_v23_executor")
    batches = tuple(
        {**batch, executor.ACTION_PRIOR_M_KEY: torch.zeros(9)}
        for batch in fixture._microbatches()
    )
    monkeypatch.setattr(
        executor._engine, "_validate_batch_query_identity_v13", lambda *_: None
    )
    executor.validate_microbatches_for_engine_v23(
        SimpleNamespace(torch=torch, training_module=training),
        object(),
        batches,
    )
    assert training._validate_microbatches_v21 is training._v21._validate_microbatches_v21


def test_integrity_validates_full_route_and_records_actual_diagnostics() -> None:
    executor = _executor("_v23_executor_integrity")
    base_routes = {
        name: {
            "preclip_l2": 1.0,
            "applied_scale": 1.0,
            "parameter_tensor_count": 1,
            "absent_tensor_gradient_count": 0,
        }
        for name in ("camera_shared", "joint_shared", "representation", "predictor")
    }
    auxiliary_route = {
        "preclip_l2": 2.0,
        "applied_scale": 0.5,
        "parameter_tensor_count": 109,
        "absent_tensor_gradient_count": 0,
    }
    accounting = {
        name: multiplier
        for name, multiplier in executor.ACCOUNTING_MULTIPLIERS_V23.items()
    }
    losses = {
        "S": 0.1,
        "P": 0.2,
        "U": 0.3,
        "R": 0.4,
        "O": 0.5,
        "F": 0.1,
        "J_rank": 0.9,
        "J23": 1.0,
        "N": 1.5,
        "C": 0.25,
        "L": 2.75,
    }
    result = SimpleNamespace(
        accounting=accounting,
        gradient_routes={
            **base_routes,
            executor.STATE_RESIDUAL_SURVIVAL_ROUTE_NAME: auxiliary_route,
        },
        mean_losses=losses,
        state_residual_survival_diagnostics=_diagnostics(),
        ranking_active_microbatches=4,
        ranking_eligible_pairs=10,
        survival_supervised_decisions=20,
        target_gradient_tensor_count=0,
        optimizer_steps_this_update=1,
        ema_steps_this_update=1,
    )
    runtime = SimpleNamespace(torch=torch)
    original = executor._original_validate_update_integrity
    def inherited_validator(_runtime, _model, projected, **_kwargs):
        # Exercise the exact inherited arithmetic check that the compatibility
        # projection must satisfy; the actual V23 diagnostics remain distinct.
        inherited = executor._base._validate_scene_diagnostics(
            projected.scene_innovation_diagnostics
        )
        assert inherited["positive_energy_mean"] == losses["F"]
        assert inherited["negative_energy_mean"] == losses["F"]
        assert inherited["advantage_mean"] == 0.0
        return {
            "update": 1,
            "passed": True,
            "gradient_routes": base_routes,
            "scene_innovation_diagnostics": inherited,
            "v21_same_action_cross_scene_contrastive_innovation": {},
        }

    executor._original_validate_update_integrity = inherited_validator
    try:
        receipt = executor.validate_update_integrity_v23(
            runtime,
            _model(),
            result,
            update=1,
            access_receipt={},
        )
    finally:
        executor._original_validate_update_integrity = original
    assert receipt["passed"] is True
    route = receipt["gradient_routes"][executor.STATE_RESIDUAL_SURVIVAL_ROUTE_NAME]
    assert route["parameter_tensor_count"] == 109
    mechanism = receipt[
        "v23_action_prior_residualized_wrong_scene_survival_output"
    ]
    assert mechanism["encoder_gradient_from_j23"] is True
    assert mechanism["survival_head_gradient_from_j23"] is True
    assert mechanism["semantic_head_gradient_from_j23"] is False
    assert runtime.state_residual_survival_diagnostics_v23[1] == _diagnostics()


def test_diagnostics_fail_closed_on_inconsistent_means() -> None:
    executor = _executor("_v23_executor_diagnostics")
    bad = _diagnostics()
    bad["prior_advantage_mean"] = 0.2
    with pytest.raises(RuntimeError, match="diagnostics are inconsistent"):
        executor._validate_state_residual_survival_diagnostics(bad)


def test_observation_only_relabels_cached_diagnostics_without_rescoring() -> None:
    executor = _executor("_v23_executor_observation")
    runtime = SimpleNamespace(
        state_residual_survival_diagnostics_v23={100: _diagnostics()}
    )
    original = executor._original_observation
    executor._original_observation = lambda *args, **kwargs: {
        "update": 100,
        "integrity_pass": True,
        "scene_innovation_diagnostics": {"legacy": True},
        "controls": {"kept": True},
    }
    try:
        observed = executor.observation_v23(
            runtime, object(), update=100, integrity_pass=True
        )
    finally:
        executor._original_observation = original
    assert "scene_innovation_diagnostics" not in observed
    assert observed["state_residual_survival_diagnostics"] == _diagnostics()
    assert observed["controls"] == {"kept": True}
