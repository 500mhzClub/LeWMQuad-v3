from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
EXECUTOR_PATH = ROOT / (
    "scripts/execute_go2_rgb_predictor_core_protected_survival_output_"
    "joint_jepa_v24.py"
)
TRAINING_PATH = ROOT / (
    "scripts/run_go2_rgb_predictor_core_protected_survival_output_joint_jepa_v24.py"
)


def _load(path: Path, name: str) -> Any:
    import sys

    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _executor(name: str) -> Any:
    return _load(EXECUTOR_PATH, name)


class _SizedParameter:
    def __init__(self, size: int) -> None:
        self.size = size

    def numel(self) -> int:
        return self.size


def _model(executor: Any, *, corrupt_core_name: bool = False) -> Any:
    named: list[tuple[str, _SizedParameter]] = []
    encoder_sizes = (*([1] * 79), 3_102_730)
    evidence_sizes = (*([1] * 7), 8)
    representation_sizes = (1, 1, 1, 1, 1, 3_515)
    named.extend(
        (f"encoder.p{index}", _SizedParameter(size))
        for index, size in enumerate(encoder_sizes)
    )
    named.extend(
        (f"bev_lift.evidence_head.p{index}", _SizedParameter(size))
        for index, size in enumerate(evidence_sizes)
    )
    named.extend(
        (f"bev_lift.point_projection.p{index}", _SizedParameter(size))
        for index, size in enumerate(representation_sizes[:3])
    )
    named.extend(
        (f"bev_lift.volume_block.p{index}", _SizedParameter(size))
        for index, size in enumerate(representation_sizes[3:])
    )
    named.extend(
        (f"semantic_head.p{index}", _SizedParameter(size))
        for index, size in enumerate((1, 1, 1, 1, 1, 73_981))
    )
    core_names = list(executor.PROTECTED_PREDICTOR_CORE_PARAMETER_NAMES)
    if corrupt_core_name:
        core_names[0] = "predictor.unregistered_core.weight"
    named.extend(
        (name, _SizedParameter(size))
        for name, size in zip(
            core_names,
            executor.PROTECTED_PREDICTOR_CORE_PARAMETER_SIZES,
            strict=True,
        )
    )
    named.extend(
        (
            (executor.SWEPT_PROGRESS_OUTPUT_PARAMETER_NAMES[0], _SizedParameter(64)),
            (executor.SWEPT_PROGRESS_OUTPUT_PARAMETER_NAMES[1], _SizedParameter(1)),
            ("target_encoder.p0", _SizedParameter(1)),
            ("target_bev_lift.volume_block.p0", _SizedParameter(1)),
        )
    )
    return SimpleNamespace(named_parameters=lambda: iter(named))


def _diagnostics() -> dict[str, float | int]:
    return {
        "positive_energy_sum": 12.8,
        "positive_energy_count": 128,
        "positive_energy_mean": 0.1,
        "scene_negative_energy_sum": 20.0,
        "scene_eligible_count": 100,
        "scene_negative_energy_mean": 0.2,
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


def _inherited_routes() -> dict[str, dict[str, float | int]]:
    counts = {
        "camera_shared": 88,
        "joint_shared": 88,
        "representation": 12,
        "predictor": 15,
    }
    return {
        name: {
            "preclip_l2": 1.0,
            "applied_scale": 1.0,
            "parameter_tensor_count": count,
            "absent_tensor_gradient_count": 0,
        }
        for name, count in counts.items()
    }


def _result(executor: Any) -> Any:
    losses = {
        "S": 0.1,
        "P": 0.2,
        "U": 0.3,
        "R": 0.4,
        "O": 0.5,
        "F": 0.1,
        "J_rank": 0.9,
        "J24": 1.0,
        "N": 1.5,
        "C": 0.25,
        "L": 2.75,
    }
    route = {
        "preclip_l2": 2.0,
        "applied_scale": 0.5,
        "parameter_tensor_count": 96,
        "absent_tensor_gradient_count": 0,
    }
    return SimpleNamespace(
        accounting={
            name: multiplier
            for name, multiplier in executor.ACCOUNTING_MULTIPLIERS_V24.items()
        },
        gradient_routes={
            **_inherited_routes(),
            executor.PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME: route,
        },
        mean_losses=losses,
        predictor_core_protected_survival_diagnostics=_diagnostics(),
        ranking_active_microbatches=4,
        ranking_eligible_pairs=10,
        survival_supervised_decisions=20,
        target_gradient_tensor_count=0,
        optimizer_steps_this_update=1,
        ema_steps_this_update=1,
    )


def test_denied_shell_binds_frozen_v23_and_preserves_caps_and_gates(capsys) -> None:
    executor = _executor("_v24_executor_denial")
    assert executor.main([]) == 4
    assert "DENIED_SOURCE_ONLY" in capsys.readouterr().out
    private = executor.private_adapter_receipt_v24()
    assert private["base_executor_file_sha256"] == (
        "9f816eff5353984cd8335de49bb914f23fe4affeefcdeb0cab40b210e6db1884"
    )
    assert private["base_executor_byte_count"] == 31_407
    assert private["j24_parameter_tensor_count"] == 96
    assert private["j24_parameter_count"] == 3_106_409
    assert private["protected_predictor_core_parameter_tensor_count"] == 13
    assert private["protected_predictor_core_parameter_count"] == 259_008
    assert executor.MAXIMUM_UPDATES == executor._v23.MAXIMUM_UPDATES == 1_000
    assert executor.MAXIMUM_PRESENTATIONS == executor._v23.MAXIMUM_PRESENTATIONS == 16_000
    assert executor.OBSERVATION_UPDATES == (0, 100, 400, 1_000)
    assert executor.TERMINAL_UPDATES == (400, 1_000)
    assert executor.FINAL_PHYSICAL_THRESHOLDS == executor._v23.FINAL_PHYSICAL_THRESHOLDS
    assert executor.MATCHED_UPDATE400_THRESHOLDS == (
        executor._v23.MATCHED_UPDATE400_THRESHOLDS
    )


def test_actual_training_api_has_exact_v24_route_and_no_batch_change() -> None:
    executor = _executor("_v24_executor_training_api")
    training = _load(TRAINING_PATH, "_v24_training_for_executor")
    receipt = executor.validate_training_api_v24(training)
    assert receipt["new_batch_fields_over_predecessor"] == 0
    assert receipt["predictor_core_protected_survival_route"] == (
        executor.PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME
    )
    assert receipt["j24_parameter_tensor_count"] == 96
    assert receipt["j24_parameter_count"] == 3_106_409
    assert receipt["protected_predictor_core_parameter_tensor_count"] == 13
    assert receipt["protected_predictor_core_parameter_count"] == 259_008
    assert receipt["backward_calls_per_update"] == 12
    assert "required_batch_key_count_v23" not in receipt
    assert "state_residual_survival_route" not in receipt


def test_exact_included_and_protected_parameter_inventory_is_bound() -> None:
    executor = _executor("_v24_executor_inventory")
    inventory = executor._validate_predictor_core_protected_parameter_subset(
        _model(executor)
    )
    assert inventory["included_parameter_tensor_count"] == 96
    assert inventory["included_parameter_count"] == 3_106_409
    assert inventory["protected_parameter_tensor_count"] == 13
    assert inventory["protected_parameter_count"] == 259_008
    assert inventory["protected_parameter_names"] == (
        executor.PROTECTED_PREDICTOR_CORE_PARAMETER_NAMES
    )
    with pytest.raises(RuntimeError, match="parameter subset changed"):
        executor._validate_predictor_core_protected_parameter_subset(
            _model(executor, corrupt_core_name=True)
        )


def test_integrity_uses_private_projection_and_publishes_only_v24_evidence() -> None:
    executor = _executor("_v24_executor_integrity")
    result = _result(executor)
    runtime = SimpleNamespace(torch=torch)
    projected: dict[str, Any] = {}
    original = executor._original_validate_update_integrity

    def inherited_validator(_runtime, _model, compatibility, **_kwargs):
        projected["result"] = compatibility
        assert compatibility.mean_losses["J23"] == result.mean_losses["J24"]
        assert "J24" not in compatibility.mean_losses
        route = compatibility.gradient_routes[
            executor._v23.STATE_RESIDUAL_SURVIVAL_ROUTE_NAME
        ]
        assert route["parameter_tensor_count"] == 109
        assert compatibility.accounting["state_residual_survival_grad_calls"] == 4
        _runtime.state_residual_survival_diagnostics_v23 = {
            1: dict(compatibility.state_residual_survival_diagnostics)
        }
        return {
            "schema": f"{executor.SCHEMA_PREFIX}_update_integrity_v1",
            "update": 1,
            "passed": True,
            "gradient_routes": {
                **_inherited_routes(),
                executor._v23.STATE_RESIDUAL_SURVIVAL_ROUTE_NAME: route,
            },
            "state_residual_survival_diagnostics": _diagnostics(),
            "v23_action_prior_residualized_wrong_scene_survival_output": {
                "private": True
            },
        }

    executor._original_validate_update_integrity = inherited_validator
    try:
        receipt = executor.validate_update_integrity_v24(
            runtime,
            _model(executor),
            result,
            update=1,
            access_receipt={},
        )
    finally:
        executor._original_validate_update_integrity = original

    assert projected
    assert not hasattr(runtime, "state_residual_survival_diagnostics_v23")
    assert runtime.predictor_core_protected_survival_diagnostics_v24[1] == (
        _diagnostics()
    )
    assert "state_residual_survival_diagnostics" not in receipt
    assert "v23_action_prior_residualized_wrong_scene_survival_output" not in receipt
    assert "J23" not in receipt["mean_losses"]
    assert receipt["mean_losses"]["J24"] == 1.0
    assert executor._v23.STATE_RESIDUAL_SURVIVAL_ROUTE_NAME not in (
        receipt["gradient_routes"]
    )
    route = receipt["gradient_routes"][
        executor.PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME
    ]
    assert route["parameter_tensor_count"] == 96
    mechanism = receipt["v24_predictor_core_protected_survival_output"]
    assert mechanism["protected_predictor_core_parameter_tensor_count"] == 13
    assert mechanism["protected_predictor_core_parameter_count"] == 259_008
    assert mechanism["protected_predictor_core_parameter_names"] == list(
        executor.PROTECTED_PREDICTOR_CORE_PARAMETER_NAMES
    )
    assert mechanism["predictor_core_gradient_from_j24"] is False
    assert mechanism["predictor_core_gradient_from_inherited_joint"] is True
    assert mechanism["swept_progress_output_gradient_from_j24"] is True


def test_integrity_rejects_v23_route_size_or_wrong_core_membership() -> None:
    executor = _executor("_v24_executor_fail_closed")
    bad_route = _result(executor)
    bad_route.gradient_routes[
        executor.PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME
    ]["parameter_tensor_count"] = 109
    with pytest.raises(RuntimeError, match="route failed integrity"):
        executor.validate_update_integrity_v24(
            SimpleNamespace(torch=torch),
            _model(executor),
            bad_route,
            update=1,
            access_receipt={},
        )

    with pytest.raises(RuntimeError, match="parameter subset changed"):
        executor.validate_update_integrity_v24(
            SimpleNamespace(torch=torch),
            _model(executor, corrupt_core_name=True),
            _result(executor),
            update=1,
            access_receipt={},
        )


def test_observation_relabels_without_v23_diagnostics_or_rescoring() -> None:
    executor = _executor("_v24_executor_observation")
    runtime = SimpleNamespace(
        predictor_core_protected_survival_diagnostics_v24={100: _diagnostics()}
    )
    original = executor._original_observation
    executor._original_observation = lambda *args, **kwargs: {
        "update": 100,
        "integrity_pass": True,
        "scene_innovation_diagnostics": {"private_compatibility": True},
        "controls": {"kept": True},
    }
    try:
        observed = executor.observation_v24(
            runtime, object(), update=100, integrity_pass=True
        )
    finally:
        executor._original_observation = original
    assert "scene_innovation_diagnostics" not in observed
    assert observed["predictor_core_protected_survival_diagnostics"] == _diagnostics()
    assert observed["controls"] == {"kept": True}
