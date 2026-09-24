from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

v22 = importlib.import_module(
    "scripts.execute_go2_rgb_scene_action_contrastive_innovation_joint_jepa_v22"
)
v21_fixture = importlib.import_module(
    "lewm.tests.test_execute_go2_rgb_same_action_cross_scene_contrastive_"
    "innovation_joint_jepa_v21"
)


def _accounting(update: int) -> dict[str, int]:
    return {
        name: update * multiplier
        for name, multiplier in v22.ACCOUNTING_MULTIPLIERS_V22.items()
    }


def _losses() -> dict[str, float]:
    values = {
        "S": 0.1,
        "P": 0.2,
        "U": 0.3,
        "R": 0.4,
        "O": 0.5,
        "I_fit": 1.0,
        "I_scene_rank": 0.6,
        "I_action_rank": 0.4,
        "C": 0.6,
    }
    values["I_two_axis"] = values["I_fit"] + 0.5 * (
        values["I_scene_rank"] + values["I_action_rank"]
    )
    values["N"] = sum(values[name] for name in ("S", "P", "U", "R", "O"))
    values["L"] = values["N"] + values["C"] + values["I_two_axis"]
    return values


def _diagnostics() -> dict[str, float | int]:
    return {
        "positive_energy_mean": 1.0,
        "scene_negative_energy_mean": 1.25,
        "scene_advantage_sum": 4.0,
        "scene_advantage_count": 16,
        "scene_advantage_mean": 0.25,
        "action_negative_energy_mean": 1.5,
        "action_advantage_sum": 8.0,
        "action_advantage_count": 16,
        "action_advantage_mean": 0.5,
        "nonrequested_action_count_per_row": 8,
        "action_candidate_energy_count": 128,
        "matching_predictor_gradient_cosine": 0.1,
        "valid_cell_count": 3_000,
        "high_salience_cell_count": 128,
        "low_salience_cell_count": 128,
    }


def _result(update: int = 2) -> SimpleNamespace:
    inherited_route = {
        "preclip_l2": 0.5,
        "applied_scale": 1.0,
        "parameter_tensor_count": 1,
        "absent_tensor_gradient_count": 0,
    }
    innovation_route = {
        "preclip_l2": 2.0,
        "applied_scale": 0.5,
        "parameter_tensor_count": 13,
        "absent_tensor_gradient_count": 0,
    }
    return SimpleNamespace(
        accounting=_accounting(update),
        gradient_routes={
            "camera_shared": dict(inherited_route),
            "joint_shared": dict(inherited_route),
            "representation": dict(inherited_route),
            "predictor": dict(inherited_route),
            "two_axis_innovation_predictor": innovation_route,
        },
        mean_losses=_losses(),
        two_axis_innovation_diagnostics=_diagnostics(),
        ranking_active_microbatches=4,
        ranking_eligible_pairs=8,
        survival_supervised_decisions=16,
        target_gradient_tensor_count=0,
        optimizer_steps_this_update=1,
        ema_steps_this_update=1,
    )


def test_v22_identity_predecessor_bindings_and_denied_shell(capsys) -> None:
    assert v22.SCHEMA_PREFIX.endswith("scene_action_contrastive_innovation_joint_jepa_v22")
    assert v22.OUTPUT_ROOT_RELATIVE_PATH.endswith("joint_jepa_v22/attempt_v1")
    assert v22.CURRENT_EXECUTION_AUTHORIZED is False
    assert v22.main([]) == 4
    denied = json.loads(capsys.readouterr().out)
    assert denied["status"] == "DENIED_SOURCE_ONLY"
    receipt = v22.private_adapter_receipt_v22()
    assert receipt["preregistration_commit"] == "43053ae49c28082c616f45ed857eedb727380952"
    assert receipt["v21_scientific_result_commit"] == "e5b5e56b30cee0c1eb818d52c4d886909f570f4d"
    assert receipt["new_batch_fields_over_v21"] == 0
    expected = {
        v22.PREREGISTRATION_PATH: (
            v22.PREREGISTRATION_FILE_SHA256,
            v22.PREREGISTRATION_BYTE_COUNT,
        ),
        v22.V21_SCIENTIFIC_RESULT_PATH: (
            v22.V21_SCIENTIFIC_RESULT_FILE_SHA256,
            v22.V21_SCIENTIFIC_RESULT_BYTE_COUNT,
        ),
    }
    for path, binding in expected.items():
        assert v22.BOUND_PARENT_SOURCES[path] == binding
    assert v22.validate_bound_sources_v22(ROOT, expected)["validated_path_count"] == 2


def test_update_integrity_projects_once_through_v21_and_publishes_both_axes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    captured: list[object] = []

    def inherited(runtime, model, result, **kwargs):
        captured.append(result)
        return v21_fixture._inherited_receipt()

    monkeypatch.setattr(v22, "_original_validate_update_integrity", inherited)
    runtime = SimpleNamespace(torch=torch)
    receipt = v22.validate_update_integrity_v22(
        runtime,
        v21_fixture._Model(),
        _result(),
        update=2,
        access_receipt={},
    )
    projected = captured[0]
    assert projected.accounting["scene_innovation_grad_calls"] == 8
    assert "two_axis_innovation_grad_calls" not in projected.accounting
    assert projected.mean_losses["I_rank"] == pytest.approx(0.5)
    assert projected.mean_losses["I_scene"] == pytest.approx(1.5)
    assert set(receipt["gradient_routes"]) == {
        "camera_shared",
        "joint_shared",
        "representation",
        "predictor",
        "two_axis_innovation_predictor",
    }
    assert receipt["mean_losses"] == _losses()
    assert receipt["two_axis_innovation_diagnostics"] == _diagnostics()
    assert "scene_innovation_diagnostics" not in receipt
    assert runtime.two_axis_innovation_diagnostics_v22 == {2: _diagnostics()}
    marker = receipt["v22_scene_action_contrastive_innovation"]
    assert marker["parameter_tensor_count"] == 13
    assert marker["nonrequested_action_count_per_row"] == 8
    assert marker["target_gradient_from_i_two_axis"] is False


def test_terminal_accounting_and_inherited_gates_remain_exact() -> None:
    expected = v22.validate_terminal_accounting_v22(
        _accounting(400), terminal_update=400
    )
    assert expected["presentations"] == 6_400
    assert expected["two_axis_innovation_grad_calls"] == 1_600
    assert expected["two_axis_innovation_objectives"] == 1_600
    assert v22.TERMINAL_UPDATES == (400, 1_000)
    assert v22.MAXIMUM_UPDATES == 1_000
    assert v22.MAXIMUM_PRESENTATIONS == 16_000


def test_observation_replaces_inherited_scene_surface_at_update_zero_and_later(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        v22,
        "_original_observation",
        lambda *args, **kwargs: {
            "update": kwargs["update"],
            "scene_innovation_diagnostics": None,
        },
    )
    initial = v22.observation_v22(
        SimpleNamespace(), object(), update=0, integrity_pass=True
    )
    assert initial == {"update": 0, "two_axis_innovation_diagnostics": None}
    runtime = SimpleNamespace(two_axis_innovation_diagnostics_v22={100: _diagnostics()})
    observed = v22.observation_v22(
        runtime, object(), update=100, integrity_pass=True
    )
    assert "scene_innovation_diagnostics" not in observed
    assert observed["two_axis_innovation_diagnostics"] == _diagnostics()
