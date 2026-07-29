from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from scripts import (
    run_go2_rgb_swept_progress_survival_joint_jepa_v7_hierarchical_cnn_encoder
    as runner,
)


def _receipt(update: int, *, inactive: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "schema": "lewm_v7_hierarchical_cnn_post_backward_gradient_v1",
        "update": update,
        "gradient_l2": 1.0,
        "online_parameter_count": 1_994_880,
        "online_parameter_tensor_count": 2,
        "gradient_tensor_count": 2,
        "active_parameter_tensor_count": 2 - len(inactive),
        "inactive_parameter_suffixes": list(inactive),
        "parameter_suffix_inventory_sha256": runner._inventory_sha256_v7(("a", "b")),
        "target_gradient_tensor_count": 0,
    }


def test_v7_wrapper_inherits_exact_v3_loss_schedule_and_cap() -> None:
    for name in (
        "ACTION_ORDER", "MICROBATCH_SIZE", "MICROBATCHES_PER_UPDATE",
        "PRESENTATIONS_PER_UPDATE", "MAXIMUM_UPDATES", "MAXIMUM_PRESENTATIONS",
        "OCCUPIED_CLASS_INDEX", "OCCUPIED_SAFETY_AUX_COEFFICIENT",
        "OCCUPIED_SAFETY_AUX_NORMALIZATION",
    ):
        assert getattr(runner, name) == getattr(runner._v3, name)
    assert runner.HIERARCHICAL_CNN_ENCODER_TRAINABLE_PARAMETER_COUNT_V7 == 1_994_880


def test_activity_summary_proves_every_online_tensor_activates() -> None:
    receipts = [_receipt(1, inactive=("b",))]
    receipts.extend(_receipt(update) for update in range(2, 1_001))
    observed = runner._summarize_encoder_activity_v7(receipts, ("a", "b"))
    assert observed["update_count"] == 1_000
    assert observed["all_online_parameter_tensors_received_gradient"] is True
    assert observed["latest_first_active_update"] == 2
    assert observed["minimum_active_parameter_tensor_count"] == 1
    assert observed["target_gradient_tensor_count"] == 0


def test_activity_summary_fails_if_one_tensor_never_activates() -> None:
    receipts = [_receipt(update, inactive=("b",)) for update in range(1, 1_001)]
    with pytest.raises(RuntimeError, match="b"):
        runner._summarize_encoder_activity_v7(receipts, ("a", "b"))


def test_gradient_receipt_checks_complete_online_and_frozen_target_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Parameter:
        def __init__(self, count: int, requires_grad: bool, gradient: torch.Tensor | None) -> None:
            self._count = count
            self.requires_grad = requires_grad
            self.grad = gradient

        def numel(self) -> int:
            return self._count

    online = (
        Parameter(1_000_000, True, torch.tensor([3.0])),
        Parameter(994_880, True, torch.tensor([4.0])),
    )
    target = (
        Parameter(1_000_000, False, None),
        Parameter(994_880, False, None),
    )
    model = SimpleNamespace(
        named_parameters=lambda: iter((
            ("encoder.a", online[0]), ("encoder.b", online[1]),
            ("target_encoder.a", target[0]), ("target_encoder.b", target[1]),
        )),
        ema_update_count=torch.tensor(1),
    )
    monkeypatch.setattr(runner._v3._v2._v1, "_runtime_apis", lambda: (torch, None, None))
    receipt = runner.cnn_encoder_gradient_receipt_v7(model)
    assert receipt["gradient_l2"] == 5.0
    assert receipt["active_parameter_tensor_count"] == 2
    assert receipt["target_gradient_tensor_count"] == 0


def test_driver_delegates_each_update_to_v3_and_preserves_losses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    inherited_result = SimpleNamespace(
        accounting=runner.JointTrainingAccountingV1(),
        mean_losses={name: float(index) for index, name in enumerate(("S", "P", "U", "R", "O", "L"), start=1)},
        gradient_l2={"encoder": 1.0, "lift_semantic": 2.0, "predictor": 3.0},
        ranking_active_microbatches=0,
        ranking_eligible_pairs=0,
        survival_supervised_decisions=0,
    )

    def fake_joint(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        del args, kwargs
        calls += 1
        return inherited_result

    def fake_core(*args: Any, joint_update: Any, **kwargs: Any) -> Any:
        del args, kwargs
        rows = []
        for update in range(1, 1_001):
            result = joint_update(None, None, (), accounting=runner.JointTrainingAccountingV1())
            rows.append({
                "update": update, "presentations": update * 16,
                "losses": dict(result.mean_losses), "gradient_l2": dict(result.gradient_l2),
            })
        return runner.JointTrainingAccountingV1(updates=1_000), tuple(rows), {"inherited": True}

    monkeypatch.setattr(runner, "_encoder_parameter_inventory_v7", lambda model: ((('a', object()), ('b', object())), ()))
    monkeypatch.setattr(runner._v3, "joint_training_update_v3", fake_joint)
    monkeypatch.setattr(runner._v3._v2, "_run_fixed_training_core_v2", fake_core)
    monkeypatch.setattr(runner, "cnn_encoder_gradient_receipt_v7", lambda model: _receipt(calls))
    _, trace, diagnostics = runner.run_fixed_training_v7(None, None, None, (), None, (), None)
    assert calls == 1_000
    assert trace[0]["losses"] == inherited_result.mean_losses
    assert trace[-1]["losses"] == inherited_result.mean_losses
    assert diagnostics["inherited"] is True
    assert diagnostics["hierarchical_cnn_encoder"]["latest_first_active_update"] == 1


def test_source_contains_no_reimplemented_loss_or_backward_loop() -> None:
    source = __import__("pathlib").Path(runner.__file__).read_text()
    assert source.count("_v3.joint_training_update_v3(") == 1
    assert source.count("_v3._v2._run_fixed_training_core_v2(") == 1
    assert ".backward(" not in source
    assert "fine_rgb_branch" not in source
