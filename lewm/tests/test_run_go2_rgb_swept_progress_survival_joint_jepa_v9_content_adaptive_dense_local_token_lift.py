from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from scripts import (
    run_go2_rgb_swept_progress_survival_joint_jepa_v9_content_adaptive_dense_local_token_lift
    as runner,
)


def _receipt(update: int, *, inactive: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "schema": "lewm_v9_dense_local_attention_post_backward_gradient_v1",
        "update": update,
        "gradient_l2": 1.0,
        "online_parameter_count": 16_576,
        "online_parameter_tensor_count": 7,
        "gradient_tensor_count": 7,
        "active_parameter_tensor_count": 7 - len(inactive),
        "inactive_parameter_suffixes": list(inactive),
        "parameter_suffix_inventory_sha256": runner._inventory_sha256_v9(
            runner.ATTENTION_PARAMETER_SUFFIXES_V9
        ),
        "target_gradient_tensor_count": 0,
    }


def test_v9_wrapper_inherits_exact_v3_objective_schedule_and_cap() -> None:
    for name in (
        "ACTION_ORDER",
        "MICROBATCH_SIZE",
        "MICROBATCHES_PER_UPDATE",
        "PRESENTATIONS_PER_UPDATE",
        "MAXIMUM_UPDATES",
        "MAXIMUM_PRESENTATIONS",
        "OCCUPIED_CLASS_INDEX",
        "OCCUPIED_SAFETY_AUX_COEFFICIENT",
        "OCCUPIED_SAFETY_AUX_NORMALIZATION",
    ):
        assert getattr(runner, name) == getattr(runner._v3, name)
    assert runner.MAXIMUM_UPDATES == 1_000
    assert runner.MAXIMUM_PRESENTATIONS == 16_000
    assert runner.DENSE_LOCAL_ATTENTION_ADDED_PARAMETER_COUNT_V9 == 16_576
    assert runner.ATTENTION_PARAMETER_SUFFIXES_V9 == (
        "query_projection.weight",
        "query_projection.bias",
        "key_projection.weight",
        "value_projection.weight",
        "value_projection.bias",
        "output_projection.weight",
        "output_projection.bias",
    )


def test_attention_activity_requires_every_tensor_by_update_two() -> None:
    suffixes = runner.ATTENTION_PARAMETER_SUFFIXES_V9
    receipts = [_receipt(1, inactive=(suffixes[-1],))]
    receipts.extend(_receipt(update) for update in range(2, 1_001))
    observed = runner._summarize_attention_activity_v9(receipts, suffixes)
    assert observed["schema"] == (
        "lewm_v9_dense_local_attention_training_activity_v1"
    )
    assert observed["update_count"] == 1_000
    assert observed["all_online_parameter_tensors_active_by_update_2"] is True
    assert observed["latest_first_active_update"] == 2
    assert observed["first_active_update"][suffixes[-1]] == 2
    assert observed["target_gradient_tensor_count"] == 0


def test_attention_activity_fails_if_a_tensor_activates_after_update_two() -> None:
    suffix = runner.ATTENTION_PARAMETER_SUFFIXES_V9[-1]
    receipts = [
        _receipt(update, inactive=(suffix,) if update < 3 else ())
        for update in range(1, 1_001)
    ]
    with pytest.raises(RuntimeError, match="not active by update 2"):
        runner._summarize_attention_activity_v9(
            receipts, runner.ATTENTION_PARAMETER_SUFFIXES_V9
        )


def test_attention_gradient_receipt_covers_online_and_frozen_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AttentionProjections(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.query_projection = torch.nn.Linear(64, 64, bias=True)
            self.key_projection = torch.nn.Linear(64, 64, bias=False)
            self.value_projection = torch.nn.Linear(64, 64, bias=True)
            self.output_projection = torch.nn.Linear(64, 64, bias=True)

    online_attention = AttentionProjections()
    target_attention = AttentionProjections()
    target_attention.requires_grad_(False)
    for parameter in online_attention.parameters():
        parameter.grad = torch.ones_like(parameter)
    model = SimpleNamespace(
        named_parameters=lambda: iter(
            tuple(
                (f"bev_lift.{name}", parameter)
                for name, parameter in online_attention.named_parameters()
            )
            + tuple(
                (f"target_bev_lift.{name}", parameter)
                for name, parameter in target_attention.named_parameters()
            )
        ),
        ema_update_count=torch.tensor(1),
    )
    monkeypatch.setattr(
        runner._v3._v2._v1,
        "_runtime_apis",
        lambda: (torch, None, None),
    )
    receipt = runner.attention_gradient_receipt_v9(model)
    assert receipt["schema"] == (
        "lewm_v9_dense_local_attention_post_backward_gradient_v1"
    )
    assert receipt["online_parameter_count"] == 16_576
    assert receipt["online_parameter_tensor_count"] == 7
    assert receipt["active_parameter_tensor_count"] == 7
    assert receipt["target_gradient_tensor_count"] == 0


def test_attention_gradient_receipt_rejects_a_target_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AttentionProjections(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.query_projection = torch.nn.Linear(64, 64, bias=True)
            self.key_projection = torch.nn.Linear(64, 64, bias=False)
            self.value_projection = torch.nn.Linear(64, 64, bias=True)
            self.output_projection = torch.nn.Linear(64, 64, bias=True)

    online_attention = AttentionProjections()
    target_attention = AttentionProjections()
    target_attention.requires_grad_(False)
    for parameter in online_attention.parameters():
        parameter.grad = torch.ones_like(parameter)
    target_parameter = next(target_attention.parameters())
    target_parameter.grad = torch.ones_like(target_parameter)
    model = SimpleNamespace(
        named_parameters=lambda: iter(
            tuple(
                (f"bev_lift.{name}", parameter)
                for name, parameter in online_attention.named_parameters()
            )
            + tuple(
                (f"target_bev_lift.{name}", parameter)
                for name, parameter in target_attention.named_parameters()
            )
        ),
        ema_update_count=torch.tensor(1),
    )
    monkeypatch.setattr(
        runner._v3._v2._v1,
        "_runtime_apis",
        lambda: (torch, None, None),
    )
    with pytest.raises(RuntimeError, match="target dense-local attention"):
        runner.attention_gradient_receipt_v9(model)


def test_driver_delegates_every_update_and_preserves_inherited_losses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    inherited_result = SimpleNamespace(
        accounting=runner.JointTrainingAccountingV1(),
        mean_losses={
            name: float(index)
            for index, name in enumerate(("S", "P", "U", "R", "O", "L"), start=1)
        },
        gradient_l2={
            "encoder": 1.0,
            "lift_semantic": 2.0,
            "predictor": 3.0,
        },
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
            result = joint_update(
                None,
                None,
                (),
                accounting=runner.JointTrainingAccountingV1(),
            )
            rows.append(
                {
                    "update": update,
                    "presentations": update * 16,
                    "losses": dict(result.mean_losses),
                    "gradient_l2": dict(result.gradient_l2),
                }
            )
        return (
            runner.JointTrainingAccountingV1(updates=1_000),
            tuple(rows),
            {"inherited": True},
        )

    fake_inventory = tuple(
        (suffix, object()) for suffix in runner.ATTENTION_PARAMETER_SUFFIXES_V9
    )
    monkeypatch.setattr(
        runner,
        "_attention_parameter_inventory_v9",
        lambda model: (fake_inventory, ()),
    )
    monkeypatch.setattr(runner._v3, "joint_training_update_v3", fake_joint)
    monkeypatch.setattr(
        runner._v3._v2,
        "_run_fixed_training_core_v2",
        fake_core,
    )
    monkeypatch.setattr(
        runner,
        "attention_gradient_receipt_v9",
        lambda model: _receipt(calls),
    )
    _, trace, diagnostics = runner.run_fixed_training_v9(
        None,
        None,
        None,
        (),
        None,
        (),
        None,
    )
    assert calls == 1_000
    assert trace[0]["losses"] == inherited_result.mean_losses
    assert trace[-1]["losses"] == inherited_result.mean_losses
    assert diagnostics["inherited"] is True
    key = "dense_local_attention"
    assert diagnostics[key]["latest_first_active_update"] == 1
    assert trace[0][key]["update"] == 1
    assert trace[-1][key]["update"] == 1_000


def test_driver_rejects_any_non_preregistered_training_cap() -> None:
    with pytest.raises(RuntimeError, match="1,000-update/16,000-presentation"):
        runner.run_fixed_training_v9(
            None,
            None,
            None,
            (),
            None,
            (),
            None,
            maximum_updates=999,
        )


def test_source_contains_no_reimplemented_loss_or_backward_loop() -> None:
    source = __import__("pathlib").Path(runner.__file__).read_text()
    assert source.count("_v3.joint_training_update_v3(") == 1
    assert source.count("_v3._v2._run_fixed_training_core_v2(") == 1
    assert ".backward(" not in source
    assert "fine_rgb_branch" not in source
    assert "hierarchical_cnn" not in source
