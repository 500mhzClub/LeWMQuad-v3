from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from scripts import (
    run_go2_rgb_swept_progress_survival_joint_jepa_v6_fine_rgb_bev_fusion
    as runner,
)


def _receipt(update: int) -> dict[str, Any]:
    early = update == 1
    norms = {
        "branch": 1.0,
        "conv1": 0.0 if early else 0.25,
        "conv2": 0.0 if early else 0.5,
        "output": 1.0,
    }
    return {
        "schema": "lewm_v6_fine_rgb_post_backward_gradient_v1",
        "update": update,
        "gradient_l2": norms,
        "active": {name: value > 0.0 for name, value in norms.items()},
        "target_gradient_tensor_count": 0,
    }


def test_v6_wrapper_inherits_exact_v3_loss_schedule_and_cap() -> None:
    for name in (
        "ACTION_ORDER", "MICROBATCH_SIZE", "MICROBATCHES_PER_UPDATE",
        "PRESENTATIONS_PER_UPDATE", "MAXIMUM_UPDATES", "MAXIMUM_PRESENTATIONS",
        "OCCUPIED_CLASS_INDEX", "OCCUPIED_SAFETY_AUX_COEFFICIENT",
        "OCCUPIED_SAFETY_AUX_NORMALIZATION",
    ):
        assert getattr(runner, name) == getattr(runner._v3, name)
    assert runner.FINE_RGB_BRANCH_ADDED_TRAINABLE_PARAMETER_COUNT_V6 == 12_256
    assert len(runner.FINE_RGB_ONLINE_PARAMETER_NAMES_V6) == 6
    assert len(runner.FINE_RGB_TARGET_PARAMETER_NAMES_V6) == 6


def test_activity_summary_records_zero_projection_unlock_without_per_update_gate() -> None:
    receipts = [_receipt(update) for update in range(1, 1_001)]
    receipts[500]["gradient_l2"]["conv1"] = 0.0
    receipts[500]["active"]["conv1"] = False
    observed = runner._summarize_branch_activity_v6(receipts)
    assert observed["update_count"] == 1_000
    assert observed["first_active_update"] == {
        "branch": 1, "conv1": 2, "conv2": 2, "output": 1,
    }
    assert observed["active_update_count"]["conv1"] == 998
    assert observed["target_gradient_tensor_count"] == 0


def test_activity_summary_fails_cleanly_if_an_earlier_convolution_never_unlocks() -> None:
    receipts = [_receipt(update) for update in range(1, 1_001)]
    for receipt in receipts:
        receipt["gradient_l2"]["conv1"] = 0.0
        receipt["active"]["conv1"] = False
    with pytest.raises(RuntimeError, match="conv1 never received"):
        runner._summarize_branch_activity_v6(receipts)


def test_driver_delegates_each_update_to_v3_and_preserves_its_loss_result(
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
            assert result is inherited_result
            rows.append({
                "update": update,
                "presentations": update * 16,
                "losses": dict(result.mean_losses),
                "gradient_l2": dict(result.gradient_l2),
            })
        return runner.JointTrainingAccountingV1(updates=1_000), tuple(rows), {"inherited": True}

    monkeypatch.setattr(runner._v3, "joint_training_update_v3", fake_joint)
    monkeypatch.setattr(runner._v3._v2, "_run_fixed_training_core_v2", fake_core)
    monkeypatch.setattr(
        runner, "fine_branch_gradient_receipt_v6",
        lambda model: _receipt(calls),
    )
    _, trace, diagnostics = runner.run_fixed_training_v6(
        None, None, None, (), None, (), None
    )
    assert calls == 1_000
    assert trace[0]["losses"] == inherited_result.mean_losses
    assert trace[-1]["losses"] == inherited_result.mean_losses
    assert diagnostics["inherited"] is True
    assert diagnostics["fine_rgb_branch"]["first_active_update"]["conv1"] == 2


def test_source_contains_no_reimplemented_loss_or_backward_loop() -> None:
    source = __import__("pathlib").Path(runner.__file__).read_text()
    assert source.count("_v3.joint_training_update_v3(") == 1
    assert source.count("_v3._v2._run_fixed_training_core_v2(") == 1
    assert ".backward(" not in source
    assert "hazard_ranking" not in source
