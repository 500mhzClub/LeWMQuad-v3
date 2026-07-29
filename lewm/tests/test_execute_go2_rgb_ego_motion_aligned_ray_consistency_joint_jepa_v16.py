from __future__ import annotations

from types import SimpleNamespace

import pytest

from lewm.models import (
    geometry_anchored_swept_progress_survival_joint_jepa_v14_unified_ray_survival
    as model_module,
)
from scripts import (
    execute_go2_rgb_ego_motion_aligned_ray_consistency_joint_jepa_v16 as executor,
)
from scripts import (
    run_go2_rgb_ego_motion_aligned_ray_consistency_joint_jepa_v16
    as training_module,
)


def _result(**changes: object) -> SimpleNamespace:
    losses = {
        "S": 1.0,
        "P": 2.0,
        "U": 3.0,
        "R": 4.0,
        "O": 5.0,
        "N": 15.0,
        "C_base": 2.0,
        "M": 0.5,
        "C": 2.05,
        "L": 17.05,
    }
    values = {
        "mean_losses": losses,
        "ray_consistency_shared_valid_cell_count": 100,
        "ray_consistency_positive_weight_cell_count": 80,
        "ray_consistency_weight_sum": 31.5,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_v16_adapts_only_the_registered_model_and_training_contract() -> None:
    assert executor.MAXIMUM_UPDATES == 1_000
    assert executor.MAXIMUM_PRESENTATIONS == 16_000
    assert executor.TRAINING_REQUIRED_BATCH_KEYS[-1] == (
        executor.REALIZED_RELATIVE_SE2_KEY
    )
    training = executor.validate_training_api_v16(training_module)
    model = executor.validate_model_api_v16(model_module)
    assert training["required_batch_key_count"] == 22
    assert model["online_trainable_parameter_count"] == 3_383_917
    denial = executor.execution_denial_receipt_v16()
    assert denial["status"] == "DENIED_SOURCE_ONLY"
    assert denial["scientific_payload_opened"] is False
    assert denial["reservation_created"] is False


def test_update_integrity_retains_base_checks_and_adds_exact_v16_equation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def base_validator(*_args: object, **kwargs: object) -> dict[str, object]:
        proxy = _args[2]
        observed["losses"] = dict(proxy.mean_losses)
        observed["update"] = kwargs["update"]
        return {"passed": True, "mean_losses": dict(proxy.mean_losses)}

    monkeypatch.setattr(
        executor,
        "_original_validate_update_integrity",
        base_validator,
    )
    receipt = executor.validate_update_integrity_v16(
        object(),
        object(),
        _result(),
        update=7,
        access_receipt={},
    )
    assert set(observed["losses"]) == {"S", "P", "U", "R", "O", "N", "C", "L"}
    assert observed["update"] == 7
    assert receipt["mean_losses"]["C"] == pytest.approx(2.05)
    assert receipt["ray_consistency"] == {
        "shared_valid_cell_count": 100,
        "positive_weight_cell_count": 80,
        "weight_sum": 31.5,
        "loss_weight": 0.1,
    }

    bad = _result()
    bad.mean_losses["C"] = 2.2
    with pytest.raises(RuntimeError, match="Camera loss equation"):
        executor.validate_update_integrity_v16(
            object(), object(), bad, update=7, access_receipt={}
        )


def test_update_integrity_rejects_inert_consistency_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        executor,
        "_original_validate_update_integrity",
        lambda *_args, **_kwargs: {"passed": True},
    )
    with pytest.raises(RuntimeError, match="support receipt"):
        executor.validate_update_integrity_v16(
            object(),
            object(),
            _result(
                ray_consistency_shared_valid_cell_count=0,
                ray_consistency_positive_weight_cell_count=0,
                ray_consistency_weight_sum=0.0,
            ),
            update=1,
            access_receipt={},
        )
