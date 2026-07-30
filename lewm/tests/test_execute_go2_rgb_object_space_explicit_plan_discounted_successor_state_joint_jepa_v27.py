from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import (
    execute_go2_rgb_object_space_explicit_plan_discounted_successor_state_joint_jepa_v27
    as executor,
)


ROOT = Path(__file__).resolve().parents[2]


def _authority() -> dict:
    runtime_inputs = {
        name: {} for name in executor.RUNTIME_INPUT_BINDING_NAMES
    }
    binding = {"path": "index.jsonl", "file_sha256": "a" * 64, "byte_count": 1}
    runtime_inputs.update(
        {
            "h6_train_index": dict(binding),
            "h6_validation_index": dict(binding),
        }
    )
    return executor._content_bound(
        {
            "schema": f"{executor.SCHEMA_PREFIX}_future_execution_authority_v1",
            "status": "AUTHORIZED_CERTIFIED_NARROW_EXPORT_ONE_SHOT",
            "scientific_payload_authorized": True,
            "one_shot": True,
            "maximum_updates": 400,
            "maximum_presentations": 12_800,
            "retry_authorized": False,
            "resume_authorized": False,
            "certified_source_root": executor.CERTIFIED_SOURCE_ROOT,
            "output_root": executor.OUTPUT_ROOT_RELATIVE_PATH,
            "preregistration_commit": executor.PREREGISTRATION_COMMIT,
            "runtime_data_root": str(ROOT),
            "selectors": {
                "executor_module": executor.__name__,
                "model_module": executor.MODEL_MODULE_NAME,
                "model_class": executor.MODEL_CLASS_NAME,
                "training_module": executor.TRAINING_MODULE_NAME,
                "evaluation_module": executor.EVALUATION_MODULE_NAME,
            },
            "clean_export_certification": {
                "path": executor.CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH,
                "file_sha256": "b" * 64,
                "byte_count": 1,
                "content_sha256": "c" * 64,
            },
            "runtime_inputs": runtime_inputs,
            "rgb_root_relative_path": ".generated/datagen_full/render_textured_v03",
        }
    )


def test_authority_and_one_shot_reservation_are_fail_closed(tmp_path: Path) -> None:
    authority = _authority()
    assert executor.validate_future_execution_prerequisites_v27(authority) == authority
    reservation = executor.reserve_attempt_v27(
        tmp_path, authority, created_utc="2026-07-30T00:00:00Z"
    )
    assert executor.validate_attempt_reservation_v27(reservation) == reservation
    assert reservation["maximum_updates"] == 400
    assert reservation["maximum_presentations"] == 12_800
    with pytest.raises(FileExistsError):
        executor.reserve_attempt_v27(
            tmp_path, authority, created_utc="2026-07-30T00:00:01Z"
        )

    changed = dict(authority)
    changed["maximum_updates"] = 401
    changed = executor._content_bound(
        {name: value for name, value in changed.items() if name != "content_sha256"}
    )
    with pytest.raises(PermissionError):
        executor.validate_future_execution_prerequisites_v27(changed)


def _plan_metrics() -> dict:
    return {
        "correct_ratio": 0.89,
        "all_registered_values_finite": True,
        "advantages": {
            name: {
                "equal_family_mean": 0.05 if name == "tail_advantage" else 0.01,
                "bootstrap_lower_95": 0.001,
                "positive_family_count": 6,
            }
            for name in executor.PLAN_METRIC_NAMES
        },
    }


@pytest.mark.parametrize(
    ("mutation", "passed"),
    (
        (lambda value: None, True),
        (lambda value: value.update(correct_ratio=0.90), False),
        (
            lambda value: value["advantages"]["tail_advantage"].update(
                equal_family_mean=0.049999
            ),
            False,
        ),
        (
            lambda value: value["advantages"]["wrong_plan_advantage"].update(
                equal_family_mean=0.0
            ),
            False,
        ),
        (
            lambda value: value["advantages"]["persistence_advantage"].update(
                bootstrap_lower_95=0.0
            ),
            False,
        ),
        (
            lambda value: value["advantages"]["mean_prior_advantage"].update(
                positive_family_count=5
            ),
            False,
        ),
    ),
)
def test_hard_gate_boundaries_are_exact(
    monkeypatch: pytest.MonkeyPatch, mutation, passed: bool
) -> None:
    monkeypatch.setattr(
        executor.v26,
        "evaluate_update400_gate_v26",
        lambda *_args, **_kwargs: {"passed": True},
    )
    plan = _plan_metrics()
    mutation(plan)
    result = executor.evaluate_gate_v27(
        update100_physical={},
        update400_physical={},
        update400_controls={},
        plan_metrics=plan,
        integrity_pass=True,
    )
    assert result["passed"] is passed
    assert (result["action"].startswith("PASS_")) is passed


def test_update_integrity_accepts_only_exact_mixed_accounting() -> None:
    update = 3
    route_names = (
        "camera_shared",
        "joint_shared",
        "representation",
        "predictor",
        "predictor_core_protected_survival_output",
        "explicit_plan_discounted_successor_state",
    )
    result = SimpleNamespace(
        accounting={
            "updates": update,
            "presentations": 32 * update,
            "physical_presentations": 16 * update,
            "plan_presentations": 16 * update,
            "physical_microbatch_graphs": 4 * update,
            "plan_microbatch_graphs": 4 * update,
            "autograd_grad_calls": 16 * update,
            "optimizer_steps": update,
            "ema_steps": update,
        },
        gradient_routes={
            name: {
                "preclip_l2": 1.0,
                "applied_scale": 1.0,
                "parameter_tensor_count": 1,
                "absent_tensor_gradient_count": 0,
            }
            for name in route_names
        },
        mean_losses={"C": 1.0, "N27": 1.0, "J24": 1.0, "P27": 1.0, "L27": 4.0},
        plan_diagnostics={
            "mechanism": "explicit_plan_discounted_successor_state",
            "gamma": 0.9,
            "p25_evaluation_count": 0,
            "energy_per_row": (0.1,) * 16,
        },
        target_gradient_tensor_count=0,
        optimizer_steps_this_update=1,
        ema_steps_this_update=1,
    )
    model = SimpleNamespace(
        ema_update_count=torch.tensor(update),
        target_modules=lambda: (),
        state_dict=lambda: {"online": torch.tensor((1.0,))},
    )
    runtime = SimpleNamespace(torch=torch)
    receipt = executor.validate_update_integrity_v27(
        runtime, model, result, update=update
    )
    assert receipt["passed"] is True
    assert receipt["p25_evaluation_count"] == 0

    result.accounting = {**result.accounting, "presentations": 95}
    with pytest.raises(RuntimeError, match="accounting"):
        executor.validate_update_integrity_v27(runtime, model, result, update=update)
