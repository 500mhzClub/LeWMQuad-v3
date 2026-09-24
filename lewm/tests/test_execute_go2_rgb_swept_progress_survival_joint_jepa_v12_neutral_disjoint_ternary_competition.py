from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from lewm.models.encoders import VisionEncoder
from lewm.models import (
    geometry_anchored_swept_progress_survival_joint_jepa_v10_projective_cell_volume_token_lift
    as v10_model_api,
)
from lewm.models import (
    geometry_anchored_swept_progress_survival_joint_jepa_v11_height_role_factorized_evidence_lift
    as v11_model_api,
)
from lewm.models import (
    geometry_anchored_swept_progress_survival_joint_jepa_v12_neutral_disjoint_ternary_competition
    as model_api,
)
from scripts import (
    execute_go2_rgb_swept_progress_survival_joint_jepa_v12_neutral_disjoint_ternary_competition
    as executor,
)
from scripts import run_go2_rgb_swept_progress_survival_joint_jepa_v1 as training_v1
from scripts import (
    run_go2_rgb_swept_progress_survival_joint_jepa_v3_half_occupied_safety_aux
    as training_v3,
)
from scripts import (
    run_go2_rgb_swept_progress_survival_joint_jepa_v9_content_adaptive_dense_local_token_lift
    as training_v9,
)
from scripts import (
    run_go2_rgb_swept_progress_survival_joint_jepa_v11_height_role_factorized_evidence_lift
    as training_v11,
)


def _sweep_masks() -> torch.Tensor:
    masks = torch.zeros((9, 16, 64, 64), dtype=torch.bool)
    masks[:, :, 31:33, 31:33] = True
    return masks


@pytest.fixture(scope="module")
def n320_encoder_state() -> dict[str, torch.Tensor]:
    caller_rng = torch.random.get_rng_state().clone()
    try:
        torch.random.default_generator.manual_seed(12_730)
        encoder = VisionEncoder(
            image_size=112,
            patch_size=7,
            hidden_dim=192,
            depth=6,
            n_heads=6,
            mlp_ratio=4,
            dropout=0.0,
        )
        return {
            name: value.detach().clone()
            for name, value in encoder.state_dict().items()
        }
    finally:
        torch.random.set_rng_state(caller_rng)


@pytest.fixture(scope="module")
def fresh_v10_v11_v12(
    n320_encoder_state: dict[str, torch.Tensor],
) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    masks = _sweep_masks()
    caller_rng = torch.random.get_rng_state().clone()
    v12 = model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV12(
        n320_encoder_state,
        masks,
    )
    v11 = v11_model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV11(
        n320_encoder_state,
        masks,
    )
    v10 = v10_model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV10(
        n320_encoder_state,
        masks,
    )
    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    return v10, v11, v12


def _activity_diagnostics() -> dict[str, Any]:
    return {
        "height_role_branch_attention": {
            "schema": (
                "lewm_v11_height_role_branch_attention_training_activity_v1"
            ),
            "all_online_parameter_tensors_active_by_update_2": True,
            "target_gradient_tensor_count": 0,
        },
        "factorized_semantic_axes": {
            "schema": "lewm_v11_factorized_semantic_axes_training_activity_v1",
            "all_online_parameter_tensors_active_by_update_2": True,
            "target_gradient_tensor_count": 0,
        },
    }


def test_executor_binds_v12_preregistration_zero_parameter_change_and_cap() -> None:
    assert executor.PREREGISTRATION_COMMIT == (
        "ae1568e8f434d715d379eefc3eaf644369154f76"
    )
    assert executor.OUTPUT_RELATIVE_PATH.endswith(
        "v12_neutral_disjoint_ternary_competition/attempt_v1"
    )
    for schema in (
        executor.CHECKPOINT_SCHEMA,
        executor.TRACE_SCHEMA,
        executor.RESULT_SCHEMA,
        executor.FAILURE_SCHEMA,
    ):
        assert "v12_neutral_disjoint_ternary_competition" in schema
    assert (
        executor.MICROBATCH_SIZE,
        executor.MICROBATCHES_PER_UPDATE,
        executor.PRESENTATIONS_PER_UPDATE,
        executor.MAXIMUM_UPDATES,
        executor.MAXIMUM_PRESENTATIONS,
    ) == (4, 4, 16, 1_000, 16_000)
    assert executor.FLOOR_SUPPORT_INDICES_V11 == (0, 5, 10, 15, 20)
    assert executor.ELEVATED_SUPPORT_INDICES_V11 == tuple(
        value for value in range(25) if value not in (0, 5, 10, 15, 20)
    )

    receipt = executor.neutral_disjoint_ternary_architecture_receipt_v12()
    assert receipt["predecessor"].startswith("fresh_v11_source")
    assert receipt["added_parameter_count"] == 0
    assert receipt["v11_parameter_or_buffer_change"] is False
    assert receipt["supported_cell_logits"] == {
        "unknown": "0",
        "free": "f",
        "occupied": "o",
        "normalization": "log_softmax",
    }
    receipt["added_parameter_count"] = -1
    assert executor.neutral_disjoint_ternary_architecture_receipt_v12()[
        "added_parameter_count"
    ] == 0


def test_executor_binds_exact_v12_model_v11_training_and_v11_evaluation() -> None:
    executor._validate_model_api_v12(model_api)
    executor._validate_training_core_v12(
        training_v1,
        training_v3,
        training_v9,
        training_v11,
    )
    assert model_api.GeometryAnchoredDeformableBevLiftJointJepaV1 is (
        model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV12
    )
    assert executor.evaluate_gate_v12 is executor._v11.evaluate_gate_v11
    assert executor.scientific_metrics_v12 is executor._v11.scientific_metrics_v11
    assert executor.semantic_metrics_v12 is executor._v11.semantic_metrics_v11
    assert executor.paired_control_comparison_v12 is (
        executor._v11.paired_control_comparison_v11
    )


def test_fresh_v11_witness_proves_bit_state_no_new_parameter_and_partition(
    fresh_v10_v11_v12: tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module],
) -> None:
    fresh_v10, fresh_v11, model = fresh_v10_v11_v12
    receipt = executor._state_identity_receipt_v12(
        model,
        fresh_v11,
        fresh_v10,
        torch=torch,
        model_api=model_api,
        v11_model_api=v11_model_api,
        training_v11=training_v11,
    )
    assert receipt["schema"] == (
        "lewm_v12_fresh_v11_zero_parameter_state_identity_v1"
    )
    assert receipt["predecessor_experiment_checkpoint_read"] is False
    assert receipt["added_parameter_tensor_count"] == 0
    assert receipt["added_parameter_count"] == 0
    assert receipt["v12_parameter_tensor_count"] == receipt[
        "v11_parameter_tensor_count"
    ]
    assert receipt["v12_parameter_count"] == receipt["v11_parameter_count"]
    assert receipt["all_parameter_values_bit_exact"] is True
    assert receipt["all_buffer_values_bit_exact"] is True
    assert receipt["neutral_algebra_exact"] is True
    assert receipt["branch_invalid_evidence_fixed_to_minus_20"] is True
    assert receipt["all_invalid_logits_exact"] is True

    partition = training_v1.partition_parameters_v1(model)
    initial = executor._initial_model_receipt_v12(
        model,
        partition,
        receipt,
        training_v11=training_v11,
    )
    assert initial["online_branch_attention_parameter_tensor_count"] == 14
    assert initial["target_branch_attention_parameter_tensor_count"] == 14
    assert initial["factorized_semantic_parameter_tensor_count"] == 12
    assert initial["online_branch_attention_parameter_count"] == 14_528
    assert initial["target_branch_attention_parameter_count"] == 14_528
    assert initial["factorized_semantic_parameter_count"] == 18_628
    assert initial["optimizer_parameter_membership_changed_from_v11"] is False


def test_training_delegates_once_and_physical_stage_remains_separate() -> None:
    calls = 0
    diagnostics = _activity_diagnostics()

    def fake_training(*args: Any) -> tuple[Any, tuple[dict[str, Any], ...], Any]:
        nonlocal calls
        del args
        calls += 1
        return object(), ({"update": 1},), diagnostics

    accounting, trace, observed = executor._run_fixed_training_v12(
        SimpleNamespace(run_fixed_training_v11=fake_training),
        object(),
    )
    assert calls == 1
    assert accounting is not None
    assert trace == ({"update": 1},)
    assert observed["v12_contract"]["objective"] == "S+P+U+R+O"
    assert observed["v12_contract"]["occupied_auxiliary_coefficient"] == 0.5
    assert observed["v12_contract"]["new_loss_or_weight"] is False

    staged = executor._physical_calibration_stage_v12(True)
    closed = executor._physical_calibration_stage_v12(False)
    assert staged["status"] == "STAGED_FOR_SEPARATELY_FROZEN_ONE_SHOT"
    assert staged["physical_calibration_authorized_in_this_attempt"] is False
    assert closed["status"] == "CLOSED_FULL_ARM_GATE_FAILED"


def test_write_once_source_order_authority_and_single_science_call(
    tmp_path: Path,
) -> None:
    output = executor._fresh_output_root_v12(tmp_path)
    assert output == tmp_path / executor.OUTPUT_RELATIVE_PATH
    with pytest.raises(FileExistsError, match="attempt_v1 already exists"):
        executor._fresh_output_root_v12(tmp_path)

    source = Path(executor.__file__).read_text()
    assert source.count("training_v11.run_fixed_training_v11(*args)") == 1
    assert source.count("_v1.score_role_v1(") == 1
    checkpoint_write = source.index('output / "checkpoint_update_1000.pt"')
    trace_write = source.index('output / "training_trace.json"')
    evaluation = source.index("_v1.score_role_v1(")
    result_write = source.index('output / "result.json"')
    assert checkpoint_write < trace_write < evaluation < result_write
    assert "torch.load(" not in source
    assert '"predecessor_experiment_checkpoint_read": False' in source
    assert '"only_change": "neutral_disjoint_ternary_semantic_algebra"' in source
    assert '"added_parameter_count": 0' in source
    assert '"optimizer_parameter_tensor_membership_changed": False' in source
    assert '"loss_source_or_coefficient_changed": False' in source
    assert (
        '"loss_gradient_surface_changed_by_registered_semantic_algebra": True'
        in source
    )
    assert '"schedule_changed": False' in source
    assert '"evaluation_changed": False' in source
    assert 'len(gate.get("checks", {})) != 24' in source
    assert "checkpoint_access_authorized_for_physical_calibration\": False" in source


@pytest.mark.parametrize(("passed", "expected"), ((True, 0), (False, 2)))
def test_main_returns_frozen_gate_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    passed: bool,
    expected: int,
) -> None:
    monkeypatch.setattr(
        executor,
        "execute_v12",
        lambda **kwargs: {
            "status": "PASS" if passed else "FAIL",
            "full_arm_gate": {"passed": passed},
        },
    )
    assert executor.main([]) == expected
