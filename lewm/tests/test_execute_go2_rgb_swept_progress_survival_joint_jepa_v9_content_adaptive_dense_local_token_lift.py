from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lewm.models.encoders import VisionEncoder
from lewm.models import (
    geometry_anchored_deformable_bev_lift_joint_jepa_v1 as parent_model_api,
)
from lewm.models import (
    geometry_anchored_swept_progress_survival_joint_jepa_v4_residual_local_semantic_decoder
    as v4_model_api,
)
from lewm.models import (
    geometry_anchored_swept_progress_survival_joint_jepa_v9_content_adaptive_dense_local_token_lift
    as model_api,
)
from scripts import (
    execute_go2_rgb_swept_progress_survival_joint_jepa_v9_content_adaptive_dense_local_token_lift
    as executor,
)
from scripts import (
    run_go2_rgb_swept_progress_survival_joint_jepa_v1 as training_v1,
)
from scripts import (
    run_go2_rgb_swept_progress_survival_joint_jepa_v3_half_occupied_safety_aux
    as training_v3,
)
from scripts import (
    run_go2_rgb_swept_progress_survival_joint_jepa_v9_content_adaptive_dense_local_token_lift
    as training_v9,
)


def _sweep_masks() -> torch.Tensor:
    masks = torch.zeros((9, 16, 64, 64), dtype=torch.bool)
    masks[:, :, 31:33, 31:33] = True
    return masks


@pytest.fixture(scope="module")
def n320_encoder_state() -> dict[str, torch.Tensor]:
    caller_rng = torch.random.get_rng_state().clone()
    try:
        torch.random.default_generator.manual_seed(17029)
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
def clean_v4_and_v9(
    n320_encoder_state: dict[str, torch.Tensor],
) -> tuple[torch.nn.Module, torch.nn.Module]:
    masks = _sweep_masks()
    return (
        v4_model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV4(
            n320_encoder_state, masks
        ),
        model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV9(
            n320_encoder_state, masks
        ),
    )


def test_executor_binds_preregistration_amendment_and_architecture() -> None:
    assert executor.PREREGISTRATION_COMMIT == (
        "47043472466e7a258ad0f0be854c05393e233db8"
    )
    assert executor.PREIMPLEMENTATION_AMENDMENT_COMMIT == (
        "04db6b26d46875297e3aa515fdf1d688bee2b755"
    )
    assert executor.OUTPUT_RELATIVE_PATH.endswith(
        "v9_content_adaptive_dense_local_token_lift/attempt_v1"
    )
    assert (
        executor.DENSE_LOCAL_SUPPORT_SIDE_V9,
        executor.DENSE_LOCAL_SUPPORT_COUNT_V9,
        executor.DENSE_LOCAL_SUPPORT_CENTER_INDEX_V9,
        executor.DENSE_LOCAL_ATTENTION_HEADS_V9,
        executor.DENSE_LOCAL_ATTENTION_HEAD_WIDTH_V9,
        executor.DENSE_LOCAL_ATTENTION_PARAMETER_TENSOR_COUNT_V9,
        executor.DENSE_LOCAL_ATTENTION_ADDED_PARAMETER_COUNT_V9,
    ) == (5, 25, 12, 4, 16, 7, 16_576)
    receipt = executor.dense_local_lift_architecture_receipt_v9()
    assert receipt["attention"]["key"]["bias"] is False
    assert receipt["attention"]["parameter_tensor_count"] == 7
    assert receipt["attention"]["added_parameter_count_per_lift"] == 16_576
    assert receipt["all_invalid_cells"] == (
        "excluded_from_attention_softmax_with_exact_zero_reported_weights_then_"
        "exact_inherited_null_evidence_before_consumers"
    )
    assert receipt["sampling"] == {
        "operator": "torch.nn.functional.grid_sample",
        "mode": "bilinear",
        "padding_mode": "zeros",
        "align_corners": False,
        "invalid_coordinate_xy": [2.0, 2.0],
        "reported_support_valid_mask_shape": [64, 64, 25],
        "reported_support_grid_xy_shape": [64, 64, 25, 2],
    }
    receipt["attention"]["head_count"] = -1
    assert executor.dense_local_lift_architecture_receipt_v9()["attention"][
        "head_count"
    ] == 4


def test_fresh_output_is_exactly_write_once_and_calibration_is_conditional(
    tmp_path: Path,
) -> None:
    output = executor._fresh_output_root_v9(tmp_path)
    assert output == tmp_path / executor.OUTPUT_RELATIVE_PATH
    assert output.is_dir()
    with pytest.raises(FileExistsError, match="attempt_v1 already exists"):
        executor._fresh_output_root_v9(tmp_path)
    assert executor._physical_calibration_stage_v9(True)["status"] == (
        "STAGED_FOR_SEPARATELY_FROZEN_ONE_SHOT"
    )
    assert executor._physical_calibration_stage_v9(False)["status"] == (
        "CLOSED_FULL_ARM_GATE_FAILED"
    )
    assert not executor._physical_calibration_stage_v9(True)[
        "physical_calibration_run_in_this_attempt"
    ]


def test_executor_validates_final_model_and_runner_contracts() -> None:
    executor._validate_model_api_v9(model_api)
    executor._validate_training_core_v9(training_v1, training_v3, training_v9)
    assert executor.evaluate_gate_v9 is executor._v4.evaluate_gate_v4
    assert executor.scientific_metrics_v9 is executor._v4.scientific_metrics_v4
    assert executor.semantic_metrics_v9 is executor._v4.semantic_metrics_v4
    assert executor.paired_control_comparison_v9 is (
        executor._v4.paired_control_comparison_v4
    )


def test_clean_v4_migration_and_optimizer_partition_are_receipted(
    clean_v4_and_v9: tuple[torch.nn.Module, torch.nn.Module],
) -> None:
    clean_v4, model = clean_v4_and_v9
    migration = executor._migration_receipt_v9(
        model, clean_v4, torch=torch, model_api=model_api
    )
    assert migration["all_inherited_state_tensors_bit_exact"] is True
    assert migration["removed_parameter_count_per_online_or_target_lift"] == 49_152
    assert migration["added_attention_parameter_count_per_online_or_target_lift"] == 16_576
    assert migration["added_attention_parameter_tensor_count_per_online_or_target_lift"] == 7
    assert migration["attention_initialization_bit_exact"] is True
    assert migration["key_projection_bias"] is False
    assert migration["sampling_receipt"]["attention_weights_shape"] == [
        1,
        64,
        64,
        4,
        25,
    ]
    partition = training_v1.partition_parameters_v1(model)
    initial = executor._initial_model_receipt_v9(
        model,
        partition,
        migration,
        torch=torch,
        model_api=model_api,
        inherited_semantic_method=(
            parent_model_api.GeometryAnchoredDeformableBevLiftJointJepaV1.
            semantic_logits_from_latent
        ),
    )
    assert initial["online_attention_parameter_count"] == 16_576
    assert initial["online_attention_parameter_tensor_count"] == 7
    assert initial[
        "all_online_attention_parameters_in_lift_semantic_exactly_once"
    ] is True
    assert initial[
        "all_target_attention_parameters_frozen_in_target_exactly_once"
    ] is True
    assert initial["initial_hard_sync_count"] == 1
    assert initial["initial_ema_update_count"] == 0


def test_migration_audit_fails_closed_on_inherited_tensor_mutation(
    clean_v4_and_v9: tuple[torch.nn.Module, torch.nn.Module],
) -> None:
    clean_v4, model = clean_v4_and_v9
    weight = clean_v4.semantic_head.base.weight
    original = weight.detach().clone()
    try:
        with torch.no_grad():
            weight.add_(1.0)
        with pytest.raises(RuntimeError, match="changed inherited V4 tensor"):
            executor._migration_receipt_v9(
                model, clean_v4, torch=torch, model_api=model_api
            )
    finally:
        with torch.no_grad():
            weight.copy_(original)


def test_executor_source_orders_artifacts_before_evaluation_and_has_no_legacy_stub() -> None:
    source = Path(executor.__file__).read_text()
    assert source.count("training_v9.run_fixed_training_v9(") == 1
    assert source.count("_v1.score_role_v1(") == 1
    checkpoint_write = source.index('output / "checkpoint_update_1000.pt"')
    trace_write = source.index('output / "training_trace.json"')
    evaluation = source.index("_v1.score_role_v1(")
    result_write = source.index('output / "result.json"')
    assert checkpoint_write < trace_write < evaluation < result_write
    assert "torch.load(" not in source
    assert "fine_rgb" not in source
    assert "hierarchical_cnn" not in source
    assert '"only_change": "content_adaptive_dense_local_token_lift"' in source
    assert '"input_tensorization_changed": False' in source
    assert '"losses_changed": False' in source
    assert '"schedule_changed": False' in source
    assert '"evaluation_changed": False' in source
    assert 'len(gate.get("checks", {})) != 24' in source
    assert 'training_diagnostics["dense_local_attention"]' in source
    assert '"checkpoint_access_authorized_for_physical_calibration": full_arm_passed' in source
