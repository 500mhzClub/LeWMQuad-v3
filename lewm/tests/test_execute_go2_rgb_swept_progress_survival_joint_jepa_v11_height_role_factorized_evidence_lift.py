from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lewm.models.encoders import VisionEncoder
from lewm.models import (
    geometry_anchored_swept_progress_survival_joint_jepa_v10_projective_cell_volume_token_lift
    as v10_model_api,
)
from lewm.models import (
    geometry_anchored_swept_progress_survival_joint_jepa_v11_height_role_factorized_evidence_lift
    as model_api,
)
from scripts import (
    execute_go2_rgb_swept_progress_survival_joint_jepa_v11_height_role_factorized_evidence_lift
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
        torch.random.default_generator.manual_seed(11_730)
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
def fresh_v10_v11(
    n320_encoder_state: dict[str, torch.Tensor],
) -> tuple[torch.nn.Module, torch.nn.Module]:
    masks = _sweep_masks()
    return (
        v10_model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV10(
            n320_encoder_state, masks
        ),
        model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV11(
            n320_encoder_state, masks
        ),
    )


def test_executor_binds_preregistration_roles_and_one_shot_cap() -> None:
    assert executor.PREREGISTRATION_COMMIT == (
        "b8ca8bd267e233a11f29da82842dcf5429743c18"
    )
    assert executor.OUTPUT_RELATIVE_PATH.endswith(
        "v11_height_role_factorized_evidence_lift/attempt_v1"
    )
    assert "v11_height_role_factorized_evidence_lift" in executor.CHECKPOINT_SCHEMA
    assert "v11_height_role_factorized_evidence_lift" in executor.TRACE_SCHEMA
    assert "v11_height_role_factorized_evidence_lift" in executor.RESULT_SCHEMA
    assert "v11_height_role_factorized_evidence_lift" in executor.FAILURE_SCHEMA
    assert executor.FLOOR_SUPPORT_INDICES_V11 == (0, 5, 10, 15, 20)
    assert executor.ELEVATED_SUPPORT_INDICES_V11 == tuple(
        value for value in range(25) if value not in (0, 5, 10, 15, 20)
    )
    assert (
        executor.HEIGHT_ROLE_ATTENTION_PARAMETER_COUNT_V11,
        executor.HEIGHT_ROLE_ATTENTION_PARAMETER_TENSOR_COUNT_V11,
        executor.HEIGHT_ROLE_SEMANTIC_PARAMETER_COUNT_V11,
        executor.HEIGHT_ROLE_SEMANTIC_PARAMETER_TENSOR_COUNT_V11,
    ) == (14_528, 14, 18_628, 12)
    assert (
        executor.MICROBATCH_SIZE,
        executor.MICROBATCHES_PER_UPDATE,
        executor.PRESENTATIONS_PER_UPDATE,
        executor.MAXIMUM_UPDATES,
        executor.MAXIMUM_PRESENTATIONS,
    ) == (4, 4, 16, 1_000, 16_000)

    receipt = executor.height_role_factorized_architecture_receipt_v11()
    assert receipt["geometry"]["changed_from_v10"] is False
    assert receipt["roles"]["floor_free"]["valid_cell_count"] == 2_024
    assert receipt["roles"]["elevated_occupied"]["valid_cell_count"] == 2_062
    assert receipt["roles"]["elevated_only_cell_count"] == 38
    assert receipt["roles"]["disjoint_and_exhaustive"] is True
    assert receipt["predictor_consumes_shared_role_ordered_64_channel_state"]
    receipt["roles"]["floor_free"]["valid_cell_count"] = -1
    assert executor.height_role_factorized_architecture_receipt_v11()["roles"][
        "floor_free"
    ]["valid_cell_count"] == 2_024


def test_executor_binds_exact_model_training_and_v10_evaluation_apis() -> None:
    executor._validate_model_api_v11(model_api)
    executor._validate_training_core_v11(
        training_v1, training_v3, training_v9, training_v11
    )
    assert executor.evaluate_gate_v11 is executor._v10.evaluate_gate_v10
    assert executor.scientific_metrics_v11 is executor._v10.scientific_metrics_v10
    assert executor.semantic_metrics_v11 is executor._v10.semantic_metrics_v10
    assert executor.paired_control_comparison_v11 is (
        executor._v10.paired_control_comparison_v10
    )
    assert tuple(model_api.HEIGHT_ROLE_ATTENTION_PARAMETER_SUFFIXES_V11) == tuple(
        training_v11.BRANCH_ATTENTION_PARAMETER_SUFFIXES_V11
    )
    assert tuple(model_api.HEIGHT_ROLE_SEMANTIC_PARAMETER_SUFFIXES_V11) == tuple(
        training_v11.SEMANTIC_AXIS_PARAMETER_SUFFIXES_V11
    )


def test_v10_migration_role_masks_and_parameter_partition_are_receipted(
    fresh_v10_v11: tuple[torch.nn.Module, torch.nn.Module],
) -> None:
    fresh_v10, model = fresh_v10_v11
    migration = executor._migration_receipt_v11(
        model,
        fresh_v10,
        torch=torch,
        model_api=model_api,
        training_v11=training_v11,
    )
    assert migration["schema"].startswith("lewm_v11_")
    assert migration["predecessor_experiment_checkpoint_read"] is False
    assert migration["all_common_v10_parameter_values_bit_exact"] is True
    assert migration["all_common_v10_buffer_values_bit_exact"] is True
    assert migration["online_branch_attention_parameter_count"] == 14_528
    assert migration["target_branch_attention_parameter_count"] == 14_528
    assert migration["factorized_semantic_parameter_count"] == 18_628
    sampling = migration["sampling_receipt"]
    assert sampling["floor_support_indices"] == [0, 5, 10, 15, 20]
    assert sampling["floor_valid_cell_count"] == 2_024
    assert sampling["floor_valid_mask_row_major_uint8_sha256"] == (
        "8b6b4202d04cf08de9813a4fc12deff9ea35de8d8c7adc8eb40a117593694bbc"
    )
    assert sampling["elevated_valid_cell_count"] == 2_062
    assert sampling["elevated_valid_mask_row_major_uint8_sha256"] == (
        "4ebbafb6d4dd5fb13b96df978abfa7b81bc2f879b2ba6dec2fcda38dec54e60b"
    )
    assert sampling["role_valid_overlap_cell_count"] == 2_024
    assert sampling["elevated_only_cell_count"] == 38
    assert sampling["near_field_floor_valid_cell_count"] == 184
    assert sampling["near_field_elevated_valid_cell_count"] == 222
    assert sampling["invalid_and_cross_role_attention_exact_zero"] is True
    assert sampling["finite_normalized_three_class_log_probabilities"] is True
    assert sampling["all_invalid_semantic_logits_exact_unknown"] is True

    partition = training_v1.partition_parameters_v1(model)
    initial = executor._initial_model_receipt_v11(
        model, partition, migration, torch=torch
    )
    assert initial["schema"].startswith("lewm_v11_")
    assert initial["online_branch_attention_parameter_tensor_count"] == 14
    assert initial["target_branch_attention_parameter_tensor_count"] == 14
    assert initial["factorized_semantic_parameter_tensor_count"] == 12
    assert initial[
        "all_online_replacement_parameters_in_lift_semantic_exactly_once"
    ]
    assert initial["all_target_branch_parameters_frozen_in_target_exactly_once"]
    assert initial["predictor_consumes_shared_role_ordered_64_channel_state"]


def test_training_activity_and_physical_stage_fail_closed() -> None:
    diagnostics = {
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
    branch, semantic = executor._validate_training_activity_v11(diagnostics)
    assert branch is diagnostics["height_role_branch_attention"]
    assert semantic is diagnostics["factorized_semantic_axes"]
    with pytest.raises(RuntimeError, match="not fully active"):
        executor._validate_training_activity_v11(
            {
                **diagnostics,
                "factorized_semantic_axes": {
                    **diagnostics["factorized_semantic_axes"],
                    "all_online_parameter_tensors_active_by_update_2": False,
                },
            }
        )
    assert executor._physical_calibration_stage_v11(True)["status"] == (
        "STAGED_FOR_SEPARATELY_FROZEN_ONE_SHOT"
    )
    assert executor._physical_calibration_stage_v11(False)["status"] == (
        "CLOSED_FULL_ARM_GATE_FAILED"
    )


def test_write_once_source_order_and_authority_are_explicit(tmp_path: Path) -> None:
    output = executor._fresh_output_root_v11(tmp_path)
    assert output == tmp_path / executor.OUTPUT_RELATIVE_PATH
    with pytest.raises(FileExistsError, match="attempt_v1 already exists"):
        executor._fresh_output_root_v11(tmp_path)

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
    assert '"objective": "S+P+U+R+O"' in source
    assert '"losses_changed": False' in source
    assert '"schedule_changed": False' in source
    assert '"evaluation_changed": False' in source
    assert 'len(gate.get("checks", {})) != 24' in source
    assert "GeometryAnchoredSweptProgressSurvivalJointJepaV11(" in source
