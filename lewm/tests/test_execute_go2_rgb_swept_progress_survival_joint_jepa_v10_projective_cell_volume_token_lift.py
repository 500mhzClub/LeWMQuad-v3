from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lewm.models.encoders import VisionEncoder
from lewm.models import (
    geometry_anchored_swept_progress_survival_joint_jepa_v9_content_adaptive_dense_local_token_lift
    as v9_model_api,
)
from lewm.models import (
    geometry_anchored_swept_progress_survival_joint_jepa_v10_projective_cell_volume_token_lift
    as model_api,
)
from scripts import (
    execute_go2_rgb_swept_progress_survival_joint_jepa_v10_projective_cell_volume_token_lift
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


def _sweep_masks() -> torch.Tensor:
    masks = torch.zeros((9, 16, 64, 64), dtype=torch.bool)
    masks[:, :, 31:33, 31:33] = True
    return masks


@pytest.fixture(scope="module")
def n320_encoder_state() -> dict[str, torch.Tensor]:
    caller_rng = torch.random.get_rng_state().clone()
    try:
        torch.random.default_generator.manual_seed(10_729)
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
def fresh_v9_v10(
    n320_encoder_state: dict[str, torch.Tensor],
) -> tuple[torch.nn.Module, torch.nn.Module]:
    masks = _sweep_masks()
    return (
        v9_model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV9(
            n320_encoder_state, masks
        ),
        model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV10(
            n320_encoder_state, masks
        ),
    )


def test_executor_binds_preregistration_and_frozen_v10_architecture() -> None:
    assert executor.PREREGISTRATION_COMMIT == (
        "b9eaae6560c42e588c86fb8bf949cc95bd9e29e9"
    )
    assert executor.OUTPUT_RELATIVE_PATH.endswith(
        "v10_projective_cell_volume_token_lift/attempt_v1"
    )
    assert "v10_projective_cell_volume_token_lift" in executor.CHECKPOINT_SCHEMA
    assert "v10_projective_cell_volume_token_lift" in executor.TRACE_SCHEMA
    assert "v10_projective_cell_volume_token_lift" in executor.RESULT_SCHEMA
    assert "v10_projective_cell_volume_token_lift" in executor.FAILURE_SCHEMA
    assert (
        executor.CELL_VOLUME_HORIZONTAL_SUPPORT_COUNT_V10,
        executor.CELL_VOLUME_HEIGHT_COUNT_V10,
        executor.CELL_VOLUME_SUPPORT_COUNT_V10,
        executor.CELL_VOLUME_ATTENTION_HEADS_V10,
        executor.CELL_VOLUME_ATTENTION_HEAD_WIDTH_V10,
        executor.CELL_VOLUME_ATTENTION_PARAMETER_TENSOR_COUNT_V10,
        executor.CELL_VOLUME_ATTENTION_ADDED_PARAMETER_COUNT_V10,
    ) == (5, 5, 25, 4, 16, 7, 16_576)
    receipt = executor.cell_volume_lift_architecture_receipt_v10()
    assert receipt["only_change_from_v9"] == (
        "registered_3d_support_geometry_and_masked_mean_base"
    )
    assert receipt["geometry"]["cell_valid_count"] == 2_062
    assert receipt["geometry"]["near_field_lte_2m_valid_cell_count"] == 222
    assert receipt["aggregation"]["base"] == (
        "arithmetic_mean_of_valid_samples_with_invalid_exact_zero"
    )
    assert receipt["new_loss_or_head"] is False
    receipt["geometry"]["cell_valid_count"] = -1
    assert executor.cell_volume_lift_architecture_receipt_v10()["geometry"][
        "cell_valid_count"
    ] == 2_062


def test_executor_reuses_exact_v9_v4_training_and_evaluation_contracts() -> None:
    executor._validate_model_api_v10(model_api)
    executor._validate_training_core_v10(training_v1, training_v3, training_v9)
    assert executor.evaluate_gate_v10 is executor._v9.evaluate_gate_v9
    assert executor.scientific_metrics_v10 is executor._v9.scientific_metrics_v9
    assert executor.semantic_metrics_v10 is executor._v9.semantic_metrics_v9
    assert executor.paired_control_comparison_v10 is (
        executor._v9.paired_control_comparison_v9
    )
    assert executor.AUXILIARY_OBJECTIVE == executor._v9.AUXILIARY_OBJECTIVE
    assert (
        executor.MICROBATCH_SIZE,
        executor.MICROBATCHES_PER_UPDATE,
        executor.PRESENTATIONS_PER_UPDATE,
        executor.MAXIMUM_UPDATES,
        executor.MAXIMUM_PRESENTATIONS,
    ) == (4, 4, 16, 1_000, 16_000)


def test_geometry_migration_and_initial_receipts_are_v10_and_partitioned(
    fresh_v9_v10: tuple[torch.nn.Module, torch.nn.Module],
) -> None:
    fresh_v9, model = fresh_v9_v10
    migration = executor._migration_receipt_v10(
        model, fresh_v9, torch=torch, model_api=model_api
    )
    assert migration["schema"].startswith("lewm_v10_")
    assert migration["all_v9_parameter_names_and_values_bit_exact"] is True
    assert migration["all_common_buffers_bit_exact"] is True
    sampling = migration["sampling_receipt"]
    assert sampling["schema"].startswith("lewm_v10_")
    assert sampling["cell_valid_count"] == 2_062
    assert sampling["cell_valid_mask_row_major_uint8_sha256"] == (
        "4ebbafb6d4dd5fb13b96df978abfa7b81bc2f879b2ba6dec2fcda38dec54e60b"
    )
    assert sampling["near_field_lte_2m_cell_count"] == 1_016
    assert sampling["near_field_lte_2m_valid_cell_count"] == 222
    assert sampling["all_invalid_semantic_logits_exact_unknown"] is True

    partition = training_v1.partition_parameters_v1(model)
    initial = executor._initial_model_receipt_v10(
        model, partition, migration, torch=torch
    )
    assert initial["schema"].startswith("lewm_v10_")
    assert initial["architecture"]["schema"].startswith("lewm_v10_")
    assert initial["online_attention_parameter_count"] == 16_576
    assert initial["target_attention_parameter_count"] == 16_576
    assert initial["all_online_attention_parameters_in_lift_semantic_exactly_once"]
    assert initial["all_target_attention_parameters_frozen_in_target_exactly_once"]
    semantic_mask = initial["inherited_v4_decoder"]["visibility_mask"]
    assert semantic_mask == {
        "schema": "lewm_v10_cell_volume_semantic_validity_mask_v1",
        "shape": [64, 64],
        "dtype": "bool",
        "true_cell_count": 2_062,
        "sha256": "4ebbafb6d4dd5fb13b96df978abfa7b81bc2f879b2ba6dec2fcda38dec54e60b",
        "application": "v10_post_decoder_cell_volume_validity",
        "invalid_logits": [0.0, -20.0, -20.0],
    }


def test_reused_attention_receipts_are_explicitly_rebadged_v10() -> None:
    suffix_hash = executor._v9._names_sha256_v9(
        executor.ATTENTION_PARAMETER_SUFFIXES_V10
    )
    gradient = {
        "schema": "lewm_v9_dense_local_attention_post_backward_gradient_v1",
        "update": 1,
        "parameter_suffix_inventory_sha256": suffix_hash,
    }
    rebadged_gradient = executor._rebadge_attention_receipt_v10(gradient)
    assert rebadged_gradient["schema"] == (
        "lewm_v10_cell_volume_attention_post_backward_gradient_v1"
    )
    assert rebadged_gradient["implementation"] == (
        "unchanged_v9_attention_gradient_receipt"
    )
    activity = {
        "schema": "lewm_v9_dense_local_attention_training_activity_v1",
        "update_count": 1_000,
    }
    rebadged_activity = executor._rebadge_attention_activity_v10(activity)
    assert rebadged_activity["schema"] == (
        "lewm_v10_cell_volume_attention_training_activity_v1"
    )
    with pytest.raises(RuntimeError, match="schema changed"):
        executor._rebadge_attention_receipt_v10({"schema": "lewm_v8_wrong"})


def test_write_once_calibration_and_source_order_are_fail_closed(
    tmp_path: Path,
) -> None:
    output = executor._fresh_output_root_v10(tmp_path)
    assert output == tmp_path / executor.OUTPUT_RELATIVE_PATH
    with pytest.raises(FileExistsError, match="attempt_v1 already exists"):
        executor._fresh_output_root_v10(tmp_path)
    assert executor._physical_calibration_stage_v10(True)["status"] == (
        "STAGED_FOR_SEPARATELY_FROZEN_ONE_SHOT"
    )
    assert executor._physical_calibration_stage_v10(False)["status"] == (
        "CLOSED_FULL_ARM_GATE_FAILED"
    )

    source = Path(executor.__file__).read_text()
    assert source.count("training_v9.run_fixed_training_v9(*args)") == 1
    assert source.count("_v1.score_role_v1(") == 1
    checkpoint_write = source.index('output / "checkpoint_update_1000.pt"')
    trace_write = source.index('output / "training_trace.json"')
    evaluation = source.index("_v1.score_role_v1(")
    result_write = source.index('output / "result.json"')
    assert checkpoint_write < trace_write < evaluation < result_write
    assert "torch.load(" not in source
    assert '"objective": "S+P+U+R+O"' in source
    assert '"new_loss_or_head": False' in source
    assert '"losses_changed": False' in source
    assert '"optimizer_rules_changed": False' in source
    assert '"schedule_changed": False' in source
    assert '"evaluation_changed": False' in source
    assert 'len(gate.get("checks", {})) != 24' in source
    assert "GeometryAnchoredSweptProgressSurvivalJointJepaV10(" in source
