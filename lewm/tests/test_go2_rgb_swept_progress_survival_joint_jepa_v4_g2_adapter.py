from __future__ import annotations

import copy
import hashlib
import io
from typing import Any

import numpy as np
import pytest
import torch

from lewm.benchmarks import (
    go2_rgb_swept_progress_survival_joint_jepa_v4_g2_adapter as adapter,
)
from lewm.benchmarks.traversability_metrics import TraversabilityThresholds
from lewm.models.encoders import VisionEncoder
from lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v4_residual_local_semantic_decoder import (
    GeometryAnchoredSweptProgressSurvivalJointJepaV4,
)


def _sweep_masks() -> torch.Tensor:
    masks = torch.zeros((9, 16, 64, 64), dtype=torch.bool)
    masks[:, :, 31:33, 31:33] = True
    return masks


@pytest.fixture(scope="module")
def synthetic_model() -> GeometryAnchoredSweptProgressSurvivalJointJepaV4:
    caller_rng = torch.random.get_rng_state().clone()
    try:
        torch.random.default_generator.manual_seed(9917)
        encoder = VisionEncoder(
            image_size=112,
            patch_size=7,
            hidden_dim=192,
            depth=6,
            n_heads=6,
            mlp_ratio=4,
            dropout=0.0,
        )
        encoder_state = {
            name: value.detach().clone()
            for name, value in encoder.state_dict().items()
        }
        model = GeometryAnchoredSweptProgressSurvivalJointJepaV4(
            encoder_state,
            _sweep_masks(),
        )
        model.ema_update_count.fill_(1_000)
        model.eval().requires_grad_(False)
        return model
    finally:
        torch.random.set_rng_state(caller_rng)


@pytest.fixture(scope="module")
def checkpoint_payload(
    synthetic_model: GeometryAnchoredSweptProgressSurvivalJointJepaV4,
) -> dict[str, Any]:
    anchor = synthetic_model.bev_lift.anchor_in_frustum.detach().cpu().contiguous()
    visibility_sha256 = hashlib.sha256(
        anchor.numpy().tobytes(order="C")
    ).hexdigest()
    return {
        "schema": adapter.CHECKPOINT_SCHEMA,
        "development_only": True,
        "resume_authorized": False,
        "qualified": False,
        "constructor_initialization_seed": 20_260_712,
        "semantic_decoder_initialization_seed": 20_260_713,
        "experiment_seed": 20_260_728,
        "initialization_source": "exact_n320_encoder_only",
        "predecessor_experiment_checkpoint_read": False,
        "auxiliary_objective": copy.deepcopy(adapter._AUXILIARY_OBJECTIVE),
        "initial_semantic_decoder": {
            "architecture": copy.deepcopy(
                adapter._SEMANTIC_DECODER_ARCHITECTURE
            ),
            "initial_residual_output_exactly_zero": True,
            "semantic_parameter_count": 37_318,
            "added_parameter_count": 37_123,
            "all_semantic_parameters_in_lift_semantic_exactly_once": True,
            "visibility_mask": {
                "shape": [64, 64],
                "dtype": "bool",
                "true_cell_count": int(anchor.sum().item()),
                "sha256": visibility_sha256,
                "application": "inherited_post_logits",
            },
        },
        "accounting": copy.deepcopy(adapter._TERMINAL_ACCOUNTING),
        "model_state_dict": {
            name: value.detach().cpu().contiguous().clone()
            for name, value in synthetic_model.state_dict().items()
        },
    }


def _serialize(payload: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


@pytest.fixture(scope="module")
def checkpoint_bytes(checkpoint_payload: dict[str, Any]) -> bytes:
    return _serialize(checkpoint_payload)


@pytest.fixture(scope="module")
def loaded_model(
    checkpoint_bytes: bytes,
) -> GeometryAnchoredSweptProgressSurvivalJointJepaV4:
    return adapter.load_checkpoint(checkpoint_bytes)


def test_load_checkpoint_strictly_reconstructs_frozen_v4(
    loaded_model: GeometryAnchoredSweptProgressSurvivalJointJepaV4,
    checkpoint_payload: dict[str, Any],
) -> None:
    assert type(loaded_model) is GeometryAnchoredSweptProgressSurvivalJointJepaV4
    assert not loaded_model.training
    assert not any(parameter.requires_grad for parameter in loaded_model.parameters())
    assert loaded_model.action_vocabulary == adapter.ACTION_ORDER
    assert loaded_model.ema_update_count.item() == 1_000
    assert loaded_model.state_dict().keys() == checkpoint_payload[
        "model_state_dict"
    ].keys()


def test_load_checkpoint_rejects_open_schema_and_nonfinite_state(
    checkpoint_payload: dict[str, Any],
) -> None:
    open_schema = dict(checkpoint_payload)
    open_schema["unexpected"] = True
    with pytest.raises(ValueError, match="keys changed"):
        adapter.load_checkpoint(_serialize(open_schema))

    nonfinite = dict(checkpoint_payload)
    nonfinite_state = dict(checkpoint_payload["model_state_dict"])
    name = "semantic_head.residual_output.bias"
    nonfinite_state[name] = nonfinite_state[name].clone()
    nonfinite_state[name][0] = float("nan")
    nonfinite["model_state_dict"] = nonfinite_state
    with pytest.raises(FloatingPointError, match="nonfinite"):
        adapter.load_checkpoint(_serialize(nonfinite))


def test_infer_one_exposes_physical_head_and_joint_predictor(
    loaded_model: GeometryAnchoredSweptProgressSurvivalJointJepaV4,
) -> None:
    counters_before = (
        loaded_model.target_hard_sync_count.clone(),
        loaded_model.ema_update_count.clone(),
    )
    output = adapter.infer_one(
        loaded_model,
        {
            "schema": adapter.INFERENCE_INPUT_SCHEMA,
            "rgb_f32_chw": np.zeros((3, 112, 112), dtype=np.float32).tolist(),
        },
    )
    logits = np.asarray(output["physical_logits_f32_chw"])
    probabilities = np.asarray(output["physical_probabilities_f32_chw"])
    anchor = np.asarray(output["anchor_in_frustum_bool_hw"])
    survival = np.asarray(output["all_action_survival_logits_f32"])
    assert output["class_order"] == ["unknown", "free", "occupied"]
    assert output["action_order"] == list(adapter.ACTION_ORDER)
    assert logits.shape == probabilities.shape == (3, 64, 64)
    assert anchor.shape == (64, 64) and anchor.dtype == np.bool_
    assert survival.shape == (9, 16)
    assert np.isfinite(logits).all() and np.isfinite(survival).all()
    assert np.allclose(probabilities.sum(axis=0), 1.0, atol=1e-6)
    expected_invalid = np.asarray((0.0, -20.0, -20.0))[:, None]
    assert np.array_equal(
        logits[:, ~anchor],
        np.broadcast_to(expected_invalid, logits[:, ~anchor].shape),
    )
    validated_probabilities, validated_anchor = adapter._validated_output(output)
    assert np.allclose(validated_probabilities, probabilities)
    assert np.array_equal(validated_anchor, anchor)
    assert torch.equal(loaded_model.target_hard_sync_count, counters_before[0])
    assert torch.equal(loaded_model.ema_update_count, counters_before[1])


def test_infer_one_rejects_target_leakage_and_raw_range(
    loaded_model: GeometryAnchoredSweptProgressSurvivalJointJepaV4,
) -> None:
    rgb = np.zeros((3, 112, 112), dtype=np.float32).tolist()
    with pytest.raises(ValueError, match="extra"):
        adapter.infer_one(
            loaded_model,
            {
                "schema": adapter.INFERENCE_INPUT_SCHEMA,
                "rgb_f32_chw": rgb,
                "targets": {"labels": []},
            },
        )
    outside_range = np.zeros((3, 112, 112), dtype=np.float32)
    outside_range[0, 0, 0] = 3.0
    with pytest.raises(ValueError, match="normalized RGB range"):
        adapter.infer_one(
            loaded_model,
            {
                "schema": adapter.INFERENCE_INPUT_SCHEMA,
                "rgb_f32_chw": outside_range.tolist(),
            },
        )


def _perfect_physical_output(labels: np.ndarray) -> dict[str, Any]:
    probabilities = np.full((3, 64, 64), 0.01, dtype=np.float64)
    probabilities[0, labels == 0] = 0.98
    probabilities[1, labels == 1] = 0.98
    probabilities[2, labels == 2] = 0.98
    logits = np.log(probabilities)
    return {
        "schema": adapter.INFERENCE_OUTPUT_SCHEMA,
        "class_order": list(adapter.CLASS_ORDER),
        "action_order": list(adapter.ACTION_ORDER),
        "physical_logits_f32_chw": logits.tolist(),
        "physical_probabilities_f32_chw": probabilities.tolist(),
        "anchor_in_frustum_bool_hw": np.ones((64, 64), dtype=np.bool_).tolist(),
        "all_action_survival_logits_f32": np.zeros((9, 16)).tolist(),
    }


def test_physical_batch_scoring_reuses_metrics_and_excludes_routing() -> None:
    labels = np.zeros((1, 64, 64), dtype=np.int64)
    labels[:, :24] = 1
    labels[:, 24:44] = 2
    output = _perfect_physical_output(labels[0])
    thresholds = TraversabilityThresholds(
        free_probability_min=0.9,
        occupied_probability_max=0.05,
        unknown_probability_max=0.05,
        occupied_detection_min=0.9,
    )
    scored = adapter.score_physical_evidence_batch(
        [output],
        labels,
        np.ones_like(labels, dtype=np.float64),
        thresholds=thresholds,
        evaluation_mask=np.ones_like(labels, dtype=np.bool_),
    )
    assert scored["occupancy_target_space"] == "observable_physical_occupancy"
    assert scored["physical_evidence"][
        "admitted_observable_physical_free_precision"
    ] == 1.0
    assert scored["physical_evidence"][
        "directly_observable_physical_obstacle_recall_within_2m"
    ] == 1.0
    assert scored["g2"]["passes"] is False
    assert scored["g2"]["checks"] == {
        "heldout_probability_calibration_applied": False,
        "admitted_observable_physical_free_precision_ge_0_99": True,
        "directly_observable_physical_obstacle_recall_within_2m_ge_0_95": True,
        "useful_observable_physical_free_recall_ge_0_90": True,
    }
    assert scored["g2"]["routing_included"] is False
    assert scored["routing"] == adapter.routing_not_applicable_receipt()
    assert scored["routing"]["status"] == "NOT_APPLICABLE"


def test_physical_batch_scoring_rejects_probability_logit_disagreement() -> None:
    labels = np.ones((1, 64, 64), dtype=np.int64)
    output = _perfect_physical_output(labels[0])
    output["physical_probabilities_f32_chw"][1][0][0] = 0.5
    with pytest.raises(ValueError, match="sum to one|match the supplied logits"):
        adapter.score_physical_evidence_batch(
            [output],
            labels,
            np.ones_like(labels, dtype=np.float64),
            thresholds=TraversabilityThresholds(0.9, 0.05, 0.05, 0.9),
        )
