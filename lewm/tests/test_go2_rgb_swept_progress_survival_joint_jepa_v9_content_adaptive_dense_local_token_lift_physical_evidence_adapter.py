from __future__ import annotations

import copy
import hashlib
import io
from typing import Any

import pytest
import torch

from lewm.benchmarks import (
    go2_rgb_swept_progress_survival_joint_jepa_v9_content_adaptive_dense_local_token_lift_physical_evidence_adapter
    as adapter,
)
from lewm.models.encoders import VisionEncoder
from lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v9_content_adaptive_dense_local_token_lift import (
    GeometryAnchoredSweptProgressSurvivalJointJepaV9,
)


def _sweep_masks() -> torch.Tensor:
    masks = torch.zeros((9, 16, 64, 64), dtype=torch.bool)
    masks[:, :, 31:33, 31:33] = True
    return masks


@pytest.fixture(scope="module")
def synthetic_model() -> GeometryAnchoredSweptProgressSurvivalJointJepaV9:
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
        model = GeometryAnchoredSweptProgressSurvivalJointJepaV9(
            encoder_state,
            _sweep_masks(),
        )
        model.ema_update_count.fill_(1_000)
        return model.eval().requires_grad_(False)
    finally:
        torch.random.set_rng_state(caller_rng)


def _initial_v9_receipt(
    model: GeometryAnchoredSweptProgressSurvivalJointJepaV9,
) -> dict[str, Any]:
    anchor = model.bev_lift.anchor_in_frustum.detach().cpu().contiguous()
    visibility_sha256 = hashlib.sha256(
        anchor.numpy().tobytes(order="C")
    ).hexdigest()
    migration = {
        "source": "fresh clean V4 construction with identical N320 state and masks",
        "removed_state_names": copy.deepcopy(adapter._REMOVED_STATE_NAMES),
        "added_state_names": copy.deepcopy(adapter._ADDED_STATE_NAMES),
        "all_inherited_state_tensors_bit_exact": True,
        "inherited_state_tensor_count": 220,
        "inherited_state_name_inventory_sha256": (
            "55439423b29b61060e9a89279f0f19ecd4cf81cafb64bf3c15769a565647602c"
        ),
        "removed_parameter_count_per_online_or_target_lift": 49_152,
        "added_attention_parameter_count_per_online_or_target_lift": 16_576,
        "added_attention_parameter_tensor_count_per_online_or_target_lift": 7,
        "online_target_attention_initial_copy_exact": True,
        "online_target_support_offsets_initial_copy_exact": True,
        "target_attention_initial_gradient_tensor_count": 0,
        "attention_initialization_bit_exact": True,
        "attention_biases_exact_zero": True,
        "key_projection_bias": False,
        "sampling_receipt": copy.deepcopy(adapter._INITIAL_SAMPLING_RECEIPT),
        "caller_cpu_rng_state_restored": True,
    }
    return {
        "architecture": copy.deepcopy(adapter._DENSE_LOCAL_LIFT_ARCHITECTURE),
        "migration": migration,
        "inherited_v4_decoder": {
            "architecture": copy.deepcopy(
                adapter._INHERITED_DECODER_ARCHITECTURE
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
        "online_attention_parameter_count": 16_576,
        "online_attention_parameter_tensor_count": 7,
        "target_attention_parameter_count": 16_576,
        "target_attention_parameter_tensor_count": 7,
        "attention_parameter_suffix_inventory_sha256": (
            adapter._ATTENTION_PARAMETER_INVENTORY_SHA256
        ),
        "all_online_attention_parameters_in_lift_semantic_exactly_once": True,
        "all_target_attention_parameters_frozen_in_target_exactly_once": True,
        "target_initial_copy_exact": True,
        "initial_hard_sync_count": 1,
        "initial_ema_update_count": 0,
    }


def _attention_activity() -> dict[str, Any]:
    return {
        "schema": "lewm_v9_dense_local_attention_training_activity_v1",
        "update_count": 1_000,
        "online_parameter_count": 16_576,
        "online_parameter_tensor_count": 7,
        "parameter_suffix_inventory_sha256": (
            adapter._ATTENTION_PARAMETER_INVENTORY_SHA256
        ),
        "all_online_parameter_tensors_active_by_update_2": True,
        "first_active_update": {
            name: 1 for name in adapter._ATTENTION_PARAMETER_SUFFIXES
        },
        "latest_first_active_update": 1,
        "active_update_count": 1_000,
        "minimum_active_parameter_tensor_count": 7,
        "maximum_active_parameter_tensor_count": 7,
        "minimum_gradient_l2": 0.05,
        "maximum_gradient_l2": 0.25,
        "target_gradient_tensor_count": 0,
    }


@pytest.fixture(scope="module")
def checkpoint_payload(
    synthetic_model: GeometryAnchoredSweptProgressSurvivalJointJepaV9,
) -> dict[str, Any]:
    activity = _attention_activity()
    return {
        "schema": adapter.CHECKPOINT_SCHEMA,
        "development_only": True,
        "resume_authorized": False,
        "qualified": False,
        "preregistration_commit": adapter.V9_PREREGISTRATION_COMMIT,
        "preimplementation_amendment_commit": (
            adapter.V9_PREIMPLEMENTATION_AMENDMENT_COMMIT
        ),
        "constructor_initialization_seed": 20_260_712,
        "semantic_decoder_initialization_seed": 20_260_713,
        "dense_local_attention_initialization_seed": 20_260_729,
        "experiment_seed": 20_260_728,
        "initialization_source": (
            "exact_n320_encoder_and_clean_v4_with_preregistered_lift_replacement"
        ),
        "predecessor_experiment_checkpoint_read": False,
        "inherited_occupied_auxiliary": copy.deepcopy(
            adapter._INHERITED_OCCUPIED_AUXILIARY
        ),
        "initial_v9_model": _initial_v9_receipt(synthetic_model),
        "dense_local_attention_activity": activity,
        "training_diagnostics": {
            "ranking_active_microbatch_count": 4_000,
            "ranking_eligible_pair_count": 284_795,
            "survival_supervised_decision_count": 1_318_068,
            "minimum_gradient_l2": {
                "encoder": 1.0,
                "lift_semantic": 1.1,
                "predictor": 1.2,
            },
            "maximum_gradient_l2": {
                "encoder": 10.0,
                "lift_semantic": 11.0,
                "predictor": 12.0,
            },
            "dense_local_attention": copy.deepcopy(activity),
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


def test_load_checkpoint_uses_weights_only_and_returns_exact_frozen_cpu_v9(
    checkpoint_payload: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    original_load = torch.load

    def tracked_load(*args: Any, **kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(adapter.torch, "load", tracked_load)
    model = adapter.load_checkpoint(_serialize(checkpoint_payload))
    assert calls == [{"map_location": "cpu", "weights_only": True}]
    assert type(model) is GeometryAnchoredSweptProgressSurvivalJointJepaV9
    assert not model.training
    assert not any(module.training for module in model.modules())
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert all(tensor.device.type == "cpu" for tensor in model.state_dict().values())
    assert model.target_hard_sync_count.item() == 1
    assert model.ema_update_count.item() == 1_000
    assert model.state_dict().keys() == checkpoint_payload["model_state_dict"].keys()
    for name, expected in checkpoint_payload["model_state_dict"].items():
        actual = model.state_dict()[name]
        assert actual.dtype == expected.dtype, name
        assert torch.equal(
            actual.reshape(-1).view(torch.uint8),
            expected.reshape(-1).view(torch.uint8),
        ), name
    assert adapter.PHYSICAL_CALIBRATION_PREREGISTRATION_COMMIT == (
        "2f561d26f0b6ca154b6f4eab00dba228f8bc8c9e"
    )
    assert adapter.PHYSICAL_CALIBRATION_SOURCE_CLOSURE_AMENDMENT_COMMIT == (
        "b2465b2148b999b216078d53fe9bd556e63703e0"
    )


def test_load_checkpoint_rejects_top_level_provenance_accounting_and_auxiliary(
    checkpoint_payload: dict[str, Any],
) -> None:
    extra = dict(checkpoint_payload)
    extra["unexpected"] = True
    with pytest.raises(ValueError, match="checkpoint keys changed"):
        adapter.load_checkpoint(_serialize(extra))

    wrong_commit = dict(checkpoint_payload)
    wrong_commit["preregistration_commit"] = "0" * 40
    with pytest.raises(ValueError, match="preregistration_commit"):
        adapter.load_checkpoint(_serialize(wrong_commit))

    wrong_accounting = dict(checkpoint_payload)
    accounting = dict(checkpoint_payload["accounting"])
    accounting["updates"] = 999
    wrong_accounting["accounting"] = accounting
    with pytest.raises(ValueError, match="accounting.updates"):
        adapter.load_checkpoint(_serialize(wrong_accounting))

    wrong_auxiliary = dict(checkpoint_payload)
    auxiliary = dict(checkpoint_payload["inherited_occupied_auxiliary"])
    auxiliary["coefficient"] = 0.25
    wrong_auxiliary["inherited_occupied_auxiliary"] = auxiliary
    with pytest.raises(ValueError, match="inherited_occupied_auxiliary"):
        adapter.load_checkpoint(_serialize(wrong_auxiliary))


def test_load_checkpoint_rejects_initial_receipt_and_attention_activity_mutation(
    checkpoint_payload: dict[str, Any],
) -> None:
    changed_architecture = dict(checkpoint_payload)
    initial = copy.deepcopy(checkpoint_payload["initial_v9_model"])
    initial["architecture"]["geometry"]["support_count"] = 24
    changed_architecture["initial_v9_model"] = initial
    with pytest.raises(ValueError, match="support_count"):
        adapter.load_checkpoint(_serialize(changed_architecture))

    late_attention = dict(checkpoint_payload)
    activity = copy.deepcopy(checkpoint_payload["dense_local_attention_activity"])
    activity["first_active_update"]["key_projection.weight"] = 3
    late_attention["dense_local_attention_activity"] = activity
    with pytest.raises(ValueError, match="not active by update 2"):
        adapter.load_checkpoint(_serialize(late_attention))

    target_gradient = dict(checkpoint_payload)
    activity = copy.deepcopy(checkpoint_payload["dense_local_attention_activity"])
    activity["target_gradient_tensor_count"] = 1
    target_gradient["dense_local_attention_activity"] = activity
    with pytest.raises(ValueError, match="target_gradient_tensor_count"):
        adapter.load_checkpoint(_serialize(target_gradient))

    mismatched_diagnostics = dict(checkpoint_payload)
    diagnostics = copy.deepcopy(checkpoint_payload["training_diagnostics"])
    diagnostics["dense_local_attention"]["maximum_gradient_l2"] = 0.5
    mismatched_diagnostics["training_diagnostics"] = diagnostics
    with pytest.raises(ValueError, match="dense_local_attention"):
        adapter.load_checkpoint(_serialize(mismatched_diagnostics))


def test_load_checkpoint_rejects_nonfinite_noncontiguous_and_open_state(
    checkpoint_payload: dict[str, Any],
) -> None:
    nonfinite_payload = dict(checkpoint_payload)
    nonfinite_state = dict(checkpoint_payload["model_state_dict"])
    name = "semantic_head.residual_output.bias"
    nonfinite_state[name] = nonfinite_state[name].clone()
    nonfinite_state[name][0] = float("nan")
    nonfinite_payload["model_state_dict"] = nonfinite_state
    with pytest.raises(FloatingPointError, match="nonfinite"):
        adapter.load_checkpoint(_serialize(nonfinite_payload))

    noncontiguous_payload = dict(checkpoint_payload)
    noncontiguous_state = dict(checkpoint_payload["model_state_dict"])
    name = "encoder.pos_embed"
    noncontiguous_state[name] = noncontiguous_state[name].transpose(1, 2)
    assert not noncontiguous_state[name].is_contiguous()
    noncontiguous_payload["model_state_dict"] = noncontiguous_state
    with pytest.raises(TypeError, match="not dense contiguous"):
        adapter.load_checkpoint(_serialize(noncontiguous_payload))

    open_state_payload = dict(checkpoint_payload)
    open_state = dict(checkpoint_payload["model_state_dict"])
    open_state["unexpected"] = torch.zeros(())
    open_state_payload["model_state_dict"] = open_state
    with pytest.raises(RuntimeError, match="Unexpected key"):
        adapter.load_checkpoint(_serialize(open_state_payload))


def test_load_checkpoint_rejects_terminal_counter_or_decoder_receipt_mismatch(
    checkpoint_payload: dict[str, Any],
) -> None:
    wrong_counter_payload = dict(checkpoint_payload)
    wrong_counter_state = dict(checkpoint_payload["model_state_dict"])
    wrong_counter_state["ema_update_count"] = torch.tensor(999, dtype=torch.long)
    wrong_counter_payload["model_state_dict"] = wrong_counter_state
    with pytest.raises(ValueError, match="target-update counters"):
        adapter.load_checkpoint(_serialize(wrong_counter_payload))

    wrong_visibility_payload = dict(checkpoint_payload)
    initial = copy.deepcopy(checkpoint_payload["initial_v9_model"])
    initial["inherited_v4_decoder"]["visibility_mask"]["sha256"] = "0" * 64
    wrong_visibility_payload["initial_v9_model"] = initial
    with pytest.raises(ValueError, match="visibility SHA-256"):
        adapter.load_checkpoint(_serialize(wrong_visibility_payload))
