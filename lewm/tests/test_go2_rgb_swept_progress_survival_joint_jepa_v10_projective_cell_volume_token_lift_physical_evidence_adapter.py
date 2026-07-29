from __future__ import annotations

import copy
import io
from typing import Any

import pytest
import torch

from lewm.benchmarks import (
    go2_rgb_swept_progress_survival_joint_jepa_v10_projective_cell_volume_token_lift_physical_evidence_adapter
    as adapter,
)
from lewm.models.encoders import VisionEncoder
from lewm.models import (
    geometry_anchored_swept_progress_survival_joint_jepa_v4_residual_local_semantic_decoder
    as v4_model_api,
)
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


def _sweep_masks() -> torch.Tensor:
    masks = torch.zeros((9, 16, 64, 64), dtype=torch.bool)
    masks[:, :, 31:33, 31:33] = True
    return masks


@pytest.fixture(scope="module")
def synthetic_model_and_initial() -> tuple[torch.nn.Module, dict[str, Any]]:
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
        masks = _sweep_masks()
        model = model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV10(
            encoder_state, masks
        )
        fresh_v9 = v9_model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV9(
            encoder_state, masks
        )
        clean_v4 = v4_model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV4(
            encoder_state, masks
        )
        nested = executor._v9._migration_receipt_v9(
            fresh_v9, clean_v4, torch=torch, model_api=v9_model_api
        )
        migration = dict(
            executor._migration_receipt_v10(
                model, fresh_v9, torch=torch, model_api=model_api
            )
        )
        migration["fresh_v9_clean_v4_migration"] = nested
        migration["caller_cpu_rng_state_restored"] = True
        initial = dict(
            executor._initial_model_receipt_v10(
                model,
                training_v1.partition_parameters_v1(model),
                migration,
                torch=torch,
            )
        )
        model.ema_update_count.fill_(1_000)
        return model.eval().requires_grad_(False), initial
    finally:
        torch.random.set_rng_state(caller_rng)


def _activity() -> dict[str, Any]:
    return {
        "schema": "lewm_v10_cell_volume_attention_training_activity_v1",
        "update_count": 1_000,
        "online_parameter_count": 16_576,
        "online_parameter_tensor_count": 7,
        "parameter_suffix_inventory_sha256": adapter._ATTENTION_PARAMETER_INVENTORY_SHA256,
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
        "implementation": "unchanged_v9_attention_gradient_receipts",
    }


@pytest.fixture(scope="module")
def checkpoint_payload(
    synthetic_model_and_initial: tuple[torch.nn.Module, dict[str, Any]],
) -> dict[str, Any]:
    model, initial = synthetic_model_and_initial
    activity = _activity()
    return {
        "schema": adapter.CHECKPOINT_SCHEMA,
        "development_only": True,
        "resume_authorized": False,
        "qualified": False,
        "preregistration_commit": adapter.V10_PREREGISTRATION_COMMIT,
        "constructor_initialization_seed": 20_260_712,
        "semantic_decoder_initialization_seed": 20_260_713,
        "cell_volume_attention_initialization_seed": 20_260_729,
        "experiment_seed": 20_260_728,
        "initialization_source": (
            "exact_n320_encoder_and_fresh_v9_v4_with_only_preregistered_"
            "geometry_replacement"
        ),
        "predecessor_experiment_checkpoint_read": False,
        "objective": "S+P+U+R+O",
        "inherited_occupied_auxiliary": copy.deepcopy(
            adapter._INHERITED_OCCUPIED_AUXILIARY
        ),
        "initial_v10_model": initial,
        "cell_volume_attention_activity": activity,
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
            "v10_contract": {
                "schema": "lewm_v10_unchanged_joint_training_contract_v1",
                "objective": "S+P+U+R+O",
                "occupied_auxiliary_coefficient": 0.5,
                "new_loss_or_head": False,
                "training_core": "unchanged_v9_wrapper_over_v3_v4",
            },
        },
        "accounting": copy.deepcopy(adapter._TERMINAL_ACCOUNTING),
        "model_state_dict": {
            name: value.detach().cpu().contiguous().clone()
            for name, value in model.state_dict().items()
        },
    }


def _serialize(payload: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def test_load_checkpoint_returns_exact_frozen_cpu_v10_and_uses_weights_only(
    checkpoint_payload: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    original = torch.load

    def tracked(*args: Any, **kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(adapter.torch, "load", tracked)
    model = adapter.load_checkpoint(_serialize(checkpoint_payload))
    assert calls == [{"map_location": "cpu", "weights_only": True}]
    assert type(model) is model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV10
    assert not model.training
    assert not any(module.training for module in model.modules())
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert all(value.device.type == "cpu" for value in model.state_dict().values())
    assert model.target_hard_sync_count.item() == 1
    assert model.ema_update_count.item() == 1_000
    assert adapter.PHYSICAL_CALIBRATION_PREREGISTRATION_COMMIT == (
        "6bc4dca93daf0e220bbaa4fc524470addb880e21"
    )


def test_load_checkpoint_rejects_metadata_activity_and_migration_mutations(
    checkpoint_payload: dict[str, Any],
) -> None:
    wrong_objective = copy.deepcopy(checkpoint_payload)
    wrong_objective["objective"] = "S+P+U+R"
    with pytest.raises(ValueError, match="objective"):
        adapter.load_checkpoint(_serialize(wrong_objective))

    target_gradient = copy.deepcopy(checkpoint_payload)
    target_gradient["cell_volume_attention_activity"][
        "target_gradient_tensor_count"
    ] = 1
    with pytest.raises(ValueError, match="target_gradient_tensor_count"):
        adapter.load_checkpoint(_serialize(target_gradient))

    wrong_mask = copy.deepcopy(checkpoint_payload)
    wrong_mask["initial_v10_model"]["migration"]["sampling_receipt"][
        "cell_valid_count"
    ] = 2_061
    with pytest.raises(ValueError, match="cell_valid_count"):
        adapter.load_checkpoint(_serialize(wrong_mask))


def test_load_checkpoint_rejects_open_or_nonfinite_state(
    checkpoint_payload: dict[str, Any],
) -> None:
    nonfinite = copy.deepcopy(checkpoint_payload)
    nonfinite["model_state_dict"]["semantic_head.residual_output.bias"][0] = float(
        "nan"
    )
    with pytest.raises(FloatingPointError, match="nonfinite"):
        adapter.load_checkpoint(_serialize(nonfinite))

    opened = copy.deepcopy(checkpoint_payload)
    opened["model_state_dict"]["unexpected"] = torch.zeros(())
    with pytest.raises(RuntimeError, match="Unexpected key"):
        adapter.load_checkpoint(_serialize(opened))


def test_loaded_mechanism_uses_cell_volume_not_ground_anchor_semantic_mask(
    synthetic_model_and_initial: tuple[torch.nn.Module, dict[str, Any]],
) -> None:
    model, _initial = synthetic_model_and_initial
    adapter._validate_loaded_mechanism(model)
    with torch.inference_mode():
        sampling = model.bev_lift.forward_with_sampling(torch.zeros((1, 256, 192)))
        raw = model.semantic_head(sampling.latent)
        logits = model.semantic_logits_from_latent(sampling.latent)
    newly_valid = sampling.cell_valid_mask & ~sampling.anchor_in_frustum
    assert bool(newly_valid.any())
    mask = newly_valid[:, None].expand_as(logits)
    assert torch.equal(logits.masked_select(mask), raw.masked_select(mask))
    assert int(model.bev_lift.cell_valid_mask.sum()) == 2_062
    assert int(model.bev_lift.anchor_in_frustum.sum()) < 2_062
