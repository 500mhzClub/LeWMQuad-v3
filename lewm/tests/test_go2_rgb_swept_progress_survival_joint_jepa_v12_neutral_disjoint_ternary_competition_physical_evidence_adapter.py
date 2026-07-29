from __future__ import annotations

import copy
import io
from typing import Any

import pytest
import torch

from lewm.benchmarks import (
    go2_rgb_swept_progress_survival_joint_jepa_v12_neutral_disjoint_ternary_competition_physical_evidence_adapter
    as adapter,
)
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
    run_go2_rgb_swept_progress_survival_joint_jepa_v11_height_role_factorized_evidence_lift
    as training_v11,
)


def _sweep_masks() -> torch.Tensor:
    masks = torch.zeros((9, 16, 64, 64), dtype=torch.bool)
    masks[:, :, 31:33, 31:33] = True
    return masks


@pytest.fixture(scope="module")
def synthetic_model_and_initial() -> tuple[torch.nn.Module, dict[str, Any]]:
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
        encoder_state = {
            name: value.detach().clone()
            for name, value in encoder.state_dict().items()
        }
        masks = _sweep_masks()
        model = model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV12(
            encoder_state, masks
        )
        fresh_v11 = (
            v11_model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV11(
                encoder_state, masks
            )
        )
        fresh_v10 = (
            v10_model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV10(
                encoder_state, masks
            )
        )
        identity = executor._state_identity_receipt_v12(
            model,
            fresh_v11,
            fresh_v10,
            torch=torch,
            model_api=model_api,
            v11_model_api=v11_model_api,
            training_v11=training_v11,
        )
        initial = dict(
            executor._initial_model_receipt_v12(
                model,
                training_v1.partition_parameters_v1(model),
                identity,
                training_v11=training_v11,
            )
        )
        model.ema_update_count.fill_(1_000)
        return model.eval().requires_grad_(False), initial
    finally:
        torch.random.set_rng_state(caller_rng)


def _activity(
    *,
    schema: str,
    suffixes: tuple[str, ...],
    parameter_count: int,
    target_parameter_tensor_count: int,
) -> dict[str, Any]:
    return {
        "schema": schema,
        "update_count": 1_000,
        "online_parameter_count": parameter_count,
        "online_parameter_tensor_count": len(suffixes),
        "parameter_suffix_inventory_sha256": adapter._names_sha256(suffixes),
        "all_online_parameter_tensors_active_by_update_2": True,
        "first_active_update": {name: 1 for name in suffixes},
        "latest_first_active_update": 1,
        "active_update_count": 1_000,
        "minimum_active_parameter_tensor_count": len(suffixes),
        "maximum_active_parameter_tensor_count": len(suffixes),
        "minimum_gradient_l2": 0.05,
        "maximum_gradient_l2": 0.25,
        "target_parameter_tensor_count": target_parameter_tensor_count,
        "target_gradient_tensor_count": 0,
    }


@pytest.fixture(scope="module")
def checkpoint_payload(
    synthetic_model_and_initial: tuple[torch.nn.Module, dict[str, Any]],
) -> dict[str, Any]:
    model, initial = synthetic_model_and_initial
    branch = _activity(
        schema="lewm_v11_height_role_branch_attention_training_activity_v1",
        suffixes=adapter._BRANCH_ATTENTION_PARAMETER_SUFFIXES,
        parameter_count=model_api.HEIGHT_ROLE_ATTENTION_PARAMETER_COUNT_V11,
        target_parameter_tensor_count=(
            model_api.HEIGHT_ROLE_ATTENTION_PARAMETER_TENSOR_COUNT_V11
        ),
    )
    semantic = _activity(
        schema="lewm_v11_factorized_semantic_axes_training_activity_v1",
        suffixes=adapter._SEMANTIC_AXIS_PARAMETER_SUFFIXES,
        parameter_count=model_api.HEIGHT_ROLE_SEMANTIC_PARAMETER_COUNT_V11,
        target_parameter_tensor_count=0,
    )
    for name in semantic["first_active_update"]:
        if ".local." in name:
            semantic["first_active_update"][name] = 2
    semantic["latest_first_active_update"] = 2
    semantic["minimum_active_parameter_tensor_count"] = 8
    diagnostics = {
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
        "height_role_branch_attention": copy.deepcopy(branch),
        "factorized_semantic_axes": copy.deepcopy(semantic),
        "v12_contract": {
            "schema": "lewm_v12_unchanged_joint_training_contract_v1",
            "training_helper": (
                "scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v11_"
                "height_role_factorized_evidence_lift"
            ),
            "objective": "S+P+U+R+O",
            "occupied_auxiliary_coefficient": 0.5,
            "new_loss_or_weight": False,
            "height_role_branch_attention": copy.deepcopy(branch),
            "factorized_semantic_axes": copy.deepcopy(semantic),
        },
    }
    return {
        "schema": adapter.CHECKPOINT_SCHEMA,
        "development_only": True,
        "resume_authorized": False,
        "qualified": False,
        "preregistration_commit": adapter.V12_PREREGISTRATION_COMMIT,
        "constructor_initialization_seed": 20_260_712,
        "height_role_initialization_seed": 20_260_730,
        "experiment_seed": 20_260_728,
        "initialization_source": (
            "accepted_n320_encoder_and_fresh_v11_source_state_with_only_"
            "zero_parameter_neutral_ternary_algebra"
        ),
        "predecessor_experiment_checkpoint_read": False,
        "objective": "S+P+U+R+O",
        "inherited_occupied_auxiliary": copy.deepcopy(
            adapter._INHERITED_OCCUPIED_AUXILIARY
        ),
        "initial_v12_model": initial,
        "height_role_branch_attention_activity": branch,
        "factorized_semantic_axes_activity": semantic,
        "training_diagnostics": diagnostics,
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


def test_load_checkpoint_returns_exact_frozen_cpu_v12_and_uses_weights_only(
    checkpoint_payload: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    original_load = torch.load

    def tracked_load(*args: Any, **kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        return original_load(*args, **kwargs)

    def forbidden_predictor(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("physical checkpoint validation called the predictor")

    monkeypatch.setattr(adapter.torch, "load", tracked_load)
    monkeypatch.setattr(model_api.SweptProgressSurvivalHeadV1, "forward", forbidden_predictor)
    model = adapter.load_checkpoint(_serialize(checkpoint_payload))
    assert calls == [{"map_location": "cpu", "weights_only": True}]
    assert type(model) is model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV12
    assert not model.training
    assert not any(module.training for module in model.modules())
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert all(value.device.type == "cpu" for value in model.state_dict().values())
    assert model.target_hard_sync_count.item() == 1
    assert model.ema_update_count.item() == 1_000
    assert adapter.PHYSICAL_CALIBRATION_PREREGISTRATION_COMMIT == (
        "c63e98162a1b03a33225e6e0a04b67a357c7ed89"
    )
    assert adapter.__all__ == [
        "CHECKPOINT_SCHEMA",
        "PHYSICAL_CALIBRATION_PREREGISTRATION_COMMIT",
        "V12_PREREGISTRATION_COMMIT",
        "load_checkpoint",
    ]


def test_load_checkpoint_rejects_metadata_activity_and_identity_mutations(
    checkpoint_payload: dict[str, Any],
) -> None:
    wrong_objective = copy.deepcopy(checkpoint_payload)
    wrong_objective["objective"] = "S+P+U+R"
    with pytest.raises(ValueError, match="objective"):
        adapter.load_checkpoint(_serialize(wrong_objective))

    target_gradient = copy.deepcopy(checkpoint_payload)
    target_gradient["height_role_branch_attention_activity"][
        "target_gradient_tensor_count"
    ] = 1
    with pytest.raises(ValueError, match="target_gradient_tensor_count"):
        adapter.load_checkpoint(_serialize(target_gradient))

    late_semantic = copy.deepcopy(checkpoint_payload)
    suffix = adapter._SEMANTIC_AXIS_PARAMETER_SUFFIXES[0]
    late_semantic["factorized_semantic_axes_activity"]["first_active_update"][
        suffix
    ] = 3
    with pytest.raises(ValueError, match="not active by update 2"):
        adapter.load_checkpoint(_serialize(late_semantic))

    wrong_identity = copy.deepcopy(checkpoint_payload)
    wrong_identity["initial_v12_model"]["fresh_v11_state_identity"][
        "neutral_algebra_exact"
    ] = False
    with pytest.raises(ValueError, match="neutral_algebra_exact"):
        adapter.load_checkpoint(_serialize(wrong_identity))


def test_load_checkpoint_rejects_open_nonfinite_or_wrong_counter_state(
    checkpoint_payload: dict[str, Any],
) -> None:
    nonfinite = copy.deepcopy(checkpoint_payload)
    nonfinite["model_state_dict"][
        "semantic_head.free_axis.residual_output.bias"
    ][0] = float("nan")
    with pytest.raises(FloatingPointError, match="nonfinite"):
        adapter.load_checkpoint(_serialize(nonfinite))

    opened = copy.deepcopy(checkpoint_payload)
    opened["model_state_dict"]["unexpected"] = torch.zeros(())
    with pytest.raises(RuntimeError, match="Unexpected key"):
        adapter.load_checkpoint(_serialize(opened))

    wrong_counter = copy.deepcopy(checkpoint_payload)
    wrong_counter["model_state_dict"]["ema_update_count"].fill_(999)
    with pytest.raises(ValueError, match="target-update counters"):
        adapter.load_checkpoint(_serialize(wrong_counter))


def test_loaded_mechanism_preserves_role_masks_neutral_algebra_and_state(
    synthetic_model_and_initial: tuple[torch.nn.Module, dict[str, Any]],
) -> None:
    model, _initial = synthetic_model_and_initial
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    adapter._validate_loaded_mechanism(model)
    after = model.state_dict()
    assert tuple(before) == tuple(after)
    assert all(adapter._tensor_bit_exact(before[name], after[name]) for name in before)
    lift = model.bev_lift
    assert tuple(torch.nonzero(lift.floor_support_role_mask).flatten().tolist()) == (
        0,
        5,
        10,
        15,
        20,
    )
    assert not bool(
        (lift.floor_support_role_mask & lift.elevated_support_role_mask).any()
    )
    assert bool(
        (lift.floor_support_role_mask | lift.elevated_support_role_mask).all()
    )
    with torch.inference_mode():
        sampling = lift.forward_with_sampling(torch.zeros((1, 256, 192)))
        free, occupied = model.semantic_head.evidence_logits(sampling.latent)
        logits = model.semantic_logits_from_latent(sampling.latent)
    expected_free = torch.where(
        sampling.floor_cell_valid_mask,
        free,
        torch.full_like(free, -20.0),
    )
    expected_occupied = torch.where(
        sampling.elevated_cell_valid_mask,
        occupied,
        torch.full_like(occupied, -20.0),
    )
    expected = model_api.neutral_disjoint_ternary_log_probabilities_v12(
        expected_free, expected_occupied
    )
    expected = torch.where(
        sampling.cell_valid_mask[:, None],
        expected,
        expected.new_tensor((0.0, -20.0, -20.0))[None, :, None, None],
    )
    assert torch.equal(logits, expected)


@pytest.mark.parametrize("encoded", [b"", bytearray(b"not exact bytes")])
def test_load_checkpoint_requires_nonempty_exact_bytes(encoded: object) -> None:
    with pytest.raises(TypeError, match="nonempty exact bytes"):
        adapter.load_checkpoint(encoded)  # type: ignore[arg-type]
