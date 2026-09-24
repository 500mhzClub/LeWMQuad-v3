from __future__ import annotations

import hashlib
import json
import math

import pytest
import torch
import torch.nn.functional as F

from lewm.models.egomotion_bev_jepa import (
    BevDecoder,
    DYNAMIC_PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT,
    EgomotionBevJepa,
    GLOBAL_CROSS_ATTENTION_LIFT,
    PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT,
    PROJECTIVE_COLUMN_ATTENTION_LIFT,
    PROJECTIVE_FOOTPRINT_ATTENTION_LIFT,
    _cell_square_horizontal_offsets,
    _dynamic_projective_cell_square_attention_geometry,
    bev_variance_floor_loss,
    build_projective_query_support_contract,
    validate_projective_query_support_binding,
    warp_bev_current_to_next,
)
from lewm.benchmarks.go2_dynamic_cell_square_projection import (
    build_dynamic_cell_square_support_mask,
)


def _model() -> EgomotionBevJepa:
    return EgomotionBevJepa(
        image_size=28,
        patch_size=14,
        encoder_dim=12,
        encoder_depth=1,
        encoder_heads=3,
        bev_dim=8,
        bev_size=(8, 8),
        forward_range_m=(-0.5, 0.5),
        left_range_m=(-0.5, 0.5),
        action_dim=4,
        predictor_hidden_dim=12,
        target_ema_momentum=0.5,
        variance_target_std=0.2,
    )


def _projective_decoder(**overrides) -> BevDecoder:
    kwargs = {
        "token_dim": 6,
        "bev_dim": 8,
        "token_side": 3,
        "bev_size": (2, 3),
        "forward_range_m": (1.0, 2.0),
        "left_range_m": (-1.0, 1.0),
        "attention_heads": 2,
        "lift_type": PROJECTIVE_COLUMN_ATTENTION_LIFT,
        "projective_horizontal_fov_deg": 90.0,
        "projective_vertical_fov_deg": 90.0,
        "projective_camera_xyz_body_m": (0.0, 0.0, 0.0),
        "projective_camera_rpy_body_rad": (0.0, 0.0, 0.0),
        "projective_near_m": 0.1,
        "projective_vertical_anchor_z_body_m": (0.0,),
    }
    kwargs.update(overrides)
    return BevDecoder(**kwargs)


def _footprint_decoder(**overrides) -> BevDecoder:
    kwargs = {
        "lift_type": PROJECTIVE_FOOTPRINT_ATTENTION_LIFT,
        "projective_footprint_radius_m": 0.5,
        "projective_footprint_perimeter_samples": 4,
    }
    kwargs.update(overrides)
    return _projective_decoder(**kwargs)


def _cell_square_decoder(**overrides) -> BevDecoder:
    kwargs = {
        "lift_type": PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT,
        "projective_output_cell_size_m": 0.1,
    }
    kwargs.update(overrides)
    return _projective_decoder(**kwargs)


def _dynamic_cell_square_decoder(**overrides) -> BevDecoder:
    kwargs = {
        "lift_type": DYNAMIC_PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT,
        "projective_output_cell_size_m": 0.1,
    }
    kwargs.update(overrides)
    return _projective_decoder(**kwargs)


def _tilt_quaternion_xyzw(
    *,
    roll_rad: float = 0.0,
    pitch_rad: float = 0.0,
) -> tuple[float, float, float, float]:
    half_roll = 0.5 * float(roll_rad)
    half_pitch = 0.5 * float(pitch_rad)
    return (
        math.sin(half_roll) * math.cos(half_pitch),
        math.cos(half_roll) * math.sin(half_pitch),
        -math.sin(half_roll) * math.sin(half_pitch),
        math.cos(half_roll) * math.cos(half_pitch),
    )


def _registered_dynamic_geometry(
    quaternions: torch.Tensor,
    yaws: torch.Tensor,
    *,
    token_side: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    forward = torch.linspace(-0.95, 5.35, 64)
    left = torch.linspace(-3.15, 3.15, 64)
    forward_grid, left_grid = torch.meshgrid(forward, left, indexing="ij")
    return _dynamic_projective_cell_square_attention_geometry(
        metric_forward_grid=forward_grid,
        metric_left_grid=left_grid,
        token_side=token_side,
        horizontal_fov_deg=78.323,
        vertical_fov_deg=62.8370386364,
        camera_xyz_body_m=(0.326, 0.0, 0.043),
        camera_rpy_body_rad=(0.0, 0.0, 0.0),
        near_m=0.05,
        vertical_anchor_z_body_m=(-0.333, -0.133, 0.067, 0.267, 0.467),
        horizontal_offsets_body_m=_cell_square_horizontal_offsets(0.1),
        sigma_tokens=2.0,
        bias_floor=-6.0,
        base_quat_world_xyzw=quaternions,
        stored_base_yaw_rad=yaws,
    )


def _physical_manifest(*, cell_size_m: float = 0.1) -> dict:
    aggregation = {
        "schema": "lewm_observable_physical_aggregation_v1",
        "source_cell_size_m": 0.05,
        "output_cell_size_m": cell_size_m,
        "free_rule": "unit free rule",
        "occupied_rule": "unit occupied rule",
        "known_class_precedence": "OCCUPIED_then_FREE_else_UNKNOWN",
        "collision_geometry_veto": "unit collision veto",
    }
    encoded = json.dumps(
        aggregation, sort_keys=True, separators=(",", ":")
    ).encode()
    aggregation["contract_sha256"] = hashlib.sha256(encoded).hexdigest()
    return {
        "schema": "lewm_go2_paired_navigation_dataset_v3",
        "local_grid": {"cell_size_m": cell_size_m},
        "label_semantics": {
            "label_contract": "observable_physical_occupancy_v3",
            "target_occupancy_space": "observable_physical_occupancy",
            "per_frame_configuration_classes_supervised": False,
            "physical_aggregation": aggregation,
        },
    }


def _projective_model(**overrides) -> EgomotionBevJepa:
    kwargs = {
        "image_size": 28,
        "patch_size": 14,
        "encoder_dim": 12,
        "encoder_depth": 1,
        "encoder_heads": 3,
        "bev_dim": 8,
        "bev_size": (4, 4),
        "forward_range_m": (0.1, 0.7),
        "left_range_m": (-0.3, 0.3),
        "action_dim": 4,
        "bev_attention_heads": 2,
        "bev_lift_type": PROJECTIVE_COLUMN_ATTENTION_LIFT,
        "projective_horizontal_fov_deg": 100.0,
        "projective_vertical_fov_deg": 100.0,
        "projective_camera_xyz_body_m": (0.0, 0.0, 0.1),
        "projective_camera_rpy_body_rad": (0.0, 0.0, 0.0),
        "projective_near_m": 0.05,
        "projective_vertical_anchor_z_body_m": (-0.1, 0.1, 0.3),
        "predictor_hidden_dim": 12,
        "target_ema_momentum": 0.5,
        "variance_target_std": 0.2,
    }
    kwargs.update(overrides)
    return EgomotionBevJepa(**kwargs)


def _legacy_decoder_reference(decoder: BevDecoder, patch_tokens: torch.Tensor) -> torch.Tensor:
    tokens = decoder.token_project(patch_tokens)
    queries = decoder.coordinate_query(
        decoder.coordinate_features.to(dtype=patch_tokens.dtype)
    )
    queries = queries + decoder.query_bias.to(dtype=queries.dtype)
    queries = queries[None].expand(patch_tokens.shape[0], -1, -1)
    attended, _weights = decoder.cross_attention(
        queries,
        tokens,
        tokens,
        need_weights=False,
    )
    features = decoder.query_norm(queries + attended)
    features = features.transpose(1, 2).reshape(
        patch_tokens.shape[0],
        -1,
        decoder.bev_size[0],
        decoder.bev_size[1],
    )
    return decoder.refine(features)


def test_bev_warp_identity_and_forward_translation() -> None:
    current = torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5)
    identity, identity_mask = warp_bev_current_to_next(
        current,
        torch.zeros(1, 3),
        forward_range_m=(-0.2, 0.2),
        left_range_m=(-0.2, 0.2),
    )
    translated, translated_mask = warp_bev_current_to_next(
        current,
        torch.tensor([[0.1, 0.0, 0.0]]),
        forward_range_m=(-0.2, 0.2),
        left_range_m=(-0.2, 0.2),
    )

    assert torch.allclose(identity, current)
    assert identity_mask.all()
    assert torch.allclose(translated[:, :, :-1], current[:, :, 1:])
    assert not translated_mask[:, :, -1].any()


def test_bev_warp_lateral_and_positive_yaw_conventions() -> None:
    current = torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5)
    translated, translated_mask = warp_bev_current_to_next(
        current,
        torch.tensor([[0.0, 0.1, 0.0]]),
        forward_range_m=(-0.2, 0.2),
        left_range_m=(-0.2, 0.2),
    )
    assert torch.allclose(translated[:, :, :, :-1], current[:, :, :, 1:])
    assert not translated_mask[:, :, :, -1].any()

    impulse = torch.zeros(1, 1, 5, 5)
    impulse[0, 0, 3, 2] = 1.0  # 0.1 m forward in the current frame.
    rotated, _mask = warp_bev_current_to_next(
        impulse,
        torch.tensor([[0.0, 0.0, math.pi / 2.0]]),
        forward_range_m=(-0.2, 0.2),
        left_range_m=(-0.2, 0.2),
    )
    # A current-frame point ahead is to the right in the next frame after a
    # positive (left) 90-degree base rotation.
    assert rotated[0, 0, 2, 1] == pytest.approx(1.0, abs=1e-6)


def test_variance_floor_rejects_collapse() -> None:
    collapsed = torch.zeros(2, 4, 3, 3)
    varied = torch.randn(32, 4, 3, 3)
    assert bev_variance_floor_loss(collapsed) > 0.9
    assert bev_variance_floor_loss(varied) < 0.2


def test_variance_floor_rejects_input_independent_spatial_template() -> None:
    torch.manual_seed(3)
    spatial_template = torch.randn(1, 4, 3, 3).expand(16, -1, -1, -1)
    assert bev_variance_floor_loss(spatial_template) > 0.9


def test_bev_decoder_each_metric_query_can_use_every_image_token() -> None:
    torch.manual_seed(5)
    decoder = BevDecoder(
        token_dim=6,
        bev_dim=8,
        token_side=2,
        bev_size=(4, 4),
        forward_range_m=(-0.1, 0.2),
        left_range_m=(-0.2, 0.1),
        attention_heads=2,
    )
    tokens = torch.randn(1, 4, 6, requires_grad=True)
    decoder(tokens)[0, :, -1, -1].sum().backward()

    token_gradient = tokens.grad.abs().sum(dim=-1)
    assert torch.all(token_gradient > 0)


def test_bev_decoder_legacy_default_is_exact_reference_and_rng_equivalent() -> None:
    kwargs = {
        "token_dim": 6,
        "bev_dim": 8,
        "token_side": 2,
        "bev_size": (4, 4),
        "forward_range_m": (-0.1, 0.2),
        "left_range_m": (-0.2, 0.1),
        "attention_heads": 2,
    }
    torch.manual_seed(101)
    implicit = BevDecoder(**kwargs).eval()
    torch.manual_seed(101)
    explicit = BevDecoder(
        **kwargs,
        lift_type=GLOBAL_CROSS_ATTENTION_LIFT,
    ).eval()
    tokens = torch.randn(2, 4, 6)

    assert implicit.state_dict().keys() == explicit.state_dict().keys()
    assert all(
        torch.equal(implicit.state_dict()[name], explicit.state_dict()[name])
        for name in implicit.state_dict()
    )
    assert torch.equal(implicit(tokens), _legacy_decoder_reference(implicit, tokens))
    assert torch.equal(implicit(tokens), explicit(tokens))
    assert implicit.projective_attention_bias is None
    assert implicit.projective_query_visibility is None


def test_projective_column_projection_has_correct_signs_and_fov_boundaries() -> None:
    decoder = _projective_decoder()
    bias = decoder.projective_attention_bias
    visibility = decoder.projective_query_visibility
    assert bias is not None
    assert visibility is not None

    # BEV columns are body-right, center, body-left. Positive body-left projects
    # to image-left under x-forward/y-left/z-up camera axes.
    closest_token = bias.argmax(dim=1).reshape(2, 3)
    assert closest_token[0].tolist() == [5, 4, 3]
    # The first row's +/-45-degree cells lie exactly on the 90-degree FOV edges.
    assert visibility.reshape(2, 3)[0].all()


def test_projective_column_center_projection_has_exact_gaussian_prior() -> None:
    decoder = _projective_decoder()
    assert decoder.projective_attention_bias is not None
    expected = torch.tensor(
        [
            [-1.0, -0.5, -1.0],
            [-0.5, 0.0, -0.5],
            [-1.0, -0.5, -1.0],
        ]
    )
    assert torch.allclose(
        decoder.projective_attention_bias[1].reshape(3, 3),
        expected,
        rtol=0.0,
        atol=1e-6,
    )


def test_projective_footprint_offsets_are_deterministic_and_cardinal_aligned() -> None:
    decoder = _footprint_decoder(
        projective_footprint_radius_m=0.47,
        projective_footprint_perimeter_samples=4,
    )

    expected = torch.tensor(
        ((0.0, 0.0), (0.47, 0.0), (0.0, 0.47), (-0.47, 0.0), (0.0, -0.47)),
        dtype=torch.float64,
    )
    actual = torch.tensor(
        decoder.projective_horizontal_offsets_body_m,
        dtype=torch.float64,
    )
    assert decoder.projective_footprint_radius_m == pytest.approx(0.47)
    assert decoder.projective_footprint_perimeter_samples == 4
    assert torch.allclose(actual, expected, rtol=0.0, atol=1e-15)


def test_projective_cell_square_uses_center_and_exact_output_corners() -> None:
    decoder = _cell_square_decoder()
    expected = torch.tensor(
        (
            (0.0, 0.0),
            (-0.05, -0.05),
            (-0.05, 0.05),
            (0.05, -0.05),
            (0.05, 0.05),
        ),
        dtype=torch.float64,
    )
    actual = torch.tensor(
        decoder.projective_horizontal_offsets_body_m,
        dtype=torch.float64,
    )

    assert decoder.projective_output_cell_size_m == pytest.approx(0.1)
    assert decoder.projective_footprint_radius_m is None
    assert decoder.projective_footprint_perimeter_samples is None
    assert torch.equal(actual, expected)


def test_dynamic_cell_square_level_support_matches_frozen_geometry() -> None:
    bias, visibility = _registered_dynamic_geometry(
        torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        torch.tensor([0.0]),
    )

    payload = bytes(visibility[0].to(torch.uint8).tolist())
    assert bias.shape == (1, 4096, 16)
    assert visibility.shape == (1, 4096)
    assert int(visibility.sum()) == 2062
    assert hashlib.sha256(payload).hexdigest() == (
        "4ebbafb6d4dd5fb13b96df978abfa7b81bc2f879b2ba6dec2fcda38dec54e60b"
    )
    assert torch.isfinite(bias).all()


@pytest.mark.parametrize(
    ("roll_rad", "pitch_rad"),
    ((0.17, 0.0), (0.0, -0.13), (-0.11, 0.09)),
)
def test_dynamic_cell_square_visibility_matches_stdlib_for_tilts(
    roll_rad: float,
    pitch_rad: float,
) -> None:
    quaternion = _tilt_quaternion_xyzw(
        roll_rad=roll_rad,
        pitch_rad=pitch_rad,
    )
    _bias, visibility = _registered_dynamic_geometry(
        torch.tensor([quaternion]),
        torch.tensor([0.0]),
    )
    reference = build_dynamic_cell_square_support_mask(quaternion, 0.0)
    reference_tensor = torch.tensor(reference, dtype=torch.bool).reshape(-1)

    assert torch.equal(visibility[0].cpu(), reference_tensor)


def test_dynamic_visibility_rejects_float32_near_boundary_false_positive() -> None:
    quaternion = (
        -0.2109752693,
        -0.1558566282,
        -0.3949682858,
        0.8804534062,
    )
    yaw = -0.7777720055
    _bias, visibility = _registered_dynamic_geometry(
        torch.tensor([quaternion], dtype=torch.float64),
        torch.tensor([yaw], dtype=torch.float64),
    )
    reference = torch.tensor(
        build_dynamic_cell_square_support_mask(quaternion, yaw),
        dtype=torch.bool,
    )

    assert not bool(visibility[0].reshape(64, 64)[45, 55])
    assert torch.equal(visibility[0].reshape(64, 64).cpu(), reference)


def test_dynamic_attitude_tolerances_are_checked_before_float32_cast() -> None:
    inside_norm = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0 + 0.999e-5]], dtype=torch.float64
    )
    outside_norm = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0 + 1.001e-5]], dtype=torch.float64
    )
    tokens = torch.randn(1, 9, 6)
    decoder = _dynamic_cell_square_decoder()

    decoder(
        tokens,
        base_quat_world_xyzw=inside_norm,
        stored_base_yaw_rad=torch.tensor([0.0], dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="quaternion norm"):
        decoder(
            tokens,
            base_quat_world_xyzw=outside_norm,
            stored_base_yaw_rad=torch.tensor([0.0], dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="stored base yaw disagrees"):
        decoder(
            tokens,
            base_quat_world_xyzw=torch.tensor(
                [[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64
            ),
            stored_base_yaw_rad=torch.tensor([1.001e-5], dtype=torch.float64),
        )


def test_dynamic_cell_square_decoder_batch_matches_single_frames() -> None:
    torch.manual_seed(311)
    decoder = _dynamic_cell_square_decoder().eval()
    tokens = torch.randn(2, 9, 6)
    quaternions = torch.tensor(
        [
            _tilt_quaternion_xyzw(roll_rad=0.12),
            _tilt_quaternion_xyzw(pitch_rad=-0.08),
        ]
    )
    yaws = torch.zeros(2)

    batched = decoder(
        tokens,
        base_quat_world_xyzw=quaternions,
        stored_base_yaw_rad=yaws,
    )
    singles = torch.cat(
        [
            decoder(
                tokens[index : index + 1],
                base_quat_world_xyzw=quaternions[index : index + 1],
                stored_base_yaw_rad=yaws[index : index + 1],
            )
            for index in range(2)
        ],
        dim=0,
    )

    torch.testing.assert_close(batched, singles, rtol=1e-5, atol=1e-6)


def test_dynamic_attitude_api_fails_closed_and_legacy_rejects_it() -> None:
    tokens = torch.randn(1, 9, 6)
    quaternion = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    yaw = torch.tensor([0.0])

    with pytest.raises(ValueError, match="requires base quaternion and stored yaw"):
        _dynamic_cell_square_decoder()(tokens)
    with pytest.raises(ValueError, match="requires base quaternion and stored yaw"):
        _dynamic_cell_square_decoder()(
            tokens,
            base_quat_world_xyzw=quaternion,
        )
    with pytest.raises(ValueError, match="invalid for legacy"):
        _cell_square_decoder()(
            tokens,
            base_quat_world_xyzw=quaternion,
            stored_base_yaw_rad=yaw,
        )
    with pytest.raises(ValueError, match="must be tensors"):
        _dynamic_cell_square_decoder()(
            tokens,
            base_quat_world_xyzw=[[0.0, 0.0, 0.0, 1.0]],  # type: ignore[arg-type]
            stored_base_yaw_rad=yaw,
        )
def test_projective_query_support_is_manifest_derived_hashed_and_bound() -> None:
    manifest = _physical_manifest()
    support = build_projective_query_support_contract(manifest)
    model_config = {
        "bev_lift_type": PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT,
        "projective_output_cell_size_m": 0.1,
    }
    output_contract = {
        "projective_query_support_contract_sha256": support["contract_sha256"]
    }

    assert support["uses_body_footprint"] is False
    assert support["support_point_count"] == 5
    assert support["output_cell_half_extent_m"] == pytest.approx(0.05)
    assert support["physical_aggregation_contract"]["contract_sha256"] == (
        manifest["label_semantics"]["physical_aggregation"]["contract_sha256"]
    )
    assert validate_projective_query_support_binding(
        model_config=model_config,
        projective_query_support=support,
        dataset_manifest=manifest,
        occupancy_output_contract=output_contract,
    ) == support

    dynamic_support = build_projective_query_support_contract(
        manifest,
        lift_type=DYNAMIC_PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT,
    )
    assert dynamic_support["lift_type"] == (
        DYNAMIC_PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT
    )
    assert dynamic_support["contract_sha256"] != support["contract_sha256"]
    assert validate_projective_query_support_binding(
        model_config={
            "bev_lift_type": DYNAMIC_PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT,
            "projective_output_cell_size_m": 0.1,
        },
        projective_query_support=dynamic_support,
        dataset_manifest=manifest,
        occupancy_output_contract={
            "projective_query_support_contract_sha256": dynamic_support[
                "contract_sha256"
            ]
        },
    ) == dynamic_support

    changed = dict(support)
    changed["uses_body_footprint"] = True
    with pytest.raises(ValueError, match="differs"):
        validate_projective_query_support_binding(
            model_config=model_config,
            projective_query_support=changed,
            dataset_manifest=manifest,
            occupancy_output_contract=output_contract,
        )


def test_old_lifts_reject_cell_square_support_but_allow_old_checkpoints() -> None:
    manifest = _physical_manifest()
    support = build_projective_query_support_contract(manifest)
    assert validate_projective_query_support_binding(
        model_config={"bev_lift_type": PROJECTIVE_COLUMN_ATTENTION_LIFT},
        projective_query_support=None,
        dataset_manifest=manifest,
    ) is None
    with pytest.raises(ValueError, match="invalid for this lift type"):
        validate_projective_query_support_binding(
            model_config={"bev_lift_type": PROJECTIVE_COLUMN_ATTENTION_LIFT},
            projective_query_support=support,
            dataset_manifest=manifest,
        )


def test_projective_footprint_prior_covers_cell_center_and_lateral_extent() -> None:
    center_only = _projective_decoder()
    footprint = _footprint_decoder()
    center_query = 1  # x=1 m, y=0 m.

    assert center_only.projective_attention_bias is not None
    assert footprint.projective_attention_bias is not None
    center_bias = center_only.projective_attention_bias[center_query].reshape(3, 3)
    footprint_bias = footprint.projective_attention_bias[center_query].reshape(3, 3)

    # The center support remains exact while +/-0.5 m body-left support expands
    # the soft prior into both lateral image columns.
    assert footprint_bias[1, 1] == center_bias[1, 1] == 0.0
    assert footprint_bias[1, 0] > center_bias[1, 0]
    assert footprint_bias[1, 2] > center_bias[1, 2]
    assert footprint_bias[1, 0] == pytest.approx(-0.03125, abs=1e-6)
    assert footprint_bias[1, 2] == pytest.approx(-0.03125, abs=1e-6)


def test_projective_footprint_visibility_uses_any_visible_support_point() -> None:
    shared = {
        "bev_size": (2, 2),
        "forward_range_m": (1.0, 2.0),
        "left_range_m": (1.5, 1.6),
    }
    center_only = _projective_decoder(**shared)
    footprint = _footprint_decoder(
        **shared,
        projective_footprint_radius_m=0.6,
    )

    assert center_only.projective_query_visibility is not None
    assert footprint.projective_query_visibility is not None
    assert not bool(center_only.projective_query_visibility[0])
    # The center is outside the 45-degree half-FOV, but the footprint's
    # body-right support lies at (x=1.0, y=0.9) and is visible.
    assert bool(footprint.projective_query_visibility[0])


def test_projective_footprint_keeps_finite_global_attention_floor() -> None:
    torch.manual_seed(113)
    decoder = _footprint_decoder(projective_attention_bias_floor=-7.0)
    tokens = torch.randn(1, 9, 6, requires_grad=True)

    assert decoder.projective_attention_bias is not None
    assert decoder.projective_attention_bias.shape == (6, 9)
    assert torch.isfinite(decoder.projective_attention_bias).all()
    assert float(decoder.projective_attention_bias.min()) >= -7.0
    decoder(tokens)[0, :, 0, 1].sum().backward()
    assert tokens.grad is not None
    assert torch.all(tokens.grad.abs().sum(dim=-1) > 0)


@pytest.mark.parametrize(
    ("anchor_z", "expected_token"),
    ((1.0, 1), (-1.0, 7)),
)
def test_projective_column_projection_maps_positive_z_upward(
    anchor_z: float,
    expected_token: int,
) -> None:
    decoder = _projective_decoder(
        left_range_m=(-0.1, 0.1),
        projective_vertical_anchor_z_body_m=(anchor_z,),
    )
    assert decoder.projective_attention_bias is not None
    # Query one is the center column at x=1 m. z=+/-1 lies on the top/bottom edge.
    assert int(decoder.projective_attention_bias[1].argmax()) == expected_token
    assert bool(decoder.projective_query_visibility[1])


def test_projective_column_near_boundary_is_inclusive() -> None:
    decoder = _projective_decoder(
        bev_size=(2, 3),
        forward_range_m=(0.05, 0.1),
        left_range_m=(-0.01, 0.01),
        projective_near_m=0.1,
    )
    assert decoder.projective_query_visibility is not None
    visibility = decoder.projective_query_visibility.reshape(2, 3)
    assert not visibility[0].any()
    assert visibility[1].all()


def test_projective_column_all_invisible_is_query_only() -> None:
    decoder = _projective_decoder(
        projective_camera_rpy_body_rad=(0.0, 0.0, math.pi),
    ).eval()
    first_tokens = torch.randn(1, 9, 6, requires_grad=True)
    second_tokens = torch.randn(1, 9, 6)

    first = decoder(first_tokens)
    second = decoder(second_tokens)
    queries = decoder.coordinate_query(decoder.coordinate_features)
    queries = decoder.query_norm(queries + decoder.query_bias)[None]
    expected = decoder.refine(
        queries.transpose(1, 2).reshape(1, 8, *decoder.bev_size)
    )

    assert decoder.projective_query_visibility is not None
    assert not decoder.projective_query_visibility.any()
    assert torch.equal(first, second)
    assert torch.equal(first, expected)
    first.sum().backward()
    assert first_tokens.grad is not None
    assert torch.count_nonzero(first_tokens.grad) == 0


def test_projective_column_visible_queries_retain_all_token_gradients() -> None:
    torch.manual_seed(103)
    decoder = _projective_decoder()
    tokens = torch.randn(1, 9, 6, requires_grad=True)
    decoder(tokens)[0, :, 0, 1].sum().backward()

    assert decoder.projective_attention_bias is not None
    assert torch.isfinite(decoder.projective_attention_bias).all()
    assert not decoder.projective_attention_bias.requires_grad
    assert decoder.projective_query_visibility is not None
    assert not decoder.projective_query_visibility.requires_grad
    assert tokens.grad is not None
    assert torch.all(tokens.grad.abs().sum(dim=-1) > 0)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"lift_type": "unknown"}, "lift_type"),
        (
            {
                "lift_type": GLOBAL_CROSS_ATTENTION_LIFT,
                "projective_horizontal_fov_deg": 90.0,
            },
            "require",
        ),
        (
            {
                "lift_type": GLOBAL_CROSS_ATTENTION_LIFT,
                "projective_horizontal_fov_deg": None,
                "projective_vertical_fov_deg": None,
                "projective_camera_xyz_body_m": None,
                "projective_camera_rpy_body_rad": None,
                "projective_near_m": None,
                "projective_vertical_anchor_z_body_m": None,
                "projective_attention_sigma_tokens": 2.0,
            },
            "tuning",
        ),
        ({"projective_horizontal_fov_deg": None}, "missing"),
        ({"projective_horizontal_fov_deg": 180.0}, "horizontal"),
        ({"projective_vertical_fov_deg": float("nan")}, "finite"),
        ({"projective_near_m": 0.0}, "positive"),
        ({"projective_vertical_anchor_z_body_m": (0.1, 0.0)}, "increasing"),
        ({"projective_attention_sigma_tokens": 0.0}, "positive"),
        ({"projective_attention_bias_floor": 0.0}, "negative"),
        (
            {"projective_footprint_radius_m": 0.47},
            "footprint projection parameters require",
        ),
        (
            {"lift_type": PROJECTIVE_FOOTPRINT_ATTENTION_LIFT},
            "missing",
        ),
        (
            {
                "lift_type": PROJECTIVE_FOOTPRINT_ATTENTION_LIFT,
                "projective_footprint_radius_m": 0.0,
                "projective_footprint_perimeter_samples": 4,
            },
            "radius.*positive",
        ),
        (
            {
                "lift_type": PROJECTIVE_FOOTPRINT_ATTENTION_LIFT,
                "projective_footprint_radius_m": 0.47,
                "projective_footprint_perimeter_samples": True,
            },
            "integer",
        ),
        (
            {
                "lift_type": PROJECTIVE_FOOTPRINT_ATTENTION_LIFT,
                "projective_footprint_radius_m": 0.47,
                "projective_footprint_perimeter_samples": 6,
            },
            "multiple of four",
        ),
        (
            {
                "lift_type": PROJECTIVE_FOOTPRINT_ATTENTION_LIFT,
                "projective_footprint_radius_m": 0.47,
                "projective_footprint_perimeter_samples": 68,
            },
            "between four and 64",
        ),
    ),
)
def test_projective_column_configuration_fails_closed(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _projective_decoder(**overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        (
            {"lift_type": PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT},
            "requires projective_output_cell_size_m",
        ),
        (
            {
                "lift_type": PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT,
                "projective_output_cell_size_m": 0.0,
            },
            "positive",
        ),
        (
            {
                "lift_type": PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT,
                "projective_output_cell_size_m": 0.2,
            },
            "registered 0.10 m",
        ),
        (
            {
                "lift_type": PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT,
                "projective_output_cell_size_m": 0.1,
                "projective_footprint_radius_m": 0.47,
            },
            "must not use body-footprint",
        ),
        (
            {"projective_output_cell_size_m": 0.1},
            "output-cell projection parameters require",
        ),
    ),
)
def test_projective_cell_square_configuration_fails_closed(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _projective_decoder(**overrides)


def test_projective_buffers_are_nonpersistent_and_ema_copied() -> None:
    model = _projective_model()
    online_bias = model.bev_decoder.projective_attention_bias
    target_bias = model.target_bev_decoder.projective_attention_bias
    online_visibility = model.bev_decoder.projective_query_visibility
    target_visibility = model.target_bev_decoder.projective_query_visibility

    assert online_bias is not None and target_bias is not None
    assert online_visibility is not None and target_visibility is not None
    assert torch.equal(online_bias, target_bias)
    assert torch.equal(online_visibility, target_visibility)
    assert online_bias.data_ptr() != target_bias.data_ptr()
    named_buffers = dict(model.bev_decoder.named_buffers())
    assert named_buffers["projective_attention_bias"] is online_bias
    assert named_buffers["projective_query_visibility"] is online_visibility
    assert not any(
        "projective_attention_bias" in name
        or "projective_query_visibility" in name
        for name in model.state_dict()
    )

    with torch.no_grad():
        online_bias.add_(0.25)
        online_visibility.logical_not_()
    assert not torch.equal(online_bias, target_bias)
    assert not torch.equal(online_visibility, target_visibility)
    model.update_target_encoder()
    assert torch.equal(online_bias, target_bias)
    assert torch.equal(online_visibility, target_visibility)


def test_projective_geometry_adds_no_parameters_or_rng_consumption() -> None:
    shared = {
        "token_dim": 6,
        "bev_dim": 8,
        "token_side": 3,
        "bev_size": (2, 3),
        "forward_range_m": (1.0, 2.0),
        "left_range_m": (-1.0, 1.0),
        "attention_heads": 2,
    }
    torch.manual_seed(109)
    legacy = BevDecoder(**shared)
    torch.manual_seed(109)
    projective = _projective_decoder()
    torch.manual_seed(109)
    footprint = _footprint_decoder()
    footprint_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(109)
    cell_square = _cell_square_decoder()
    cell_square_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(109)
    dynamic_cell_square = _dynamic_cell_square_decoder()
    dynamic_cell_square_rng = torch.random.get_rng_state().clone()

    assert legacy.state_dict().keys() == projective.state_dict().keys()
    assert all(
        torch.equal(legacy.state_dict()[name], projective.state_dict()[name])
        for name in legacy.state_dict()
    )
    assert sum(parameter.numel() for parameter in legacy.parameters()) == sum(
        parameter.numel() for parameter in projective.parameters()
    )
    assert projective.state_dict().keys() == footprint.state_dict().keys()
    assert all(
        torch.equal(projective.state_dict()[name], footprint.state_dict()[name])
        for name in projective.state_dict()
    )
    assert projective.state_dict().keys() == cell_square.state_dict().keys()
    assert all(
        torch.equal(projective.state_dict()[name], cell_square.state_dict()[name])
        for name in projective.state_dict()
    )
    assert torch.equal(footprint_rng, cell_square_rng)
    assert torch.equal(cell_square_rng, dynamic_cell_square_rng)
    assert cell_square.state_dict().keys() == dynamic_cell_square.state_dict().keys()
    assert all(
        torch.equal(
            cell_square.state_dict()[name],
            dynamic_cell_square.state_dict()[name],
        )
        for name in cell_square.state_dict()
    )
    assert footprint.projective_attention_bias is not None
    assert projective.projective_attention_bias is not None
    assert footprint.projective_attention_bias.shape == (
        projective.projective_attention_bias.shape
    )
    regenerated = _projective_decoder()
    regenerated.load_state_dict(projective.state_dict(), strict=True)
    assert torch.equal(
        regenerated.projective_attention_bias,
        projective.projective_attention_bias,
    )
    assert torch.equal(
        regenerated.projective_query_visibility,
        projective.projective_query_visibility,
    )


def test_projective_footprint_model_config_reaches_online_and_target_decoders() -> None:
    model = _projective_model(
        bev_lift_type=PROJECTIVE_FOOTPRINT_ATTENTION_LIFT,
        projective_footprint_radius_m=0.47,
        projective_footprint_perimeter_samples=8,
    )

    for decoder in (model.bev_decoder, model.target_bev_decoder):
        assert decoder.lift_type == PROJECTIVE_FOOTPRINT_ATTENTION_LIFT
        assert decoder.projective_footprint_radius_m == pytest.approx(0.47)
        assert decoder.projective_footprint_perimeter_samples == 8
        assert len(decoder.projective_horizontal_offsets_body_m) == 9
        assert decoder.projective_attention_bias is not None
        assert decoder.projective_attention_bias.shape == (16, 4)


def test_projective_cell_square_model_config_reaches_both_decoders() -> None:
    model = _projective_model(
        bev_lift_type=PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT,
        projective_output_cell_size_m=0.1,
    )

    for decoder in (model.bev_decoder, model.target_bev_decoder):
        assert decoder.lift_type == PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT
        assert decoder.projective_output_cell_size_m == pytest.approx(0.1)
        assert len(decoder.projective_horizontal_offsets_body_m) == 5
        assert decoder.projective_footprint_radius_m is None
        assert decoder.projective_attention_bias is not None


def test_dynamic_model_preserves_tensor_contracts_and_gradients() -> None:
    torch.manual_seed(313)
    model = _projective_model(
        bev_lift_type=DYNAMIC_PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT,
        projective_output_cell_size_m=0.1,
    )
    current = torch.randn(2, 3, 28, 28)
    nxt = torch.randn(2, 3, 28, 28)
    action = torch.eye(4)[:2]
    delta = torch.tensor([[0.05, 0.0, 0.1], [0.0, -0.05, -0.1]])
    labels = torch.randint(0, 3, (2, 4, 4))
    mask = torch.ones(2, 4, 4, dtype=torch.bool)
    current_quaternion = torch.tensor(
        [
            _tilt_quaternion_xyzw(roll_rad=0.08),
            _tilt_quaternion_xyzw(pitch_rad=-0.06),
        ]
    )
    next_quaternion = torch.tensor(
        [
            _tilt_quaternion_xyzw(roll_rad=0.11),
            _tilt_quaternion_xyzw(pitch_rad=-0.09),
        ]
    )
    yaw = torch.zeros(2)

    logits = model.occupancy_logits(current, current_quaternion, yaw)
    output = model(
        current,
        nxt,
        action,
        delta,
        commanded_delta_pose_current=delta,
        current_base_quat_world_xyzw=current_quaternion,
        current_stored_base_yaw_rad=yaw,
        next_base_quat_world_xyzw=next_quaternion,
        next_stored_base_yaw_rad=yaw,
        current_occupancy=labels,
        next_occupancy=labels,
        current_occupancy_mask=mask,
        next_occupancy_mask=mask,
    )
    output["loss"].backward()

    assert logits.shape == (2, 3, 4, 4)
    assert torch.isfinite(logits).all()
    assert output["current_bev"].shape == (2, 8, 4, 4)
    assert output["target_next_bev"].shape == (2, 8, 4, 4)
    assert model.bev_decoder.token_project.weight.grad is not None
    assert model.occupancy_head.weight.grad is not None
    assert all(
        parameter.grad is None for parameter in model.target_bev_decoder.parameters()
    )


def test_dynamic_online_and_target_bev_are_attitude_sensitive() -> None:
    torch.manual_seed(317)
    model = _projective_model(
        bev_lift_type=DYNAMIC_PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT,
        projective_output_cell_size_m=0.1,
    ).eval()
    image = torch.randn(1, 3, 28, 28)
    level = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
    tilted = torch.tensor(
        [_tilt_quaternion_xyzw(roll_rad=0.28, pitch_rad=-0.17)],
        dtype=torch.float64,
    )
    yaw = torch.zeros(1, dtype=torch.float64)

    online_level = model._encode_online(image, level, yaw)
    online_tilted = model._encode_online(image, tilted, yaw)
    target_level = model._encode_target(image, level, yaw)
    target_tilted = model._encode_target(image, tilted, yaw)

    assert not torch.equal(online_level, online_tilted)
    assert not torch.equal(target_level, target_tilted)


def test_projective_model_preserves_jepa_and_occupancy_tensor_contracts() -> None:
    torch.manual_seed(107)
    model = _projective_model()
    current = torch.randn(2, 3, 28, 28)
    nxt = torch.randn(2, 3, 28, 28)
    action = torch.eye(4)[:2]
    delta = torch.tensor([[0.05, 0.0, 0.1], [0.0, -0.05, -0.1]])
    labels = torch.randint(0, 3, (2, 4, 4))
    mask = torch.ones(2, 4, 4, dtype=torch.bool)

    output = model(
        current,
        nxt,
        action,
        delta,
        commanded_delta_pose_current=delta,
        current_occupancy=labels,
        next_occupancy=labels,
        current_occupancy_mask=mask,
        next_occupancy_mask=mask,
    )
    output["loss"].backward()

    assert output["current_bev"].shape == (2, 8, 4, 4)
    assert output["target_next_bev"].shape == (2, 8, 4, 4)
    assert output["predicted_next_bev"].shape == (2, 8, 4, 4)
    assert output["current_occupancy_logits"].shape == (2, 3, 4, 4)
    assert output["next_occupancy_logits"].shape == (2, 3, 4, 4)
    assert output["prediction_overlap_mask"].shape == (2, 1, 4, 4)
    assert output["prediction_overlap_mask"].dtype is torch.bool
    assert model.bev_decoder.token_project.weight.grad is not None
    assert model.occupancy_head.weight.grad is not None
    assert any(parameter.grad is not None for parameter in model.predictor.parameters())
    assert all(parameter.grad is None for parameter in model.target_bev_decoder.parameters())


def test_model_trains_predictive_encoder_and_occupancy_head() -> None:
    torch.manual_seed(7)
    model = _model()
    current = torch.randn(2, 3, 28, 28)
    nxt = torch.randn(2, 3, 28, 28)
    action = torch.eye(4)[:2]
    delta = torch.tensor([[0.05, 0.0, 0.1], [0.0, -0.05, -0.1]])
    labels = torch.randint(0, 3, (2, 8, 8))
    mask = torch.ones(2, 8, 8, dtype=torch.bool)

    output = model(
        current,
        nxt,
        action,
        delta,
        commanded_delta_pose_current=delta,
        current_occupancy=labels,
        next_occupancy=labels,
        current_occupancy_mask=mask,
        next_occupancy_mask=mask,
        occupancy_unknown_known_weights=torch.tensor([1.0, 2.0]),
        occupancy_free_occupied_weights=torch.tensor([1.0, 3.0]),
    )
    output["loss"].backward()

    assert output["current_occupancy_logits"].shape == (2, 3, 8, 8)
    assert output["prediction_overlap_mask"].dtype == torch.bool
    assert model.encoder.patch_embed.weight.grad is not None
    assert model.occupancy_head.weight.grad is not None
    assert any(parameter.grad is not None for parameter in model.predictor.parameters())
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())
    assert all(
        parameter.grad is None for parameter in model.target_bev_decoder.parameters()
    )
    assert not output["target_next_bev"].requires_grad


def test_occupancy_mask_is_boolean_and_single_target_is_not_halved() -> None:
    torch.manual_seed(11)
    model = _model()
    current = torch.randn(2, 3, 28, 28)
    nxt = torch.randn(2, 3, 28, 28)
    action = torch.eye(4)[:2]
    delta = torch.zeros(2, 3)
    labels = torch.randint(0, 3, (2, 8, 8))
    mask = torch.ones(2, 8, 8, dtype=torch.bool)
    output = model(
        current,
        nxt,
        action,
        delta,
        commanded_delta_pose_current=delta,
        current_occupancy=labels,
        current_occupancy_mask=mask,
    )
    expected = model._occupancy_loss(
        output["current_occupancy_logits"], labels, mask, None
    )
    assert torch.allclose(output["occupancy_loss"], expected)

    with pytest.raises(ValueError, match="boolean"):
        model(
            current,
            nxt,
            action,
            delta,
            commanded_delta_pose_current=delta,
            current_occupancy=labels,
            current_occupancy_mask=mask.float(),
        )


def test_decomposed_loss_balances_binary_classes_under_imbalance() -> None:
    labels = torch.tensor([[[0, 0, 0, 0, 0, 0, 1, 1, 1, 2]]])
    logits = torch.linspace(-1.4, 1.7, 30).reshape(1, 3, 1, 10)
    unknown_known_weights = torch.tensor([1.0 / 6.0, 1.0 / 4.0])
    free_occupied_weights = torch.tensor([1.0 / 3.0, 1.0])

    actual = EgomotionBevJepa._occupancy_loss(
        logits,
        labels,
        None,
        None,
        unknown_known_weights=unknown_known_weights,
        free_occupied_weights=free_occupied_weights,
    )

    known_logits = torch.logsumexp(logits[:, 1:], dim=1)
    unknown_known_logits = torch.stack((logits[:, 0], known_logits), dim=1)
    unknown_known_labels = (labels != 0).long()
    unknown_known_cell_loss = F.cross_entropy(
        unknown_known_logits,
        unknown_known_labels,
        reduction="none",
    )
    unknown_known_expected = 0.5 * (
        unknown_known_cell_loss[labels == 0].mean()
        + unknown_known_cell_loss[labels != 0].mean()
    )
    free_occupied_cell_loss = F.cross_entropy(
        logits[:, 1:],
        (labels - 1).clamp_min(0),
        reduction="none",
    )
    free_occupied_expected = 0.5 * (
        free_occupied_cell_loss[labels == 1].mean()
        + free_occupied_cell_loss[labels == 2].mean()
    )
    expected = 0.5 * unknown_known_expected + 0.5 * free_occupied_expected

    assert torch.allclose(actual, expected)


def test_decomposed_loss_masks_before_weight_normalization() -> None:
    labels = torch.tensor([[[0, 1, 2, 0]]])
    mask = torch.tensor([[[True, True, True, False]]])
    logits = torch.tensor(
        [
            [
                [[1.0, -0.2, 0.1, 0.0]],
                [[0.0, 1.2, -0.4, 0.0]],
                [[-1.0, 0.1, 1.4, 0.0]],
            ]
        ]
    )
    changed_masked_logits = logits.clone()
    changed_masked_logits[:, :, 0, -1] = torch.tensor([100.0, -100.0, 50.0])
    kwargs = {
        "unknown_known_weights": torch.tensor([7.0, 2.0]),
        "free_occupied_weights": torch.tensor([3.0, 11.0]),
    }

    reference = EgomotionBevJepa._occupancy_loss(
        logits,
        labels,
        mask,
        None,
        **kwargs,
    )
    changed = EgomotionBevJepa._occupancy_loss(
        changed_masked_logits,
        labels,
        mask,
        None,
        **kwargs,
    )
    empty = EgomotionBevJepa._occupancy_loss(
        logits,
        labels,
        torch.zeros_like(mask),
        None,
        **kwargs,
    )

    assert torch.allclose(reference, changed)
    assert empty == 0.0


def test_decomposed_occupancy_loss_without_known_cells_is_finite() -> None:
    logits = torch.tensor(
        [[[[0.2, -0.5]], [[0.1, 0.4]], [[-0.3, 0.7]]]],
        requires_grad=True,
    )
    labels = torch.zeros((1, 1, 2), dtype=torch.long)
    actual = EgomotionBevJepa._occupancy_loss(
        logits,
        labels,
        None,
        None,
        unknown_known_weights=torch.tensor([4.0, 1.0]),
        free_occupied_weights=torch.tensor([2.0, 9.0]),
    )
    unknown_known_logits = torch.stack(
        (logits[:, 0], torch.logsumexp(logits[:, 1:], dim=1)),
        dim=1,
    )
    expected = 0.5 * F.cross_entropy(
        unknown_known_logits,
        torch.zeros_like(labels),
    )

    assert torch.isfinite(actual)
    assert torch.allclose(actual, expected)
    actual.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_decomposed_occupancy_weight_validation() -> None:
    logits = torch.zeros(1, 3, 1, 1)
    labels = torch.zeros(1, 1, 1, dtype=torch.long)
    valid = torch.ones(2)

    with pytest.raises(ValueError, match="supplied together"):
        EgomotionBevJepa._occupancy_loss(
            logits,
            labels,
            None,
            None,
            unknown_known_weights=valid,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        EgomotionBevJepa._occupancy_loss(
            logits,
            labels,
            None,
            torch.ones(3),
            unknown_known_weights=valid,
            free_occupied_weights=valid,
        )
    with pytest.raises(ValueError, match=r"shape \(2,\)"):
        EgomotionBevJepa._occupancy_loss(
            logits,
            labels,
            None,
            None,
            unknown_known_weights=torch.ones(3),
            free_occupied_weights=valid,
        )
    with pytest.raises(ValueError, match="floating point"):
        EgomotionBevJepa._occupancy_loss(
            logits,
            labels,
            None,
            None,
            unknown_known_weights=torch.ones(2, dtype=torch.long),
            free_occupied_weights=valid,
        )
    with pytest.raises(ValueError, match="finite"):
        EgomotionBevJepa._occupancy_loss(
            logits,
            labels,
            None,
            None,
            unknown_known_weights=torch.tensor([1.0, float("nan")]),
            free_occupied_weights=valid,
        )
    with pytest.raises(ValueError, match="nonnegative"):
        EgomotionBevJepa._occupancy_loss(
            logits,
            labels,
            None,
            None,
            unknown_known_weights=torch.tensor([1.0, -1.0]),
            free_occupied_weights=valid,
        )


def test_prediction_diagnostics_use_geometric_matched_masks() -> None:
    torch.manual_seed(13)
    model = _model()
    with torch.no_grad():
        torch.nn.init.normal_(model.predictor.net[-1].weight, std=0.05)
    current = torch.randn(2, 3, 28, 28)
    nxt = torch.randn(2, 3, 28, 28)
    action = torch.eye(4)[:2]
    wrong_action = torch.roll(action, shifts=1, dims=1)
    delta = torch.tensor([[0.1, 0.0, 0.0], [0.0, -0.1, 0.1]])
    wrong_delta = torch.tensor([[-0.4, 0.0, 0.0], [0.0, 0.4, -0.2]])
    prediction_mask = torch.ones(2, 8, 8, dtype=torch.bool)
    prediction_mask[:, 0] = False

    output = model(
        current,
        nxt,
        action,
        delta,
        commanded_delta_pose_current=delta,
        next_prediction_mask=prediction_mask,
        diagnostic_wrong_action=wrong_action,
        diagnostic_wrong_action_delta_pose_current=wrong_delta,
        diagnostic_wrong_commanded_delta_pose_current=wrong_delta,
    )

    assert output["prediction_valid_mask"].dtype == torch.bool
    assert output["wrong_delta_matched_mask"].dtype == torch.bool
    assert torch.all(
        ~output["wrong_delta_matched_mask"] | output["prediction_valid_mask"]
    )
    assert output["wrong_delta_valid_cells"] < output["prediction_valid_cells"]
    assert output["wrong_action_prediction_sensitivity"] > 0
    assert output["wrong_delta_prediction_sensitivity"] > 0
    assert output["action_contrast_loss"] >= 0
    assert output["wrong_action_contrast_loss"] >= 0
    assert output["zero_action_contrast_loss"] >= 0
    assert torch.all(
        ~output["zero_action_matched_mask"] | output["prediction_valid_mask"]
    )
    for name in (
        "prediction_to_persistence_ratio",
        "wrong_action_advantage_over_target_change",
        "wrong_delta_advantage_over_target_change",
    ):
        assert torch.isfinite(output[name])


def test_promoted_prediction_does_not_consume_realized_future_odometry() -> None:
    torch.manual_seed(17)
    model = _model().eval()
    with torch.no_grad():
        torch.nn.init.normal_(model.predictor.net[-1].weight, std=0.05)
    current = torch.randn(2, 3, 28, 28)
    nxt = torch.randn(2, 3, 28, 28)
    action = torch.eye(4)[:2]
    commanded_delta = torch.tensor([[0.1, 0.0, 0.0], [0.0, 0.1, 0.1]])
    first = model(
        current,
        nxt,
        action,
        torch.zeros(2, 3),
        commanded_delta_pose_current=commanded_delta,
    )
    second = model(
        current,
        nxt,
        action,
        torch.tensor([[0.4, 0.0, 0.2], [-0.3, 0.2, -0.4]]),
        commanded_delta_pose_current=commanded_delta,
    )

    assert torch.allclose(first["predicted_next_bev"], second["predicted_next_bev"])
    assert torch.allclose(
        first["commanded_warped_current_bev"],
        second["commanded_warped_current_bev"],
    )
    assert not torch.allclose(
        first["realized_warped_current_bev"],
        second["realized_warped_current_bev"],
    )


def test_ema_target_updates_and_stays_in_eval_mode() -> None:
    model = _model()
    before = model.target_encoder.patch_embed.weight.detach().clone()
    with torch.no_grad():
        model.encoder.patch_embed.weight.add_(2.0)
    model.train()
    model.update_target_encoder()

    assert not model.target_encoder.training
    assert torch.allclose(model.target_encoder.patch_embed.weight, before + 1.0)
