from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.models.encoders import VisionEncoder
from lewm.models.geometry_anchored_deformable_bev_lift_joint_jepa_v1 import (
    GeometryAnchoredDeformableBevLiftJointJepaV1,
    GeometryAnchoredDeformableBevLiftJointJepaV1Config,
    GeometryAnchoredDeformableBevLiftV1,
    latent_energy_per_row,
)
from lewm.models import (
    geometry_anchored_global_action_indexed_rigid_bev_transport_joint_jepa_v1
    as model_api,
)
from lewm.models.geometry_anchored_global_action_indexed_rigid_bev_transport_joint_jepa_v1 import (
    GeometryAnchoredGlobalActionIndexedRigidBevTransportJointJepaV1,
    GlobalActionIndexedRigidBevTransportPredictorV1,
)


PREDICTOR_PARAMETER_NAMES = (
    "raw_twist",
    "residual_blocks.0.conv1.weight",
    "residual_blocks.0.conv1.bias",
    "residual_blocks.0.conv2.weight",
    "residual_blocks.0.conv2.bias",
    "residual_blocks.1.conv1.weight",
    "residual_blocks.1.conv1.bias",
    "residual_blocks.1.conv2.weight",
    "residual_blocks.1.conv2.bias",
    "residual_head.weight",
    "residual_head.bias",
)


@pytest.fixture(scope="module")
def n320_encoder_state() -> dict[str, torch.Tensor]:
    caller_rng = torch.random.get_rng_state().clone()
    try:
        torch.random.default_generator.manual_seed(33017)
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


def _predictor() -> GlobalActionIndexedRigidBevTransportPredictorV1:
    caller_rng = torch.random.get_rng_state().clone()
    try:
        torch.random.default_generator.manual_seed(20260712)
        return GlobalActionIndexedRigidBevTransportPredictorV1(
            GeometryAnchoredDeformableBevLiftJointJepaV1Config()
        )
    finally:
        torch.random.set_rng_state(caller_rng)


def _one_hot(indices: list[int]) -> torch.Tensor:
    return F.one_hot(torch.tensor(indices), num_classes=9).to(torch.float32)


def _all_action_predictions(
    predictor: GlobalActionIndexedRigidBevTransportPredictorV1,
    latent: torch.Tensor,
) -> torch.Tensor:
    batch = latent.shape[0]
    repeated = latent[:, None].expand(-1, 9, -1, -1, -1).reshape(
        batch * 9, 64, 64, 64
    )
    actions = torch.eye(9, dtype=torch.float32)[None].expand(
        batch, -1, -1
    ).reshape(batch * 9, 9)
    return predictor(repeated, actions).reshape(batch, 9, 64, 64, 64)


def _tensor_bytes(value: torch.Tensor) -> bytes:
    flat = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return flat.numpy().tobytes()


def _assert_state_bytes_equal(
    first: torch.nn.Module, second: torch.nn.Module
) -> None:
    first_state = first.state_dict()
    second_state = second.state_dict()
    assert first_state.keys() == second_state.keys()
    for name in first_state:
        assert first_state[name].shape == second_state[name].shape
        assert first_state[name].dtype == second_state[name].dtype
        assert _tensor_bytes(first_state[name]) == _tensor_bytes(second_state[name])


def test_predictor_inventory_initialization_and_forbidden_shortcuts() -> None:
    predictor = _predictor()
    named = tuple(name for name, _ in predictor.named_parameters())
    assert named == PREDICTOR_PARAMETER_NAMES
    assert len(tuple(predictor.parameters())) == 11
    assert sum(parameter.numel() for parameter in predictor.parameters()) == 184_667
    assert predictor.raw_twist.shape == (9, 3)
    assert torch.equal(predictor.raw_twist, torch.zeros((9, 3)))
    assert torch.equal(
        predictor.residual_head.weight,
        torch.zeros_like(predictor.residual_head.weight),
    )
    assert torch.equal(
        predictor.residual_head.bias,
        torch.zeros_like(predictor.residual_head.bias),
    )
    assert len(predictor.residual_blocks) == 2
    assert not hasattr(predictor, "action_embedding")
    assert not hasattr(predictor, "input_projection")
    assert not hasattr(predictor, "flow")
    assert not hasattr(predictor, "pose")
    assert not hasattr(predictor, "coordinate_features")


def test_predictor_bytes_follow_the_exact_single_seed_stream(
    n320_encoder_state: dict[str, torch.Tensor],
) -> None:
    config = GeometryAnchoredDeformableBevLiftJointJepaV1Config()
    caller_rng = torch.random.get_rng_state().clone()
    try:
        torch.random.default_generator.manual_seed(config.initialization_seed)
        GeometryAnchoredDeformableBevLiftV1(config)
        nn.Conv2d(config.bev_dim, config.state_classes, kernel_size=1, bias=True)
        raw_twist = torch.zeros((config.action_dim, 3), dtype=torch.float32)
        residual_convolutions = [
            nn.Conv2d(
                config.bev_dim,
                config.bev_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            )
            for _ in range(4)
        ]
        residual_head = nn.Conv2d(
            config.bev_dim,
            config.bev_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        nn.init.zeros_(residual_head.weight)
        nn.init.zeros_(residual_head.bias)
        expected = {
            "raw_twist": raw_twist,
            "residual_blocks.0.conv1.weight": residual_convolutions[0].weight,
            "residual_blocks.0.conv1.bias": residual_convolutions[0].bias,
            "residual_blocks.0.conv2.weight": residual_convolutions[1].weight,
            "residual_blocks.0.conv2.bias": residual_convolutions[1].bias,
            "residual_blocks.1.conv1.weight": residual_convolutions[2].weight,
            "residual_blocks.1.conv1.bias": residual_convolutions[2].bias,
            "residual_blocks.1.conv2.weight": residual_convolutions[3].weight,
            "residual_blocks.1.conv2.bias": residual_convolutions[3].bias,
            "residual_head.weight": residual_head.weight,
            "residual_head.bias": residual_head.bias,
        }
    finally:
        torch.random.set_rng_state(caller_rng)

    model = GeometryAnchoredGlobalActionIndexedRigidBevTransportJointJepaV1(
        n320_encoder_state
    )
    observed = model.predictor.state_dict()
    assert tuple(observed) == PREDICTOR_PARAMETER_NAMES
    assert tuple(
        name for name, _ in model.named_parameters() if name.startswith("predictor.")
    ) == tuple(f"predictor.{name}" for name in PREDICTOR_PARAMETER_NAMES)
    for name in PREDICTOR_PARAMETER_NAMES:
        assert observed[name].shape == expected[name].shape
        assert observed[name].dtype == expected[name].dtype
        assert _tensor_bytes(observed[name]) == _tensor_bytes(expected[name])


def test_new_model_preserves_rng_and_representation_bytes(
    n320_encoder_state: dict[str, torch.Tensor],
) -> None:
    torch.random.default_generator.manual_seed(713)
    caller_rng = torch.random.get_rng_state().clone()
    predecessor = GeometryAnchoredDeformableBevLiftJointJepaV1(n320_encoder_state)
    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    replacement = GeometryAnchoredGlobalActionIndexedRigidBevTransportJointJepaV1(
        n320_encoder_state
    )
    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    assert isinstance(
        replacement.predictor, GlobalActionIndexedRigidBevTransportPredictorV1
    )
    for predecessor_component, replacement_component in (
        (predecessor.encoder, replacement.encoder),
        (predecessor.bev_lift, replacement.bev_lift),
        (predecessor.semantic_head, replacement.semantic_head),
        (predecessor.target_encoder, replacement.target_encoder),
        (predecessor.target_bev_lift, replacement.target_bev_lift),
    ):
        _assert_state_bytes_equal(predecessor_component, replacement_component)
    assert _tensor_bytes(predecessor.target_hard_sync_count) == _tensor_bytes(
        replacement.target_hard_sync_count
    )
    assert _tensor_bytes(predecessor.ema_update_count) == _tensor_bytes(
        replacement.ema_update_count
    )


def test_zero_twist_symmetry_and_independent_reference_warp() -> None:
    predictor = _predictor().eval()
    latent = torch.randn(2, 64, 64, 64)
    actions = _one_hot([0, 8])
    observed = predictor(latent, actions)
    identity = torch.tensor(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), dtype=torch.float32
    )[None].expand(2, -1, -1)
    reference_grid = F.affine_grid(identity, latent.shape, align_corners=False)
    reference = F.grid_sample(
        latent,
        reference_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    assert torch.allclose(observed, reference, atol=2e-5, rtol=2e-5)

    all_actions = _all_action_predictions(predictor, latent[:1])
    assert torch.allclose(
        all_actions,
        all_actions[:, :1].expand_as(all_actions),
        atol=2e-5,
        rtol=2e-5,
    )


def test_bounded_rigid_affine_matches_independent_formula() -> None:
    predictor = _predictor()
    raw = torch.tensor(
        (
            (100.0, -100.0, 100.0),
            (-0.7, 0.4, -0.2),
        ),
        dtype=torch.float32,
    )
    with torch.no_grad():
        predictor.raw_twist[2].copy_(raw[0])
        predictor.raw_twist[7].copy_(raw[1])
    latent = torch.zeros(2, 64, 64, 64)
    actions = _one_hot([2, 7])
    observed = predictor.selected_affine(latent, actions)

    forward = 8.0 * torch.tanh(raw[:, 0])
    left = 8.0 * torch.tanh(raw[:, 1])
    theta = (math.pi / 4.0) * torch.tanh(raw[:, 2])
    cosine = torch.cos(theta)
    sine = torch.sin(theta)
    expected = torch.stack(
        (
            torch.stack((cosine, -sine, 2.0 * left / 64.0), dim=1),
            torch.stack((sine, cosine, 2.0 * forward / 64.0), dim=1),
        ),
        dim=1,
    )
    assert torch.allclose(observed, expected, atol=2e-5, rtol=2e-5)
    rotation = observed[:, :, :2]
    identity = torch.eye(2)[None].expand(2, -1, -1)
    assert torch.allclose(
        rotation.transpose(1, 2) @ rotation,
        identity,
        atol=2e-6,
        rtol=2e-6,
    )
    assert torch.allclose(
        torch.linalg.det(rotation), torch.ones(2), atol=2e-6, rtol=2e-6
    )
    assert float(forward.detach().abs().max()) <= 8.0 + 1e-6
    assert float(left.detach().abs().max()) <= 8.0 + 1e-6
    assert float(theta.detach().abs().max()) <= math.pi / 4.0 + 1e-6
    assert float(observed.detach()[:, :, 2].abs().max()) <= 0.25 + 1e-6


def test_combined_twist_transport_matches_registered_sampling_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = _predictor().eval()
    latent = torch.randn(1, 64, 64, 64)
    raw = torch.tensor((0.37, -0.44, 0.29), dtype=torch.float32)
    with torch.no_grad():
        predictor.raw_twist[6].copy_(raw)

    affine_calls: list[tuple[tuple[int, ...], bool]] = []
    sample_calls: list[tuple[str, str, bool]] = []
    original_affine_grid = F.affine_grid
    original_grid_sample = F.grid_sample

    def recorded_affine_grid(
        theta: torch.Tensor,
        size: torch.Size,
        *,
        align_corners: bool,
    ) -> torch.Tensor:
        affine_calls.append((tuple(size), align_corners))
        return original_affine_grid(theta, size, align_corners=align_corners)

    def recorded_grid_sample(
        input_value: torch.Tensor,
        grid: torch.Tensor,
        *,
        mode: str,
        padding_mode: str,
        align_corners: bool,
    ) -> torch.Tensor:
        sample_calls.append((mode, padding_mode, align_corners))
        return original_grid_sample(
            input_value,
            grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )

    monkeypatch.setattr(F, "affine_grid", recorded_affine_grid)
    monkeypatch.setattr(F, "grid_sample", recorded_grid_sample)
    observed = predictor.transport(latent, _one_hot([6]))
    assert affine_calls == [((1, 64, 64, 64), False)]
    assert sample_calls == [("bilinear", "zeros", False)]

    forward_cells = 8.0 * torch.tanh(raw[0])
    left_cells = 8.0 * torch.tanh(raw[1])
    theta = (math.pi / 4.0) * torch.tanh(raw[2])
    cosine = torch.cos(theta)
    sine = torch.sin(theta)
    reference_affine = torch.stack(
        (
            torch.stack((cosine, -sine, 2.0 * left_cells / 64.0)),
            torch.stack((sine, cosine, 2.0 * forward_cells / 64.0)),
        )
    )[None]
    reference_grid = original_affine_grid(
        reference_affine, latent.shape, align_corners=False
    )
    reference = original_grid_sample(
        latent,
        reference_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    assert torch.allclose(observed, reference, atol=2e-5, rtol=2e-5)


def test_one_row_isolation_and_action_permutation_algebra() -> None:
    predictor = _predictor().eval()
    latent = torch.randn(1, 64, 64, 64)
    baseline = _all_action_predictions(predictor, latent)
    with torch.no_grad():
        predictor.raw_twist[4].copy_(torch.tensor((0.3, -0.4, 0.2)))
    changed = _all_action_predictions(predictor, latent)
    for action_index in range(9):
        if action_index == 4:
            assert not torch.allclose(changed[:, action_index], baseline[:, action_index])
        else:
            assert torch.equal(changed[:, action_index], baseline[:, action_index])

    with torch.no_grad():
        predictor.raw_twist.copy_(
            torch.linspace(-0.8, 0.8, 27, dtype=torch.float32).reshape(9, 3)
        )
    original = _all_action_predictions(predictor, latent)
    permutation = torch.tensor((4, 0, 8, 2, 6, 1, 7, 3, 5))
    permuted_predictor = copy.deepcopy(predictor)
    with torch.no_grad():
        permuted_predictor.raw_twist.copy_(predictor.raw_twist[permutation])
    permuted = _all_action_predictions(permuted_predictor, latent)
    assert torch.allclose(
        permuted,
        original[:, permutation],
        atol=2e-5,
        rtol=2e-5,
    )

    actions = _one_hot([0, 4, 8])
    action_permuted = actions[:, permutation]
    repeated = latent.expand(3, -1, -1, -1)
    assert torch.allclose(
        permuted_predictor(repeated, action_permuted),
        predictor(repeated, actions),
        atol=2e-5,
        rtol=2e-5,
    )


def test_historical_alias_instantiates_replacement_and_preserves_action_order(
    n320_encoder_state: dict[str, torch.Tensor],
) -> None:
    assert model_api.GeometryAnchoredDeformableBevLiftJointJepaV1 is (
        model_api.GeometryAnchoredGlobalActionIndexedRigidBevTransportJointJepaV1
    )
    model = model_api.GeometryAnchoredDeformableBevLiftJointJepaV1(
        n320_encoder_state
    ).eval()
    assert type(model) is (
        model_api.GeometryAnchoredGlobalActionIndexedRigidBevTransportJointJepaV1
    )
    assert isinstance(
        model.predictor, model_api.GlobalActionIndexedRigidBevTransportPredictorV1
    )
    with torch.no_grad():
        model.predictor.raw_twist.copy_(
            torch.linspace(-0.6, 0.6, 27, dtype=torch.float32).reshape(9, 3)
        )
    latent = torch.randn(1, 64, 64, 64)
    observed = model.predict_all_actions(latent)
    expected = torch.stack(
        [model.predict(latent, _one_hot([index])) for index in range(9)], dim=1
    )
    assert observed.shape == (1, 9, 64, 64, 64)
    assert torch.allclose(observed, expected, atol=2e-5, rtol=2e-5)


def test_spatial_impulse_is_globally_transported_not_constantly_added() -> None:
    predictor = _predictor().eval()
    latent = torch.zeros(1, 64, 64, 64)
    latent[0, 0, 20, 20] = 1.0
    latent[0, 0, 40, 40] = 0.75
    with torch.no_grad():
        predictor.raw_twist[3, 1] = math.atanh(0.5)
    shifted = predictor(latent, _one_hot([3]))
    identity = predictor(latent, _one_hot([0]))
    plane = shifted[0, 0]
    assert plane[20, 16].item() == pytest.approx(1.0, abs=2e-5)
    assert plane[40, 36].item() == pytest.approx(0.75, abs=2e-5)
    assert plane.argmax().item() == 20 * 64 + 16
    delta = shifted - identity
    assert float(delta.detach().abs().sum()) > 0.0
    assert float(delta.detach().var()) > 0.0
    assert not torch.equal(
        delta,
        delta.mean(dim=(-2, -1), keepdim=True).expand_as(delta),
    )


def test_initial_gradient_paths_and_registered_corrector_reachability(
    n320_encoder_state: dict[str, torch.Tensor],
) -> None:
    model = GeometryAnchoredGlobalActionIndexedRigidBevTransportJointJepaV1(
        n320_encoder_state
    ).train()
    online = model.encode_online(torch.rand(1, 3, 112, 112))
    target = torch.randn_like(online)
    prediction = model.predict(online, _one_hot([5]))
    latent_energy_per_row(prediction, target).mean().backward()

    selected_gradient = model.predictor.raw_twist.grad
    assert selected_gradient is not None and bool(torch.isfinite(selected_gradient).all())
    assert float(selected_gradient[5].abs().sum()) > 0.0
    assert torch.equal(
        selected_gradient[torch.arange(9) != 5],
        torch.zeros_like(selected_gradient[torch.arange(9) != 5]),
    )
    for parameter in (
        model.predictor.residual_head.weight,
        model.predictor.residual_head.bias,
        model.encoder.patch_embed.weight,
        model.bev_lift.token_projection.weight,
    ):
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
        assert float(parameter.grad.abs().sum()) > 0.0
    for block in model.predictor.residual_blocks:
        for convolution in (block.conv1, block.conv2):
            for parameter in convolution.parameters():
                assert parameter.grad is not None
                assert torch.equal(parameter.grad, torch.zeros_like(parameter.grad))
    assert all(
        parameter.grad is None
        for module in model.target_modules()
        for parameter in module.parameters()
    )

    reachable = copy.deepcopy(model.predictor)
    reachable.zero_grad(set_to_none=True)
    with torch.no_grad():
        reachable.residual_head.weight.zero_()
        reachable.residual_head.bias.zero_()
        channel_index = torch.arange(64)
        reachable.residual_head.weight[channel_index, channel_index, 1, 1] = 1.0 / 64.0
    source = torch.randn(1, 64, 64, 64, requires_grad=True)
    reached = reachable(source, _one_hot([5]))
    reached.square().mean().backward()
    for block in reachable.residual_blocks:
        for convolution in (block.conv1, block.conv2):
            for parameter in convolution.parameters():
                assert parameter.grad is not None
                assert bool(torch.isfinite(parameter.grad).all())
                assert float(parameter.grad.abs().sum()) > 0.0


def test_all_nine_action_objective_reaches_every_twist_row() -> None:
    predictor = _predictor()
    latent = torch.randn(1, 64, 64, 64)
    target = torch.randn(1, 64, 64, 64)
    predictions = _all_action_predictions(predictor, latent)
    targets = target[:, None].expand_as(predictions)
    energies = latent_energy_per_row(predictions, targets)
    scale = energies.detach().mean().clamp_min(1e-6)
    loss = F.cross_entropy(-energies / scale, torch.tensor((6,))) / math.log(9.0)
    loss.backward()
    gradient = predictor.raw_twist.grad
    assert gradient is not None and bool(torch.isfinite(gradient).all())
    assert bool((gradient.abs().sum(dim=1) > 0.0).all())


@pytest.mark.parametrize(
    ("latent", "action", "error", "message"),
    (
        (torch.zeros(0, 64, 64, 64), torch.zeros(0, 9), ValueError, "at least one"),
        (torch.zeros(1, 63, 64, 64), _one_hot([0]), ValueError, "shape"),
        (torch.zeros(1, 64, 64, 64, dtype=torch.float64), torch.eye(9, dtype=torch.float64)[:1], TypeError, "float32"),
        (torch.full((1, 64, 64, 64), float("nan")), _one_hot([0]), FloatingPointError, "nonfinite"),
        (torch.zeros(1, 64, 64, 64), torch.zeros(1, 9), ValueError, "exactly one"),
        (torch.zeros(1, 64, 64, 64), torch.full((1, 9), 1.0 / 9.0), ValueError, "zeros and ones"),
    ),
)
def test_input_validation(
    latent: torch.Tensor,
    action: torch.Tensor,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        _predictor()(latent, action)


def test_nonfinite_twist_is_rejected() -> None:
    predictor = _predictor()
    with torch.no_grad():
        predictor.raw_twist[0, 0] = float("inf")
    with pytest.raises(FloatingPointError, match="raw_twist"):
        predictor(torch.zeros(1, 64, 64, 64), _one_hot([0]))
