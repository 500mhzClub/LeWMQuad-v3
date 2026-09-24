from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import lewm.models.patch_whitened_action_residual_jepa as mechanism
from lewm.models.patch_whitened_action_residual_jepa import (
    ACTION_DIM,
    ACTION_GATE_BIAS,
    ACTION_GATE_INITIALIZATION_SEED,
    ACTION_GATE_WEIGHT_STD,
    DENSE_PAIRWISE_DISPLACEMENT_BOUND,
    DENSE_PAIRWISE_HEAD_CHANNELS,
    DENSE_PAIRWISE_INVERSE_INITIALIZATION_SEED,
    DENSE_PAIRWISE_LAYER_NORM_EPS,
    FLOW_GRID_SCALE,
    HOLD_ACTION_INDEX,
    LATENT_DIM,
    MAXIMUM_FLOW_CELL_DISPLACEMENT,
    RESIDUAL_ALPHA,
    TOKEN_COUNT,
    TOKEN_SIDE,
    ActionConditionedNextTargetRetrievalTerms,
    ActionIndexedLosses,
    ActionIndexedPredictions,
    ActionConditionedLatentFlow,
    DensePairwiseSpatialCostVolumeInverseHead,
    DensePairwiseSpatialCostVolumeInverseTerms,
    action_conditioned_next_target_retrieval_terms,
    action_independent_trunk,
    action_indexed_energy_nll,
    bounded_flow_cells,
    dense_pairwise_spatial_cost_volume_inverse_terms,
    flow_residual_reconstruct,
    initialize_action_gate_rows,
    normalized_token_l2_energy,
    patch_whitening_terms,
    predict_action_conditioned_flow_warps,
    relative_action_embeddings,
    requested_action_indices,
    warp_ema_current_latents,
)
from lewm.models.encoders import VisionEncoder
from lewm.models.phase2d_spatial_lewm import LinearTokenProjector
from lewm.models.predictor import ActionEmbedder


class _GateBlock(nn.Module):
    def __init__(self, output_rows: int = 6 * LATENT_DIM):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(LATENT_DIM, output_rows),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)


class _GatePredictor(nn.Module):
    def __init__(self, blocks: int = 2, *, output_rows: int = 6 * LATENT_DIM):
        super().__init__()
        self.blocks = nn.ModuleList([
            _GateBlock(output_rows) for _ in range(blocks)
        ])


def _one_hot(indices: list[int]) -> torch.Tensor:
    return F.one_hot(
        torch.tensor(indices), num_classes=ACTION_DIM
    ).to(torch.float32)


def _non_gate_mask() -> torch.Tensor:
    mask = torch.ones(6 * LATENT_DIM, dtype=torch.bool)
    mask[2 * LATENT_DIM : 3 * LATENT_DIM] = False
    mask[5 * LATENT_DIM : 6 * LATENT_DIM] = False
    return mask


def test_gate_initialization_matches_isolated_cpu_draw_order() -> None:
    torch.manual_seed(911)
    predictor = _GatePredictor()
    global_before = torch.random.get_rng_state().clone()

    receipt = initialize_action_gate_rows(predictor)

    assert torch.equal(torch.random.get_rng_state(), global_before)
    assert receipt == {
        "seed": ACTION_GATE_INITIALIZATION_SEED,
        "block_count": 2,
        "latent_dim": LATENT_DIM,
        "attention_gate_rows": [2 * LATENT_DIM, 3 * LATENT_DIM],
        "mlp_gate_rows": [5 * LATENT_DIM, 6 * LATENT_DIM],
        "weight_std": ACTION_GATE_WEIGHT_STD,
        "bias": ACTION_GATE_BIAS,
        "changed_weight_scalar_count": 4 * LATENT_DIM * LATENT_DIM,
        "changed_bias_scalar_count": 4 * LATENT_DIM,
    }

    generator = torch.Generator(device="cpu")
    generator.manual_seed(ACTION_GATE_INITIALIZATION_SEED)
    non_gate = _non_gate_mask()
    expected_gate_weights: list[torch.Tensor] = []
    for block in predictor.blocks:
        linear = block.adaLN_modulation[-1]
        expected_attention = torch.randn(
            (LATENT_DIM, LATENT_DIM),
            generator=generator,
            dtype=torch.float32,
        ) * ACTION_GATE_WEIGHT_STD
        expected_mlp = torch.randn(
            (LATENT_DIM, LATENT_DIM),
            generator=generator,
            dtype=torch.float32,
        ) * ACTION_GATE_WEIGHT_STD
        expected_gate_weights.append(expected_attention)
        assert torch.equal(
            linear.weight[2 * LATENT_DIM : 3 * LATENT_DIM],
            expected_attention,
        )
        assert torch.equal(
            linear.weight[5 * LATENT_DIM : 6 * LATENT_DIM],
            expected_mlp,
        )
        assert torch.count_nonzero(linear.weight[non_gate]).item() == 0
        assert torch.count_nonzero(linear.bias[non_gate]).item() == 0
        assert torch.all(
            linear.bias[2 * LATENT_DIM : 3 * LATENT_DIM]
            == ACTION_GATE_BIAS
        )
        assert torch.all(
            linear.bias[5 * LATENT_DIM : 6 * LATENT_DIM]
            == ACTION_GATE_BIAS
        )
    assert not torch.equal(expected_gate_weights[0], expected_gate_weights[1])


def test_gate_initialization_is_repeatable_but_not_reentrant() -> None:
    left = _GatePredictor()
    right = _GatePredictor()
    initialize_action_gate_rows(left)
    initialize_action_gate_rows(right)
    for left_parameter, right_parameter in zip(
        left.parameters(), right.parameters(), strict=True
    ):
        assert torch.equal(left_parameter, right_parameter)

    with pytest.raises(ValueError, match="not all-zero"):
        initialize_action_gate_rows(left)
    with pytest.raises(ValueError, match="weight must have shape"):
        initialize_action_gate_rows(
            _GatePredictor(output_rows=6 * LATENT_DIM - 1)
        )


def test_requested_actions_are_exact_uniform_vocabulary_indices() -> None:
    requested = _one_hot([0, 6, ACTION_DIM - 1])
    assert torch.equal(
        requested_action_indices(requested),
        torch.tensor([0, 6, 8]),
    )
    with pytest.raises(ValueError, match="exact one-hot"):
        requested_action_indices(torch.zeros(2, ACTION_DIM))
    with pytest.raises(TypeError, match="floating point"):
        requested_action_indices(
            F.one_hot(torch.tensor([0]), num_classes=ACTION_DIM)
        )


def test_flow_wrapper_is_zero_bias_free_rng_free_and_has_exact_grid() -> None:
    shared = LinearTokenProjector(LATENT_DIM)
    torch.manual_seed(812)
    rng_before = torch.random.get_rng_state().clone()

    wrapper = ActionConditionedLatentFlow(shared)

    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert wrapper.shared_projector is shared
    assert tuple(wrapper.flow_weight.shape) == (2, LATENT_DIM)
    assert wrapper.flow_weight.numel() == 384
    assert torch.count_nonzero(wrapper.flow_weight).item() == 0
    assert tuple(wrapper.identity_grid_xy.shape) == (1, 16, 16, 2)
    assert "identity_grid_xy" not in wrapper.state_dict()
    assert [name for name, _ in wrapper.named_parameters()] == [
        "flow_weight",
        "shared_projector.linear.weight",
        "shared_projector.linear.bias",
    ]
    assert not hasattr(wrapper, "bias")
    assert not hasattr(wrapper, "action_embed")


def _spatial_latents(batch: int) -> torch.Tensor:
    coordinates = torch.linspace(-1.0, 1.0, TOKEN_SIDE)
    rows, columns = torch.meshgrid(
        coordinates,
        coordinates,
        indexing="ij",
    )
    base = torch.linspace(0.25, 1.25, LATENT_DIM)
    horizontal = torch.linspace(-0.7, 0.9, LATENT_DIM)
    vertical = torch.cos(torch.linspace(0.0, 2.0 * math.pi, LATENT_DIM))
    tokens = (
        base[None]
        + columns.reshape(TOKEN_COUNT, 1) * horizontal[None]
        + rows.reshape(TOKEN_COUNT, 1) * vertical[None]
    )
    tokens = F.normalize(tokens, dim=-1)
    return torch.stack(
        [torch.roll(tokens, shifts=index, dims=0) for index in range(batch)]
    )


def test_flow_grid_axes_bound_border_and_post_warp_residual() -> None:
    assert RESIDUAL_ALPHA == 0.1 / math.sqrt(192)
    assert TOKEN_SIDE == 16
    assert TOKEN_COUNT == 256
    assert MAXIMUM_FLOW_CELL_DISPLACEMENT == 1.0
    assert FLOW_GRID_SCALE == 2.0 / 15.0
    wrapper = ActionConditionedLatentFlow(
        LinearTokenProjector(LATENT_DIM)
    )
    values = (
        100.0 * torch.arange(TOKEN_SIDE)[:, None]
        + torch.arange(TOKEN_SIDE)[None, :]
    ).reshape(1, TOKEN_COUNT, 1).expand(-1, -1, LATENT_DIM)
    ema_current = values.to(torch.float32).requires_grad_()
    flows = torch.zeros(1, ACTION_DIM, TOKEN_COUNT, 2)
    flows[:, 0, :, 0] = 1.0
    flows[:, 1, :, 1] = 1.0
    warped = warp_ema_current_latents(wrapper, ema_current, flows)

    source_map = values.reshape(1, TOKEN_SIDE, TOKEN_SIDE, LATENT_DIM)
    expected_right = torch.cat(
        (source_map[:, :, 1:], source_map[:, :, -1:]),
        dim=2,
    ).reshape(1, TOKEN_COUNT, LATENT_DIM)
    expected_down = torch.cat(
        (source_map[:, 1:], source_map[:, -1:]),
        dim=1,
    ).reshape(1, TOKEN_COUNT, LATENT_DIM)
    assert torch.allclose(warped[:, 0], expected_right, atol=1e-5)
    assert torch.allclose(warped[:, 1], expected_down, atol=1e-5)
    assert torch.allclose(warped[:, HOLD_ACTION_INDEX], values, atol=1e-5)
    assert torch.allclose(
        warped[:, 0].reshape(1, TOKEN_SIDE, TOKEN_SIDE, LATENT_DIM)[
            :, :, -1
        ],
        source_map[:, :, -1],
        atol=1e-5,
    )

    raw = torch.full((1, ACTION_DIM, TOKEN_COUNT, 2), 100.0)
    bounded = bounded_flow_cells(raw)
    assert bounded.max().item() <= 1.0
    assert bounded.min().item() >= -1.0
    with pytest.raises(FloatingPointError, match="out of bounds"):
        warp_ema_current_latents(wrapper, values, flows * 1.01)

    residual = torch.randn(
        1,
        ACTION_DIM,
        TOKEN_COUNT,
        LATENT_DIM,
        requires_grad=True,
    )
    result = flow_residual_reconstruct(warped, residual)
    expected = F.normalize(
        warped + RESIDUAL_ALPHA * residual,
        dim=-1,
        eps=1e-8,
    )
    assert torch.allclose(result, expected)
    assert torch.allclose(
        result.norm(dim=-1),
        torch.ones(1, ACTION_DIM, TOKEN_COUNT),
        atol=1e-6,
    )
    (warped[:, 0, :, 0].sum() + result[..., 0].sum()).backward()
    assert ema_current.grad is None
    assert residual.grad is not None
    assert torch.isfinite(residual.grad).all()


class _ForbiddenActionEmbed(nn.Module):
    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        raise AssertionError("shared trunk must bypass action_embed")


class _FlowBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(LATENT_DIM, LATENT_DIM, bias=False)

    def forward(
        self,
        state: torch.Tensor,
        condition: torch.Tensor,
        *,
        causal: bool,
    ) -> torch.Tensor:
        assert causal is False
        assert torch.count_nonzero(condition).item() == 0
        return state + 0.05 * self.projection(state)


class _FlowPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.latent_dim = LATENT_DIM
        self.num_spatial_tokens = TOKEN_COUNT
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, TOKEN_COUNT, LATENT_DIM)
        )
        self.input_drop = nn.Identity()
        self.blocks = nn.ModuleList([_FlowBlock()])
        self.norm = nn.LayerNorm(LATENT_DIM)
        self.action_embed = ActionEmbedder(
            input_dim=ACTION_DIM,
            smoothed_dim=10,
            emb_dim=LATENT_DIM,
        )


def _spatial_fixture() -> tuple[
    _FlowPredictor,
    ActionConditionedLatentFlow,
]:
    torch.manual_seed(101)
    predictor = _FlowPredictor()
    projector = ActionConditionedLatentFlow(
        LinearTokenProjector(LATENT_DIM)
    )
    return predictor, projector


def test_action_independent_trunk_bypasses_embed_and_uses_exact_zero() -> None:
    predictor, _ = _spatial_fixture()
    captured: list[tuple[torch.Tensor, bool]] = []

    def capture_condition(
        _module: nn.Module,
        args: tuple[torch.Tensor, torch.Tensor],
        kwargs: dict[str, object],
    ) -> None:
        captured.append((args[1].detach().clone(), bool(kwargs["causal"])))

    handle = predictor.blocks[0].register_forward_pre_hook(
        capture_condition,
        with_kwargs=True,
    )
    predictor.action_embed = _ForbiddenActionEmbed()
    state = torch.randn(2, TOKEN_COUNT, LATENT_DIM)
    hidden = action_independent_trunk(predictor, state)
    handle.remove()

    assert hidden.shape == state.shape
    assert len(captured) == 1
    assert torch.count_nonzero(captured[0][0]).item() == 0
    assert captured[0][1] is False


def test_action_embeddings_are_distinct_with_exact_hold_reference() -> None:
    predictor, _ = _spatial_fixture()
    relative = relative_action_embeddings(
        predictor,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert relative.shape == (ACTION_DIM, LATENT_DIM)
    assert torch.count_nonzero(relative[HOLD_ACTION_INDEX]).item() == 0
    for action in range(ACTION_DIM):
        if action != HOLD_ACTION_INDEX:
            assert torch.count_nonzero(relative[action]).item() > 0
    for left in range(ACTION_DIM):
        for right in range(left + 1, ACTION_DIM):
            assert not torch.equal(relative[left], relative[right])


def test_all_flow_predictions_start_bitwise_equal_and_gather_exactly() -> None:
    predictor, projector = _spatial_fixture()
    state = torch.randn(2, TOKEN_COUNT, LATENT_DIM)
    skip = _spatial_latents(2)
    requested = _one_hot([0, 6])

    predictions = predict_action_conditioned_flow_warps(
        predictor,
        projector,
        state,
        requested,
        skip,
    )

    assert predictions.all_predictions.shape == (
        2, ACTION_DIM, TOKEN_COUNT, LATENT_DIM
    )
    assert predictions.all_flows_cell.shape == (
        2, ACTION_DIM, TOKEN_COUNT, 2
    )
    assert torch.count_nonzero(predictions.all_flows_cell).item() == 0
    assert predictions.executed.shape == (2, TOKEN_COUNT, LATENT_DIM)
    assert predictions.controls.shape == (
        2, ACTION_DIM - 1, TOKEN_COUNT, LATENT_DIM
    )
    assert predictions.control_indices.shape == (2, ACTION_DIM - 1)
    comparison_count = 0
    for left in range(ACTION_DIM):
        for right in range(left + 1, ACTION_DIM):
            comparison_count += 1
            assert torch.equal(
                predictions.all_predictions[:, left],
                predictions.all_predictions[:, right],
            )
    assert comparison_count == 36
    assert torch.equal(
        predictions.executed,
        predictions.all_predictions[
            torch.arange(2), torch.tensor([0, 6])
        ],
    )
    assert torch.equal(
        predictions.controls,
        predictions.all_predictions.gather(
            1,
            predictions.control_indices[:, :, None, None].expand(
                -1, -1, TOKEN_COUNT, LATENT_DIM
            ),
        ),
    )


def test_shared_flow_separates_nonhold_actions_and_hold_stays_zero() -> None:
    predictor, projector = _spatial_fixture()
    state = torch.randn(1, TOKEN_COUNT, LATENT_DIM)
    skip = _spatial_latents(1)
    requested = _one_hot([2])
    baseline = predict_action_conditioned_flow_warps(
        predictor, projector, state, requested, skip
    )

    with torch.no_grad():
        projector.flow_weight.copy_(
            torch.stack((
                torch.linspace(-0.02, 0.02, LATENT_DIM),
                torch.linspace(0.015, -0.015, LATENT_DIM),
            ))
        )
    changed = predict_action_conditioned_flow_warps(
        predictor, projector, state, requested, skip
    )

    assert torch.count_nonzero(
        changed.all_flows_cell[:, HOLD_ACTION_INDEX]
    ).item() == 0
    assert torch.count_nonzero(
        changed.all_flows_cell[:, 0]
    ).item() > 0
    assert not torch.equal(
        changed.all_flows_cell[:, 0],
        changed.all_flows_cell[:, 1],
    )
    assert torch.equal(
        changed.all_predictions[:, HOLD_ACTION_INDEX],
        baseline.all_predictions[:, HOLD_ACTION_INDEX],
    )


def test_zero_flow_projection_opens_then_state_and_action_become_live() -> None:
    predictor, projector = _spatial_fixture()
    state = torch.randn(
        1,
        TOKEN_COUNT,
        LATENT_DIM,
        requires_grad=True,
    )
    skip = _spatial_latents(1)
    requested = _one_hot([0])
    predictions = predict_action_conditioned_flow_warps(
        predictor,
        projector,
        state,
        requested,
        skip,
    )
    target_flow = torch.zeros_like(predictions.all_flows_cell)
    target_flow[:, 0, :, 0] = 0.5
    desired = warp_ema_current_latents(
        projector,
        skip,
        target_flow,
    )[:, 0].detach()
    warped = warp_ema_current_latents(
        projector,
        skip,
        predictions.all_flows_cell,
    )[:, 0]
    action_parameters = tuple(predictor.action_embed.parameters())
    initial_loss = (warped - desired).square().mean()
    initial_gradients = torch.autograd.grad(
        initial_loss,
        (projector.flow_weight, state, *action_parameters),
        allow_unused=True,
    )
    initial_flow_gradient = initial_gradients[0]
    assert initial_flow_gradient is not None
    assert torch.isfinite(initial_flow_gradient).all()
    assert torch.count_nonzero(initial_flow_gradient).item() > 0
    assert initial_gradients[1] is None or (
        torch.count_nonzero(initial_gradients[1]).item() == 0
    )
    assert all(
        gradient is None or torch.count_nonzero(gradient).item() == 0
        for gradient in initial_gradients[2:]
    )

    with torch.no_grad():
        projector.flow_weight.copy_(
            -0.01
            * initial_flow_gradient
            / initial_flow_gradient.norm().clamp_min(1e-8)
        )
    post_state = state.detach().clone().requires_grad_()
    post_predictions = predict_action_conditioned_flow_warps(
        predictor,
        projector,
        post_state,
        requested,
        skip,
    )
    post_warped = warp_ema_current_latents(
        projector,
        skip,
        post_predictions.all_flows_cell,
    )[:, 0]
    post_loss = (post_warped - desired).square().mean()
    post_gradients = torch.autograd.grad(
        post_loss,
        (post_state, *action_parameters),
        allow_unused=True,
    )
    assert post_gradients[0] is not None
    assert torch.count_nonzero(post_gradients[0]).item() > 0
    assert any(
        gradient is not None
        and torch.count_nonzero(gradient).item() > 0
        for gradient in post_gradients[1:]
    )


def test_executed_path_is_live_while_wrong_shared_state_is_detached() -> None:
    predictor, projector = _spatial_fixture()
    with torch.no_grad():
        projector.flow_weight.copy_(
            torch.stack((
                torch.linspace(-0.03, 0.03, LATENT_DIM),
                torch.linspace(0.02, -0.02, LATENT_DIM),
            ))
        )
    state = torch.randn(
        1,
        TOKEN_COUNT,
        LATENT_DIM,
        requires_grad=True,
    )
    skip = _spatial_latents(1).requires_grad_()
    predictions = predict_action_conditioned_flow_warps(
        predictor, projector, state, _one_hot([2]), skip
    )
    shared_weight = projector.shared_projector.linear.weight
    action_parameters = tuple(predictor.action_embed.parameters())

    wrong_gradients = torch.autograd.grad(
        predictions.all_predictions[:, 4, :, 0].sum(),
        (
            state,
            shared_weight,
            projector.flow_weight,
            *action_parameters,
            skip,
        ),
        retain_graph=True,
        allow_unused=True,
    )
    assert wrong_gradients[0] is None or (
        torch.count_nonzero(wrong_gradients[0]).item() == 0
    )
    assert wrong_gradients[1] is None or (
        torch.count_nonzero(wrong_gradients[1]).item() == 0
    )
    assert wrong_gradients[2] is not None
    assert torch.count_nonzero(wrong_gradients[2]).item() > 0
    assert any(
        gradient is not None
        and torch.count_nonzero(gradient).item() > 0
        for gradient in wrong_gradients[3:-1]
    )
    assert wrong_gradients[-1] is None

    executed_gradients = torch.autograd.grad(
        predictions.executed[..., 0].sum(),
        (
            state,
            shared_weight,
            projector.flow_weight,
            *action_parameters,
            skip,
        ),
        allow_unused=True,
    )
    assert executed_gradients[0] is not None
    assert torch.count_nonzero(executed_gradients[0]).item() > 0
    assert executed_gradients[1] is not None
    assert torch.count_nonzero(executed_gradients[1]).item() > 0
    assert executed_gradients[2] is not None
    assert torch.count_nonzero(executed_gradients[2]).item() > 0
    assert any(
        gradient is not None
        and torch.count_nonzero(gradient).item() > 0
        for gradient in executed_gradients[3:-1]
    )
    assert executed_gradients[-1] is None


def _manual_whitening(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, patches, dim = tokens.shape
    q = (tokens - tokens.mean(dim=0, keepdim=True)).reshape(
        batch * patches, dim
    )
    a = q / torch.sqrt(q.square().mean().detach() + 1e-4)
    covariance = a.T @ a / (batch * patches - 1)
    diagonal = covariance.diagonal()
    variance = F.relu(1 - torch.sqrt(diagonal + 1e-4)).mean()
    off_diagonal = covariance.square().masked_select(
        ~torch.eye(dim, dtype=torch.bool)
    ).sum() / dim
    return variance, off_diagonal


def test_whitening_matches_exact_registered_formula() -> None:
    tokens = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 5.0]],
            [[2.0, 0.0], [7.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    expected_variance, expected_covariance = _manual_whitening(tokens)
    terms = patch_whitening_terms(tokens)

    assert torch.equal(terms.variance, expected_variance)
    assert torch.equal(terms.covariance, expected_covariance)

    constant = patch_whitening_terms(torch.ones(2, 3, 4))
    assert constant.variance.item() == pytest.approx(0.99)
    assert constant.covariance.item() == pytest.approx(0.0)


def test_whitening_detects_correlated_rank_and_ignores_patch_offsets() -> None:
    batch, patches, dim = 9, 3, 8
    basis_seed = torch.zeros(batch, batch)
    basis_seed[:, 0] = 1.0
    basis_seed[:, 1:] = torch.eye(batch)[:, : batch - 1]
    orthogonal, _ = torch.linalg.qr(basis_seed)
    isotropic = orthogonal[:, None, 1:].expand(-1, patches, -1).contiguous()
    scalar = torch.linspace(-1.0, 1.0, batch)
    rank_one = scalar[:, None, None].expand(batch, patches, dim).contiguous()

    isotropic_terms = patch_whitening_terms(isotropic.to(torch.float32))
    rank_one_terms = patch_whitening_terms(rank_one.to(torch.float32))

    assert isotropic_terms.covariance.item() < 1e-5
    assert rank_one_terms.variance.item() < 1e-4
    assert rank_one_terms.covariance.item() > 5.0

    torch.manual_seed(19)
    tokens = torch.randn(4, 5, 6)
    offsets = torch.randn(1, 5, 6) * 20
    baseline = patch_whitening_terms(tokens)
    shifted = patch_whitening_terms(tokens + offsets)
    permuted = patch_whitening_terms(tokens[torch.tensor([2, 0, 3, 1])])
    assert torch.allclose(baseline.variance, shifted.variance, atol=1e-6)
    assert torch.allclose(baseline.covariance, shifted.covariance, atol=1e-6)
    assert torch.allclose(baseline.variance, permuted.variance, atol=1e-7)
    assert torch.allclose(baseline.covariance, permuted.covariance, atol=1e-6)


def test_whitening_is_float32_differentiable_and_validates_shapes() -> None:
    tokens = torch.randn(4, 5, 6, requires_grad=True)
    terms = patch_whitening_terms(tokens)
    (0.5 * terms.variance + 0.02 * terms.covariance).backward()
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()
    assert torch.count_nonzero(tokens.grad).item() > 0

    with pytest.raises(TypeError, match="float32"):
        patch_whitening_terms(torch.randn(2, 3, 4, dtype=torch.float64))
    with pytest.raises(ValueError, match="B >= 2"):
        patch_whitening_terms(torch.randn(1, 3, 4))
    with pytest.raises(ValueError, match=r"\(B, N, D\)"):
        patch_whitening_terms(torch.randn(2, 3))


def _predictions_from_energies(
    executed_indices: list[int],
    energies: torch.Tensor,
) -> ActionIndexedPredictions:
    batch = len(executed_indices)
    indices = torch.tensor(executed_indices, dtype=torch.long)
    all_predictions = energies.sqrt()[:, :, None, None].expand(
        batch, ACTION_DIM, 1, LATENT_DIM
    )
    candidates = torch.arange(ACTION_DIM).unsqueeze(0).expand(batch, -1)
    control_indices = candidates[
        candidates != indices[:, None]
    ].reshape(batch, ACTION_DIM - 1)
    return ActionIndexedPredictions(
        executed_indices=indices,
        all_predictions=all_predictions,
        all_flows_cell=torch.zeros(
            batch,
            ACTION_DIM,
            TOKEN_COUNT,
            2,
        ),
        executed=all_predictions[
            torch.arange(batch), indices
        ],
        control_indices=control_indices,
        controls=all_predictions.gather(
            1,
            control_indices[:, :, None, None].expand(
                -1, -1, 1, LATENT_DIM
            ),
        ),
    )


def test_energy_nll_matches_exact_manual_formula() -> None:
    energies = torch.tensor(
        [
            [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
            [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10],
        ],
        dtype=torch.float32,
    )
    predictions = _predictions_from_energies([0, 8], energies)
    losses = action_indexed_energy_nll(
        predictions,
        torch.zeros(2, 1, LATENT_DIM),
    )
    expected_scale = energies.mean(dim=1).detach().clamp_min(1e-8)
    expected_logits = -energies / expected_scale[:, None]
    expected_per_row = expected_scale * F.cross_entropy(
        expected_logits,
        torch.tensor([0, 8]),
        reduction="none",
    )
    expected_jepa = torch.tensor([energies[0, 0], energies[1, 8]]).mean()

    assert isinstance(losses, ActionIndexedLosses)
    assert torch.allclose(losses.energies, energies)
    assert torch.allclose(losses.row_scale, expected_scale)
    assert not losses.row_scale.requires_grad
    assert torch.allclose(losses.logits, expected_logits)
    assert torch.allclose(losses.identification_per_row, expected_per_row)
    assert torch.allclose(losses.identification, expected_per_row.mean())
    assert torch.allclose(losses.jepa, expected_jepa)
    assert torch.allclose(
        losses.total,
        expected_jepa + expected_per_row.mean(),
    )


def test_equal_energy_nll_attracts_executed_and_repels_every_wrong() -> None:
    equal = torch.full(
        (1, ACTION_DIM),
        0.5,
        dtype=torch.float32,
        requires_grad=True,
    )
    predictions = _predictions_from_energies([3], equal)
    target = torch.zeros(
        1, 1, LATENT_DIM, requires_grad=True
    )
    losses = action_indexed_energy_nll(predictions, target)
    energy_gradient = torch.autograd.grad(
        losses.identification,
        losses.energies,
        retain_graph=True,
    )[0]

    assert energy_gradient[0, 3].item() == pytest.approx(8.0 / 9.0)
    wrong = torch.cat(
        [energy_gradient[0, :3], energy_gradient[0, 4:]]
    )
    assert wrong.tolist() == pytest.approx([-1.0 / 9.0] * 8)
    prediction_and_target_gradients = torch.autograd.grad(
        losses.total,
        (predictions.all_predictions, target),
        allow_unused=True,
    )
    assert prediction_and_target_gradients[0] is not None
    assert torch.isfinite(prediction_and_target_gradients[0]).all()
    assert prediction_and_target_gradients[1] is None


def _retrieval_predictions(
    all_predictions: torch.Tensor,
    executed_indices: list[int],
) -> ActionIndexedPredictions:
    batch = all_predictions.shape[0]
    indices = torch.tensor(
        executed_indices,
        dtype=torch.long,
        device=all_predictions.device,
    )
    rows = torch.arange(batch, device=all_predictions.device)
    candidates = torch.arange(
        ACTION_DIM,
        device=all_predictions.device,
    ).unsqueeze(0).expand(batch, -1)
    control_indices = candidates[
        candidates != indices[:, None]
    ].reshape(batch, ACTION_DIM - 1)
    controls = all_predictions.gather(
        1,
        control_indices[:, :, None, None].expand(
            -1,
            -1,
            TOKEN_COUNT,
            LATENT_DIM,
        ),
    )
    return ActionIndexedPredictions(
        executed_indices=indices,
        all_predictions=all_predictions,
        all_flows_cell=torch.zeros(
            batch,
            ACTION_DIM,
            TOKEN_COUNT,
            2,
            dtype=all_predictions.dtype,
            device=all_predictions.device,
        ),
        executed=all_predictions[rows, indices],
        control_indices=control_indices,
        controls=controls,
    )


def test_normalized_token_l2_energy_matches_exact_registered_formula() -> None:
    query = torch.zeros(3, TOKEN_COUNT, LATENT_DIM)
    target = torch.zeros_like(query)
    query[:, :, 0] = 1.0
    target[0, :, 0] = 1.0
    target[1, :, 0] = -1.0
    target[2, :, 1] = 1.0

    energy = normalized_token_l2_energy(query, target)

    assert torch.equal(energy, torch.tensor([0.0, 4.0, 2.0]))
    assert torch.equal(
        energy,
        normalized_token_l2_energy(target, query),
    )
    assert torch.equal(
        energy,
        (query - target).square().sum(dim=-1).mean(dim=-1),
    )
    with pytest.raises(ValueError, match="L2-normalized"):
        normalized_token_l2_energy(query * 2.0, target)
    with pytest.raises(TypeError, match="float32"):
        normalized_token_l2_energy(query.double(), target.double())


def test_v10_factorized_retrieval_matches_exact_formula_and_candidate_order() -> None:
    torch.manual_seed(761)
    batch = 2
    all_predictions = F.normalize(
        torch.randn(batch, ACTION_DIM, TOKEN_COUNT, LATENT_DIM),
        p=2,
        dim=-1,
        eps=1e-8,
    )
    predictions = _retrieval_predictions(all_predictions, [2, HOLD_ACTION_INDEX])
    ema_current = F.normalize(
        torch.randn(batch, TOKEN_COUNT, LATENT_DIM),
        p=2,
        dim=-1,
        eps=1e-8,
    )
    ema_next = F.normalize(
        torch.randn(batch, TOKEN_COUNT, LATENT_DIM),
        p=2,
        dim=-1,
        eps=1e-8,
    )
    ema_deranged = F.normalize(
        torch.randn(batch, TOKEN_COUNT, LATENT_DIM),
        p=2,
        dim=-1,
        eps=1e-8,
    )
    non_hold = torch.tensor([True, False])

    terms = action_conditioned_next_target_retrieval_terms(
        predictions,
        ema_current,
        ema_next,
        ema_deranged,
        non_hold,
    )

    expected_action_energies = (
        all_predictions - ema_next[:, None]
    ).square().sum(dim=-1).mean(dim=-1)
    expected_target_candidates = torch.stack(
        (ema_next, ema_deranged, ema_current),
        dim=1,
    )
    expected_target_energies = (
        predictions.executed[:, None] - expected_target_candidates
    ).square().sum(dim=-1).mean(dim=-1)
    expected_action_rows = F.cross_entropy(
        -expected_action_energies,
        predictions.executed_indices,
        reduction="none",
    )
    expected_target_rows = torch.stack((
        F.cross_entropy(
            -expected_target_energies[0:1],
            torch.zeros(1, dtype=torch.long),
        ),
        F.cross_entropy(
            -expected_target_energies[1:2, :2],
            torch.zeros(1, dtype=torch.long),
        ),
    ))
    expected_jepa = (
        all_predictions - ema_next[:, None]
    ).square().mean(dim=(2, 3)).gather(
        1,
        predictions.executed_indices[:, None],
    ).mean()

    assert isinstance(terms, ActionConditionedNextTargetRetrievalTerms)
    assert torch.equal(terms.action_energies, expected_action_energies)
    assert torch.equal(terms.action_logits, -terms.action_energies)
    assert torch.equal(terms.action_nll_per_row, expected_action_rows)
    assert torch.equal(terms.action_retrieval, expected_action_rows.mean())
    assert torch.equal(terms.target_energies, expected_target_energies)
    assert torch.equal(terms.target_logits, -terms.target_energies)
    assert torch.equal(terms.target_nll_per_row, expected_target_rows)
    assert torch.equal(terms.target_retrieval, expected_target_rows.mean())
    assert torch.equal(
        terms.target_candidate_mask,
        torch.tensor([[True, True, True], [True, True, False]]),
    )
    assert torch.equal(terms.jepa, expected_jepa)
    assert torch.equal(
        terms.total,
        expected_jepa
        + expected_action_rows.mean()
        + expected_target_rows.mean(),
    )
    assert bool((terms.action_energies >= 0.0).all())
    assert bool((terms.action_energies <= 4.0).all())
    assert bool((terms.target_energies >= 0.0).all())
    assert bool((terms.target_energies <= 4.0).all())
    for wrong_non_hold in (
        torch.tensor([False, False]),
        torch.tensor([True, True]),
    ):
        with pytest.raises(ValueError, match="exactly match"):
            action_conditioned_next_target_retrieval_terms(
                predictions,
                ema_current,
                ema_next,
                ema_deranged,
                wrong_non_hold,
            )


def test_v10_equal_energy_gradients_and_hold_mask_are_exact() -> None:
    batch = 2
    all_predictions = torch.zeros(
        batch,
        ACTION_DIM,
        TOKEN_COUNT,
        LATENT_DIM,
    )
    all_predictions[..., 0] = 1.0
    all_predictions.requires_grad_()
    predictions = _retrieval_predictions(
        all_predictions,
        [3, HOLD_ACTION_INDEX],
    )
    targets = torch.zeros(batch, TOKEN_COUNT, LATENT_DIM)
    targets[..., 1] = 1.0
    terms = action_conditioned_next_target_retrieval_terms(
        predictions,
        targets,
        targets,
        targets,
        torch.tensor([True, False]),
    )

    action_gradient = torch.autograd.grad(
        terms.action_retrieval,
        terms.action_energies,
        retain_graph=True,
    )[0]
    target_gradient = torch.autograd.grad(
        terms.target_retrieval,
        terms.target_energies,
        retain_graph=True,
    )[0]

    assert terms.action_nll_per_row.tolist() == pytest.approx(
        [math.log(9.0), math.log(9.0)]
    )
    for row, executed in enumerate((3, HOLD_ACTION_INDEX)):
        assert action_gradient[row, executed].item() == pytest.approx(4.0 / 9.0)
        wrong = torch.cat((
            action_gradient[row, :executed],
            action_gradient[row, executed + 1 :],
        ))
        assert wrong.tolist() == pytest.approx([-1.0 / 18.0] * 8)
    assert terms.target_nll_per_row.tolist() == pytest.approx(
        [math.log(3.0), math.log(2.0)]
    )
    assert target_gradient[0].tolist() == pytest.approx(
        [1.0 / 3.0, -1.0 / 6.0, -1.0 / 6.0]
    )
    assert target_gradient[1].tolist() == pytest.approx(
        [1.0 / 4.0, -1.0 / 4.0, 0.0]
    )
    prediction_gradient = torch.autograd.grad(
        terms.total,
        all_predictions,
    )[0]
    assert torch.isfinite(prediction_gradient).all()
    assert torch.count_nonzero(prediction_gradient).item() > 0


def test_v10_retrieval_targets_are_detached_and_each_ce_reaches_queries() -> None:
    torch.manual_seed(9203)
    batch = 2
    prediction_raw = torch.randn(
        batch,
        ACTION_DIM,
        TOKEN_COUNT,
        LATENT_DIM,
        requires_grad=True,
    )
    all_predictions = F.normalize(
        prediction_raw,
        p=2,
        dim=-1,
        eps=1e-8,
    )
    predictions = _retrieval_predictions(all_predictions, [1, 7])
    target_raw = [
        torch.randn(
            batch,
            TOKEN_COUNT,
            LATENT_DIM,
            requires_grad=True,
        )
        for _ in range(3)
    ]
    ema_current, ema_next, ema_deranged = [
        F.normalize(value, p=2, dim=-1, eps=1e-8)
        for value in target_raw
    ]
    terms = action_conditioned_next_target_retrieval_terms(
        predictions,
        ema_current,
        ema_next,
        ema_deranged,
        torch.tensor([True, True]),
    )

    action_query_gradient = torch.autograd.grad(
        terms.action_retrieval,
        prediction_raw,
        retain_graph=True,
    )[0]
    target_query_gradient = torch.autograd.grad(
        terms.target_retrieval,
        prediction_raw,
        retain_graph=True,
    )[0]
    target_gradients = torch.autograd.grad(
        terms.total,
        target_raw,
        allow_unused=True,
    )

    for gradient in (action_query_gradient, target_query_gradient):
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient).item() > 0
    assert target_gradients == (None, None, None)


def test_v10_each_ce_and_their_sum_reach_the_registered_live_paths() -> None:
    torch.manual_seed(8119)
    encoder = VisionEncoder(
        image_size=16,
        patch_size=1,
        hidden_dim=LATENT_DIM,
        depth=1,
        n_heads=3,
        mlp_ratio=1,
        dropout=0.0,
    )
    predictor, projector = _spatial_fixture()
    with torch.no_grad():
        projector.flow_weight.copy_(
            torch.stack((
                torch.linspace(-0.02, 0.02, LATENT_DIM),
                torch.linspace(0.015, -0.015, LATENT_DIM),
            ))
        )
    online_state = F.normalize(
        encoder.forward_tokens(torch.randn(1, 3, 16, 16))[:, 1:],
        p=2,
        dim=-1,
        eps=1e-8,
    )
    ema_current, ema_next, ema_deranged = (
        F.normalize(
            torch.randn(1, TOKEN_COUNT, LATENT_DIM),
            p=2,
            dim=-1,
            eps=1e-8,
        )
        for _ in range(3)
    )
    predictions = predict_action_conditioned_flow_warps(
        predictor,
        projector,
        online_state,
        _one_hot([2]),
        ema_current,
    )
    terms = action_conditioned_next_target_retrieval_terms(
        predictions,
        ema_current,
        ema_next,
        ema_deranged,
        torch.tensor([True]),
    )
    action_parameters = tuple(predictor.action_embed.parameters())

    for loss in (terms.action_retrieval, terms.target_retrieval):
        gradients = torch.autograd.grad(
            loss,
            (projector.flow_weight, *action_parameters),
            retain_graph=True,
            allow_unused=True,
        )
        _assert_finite_nonzero_gradient(gradients[0])
        assert any(
            gradient is not None
            and bool(torch.isfinite(gradient).all())
            and torch.count_nonzero(gradient).item() > 0
            for gradient in gradients[1:]
        )

    encoder_gradient = torch.autograd.grad(
        terms.action_retrieval + terms.target_retrieval,
        encoder.patch_embed.weight,
    )[0]
    _assert_finite_nonzero_gradient(encoder_gradient)


def _assert_finite_nonzero_gradient(gradient: torch.Tensor | None) -> None:
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient).item() > 0


def test_dense_pairwise_head_exact_initialization_and_rng_isolation() -> None:
    torch.manual_seed(3701)
    cpu_before = torch.random.get_rng_state().clone()
    accelerator_before = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_initialized()
        else None
    )

    head = DensePairwiseSpatialCostVolumeInverseHead()

    assert torch.equal(torch.random.get_rng_state(), cpu_before)
    if accelerator_before is not None:
        accelerator_after = torch.cuda.get_rng_state_all()
        assert len(accelerator_after) == len(accelerator_before)
        assert all(
            torch.equal(after, before)
            for after, before in zip(
                accelerator_after,
                accelerator_before,
                strict=True,
            )
        )

    repeat = DensePairwiseSpatialCostVolumeInverseHead()
    assert torch.equal(torch.random.get_rng_state(), cpu_before)
    for left, right in zip(
        head.parameters(),
        repeat.parameters(),
        strict=True,
    ):
        assert torch.equal(left, right)

    assert isinstance(head.channel_projection, nn.Conv2d)
    assert head.channel_projection.in_channels == TOKEN_COUNT
    assert head.channel_projection.out_channels == DENSE_PAIRWISE_HEAD_CHANNELS
    assert head.channel_projection.kernel_size == (1, 1)
    assert head.channel_projection.stride == (1, 1)
    assert head.channel_projection.padding == (0, 0)
    assert head.channel_projection.bias is None
    assert isinstance(head.channel_activation, nn.GELU)
    assert head.channel_activation.approximate == "none"
    assert isinstance(head.spatial_projection, nn.Conv2d)
    assert head.spatial_projection.in_channels == DENSE_PAIRWISE_HEAD_CHANNELS
    assert head.spatial_projection.out_channels == DENSE_PAIRWISE_HEAD_CHANNELS
    assert head.spatial_projection.kernel_size == (3, 3)
    assert head.spatial_projection.stride == (1, 1)
    assert head.spatial_projection.padding == (1, 1)
    assert head.spatial_projection.bias is None
    assert isinstance(head.spatial_activation, nn.GELU)
    assert head.spatial_activation.approximate == "none"
    assert isinstance(head.pool, nn.AvgPool2d)
    assert head.pool.kernel_size == 4
    assert head.pool.stride == 4
    assert head.pool.padding == 0
    assert isinstance(head.classifier, nn.Linear)
    assert head.classifier.in_features == 256
    assert head.classifier.out_features == ACTION_DIM
    assert head.classifier.bias is not None

    parameters = tuple(head.parameters())
    assert sum(parameter.numel() for parameter in parameters) == 8_713
    assert all(
        parameter.device == torch.device("cpu")
        and parameter.dtype == torch.float32
        for parameter in parameters
    )
    for weight in (
        head.channel_projection.weight,
        head.spatial_projection.weight,
        head.classifier.weight,
    ):
        assert torch.count_nonzero(weight).item() == weight.numel()
    assert torch.count_nonzero(head.classifier.bias).item() == 0

    generator = torch.Generator(device="cpu")
    generator.manual_seed(DENSE_PAIRWISE_INVERSE_INITIALIZATION_SEED)
    expected_channel = torch.empty_like(head.channel_projection.weight)
    expected_spatial = torch.empty_like(head.spatial_projection.weight)
    expected_classifier = torch.empty_like(head.classifier.weight)
    nn.init.kaiming_normal_(
        expected_channel,
        a=0,
        mode="fan_in",
        nonlinearity="relu",
        generator=generator,
    )
    nn.init.kaiming_normal_(
        expected_spatial,
        a=0,
        mode="fan_in",
        nonlinearity="relu",
        generator=generator,
    )
    nn.init.normal_(
        expected_classifier,
        mean=0.0,
        std=1.0 / 16.0,
        generator=generator,
    )
    assert torch.equal(head.channel_projection.weight, expected_channel)
    assert torch.equal(head.spatial_projection.weight, expected_spatial)
    assert torch.equal(head.classifier.weight, expected_classifier)

    axis = torch.linspace(
        -1.0,
        1.0,
        TOKEN_SIDE,
        device="cpu",
        dtype=torch.float32,
    )
    rows_y, columns_x = torch.meshgrid(axis, axis, indexing="ij")
    expected_coordinates = torch.stack(
        [rows_y.flatten(), columns_x.flatten()],
        dim=-1,
    )
    assert torch.equal(head.coordinates_yx, expected_coordinates)
    assert tuple(head.coordinates_yx.shape) == (TOKEN_COUNT, 2)
    assert "coordinates_yx" not in head.state_dict()


def test_dense_pairwise_terms_match_exact_registered_formula() -> None:
    torch.manual_seed(1701)
    head = DensePairwiseSpatialCostVolumeInverseHead()
    current = torch.randn(2, TOKEN_COUNT, LATENT_DIM)
    next_state = (
        torch.roll(current, shifts=(3, 11), dims=(1, 2))
        + 0.13 * torch.randn_like(current)
    )
    labels = torch.tensor([1, 7], dtype=torch.long)
    row_scale = torch.tensor(
        [0.35, 1.4],
        dtype=torch.float32,
        requires_grad=True,
    )

    terms = dense_pairwise_spatial_cost_volume_inverse_terms(
        head,
        current,
        next_state,
        labels,
        row_scale,
    )

    normalized_current = F.layer_norm(
        current,
        (LATENT_DIM,),
        weight=None,
        bias=None,
        eps=DENSE_PAIRWISE_LAYER_NORM_EPS,
    )
    normalized_next = F.layer_norm(
        next_state,
        (LATENT_DIM,),
        weight=None,
        bias=None,
        eps=DENSE_PAIRWISE_LAYER_NORM_EPS,
    )
    expected_current_next = (
        normalized_current @ normalized_next.transpose(1, 2)
    ) / math.sqrt(LATENT_DIM)
    expected_current_current = (
        normalized_current @ normalized_current.transpose(1, 2)
    ) / math.sqrt(LATENT_DIM)
    expected_next_probabilities = F.softmax(
        expected_current_next,
        dim=-1,
    )
    expected_current_probabilities = F.softmax(
        expected_current_current,
        dim=-1,
    )
    expected_difference = (
        expected_next_probabilities - expected_current_probabilities
    )
    expected_volume = expected_difference.transpose(1, 2).reshape(
        2,
        TOKEN_COUNT,
        TOKEN_SIDE,
        TOKEN_SIDE,
    ).contiguous()
    expected_displacement = (
        expected_difference @ head.coordinates_yx
    ).reshape(
        2,
        TOKEN_SIDE,
        TOKEN_SIDE,
        2,
    ).permute(0, 3, 1, 2).contiguous()
    expected_logits = head(expected_volume)
    expected_rows = F.cross_entropy(
        expected_logits,
        labels,
        reduction="none",
    )

    assert isinstance(
        terms,
        DensePairwiseSpatialCostVolumeInverseTerms,
    )
    assert torch.equal(
        terms.current_next_cost_volume,
        expected_current_next,
    )
    assert torch.equal(
        terms.current_current_cost_volume,
        expected_current_current,
    )
    assert torch.equal(
        terms.current_next_probabilities,
        expected_next_probabilities,
    )
    assert torch.equal(
        terms.current_current_probabilities,
        expected_current_probabilities,
    )
    assert torch.equal(terms.probability_difference, expected_difference)
    assert torch.equal(terms.volume, expected_volume)
    assert terms.volume.is_contiguous()
    assert torch.equal(terms.displacement, expected_displacement)
    assert terms.displacement.is_contiguous()
    assert torch.equal(terms.logits, expected_logits)
    assert torch.equal(terms.nll_per_row, expected_rows)
    assert torch.equal(terms.unscaled_nll, expected_rows.mean())
    assert torch.equal(
        terms.loss,
        (row_scale.detach() * expected_rows).mean(),
    )
    assert torch.autograd.grad(
        terms.loss,
        row_scale,
        allow_unused=True,
    )[0] is None

    assert terms.current_next_cost_volume.shape == (2, 256, 256)
    assert terms.current_current_cost_volume.shape == (2, 256, 256)
    assert terms.volume.shape == (2, 256, 16, 16)
    assert terms.displacement.shape == (2, 2, 16, 16)
    assert terms.logits.shape == (2, ACTION_DIM)
    assert torch.isfinite(terms.volume).all()
    assert terms.volume.min().item() >= -1.0
    assert terms.volume.max().item() <= 1.0
    assert torch.allclose(
        terms.volume.sum(dim=1),
        torch.zeros_like(terms.volume.sum(dim=1)),
        rtol=0.0,
        atol=1e-6,
    )
    assert torch.isfinite(terms.displacement).all()
    assert terms.displacement.abs().max().item() <= (
        DENSE_PAIRWISE_DISPLACEMENT_BOUND
    )


def test_dense_pairwise_index_coded_source_target_layout_is_exact() -> None:
    coded_difference = torch.linspace(
        -1.0,
        1.0,
        TOKEN_COUNT * TOKEN_COUNT,
        dtype=torch.float32,
    ).reshape(1, TOKEN_COUNT, TOKEN_COUNT)
    volume = mechanism._dense_pairwise_probability_difference_volume(
        coded_difference
    )

    assert volume.is_contiguous()
    for source in range(TOKEN_COUNT):
        source_y, source_x = divmod(source, TOKEN_SIDE)
        assert torch.equal(
            volume[0, :, source_y, source_x],
            coded_difference[0, source, :],
        )


def test_dense_pairwise_action_labels_change_only_ce_and_gradient() -> None:
    torch.manual_seed(829)
    head = DensePairwiseSpatialCostVolumeInverseHead()
    current = torch.randn(
        3,
        TOKEN_COUNT,
        LATENT_DIM,
        requires_grad=True,
    )
    next_state = torch.randn(
        3,
        TOKEN_COUNT,
        LATENT_DIM,
        requires_grad=True,
    )
    scale = torch.tensor([0.4, 0.7, 1.1])
    probe = dense_pairwise_spatial_cost_volume_inverse_terms(
        head,
        current,
        next_state,
        torch.zeros(3, dtype=torch.long),
        scale,
    )
    best_labels = probe.logits.detach().argmax(dim=1)
    worst_labels = probe.logits.detach().argmin(dim=1)
    assert torch.all(best_labels != worst_labels)

    best = dense_pairwise_spatial_cost_volume_inverse_terms(
        head,
        current,
        next_state,
        best_labels,
        scale,
    )
    worst = dense_pairwise_spatial_cost_volume_inverse_terms(
        head,
        current,
        next_state,
        worst_labels,
        scale,
    )
    label_blind_fields = (
        "current_next_cost_volume",
        "current_current_cost_volume",
        "current_next_probabilities",
        "current_current_probabilities",
        "probability_difference",
        "volume",
        "displacement",
        "logits",
    )
    for field in label_blind_fields:
        assert torch.equal(getattr(best, field), getattr(worst, field))
    assert not torch.equal(best.nll_per_row, worst.nll_per_row)
    assert best.unscaled_nll.item() < worst.unscaled_nll.item()

    best_gradient = torch.autograd.grad(
        best.loss,
        head.classifier.weight,
        retain_graph=True,
    )[0]
    worst_gradient = torch.autograd.grad(
        worst.loss,
        head.classifier.weight,
    )[0]
    assert not torch.equal(best_gradient, worst_gradient)


def test_dense_pairwise_both_live_branches_and_every_head_parameter_are_live() -> None:
    torch.manual_seed(9107)
    head = DensePairwiseSpatialCostVolumeInverseHead()
    current = torch.randn(
        2,
        TOKEN_COUNT,
        LATENT_DIM,
        requires_grad=True,
    )
    next_state = torch.randn(
        2,
        TOKEN_COUNT,
        LATENT_DIM,
        requires_grad=True,
    )
    ema_tensor = torch.randn(
        2,
        TOKEN_COUNT,
        LATENT_DIM,
        requires_grad=True,
    )
    terms = dense_pairwise_spatial_cost_volume_inverse_terms(
        head,
        current,
        next_state,
        torch.tensor([0, 0], dtype=torch.long),
        torch.tensor([0.5, 1.0]),
    )
    gradients = torch.autograd.grad(
        terms.loss,
        (
            current,
            next_state,
            head.channel_projection.weight,
            head.spatial_projection.weight,
            head.classifier.weight,
            head.classifier.bias,
            ema_tensor,
        ),
        allow_unused=True,
    )

    for gradient in gradients[:-1]:
        _assert_finite_nonzero_gradient(gradient)
    assert gradients[-1] is None


def _encoder_online_geometry(
    encoder: VisionEncoder,
    rgb: torch.Tensor,
) -> torch.Tensor:
    return F.normalize(
        encoder.forward_tokens(rgb)[:, 1:],
        p=2,
        dim=-1,
        eps=1e-8,
    )


def test_dense_pairwise_each_branch_reaches_real_shared_encoder_parameters() -> None:
    torch.manual_seed(601)
    encoder = VisionEncoder(
        image_size=16,
        patch_size=1,
        hidden_dim=LATENT_DIM,
        depth=1,
        n_heads=3,
        mlp_ratio=1,
        dropout=0.0,
    )
    head = DensePairwiseSpatialCostVolumeInverseHead()
    current_rgb = torch.randn(1, 3, 16, 16)
    next_rgb = torch.randn(1, 3, 16, 16)
    labels = torch.tensor([4], dtype=torch.long)
    scale = torch.ones(1)

    current_live = _encoder_online_geometry(encoder, current_rgb)
    next_detached = _encoder_online_geometry(
        encoder,
        next_rgb,
    ).detach()
    current_terms = dense_pairwise_spatial_cost_volume_inverse_terms(
        head,
        current_live,
        next_detached,
        labels,
        scale,
    )
    current_gradient = torch.autograd.grad(
        current_terms.loss,
        encoder.patch_embed.weight,
    )[0]
    _assert_finite_nonzero_gradient(current_gradient)

    current_detached = _encoder_online_geometry(
        encoder,
        current_rgb,
    ).detach()
    next_live = _encoder_online_geometry(encoder, next_rgb)
    next_terms = dense_pairwise_spatial_cost_volume_inverse_terms(
        head,
        current_detached,
        next_live,
        labels,
        scale,
    )
    next_gradient = torch.autograd.grad(
        next_terms.loss,
        encoder.patch_embed.weight,
    )[0]
    _assert_finite_nonzero_gradient(next_gradient)


def test_dense_pairwise_identity_is_exact_zero_and_head_is_spatial() -> None:
    torch.manual_seed(2207)
    head = DensePairwiseSpatialCostVolumeInverseHead()
    state = torch.randn(2, TOKEN_COUNT, LATENT_DIM)
    identity = dense_pairwise_spatial_cost_volume_inverse_terms(
        head,
        state,
        state,
        torch.tensor([0, 8], dtype=torch.long),
        torch.ones(2),
    )
    assert torch.count_nonzero(identity.probability_difference).item() == 0
    assert torch.count_nonzero(identity.volume).item() == 0
    assert torch.count_nonzero(identity.displacement).item() == 0
    assert torch.count_nonzero(identity.logits).item() == 0

    asymmetric_volume = torch.randn(
        2,
        TOKEN_COUNT,
        TOKEN_SIDE,
        TOKEN_SIDE,
    )
    observed: list[
        tuple[str, tuple[int, ...], tuple[int, ...]]
    ] = []

    def record(name: str):
        def hook(
            _module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            observed.append(
                (name, tuple(inputs[0].shape), tuple(output.shape))
            )
        return hook

    modules = (
        ("conv1", head.channel_projection),
        ("gelu1", head.channel_activation),
        ("conv3", head.spatial_projection),
        ("gelu2", head.spatial_activation),
        ("pool", head.pool),
        ("linear", head.classifier),
    )
    handles = [
        module.register_forward_hook(record(name))
        for name, module in modules
    ]
    baseline_logits = head(asymmetric_volume)
    for handle in handles:
        handle.remove()

    rolled_logits = head(
        torch.roll(
            asymmetric_volume,
            shifts=(1, 3),
            dims=(2, 3),
        )
    )
    assert not torch.equal(baseline_logits, rolled_logits)
    assert [name for name, _, _ in observed] == [
        "conv1",
        "gelu1",
        "conv3",
        "gelu2",
        "pool",
        "linear",
    ]
    assert observed[0][1][2:] == (16, 16)
    assert observed[0][2][2:] == (16, 16)
    assert observed[1][2][2:] == (16, 16)
    assert observed[2][1][2:] == (16, 16)
    assert observed[2][2][2:] == (16, 16)
    assert observed[3][2][2:] == (16, 16)
    assert observed[4][1][2:] == (16, 16)
    assert observed[4][2][2:] == (4, 4)

    permuted_diagnostic = identity.displacement.flip((1, 2, 3))
    assert permuted_diagnostic.shape == identity.displacement.shape
    assert torch.equal(head(identity.volume), identity.logits)
    assert [
        type(module) for module in head.children()
    ] == [
        nn.Conv2d,
        nn.GELU,
        nn.Conv2d,
        nn.GELU,
        nn.AvgPool2d,
        nn.Linear,
    ]


def test_v9_excludes_closed_v7_v8_correspondence_mechanisms() -> None:
    assert not hasattr(
        mechanism,
        "ActionConditionedLocalCorrespondenceTransport",
    )
    assert not hasattr(mechanism, "CorrespondenceTargets")
    assert not hasattr(mechanism, "CorrespondenceTerms")
    assert not hasattr(
        mechanism,
        "CorrespondenceActionIdentificationTerms",
    )
