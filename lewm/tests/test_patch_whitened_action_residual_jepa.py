from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.models.patch_whitened_action_residual_jepa import (
    ACTION_DIM,
    ACTION_GATE_BIAS,
    ACTION_GATE_INITIALIZATION_SEED,
    ACTION_GATE_WEIGHT_STD,
    HOLD_ACTION_INDEX,
    LATENT_DIM,
    RESIDUAL_ALPHA,
    ActionResidualLosses,
    ResidualPredictions,
    action_residual_losses,
    build_action_layout,
    initialize_action_gate_rows,
    patch_whitening_terms,
    predict_live_and_control_residuals,
    residual_reconstruct,
)
from lewm.models.phase2d_spatial_lewm import LinearTokenProjector
from lewm.models.spatial_predictor import SpatialTokenPredictor


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


class _ToyPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.action = nn.Linear(ACTION_DIM, LATENT_DIM, bias=False)

    def predict_step(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        return state + self.action(actions)[:, None]


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


def test_action_layout_contains_only_real_one_hot_controls() -> None:
    requested = _one_hot([0, HOLD_ACTION_INDEX, ACTION_DIM - 1])
    layout = build_action_layout(requested)

    assert layout.all_actions.shape == (3, ACTION_DIM, ACTION_DIM)
    assert torch.equal(layout.all_actions[0], torch.eye(ACTION_DIM))
    assert torch.equal(layout.requested_indices, torch.tensor([0, 6, 8]))
    assert layout.control_indices.shape == (3, ACTION_DIM - 1)
    assert layout.control_actions.shape == (3, ACTION_DIM - 1, ACTION_DIM)
    assert torch.all(layout.control_actions.sum(dim=-1) == 1)
    assert torch.all((layout.control_actions == 0) | (layout.control_actions == 1))
    assert torch.all(
        layout.control_indices != layout.requested_indices[:, None]
    )
    assert torch.equal(
        layout.wrong_loss_mask.sum(dim=1),
        torch.tensor([7, 8, 7]),
    )
    assert torch.equal(
        layout.non_hold_mask,
        torch.tensor([True, False, True]),
    )
    assert torch.equal(
        (layout.control_indices == HOLD_ACTION_INDEX).sum(dim=1),
        torch.tensor([1, 0, 1]),
    )

    with pytest.raises(ValueError, match="exact one-hot"):
        build_action_layout(torch.zeros(2, ACTION_DIM))
    with pytest.raises(TypeError, match="floating point"):
        build_action_layout(
            F.one_hot(torch.tensor([0]), num_classes=ACTION_DIM)
        )


def test_residual_reconstruction_has_fixed_scale_and_detached_skip() -> None:
    assert RESIDUAL_ALPHA == 0.1 / math.sqrt(192)
    ema_current = F.normalize(
        torch.randn(2, 3, LATENT_DIM), dim=-1
    ).requires_grad_()
    residual = torch.randn(2, 3, LATENT_DIM, requires_grad=True)

    result = residual_reconstruct(ema_current, residual)
    expected = F.normalize(
        ema_current.detach() + RESIDUAL_ALPHA * residual,
        dim=-1,
        eps=1e-8,
    )

    assert torch.allclose(result, expected)
    assert torch.allclose(
        result.norm(dim=-1), torch.ones(2, 3), atol=1e-6
    )
    result.square().sum().backward()
    assert ema_current.grad is None
    assert residual.grad is not None
    assert torch.isfinite(residual.grad).all()

    controls = residual_reconstruct(
        ema_current.detach(),
        torch.randn(2, 8, 3, LATENT_DIM),
    )
    assert controls.shape == (2, 8, 3, LATENT_DIM)
    with pytest.raises(ValueError, match="aligned"):
        residual_reconstruct(
            ema_current.detach(),
            torch.randn(2, 8, 4, LATENT_DIM),
        )


def test_live_and_control_predictions_use_projector_and_isolate_gradients() -> None:
    torch.manual_seed(73)
    predictor = _ToyPredictor()
    projector = nn.Linear(LATENT_DIM, LATENT_DIM)
    with torch.no_grad():
        projector.weight.copy_(torch.eye(LATENT_DIM))
        projector.bias.fill_(123.0)
    state = torch.randn(2, 3, LATENT_DIM, requires_grad=True)
    ema_current = F.normalize(
        torch.randn(2, 3, LATENT_DIM), dim=-1
    ).requires_grad_()
    requested = _one_hot([0, HOLD_ACTION_INDEX])

    predictions = predict_live_and_control_residuals(
        predictor,
        projector,
        state,
        requested,
        ema_current,
    )

    assert predictions.true.shape == (2, 3, LATENT_DIM)
    assert predictions.controls.shape == (2, 8, 3, LATENT_DIM)
    manual_raw = predictor.predict_step(state, requested)
    manual_residual = projector(manual_raw)
    manual = F.normalize(
        ema_current.detach() + RESIDUAL_ALPHA * manual_residual,
        dim=-1,
        eps=1e-8,
    )
    assert torch.allclose(predictions.true, manual)

    projector_without_bias = nn.Linear(LATENT_DIM, LATENT_DIM)
    with torch.no_grad():
        projector_without_bias.weight.copy_(projector.weight)
        projector_without_bias.bias.fill_(-987.0)
    comparison = predict_live_and_control_residuals(
        predictor,
        projector_without_bias,
        state,
        requested,
        ema_current,
    )
    assert not torch.allclose(predictions.true, comparison.true)
    assert not torch.allclose(predictions.controls, comparison.controls)

    control_gradient = torch.autograd.grad(
        predictions.controls[..., 0].sum(),
        (state, predictor.action.weight, projector.bias, ema_current),
        retain_graph=True,
        allow_unused=True,
    )
    assert control_gradient[0] is None
    assert control_gradient[1] is not None
    assert torch.count_nonzero(control_gradient[1]).item() > 0
    assert control_gradient[2] is not None
    assert torch.count_nonzero(control_gradient[2]).item() > 0
    assert control_gradient[3] is None
    true_state_gradient = torch.autograd.grad(
        predictions.true[..., 0].sum(),
        state,
    )[0]
    assert torch.count_nonzero(true_state_gradient).item() > 0


def test_helpers_integrate_with_registered_spatial_predictor() -> None:
    torch.manual_seed(101)
    predictor = SpatialTokenPredictor(
        latent_dim=LATENT_DIM,
        cmd_dim=ACTION_DIM,
        num_spatial_tokens=4,
        n_layers=2,
        n_heads=1,
        dim_head=32,
        mlp_dim=64,
        dropout=0.0,
    )
    projector = LinearTokenProjector(LATENT_DIM)
    initialize_action_gate_rows(predictor)
    shared_state = torch.randn(1, 4, LATENT_DIM).expand(2, -1, -1).clone()
    shared_skip = F.normalize(
        torch.randn(1, 4, LATENT_DIM), dim=-1
    ).expand(2, -1, -1).clone()

    predictions = predict_live_and_control_residuals(
        predictor,
        projector,
        shared_state,
        _one_hot([0, 1]),
        shared_skip,
    )

    assert predictions.true.shape == (2, 4, LATENT_DIM)
    assert predictions.controls.shape == (2, 8, 4, LATENT_DIM)
    assert not torch.equal(predictions.true[0], predictions.true[1])


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
    requested_indices: list[int],
    true_energy: torch.Tensor,
    control_energy: torch.Tensor,
) -> ResidualPredictions:
    batch = len(requested_indices)
    layout = build_action_layout(_one_hot(requested_indices))
    true = true_energy.sqrt()[:, None, None].expand(
        batch, 1, LATENT_DIM
    )
    controls = control_energy.sqrt()[:, :, None, None].expand(
        batch, ACTION_DIM - 1, 1, LATENT_DIM
    )
    return ResidualPredictions(layout=layout, true=true, controls=controls)


def test_action_hinges_use_row_means_and_real_hold_separately() -> None:
    requested_indices = [0, HOLD_ACTION_INDEX]
    layout = build_action_layout(_one_hot(requested_indices))
    true_energy = torch.full((2,), 0.95)
    control_energy = torch.empty(2, ACTION_DIM - 1)
    control_energy[0].fill_(0.50)
    control_energy[0, layout.control_indices[0] == HOLD_ACTION_INDEX] = 0.80
    control_energy[1].fill_(0.75)
    predictions = _predictions_from_energies(
        requested_indices, true_energy, control_energy
    )

    losses = action_residual_losses(
        predictions,
        torch.zeros(2, 1, LATENT_DIM),
    )

    assert isinstance(losses, ActionResidualLosses)
    assert losses.jepa.item() == pytest.approx(0.95, abs=2e-6)
    assert losses.wrong_per_row.tolist() == pytest.approx([0.50, 0.25])
    assert losses.wrong.item() == pytest.approx(0.375)
    assert losses.hold.item() == pytest.approx(0.20)


def test_action_hinge_boundary_empty_hold_and_live_reference_gradient() -> None:
    boundary_layout = build_action_layout(_one_hot([0]))
    boundary_predictions = _predictions_from_energies(
        [0],
        torch.tensor([0.95]),
        torch.ones(1, ACTION_DIM - 1),
    )
    boundary = action_residual_losses(
        boundary_predictions,
        torch.zeros(1, 1, LATENT_DIM),
    )
    assert boundary.wrong.item() == pytest.approx(0.0, abs=1e-6)
    assert boundary.hold.item() == pytest.approx(0.0, abs=1e-6)
    assert boundary_layout.wrong_loss_mask.sum().item() == 7

    all_hold_predictions = _predictions_from_energies(
        [HOLD_ACTION_INDEX, HOLD_ACTION_INDEX],
        torch.tensor([0.95, 0.95]),
        torch.full((2, ACTION_DIM - 1), 0.75),
    )
    all_hold = action_residual_losses(
        all_hold_predictions,
        torch.zeros(2, 1, LATENT_DIM),
    )
    assert all_hold.hold.shape == torch.Size([])
    assert all_hold.hold.dtype == torch.float32
    assert all_hold.hold.item() == 0.0
    assert all_hold.wrong.item() == pytest.approx(0.25)

    layout = build_action_layout(_one_hot([0]))
    true = torch.full(
        (1, 1, LATENT_DIM),
        math.sqrt(0.95),
        requires_grad=True,
    )
    controls = torch.full(
        (1, ACTION_DIM - 1, 1, LATENT_DIM),
        math.sqrt(0.50),
        requires_grad=True,
    )
    target = torch.zeros(
        1, 1, LATENT_DIM, requires_grad=True
    )
    predictions = ResidualPredictions(
        layout=layout,
        true=true,
        controls=controls,
    )
    losses = action_residual_losses(predictions, target)
    wrong_and_hold_gradients = torch.autograd.grad(
        losses.wrong + losses.hold,
        (true, controls, target),
        retain_graph=True,
        allow_unused=True,
    )
    assert wrong_and_hold_gradients[0] is not None
    assert torch.isfinite(wrong_and_hold_gradients[0]).all()
    assert torch.count_nonzero(wrong_and_hold_gradients[0]).item() > 0
    assert wrong_and_hold_gradients[1] is not None
    assert torch.isfinite(wrong_and_hold_gradients[1]).all()
    assert torch.count_nonzero(wrong_and_hold_gradients[1]).item() > 0
    assert wrong_and_hold_gradients[2] is None
    jepa_gradients = torch.autograd.grad(
        losses.jepa,
        (true, target),
        allow_unused=True,
    )
    assert jepa_gradients[0] is not None
    assert torch.count_nonzero(jepa_gradients[0]).item() > 0
    assert jepa_gradients[1] is None
