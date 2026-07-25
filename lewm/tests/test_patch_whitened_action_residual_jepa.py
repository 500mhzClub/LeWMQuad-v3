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
    LATENT_DIM,
    RESIDUAL_ALPHA,
    ActionIndexedLosses,
    ActionIndexedPredictions,
    ActionIndexedResidualOperators,
    action_independent_trunk,
    action_indexed_energy_nll,
    initialize_action_gate_rows,
    patch_whitening_terms,
    predict_action_indexed_residuals,
    requested_action_indices,
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


def test_action_operator_wrapper_is_zero_bias_free_and_rng_free() -> None:
    shared = nn.Linear(LATENT_DIM, LATENT_DIM)
    torch.manual_seed(812)
    rng_before = torch.random.get_rng_state().clone()

    wrapper = ActionIndexedResidualOperators(shared)

    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert wrapper.shared_projector is shared
    assert tuple(wrapper.action_weights.shape) == (
        ACTION_DIM,
        LATENT_DIM,
        LATENT_DIM,
    )
    assert wrapper.action_weights.numel() == 331_776
    assert torch.count_nonzero(wrapper.action_weights).item() == 0
    assert [name for name, _ in wrapper.named_parameters()] == [
        "action_weights",
        "shared_projector.weight",
        "shared_projector.bias",
    ]
    assert not hasattr(wrapper, "bias")


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


class _ForbiddenActionEmbed(nn.Module):
    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        raise AssertionError("V4 trunk must bypass action_embed")


def _spatial_fixture() -> tuple[
    SpatialTokenPredictor,
    ActionIndexedResidualOperators,
]:
    torch.manual_seed(101)
    predictor = SpatialTokenPredictor(
        latent_dim=LATENT_DIM,
        cmd_dim=ACTION_DIM,
        num_spatial_tokens=3,
        n_layers=1,
        n_heads=1,
        dim_head=32,
        mlp_dim=64,
        dropout=0.0,
    )
    initialize_action_gate_rows(predictor)
    predictor.action_embed = _ForbiddenActionEmbed()
    projector = ActionIndexedResidualOperators(
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
    state = torch.randn(2, 3, LATENT_DIM)
    hidden = action_independent_trunk(predictor, state)
    handle.remove()

    assert hidden.shape == state.shape
    assert len(captured) == 1
    assert torch.count_nonzero(captured[0][0]).item() == 0
    assert captured[0][1] is False


def test_all_action_predictions_start_bitwise_equal_and_gather_exactly() -> None:
    predictor, projector = _spatial_fixture()
    state = torch.randn(2, 3, LATENT_DIM)
    skip = F.normalize(torch.randn(2, 3, LATENT_DIM), dim=-1)
    requested = _one_hot([0, 6])

    predictions = predict_action_indexed_residuals(
        predictor,
        projector,
        state,
        requested,
        skip,
    )

    assert predictions.all_predictions.shape == (
        2, ACTION_DIM, 3, LATENT_DIM
    )
    assert predictions.executed.shape == (2, 3, LATENT_DIM)
    assert predictions.controls.shape == (
        2, ACTION_DIM - 1, 3, LATENT_DIM
    )
    assert predictions.control_indices.shape == (2, ACTION_DIM - 1)
    for action in range(1, ACTION_DIM):
        assert torch.equal(
            predictions.all_predictions[:, 0],
            predictions.all_predictions[:, action],
        )
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
                -1, -1, 3, LATENT_DIM
            ),
        ),
    )


def test_action_operator_changes_only_its_candidate() -> None:
    predictor, projector = _spatial_fixture()
    state = torch.randn(1, 3, LATENT_DIM)
    skip = F.normalize(torch.randn(1, 3, LATENT_DIM), dim=-1)
    requested = _one_hot([2])
    baseline = predict_action_indexed_residuals(
        predictor, projector, state, requested, skip
    ).all_predictions.detach()

    with torch.no_grad():
        projector.action_weights[4].copy_(torch.eye(LATENT_DIM))
    changed = predict_action_indexed_residuals(
        predictor, projector, state, requested, skip
    ).all_predictions.detach()

    assert not torch.equal(changed[:, 4], baseline[:, 4])
    for action in range(ACTION_DIM):
        if action != 4:
            assert torch.equal(changed[:, action], baseline[:, action])


def test_executed_path_is_live_while_wrong_shared_path_is_detached() -> None:
    predictor, projector = _spatial_fixture()
    state = torch.randn(1, 3, LATENT_DIM, requires_grad=True)
    skip = F.normalize(
        torch.randn(1, 3, LATENT_DIM), dim=-1
    ).requires_grad_()
    predictions = predict_action_indexed_residuals(
        predictor, projector, state, _one_hot([2]), skip
    )
    shared_weight = projector.shared_projector.linear.weight

    wrong_gradients = torch.autograd.grad(
        predictions.all_predictions[:, 4, :, 0].sum(),
        (state, shared_weight, projector.action_weights, skip),
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
    assert torch.count_nonzero(wrong_gradients[2][4]).item() > 0
    assert torch.count_nonzero(wrong_gradients[2][2]).item() == 0
    assert wrong_gradients[3] is None

    executed_gradients = torch.autograd.grad(
        predictions.executed[..., 0].sum(),
        (state, shared_weight, projector.action_weights, skip),
        allow_unused=True,
    )
    assert executed_gradients[0] is not None
    assert torch.count_nonzero(executed_gradients[0]).item() > 0
    assert executed_gradients[1] is not None
    assert torch.count_nonzero(executed_gradients[1]).item() > 0
    assert executed_gradients[2] is not None
    assert torch.count_nonzero(executed_gradients[2][2]).item() > 0
    assert torch.count_nonzero(executed_gradients[2][4]).item() == 0
    assert executed_gradients[3] is None


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
