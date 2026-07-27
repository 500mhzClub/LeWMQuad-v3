from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.benchmarks import (
    go2_geometry_anchored_two_mode_event_delta_joint_jepa_v1 as gate_contract,
)
from lewm.models.encoders import VisionEncoder
from lewm.models.geometry_anchored_deformable_bev_lift_joint_jepa_v1 import (
    GeometryAnchoredDeformableBevLiftJointJepaV1,
    GeometryAnchoredDeformableBevLiftJointJepaV1Config,
    GeometryAnchoredDeformableBevLiftV1,
)
from lewm.models import geometry_anchored_two_mode_event_delta_joint_jepa_v1 as api


PREDICTOR_PARAMETER_NAMES = (
    "action_embedding.weight",
    "input_projection.weight",
    "input_projection.bias",
    "residual_blocks.0.conv1.weight",
    "residual_blocks.0.conv1.bias",
    "residual_blocks.0.conv2.weight",
    "residual_blocks.0.conv2.bias",
    "residual_blocks.1.conv1.weight",
    "residual_blocks.1.conv1.bias",
    "residual_blocks.1.conv2.weight",
    "residual_blocks.1.conv2.bias",
    "event_mean_head.weight",
    "event_mean_head.bias",
    "event_logit_head.weight",
    "event_logit_head.bias",
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


def _predictor() -> api.TwoModeEventDeltaPredictorV1:
    caller_rng = torch.random.get_rng_state().clone()
    try:
        torch.random.default_generator.manual_seed(20260712)
        return api.TwoModeEventDeltaPredictorV1(
            GeometryAnchoredDeformableBevLiftJointJepaV1Config()
        )
    finally:
        torch.random.set_rng_state(caller_rng)


def _one_hot(indices: list[int]) -> torch.Tensor:
    return F.one_hot(torch.tensor(indices), num_classes=9).to(torch.float32)


def _all_actions(
    predictor: api.TwoModeEventDeltaPredictorV1,
    latent: torch.Tensor,
) -> api.EventDeltaPrediction:
    batch = latent.shape[0]
    repeated = latent[:, None].expand(-1, 9, -1, -1, -1).reshape(
        batch * 9, 64, 64, 64
    )
    actions = torch.eye(9, dtype=torch.float32)[None].expand(
        batch, -1, -1
    ).reshape(batch * 9, 9)
    prediction = predictor(repeated, actions)
    return api.EventDeltaPrediction(
        prediction.mu_event.reshape(batch, 9, 64, 64, 64),
        prediction.event_logit.reshape(batch, 9, 1, 64, 64),
    )


def _tensor_bytes(value: torch.Tensor) -> bytes:
    flat = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return flat.numpy().tobytes()


def _assert_module_bytes_equal(first: nn.Module, second: nn.Module) -> None:
    first_state = first.state_dict()
    second_state = second.state_dict()
    assert first_state.keys() == second_state.keys()
    for name in first_state:
        assert first_state[name].shape == second_state[name].shape
        assert first_state[name].dtype == second_state[name].dtype
        assert _tensor_bytes(first_state[name]) == _tensor_bytes(second_state[name])


def _manual_smooth_l1(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    difference = (first - second).abs()
    return torch.where(
        difference < 1.0,
        0.5 * difference.square(),
        difference - 0.5,
    )


def _passing_event_gate_metrics() -> dict[str, float | int | bool]:
    """Synthetic CPU-only metrics with all twenty event conjuncts passing."""

    return {
        "event_balanced_energy": 0.8,
        "action_nll": 1.0,
        "action_macro_balanced_accuracy": 0.5,
        "hardest_wrong_positive_family_count": 8,
        "mean_executed_action_energy": 0.8,
        "mean_wrong_action_energy": 1.0,
        "mean_non_hold_executed_action_energy": 0.8,
        "mean_non_hold_hold_action_energy": 1.0,
        "target_nll": 0.5,
        "target_strict_win_rate": 0.8,
        "target_positive_family_count": 8,
        "context_true_energy": 0.8,
        "context_swap_energy": 1.0,
        "context_nll": 0.5,
        "context_true_to_swap_energy_ratio": 0.8,
        "context_true_strict_win_rate": 0.8,
        "context_positive_family_count": 8,
        "state_true_energy": 0.8,
        "state_template_energy": 1.0,
        "state_true_to_template_energy_ratio": 0.8,
        "state_positive_family_count": 8,
        "executed_action_energy": 0.8,
        "state_only_energy": 1.0,
        "executed_to_state_only_energy_ratio": 0.8,
        "state_only_positive_family_count": 8,
        "two_mode_energy": 0.8,
        "matched_single_mean_energy": 1.0,
        "two_mode_to_matched_single_ratio": 0.8,
        "matched_single_positive_family_count": 8,
        "event_changed_energy": 0.8,
        "zero_changed_energy": 1.0,
        "event_over_zero_changed_positive_family_count": 8,
        "zero_static_energy": 0.8,
        "event_static_energy": 1.0,
        "zero_over_event_static_positive_family_count": 8,
        "mixture_overall_energy": 0.8,
        "zero_overall_energy": 1.0,
        "event_overall_energy": 1.0,
        "mixture_beats_zero_family_count": 8,
        "mixture_beats_event_family_count": 8,
        "mu_event_changed_abs": 0.1,
        "prior_changed_mean": 0.6,
        "prior_static_mean": 0.4,
        "prior_mean": 0.5,
        "prior_spatial_variance": 0.1,
        "prior_context_difference": 0.1,
        "prior_context_difference_positive_family_count": 8,
        "posterior_changed_mean": 0.6,
        "posterior_static_mean": 0.4,
        "posterior_mean": 0.5,
        "posterior_event_and_zero_family_count": 8,
        "all_action_predictor_training_forward_count": 2_400,
        "context_swap_predictor_training_forward_count": 2_400,
        "semantic_term_evaluation_count": 4_000,
        "event_persistence_term_evaluation_count": 2_400,
        "action_term_evaluation_count": 2_400,
        "target_term_evaluation_count": 2_400,
        "context_term_evaluation_count": 2_400,
        "registered_scalar_term_evaluation_count": 13_600,
        "combined_objective_evaluation_count": 4_000,
        "backward_call_count": 4_000,
        "online_encoder_lift_training_forward_count": 10_400,
        "semantic_head_training_forward_count": 8_000,
        "target_encoder_lift_training_forward_count": 7_200,
        "warning_policy_exact": True,
        "state_hash_accounting_exact": True,
        "receipt_schema_accounting_exact": True,
        "access_and_custody_accounting_exact": True,
        "shared_gradient_ratio_evaluation_count": 600,
        "shared_gradient_ratio_pass_count": 600,
        "shared_gradient_ratio_failure_count": 0,
        "action_embedding_dynamics_gradient_update_count": 600,
        "predictor_trunk_dynamics_gradient_update_count": 600,
        "event_mean_head_dynamics_gradient_update_count": 600,
        "event_logit_head_dynamics_gradient_update_count": 600,
    }


def test_constant_state_action_template_fixture_fails_context_and_state_gates() -> None:
    metrics = _passing_event_gate_metrics()
    assert all(gate_contract._final_event_conjuncts(metrics, 1.0).values())

    # A row-independent state and its action template are identical.  The
    # same command on a swapped context is identical too, so neither control
    # has a strict margin.  These are constructed CPU tensors only.
    constant_energy = torch.full((8,), 0.8, dtype=torch.float32)
    identical = float(constant_energy.mean())
    metrics.update({
        "context_true_energy": identical,
        "context_swap_energy": identical,
        "context_nll": float(torch.log(torch.tensor(2.0))),
        "context_true_to_swap_energy_ratio": 1.0,
        "context_true_strict_win_rate": 0.0,
        "context_positive_family_count": 0,
        "state_true_energy": identical,
        "state_template_energy": identical,
        "state_true_to_template_energy_ratio": 1.0,
        "state_positive_family_count": 0,
    })
    conjuncts = gate_contract._final_event_conjuncts(metrics, 1.0)
    assert conjuncts["event_08_true_context_beats_same_command_swap"] is False
    assert conjuncts["event_09_true_state_beats_action_template"] is False


def test_action_ignored_fixture_fails_action_and_state_only_gates() -> None:
    metrics = _passing_event_gate_metrics()

    # With all nine action energies equal, the action posterior is uniform,
    # wrong/HOLD margins vanish, and uniform action marginalization is exactly
    # the executed energy.  This explicitly catches an action-ignored model.
    equal_action_energies = torch.full((8, 9), 0.8, dtype=torch.float32)
    executed = float(equal_action_energies[:, 0].mean())
    state_only = float(
        (-torch.logsumexp(-equal_action_energies, dim=1)
         + torch.log(torch.tensor(9.0))).mean()
    )
    metrics.update({
        "action_nll": float(torch.log(torch.tensor(9.0))),
        "action_macro_balanced_accuracy": 1.0 / 9.0,
        "hardest_wrong_positive_family_count": 0,
        "mean_executed_action_energy": executed,
        "mean_wrong_action_energy": executed,
        "mean_non_hold_executed_action_energy": executed,
        "mean_non_hold_hold_action_energy": executed,
        "executed_action_energy": executed,
        "state_only_energy": state_only,
        "executed_to_state_only_energy_ratio": 1.0,
        "state_only_positive_family_count": 0,
    })
    conjuncts = gate_contract._final_event_conjuncts(metrics, 1.0)
    for name in (
        "event_02_action_nll_strictly_below_point95_log9",
        "event_03_action_macro_balanced_accuracy_strictly_above_two_ninths",
        "event_04_executed_beats_hardest_wrong_in_at_least_6_families",
        "event_05_mean_wrong_energy_strictly_above_executed",
        "event_06_nonhold_HOLD_energy_strictly_above_executed",
        "event_10_executed_action_beats_state_only_mixture",
    ):
        assert conjuncts[name] is False


def test_predictor_inventory_initialization_and_mode_identity() -> None:
    predictor = _predictor()
    assert tuple(name for name, _ in predictor.named_parameters()) == (
        PREDICTOR_PARAMETER_NAMES
    )
    assert len(tuple(predictor.parameters())) == 15
    assert sum(value.numel() for value in predictor.parameters()) == 231_505
    assert tuple(predictor.action_embedding.weight.shape) == (9, 16)
    assert torch.equal(
        predictor.action_embedding.weight,
        torch.zeros_like(predictor.action_embedding.weight),
    )
    assert tuple(predictor.input_projection.weight.shape) == (64, 80, 3, 3)
    assert tuple(predictor.event_mean_head.weight.shape) == (64, 64, 3, 3)
    assert bool(torch.isfinite(predictor.event_mean_head.weight).all())
    assert float(predictor.event_mean_head.weight.detach().abs().sum()) > 0.0
    assert torch.equal(
        predictor.event_mean_head.bias,
        torch.zeros_like(predictor.event_mean_head.bias),
    )
    assert tuple(predictor.event_logit_head.weight.shape) == (1, 64, 3, 3)
    assert torch.equal(
        predictor.event_logit_head.weight,
        torch.zeros_like(predictor.event_logit_head.weight),
    )
    assert torch.equal(
        predictor.event_logit_head.bias,
        torch.zeros_like(predictor.event_logit_head.bias),
    )
    for forbidden in (
        "zero_event_head", "scale_head", "variance", "flow", "warp", "transport"
    ):
        assert not hasattr(predictor, forbidden)


def test_exact_single_seed_stream_and_representation_identity(
    n320_encoder_state: dict[str, torch.Tensor],
) -> None:
    config = GeometryAnchoredDeformableBevLiftJointJepaV1Config()
    caller_rng = torch.random.get_rng_state().clone()
    try:
        torch.random.default_generator.manual_seed(config.initialization_seed)
        GeometryAnchoredDeformableBevLiftV1(config)
        nn.Conv2d(config.bev_dim, config.state_classes, kernel_size=1, bias=True)
        action_embedding = nn.Embedding(9, 16)
        nn.init.zeros_(action_embedding.weight)
        input_projection = nn.Conv2d(80, 64, 3, stride=1, padding=1, bias=True)
        residual_convolutions = [
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=True)
            for _ in range(4)
        ]
        event_mean_head = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=True)
        nn.init.normal_(event_mean_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(event_mean_head.bias)
        event_logit_head = nn.Conv2d(64, 1, 3, stride=1, padding=1, bias=True)
        nn.init.zeros_(event_logit_head.weight)
        nn.init.zeros_(event_logit_head.bias)
        expected = {
            "action_embedding.weight": action_embedding.weight,
            "input_projection.weight": input_projection.weight,
            "input_projection.bias": input_projection.bias,
            "residual_blocks.0.conv1.weight": residual_convolutions[0].weight,
            "residual_blocks.0.conv1.bias": residual_convolutions[0].bias,
            "residual_blocks.0.conv2.weight": residual_convolutions[1].weight,
            "residual_blocks.0.conv2.bias": residual_convolutions[1].bias,
            "residual_blocks.1.conv1.weight": residual_convolutions[2].weight,
            "residual_blocks.1.conv1.bias": residual_convolutions[2].bias,
            "residual_blocks.1.conv2.weight": residual_convolutions[3].weight,
            "residual_blocks.1.conv2.bias": residual_convolutions[3].bias,
            "event_mean_head.weight": event_mean_head.weight,
            "event_mean_head.bias": event_mean_head.bias,
            "event_logit_head.weight": event_logit_head.weight,
            "event_logit_head.bias": event_logit_head.bias,
        }
    finally:
        torch.random.set_rng_state(caller_rng)

    torch.random.default_generator.manual_seed(713)
    rng_before = torch.random.get_rng_state().clone()
    predecessor = GeometryAnchoredDeformableBevLiftJointJepaV1(n320_encoder_state)
    replacement = api.GeometryAnchoredTwoModeEventDeltaJointJepaV1(
        n320_encoder_state
    )
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert tuple(replacement.predictor.state_dict()) == PREDICTOR_PARAMETER_NAMES
    for name, observed in replacement.predictor.state_dict().items():
        assert _tensor_bytes(observed) == _tensor_bytes(expected[name])
    for before, after in (
        (predecessor.encoder, replacement.encoder),
        (predecessor.bev_lift, replacement.bev_lift),
        (predecessor.semantic_head, replacement.semantic_head),
        (predecessor.target_encoder, replacement.target_encoder),
        (predecessor.target_bev_lift, replacement.target_bev_lift),
    ):
        _assert_module_bytes_equal(before, after)
    assert api.GeometryAnchoredDeformableBevLiftJointJepaV1 is (
        api.GeometryAnchoredTwoModeEventDeltaJointJepaV1
    )


def test_affine_free_per_cell_normalization_matches_independent_reference() -> None:
    generator = torch.Generator().manual_seed(81)
    latent = torch.randn(2, 64, 64, 64, generator=generator)
    observed = api.normalize_latent_per_cell(latent)
    expected = F.layer_norm(
        latent.permute(0, 2, 3, 1),
        (64,),
        weight=None,
        bias=None,
        eps=1e-5,
    ).permute(0, 3, 1, 2)
    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)
    assert observed.dtype == torch.float32
    with pytest.raises(ValueError, match="shape"):
        api.normalize_latent_per_cell(torch.zeros(1, 64, 63, 64))
    with pytest.raises(TypeError, match="float32"):
        api.normalize_latent_per_cell(torch.zeros(1, 64, 64, 64).double())
    with pytest.raises(FloatingPointError, match="nonfinite"):
        api.normalize_latent_per_cell(
            torch.full((1, 64, 64, 64), float("nan"))
        )


def test_one_and_all_action_shapes_and_exact_update_zero_symmetry(
    n320_encoder_state: dict[str, torch.Tensor],
) -> None:
    model = api.GeometryAnchoredTwoModeEventDeltaJointJepaV1(
        n320_encoder_state
    ).eval()
    latent = api.normalize_latent_per_cell(
        torch.randn(1, 64, 64, 64, generator=torch.Generator().manual_seed(17))
    )
    one = model.predict(latent, _one_hot([6]))
    assert tuple(one.mu_event.shape) == (1, 64, 64, 64)
    assert tuple(one.event_logit.shape) == (1, 1, 64, 64)
    all_actions = model.predict_all_actions(latent)
    assert tuple(all_actions.mu_event.shape) == (1, 9, 64, 64, 64)
    assert tuple(all_actions.event_logit.shape) == (1, 9, 1, 64, 64)
    assert torch.equal(
        all_actions.mu_event,
        all_actions.mu_event[:, :1].expand_as(all_actions.mu_event),
    )
    assert torch.equal(
        all_actions.event_logit,
        torch.zeros_like(all_actions.event_logit),
    )
    assert torch.equal(
        api.event_prior_probability(all_actions),
        torch.full((1, 9, 64, 64), 0.5, dtype=torch.float32),
    )
    assert torch.equal(one.mu_event, all_actions.mu_event[:, 6])
    assert torch.equal(one.event_logit, all_actions.event_logit[:, 6])
    assert model.predict_event is not None
    assert model.predict_all_action_event_deltas is not None


def test_component_energies_match_independent_formula_and_reject_broadcast() -> None:
    generator = torch.Generator().manual_seed(413)
    target = torch.randn(1, 64, 64, 64, generator=generator)
    mu = torch.randn(1, 64, 64, 64, generator=generator)
    logit = torch.randn(1, 1, 64, 64, generator=generator)
    prediction = api.EventDeltaPrediction(mu, logit)
    observed = api.event_delta_cell_energies(target, prediction)
    expected_zero = _manual_smooth_l1(target, torch.zeros_like(target)).mean(1)
    expected_event = _manual_smooth_l1(target, mu).mean(1)
    torch.testing.assert_close(observed.zero_event, expected_zero, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        observed.learned_event, expected_event, rtol=0.0, atol=0.0
    )

    all_prediction = api.EventDeltaPrediction(
        mu[:, None].expand(-1, 9, -1, -1, -1).clone(),
        logit[:, None].expand(-1, 9, -1, -1, -1).clone(),
    )
    all_energy = api.event_delta_cell_energies(target, all_prediction)
    assert tuple(all_energy.zero_event.shape) == (1, 9, 64, 64)
    assert tuple(all_energy.learned_event.shape) == (1, 9, 64, 64)
    assert torch.equal(
        all_energy.zero_event,
        observed.zero_event[:, None].expand_as(all_energy.zero_event),
    )
    with pytest.raises(ValueError, match="singleton"):
        api.two_mode_event_energy(
            all_energy.zero_event,
            all_energy.learned_event,
            torch.zeros(1, 1, 64, 64),
            0.2,
        )
    with pytest.raises(ValueError, match="singleton"):
        api.two_mode_event_energy(
            observed.zero_event,
            observed.learned_event,
            torch.zeros(1, 64, 64),
            0.2,
        )


def test_stable_mixture_posterior_persistence_identity_and_extreme_logits() -> None:
    generator = torch.Generator().manual_seed(99)
    zero = torch.rand(2, 64, 64, generator=generator)
    learned = torch.rand(2, 64, 64, generator=generator)
    logit = torch.empty(2, 1, 64, 64)
    logit[0].fill_(1000.0)
    logit[1].fill_(-1000.0)
    temperature = 0.137
    observed = api.two_mode_event_energy(zero, learned, logit, temperature)
    ell = logit[:, 0]
    stacked = torch.stack(
        (
            F.logsigmoid(-ell) - zero / temperature,
            F.logsigmoid(ell) - learned / temperature,
        ),
        dim=0,
    )
    expected = -temperature * torch.logsumexp(stacked, dim=0)
    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)
    posterior = api.event_posterior_responsibility(
        zero, learned, logit, temperature
    )
    expected_log_odds = stacked[1] - stacked[0]
    torch.testing.assert_close(
        posterior, torch.sigmoid(expected_log_odds), rtol=0.0, atol=0.0
    )
    assert bool(torch.isfinite(observed).all())
    assert bool(torch.isfinite(posterior).all())

    arbitrary_logit = torch.randn(2, 1, 64, 64, generator=generator) * 20.0
    persistence = api.two_mode_event_energy(
        zero, zero.clone(), arbitrary_logit, temperature
    )
    torch.testing.assert_close(persistence, zero, atol=2e-6, rtol=2e-6)
    symmetric = api.event_posterior_responsibility(
        zero, zero.clone(), torch.zeros_like(arbitrary_logit), temperature
    )
    assert torch.equal(symmetric, torch.full_like(symmetric, 0.5))
    with pytest.raises(FloatingPointError, match="positive"):
        api.two_mode_event_energy(zero, learned, logit, 0.0)


def test_change_weight_balanced_reductions_and_matched_single_reference() -> None:
    generator = torch.Generator().manual_seed(710)
    persistence = torch.rand(2, 64, 64, generator=generator) + 0.01
    cell = torch.rand(2, 64, 64, generator=generator)
    temperature = 0.3
    weight = api.change_weight(persistence, temperature)
    expected_weight = persistence / (persistence + temperature)
    torch.testing.assert_close(weight, expected_weight, rtol=0.0, atol=0.0)
    reduced = api.changed_static_balanced_energy_per_row(cell, weight)
    expected_changed = (weight * cell).sum((-2, -1)) / weight.sum((-2, -1))
    expected_static = ((1.0 - weight) * cell).sum((-2, -1)) / (
        1.0 - weight
    ).sum((-2, -1))
    torch.testing.assert_close(reduced.changed, expected_changed, rtol=0.0, atol=0.0)
    torch.testing.assert_close(reduced.static, expected_static, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        reduced.balanced,
        0.5 * expected_changed + 0.5 * expected_static,
        rtol=0.0,
        atol=0.0,
    )
    all_cell = cell[:, None].expand(-1, 9, -1, -1).clone()
    all_reduced = api.changed_static_balanced_energy_per_row(all_cell, weight)
    assert tuple(all_reduced.balanced.shape) == (2, 9)
    assert torch.equal(
        all_reduced.balanced,
        reduced.balanced[:, None].expand_as(all_reduced.balanced),
    )
    with pytest.raises(FloatingPointError, match="denominator"):
        api.changed_static_balanced_energy_per_row(
            cell, torch.zeros_like(weight)
        )

    target = torch.randn(2, 64, 64, 64, generator=generator)
    prediction = api.EventDeltaPrediction(
        torch.randn(2, 64, 64, 64, generator=generator),
        torch.randn(2, 1, 64, 64, generator=generator),
    )
    observed = api.matched_single_mean_cell_energy(target, prediction)
    probability = torch.sigmoid(prediction.event_logit[:, 0])
    matched = probability[:, None] * prediction.mu_event
    expected = _manual_smooth_l1(target, matched).mean(dim=1)
    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)


def test_action_permutation_equivariance_uses_only_embedding_rows() -> None:
    predictor = _predictor().eval()
    with torch.no_grad():
        predictor.action_embedding.weight.copy_(
            torch.linspace(-0.4, 0.4, 9 * 16).reshape(9, 16)
        )
        predictor.event_logit_head.weight.fill_(1e-3)
    latent = torch.randn(
        1, 64, 64, 64, generator=torch.Generator().manual_seed(971)
    )
    original = _all_actions(predictor, latent)
    permutation = torch.tensor((4, 0, 8, 2, 6, 1, 7, 3, 5))
    permuted_predictor = copy.deepcopy(predictor)
    with torch.no_grad():
        permuted_predictor.action_embedding.weight.copy_(
            predictor.action_embedding.weight[permutation]
        )
    permuted = _all_actions(permuted_predictor, latent)
    torch.testing.assert_close(
        permuted.mu_event, original.mu_event[:, permutation], atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        permuted.event_logit,
        original.event_logit[:, permutation],
        atol=0.0,
        rtol=0.0,
    )
    actions = _one_hot([0, 4, 8])
    action_permuted = actions[:, permutation]
    repeated = latent.expand(3, -1, -1, -1)
    expected = predictor(repeated, actions)
    observed = permuted_predictor(repeated, action_permuted)
    torch.testing.assert_close(observed.mu_event, expected.mu_event, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        observed.event_logit, expected.event_logit, atol=0.0, rtol=0.0
    )


@pytest.mark.parametrize(
    ("latent", "action", "error", "message"),
    (
        (torch.zeros(0, 64, 64, 64), torch.zeros(0, 9), ValueError, "at least one"),
        (torch.zeros(1, 63, 64, 64), _one_hot([0]), ValueError, "shape"),
        (
            torch.zeros(1, 64, 64, 64, dtype=torch.float64),
            torch.eye(9, dtype=torch.float64)[:1],
            TypeError,
            "float32",
        ),
        (
            torch.full((1, 64, 64, 64), float("nan")),
            _one_hot([0]),
            FloatingPointError,
            "nonfinite",
        ),
        (torch.zeros(1, 64, 64, 64), torch.zeros(1, 9), ValueError, "exactly one"),
        (
            torch.zeros(1, 64, 64, 64),
            torch.full((1, 9), 1.0 / 9.0),
            ValueError,
            "zeros and ones",
        ),
    ),
)
def test_predictor_input_guards(
    latent: torch.Tensor,
    action: torch.Tensor,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        _predictor()(latent, action)


def test_dynamics_gradient_routes_reach_online_representation_and_every_predictor_submodule(
    n320_encoder_state: dict[str, torch.Tensor],
) -> None:
    model = api.GeometryAnchoredTwoModeEventDeltaJointJepaV1(
        n320_encoder_state
    ).train()
    generator = torch.Generator().manual_seed(1201)
    current_rgb = torch.rand(1, 3, 112, 112, generator=generator)
    next_rgb = torch.rand(1, 3, 112, 112, generator=generator)
    fixed_rgb = torch.rand(1, 3, 112, 112, generator=generator)
    online_current = api.normalize_latent_per_cell(model.encode_online(current_rgb))
    with torch.no_grad():
        target_current = api.normalize_latent_per_cell(model.encode_target(current_rgb))
        target_next = api.normalize_latent_per_cell(model.encode_target(next_rgb))
        fixed_context = api.normalize_latent_per_cell(
            model.encode_online(fixed_rgb).detach()
        )
    assert not fixed_context.requires_grad
    target_delta = (target_next - target_current).detach()
    action = _one_hot([5])
    prediction = model.predict(online_current, action)
    component = api.event_delta_cell_energies(target_delta, prediction)
    temperature = 0.15
    weight = api.change_weight(component.zero_event, temperature)
    mixed = api.two_mode_event_energy(
        component.zero_event,
        component.learned_event,
        prediction.event_logit,
        temperature,
    )
    main = api.changed_static_balanced_energy_per_row(mixed, weight).balanced.mean()
    main.backward()

    encoder_gradients = [
        value.grad for value in model.encoder.parameters() if value.grad is not None
    ]
    lift_gradients = [
        value.grad for value in model.bev_lift.parameters() if value.grad is not None
    ]
    assert encoder_gradients and lift_gradients
    assert sum(float(value.abs().sum()) for value in encoder_gradients) > 0.0
    assert sum(float(value.abs().sum()) for value in lift_gradients) > 0.0
    assert all(bool(torch.isfinite(value).all()) for value in encoder_gradients)
    assert all(bool(torch.isfinite(value).all()) for value in lift_gradients)

    predictor_modules = (
        model.predictor.action_embedding,
        model.predictor.input_projection,
        model.predictor.residual_blocks[0].conv1,
        model.predictor.residual_blocks[0].conv2,
        model.predictor.residual_blocks[1].conv1,
        model.predictor.residual_blocks[1].conv2,
        model.predictor.event_mean_head,
        model.predictor.event_logit_head,
    )
    for module in predictor_modules:
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        assert gradients
        assert all(bool(torch.isfinite(value).all()) for value in gradients)
        assert sum(float(value.abs().sum()) for value in gradients) > 0.0
    assert all(parameter.grad is None for parameter in model.semantic_head.parameters())
    assert all(
        parameter.grad is None
        for target_module in model.target_modules()
        for parameter in target_module.parameters()
    )

    # The same-command context control owns an autograd-enabled predictor
    # call, but its alternative online context is explicitly detached.  Prove
    # that this call alone reaches every predictor submodule and cannot route
    # a gradient back into either representation branch.
    model.zero_grad(set_to_none=True)
    fixed_prediction = model.predict(fixed_context, action)
    fixed_component = api.event_delta_cell_energies(
        target_delta, fixed_prediction
    )
    fixed_mixed = api.two_mode_event_energy(
        fixed_component.zero_event,
        fixed_component.learned_event,
        fixed_prediction.event_logit,
        temperature,
    )
    context = api.changed_static_balanced_energy_per_row(
        fixed_mixed, weight
    ).balanced.mean()
    context.backward()
    for module in predictor_modules:
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        assert gradients
        assert all(bool(torch.isfinite(value).all()) for value in gradients)
        assert sum(float(value.abs().sum()) for value in gradients) > 0.0
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert all(parameter.grad is None for parameter in model.bev_lift.parameters())
    assert all(parameter.grad is None for parameter in model.semantic_head.parameters())
    assert all(
        parameter.grad is None
        for target_module in model.target_modules()
        for parameter in target_module.parameters()
    )
