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
    CENTER_OFFSET_INDEX,
    HOLD_ACTION_INDEX,
    LATENT_DIM,
    NEIGHBOR_COUNT,
    NONCENTER_NEIGHBOR_COUNT,
    RESIDUAL_ALPHA,
    TOKEN_COUNT,
    TOKEN_SIDE,
    ActionIndexedLosses,
    ActionIndexedPredictions,
    ActionConditionedLocalCorrespondenceTransport,
    CorrespondenceTargets,
    CorrespondenceTerms,
    action_independent_trunk,
    action_indexed_energy_nll,
    centered_log_soft_cross_entropy,
    initialize_action_gate_rows,
    local_correspondence_targets,
    local_correspondence_terms,
    patch_whitening_terms,
    predict_action_conditioned_local_transports,
    relative_action_embeddings,
    requested_action_indices,
)
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


def test_transport_wrapper_is_zero_bias_free_rng_free_with_exact_neighbors() -> None:
    shared = LinearTokenProjector(LATENT_DIM)
    torch.manual_seed(812)
    rng_before = torch.random.get_rng_state().clone()

    wrapper = ActionConditionedLocalCorrespondenceTransport(shared)

    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert wrapper.shared_projector is shared
    assert tuple(wrapper.transport_weight.shape) == (
        NONCENTER_NEIGHBOR_COUNT,
        LATENT_DIM,
    )
    assert wrapper.transport_weight.numel() == 1536
    assert torch.count_nonzero(wrapper.transport_weight).item() == 0
    assert tuple(wrapper.neighbor_indices.shape) == (
        TOKEN_COUNT,
        NEIGHBOR_COUNT,
    )
    assert wrapper.neighbor_indices.dtype == torch.long
    assert "neighbor_indices" not in wrapper.state_dict()
    assert [name for name, _ in wrapper.named_parameters()] == [
        "transport_weight",
        "shared_projector.linear.weight",
        "shared_projector.linear.bias",
    ]
    assert not hasattr(wrapper, "bias")
    assert not hasattr(wrapper, "action_embed")
    assert not hasattr(wrapper, "flow_weight")
    assert not hasattr(wrapper, "inverse_weight")


def test_neighbor_table_has_exact_order_and_clamped_border_duplicates() -> None:
    wrapper = ActionConditionedLocalCorrespondenceTransport(
        LinearTokenProjector(LATENT_DIM)
    )
    expected_rows: list[list[int]] = []
    offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 0),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    for row in range(TOKEN_SIDE):
        for column in range(TOKEN_SIDE):
            expected_rows.append([
                TOKEN_SIDE * min(max(row + dy, 0), TOKEN_SIDE - 1)
                + min(max(column + dx, 0), TOKEN_SIDE - 1)
                for dy, dx in offsets
            ])
    expected = torch.tensor(expected_rows, dtype=torch.long)
    assert torch.equal(wrapper.neighbor_indices, expected)
    assert wrapper.neighbor_indices[0].tolist() == [
        0, 0, 1, 0, 0, 1, 16, 16, 17
    ]
    assert wrapper.neighbor_indices[17].tolist() == [
        0, 1, 2, 16, 17, 18, 32, 33, 34
    ]
    assert wrapper.neighbor_indices[-1].tolist() == [
        238, 239, 239, 254, 255, 255, 254, 255, 255
    ]


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


def _next_spatial_latents(current: torch.Tensor) -> torch.Tensor:
    feature_axis = torch.linspace(-0.2, 0.3, LATENT_DIM)
    token_axis = torch.linspace(-0.1, 0.15, TOKEN_COUNT)[:, None]
    return torch.stack([
        F.normalize(
            current[index]
            + float(index + 1) * token_axis * feature_axis[None],
            dim=-1,
        )
        for index in range(current.shape[0])
    ])


class _ForbiddenActionEmbed(nn.Module):
    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        raise AssertionError("shared trunk must bypass action_embed")


class _TransportBlock(nn.Module):
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


class _TransportPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.latent_dim = LATENT_DIM
        self.num_spatial_tokens = TOKEN_COUNT
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, TOKEN_COUNT, LATENT_DIM)
        )
        self.input_drop = nn.Identity()
        self.blocks = nn.ModuleList([_TransportBlock()])
        self.norm = nn.LayerNorm(LATENT_DIM)
        self.action_embed = ActionEmbedder(
            input_dim=ACTION_DIM,
            smoothed_dim=10,
            emb_dim=LATENT_DIM,
        )


def _spatial_fixture() -> tuple[
    _TransportPredictor,
    ActionConditionedLocalCorrespondenceTransport,
]:
    torch.manual_seed(101)
    predictor = _TransportPredictor()
    projector = ActionConditionedLocalCorrespondenceTransport(
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


def test_local_correspondence_target_matches_formula_and_detaches_ema() -> None:
    _, projector = _spatial_fixture()
    current = _spatial_latents(2).requires_grad_()
    next_state = _next_spatial_latents(current.detach()).requires_grad_()

    targets = local_correspondence_targets(
        projector,
        current,
        next_state,
    )
    normalized_current = F.layer_norm(
        current.detach(),
        (LATENT_DIM,),
        weight=None,
        bias=None,
        eps=1e-5,
    )
    normalized_next = F.layer_norm(
        next_state.detach(),
        (LATENT_DIM,),
        weight=None,
        bias=None,
        eps=1e-5,
    )
    expected_logits = (
        normalized_current[:, projector.neighbor_indices]
        * normalized_next[:, :, None]
    ).sum(dim=-1) / math.sqrt(LATENT_DIM)
    expected_probabilities = torch.softmax(expected_logits, dim=-1)
    expected_kl = (
        expected_probabilities
        * (expected_probabilities.log() + math.log(NEIGHBOR_COUNT))
    ).sum(dim=-1).mean()

    assert isinstance(targets, CorrespondenceTargets)
    assert torch.equal(targets.logits, expected_logits)
    assert torch.equal(targets.probabilities, expected_probabilities)
    assert torch.equal(targets.mean_kl_to_uniform, expected_kl)
    assert targets.mean_kl_to_uniform.item() > 0
    assert torch.all(targets.probabilities > 0)
    assert torch.allclose(
        targets.probabilities.sum(dim=-1),
        torch.ones(2, TOKEN_COUNT),
        atol=1e-6,
        rtol=0,
    )
    assert not targets.logits.requires_grad
    assert not targets.probabilities.requires_grad
    assert not targets.mean_kl_to_uniform.requires_grad
    assert current.grad is None
    assert next_state.grad is None


def test_centered_log_soft_cross_entropy_is_exact_and_broadcastable() -> None:
    torch.manual_seed(419)
    target_logits = torch.randn(2, TOKEN_COUNT, NEIGHBOR_COUNT)
    targets = torch.softmax(target_logits, dim=-1).requires_grad_()
    student_logits = torch.randn(
        2,
        ACTION_DIM,
        TOKEN_COUNT,
        NEIGHBOR_COUNT,
        requires_grad=True,
    )
    centered = centered_log_soft_cross_entropy(
        targets[:, None],
        student_logits,
    )
    expected = -(
        targets.detach()[:, None]
        * F.log_softmax(student_logits, dim=-1)
    ).sum(dim=-1)

    assert CENTER_OFFSET_INDEX == 4
    assert centered.shape == (2, ACTION_DIM, TOKEN_COUNT)
    assert torch.allclose(centered, expected, atol=2e-7, rtol=1e-6)

    uniform_logits = torch.zeros(2, TOKEN_COUNT, NEIGHBOR_COUNT)
    left = centered_log_soft_cross_entropy(
        targets.detach(),
        uniform_logits,
    )
    right_targets = torch.roll(targets.detach(), shifts=3, dims=-1)
    right = centered_log_soft_cross_entropy(
        right_targets,
        uniform_logits,
    )
    assert torch.equal(left, right)
    assert torch.equal(
        left,
        -F.log_softmax(uniform_logits, dim=-1)[..., CENTER_OFFSET_INDEX],
    )

    centered.mean().backward()
    assert targets.grad is None
    assert student_logits.grad is not None
    assert torch.count_nonzero(student_logits.grad).item() > 0


def test_zero_transport_is_exact_identity_uniform_and_gathers_exactly() -> None:
    predictor, projector = _spatial_fixture()
    state = torch.randn(2, TOKEN_COUNT, LATENT_DIM)
    ema_current = _spatial_latents(2)
    requested = _one_hot([0, HOLD_ACTION_INDEX])

    predictions = predict_action_conditioned_local_transports(
        predictor,
        projector,
        state,
        requested,
        ema_current,
    )
    uniform = torch.softmax(
        torch.zeros_like(predictions.all_transport_logits),
        dim=-1,
    )

    assert predictions.all_predictions.shape == (
        2, ACTION_DIM, TOKEN_COUNT, LATENT_DIM
    )
    assert predictions.all_transport_logits.shape == (
        2, ACTION_DIM, TOKEN_COUNT, NEIGHBOR_COUNT
    )
    assert predictions.all_transport_probabilities.shape == (
        2, ACTION_DIM, TOKEN_COUNT, NEIGHBOR_COUNT
    )
    assert predictions.all_expected_offsets.shape == (
        2, ACTION_DIM, TOKEN_COUNT, 2
    )
    assert predictions.all_transports.shape == (
        2, ACTION_DIM, TOKEN_COUNT, LATENT_DIM
    )
    assert torch.count_nonzero(predictions.all_transport_logits).item() == 0
    assert torch.equal(predictions.all_transport_probabilities, uniform)
    assert torch.count_nonzero(predictions.all_expected_offsets).item() == 0
    assert torch.equal(
        predictions.all_transports,
        ema_current[:, None].expand_as(predictions.all_transports),
    )
    assert predictions.executed.shape == (2, TOKEN_COUNT, LATENT_DIM)
    assert predictions.controls.shape == (
        2, ACTION_DIM - 1, TOKEN_COUNT, LATENT_DIM
    )
    for action in range(1, ACTION_DIM):
        assert torch.equal(
            predictions.all_predictions[:, 0],
            predictions.all_predictions[:, action],
        )
    assert torch.equal(
        predictions.executed,
        predictions.all_predictions[
            torch.arange(2), torch.tensor([0, HOLD_ACTION_INDEX])
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


def _install_deterministic_transport_weight(
    projector: ActionConditionedLocalCorrespondenceTransport,
) -> None:
    with torch.no_grad():
        projector.transport_weight.copy_(
            torch.linspace(
                -0.03,
                0.04,
                projector.transport_weight.numel(),
            ).reshape_as(projector.transport_weight)
        )


def test_active_transport_matches_centered_softmax_math_and_bounds() -> None:
    predictor, projector = _spatial_fixture()
    _install_deterministic_transport_weight(projector)
    state = torch.randn(2, TOKEN_COUNT, LATENT_DIM)
    ema_current = _spatial_latents(2)
    requested = _one_hot([2, 8])

    predictions = predict_action_conditioned_local_transports(
        predictor,
        projector,
        state,
        requested,
        ema_current,
    )
    hidden = action_independent_trunk(predictor, state)
    relative = relative_action_embeddings(
        predictor,
        device=state.device,
        dtype=state.dtype,
    )
    interactions = hidden[:, None] * relative[None, :, None]
    noncenter = F.linear(
        interactions,
        projector.transport_weight,
        bias=None,
    )
    expected_logits = torch.cat(
        (
            noncenter[..., :CENTER_OFFSET_INDEX],
            -noncenter.sum(dim=-1, keepdim=True),
            noncenter[..., CENTER_OFFSET_INDEX:],
        ),
        dim=-1,
    )
    expected_probabilities = torch.softmax(expected_logits, dim=-1)
    uniform = torch.softmax(torch.zeros_like(expected_logits), dim=-1)
    coefficients = expected_probabilities - uniform
    neighbor_delta = (
        ema_current[:, projector.neighbor_indices]
        - ema_current[:, :, None]
    )
    expected_transports = (
        ema_current[:, None]
        + torch.matmul(
            coefficients.permute(0, 2, 1, 3),
            neighbor_delta,
        ).permute(0, 2, 1, 3)
    )
    offsets = torch.tensor(
        [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 0), (0, 1),
            (1, -1), (1, 0), (1, 1),
        ],
        dtype=torch.float32,
    )
    expected_offsets = torch.matmul(coefficients, offsets)

    assert torch.allclose(
        predictions.all_transport_logits,
        expected_logits,
    )
    assert torch.allclose(
        predictions.all_transport_probabilities,
        expected_probabilities,
    )
    assert torch.allclose(predictions.all_transports, expected_transports)
    assert torch.allclose(
        predictions.all_expected_offsets,
        expected_offsets,
    )
    assert predictions.all_expected_offsets.abs().max().item() <= 1.0
    assert torch.allclose(
        predictions.all_transport_logits.sum(dim=-1),
        torch.zeros(2, ACTION_DIM, TOKEN_COUNT),
        atol=2e-7,
        rtol=0,
    )
    hold = predictions.all_transport_probabilities[:, HOLD_ACTION_INDEX]
    assert torch.equal(hold, uniform[:, HOLD_ACTION_INDEX])
    assert torch.count_nonzero(
        predictions.all_expected_offsets[:, HOLD_ACTION_INDEX]
    ).item() == 0
    assert torch.equal(
        predictions.all_transports[:, HOLD_ACTION_INDEX],
        ema_current,
    )
    for action in range(ACTION_DIM):
        if action != HOLD_ACTION_INDEX:
            assert not torch.equal(
                predictions.all_transport_probabilities[:, action],
                hold,
            )


def test_correspondence_terms_match_exact_executed_formula_and_scale_detaches() -> None:
    predictor, projector = _spatial_fixture()
    _install_deterministic_transport_weight(projector)
    current = _spatial_latents(3)
    targets = local_correspondence_targets(
        projector,
        current,
        _next_spatial_latents(current),
    )
    predictions = predict_action_conditioned_local_transports(
        predictor,
        projector,
        torch.randn(3, TOKEN_COUNT, LATENT_DIM),
        _one_hot([0, 4, 8]),
        current,
    )
    row_scale = torch.tensor(
        [0.15, 0.25, 0.40],
        requires_grad=True,
    )

    terms = local_correspondence_terms(
        targets,
        predictions,
        row_scale,
    )
    executed_logits = predictions.all_transport_logits[
        torch.arange(3),
        torch.tensor([0, 4, 8]),
    ]
    expected_token = centered_log_soft_cross_entropy(
        targets.probabilities,
        executed_logits,
    )
    expected_per_row = expected_token.mean(dim=1)

    assert isinstance(terms, CorrespondenceTerms)
    assert torch.equal(terms.cross_entropy_per_row, expected_per_row)
    assert torch.equal(
        terms.centered_cross_entropy,
        expected_per_row.mean(),
    )
    assert torch.equal(
        terms.loss,
        (row_scale.detach() * expected_per_row).mean(),
    )
    assert torch.autograd.grad(
        terms.loss,
        row_scale,
        allow_unused=True,
    )[0] is None


def test_zero_transport_opens_then_online_and_action_paths_become_live() -> None:
    predictor, projector = _spatial_fixture()
    current = _spatial_latents(2).requires_grad_()
    next_state = _next_spatial_latents(current.detach()).requires_grad_()
    online = torch.randn(
        2,
        TOKEN_COUNT,
        LATENT_DIM,
        requires_grad=True,
    )
    requested = _one_hot([0, 3])
    action_parameters = tuple(predictor.action_embed.parameters())
    targets = local_correspondence_targets(
        projector,
        current,
        next_state,
    )
    predictions = predict_action_conditioned_local_transports(
        predictor,
        projector,
        online,
        requested,
        current,
    )
    terms = local_correspondence_terms(
        targets,
        predictions,
        torch.tensor([0.2, 0.4]),
    )
    initial_gradients = torch.autograd.grad(
        terms.loss,
        (
            projector.transport_weight,
            online,
            *action_parameters,
            current,
            next_state,
        ),
        allow_unused=True,
    )
    transport_gradient = initial_gradients[0]

    assert transport_gradient is not None
    assert torch.isfinite(transport_gradient).all()
    assert torch.count_nonzero(transport_gradient).item() > 0
    assert all(
        gradient is None
        or (
            torch.isfinite(gradient).all()
            and torch.count_nonzero(gradient).item() == 0
        )
        for gradient in initial_gradients[1:]
    )

    with torch.no_grad():
        projector.transport_weight.copy_(
            -0.01
            * transport_gradient
            / transport_gradient.norm().clamp_min(1e-8)
        )
    active_online = online.detach().clone().requires_grad_()
    active_current = current.detach().clone().requires_grad_()
    active_next = next_state.detach().clone().requires_grad_()
    active_targets = local_correspondence_targets(
        projector,
        active_current,
        active_next,
    )
    active_predictions = predict_action_conditioned_local_transports(
        predictor,
        projector,
        active_online,
        requested,
        active_current,
    )
    active_terms = local_correspondence_terms(
        active_targets,
        active_predictions,
        torch.tensor([0.2, 0.4]),
    )
    active_gradients = torch.autograd.grad(
        active_terms.loss,
        (
            projector.transport_weight,
            active_online,
            *action_parameters,
            active_current,
            active_next,
        ),
        allow_unused=True,
    )

    assert active_gradients[0] is not None
    assert torch.count_nonzero(active_gradients[0]).item() > 0
    assert active_gradients[1] is not None
    assert torch.count_nonzero(active_gradients[1]).item() > 0
    assert any(
        gradient is not None
        and torch.count_nonzero(gradient).item() > 0
        for gradient in active_gradients[2:-2]
    )
    assert active_gradients[-2] is None
    assert active_gradients[-1] is None


def test_executed_online_path_is_live_while_wrong_online_path_is_detached() -> None:
    predictor, projector = _spatial_fixture()
    _install_deterministic_transport_weight(projector)
    online = torch.randn(
        1,
        TOKEN_COUNT,
        LATENT_DIM,
        requires_grad=True,
    )
    ema_current = _spatial_latents(1).requires_grad_()
    predictions = predict_action_conditioned_local_transports(
        predictor,
        projector,
        online,
        _one_hot([2]),
        ema_current,
    )
    action_parameters = tuple(predictor.action_embed.parameters())

    wrong_gradients = torch.autograd.grad(
        predictions.all_transport_logits[:, 4, :, 0].sum(),
        (
            online,
            projector.transport_weight,
            *action_parameters,
            ema_current,
        ),
        retain_graph=True,
        allow_unused=True,
    )
    assert wrong_gradients[0] is None or (
        torch.count_nonzero(wrong_gradients[0]).item() == 0
    )
    assert wrong_gradients[1] is not None
    assert torch.count_nonzero(wrong_gradients[1]).item() > 0
    assert any(
        gradient is not None
        and torch.count_nonzero(gradient).item() > 0
        for gradient in wrong_gradients[2:-1]
    )
    assert wrong_gradients[-1] is None

    executed_gradients = torch.autograd.grad(
        predictions.all_transport_logits[:, 2, :, 0].sum(),
        (
            online,
            projector.transport_weight,
            *action_parameters,
            ema_current,
        ),
        allow_unused=True,
    )
    assert executed_gradients[0] is not None
    assert torch.count_nonzero(executed_gradients[0]).item() > 0
    assert executed_gradients[1] is not None
    assert torch.count_nonzero(executed_gradients[1]).item() > 0
    assert any(
        gradient is not None
        and torch.count_nonzero(gradient).item() > 0
        for gradient in executed_gradients[2:-1]
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
        all_transport_logits=torch.zeros(
            batch, ACTION_DIM, 1, NEIGHBOR_COUNT
        ),
        all_transport_probabilities=torch.full(
            (batch, ACTION_DIM, 1, NEIGHBOR_COUNT),
            1.0 / NEIGHBOR_COUNT,
        ),
        all_expected_offsets=torch.zeros(batch, ACTION_DIM, 1, 2),
        all_transports=torch.zeros(
            batch, ACTION_DIM, 1, LATENT_DIM
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
