from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from lewm.safety import plan_aware_monotone_jepa_cost_v1 as subject
from lewm.safety import plan_aware_monotone_jepa_cost_v1_contract as contract
from scripts import evaluate_plan_aware_monotone_jepa_cost_v1 as evaluator


def _pairwise_targets(utilities: torch.Tensor) -> torch.Tensor:
    return torch.sign(utilities.unsqueeze(-1) - utilities.unsqueeze(-2))


def _components(leading: tuple[int, ...] = (2, 3)) -> dict[str, torch.Tensor]:
    count = math.prod(leading)

    def values(width: int, offset: float) -> torch.Tensor:
        return torch.arange(count * width, dtype=torch.float32).reshape(
            *leading, width
        ) + offset

    return {
        "waypoint_features": values(6, 10.0),
        "route_role_one_hot": values(3, 20.0),
        "requested_action_plan": values(45, 30.0).reshape(*leading, 3, 5, 3),
        "applied_action_plan": values(45, 40.0).reshape(*leading, 3, 5, 3),
        "previous_command": values(3, 50.0),
        "observed_control_history": values(30, 60.0).reshape(*leading, 3, 5, 2),
        "deterministic_kinematic_outcome": values(6, 70.0),
    }


def test_exact_feature_contract_and_active_query_channels() -> None:
    pieces = _components()
    base, query = subject.assemble_base_and_query_features(**pieces)
    assert base.shape == (2, 3, subject.BASE_FEATURE_DIM)
    assert query.shape == (2, 3, subject.QUERY_FEATURE_DIM)

    for name, value in subject.BASE_FEATURE_SLICES.items():
        assert value.stop is not None and value.start is not None
        assert value.stop > value.start, name
    assert list(subject.BASE_FEATURE_SLICES.values())[0].start == 0
    assert list(subject.BASE_FEATURE_SLICES.values())[-1].stop == 138
    assert list(subject.QUERY_FEATURE_SLICES.values())[0].start == 0
    assert list(subject.QUERY_FEATURE_SLICES.values())[-1].stop == 71

    base_applied = base[..., subject.BASE_FEATURE_SLICES["applied_action_plan_3x5x3"]]
    assert torch.equal(base_applied, pieces["applied_action_plan"].flatten(-3))
    query_applied = query[
        ...,
        subject.QUERY_FEATURE_SLICES[
            "applied_action_plan_active_vx_yaw_rate_3x5x2"
        ],
    ]
    assert torch.equal(
        query_applied,
        pieces["applied_action_plan"][..., (0, 2)].flatten(-3),
    )
    query_previous = query[
        ..., subject.QUERY_FEATURE_SLICES["previous_command_active_vx_yaw_rate"]
    ]
    assert torch.equal(query_previous, pieces["previous_command"][..., (0, 2)])


def test_feature_assembly_rejects_shape_dtype_and_device_drift() -> None:
    pieces = _components((1,))
    pieces["previous_command"] = torch.zeros(1, 2)
    with pytest.raises(subject.PlanAwareCostContractError, match="previous_command"):
        subject.assemble_base_and_query_features(**pieces)

    pieces = _components((1,))
    pieces["route_role_one_hot"] = pieces["route_role_one_hot"].double()
    with pytest.raises(subject.PlanAwareCostContractError, match="dtype and device"):
        subject.assemble_base_and_query_features(**pieces)


def test_parameter_counts_caps_and_matched_deterministic_initialisation() -> None:
    no_latent, jepa = subject.build_matched_rankers()
    again_no_latent, again_jepa = subject.build_matched_rankers()

    assert subject.parameter_count(no_latent) == subject.NO_LATENT_PARAMETER_COUNT == 26_113
    assert subject.parameter_count(jepa) == subject.JEPA_LATENT_PARAMETER_COUNT == 232_514
    assert subject.parameter_count(no_latent) < subject.NO_LATENT_PARAMETER_CAP_EXCLUSIVE == 250_000
    assert subject.parameter_count(jepa) < subject.JEPA_LATENT_PARAMETER_CAP_EXCLUSIVE == 500_000
    assert subject.matched_base_initialisation(no_latent, jepa)
    subject.assert_matched_initialisation(no_latent, jepa)
    assert subject.shared_base_digest(no_latent) == subject.shared_base_digest(jepa)
    assert all(
        torch.equal(no_latent.state_dict()[name], again_no_latent.state_dict()[name])
        for name in no_latent.state_dict()
    )
    assert all(
        torch.equal(jepa.state_dict()[name], again_jepa.state_dict()[name])
        for name in jepa.state_dict()
    )
    assert not any(isinstance(module, torch.nn.Dropout) for module in jepa.modules())


def test_condition_keys_are_exactly_derived_and_shared_base_is_separate() -> None:
    keys = subject.condition_seed_keys(subject.MODEL_SEED)
    assert set(keys) == {
        subject.SHARED_BASE_KEY_ID,
        subject.KINEMATIC_PLUS_NO_LATENT_RESIDUAL,
        subject.KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL,
    }
    assert all(len(value) == 64 for value in keys.values())
    assert len(set(keys.values())) == 3
    no_latent, jepa = subject.build_matched_rankers(subject.MODEL_SEED)
    assert no_latent.shared_base_seed_key == jepa.shared_base_seed_key
    assert no_latent.shared_base_seed_key == keys[subject.SHARED_BASE_KEY_ID]
    assert no_latent.condition_seed_key == keys[subject.KINEMATIC_PLUS_NO_LATENT_RESIDUAL]
    assert jepa.condition_seed_key == keys[subject.KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL]
    assert subject.model_contract()["initialisation"]["condition_seed_keys"] == keys
    assert keys[subject.SHARED_BASE_KEY_ID] == contract.CONDITION_KEYED_SEEDS[
        "shared_base_subkey"
    ]["keyed_seed_sha256"]
    for condition in (
        subject.KINEMATIC_PLUS_NO_LATENT_RESIDUAL,
        subject.KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL,
    ):
        assert keys[condition] == contract.CONDITION_KEYED_SEEDS["condition_keys"][
            condition
        ]["keyed_seed_sha256"]


def test_model_construction_does_not_advance_callers_cpu_rng() -> None:
    torch.manual_seed(1234)
    expected = torch.rand(5)
    torch.manual_seed(1234)
    subject.build_matched_rankers()
    observed = torch.rand(5)
    assert torch.equal(observed, expected)


def test_initial_zero_residuals_reproduce_kinematic_anchor_exactly() -> None:
    no_latent, jepa = subject.build_matched_rankers()
    base = torch.randn(2, subject.BASE_FEATURE_DIM)
    query = torch.randn(2, subject.QUERY_FEATURE_DIM)
    tokens = torch.randn(
        2,
        len(subject.TIMEPOINTS),
        subject.TOKENS_PER_TIMEPOINT,
        subject.TOKEN_DIM,
    )
    anchor = torch.tensor([0.0, -3.0])
    with torch.no_grad():
        assert torch.equal(no_latent(base, anchor), anchor)
        components = jepa.score_components(
            base, query, anchor, tokens[0, 0], tokens[:, 1:]
        )
    assert torch.equal(components.base_residual, torch.zeros_like(anchor))
    assert torch.equal(components.latent_residual, torch.zeros_like(anchor))
    assert torch.equal(components.score, anchor)
    assert components.attention_weights.shape == (
        2,
        len(subject.TIMEPOINTS),
        subject.TOKENS_PER_TIMEPOINT,
    )
    assert torch.allclose(
        components.attention_weights.sum(dim=-1),
        torch.ones(2, len(subject.TIMEPOINTS)),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_zero_latent_branch_reproduces_its_own_kinematic_plus_base_path() -> None:
    _, jepa = subject.build_matched_rankers()
    with torch.no_grad():
        jepa.base_residual.layers[-1].bias.fill_(0.25)
        jepa.latent_residual_mlp[-1].weight.zero_()
        jepa.latent_residual_mlp[-1].bias.zero_()
    base = torch.randn(1, subject.BASE_FEATURE_DIM)
    query = torch.randn(1, subject.QUERY_FEATURE_DIM)
    tokens = torch.randn(
        1,
        len(subject.TIMEPOINTS),
        subject.TOKENS_PER_TIMEPOINT,
        subject.TOKEN_DIM,
    )
    anchor = torch.tensor([-2.0])
    with torch.no_grad():
        components = jepa.score_components(
            base, query, anchor, tokens[0, 0], tokens[:, 1:]
        )
        direct_base_path = jepa.kinematic_plus_base_score(base, anchor)
    assert torch.equal(components.latent_residual, torch.zeros_like(anchor))
    assert torch.equal(components.score, components.kinematic_plus_base_score)
    assert torch.equal(components.kinematic_plus_base_score, direct_base_path)
    assert torch.equal(components.kinematic_plus_base_score, anchor + 0.25)


def test_latent_branch_uses_one_shared_projection_and_scaled_dot_attention() -> None:
    _, jepa = subject.build_matched_rankers()
    assert isinstance(jepa.token_layer_norm, torch.nn.LayerNorm)
    assert jepa.token_layer_norm.normalized_shape == (subject.TOKEN_DIM,)
    assert jepa.shared_token_projection.in_features == subject.TOKEN_DIM
    assert jepa.shared_token_projection.out_features == subject.LATENT_WIDTH
    assert jepa.query_projection.in_features == subject.QUERY_FEATURE_DIM
    assert jepa.query_projection.out_features == subject.LATENT_WIDTH
    assert sum(
        isinstance(module, torch.nn.Linear)
        and module.in_features == subject.TOKEN_DIM
        and module.out_features == subject.LATENT_WIDTH
        for module in jepa.modules()
    ) == 1
    assert jepa.latent_residual_mlp[0].in_features == 4 * 64 + 138
    assert jepa.latent_residual_mlp[0].out_features == 256
    assert isinstance(jepa.latent_residual_mlp[1], torch.nn.GELU)
    assert jepa.latent_residual_mlp[2].in_features == 256
    assert jepa.latent_residual_mlp[2].out_features == 128
    assert isinstance(jepa.latent_residual_mlp[3], torch.nn.GELU)
    assert jepa.latent_residual_mlp[4].in_features == 128
    assert jepa.latent_residual_mlp[4].out_features == 1


def test_token_grid_view_preserves_known_y_times_32_plus_x_indices() -> None:
    flat_indices = torch.arange(
        subject.TOKENS_PER_TIMEPOINT, dtype=torch.float32
    ).reshape(subject.TOKENS_PER_TIMEPOINT, 1)
    tokens = flat_indices.expand(subject.TOKENS_PER_TIMEPOINT, subject.TOKEN_DIM)
    grid = subject.reshape_tokens_to_spatial_grid(tokens)
    assert subject.TOKEN_GRID_SHAPE == (24, 32)
    assert grid.shape == (24, 32, subject.TOKEN_DIM)
    for y, x in ((0, 0), (0, 31), (7, 11), (23, 31)):
        assert grid[y, x, 0].item() == y * 32 + x
        assert grid[y, x, -1].item() == y * 32 + x
    flattened = subject.flatten_spatial_token_grid(grid)
    assert torch.equal(flattened, tokens)


def test_machine_readable_model_contract_matches_live_architecture() -> None:
    contract = subject.model_contract()
    assert contract["schema"] == "plan_aware_monotone_jepa_cost_v1.model_contract.v1"
    assert contract["goal_tokens_used"] is False
    assert contract["base_features"]["dim"] == 138
    assert contract["query_features"]["dim"] == 71
    no_latent = contract["models"][subject.KINEMATIC_PLUS_NO_LATENT_RESIDUAL]
    latent = contract["models"][subject.KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL]
    assert no_latent["trainable_parameters"] == 26_113
    assert latent["trainable_parameters"] == 232_514
    assert latent["latent_residual_mlp"] == [394, 256, 128, 1]
    assert latent["attention"]["timepoints"] == ["CURRENT", "H1", "H2", "H3"]
    assert latent["attention"]["token_grid_shape"] == [24, 32]
    assert latent["attention"]["token_grid_storage_order"] == "row-major"
    assert latent["attention"]["token_flat_index"] == "y * 32 + x"
    assert latent["attention"]["token_grid_transform"] == (
        "view [...,768,1024] as [...,24,32,1024], then flatten the spatial "
        "grid in the same row-major order before shared LayerNorm/projection/attention"
    )
    selected = subject.model_contract(subject.KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL)
    assert selected["selected_model"] == subject.KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL


def test_latent_interface_rejects_wrong_timepoint_or_token_shape() -> None:
    _, jepa = subject.build_matched_rankers()
    base = torch.zeros(1, subject.BASE_FEATURE_DIM)
    query = torch.zeros(1, subject.QUERY_FEATURE_DIM)
    anchor = torch.zeros(1)
    current = torch.zeros(subject.TOKENS_PER_TIMEPOINT, subject.TOKEN_DIM)
    wrong = torch.zeros(1, 2, subject.TOKENS_PER_TIMEPOINT, subject.TOKEN_DIM)
    with pytest.raises(subject.PlanAwareCostContractError, match="future_tokens"):
        jepa(base, query, anchor, current, wrong)


def test_kinematic_rank_score_orientation() -> None:
    rank_costs = torch.tensor([[0.0, 1.0, 11.0]])
    assert torch.equal(
        subject.kinematic_rank_scores(rank_costs),
        torch.tensor([[-0.0, -1.0, -11.0]]),
    )


def test_route_only_loss_matches_registered_formula_and_prefers_correct_order() -> None:
    utilities = torch.tensor([[1.0, 0.5, 0.0]], dtype=torch.float64)
    anchor = torch.zeros_like(utilities)
    aligned = torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float64, requires_grad=True)
    reversed_scores = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float64)

    result = subject.route_only_loss(
        scores=aligned,
        route_utilities=utilities,
        kinematic_anchor=anchor,
        pairwise_targets=_pairwise_targets(utilities),
    )
    reversed_result = subject.route_only_loss(
        scores=reversed_scores,
        route_utilities=utilities,
        kinematic_anchor=anchor,
        pairwise_targets=_pairwise_targets(utilities),
    )
    manual_pairwise = torch.stack(
        [
            F.softplus(torch.tensor(-1.0)),
            F.softplus(torch.tensor(-2.0)),
            F.softplus(torch.tensor(-1.0)),
        ]
    ).double().mean()
    target = torch.softmax(utilities, dim=-1)
    manual_listwise = -(target * torch.log_softmax(aligned, dim=-1)).sum(dim=-1).mean()
    manual_residual = aligned.square().mean()
    assert result.ordered_pair_count == 3
    assert result.candidate_count == 3
    assert torch.allclose(result.pairwise, manual_pairwise, atol=1.0e-7, rtol=0.0)
    assert torch.allclose(result.listwise, manual_listwise, atol=1.0e-12, rtol=0.0)
    assert torch.allclose(result.residual_l2, manual_residual, atol=1.0e-12, rtol=0.0)
    assert torch.allclose(
        result.total,
        result.pairwise + 0.5 * result.listwise + 1.0e-3 * result.residual_l2,
        atol=1.0e-12,
        rtol=0.0,
    )
    assert result.total < reversed_result.total
    alias = subject.route_ordering_loss(
        aligned, utilities, anchor, _pairwise_targets(utilities)
    )
    assert torch.equal(alias["loss"], result.total)
    assert torch.equal(alias["pair"], result.pairwise)
    assert torch.equal(alias["list"], result.listwise)
    assert torch.equal(alias["residual"], result.residual_l2)
    result.total.backward()
    assert aligned.grad is not None and torch.isfinite(aligned.grad).all()


def test_route_only_loss_mask_and_tied_pair_handling() -> None:
    scores = torch.tensor([[0.5, -2.0, 0.5], [1.0, 2.0, 3.0]])
    utilities = torch.tensor([[1.0, 100.0, 1.0], [0.0, 0.0, 0.0]])
    anchor = scores.clone()
    mask = torch.tensor([[True, False, True], [True, True, True]])
    result = subject.route_only_loss(
        scores=scores,
        route_utilities=utilities,
        kinematic_anchor=anchor,
        pairwise_targets=torch.zeros(2, 3, 3),
        admissible_mask=mask,
    )
    assert result.ordered_pair_count == 0
    assert result.candidate_count == 5
    assert result.pairwise.item() == 0.0
    assert result.residual_l2.item() == 0.0
    assert torch.isfinite(result.total)


def test_route_only_loss_rejects_empty_state_population() -> None:
    scores = torch.zeros(2, 3)
    mask = torch.tensor([[True, False, False], [False, False, False]])
    with pytest.raises(subject.PlanAwareCostContractError, match="at least one"):
        subject.route_only_loss(
            scores=scores,
            route_utilities=scores,
            kinematic_anchor=scores,
            pairwise_targets=torch.zeros(2, 3, 3),
            admissible_mask=mask,
        )


def test_pairwise_target_must_follow_conditioned_borda_utility_difference() -> None:
    scores = torch.tensor([[2.0, 1.0, 0.0]])
    utilities = torch.tensor([[1.0, 0.5, 0.0]])
    targets = _pairwise_targets(utilities)
    targets[0, 0, 1] = 0.0
    targets[0, 1, 0] = 0.0
    with pytest.raises(subject.PlanAwareCostContractError, match="margin-Borda"):
        subject.route_only_loss(
            scores=scores,
            route_utilities=utilities,
            kinematic_anchor=torch.zeros_like(scores),
            pairwise_targets=targets,
        )


def test_pairwise_target_contract_rejects_nonantisymmetric_matrix() -> None:
    values = torch.zeros(1, 3)
    targets = torch.zeros(1, 3, 3)
    targets[0, 0, 1] = 1.0
    with pytest.raises(subject.PlanAwareCostContractError, match="antisymmetric"):
        subject.route_only_loss(
            scores=values,
            route_utilities=values,
            kinematic_anchor=values,
            pairwise_targets=targets,
        )


def test_evaluator_payload_orders_different_borda_utility_despite_direct_tie() -> None:
    # A~B and A~C under the 0.03 m primitive margin, while B>C. B therefore has
    # larger conditioned Borda utility and the Section-12 pairwise target orders it.
    rows = []
    for candidate, progress in enumerate((0.0, 0.02, -0.02)):
        rows.append(
            {
                "state_id": "synthetic-0",
                "candidate_index": candidate,
                "oracle_viability_admissible": True,
                "completed": False,
                "p_d": progress,
                "p_theta": 0.0,
                "base_features": np.zeros(subject.BASE_FEATURE_DIM, np.float32),
                "query_features": np.zeros(subject.QUERY_FEATURE_DIM, np.float32),
                "kinematic_anchor": -float(candidate),
            }
        )
    payload = evaluator.state_training_payload(rows)
    assert payload["pairwise_targets"][0, 1] == -1.0
    assert payload["utility"][0] != payload["utility"][1]
    assert np.array_equal(
        payload["pairwise_targets"], -payload["pairwise_targets"].T
    )
