from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from lewm.models.go2_rgb_fixed_teacher_dual_domain_trajectory_h4_jepa_v1 import (
    DualDomainTrajectoryH4JEPA,
    DualDomainTrajectoryH4JEPAConfig,
    LocalInnovationTrajectoryH4JEPAOutput,
)
from lewm.models.go2_rgb_fixed_teacher_local_innovation_trajectory_h4_jepa_v1 import (
    LocalInnovationTrajectoryH4JEPA,
    LocalInnovationTrajectoryH4JEPAConfig,
    fixed_teacher_local_innovations,
    trajectory_energy_score,
)
from scripts.run_go2_recurrent_h4_joint_jepa_v1 import _parameter_groups


def _model() -> DualDomainTrajectoryH4JEPA:
    torch.manual_seed(101)
    return DualDomainTrajectoryH4JEPA(
        config=DualDomainTrajectoryH4JEPAConfig(
            image_size=8,
            patch_size=4,
            feature_dim=12,
            encoder_depth=1,
            encoder_heads=3,
            recurrent_spatial_heads=3,
            cross_attention_heads=3,
        )
    )


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(102)
    history = torch.randn(2, 3, 3, 8, 8, generator=generator)
    future_rgb = torch.randn(2, 4, 3, 8, 8, generator=generator)
    past = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    future = torch.tensor([[4, 5, 6, 7], [8, 0, 1, 2]], dtype=torch.long)
    return history, past, future, future_rgb


def test_config_and_architecture_are_exactly_inherited() -> None:
    config = DualDomainTrajectoryH4JEPAConfig()
    assert config.local_innovation_score_weight == 0.5
    assert config.cumulative_trajectory_score_weight == 0.5
    assert config.cyclic_wrong_action_margin == 0.05
    assert config.history_margin == 0.03
    with pytest.raises(ValueError):
        DualDomainTrajectoryH4JEPAConfig(local_innovation_score_weight=0.6)

    model = _model().eval()
    history, past, future, _future_rgb = _batch()
    output = model(history, past, future)
    assert isinstance(output, LocalInnovationTrajectoryH4JEPAOutput)
    current = F.normalize(output.history_latents[:, 2], dim=-1)
    expected = current[:, None, None].expand_as(output.trajectory_latents)
    torch.testing.assert_close(
        output.trajectory_latents,
        expected,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        output.trajectory_innovations,
        torch.zeros_like(output.trajectory_innovations),
        rtol=0.0,
        atol=0.0,
    )


def test_auxiliary_fit_is_exact_half_local_plus_half_cumulative() -> None:
    model = _model().train()
    history, past, future, future_rgb = _batch()
    output = model(history, past, future)
    target = model.encode_target(future_rgb)
    losses = model.training_auxiliary_losses(
        history_rgb=history,
        past_actions=past,
        future_actions=future,
        target_latents=target,
        output=output,
    )
    assert set(losses) == {
        "history_teacher_alignment",
        "half_future_teacher_local_innovation_energy_score",
        "half_future_teacher_cumulative_trajectory_energy_score",
        "dual_domain_cyclic_wrong_action_score_ranking",
        "dual_domain_history_counterfactual_score_ranking",
    }

    teacher_history = model._encode_fixed_teacher_history(history)
    target_innovations = fixed_teacher_local_innovations(
        teacher_history[:, 2],
        target,
    )
    _local_horizon, _local_joint, local = trajectory_energy_score(
        output.trajectory_innovations,
        target_innovations,
    )
    _cumulative_horizon, _cumulative_joint, cumulative = trajectory_energy_score(
        output.trajectory_latents,
        target,
    )
    torch.testing.assert_close(
        losses["half_future_teacher_local_innovation_energy_score"],
        0.5 * local.mean(),
    )
    torch.testing.assert_close(
        losses["half_future_teacher_cumulative_trajectory_energy_score"],
        0.5 * cumulative.mean(),
    )
    torch.testing.assert_close(
        losses["dual_domain_cyclic_wrong_action_score_ranking"],
        torch.tensor(0.05),
    )
    torch.testing.assert_close(
        losses["dual_domain_history_counterfactual_score_ranking"],
        torch.tensor(0.03),
    )


def test_each_domain_opens_head_gradient_and_target_remains_fixed() -> None:
    model = _model().train()
    history, past, future, future_rgb = _batch()
    output = model(history, past, future)
    losses = model.training_auxiliary_losses(
        history_rgb=history,
        past_actions=past,
        future_actions=future,
        target_latents=model.encode_target(future_rgb),
        output=output,
    )
    final = model.prediction_projector[-1]
    for name in (
        "half_future_teacher_local_innovation_energy_score",
        "half_future_teacher_cumulative_trajectory_energy_score",
        "dual_domain_cyclic_wrong_action_score_ranking",
    ):
        gradient = torch.autograd.grad(
            losses[name],
            final.weight,
            retain_graph=True,
        )[0]
        assert bool(torch.isfinite(gradient).all())
        assert float(gradient.norm()) > 0.0
    sum(losses.values()).backward()
    assert final.weight.grad is not None
    assert float(final.weight.grad.norm()) > 0.0
    assert all(
        parameter.grad is None for parameter in model.target_encoder.parameters()
    )


def test_counterfactual_hinges_use_the_complete_mixed_score() -> None:
    model = _model().train()
    with torch.no_grad():
        model.encoder.patch_embed.weight.add_(0.03)
        model.prediction_projector[-1].weight.normal_(std=0.02)
        model.prediction_projector[-1].bias.zero_()
    history, past, future, future_rgb = _batch()
    output = model(history, past, future)
    target = model.encode_target(future_rgb)
    losses = model.training_auxiliary_losses(
        history_rgb=history,
        past_actions=past,
        future_actions=future,
        target_latents=target,
        output=output,
    )
    teacher_history = model._encode_fixed_teacher_history(history)
    target_innovations = fixed_teacher_local_innovations(
        teacher_history[:, 2],
        target,
    )
    normalized_e2 = F.normalize(output.history_latents[:, 2], dim=-1)

    def mixed(atoms: torch.Tensor) -> torch.Tensor:
        anchor = normalized_e2[:, None, None].expand(
            -1,
            atoms.shape[1],
            1,
            -1,
            -1,
        )
        innovations = torch.cat(
            (anchor, atoms),
            dim=2,
        ).diff(dim=2)
        _lh, _lj, local = trajectory_energy_score(
            innovations,
            target_innovations,
        )
        _ch, _cj, cumulative = trajectory_energy_score(atoms, target)
        return 0.5 * local + 0.5 * cumulative

    real = mixed(output.trajectory_latents)
    persistence_atoms = teacher_history[:, 2, None, None].expand_as(
        output.trajectory_latents
    )
    _plh, _plj, persistence_local = trajectory_energy_score(
        torch.zeros_like(output.trajectory_innovations),
        target_innovations,
    )
    _pch, _pcj, persistence_cumulative = trajectory_energy_score(
        persistence_atoms,
        target,
    )
    denominator = (0.5 * persistence_local + 0.5 * persistence_cumulative).detach()
    denominator = denominator.clamp_min(model.config.normalization_epsilon)

    wrong_atoms = model.predict_trajectory_atoms_from_belief(
        output.belief_latents,
        (future + 1) % model.config.action_count,
    )
    expected_action = F.relu(0.05 + real / denominator - mixed(wrong_atoms) / denominator)
    expected_action = expected_action.mean()

    reversed_belief = model._belief_from_encoded_history(
        output.history_latents[:, [1, 0, 2]],
        past.flip(dims=(1,)),
    )
    reset_history = output.history_latents[:, 2:3].expand(-1, 3, -1, -1)
    hold_index = model.action_vocabulary.index("hold")
    reset_belief = model._belief_from_encoded_history(
        reset_history,
        torch.full_like(past, hold_index),
    )
    reversed_atoms = model.predict_trajectory_atoms_from_belief(
        reversed_belief,
        future,
    )
    reset_atoms = model.predict_trajectory_atoms_from_belief(
        reset_belief,
        future,
    )
    expected_history = F.relu(
        0.03
        + real / denominator
        - torch.minimum(mixed(reversed_atoms), mixed(reset_atoms)) / denominator
    ).mean()
    torch.testing.assert_close(
        losses["dual_domain_cyclic_wrong_action_score_ranking"],
        expected_action,
    )
    torch.testing.assert_close(
        losses["dual_domain_history_counterfactual_score_ranking"],
        expected_history,
    )


def test_zero_target_change_rows_are_clamped_not_filtered() -> None:
    model = _model().train()
    history, past, future, _future_rgb = _batch()
    output = model(history, past, future)
    teacher_history = model._encode_fixed_teacher_history(history)
    target = teacher_history[:, 2:3].expand(-1, 4, -1, -1).contiguous()
    losses = model.training_auxiliary_losses(
        history_rgb=history,
        past_actions=past,
        future_actions=future,
        target_latents=target,
        output=output,
    )
    torch.testing.assert_close(
        losses["half_future_teacher_local_innovation_energy_score"],
        torch.zeros_like(
            losses["half_future_teacher_local_innovation_energy_score"]
        ),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        losses["half_future_teacher_cumulative_trajectory_energy_score"],
        torch.zeros_like(
            losses["half_future_teacher_cumulative_trajectory_energy_score"]
        ),
        rtol=0.0,
        atol=1e-7,
    )
    torch.testing.assert_close(
        losses["dual_domain_cyclic_wrong_action_score_ranking"],
        torch.tensor(0.05),
    )
    torch.testing.assert_close(
        losses["dual_domain_history_counterfactual_score_ranking"],
        torch.tensor(0.03),
    )
    assert all(bool(torch.isfinite(value)) for value in losses.values())


def test_history_ranking_reaches_history_and_uses_no_new_parameters() -> None:
    model = _model().train()
    with torch.no_grad():
        model.prediction_projector[-1].weight.normal_(std=1e-4)
        model.prediction_projector[-1].bias.zero_()
    history, past, future, future_rgb = _batch()
    output = model(history, past, future)
    losses = model.training_auxiliary_losses(
        history_rgb=history,
        past_actions=past,
        future_actions=future,
        target_latents=model.encode_target(future_rgb),
        output=output,
    )
    ranking = losses["dual_domain_history_counterfactual_score_ranking"]
    assert float(ranking.detach()) > 0.0
    ranking.backward()
    gradients = (
        model.initial_belief.time_embedding.weight.grad,
        model.initial_belief.encoder[0].self_attn.in_proj_weight.grad,
        model.future_cell.decoder[0].multihead_attn.in_proj_weight.grad,
        model.prediction_projector[-1].weight.grad,
    )
    assert all(value is not None for value in gradients)
    assert all(
        bool(torch.isfinite(value).all()) and float(value.norm()) > 0.0
        for value in gradients
        if value is not None
    )

    fresh = _model()
    local = LocalInnovationTrajectoryH4JEPA(
        config=LocalInnovationTrajectoryH4JEPAConfig(
            image_size=8,
            patch_size=4,
            feature_dim=12,
            encoder_depth=1,
            encoder_heads=3,
            recurrent_spatial_heads=3,
            cross_attention_heads=3,
        )
    )
    dual_inventory = {
        name: tuple(parameter.shape)
        for name, parameter in fresh.named_parameters()
        if parameter.requires_grad
    }
    local_inventory = {
        name: tuple(parameter.shape)
        for name, parameter in local.named_parameters()
        if parameter.requires_grad
    }
    assert dual_inventory == local_inventory
    groups = _parameter_groups(fresh)
    assert set(groups) == {"encoder", "history", "predictor"}
    assert {id(parameter) for parameter in fresh.parameters() if parameter.requires_grad} == {
        id(parameter)
        for parameters in groups.values()
        for parameter in parameters
    }
