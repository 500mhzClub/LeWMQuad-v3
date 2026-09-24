from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lewm.benchmarks import go2_dinov2_physical_readout_calibration_v1 as calibration
from lewm.benchmarks import go2_grounded_dense_dino_joint_jepa_v1 as subject
from lewm.benchmarks import go2_matched_branch_physical_outcome_screen_v1 as prior


def test_statistics_round_trip_and_identity_validation() -> None:
    inputs = torch.arange(4 * 9 * 12, dtype=torch.float32).reshape(4, 9, 12)
    targets = torch.arange(4 * 9 * 4, dtype=torch.float32).reshape(4, 9, 4) / 10.0
    input_stats = subject.fit_input_statistics_v1(inputs)
    outcome_stats = subject.fit_outcome_statistics_v1(targets)

    subject.validate_input_statistics_v1(input_stats)
    subject.validate_outcome_statistics_v1(outcome_stats)
    normalized = subject.normalize_physical_inputs_v1(inputs, input_stats)
    assert torch.allclose(normalized.mean(dim=(0, 1)), torch.zeros(12), atol=1.0e-6)
    residuals = subject.standardize_outcome_residuals_v1(targets, outcome_stats)
    decoded = subject.decode_standardized_outcomes_v1(residuals, outcome_stats)
    assert torch.allclose(decoded, targets, atol=1.0e-6)

    damaged = dict(input_stats)
    damaged["mean"] = input_stats["mean"].clone()
    damaged["mean"][0] += 1.0
    with pytest.raises(subject.GroundedDenseDINOJointJEPAError, match="identity"):
        subject.validate_input_statistics_v1(damaged)


def test_action_permutation_preserves_outcome_standardize_decode() -> None:
    generator = torch.Generator().manual_seed(7)
    targets = torch.randn((5, 9, 4), generator=generator)
    stats = subject.fit_outcome_statistics_v1(targets)
    canonical = subject.standardize_outcome_residuals_v1(targets, stats)
    permutation = torch.tensor([8, 0, 6, 2, 7, 4, 1, 5, 3])
    permuted_targets = targets[:, permutation]
    action_ids = permutation.unsqueeze(0).expand(targets.shape[0], -1)
    permuted = subject.standardize_outcome_residuals_v1(
        permuted_targets, stats, action_ids=action_ids
    )
    assert torch.allclose(permuted, canonical[:, permutation])
    assert torch.allclose(
        subject.decode_standardized_outcomes_v1(
            permuted, stats, action_ids=action_ids
        ),
        permuted_targets,
    )


def test_predicted_physical_cost_has_registered_signs_and_gradient() -> None:
    outcomes = torch.zeros((1, 9, 4), dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        outcomes[0, 0, 0] = 0.9
        outcomes[0, 1, 0] = 0.2
        outcomes[0, 0, 3] = -10.0
        outcomes[0, 1, 3] = 10.0
    goal = torch.tensor([[1.0, 0.0]])
    costs = subject.predicted_physical_cost_v1(outcomes, goal)
    assert costs.shape == (1, 9)
    assert costs[0, 0] < costs[0, 1]
    assert float(costs[0, 0].detach()) == pytest.approx(0.1)
    assert float(costs[0, 1].detach()) == pytest.approx(0.9)
    costs.sum().backward()
    assert outcomes.grad is not None and torch.isfinite(outcomes.grad).all()


def test_strict_rank_loss_sign_ties_and_branch_permutation() -> None:
    ranks = torch.tensor([[0, 0, 1, 2, 2, 3, 4, 4, 5]], dtype=torch.long)
    good = torch.tensor([[0.0, 0.3, 1.0, 2.0, 2.2, 3.0, 4.0, 4.1, 5.0]])
    bad = -good
    good_loss = subject.strict_rank_pairwise_softplus_loss_v1(good, ranks)
    bad_loss = subject.strict_rank_pairwise_softplus_loss_v1(bad, ranks)
    assert good_loss < bad_loss

    strict = ranks.unsqueeze(-1) < ranks.unsqueeze(-2)
    differences = good.unsqueeze(-1) - good.unsqueeze(-2)
    manual = torch.nn.functional.softplus(differences[strict] / 0.05).mean()
    assert good_loss == pytest.approx(float(manual))
    assert int(strict.sum()) == sum(
        int(left < right)
        for left in ranks[0].tolist()
        for right in ranks[0].tolist()
    )

    permutation = torch.tensor([5, 2, 8, 0, 7, 4, 1, 6, 3])
    permuted_loss = subject.strict_rank_pairwise_softplus_loss_v1(
        good[:, permutation], ranks[:, permutation]
    )
    assert permuted_loss == pytest.approx(float(good_loss))


def _orthogonal_successors() -> torch.Tensor:
    result = torch.zeros((1, 9, 256, 384), dtype=torch.float32)
    for action in range(9):
        result[0, action, :, action] = 1.0
    return result


def test_dense_jepa_losses_and_retrieval_use_action_correspondence() -> None:
    target = _orthogonal_successors().requires_grad_(True)
    predicted = target.detach().clone().requires_grad_(True)
    assert float(
        subject.dense_patch_cosine_loss_v1(predicted, target).detach()
    ) == pytest.approx(0.0)
    matched = subject.within_state_action_infonce_loss_v1(predicted, target)
    shifted_target = target.detach().roll(1, dims=1)
    shifted = subject.within_state_action_infonce_loss_v1(predicted, shifted_target)
    assert matched < shifted
    assert subject.true_successor_branch_retrieval_v1(predicted, target) == pytest.approx(1.0)
    assert subject.true_successor_branch_retrieval_v1(
        predicted, shifted_target
    ) == pytest.approx(0.0)
    (matched + subject.dense_patch_cosine_loss_v1(predicted, target)).backward()
    assert predicted.grad is not None and torch.isfinite(predicted.grad).all()
    assert target.grad is None


def test_trunk_layout_uses_plan_identities_and_context_only_fails_closed() -> None:
    context_ids = ("c0", "c1", "c2")
    target_ids = tuple(f"t{index}" for index in range(9))
    # Plan storage order is deliberately unrelated to state/action order.
    plan_ids = ("t4", "c1", "t8", "c0", "t0", "t2", "c2", "t1", "t7", "t3", "t6", "t5")
    index = {artifact_id: offset for offset, artifact_id in enumerate(plan_ids)}
    state = SimpleNamespace(
        context_artifact_indices=tuple(index[item] for item in context_ids),
        target_artifact_indices=tuple(index[item] for item in target_ids),
    )
    plan = SimpleNamespace(artifact_ids=plan_ids, states=(state,))

    supplied_ids = tuple(reversed(plan_ids))
    supplied = torch.stack(
        [torch.full((257, 384), float(index[item])) for item in supplied_ids]
    )
    layout = subject.extract_dense_trunk_layout_v1(
        plan, supplied_ids, supplied, include_successors=True
    )
    assert layout.context_trunk_tokens.shape == (1, 3, 257, 384)
    assert layout.successor_trunk_tokens is not None
    assert layout.successor_trunk_tokens.shape == (1, 9, 257, 384)
    assert layout.context_artifact_ids == (context_ids,)
    assert layout.successor_artifact_ids == (target_ids,)
    assert layout.context_trunk_tokens[0, :, 0, 0].tolist() == [
        float(index[item]) for item in context_ids
    ]
    assert layout.successor_trunk_tokens[0, :, 0, 0].tolist() == [
        float(index[item]) for item in target_ids
    ]

    context_tokens = torch.stack(
        [torch.full((257, 384), float(index[item])) for item in reversed(context_ids)]
    )
    context_layout = subject.extract_dense_trunk_layout_v1(
        plan, tuple(reversed(context_ids)), context_tokens, include_successors=False
    )
    assert context_layout.successor_trunk_tokens is None
    with pytest.raises(subject.GroundedDenseDINOJointJEPAError, match="context-only"):
        subject.extract_dense_trunk_layout_v1(
            plan, supplied_ids, supplied, include_successors=False
        )


def _physical_plan() -> SimpleNamespace:
    states = []
    groups = []
    for state_index in range(128):
        family = calibration.FAMILIES[state_index // 16]
        scene = f"{family}-{(state_index % 16) // 8}"
        ranks = tuple(range(9))
        states.append(
            SimpleNamespace(
                state_id=f"s{state_index}",
                family=family,
                scene_id=scene,
                relative_target_xy_body_m=(1.0, 0.0),
                dense_ranks=ranks,
            )
        )
        branches = tuple(
            SimpleNamespace(
                oracle_dense_rank=action,
                labels=SimpleNamespace(
                    fell=False,
                    tipped=False,
                    target_progress_m=float(8 - action),
                    path_length_m=float(action),
                    planar_clearance_proxy_min_m=None,
                    grid_recoverability_proxy=None,
                ),
            )
            for action in range(9)
        )
        groups.append(
            SimpleNamespace(
                state_id=f"s{state_index}",
                family=family,
                scene_id=scene,
                branches=branches,
            )
        )
    return SimpleNamespace(states=tuple(states), groups=tuple(groups))


def test_physical_scoring_has_exact_existing_tie_and_action_order_parity() -> None:
    plan = _physical_plan()
    outcomes = torch.zeros((128, 9, 4), dtype=torch.float32)
    actual = subject.physical_score_matrix_v1(plan, outcomes)
    expected = prior.physical_score_matrix_v1(plan, outcomes)
    assert np.array_equal(actual, expected)
    assert np.array_equal(actual[0], np.arange(9, dtype=np.float64))
    report = subject.report_physical_scores_v1(plan, actual)
    assert report["summary"]["groups"] == 128
    assert report["summary"]["normalized_rank_regret"] == 0.0
    assert report["summary"]["oracle_equivalent_selection_rate"] == 1.0


def _bootstrap_rows(delta: float) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    candidate = []
    baseline = []
    state_index = 0
    for family_index, family in enumerate(calibration.FAMILIES):
        for scene_index in range(2):
            scene = f"{family}-{scene_index}"
            for _ in range(8):
                baseline_value = 0.30 + family_index * 0.001 + scene_index * 0.002
                common = {
                    "state_id": f"state-{state_index}",
                    "scene_id": scene,
                    "family": family,
                }
                baseline.append({**common, "normalized_rank_regret": baseline_value})
                candidate.append(
                    {**common, "normalized_rank_regret": baseline_value + delta}
                )
                state_index += 1
    return candidate, baseline


def test_family_equal_scene_bootstrap_is_deterministic_and_uses_scene_units() -> None:
    candidate, baseline = _bootstrap_rows(-0.025)
    first = subject.paired_family_scene_bootstrap_v1(candidate, baseline)
    second = subject.paired_family_scene_bootstrap_v1(
        list(reversed(candidate)), list(reversed(baseline))
    )
    assert first == second
    assert first["seed"] == 2026080407
    assert first["resamples"] == 10_000
    assert first["paired_states"] == 128
    assert first["scene_clusters"] == 16
    assert first["family_strata"] == 8
    assert first["mean_delta"] == pytest.approx(-0.025)
    assert first["upper_95"] < 0.0


def _report(regret: float, *, oracle_rate: float | None = None) -> dict[str, object]:
    summary: dict[str, object] = {"normalized_rank_regret": regret}
    if oracle_rate is not None:
        summary["oracle_equivalent_selection_rate"] = oracle_rate
    return {"summary": summary}


def test_fixed_gate_requires_every_preregistered_threshold() -> None:
    passing = subject.fixed_gate_v1(
        joint_report=_report(0.12),
        task_report=_report(0.14),
        matched_report=_report(0.13),
        random_report=_report(0.49),
        oracle_report=_report(0.0, oracle_rate=1.0),
        joint_vs_task={"mean_delta": -0.02, "upper_95": -0.001},
        joint_vs_matched={"mean_delta": -0.01, "upper_95": -0.0001},
        integrity_passed=True,
    )
    assert passing["passed"] is True
    assert all(gate["passed"] for gate in passing["gates"].values())

    for gate_name, changes in (
        ("1_integrity_and_oracle", {"integrity_passed": False}),
        ("2_absolute_regret", {"joint": 0.131}),
        ("3_joint_beats_task_action_only", {"task_upper": 0.0}),
        ("4_joint_beats_matched_physical_only", {"matched_upper": 0.0}),
        ("5_joint_beats_random", {"random": 0.12}),
    ):
        joint = float(changes.get("joint", 0.12))
        task = 0.14
        matched = 0.13
        result = subject.fixed_gate_v1(
            joint_report=_report(joint),
            task_report=_report(task),
            matched_report=_report(matched),
            random_report=_report(float(changes.get("random", 0.49))),
            oracle_report=_report(0.0, oracle_rate=1.0),
            joint_vs_task={
                "mean_delta": joint - task,
                "upper_95": float(changes.get("task_upper", -0.001)),
            },
            joint_vs_matched={
                "mean_delta": joint - matched,
                "upper_95": float(changes.get("matched_upper", -0.0001)),
            },
            integrity_passed=bool(changes.get("integrity_passed", True)),
        )
        assert result["passed"] is False
        assert result["gates"][gate_name]["passed"] is False
