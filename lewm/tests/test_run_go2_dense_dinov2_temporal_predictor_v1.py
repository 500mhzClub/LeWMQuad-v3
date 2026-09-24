from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from scripts import run_go2_dense_dinov2_temporal_predictor_v1 as runner


def _primitive_document() -> dict[str, object]:
    primitives = {}
    for index, name in enumerate(runner.PRIMITIVES):
        primitives[name] = {
            "type": "velocity_block",
            "command": {
                "vx_body_mps": index / 10.0,
                "vy_body_mps": -index / 20.0,
                "yaw_rate_radps": index / 30.0,
            },
        }
    return {
        "block_size": 5,
        "command_order": [
            "vx_body_mps",
            "vy_body_mps",
            "yaw_rate_radps",
        ],
        "primitives": primitives,
    }


def test_action_conversion_is_canonical_channel_major_15d() -> None:
    table = runner.primitive_action_table_from_document(_primitive_document())
    assert table.shape == (len(runner.PRIMITIVES), 15)
    index = 3
    assert torch.allclose(table[index, :5], torch.full((5,), index / 10.0))
    assert torch.allclose(table[index, 5:10], torch.full((5,), -index / 20.0))
    assert torch.allclose(table[index, 10:], torch.full((5,), index / 30.0))


def test_epoch_sampler_is_deterministic_and_covers_each_row_once() -> None:
    first = runner.deterministic_batch_indices(
        row_count=8, batch_size=4, update_index=0, seed=11
    )
    second = runner.deterministic_batch_indices(
        row_count=8, batch_size=4, update_index=1, seed=11
    )
    repeated = runner.deterministic_batch_indices(
        row_count=8, batch_size=4, update_index=0, seed=11
    )
    assert np.array_equal(first, repeated)
    assert sorted(np.concatenate([first, second]).tolist()) == list(range(8))
    crossing = runner.deterministic_batch_indices(
        row_count=5, batch_size=4, update_index=1, seed=11
    )
    assert crossing.shape == (4,)
    assert all(0 <= int(value) < 5 for value in crossing)
    assert np.array_equal(
        crossing,
        runner.deterministic_batch_indices(
            row_count=5, batch_size=4, update_index=1, seed=11
        ),
    )


def test_wrong_action_donor_maximizes_two_horizon_contrast_deterministically() -> None:
    actions = (
        (6, 6, 0, 0),
        (6, 6, 1, 1),
        (6, 6, 1, 0),
        (6, 6, 0, 1),
    )
    donors = runner.select_wrong_action_donor_indices(actions)
    assert donors == runner.select_wrong_action_donor_indices(actions)
    assert donors[0] == 1
    assert donors[1] == 0
    for row_index, donor_index in enumerate(donors):
        assert donor_index != row_index
        assert any(
            actions[row_index][horizon] != actions[donor_index][horizon]
            for horizon in (2, 3)
        )


def test_temporal_controls_use_only_frames_zero_to_four_and_actions_zero_to_three() -> None:
    tokens = torch.stack(
        [torch.full((2, 256, 384), float(frame)) for frame in range(5)], dim=1
    )
    actions = torch.arange(2 * 4 * 15, dtype=torch.float32).reshape(2, 4, 15)
    wrong = torch.full((2, 2, 15), -1.0)
    controls = runner.build_temporal_controls(tokens, actions, wrong)

    assert controls["context"].shape == (2, 3, 256, 384)
    assert torch.equal(controls["targets"], tokens[:, 3:5])
    assert torch.equal(controls["history_actions"], actions[:, :2])
    assert torch.equal(controls["future_actions"], actions[:, 2:4])
    assert torch.equal(controls["persistence"][:, 0], tokens[:, 2])
    assert torch.equal(controls["persistence"][:, 1], tokens[:, 2])
    assert torch.equal(controls["wrong_a2_future_actions"][:, 0], wrong[:, 0])
    assert torch.equal(
        controls["wrong_a2_future_actions"][:, 1], actions[:, 3]
    )
    assert torch.equal(
        controls["wrong_a3_future_actions"][:, 0], actions[:, 2]
    )
    assert torch.equal(controls["wrong_a3_future_actions"][:, 1], wrong[:, 1])
    assert torch.count_nonzero(controls["reset_history_actions"]) == 0
    assert torch.equal(controls["current_only_context"][:, 0], tokens[:, 2])
    assert torch.equal(controls["current_only_context"][:, 2], tokens[:, 2])
    assert torch.equal(controls["reversed_context"], tokens[:, :3].flip(1))
    assert torch.equal(controls["reversed_history_actions"], actions[:, :2].flip(1))


class _SentinelPredictor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def forward(
        self,
        context: torch.Tensor,
        history_actions: torch.Tensor,
        candidate: torch.Tensor,
    ) -> torch.Tensor:
        self.calls.append(
            (context.detach().clone(), history_actions.detach().clone(), candidate.detach().clone())
        )
        value = candidate[:, 0].reshape(-1, 1, 1)
        return value.expand(-1, 256, 384).clone()


def test_free_running_rollout_has_exact_h1_h2_sentinel_alignment() -> None:
    model = _SentinelPredictor()
    context = torch.stack(
        [torch.full((1, 256, 384), value) for value in (10.0, 20.0, 30.0)],
        dim=1,
    )
    history = torch.zeros(1, 2, 15)
    history[:, 0, 0] = 1.0
    history[:, 1, 0] = 2.0
    future = torch.zeros(1, 2, 15)
    future[:, 0, 0] = 7.0
    future[:, 1, 0] = 9.0

    rollout = runner.free_running_rollout(model, context, history, future)

    assert torch.all(rollout[:, 0] == 7.0)
    assert torch.all(rollout[:, 1] == 9.0)
    assert len(model.calls) == 2
    h1_context, h1_history, h1_action = model.calls[0]
    h2_context, h2_history, h2_action = model.calls[1]
    assert torch.equal(h1_context, context)
    assert torch.equal(h1_history, history)
    assert torch.equal(h1_action, future[:, 0])
    assert torch.all(h2_context[:, 0] == 20.0)
    assert torch.all(h2_context[:, 1] == 30.0)
    assert torch.all(h2_context[:, 2] == 7.0)
    assert torch.equal(h2_history[:, 0], history[:, 1])
    assert torch.equal(h2_history[:, 1], future[:, 0])
    assert torch.equal(h2_action, future[:, 1])


def test_rollout_is_invariant_to_target_frame_mutation() -> None:
    base = torch.randn(1, 5, 256, 384)
    mutated = base.clone()
    mutated[:, 3:] = torch.randn_like(mutated[:, 3:]) * 100.0
    actions = torch.randn(1, 4, 15)
    wrong = torch.randn(1, 2, 15)
    original_controls = runner.build_temporal_controls(base, actions, wrong)
    mutated_controls = runner.build_temporal_controls(mutated, actions, wrong)
    original = runner.free_running_rollout(
        _SentinelPredictor(),
        original_controls["context"],
        original_controls["history_actions"],
        original_controls["future_actions"],
    )
    changed = runner.free_running_rollout(
        _SentinelPredictor(),
        mutated_controls["context"],
        mutated_controls["history_actions"],
        mutated_controls["future_actions"],
    )
    assert torch.equal(original, changed)
    assert original_controls["targets"].requires_grad is False


def test_cosine_error_is_per_row_and_horizon() -> None:
    target = torch.randn(3, 2, 4, 5)
    exact = runner.cosine_error_per_row_horizon(target, target)
    opposite = runner.cosine_error_per_row_horizon(-target, target)
    assert exact.shape == (3, 2)
    assert torch.allclose(exact, torch.zeros_like(exact), atol=1.0e-6)
    assert torch.allclose(opposite, torch.full_like(opposite, 2.0), atol=1.0e-6)


def _passing_metrics() -> tuple[dict[str, np.ndarray], list[str], list[str]]:
    scene_ids = ["scene-a", "scene-a", "scene-b", "scene-b"]
    family_ids = ["family-a", "family-a", "family-b", "family-b"]
    return {
        "correct": np.full((4, 2), 0.80),
        "persistence": np.full((4, 2), 1.00),
        "wrong_a2": np.full((4, 2), 0.90),
        "wrong_a3": np.full((4, 2), 0.90),
        "reset_history": np.asarray([[0.85, 0.90]] * 4, dtype=np.float64),
        "current_only": np.asarray([[0.84, 0.91]] * 4, dtype=np.float64),
        "reversed_history": np.asarray([[0.85, 0.95]] * 4, dtype=np.float64),
    }, scene_ids, family_ids


def _evaluation_for_gate() -> dict[str, object]:
    metrics, scenes, families = _passing_metrics()
    evaluation = runner.summarize_evaluation_metrics(
        metrics, scenes, families, bootstrap_draws=100, bootstrap_seed=5
    )
    evaluation["prediction_audit"] = {
        "all_finite": True,
        "row_descriptor_std": 0.1,
        "within_row_token_std": 0.2,
    }
    return evaluation


def test_scene_aggregation_bootstrap_and_ratios_are_deterministic() -> None:
    metrics, scenes, families = _passing_metrics()
    first = runner.summarize_evaluation_metrics(
        metrics, scenes, families, bootstrap_draws=100, bootstrap_seed=5
    )
    second = runner.summarize_evaluation_metrics(
        metrics, scenes, families, bootstrap_draws=100, bootstrap_seed=5
    )
    assert first == second
    assert first["controls"]["correct"]["h1"]["scene"]["point"] == 0.8
    assert first["comparisons"]["h2"]["correct_to_persistence_ratio"][
        "scene"
    ]["point"] == 0.8
    assert first["interpretation"]["wrong_action_gaps"].endswith(
        "not_same_state_counterfactual_causality"
    )
    wrong = first["comparisons"]["h1"]["wrong_a2_minus_correct"][
        "family_equal_scene"
    ]
    assert wrong["point"] == pytest.approx(0.1)
    assert wrong["normalized_by_persistence"] == pytest.approx(0.1)
    assert wrong["lower_95"] > 0.0
    persistence_advantage = first["comparisons"]["h1"][
        "persistence_minus_correct"
    ]["family_equal_scene"]
    assert persistence_advantage["lower_95"] > 0.0


def test_fixed_offline_gate_passes_all_required_controls() -> None:
    gate = runner.fixed_offline_gate(_evaluation_for_gate())
    assert gate["passes_all"] is True
    assert gate["primary_passes"] is True
    assert gate["h2_composability_passes"] is True
    assert gate["decision"] == "PASS_H1_MPC_AND_H2_COMPOSABILITY_OFFLINE_GATE"
    assert all(gate["h1_mpc_gate"]["criteria"].values())
    assert all(gate["h2_composability_gate"]["criteria"].values())


@pytest.mark.parametrize(
    "path,value",
    [
        (
            (
                "comparisons",
                "h1",
                "correct_to_persistence_ratio",
                "family_equal_scene",
                "point",
            ),
            0.96,
        ),
        (
            (
                "comparisons",
                "h1",
                "wrong_a2_minus_correct",
                "family_equal_scene",
                "normalized_by_persistence",
            ),
            0.009,
        ),
        (
            (
                "comparisons",
                "h1",
                "persistence_minus_correct",
                "family_equal_scene",
                "lower_95",
            ),
            0.0,
        ),
        (
            (
                "comparisons",
                "h1",
                "wrong_a2_minus_correct",
                "family_equal_scene",
                "lower_95",
            ),
            0.0,
        ),
    ],
)
def test_fixed_offline_gate_fails_each_h1_mechanism(
    path: tuple[str, ...], value: float
) -> None:
    evaluation = copy.deepcopy(_evaluation_for_gate())
    target = evaluation
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    gate = runner.fixed_offline_gate(evaluation)
    assert gate["passes_all"] is False
    assert gate["decision"] == "STOP_H1_MPC_OFFLINE_GATE_NOT_MET"


def test_h2_composability_is_reported_separately_from_h1_mpc_gate() -> None:
    evaluation = copy.deepcopy(_evaluation_for_gate())
    evaluation["comparisons"]["h2"]["wrong_a3_minus_correct"][
        "family_equal_scene"
    ]["normalized_by_persistence"] = 0.0
    gate = runner.fixed_offline_gate(evaluation)
    assert gate["primary_passes"] is True
    assert gate["h2_composability_passes"] is False
    assert gate["decision"] == (
        "PASS_H1_MPC_OFFLINE_GATE_H2_COMPOSABILITY_NOT_ESTABLISHED"
    )


def test_history_controls_and_collapse_audit_are_diagnostics_not_hard_gates() -> None:
    evaluation = copy.deepcopy(_evaluation_for_gate())
    for control in ("reset_history", "current_only", "reversed_history"):
        evaluation["controls"][control]["h2"]["family_equal_scene"]["point"] = 0.0
    evaluation["prediction_audit"]["row_descriptor_std"] = 0.0
    evaluation["prediction_audit"]["within_row_token_std"] = 0.0
    gate = runner.fixed_offline_gate(evaluation)
    assert gate["primary_passes"] is True
    assert gate["diagnostics_excluded_from_gate"] == [
        "reset_history",
        "current_only",
        "reversed_history",
    ]


def _set_h1_progress(
    evaluation: dict[str, object], *, ratio: float, normalized_gap: float
) -> dict[str, object]:
    result = copy.deepcopy(evaluation)
    result["comparisons"]["h1"]["correct_to_persistence_ratio"][
        "family_equal_scene"
    ]["point"] = ratio
    result["comparisons"]["h1"]["wrong_a2_minus_correct"][
        "family_equal_scene"
    ]["normalized_by_persistence"] = normalized_gap
    return result


def test_update_500_stops_without_persistence_advantage() -> None:
    evaluation = _set_h1_progress(
        _evaluation_for_gate(), ratio=1.0, normalized_gap=0.02
    )
    decision = runner.training_continuation_decision({500: evaluation})
    assert decision["should_continue"] is False
    assert decision["decision"] == "STOP_AT_U500_NO_H1_PERSISTENCE_ADVANTAGE"


def test_update_500_stops_for_nonpositive_action_gap() -> None:
    evaluation = _set_h1_progress(
        _evaluation_for_gate(), ratio=0.9, normalized_gap=0.0
    )
    decision = runner.training_continuation_decision({500: evaluation})
    assert decision["should_continue"] is False
    assert decision["decision"] == "STOP_AT_U500_NONPOSITIVE_H1_WRONG_A2_GAP"


@pytest.mark.parametrize(
    "u250_ratio,u250_gap,u500_ratio,u500_gap",
    [
        (0.90, 0.010, 0.89, 0.011),
        (0.90, 0.010, 0.895, 0.016),
    ],
)
def test_update_500_continues_for_meaningful_progress(
    u250_ratio: float, u250_gap: float, u500_ratio: float, u500_gap: float
) -> None:
    base = _evaluation_for_gate()
    decision = runner.training_continuation_decision(
        {
            250: _set_h1_progress(
                base, ratio=u250_ratio, normalized_gap=u250_gap
            ),
            500: _set_h1_progress(
                base, ratio=u500_ratio, normalized_gap=u500_gap
            ),
        }
    )
    assert decision["should_continue"] is True
    assert decision["decision"] == "CONTINUE_TO_U1000_MEANINGFUL_H1_PROGRESS"


def test_update_500_stops_when_progress_stalls() -> None:
    base = _evaluation_for_gate()
    decision = runner.training_continuation_decision(
        {
            250: _set_h1_progress(base, ratio=0.90, normalized_gap=0.010),
            500: _set_h1_progress(base, ratio=0.895, normalized_gap=0.014),
        }
    )
    assert decision["should_continue"] is False
    assert decision["decision"] == "STOP_AT_U500_H1_PROGRESS_STALLED"


def test_default_cli_contract_and_smoke_trace_override() -> None:
    parser = runner.build_parser()
    defaults = parser.parse_args([])
    assert defaults.updates == 1_000
    assert defaults.trace_updates == "0,250,500,1000"
    assert defaults.batch_size == 16
    assert defaults.learning_rate == 3.0e-4
    assert defaults.weight_decay == 1.0e-4
    assert defaults.gradient_clip_norm == 1.0
    assert runner.parse_trace_updates("0,1", updates=1) == (0, 1)
    with pytest.raises(runner.DenseDINORunnerError, match="start at zero"):
        runner.parse_trace_updates("1", updates=1)
