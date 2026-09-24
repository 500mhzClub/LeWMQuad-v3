from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from lewm.benchmarks import go2_task_coupled_recurrent_dynamics_v1 as subject
from lewm.models.go2_task_coupled_recurrent_dynamics_v1 import (
    CANDIDATE_WIDTH,
    CONTEXT_STEPS,
    MOTION_WIDTH,
    OUTPUT_WIDTH,
    TOKEN_COUNT,
    VISUAL_WIDTH,
)


def _plan(states: int, *, role: str) -> SimpleNamespace:
    rows = tuple(
        SimpleNamespace(
            state_id=f"{role}-{index}",
            relative_target_xy_body_m=(1.0 + 0.01 * index, -0.25),
            dense_ranks=tuple(range(subject.ACTION_COUNT)),
        )
        for index in range(states)
    )
    return SimpleNamespace(
        role=role,
        states=rows,
        identity_sha256=("a" if role == "train" else "b") * 64,
    )


def _role(states: int, *, role: str = "train") -> SimpleNamespace:
    physical = torch.zeros(
        states, subject.ACTION_COUNT, 12, dtype=torch.float32
    )
    for state in range(states):
        common_history = torch.arange(10, dtype=torch.float32) + 10.0 * state
        physical[state, :, :10] = common_history
        # Candidate-dependent slots are deliberately allowed to vary by branch.
        physical[state, :, 10] = torch.arange(subject.ACTION_COUNT) + state
        physical[state, :, 11] = -torch.arange(subject.ACTION_COUNT) - state
    history_commands = torch.arange(
        states * 2 * CANDIDATE_WIDTH, dtype=torch.float32
    ).reshape(states, 2, CANDIDATE_WIDTH)
    candidate_commands = torch.arange(
        states * subject.ACTION_COUNT * CANDIDATE_WIDTH, dtype=torch.float32
    ).reshape(states, subject.ACTION_COUNT, CANDIDATE_WIDTH)
    return SimpleNamespace(
        role=role,
        plan=_plan(states, role=role),
        physical_inputs=physical,
        targets=torch.zeros(
            states, subject.ACTION_COUNT, OUTPUT_WIDTH, dtype=torch.float32
        ),
        history_commands=history_commands,
        candidate_commands=candidate_commands,
        relative_goals=torch.zeros(states, 2, dtype=torch.float32),
        dense_ranks=torch.arange(subject.ACTION_COUNT, dtype=torch.long)
        .repeat(states, 1),
        identity_sha256=("c" if role == "train" else "d") * 64,
    )


def test_config_binds_the_preregistered_geometry_schedule_and_gates() -> None:
    config = subject.config_v1()

    assert config["schema"] == subject.SCHEMA
    assert config["states_per_role"] == 128
    assert config["actions"] == 9
    assert config["context_steps"] == 3
    assert config["frozen_dino_context_grid"] == [256, 384]
    assert config["model_visual_grid"] == [16, 16]
    assert config["visual_projection"] == {
        "pool": "nonoverlapping_4x4_mean_16x16_to_4x4",
        "channel_pca": [384, 16],
        "fit_role": "train_context_only",
        "standardize_projected_channels": True,
    }
    assert config["motion_step_width"] == MOTION_WIDTH
    assert config["candidate_command_width"] == CANDIDATE_WIDTH
    assert config["output_width"] == OUTPUT_WIDTH
    assert config["arms"] == [subject.NO_VISION_ARM, subject.VISUAL_ARM]
    assert config["model_seeds"] == list(subject.MODEL_SEEDS)
    assert config["shared_sampler_seed"] == subject.SAMPLER_SEED
    assert config["updates"] == 800
    assert config["batch_states"] == 8
    assert config["loss"]["all_strict_pair_rank_softplus_weight"] == 0.25
    assert config["goal_available_to_model"] is False
    assert config["successor_observation_access"] is False
    assert config["frozen_h1_thresholds"] == {
        "maximum_regret": 0.13,
        "visual_minus_task_maximum": -0.02,
        "visual_minus_no_vision_maximum": -0.01,
        "paired_upper_95_must_be_below_zero": True,
        "must_beat_random": True,
    }


def test_raw_temporal_inputs_preserve_order_and_require_branch_invariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "STATE_COUNT", 2)
    role = _role(2)

    motion, candidates = subject.raw_temporal_inputs_v1(role)

    assert motion.shape == (2, CONTEXT_STEPS, MOTION_WIDTH)
    assert torch.count_nonzero(motion[:, 0]) == 0
    assert torch.equal(motion[:, 1, :3], role.physical_inputs[:, 0, :3])
    assert torch.equal(motion[:, 1, 3:], role.history_commands[:, 0])
    assert torch.equal(motion[:, 2, :3], role.physical_inputs[:, 0, 3:6])
    assert torch.equal(motion[:, 2, 3:], role.history_commands[:, 1])
    assert candidates.is_contiguous()
    assert torch.equal(candidates, role.candidate_commands)

    # The branch-dependent candidate summary is not recurrent history.
    changed_candidate_summary = _role(2)
    changed_candidate_summary.physical_inputs[0, 1, 10:] = torch.tensor(
        [999.0, -999.0]
    )
    subject.raw_temporal_inputs_v1(changed_candidate_summary)

    changed_history = _role(2)
    changed_history.physical_inputs[0, 1, 7] += 1.0
    with pytest.raises(
        subject.TaskCoupledRecurrentDynamicsError,
        match="pre-candidate physical history differs across branches",
    ):
        subject.raw_temporal_inputs_v1(changed_history)


def test_input_statistics_are_train_only_and_normalize_shared_temporal_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "STATE_COUNT", 3)
    train = _role(3)
    train.candidate_commands[..., -1] = 7.0
    raw_motion, raw_candidates = subject.raw_temporal_inputs_v1(train)

    statistics = subject.fit_input_statistics_v1(train)
    normalized_motion, normalized_candidates = subject.normalize_temporal_inputs_v1(
        train, statistics
    )

    transition_rows = raw_motion[:, 1:].reshape(-1, MOTION_WIDTH)
    expected_motion_mean = transition_rows.mean(dim=0)
    expected_motion_scale = torch.sqrt(
        torch.mean((transition_rows - expected_motion_mean).square(), dim=0)
    )
    candidate_rows = raw_candidates.reshape(-1, CANDIDATE_WIDTH)
    expected_candidate_mean = candidate_rows.mean(dim=0)
    expected_candidate_scale = torch.sqrt(
        torch.mean((candidate_rows - expected_candidate_mean).square(), dim=0)
    )
    expected_candidate_scale[-1] = 1.0

    assert statistics["train_identity_sha256"] == train.identity_sha256
    assert torch.allclose(statistics["motion_mean"], expected_motion_mean)
    assert torch.allclose(statistics["motion_scale"], expected_motion_scale)
    assert torch.allclose(statistics["candidate_mean"], expected_candidate_mean)
    assert torch.allclose(statistics["candidate_scale"], expected_candidate_scale)
    assert torch.count_nonzero(normalized_motion[:, 0]) == 0
    assert torch.allclose(
        normalized_motion[:, 1:].reshape(-1, MOTION_WIDTH).mean(dim=0),
        torch.zeros(MOTION_WIDTH),
        atol=1.0e-6,
    )
    assert torch.allclose(
        torch.mean(
            normalized_motion[:, 1:].reshape(-1, MOTION_WIDTH).square(), dim=0
        ),
        torch.ones(MOTION_WIDTH),
        atol=1.0e-5,
    )
    assert torch.allclose(
        normalized_candidates[..., :-1].reshape(-1, CANDIDATE_WIDTH - 1).mean(
            dim=0
        ),
        torch.zeros(CANDIDATE_WIDTH - 1),
        atol=1.0e-6,
    )
    assert torch.allclose(
        torch.mean(
            normalized_candidates[..., :-1]
            .reshape(-1, CANDIDATE_WIDTH - 1)
            .square(),
            dim=0,
        ),
        torch.ones(CANDIDATE_WIDTH - 1),
        atol=1.0e-5,
    )
    assert torch.count_nonzero(normalized_candidates[..., -1]) == 0

    evaluation = _role(3, role="eval")
    with pytest.raises(
        subject.TaskCoupledRecurrentDynamicsError, match="train-only"
    ):
        subject.fit_input_statistics_v1(evaluation)
    eval_motion, eval_candidates = subject.normalize_temporal_inputs_v1(
        evaluation, statistics
    )
    assert eval_motion.shape == (3, CONTEXT_STEPS, MOTION_WIDTH)
    assert eval_candidates.shape == (3, subject.ACTION_COUNT, CANDIDATE_WIDTH)


def test_training_batches_are_shared_deterministic_complete_state_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "STATE_COUNT", 8)
    monkeypatch.setattr(subject, "BATCH_STATES", 4)
    monkeypatch.setattr(subject, "UPDATES", 7)

    first = subject.training_batches_v1()
    second = subject.training_batches_v1()

    assert len(first) == 7
    assert all(
        torch.equal(left, right) for left, right in zip(first, second, strict=True)
    )
    assert all(row.shape == (4,) and row.dtype == torch.long for row in first)
    for start in range(0, 6, 2):
        permutation = torch.cat(first[start : start + 2])
        assert torch.equal(torch.sort(permutation).values, torch.arange(8))
    with pytest.raises(
        subject.TaskCoupledRecurrentDynamicsError, match="training seed changed"
    ):
        subject.training_batches_v1(subject.SAMPLER_SEED + 1)


def test_raw_grid_pooling_and_train_only_pca_projection_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "STATE_COUNT", 2)
    spatial = torch.arange(256, dtype=torch.float32).reshape(16, 16)
    structured = spatial.reshape(1, 1, 256, 1).expand(
        2, CONTEXT_STEPS, 256, subject.RAW_VISUAL_WIDTH
    )

    pooled = subject._pool_context_grid_v1(structured, role="train")  # noqa: SLF001

    expected = spatial.reshape(4, 4, 4, 4).mean(dim=(1, 3)).reshape(TOKEN_COUNT)
    assert pooled.shape == (
        2,
        CONTEXT_STEPS,
        TOKEN_COUNT,
        subject.RAW_VISUAL_WIDTH,
    )
    assert pooled.dtype == torch.float64
    assert torch.equal(pooled[0, 0, :, 0], expected.to(torch.float64))

    generator = torch.Generator(device="cpu").manual_seed(2_026_080_415)
    train_tokens = torch.randn(
        2,
        CONTEXT_STEPS,
        subject.RAW_TOKEN_COUNT,
        subject.RAW_VISUAL_WIDTH,
        generator=generator,
        dtype=torch.float32,
    )
    projection = subject.fit_visual_projection_v1(train_tokens)
    projected = subject.project_context_tokens_v1(
        train_tokens, projection, role="train"
    )

    assert projection["training_rows"] == 2 * CONTEXT_STEPS * TOKEN_COUNT
    assert projection["mean"].shape == (subject.RAW_VISUAL_WIDTH,)
    assert projection["components"].shape == (
        subject.RAW_VISUAL_WIDTH,
        VISUAL_WIDTH,
    )
    assert projection["score_scale"].shape == (VISUAL_WIDTH,)
    assert projection["singular_values"].shape == (VISUAL_WIDTH,)
    assert projected.shape == (2, CONTEXT_STEPS, TOKEN_COUNT, VISUAL_WIDTH)
    assert projected.dtype == torch.float32
    flattened = projected.reshape(-1, VISUAL_WIDTH)
    assert torch.allclose(
        flattened.mean(dim=0), torch.zeros(VISUAL_WIDTH), atol=2.0e-6
    )
    assert torch.allclose(
        torch.mean(flattened.square(), dim=0),
        torch.ones(VISUAL_WIDTH),
        atol=2.0e-5,
    )
    for component in range(VISUAL_WIDTH):
        column = projection["components"][:, component]
        pivot = int(torch.argmax(torch.abs(column)))
        assert column[pivot] >= 0.0

    evaluation = train_tokens + 0.25
    eval_projected = subject.project_context_tokens_v1(
        evaluation, projection, role="eval"
    )
    assert eval_projected.shape == projected.shape
    with pytest.raises(
        subject.TaskCoupledRecurrentDynamicsError, match="train context tokens"
    ):
        subject.fit_visual_projection_v1(evaluation.to(torch.float64))
