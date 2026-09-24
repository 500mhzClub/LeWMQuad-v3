"""Focused fake-only tests for the exact-task frozen-DINO cost ceiling."""
from __future__ import annotations

import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts import benchmark_lewm_closed_loop_mpc as benchmark  # noqa: E402


class _FakeEncoder:
    def __init__(self) -> None:
        self.inputs: list[torch.Tensor] = []

    def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        self.inputs.append(images.detach().cpu().clone())
        batch = images.shape[0]
        channel_means = images.mean(dim=(2, 3))
        tokens = torch.zeros(
            batch,
            benchmark.DINO_TOKEN_COUNT,
            benchmark.DINO_FEATURE_DIM,
            device=images.device,
        )
        tokens[:, :, :3] = channel_means.unsqueeze(1)
        tokens[:, :, 3] = 1.0
        return {"x_norm_patchtokens": tokens}


class _FakeRobot:
    def __init__(self, pos: np.ndarray, quat: np.ndarray) -> None:
        self.pos = np.asarray(pos, dtype=np.float32).copy()
        self.quat = np.asarray(quat, dtype=np.float32).copy()

    def get_pos(self) -> np.ndarray:
        return self.pos

    def get_quat(self) -> np.ndarray:
        return self.quat

    def set_pos(self, value: np.ndarray, **_kwargs: object) -> None:
        self.pos = np.asarray(value, dtype=np.float32)[0].copy()

    def set_quat(self, value: np.ndarray, **_kwargs: object) -> None:
        self.quat = np.asarray(value, dtype=np.float32)[0].copy()


class _FakeBuild:
    def __init__(self, pos: np.ndarray, quat: np.ndarray) -> None:
        self.robot = _FakeRobot(pos, quat)


def _unit_token_batch(vectors: torch.Tensor) -> torch.Tensor:
    return vectors[:, None, :].expand(-1, benchmark.DINO_TOKEN_COUNT, -1).clone()


def test_dinov2_preprocessing_encoding_and_same_patch_cost() -> None:
    image = torch.zeros(3, benchmark.DINO_IMAGE_SIZE, benchmark.DINO_IMAGE_SIZE)
    image[0].fill_(benchmark.DINO_IMAGENET_MEAN[0])
    image[1].fill_(benchmark.DINO_IMAGENET_MEAN[1])
    image[2].fill_(benchmark.DINO_IMAGENET_MEAN[2])
    prepared = benchmark._preprocess_dinov2_images(image)
    assert prepared.shape == (1, 3, 224, 224)
    torch.testing.assert_close(prepared, torch.zeros_like(prepared), atol=1e-6, rtol=0)

    encoder = _FakeEncoder()
    encoded = benchmark._encode_dinov2_images(
        encoder,
        torch.stack((torch.zeros_like(image), torch.ones_like(image))),
        device=torch.device("cpu"),
    )
    assert encoded.shape == (2, 256, 384)
    torch.testing.assert_close(
        torch.linalg.vector_norm(encoded, dim=-1),
        torch.ones(2, 256),
        atol=1e-6,
        rtol=0,
    )
    assert len(encoder.inputs) == 1

    basis = torch.zeros(3, benchmark.DINO_FEATURE_DIM)
    basis[0, 0] = 1.0
    basis[1, 0] = -1.0
    basis[2, 1] = 1.0
    candidates = _unit_token_batch(basis)
    goal = _unit_token_batch(basis[:1])
    costs = benchmark._dinov2_same_patch_costs(candidates, goal)
    torch.testing.assert_close(costs, torch.tensor([0.0, 2.0, 1.0]))


def test_dino_provenance_binds_runtime_and_exact_cost(tmp_path: Path) -> None:
    provenance = benchmark._dino_assay_provenance(
        repo_path=tmp_path / "repo",
        repository_commit=benchmark.DINO_REPOSITORY_COMMIT,
        checkpoint_path=tmp_path / "checkpoint.pth",
        checkpoint_bytes=benchmark.DINO_CHECKPOINT_BYTES,
        checkpoint_sha256=benchmark.DINO_CHECKPOINT_SHA256,
        device=torch.device("cpu"),
    )

    assert provenance["torch_version"] == str(torch.__version__)
    assert provenance["hip_version"] == (
        None if torch.version.hip is None else str(torch.version.hip)
    )
    assert provenance["device"] == "cpu"
    assert provenance["device_name"] is None
    assert provenance["input_rgb_shape"] == [3, 224, 224]
    assert provenance["patch_output_shape"] == [256, 384]
    assert provenance["token_normalization"] == "per_patch_l2"
    assert provenance["cost_definition"] == benchmark.DINO_COST_DEFINITION
    assert provenance["feature_cache_written"] is False


def test_h1_successors_reset_each_candidate_preserve_order_and_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start_pos = np.asarray([0.5, -0.2, 0.3], dtype=np.float32)
    start_quat = benchmark._quat_wxyz_from_yaw(0.0)
    build = _FakeBuild(start_pos, start_quat)
    registry = {
        "back": np.asarray([[-0.1, 0.0, 0.0]], dtype=np.float32),
        "forward": np.asarray([[0.2, 0.0, 0.0]], dtype=np.float32),
        "hold": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
    }
    sequences = [("back",), ("forward",), ("hold",)]
    monkeypatch.setattr(
        benchmark,
        "expand_primitive_to_block",
        lambda active_registry, name: active_registry[name],
    )
    resets: list[np.ndarray] = []
    original_set_pose = benchmark._set_pose

    def _tracked_set_pose(**kwargs: object) -> None:
        resets.append(np.asarray(kwargs["pos_xyz"], dtype=np.float32).copy())
        original_set_pose(**kwargs)

    rendered_x: list[float] = []

    def _fake_render(
        _build: object,
        _pack: object,
        *,
        base_xyz_m: np.ndarray,
        base_quat_wxyz: np.ndarray,
        device: torch.device,
    ) -> torch.Tensor:
        del base_quat_wxyz
        x = float(base_xyz_m[0])
        rendered_x.append(x)
        return torch.full((3, 224, 224), x, dtype=torch.float32, device=device)

    monkeypatch.setattr(benchmark, "_set_pose", _tracked_set_pose)
    monkeypatch.setattr(benchmark, "_render_tensor_from_base", _fake_render)
    images = benchmark._render_kinematic_h1_successors(
        build=build,
        pack=object(),
        registry=registry,
        sequences=sequences,
        start_pos=start_pos,
        start_quat=start_quat,
        command_dt_s=1.0,
        grid=None,
        render_device=torch.device("cpu"),
    )

    assert images.shape == (3, 3, 224, 224)
    assert rendered_x == pytest.approx([0.4, 0.7, 0.5])
    assert len(resets) == len(sequences) + 1
    assert all(np.array_equal(reset, start_pos) for reset in resets)
    assert np.array_equal(build.robot.pos, start_pos)
    assert np.array_equal(build.robot.quat, start_quat)


def test_true_successor_shuffle_moves_scores_without_moving_candidate_rows() -> None:
    costs = np.asarray([0.0, 0.7, 0.2, 1.1, 0.4], dtype=np.float64)
    direct = benchmark._rank_dino_policy_scores(
        costs,
        policy_name="dino_true_successor",
        seed=7,
        block_index=0,
    )
    shuffled = benchmark._rank_dino_policy_scores(
        costs,
        policy_name="dino_true_successor_shuffled",
        seed=7,
        block_index=0,
    )

    assert direct["selected_candidate_index"] == 0
    sources = shuffled["score_source_candidate_indices"]
    assert sorted(sources.tolist()) == list(range(len(costs)))
    assert not np.array_equal(sources, np.arange(len(costs)))
    assert np.array_equal(shuffled["policy_candidate_scores"], costs[sources])
    selected_row = shuffled["selected_candidate_index"]
    assert sources[selected_row] == 0
    assert selected_row != 0


def test_true_successor_policy_loop_emits_complete_score_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start_pos = np.asarray([0.5, 0.0, 0.3], dtype=np.float32)
    start_quat = benchmark._quat_wxyz_from_yaw(0.0)
    build = _FakeBuild(start_pos, start_quat)
    registry = {
        "back": np.asarray([[-0.1, 0.0, 0.0]], dtype=np.float32),
        "forward": np.asarray([[0.2, 0.0, 0.0]], dtype=np.float32),
        "hold": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
    }
    sequences = [("back",), ("forward",), ("hold",)]
    monkeypatch.setattr(
        benchmark,
        "expand_primitive_to_block",
        lambda active_registry, name: active_registry[name],
    )

    def _fake_render(
        _build: object,
        _pack: object,
        *,
        base_xyz_m: np.ndarray,
        base_quat_wxyz: np.ndarray,
        device: torch.device,
    ) -> torch.Tensor:
        del base_quat_wxyz
        return torch.full(
            (3, 224, 224),
            float(base_xyz_m[0]),
            dtype=torch.float32,
            device=device,
        )

    monkeypatch.setattr(benchmark, "_render_tensor_from_base", _fake_render)
    result = benchmark._run_policy_trial(
        policy_name="dino_true_successor",
        model=torch.nn.Identity(),
        build=build,
        pack=SimpleNamespace(scene_id="fake_scene"),
        runner=None,
        registry=registry,
        goal=benchmark.GoalSpec(
            object_id="beacon",
            landmark_xy=(2.0, 0.0),
            target_xy=(2.0, 0.0),
            target_yaw_rad=0.0,
            image=torch.full((3, 224, 224), 0.7),
        ),
        start_pos=start_pos,
        start_quat=start_quat,
        sequences=sequences,
        action_tensor=torch.zeros(3, 1, 15),
        primitive_names=[sequence[0] for sequence in sequences],
        max_blocks=1,
        goal_radius_m=0.01,
        fall_z_threshold_m=0.15,
        rng=random.Random(7),
        device=torch.device("cpu"),
        command_dt_s=1.0,
        grid=None,
        oracle_shuffle_seed=7,
        dino_encoder=_FakeEncoder(),
        dino_device=torch.device("cpu"),
        dino_provenance={"checkpoint_sha256": benchmark.DINO_CHECKPOINT_SHA256},
    )

    assert result.primitive_sequence == ["forward"]
    assert result.decision_log is not None and len(result.decision_log) == 1
    decision = result.decision_log[0]
    assert decision["selected_candidate_index"] == 1
    assert decision["selected_score_source_candidate_index"] == 1
    assert decision["selected_candidate_oracle_regret_m"] == pytest.approx(0.0)
    assert len(decision["policy_candidate_scores"]) == 3
    assert decision["policy_candidate_scores"] == decision[
        "unshuffled_dino_candidate_costs"
    ]
    assert decision["score_source_candidate_indices"] == [0, 1, 2]
    assert all(np.isfinite(decision["policy_candidate_scores"]))
    assert decision["dino_cost_definition"] == benchmark.DINO_COST_DEFINITION
    assert build.robot.pos[0] == pytest.approx(0.7)


def test_persistence_is_an_exact_registered_order_tie() -> None:
    vector = torch.zeros(1, benchmark.DINO_FEATURE_DIM)
    vector[0, 0] = 1.0
    tokens = _unit_token_batch(vector)
    costs = benchmark._dino_persistence_candidate_costs(
        tokens,
        tokens,
        candidate_count=9,
    )
    ranking = benchmark._rank_dino_policy_scores(
        costs,
        policy_name="dino_persistence",
        seed=7,
        block_index=0,
    )

    assert np.array_equal(costs, np.zeros(9, dtype=np.float64))
    assert ranking["selected_candidate_index"] == 0
    assert ranking["selected_score_source_candidate_index"] == 0
    assert ranking["policy_score_margin"] == 0.0
    assert np.array_equal(
        ranking["score_source_candidate_indices"], np.arange(9, dtype=np.int64)
    )


@pytest.mark.parametrize(
    ("policies", "mode", "horizon", "goal_views", "message"),
    (
        (["dino_true_successor"], "kinematic", 2, 0, "H1"),
        (["dino_true_successor_shuffled"], "physical", 1, 0, "kinematic"),
        (["dino_persistence"], "kinematic", 1, 2, "one goal image"),
    ),
)
def test_dino_scope_rejects_unregistered_modes(
    policies: list[str],
    mode: str,
    horizon: int,
    goal_views: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        benchmark._validate_dino_policy_scope(
            policies,
            mode=mode,
            horizon=horizon,
            goal_views=goal_views,
        )


def test_non_dino_scope_is_unchanged() -> None:
    benchmark._validate_dino_policy_scope(
        ["lewm", "oracle_mpc"],
        mode="physical",
        horizon=4,
        goal_views=3,
    )


def test_dino_candidate_bank_keeps_registered_order() -> None:
    names = [
        "hold",
        "forward_slow",
        "forward_medium",
        "forward_fast",
        "arc_left",
        "arc_right",
        "yaw_left",
        "yaw_right",
        "backward",
    ]
    blocks = {
        name: np.full(15, index, dtype=np.float32)
        for index, name in enumerate(names)
    }
    sequences, _actions = benchmark._candidate_action_tensor(
        blocks,
        names,
        1,
        max_candidates=None,
        rng=random.Random(7),
        device=torch.device("cpu"),
    )
    assert sequences == [(name,) for name in names]
