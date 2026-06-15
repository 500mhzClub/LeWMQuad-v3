from __future__ import annotations

import json
from pathlib import Path

import torch

from lewm.actions import encode_active_block
from lewm.benchmarks.phase2s_swept_geometry_affordance import (
    build_phase2s_swept_geometry_affordance_examples,
    materialize_phase2s_swept_geometry_batch,
    phase2s_swept_feature_names,
    phase2s_swept_geometry_dataset_audit,
)
from lewm.models.primitive_affordance import (
    SweptGeometryPrimitiveAffordanceModel,
    factorized_affordance_losses,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def _frame_tree(tmp_path: Path, scene_id: str) -> tuple[str, str]:
    scene_dir = tmp_path / "render" / scene_id
    rgb_dir = scene_dir / "rgb"
    rgb_dir.mkdir(parents=True)
    start = rgb_dir / "frame_000001_env_00.png"
    goal = rgb_dir / "frame_000002_env_00.png"
    start.write_bytes(b"")
    goal.write_bytes(b"")
    plan = tmp_path / "rollout" / "train" / "maze" / "plan" / scene_id
    frames = plan / "frames.jsonl"
    _write_json(
        scene_dir / "summary.json",
        {"schema": "test_render_summary", "plan": str(plan / "render_replay_plan.json")},
    )
    _write_json(
        plan / "render_replay_plan.json",
        {"schema": "test_plan", "frames_jsonl": str(frames)},
    )
    frames.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "frame_index": 1,
                        "env_index": 0,
                        "base_pose_world": {
                            "position": {"x": 0.0, "y": 0.0, "z": 0.3}
                        },
                        "base_rpy_rad": {"yaw": 0.0},
                    }
                ),
                json.dumps(
                    {
                        "frame_index": 2,
                        "env_index": 0,
                        "base_pose_world": {
                            "position": {"x": 1.0, "y": 0.0, "z": 0.3}
                        },
                        "base_rpy_rad": {"yaw": 0.0},
                    }
                ),
            ]
        )
        + "\n"
    )
    return str(start), str(goal)


def _manifest(tmp_path: Path, scene_id: str) -> str:
    path = tmp_path / "scene_corpus" / scene_id / "manifest.json"
    _write_json(
        path,
        {
            "scene_id": scene_id,
            "family": "test_maze",
            "world_bounds_xy_m": [[-2.0, -2.0], [2.0, 2.0]],
            "walls": [
                {
                    "object_id": "front_wall",
                    "kind": "wall",
                    "center_xyz_m": [1.0, 0.0, 0.4],
                    "size_xyz_m": [0.1, 2.0, 0.8],
                    "yaw_rad": 0.0,
                    "material_id": "wall",
                }
            ],
            "obstacles": [],
        },
    )
    return str(path)


def _block(vx: float, yaw_rate: float = 0.0) -> list[float]:
    return encode_active_block(
        [vx] * 5,
        [0.0] * 5,
        [yaw_rate] * 5,
    ).tolist()


def _labels(
    *,
    progress: float,
    unsafe_fraction: float = 0.0,
    enters_unsafe: bool = False,
) -> dict:
    return {
        "target_progress_m": progress,
        "p05_swept_configuration_clearance_m": 0.2,
        "minimum_swept_configuration_clearance_m": 0.3,
        "unsafe_sample_fraction": unsafe_fraction,
        "enters_grid_unsafe": enters_unsafe,
        "ends_grid_unsafe": False,
        "target_recoverable": not enters_unsafe,
        "target_heading_error_rad": 0.0,
    }


def _row(
    *,
    scene_id: str,
    source_index: int,
    start_frame: str,
    goal_frame: str,
    scene_manifest: str,
    sequence: tuple[str, str],
    progress: float,
    unsafe_fraction: float = 0.0,
    enters_unsafe: bool = False,
) -> dict:
    block_by_name = {
        "forward_slow": _block(0.2),
        "backward": _block(-0.2),
        "yaw_left": _block(0.0, 0.45),
        "hold": _block(0.0),
    }
    return {
        "scene_id": scene_id,
        "family": "test_maze",
        "source_index": source_index,
        "start_frame": start_frame,
        "goal_frame": goal_frame,
        "scene_manifest": scene_manifest,
        "primitive_sequence": list(sequence),
        "active_blocks": [block_by_name[name] for name in sequence],
        "future_frames": ["future_0.png", "future_1.png"],
        "consequence_labels": _labels(
            progress=progress,
            unsafe_fraction=unsafe_fraction,
            enters_unsafe=enters_unsafe,
        ),
    }


def test_phase2s_swept_geometry_examples_materialize_per_primitive_features(
    tmp_path: Path,
) -> None:
    scene_id = "test_scene"
    start_frame, goal_frame = _frame_tree(tmp_path, scene_id)
    scene_manifest = _manifest(tmp_path, scene_id)
    rows = [
        _row(
            scene_id=scene_id,
            source_index=1,
            start_frame=start_frame,
            goal_frame=goal_frame,
            scene_manifest=scene_manifest,
            sequence=("forward_slow", "hold"),
            progress=0.3,
        ),
        _row(
            scene_id=scene_id,
            source_index=1,
            start_frame=start_frame,
            goal_frame=goal_frame,
            scene_manifest=scene_manifest,
            sequence=("forward_slow", "yaw_left"),
            progress=0.2,
        ),
        _row(
            scene_id=scene_id,
            source_index=1,
            start_frame=start_frame,
            goal_frame=goal_frame,
            scene_manifest=scene_manifest,
            sequence=("backward", "hold"),
            progress=-0.3,
            unsafe_fraction=1.0,
            enters_unsafe=True,
        ),
    ]
    primitive_names = ("forward_slow", "backward")
    feature_names = phase2s_swept_feature_names(primitive_names)

    examples = build_phase2s_swept_geometry_affordance_examples(
        rows,
        primitive_names=primitive_names,
        command_dt_s=0.1,
        max_ray_m=4.0,
    )
    batch = materialize_phase2s_swept_geometry_batch(examples, (0,))
    audit = phase2s_swept_geometry_dataset_audit(
        examples,
        split_name="train",
        feature_names=feature_names,
    )

    assert len(examples) == 1
    assert examples[0].source_pose_found
    assert examples[0].goal_pose_found
    assert examples[0].continuation_counts == (2, 1)
    assert batch.swept_geometry_features.shape == (1, 2, len(feature_names))
    assert batch.factor_targets.shape == (1, 2, 6)
    assert audit["finite_features"]
    assert audit["minimum_continuations_per_primitive"] == 1
    assert batch.swept_geometry_features[0, 0, 0] == 1.0
    assert batch.swept_geometry_features[0, 1, 1] == 1.0


def test_swept_geometry_primitive_affordance_model_backpropagates() -> None:
    torch.manual_seed(101)
    model = SweptGeometryPrimitiveAffordanceModel(
        feature_dim=14,
        factor_count=6,
        hidden_dim=16,
        depth=2,
    )
    logits = model(torch.randn(4, 3, 14))
    losses = factorized_affordance_losses(
        factor_logits=logits,
        factor_targets=torch.rand(4, 3, 6),
        factor_mask=torch.ones(4, 3, 6, dtype=torch.bool),
    )

    losses["factorized_affordance_loss"].backward()

    assert logits.shape == (4, 3, 6)
    assert model.net[-1].weight.grad is not None
