from __future__ import annotations

import json
from pathlib import Path

from scripts.build_jepa_spatial_future_dataset import build_spatial_future_dataset


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_build_spatial_future_dataset_preserves_complete_valid_sequence(tmp_path: Path) -> None:
    start = tmp_path / "start.png"
    future0 = tmp_path / "future0.png"
    future1 = tmp_path / "future1.png"
    for path in (start, future0, future1):
        path.write_bytes(b"png")
    benchmark = tmp_path / "benchmark.jsonl"
    candidate = {
        "primitive_sequence": ["forward", "hold"],
        "active_blocks": [[0.0] * 15, [0.0] * 15],
        "starts_grid_unsafe": False,
        "enters_grid_unsafe": False,
        "ends_grid_unsafe": False,
        "unsafe_sample_fraction": 0.0,
        "minimum_swept_configuration_clearance_m": 0.2,
        "p05_swept_configuration_clearance_m": 0.2,
        "clearance_gain_m": 0.0,
        "target_progress_m": 0.1,
        "target_heading_error_rad": 0.0,
        "target_recoverable": True,
    }
    _write_jsonl(
        benchmark,
        [
            {
                "scene_id": "scene",
                "family": "family",
                "split": "train",
                "start_frame": str(start),
                "local_target_frame": None,
                "counterfactual_target_cell_id": None,
                "counterfactual_horizon_blocks": 2,
                "counterfactual_oracle_index": 0,
                "counterfactual_candidates": [candidate],
            }
        ],
    )
    plan_root = tmp_path / "plans"
    plan_dir = plan_root / "train" / "family" / "scene"
    frames = plan_dir / "frames.jsonl"
    _write_jsonl(
        frames,
        [
            {
                "frame_index": block,
                "env_index": 0,
                "counterfactual_context": {
                    "source_index": 0,
                    "candidate_index": 0,
                    "block_index": block,
                    "physics_validated": False,
                },
            }
            for block in range(2)
        ],
    )
    plan = plan_dir / "render_replay_plan.json"
    plan.write_text(json.dumps({"frames_jsonl": str(frames)}))
    render_root = tmp_path / "renders"
    metadata = render_root / "frames_rendered.jsonl"
    _write_jsonl(
        metadata,
        [
            {
                "frame_index": block,
                "env_index": 0,
                "rgb_path": str(path),
                "camera_valid": True,
                "invalid_reasons": [],
                "camera_safety": {},
            }
            for block, path in enumerate((future0, future1))
        ],
    )
    (render_root / "summary.json").write_text(
        json.dumps(
            {
                "schema": "lewm_rendered_vision_v0",
                "plan": str(plan),
                "frames_rendered_jsonl": str(metadata),
            }
        )
    )
    output = tmp_path / "dataset.jsonl"

    summary = build_spatial_future_dataset(
        benchmark=benchmark,
        plan_root=plan_root,
        render_root=render_root,
        output=output,
    )
    row = json.loads(output.read_text())

    assert summary["candidate_sequences_written"] == 1
    assert row["future_frames"] == [str(future0), str(future1)]
    assert row["complete_valid_future_sequence"]
    assert all(item["observation_valid"] for item in row["future_observations"])
    assert row["is_oracle_candidate"]


def test_build_spatial_future_dataset_preserves_invalid_sequence(tmp_path: Path) -> None:
    start = tmp_path / "start.png"
    future = tmp_path / "future.png"
    for path in (start, future):
        path.write_bytes(b"png")
    benchmark = tmp_path / "benchmark.jsonl"
    candidate = {
        "primitive_sequence": ["forward"],
        "active_blocks": [[0.0] * 15],
        "starts_grid_unsafe": False,
        "enters_grid_unsafe": True,
        "ends_grid_unsafe": True,
        "unsafe_sample_fraction": 1.0,
        "minimum_swept_configuration_clearance_m": -0.1,
        "p05_swept_configuration_clearance_m": -0.1,
        "clearance_gain_m": -0.1,
        "target_progress_m": 0.1,
        "target_heading_error_rad": 0.0,
        "target_recoverable": False,
    }
    _write_jsonl(
        benchmark,
        [
            {
                "scene_id": "scene",
                "family": "family",
                "split": "train",
                "start_frame": str(start),
                "local_target_frame": None,
                "counterfactual_target_cell_id": None,
                "counterfactual_horizon_blocks": 1,
                "counterfactual_oracle_index": 0,
                "counterfactual_candidates": [candidate],
            }
        ],
    )
    plan_root = tmp_path / "plans"
    plan_dir = plan_root / "train" / "family" / "scene"
    frames = plan_dir / "frames.jsonl"
    _write_jsonl(
        frames,
        [
            {
                "frame_index": 0,
                "env_index": 0,
                "counterfactual_context": {
                    "source_index": 0,
                    "candidate_index": 0,
                    "block_index": 0,
                    "physics_validated": False,
                },
            }
        ],
    )
    plan = plan_dir / "render_replay_plan.json"
    plan.write_text(json.dumps({"frames_jsonl": str(frames)}))
    render_root = tmp_path / "renders"
    metadata = render_root / "frames_rendered.jsonl"
    _write_jsonl(
        metadata,
        [
            {
                "frame_index": 0,
                "env_index": 0,
                "rgb_path": str(future),
                "camera_valid": False,
                "invalid_reasons": ["near_forward_geometry"],
                "camera_safety": {"unsafe": True},
            }
        ],
    )
    (render_root / "summary.json").write_text(
        json.dumps(
            {
                "schema": "lewm_rendered_vision_v0",
                "plan": str(plan),
                "frames_rendered_jsonl": str(metadata),
            }
        )
    )
    output = tmp_path / "dataset.jsonl"

    summary = build_spatial_future_dataset(
        benchmark=benchmark,
        plan_root=plan_root,
        render_root=render_root,
        output=output,
    )
    row = json.loads(output.read_text())

    assert summary["candidate_sequences_written"] == 1
    assert summary["candidate_sequences_kinematic_unsafe"] == 1
    assert summary["candidate_sequences_kinematic_unsafe_complete_valid"] == 0
    assert not row["complete_valid_future_sequence"]
    assert row["future_observations"][0]["invalid_reasons"] == [
        "near_forward_geometry"
    ]
    assert row["future_observation_event"].startswith("incomplete_or_renderer_invalid")
