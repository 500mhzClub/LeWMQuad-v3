from __future__ import annotations

from pathlib import Path

from scripts.render_jepa_counterfactual_plan_root import _output_dir
from scripts.render_jepa_counterfactual_plan_root_parallel import (
    _aggregate_reports,
    _render_or_reuse_scene,
    _scene_log_path,
)


def test_counterfactual_render_root_preserves_scene_relative_path(tmp_path: Path) -> None:
    plan_root = tmp_path / "plans"
    output_root = tmp_path / "renders"
    plan = plan_root / "train" / "family" / "scene" / "render_replay_plan.json"

    assert _output_dir(plan_root, output_root, plan) == (
        output_root / "train" / "family" / "scene"
    )


def test_parallel_render_root_preserves_scene_log_relative_path(
    tmp_path: Path,
) -> None:
    plan_root = tmp_path / "plans"
    log_root = tmp_path / "logs"
    plan = plan_root / "train" / "family" / "scene" / "render_replay_plan.json"

    assert _scene_log_path(plan_root, log_root, plan) == (
        log_root / "train" / "family" / "scene.log"
    )


def test_parallel_render_root_reuses_completed_scene(tmp_path: Path) -> None:
    plan_root = tmp_path / "plans"
    output_root = tmp_path / "renders"
    plan = plan_root / "train" / "family" / "scene" / "render_replay_plan.json"
    plan.parent.mkdir(parents=True)
    plan.write_text("{}\n")
    output = output_root / "train" / "family" / "scene"
    output.mkdir(parents=True)
    (output / "summary.json").write_text(
        '{"frame_count": 12, "invalid_frame_count": 3}\n'
    )

    report = _render_or_reuse_scene(
        index=1,
        total=1,
        plan_root=plan_root,
        output_root=output_root,
        log_root=tmp_path / "logs",
        repo_root=tmp_path,
        plan=plan,
        scene_corpus=tmp_path / "scene_corpus",
        backend="vulkan",
        camera_mode="replay",
        replay_env_mode="single",
        rgb_format="png",
        store_resolution="training",
        depth_validate_only=True,
        overwrite=False,
    )

    assert report["status"] == "reused"
    assert report["render_return_code"] == 2
    assert report["frame_count"] == 12
    assert report["invalid_frame_count"] == 3


def test_parallel_render_root_aggregate_counts_successes_and_failures(
    tmp_path: Path,
) -> None:
    reports = [
        {
            "index": 2,
            "status": "failed",
            "frame_count": 0,
            "invalid_frame_count": 0,
        },
        {
            "index": 1,
            "status": "rendered",
            "frame_count": 10,
            "invalid_frame_count": 4,
        },
    ]

    aggregate = _aggregate_reports(
        plan_root=tmp_path / "plans",
        output_root=tmp_path / "renders",
        reports=reports,
        expected_scene_count=2,
        jobs=4,
    )

    assert aggregate["schema"] == "jepa_counterfactual_render_root_summary_v0"
    assert aggregate["renderer"] == "parallel_scene_subprocess"
    assert aggregate["expected_scene_count"] == 2
    assert aggregate["scene_count"] == 1
    assert aggregate["frame_count"] == 10
    assert aggregate["invalid_frame_count"] == 4
    assert aggregate["failure_count"] == 1
    assert aggregate["scenes"][0]["index"] == 1
    assert aggregate["failures"][0]["index"] == 2
