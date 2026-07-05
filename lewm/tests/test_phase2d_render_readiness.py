from __future__ import annotations

import json
from pathlib import Path

from lewm.benchmarks.phase2_data import CONFIRMATORY_SPLIT_REQUIREMENTS
from lewm.benchmarks.phase2d_render_readiness import audit_phase2d_render_readiness


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_plan_summary(path: Path, *, scene_count: int, frame_count: int) -> None:
    _write_json(
        path / "summary.json",
        {
            "schema": "jepa_counterfactual_render_plan_summary_v0",
            "scene_count": scene_count,
            "frame_count": frame_count,
            "candidate_count": frame_count // 2,
            "plans": [
                str(path / f"scene_{index}" / "render_replay_plan.json")
                for index in range(scene_count)
            ],
        },
    )


def _write_render_summary(
    path: Path,
    *,
    plan_root: Path,
    scene_count: int,
    frame_count: int,
    invalid_frame_count: int = 0,
) -> None:
    frames_per_scene = frame_count // scene_count
    for index in range(scene_count):
        scene_dir = path / f"scene_{index}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        metadata = scene_dir / "frames_rendered.jsonl"
        metadata.write_text("{}\n" * frames_per_scene)
        _write_json(
            scene_dir / "summary.json",
            {
                "schema": "lewm_rendered_vision_v0",
                "plan": str(plan_root / f"scene_{index}" / "render_replay_plan.json"),
                "frames_rendered_jsonl": str(metadata),
                "frame_count": frames_per_scene,
                "invalid_frame_count": invalid_frame_count if index == 0 else 0,
            },
        )
    _write_json(
        path / "root_summary.json",
        {
            "schema": "jepa_counterfactual_render_root_summary_v0",
            "plan_root": str(plan_root.resolve()),
            "output_root": str(path.resolve()),
            "scene_count": scene_count,
            "frame_count": frame_count,
            "invalid_frame_count": invalid_frame_count,
            "scenes": [
                {
                    "plan": str(plan_root / f"scene_{index}" / "render_replay_plan.json"),
                    "output": str(path / f"scene_{index}"),
                    "status": "rendered",
                    "render_return_code": 0 if invalid_frame_count == 0 else 2,
                    "frame_count": frames_per_scene,
                    "invalid_frame_count": (
                        invalid_frame_count if index == 0 else 0
                    ),
                }
                for index in range(scene_count)
            ],
        },
    )


def test_render_readiness_blocks_missing_render_roots(tmp_path: Path) -> None:
    plan = tmp_path / "plans" / "train"
    _write_plan_summary(plan, scene_count=1, frame_count=2)

    report = audit_phase2d_render_readiness(
        plan_roots={"train": plan},
        render_roots={},
    )

    assert not report["ready_for_spatial_future_join"]
    assert report["missing_render_roots"] == [
        "test_hard",
        "test_id",
        "train",
        "validation",
    ]


def test_render_readiness_passes_complete_valid_registered_roots(
    tmp_path: Path,
) -> None:
    plan_roots = {}
    render_roots = {}
    for split_index, split in enumerate(CONFIRMATORY_SPLIT_REQUIREMENTS):
        scene_count = split_index + 1
        plan = tmp_path / "plans" / split
        render = tmp_path / "renders" / split
        _write_plan_summary(plan, scene_count=scene_count, frame_count=scene_count * 20)
        _write_render_summary(
            render,
            plan_root=plan,
            scene_count=scene_count,
            frame_count=scene_count * 20,
        )
        plan_roots[split] = plan
        render_roots[split] = render

    report = audit_phase2d_render_readiness(
        plan_roots=plan_roots,
        render_roots=render_roots,
    )

    assert report["ready_for_spatial_future_join"]
    assert report["checks"]["all_split_renders_complete_and_accounted"]
    assert report["splits"]["validation"]["invalid_frame_count"] == 0
    assert report["splits"]["validation"]["scene_metadata_frame_sum"] == 40


def test_render_readiness_records_invalid_frames_without_blocking_accounting(
    tmp_path: Path,
) -> None:
    plan_roots = {}
    render_roots = {}
    for split in CONFIRMATORY_SPLIT_REQUIREMENTS:
        plan = tmp_path / "plans" / split
        render = tmp_path / "renders" / split
        _write_plan_summary(plan, scene_count=1, frame_count=20)
        _write_render_summary(
            render,
            plan_root=plan,
            scene_count=1,
            frame_count=20,
            invalid_frame_count=1 if split == "test_hard" else 0,
        )
        plan_roots[split] = plan
        render_roots[split] = render

    report = audit_phase2d_render_readiness(
        plan_roots=plan_roots,
        render_roots=render_roots,
    )

    assert report["ready_for_spatial_future_join"]
    assert not report["all_rendered_frames_valid"]
    assert not report["splits"]["test_hard"]["all_rendered_frames_valid"]
    assert report["splits"]["test_hard"]["invalid_frame_count"] == 1


def test_render_readiness_accepts_moved_scene_metadata_with_local_fallback(
    tmp_path: Path,
) -> None:
    plan_roots = {}
    render_roots = {}
    for split in CONFIRMATORY_SPLIT_REQUIREMENTS:
        plan = tmp_path / "plans" / split
        render = tmp_path / "renders" / split
        _write_plan_summary(plan, scene_count=1, frame_count=2)
        _write_render_summary(
            render,
            plan_root=plan,
            scene_count=1,
            frame_count=2,
        )
        if split == "train":
            summary_path = render / "scene_0" / "summary.json"
            summary = json.loads(summary_path.read_text())
            summary["frames_rendered_jsonl"] = str(
                tmp_path / "old_render_root" / split / "scene_0" / "frames_rendered.jsonl"
            )
            _write_json(summary_path, summary)
        plan_roots[split] = plan
        render_roots[split] = render

    report = audit_phase2d_render_readiness(
        plan_roots=plan_roots,
        render_roots=render_roots,
    )

    assert report["ready_for_spatial_future_join"]
    assert report["splits"]["train"]["scene_metadata_relocated_count"] == 1
    assert report["splits"]["train"]["scene_metadata_frame_sum"] == 2


def test_render_readiness_blocks_mismatched_frame_counts(
    tmp_path: Path,
) -> None:
    plan_roots = {}
    render_roots = {}
    for split in CONFIRMATORY_SPLIT_REQUIREMENTS:
        plan = tmp_path / "plans" / split
        render = tmp_path / "renders" / split
        _write_plan_summary(plan, scene_count=1, frame_count=20)
        _write_render_summary(
            render,
            plan_root=plan,
            scene_count=1,
            frame_count=18 if split == "test_id" else 20,
        )
        plan_roots[split] = plan
        render_roots[split] = render

    report = audit_phase2d_render_readiness(
        plan_roots=plan_roots,
        render_roots=render_roots,
    )

    assert not report["ready_for_spatial_future_join"]
    assert not report["splits"]["test_id"]["checks"][
        "rendered_frame_count_matches_plan"
    ]
