"""Batch-render job planning for Genesis smoke and pilot runs."""

from __future__ import annotations

from typing import Any


def build_render_jobs(
    scene_specs: list[dict[str, Any]],
    split: str,
    output_root: str,
) -> list[dict[str, Any]]:
    """Create deterministic render job descriptors without invoking Genesis."""

    jobs: list[dict[str, Any]] = []
    for index, scene_spec in enumerate(scene_specs):
        scene_id = str(scene_spec["scene_id"])
        jobs.append(
            {
                "job_id": f"{split}_{index:06d}_{scene_id}",
                "scene_id": scene_id,
                "manifest_sha256": scene_spec.get("manifest_sha256", ""),
                "split": split,
                "output_dir": f"{output_root.rstrip('/')}/{split}/{scene_id}",
            }
        )
    return jobs


def build_render_replay_jobs(
    raw_rollout_dirs: list[str],
    output_root: str,
    *,
    backend: str = "genesis",
    camera_hz: float = 10.0,
) -> list[dict[str, Any]]:
    """Create deterministic GPU render-replay job descriptors.

    These jobs are intentionally backend-neutral descriptors. A renderer worker
    takes one raw rollout, creates a per-frame replay plan, and then writes
    ``rendered_vision`` for that rollout.
    """

    jobs: list[dict[str, Any]] = []
    for index, raw_rollout_dir in enumerate(raw_rollout_dirs):
        rollout_name = raw_rollout_dir.rstrip("/").split("/")[-1]
        jobs.append(
            {
                "job_id": f"render_replay_{index:06d}_{rollout_name}",
                "backend": backend,
                "raw_rollout_dir": raw_rollout_dir,
                "camera_hz": camera_hz,
                "output_dir": f"{output_root.rstrip('/')}/{rollout_name}",
                "gpu_required": True,
            }
        )
    return jobs
