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
