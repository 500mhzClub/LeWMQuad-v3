#!/usr/bin/env python3
"""Bind one paired-navigation dataset to its actual rendered camera contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    values = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        values.append(value)
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        help=(
            "Optional legacy dataset binding. Omit to emit the stable v2 "
            "source-index contract consumed before a corrected dataset build."
        ),
    )
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_index_path = args.source_index.resolve()
    dataset_path = (
        args.dataset_manifest.resolve()
        if args.dataset_manifest is not None
        else None
    )
    dataset_by_scene = None
    if dataset_path is not None:
        dataset = read_json(dataset_path)
        if dataset.get("schema") != "lewm_go2_paired_navigation_dataset_v2":
            raise ValueError("legacy dataset binding supports only dataset v2")
        dataset_sources = dataset.get("sources")
        if not isinstance(dataset_sources, list) or not dataset_sources:
            raise ValueError("dataset has no source records")
        dataset_by_scene = {
            str(source["scene_id"]): source for source in dataset_sources
        }
    index_rows = read_jsonl(source_index_path)
    index_by_scene = {str(row["scene_id"]): row for row in index_rows}
    if len(index_by_scene) != len(index_rows):
        raise ValueError("source index contains duplicate scene IDs")
    if dataset_by_scene is not None and set(index_by_scene) != set(dataset_by_scene):
        raise ValueError("source index scene set differs from dataset")

    source_records = []
    expected_camera = None
    for scene_id in sorted(index_by_scene):
        dataset_source = (
            dataset_by_scene[scene_id]
            if dataset_by_scene is not None
            else None
        )
        index_row = index_by_scene[scene_id]
        summary_path = Path(str(index_row["render_summary_path"])).resolve()
        plan_path = Path(str(index_row["render_plan_path"])).resolve()
        summary_sha = sha256_file(summary_path)
        plan_sha = sha256_file(plan_path)
        if summary_sha != str(index_row["hashes"]["render_summary_file_sha256"]):
            raise ValueError(f"render-summary hash mismatch for {scene_id}")
        if plan_sha != str(index_row["hashes"]["render_plan_file_sha256"]):
            raise ValueError(f"render-plan hash mismatch for {scene_id}")
        if (
            dataset_source is not None
            and plan_sha != str(dataset_source["hashes"]["frame_plan_sha256"])
        ):
            raise ValueError(f"dataset render-plan hash mismatch for {scene_id}")
        summary = read_json(summary_path)
        plan = read_json(plan_path)
        camera = plan.get("camera")
        if not isinstance(camera, dict):
            raise ValueError(f"render plan has no camera object: {scene_id}")
        camera_contract = {
            "fov_axis": camera.get("fov_axis"),
            "fov_deg": float(camera.get("fov_deg")),
            "native_resolution_wh": list(camera.get("native_resolution", ())),
            "training_resolution_wh": list(camera.get("training_resolution", ())),
            "near_m": float(camera.get("near_m")),
        }
        if expected_camera is None:
            expected_camera = camera_contract
        elif camera_contract != expected_camera:
            raise ValueError("dataset mixes render-plan camera contracts")
        expected_summary = {
            "schema": "lewm_rendered_vision_v03",
            "render_status": "complete",
            "renderer": "genesis-0.3.14/vulkan",
            "resolution": 224,
            "visuals": "textured_v03",
        }
        for name, expected in expected_summary.items():
            if summary.get(name) != expected:
                raise ValueError(
                    f"unexpected render summary {name} for {scene_id}: "
                    f"{summary.get(name)!r}"
                )
        source_records.append(
            {
                "scene_id_sha256": hashlib.sha256(scene_id.encode()).hexdigest(),
                "render_plan": {
                    "path": str(plan_path),
                    "sha256": plan_sha,
                },
                "render_summary": {
                    "path": str(summary_path),
                    "sha256": summary_sha,
                },
            }
        )

    assert expected_camera is not None
    if expected_camera != {
        "fov_axis": "horizontal",
        "fov_deg": 78.323,
        "native_resolution_wh": [640, 480],
        "training_resolution_wh": [224, 224],
        "near_m": 0.05,
    }:
        raise ValueError(f"unexpected Go2 camera contract: {expected_camera}")
    aspect = 640.0 / 480.0
    intended_vertical = math.degrees(
        2.0
        * math.atan(math.tan(math.radians(78.323) * 0.5) / aspect)
    )
    core = {
        "schema": (
            "lewm_go2_source_camera_contract_v1"
            if dataset_path is not None
            else "lewm_go2_source_camera_contract_v2"
        ),
        "source_index": {
            "path": str(source_index_path),
            "sha256": sha256_file(source_index_path),
        },
        "scene_count": len(source_records),
        "scene_id_set_sha256": canonical_sha256(
            sorted(record["scene_id_sha256"] for record in source_records)
        ),
        "declared_camera": expected_camera,
        "actual_source_projection": {
            "renderer": "genesis-0.3.14/vulkan",
            "renderer_fov_axis": "vertical",
            "render_resolution_wh": [224, 224],
            "horizontal_fov_deg": 78.323,
            "vertical_fov_deg": 78.323,
        },
        "platform_projection_after_rectification": {
            "horizontal_fov_deg": 78.323,
            "vertical_fov_deg": intended_vertical,
            "center_crop_fraction_xy": [1.0, 0.75],
            "crop_before_model_resize": True,
        },
        "source_records": source_records,
        "g2_images_opened": False,
    }
    if dataset_path is not None:
        core["dataset_manifest"] = {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
        }
    else:
        renderer_source = Path(__file__).resolve().with_name("render_replay_v03.py")
        core["contract_scope"] = "source_index_before_dataset_build"
        core["legacy_render_semantics"] = {
            "summary_schema": "lewm_rendered_vision_v03",
            "genesis_camera_fov_axis": "vertical",
            "plan_fov_axis": "horizontal",
            "square_render_makes_horizontal_equal_vertical": True,
            "renderer_source_interpretation": {
                "path": str(renderer_source),
                "sha256": sha256_file(renderer_source),
            },
        }
    artifact = dict(core)
    artifact["content_sha256"] = canonical_sha256(core)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "sha256": sha256_file(args.output.resolve()),
        "content_sha256": artifact["content_sha256"],
        "scene_count": artifact["scene_count"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
