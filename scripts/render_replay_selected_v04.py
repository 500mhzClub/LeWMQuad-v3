#!/usr/bin/env python3
"""Render a committed sparse frame set with camera and geometry parity."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

import render_replay_v03 as legacy


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _find_manifest(corpus: Path, plan: dict[str, Any]) -> Path:
    scene_id = str(plan["scene_id"])
    split = str(plan.get("split") or "")
    family = str(plan.get("scene_family") or plan.get("family") or "")
    if split and family:
        candidate = corpus / split / family / scene_id / "manifest.json"
        if candidate.is_file():
            return candidate.resolve()
    hits = list(corpus.glob(f"*/*/{scene_id}/manifest.json"))
    if len(hits) != 1:
        raise FileNotFoundError(
            f"expected one manifest for {scene_id}, found {len(hits)}"
        )
    return hits[0].resolve()


def _frame_key(frame: dict[str, Any]) -> tuple[int, int]:
    return int(frame["frame_index"]), int(frame.get("env_index", 0))


def _render_object_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    visual = manifest.get("visual_randomization") or {}
    groups = (
        ("wall", manifest.get("walls") or []),
        ("obstacle", manifest.get("obstacles") or []),
        ("landmark", manifest.get("landmarks") or []),
        ("distractor", visual.get("distractor_objects") or []),
    )
    records = []
    for group, objects in groups:
        for obj in objects:
            records.append(
                {
                    "group": group,
                    "object_id": str(obj["object_id"]),
                    "kind": str(obj["kind"]),
                    "center_xyz_m": [float(value) for value in obj["center_xyz_m"]],
                    "size_xyz_m": [float(value) for value in obj["size_xyz_m"]],
                    "rpy_rad": [
                        float(obj.get("roll_rad", 0.0)),
                        float(obj.get("pitch_rad", 0.0)),
                        float(obj.get("yaw_rad", 0.0)),
                    ],
                    "material_id": str(obj.get("material_id") or ""),
                }
            )
    return sorted(records, key=lambda item: (item["group"], item["object_id"]))


def _manifest_with_collision_distractors(
    manifest: dict[str, Any],
) -> dict[str, Any]:
    visual = manifest.get("visual_randomization") or {}
    distractors = list(visual.get("distractor_objects") or [])
    merged = dict(manifest)
    merged["obstacles"] = [*(manifest.get("obstacles") or []), *distractors]
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--scene-corpus", required=True, type=Path)
    parser.add_argument("--frame-selection", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=168)
    parser.add_argument("--textures", action="store_true")
    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0:
        raise ValueError("render dimensions must be positive")
    plan_path = args.plan.resolve()
    selection_path = args.frame_selection.resolve()
    plan = json.loads(plan_path.read_text())
    selection = json.loads(selection_path.read_text())
    scene_id = str(plan["scene_id"])
    if selection.get("schema") != "lewm_go2_selected_render_frames_v1":
        raise ValueError("unsupported selected-frame schema")
    if str(selection.get("scene_id")) != scene_id:
        raise ValueError("frame selection targets a different scene")
    raw_keys = selection.get("frame_keys")
    if not isinstance(raw_keys, list) or not raw_keys:
        raise ValueError("frame selection must be a nonempty list")
    keys = sorted({(int(value[0]), int(value[1])) for value in raw_keys})
    if len(keys) != len(raw_keys):
        raise ValueError("frame selection must be sorted and unique")
    if [list(key) for key in keys] != raw_keys:
        raise ValueError("frame selection keys are not in canonical order")
    declared_key_sha = str(selection.get("frame_key_set_sha256", ""))
    if declared_key_sha != _canonical_sha256([list(key) for key in keys]):
        raise ValueError("frame selection commitment mismatch")

    camera = plan.get("camera") or {}
    if str(camera.get("fov_axis")) != "horizontal":
        raise ValueError("v04 requires a horizontal-FOV camera plan")
    horizontal_fov = float(camera["fov_deg"])
    near_m = float(camera["near_m"])
    far_m = float(camera.get("far_m", 200.0))
    vertical_fov = math.degrees(
        2.0
        * math.atan(
            math.tan(math.radians(horizontal_fov) * 0.5)
            * float(args.height)
            / float(args.width)
        )
    )
    manifest_path = _find_manifest(args.scene_corpus.resolve(), plan)
    manifest = json.loads(manifest_path.read_text())
    if str(manifest.get("scene_id")) != scene_id:
        raise ValueError("scene manifest identity mismatch")
    object_records = _render_object_records(manifest)
    rendered_object_ids = sorted(record["object_id"] for record in object_records)
    if len(rendered_object_ids) != len(set(rendered_object_ids)):
        raise ValueError("rendered object IDs must be unique within a scene")

    import genesis as gs
    from PIL import Image

    gs.init(backend=gs.vulkan, logging_level="error")
    scene, render_camera = legacy.build_scene(
        gs,
        _manifest_with_collision_distractors(manifest),
        fov=vertical_fov,
        near=near_m,
        far=far_m,
        res=(args.width, args.height),
        textures=bool(args.textures),
    )

    out = args.out.resolve()
    rgb_dir = out / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(keys)
    rendered = []
    t0 = time.time()
    frames_path = Path(str(plan["frames_jsonl"])).resolve()
    with frames_path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            frame = json.loads(line)
            key = _frame_key(frame)
            if key not in wanted:
                continue
            camera_pose = frame.get("camera_pose_world") or {}
            render_camera.set_pose(
                pos=tuple(camera_pose["position"]),
                lookat=tuple(camera_pose["lookat"]),
                up=tuple(camera_pose["up"]),
            )
            result = render_camera.render(rgb=True, depth=False)
            rgb = result[0] if isinstance(result, (tuple, list)) else result
            image = legacy._to_hwc_uint8(rgb)
            if image.shape[:2] != (args.height, args.width):
                raise RuntimeError(
                    f"renderer returned {image.shape[:2]}, expected "
                    f"{(args.height, args.width)}"
                )
            frame_index, env_index = key
            image_path = rgb_dir / f"frame_{frame_index:06d}_env_{env_index:02d}.png"
            Image.fromarray(image).save(image_path)
            rendered.append(
                {
                    "frame_index": frame_index,
                    "env_index": env_index,
                    "timestamp_ns": int(frame["timestamp_ns"]),
                    "image_sha256": _sha256_file(image_path),
                }
            )
    rendered_keys = [(item["frame_index"], item["env_index"]) for item in rendered]
    missing = sorted(wanted - set(rendered_keys))
    if missing:
        raise ValueError(f"frame metadata is missing {len(missing)} selected keys")
    rendered.sort(key=lambda item: (item["frame_index"], item["env_index"]))
    elapsed = time.time() - t0
    summary = {
        "schema": "lewm_rendered_vision_v04",
        "render_status": "complete",
        "scene_id": scene_id,
        "split": plan.get("split"),
        "family": plan.get("scene_family") or plan.get("family"),
        "frame_count": len(rendered),
        "frame_selection": {
            "path": str(selection_path),
            "sha256": _sha256_file(selection_path),
            "frame_key_set_sha256": declared_key_sha,
        },
        "rendered_frames": rendered,
        "rendered_image_set_sha256": _canonical_sha256(rendered),
        "resolution_wh": [args.width, args.height],
        "camera_projection": {
            "model": "pinhole",
            "renderer_fov_axis": "vertical",
            "horizontal_fov_deg": horizontal_fov,
            "vertical_fov_deg": vertical_fov,
            "near_m": near_m,
            "far_m": far_m,
            "runtime_rectification_required": False,
        },
        "renderer": "genesis-0.3.14/vulkan",
        "visuals": "textured_v04" if args.textures else "material_color_v04",
        "textures_enabled": bool(args.textures),
        "object_parity": {
            "schema": "lewm_render_object_parity_v1",
            "rendered_groups": ["wall", "obstacle", "landmark", "distractor"],
            "collision_distractors_rendered": True,
            "full_box_roll_pitch_yaw_rendered": True,
            "rendered_object_count": len(object_records),
            "rendered_object_ids": rendered_object_ids,
            "rendered_object_ids_sha256": _canonical_sha256(
                rendered_object_ids
            ),
            "rendered_object_records_sha256": _canonical_sha256(object_records),
        },
        "source": {
            "plan": {"path": str(plan_path), "sha256": _sha256_file(plan_path)},
            "frames_jsonl": {
                "path": str(frames_path),
                "sha256": _sha256_file(frames_path),
            },
            "scene_manifest": {
                "path": str(manifest_path),
                "sha256": _sha256_file(manifest_path),
            },
            "renderer_source": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256_file(Path(__file__).resolve()),
            },
        },
        "elapsed_s": elapsed,
        "fps": len(rendered) / elapsed if elapsed > 0.0 else None,
        "g2_model_outputs_opened": False,
    }
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out / ".render_done").write_text(_sha256_file(summary_path) + "\n")
    print(
        f"RENDER_OK {scene_id} frames={len(rendered)} "
        f"fps={summary['fps']:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
