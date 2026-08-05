#!/usr/bin/env python3
"""Audit sparse v04 renders and emit a provenance-complete source index."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    result = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        result.append(value)
    return result


def _object_ids(manifest: dict[str, Any]) -> list[str]:
    visual = manifest.get("visual_randomization") or {}
    groups = (
        manifest.get("walls") or [],
        manifest.get("obstacles") or [],
        manifest.get("landmarks") or [],
        visual.get("distractor_objects") or [],
    )
    result = sorted(str(obj["object_id"]) for group in groups for obj in group)
    if len(result) != len(set(result)):
        raise ValueError("manifest collision object IDs are not unique")
    return result


def _validate_content_hash(payload: dict[str, Any], *, name: str) -> None:
    core = dict(payload)
    declared = str(core.pop("content_sha256", ""))
    if declared != _canonical_sha256(core):
        raise ValueError(f"{name} content hash mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-plan", type=Path, required=True)
    parser.add_argument("--legacy-source-index", type=Path, required=True)
    parser.add_argument("--output-source-index", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    render_plan_path = args.render_plan.resolve()
    legacy_index_path = args.legacy_source_index.resolve()
    plan = _read_json(render_plan_path)
    if plan.get("schema") != "lewm_go2_selected_render_plan_v1":
        raise ValueError("unsupported selected-render plan schema")
    _validate_content_hash(plan, name="selected-render plan")
    tasks_path = Path(str(plan["tasks"]["path"])).resolve()
    if _sha256_file(tasks_path) != str(plan["tasks"]["sha256"]):
        raise ValueError("render task file changed after planning")
    tasks = _read_jsonl(tasks_path)
    legacy_rows = _read_jsonl(legacy_index_path)
    legacy_by_scene = {str(row["scene_id"]): row for row in legacy_rows}
    if len(legacy_by_scene) != len(legacy_rows):
        raise ValueError("legacy source index contains duplicate scenes")
    if {str(task["scene_id"]) for task in tasks} != set(legacy_by_scene):
        raise ValueError("task and source-index scene sets differ")

    output_rows = []
    role_counts: Counter[str] = Counter()
    total_frames = 0
    total_objects = 0
    renderer_sources = set()
    scene_audits = []
    for task in sorted(tasks, key=lambda value: str(value["scene_id"])):
        scene_id = str(task["scene_id"])
        role = str(task["dataset_role"])
        role_counts[role] += 1
        selection_path = Path(str(task["frame_selection_path"])).resolve()
        selection = _read_json(selection_path)
        _validate_content_hash(selection, name=f"frame selection {scene_id}")
        render_dir = Path(str(task["render_output_dir"])).resolve()
        summary_path = render_dir / "summary.json"
        done_path = render_dir / ".render_done"
        if not summary_path.is_file() or not done_path.is_file():
            raise ValueError(f"render is incomplete: {scene_id}")
        summary_sha = _sha256_file(summary_path)
        if done_path.read_text().strip() != summary_sha:
            raise ValueError(f"render completion marker mismatch: {scene_id}")
        summary = _read_json(summary_path)
        expected_count = int(task["expected_frame_count"])
        if (
            summary.get("schema") != "lewm_rendered_vision_v04"
            or summary.get("render_status") != "complete"
            or str(summary.get("scene_id")) != scene_id
            or int(summary.get("frame_count", -1)) != expected_count
        ):
            raise ValueError(f"render summary contract mismatch: {scene_id}")
        projection = summary.get("camera_projection")
        if not isinstance(projection, dict) or (
            not math.isclose(
                float(projection.get("horizontal_fov_deg")),
                78.323,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(projection.get("vertical_fov_deg")),
                62.837038636424516,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or float(projection.get("near_m")) != 0.05
            or projection.get("renderer_fov_axis") != "vertical"
            or projection.get("runtime_rectification_required") is not False
        ):
            raise ValueError(f"camera projection mismatch: {scene_id}")
        if summary.get("resolution_wh") != [224, 168]:
            raise ValueError(f"render resolution mismatch: {scene_id}")
        parity = summary.get("object_parity")
        manifest_path = Path(str(summary["source"]["scene_manifest"]["path"])).resolve()
        manifest = _read_json(manifest_path)
        expected_ids = _object_ids(manifest)
        if (
            not isinstance(parity, dict)
            or parity.get("schema") != "lewm_render_object_parity_v1"
            or parity.get("collision_distractors_rendered") is not True
            or parity.get("full_box_roll_pitch_yaw_rendered") is not True
            or parity.get("rendered_object_ids") != expected_ids
            or parity.get("rendered_object_ids_sha256")
            != _canonical_sha256(expected_ids)
            or int(parity.get("rendered_object_count", -1)) != len(expected_ids)
        ):
            raise ValueError(f"rendered object parity mismatch: {scene_id}")
        if _sha256_file(manifest_path) != str(
            summary["source"]["scene_manifest"]["sha256"]
        ):
            raise ValueError(f"rendered manifest hash mismatch: {scene_id}")
        expected_keys = [tuple(value) for value in selection["frame_keys"]]
        rendered_frames = summary.get("rendered_frames")
        if not isinstance(rendered_frames, list):
            raise ValueError(f"rendered frame records are absent: {scene_id}")
        actual_keys = [
            (int(frame["frame_index"]), int(frame["env_index"]))
            for frame in rendered_frames
        ]
        if actual_keys != expected_keys:
            raise ValueError(f"rendered frame set mismatch: {scene_id}")
        for frame in rendered_frames:
            image_path = render_dir / "rgb" / (
                f"frame_{int(frame['frame_index']):06d}_"
                f"env_{int(frame['env_index']):02d}.png"
            )
            if _sha256_file(image_path) != str(frame["image_sha256"]):
                raise ValueError(f"rendered image hash mismatch: {image_path}")
        renderer_source = summary["source"]["renderer_source"]
        renderer_source_path = Path(str(renderer_source["path"])).resolve()
        if _sha256_file(renderer_source_path) != str(renderer_source["sha256"]):
            raise ValueError("renderer source changed during the render batch")
        renderer_sources.add(
            (str(renderer_source_path), str(renderer_source["sha256"]))
        )
        total_frames += expected_count
        total_objects += len(expected_ids)

        legacy = dict(legacy_by_scene[scene_id])
        legacy["schema"] = "lewm_go2_navigation_source_v2_sparse_rgb"
        legacy["rendered_frame_count"] = expected_count
        legacy["render_summary_path"] = str(summary_path)
        legacy["rgb_dir"] = str(render_dir / "rgb")
        legacy["hashes"] = dict(legacy["hashes"])
        legacy["hashes"]["render_summary_file_sha256"] = summary_sha
        legacy["image_validation"] = "all_committed_sparse_frames_sha256"
        legacy["render_contract"] = {
            "schema": summary["schema"],
            "frame_key_set_sha256": selection["frame_key_set_sha256"],
            "rendered_image_set_sha256": summary["rendered_image_set_sha256"],
            "object_ids_sha256": parity["rendered_object_ids_sha256"],
            "camera_projection": projection,
            "resolution_wh": summary["resolution_wh"],
            "renderer_source_sha256": renderer_source["sha256"],
        }
        output_rows.append(legacy)
        scene_audits.append(
            {
                "scene_id_sha256": task["scene_id_sha256"],
                "dataset_role": role,
                "frame_count": expected_count,
                "frame_key_set_sha256": selection["frame_key_set_sha256"],
                "summary_sha256": summary_sha,
                "object_ids_sha256": parity["rendered_object_ids_sha256"],
            }
        )

    if len(renderer_sources) != 1:
        raise ValueError("render batch mixes renderer source identities")
    output_index_path = args.output_source_index.resolve()
    output_index_path.parent.mkdir(parents=True, exist_ok=True)
    output_text = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in output_rows
    )
    if output_index_path.exists() and output_index_path.read_text() != output_text:
        raise FileExistsError("refusing to replace a different source index")
    output_index_path.write_text(output_text)
    renderer_path, renderer_sha = next(iter(renderer_sources))
    report_core = {
        "schema": "lewm_go2_selected_render_audit_v1",
        "render_plan": {
            "path": str(render_plan_path),
            "sha256": _sha256_file(render_plan_path),
            "content_sha256": plan["content_sha256"],
        },
        "legacy_source_index": {
            "path": str(legacy_index_path),
            "sha256": _sha256_file(legacy_index_path),
        },
        "output_source_index": {
            "path": str(output_index_path),
            "sha256": _sha256_file(output_index_path),
        },
        "renderer_source": {"path": renderer_path, "sha256": renderer_sha},
        "scene_count": len(output_rows),
        "frame_count": total_frames,
        "rendered_object_instance_count": total_objects,
        "role_scene_counts": dict(sorted(role_counts.items())),
        "scene_audits": scene_audits,
        "camera_projection": {
            "resolution_wh": [224, 168],
            "horizontal_fov_deg": 78.323,
            "vertical_fov_deg": 62.837038636424516,
            "near_m": 0.05,
            "runtime_rectification_required": False,
        },
        "object_contract": {
            "rendered_groups": ["wall", "obstacle", "landmark", "distractor"],
            "collision_distractors_rendered": True,
            "full_box_roll_pitch_yaw_rendered": True,
        },
        "g2_row_metadata_read": True,
        "g2_image_bytes_hashed_for_integrity": True,
        "g2_images_decoded_or_inspected": False,
        "g2_image_content_metrics_computed": False,
        "g2_label_shards_opened": False,
        "g2_model_outputs_opened": False,
    }
    report = {
        **report_core,
        "content_sha256": _canonical_sha256(report_core),
    }
    report_path = args.output_report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if report_path.exists() and report_path.read_text() != report_text:
        raise FileExistsError("refusing to replace a different render audit")
    report_path.write_text(report_text)
    print(
        json.dumps(
            {
                "source_index": str(output_index_path),
                "source_index_sha256": _sha256_file(output_index_path),
                "report": str(report_path),
                "report_sha256": _sha256_file(report_path),
                "content_sha256": report["content_sha256"],
                "scene_count": len(output_rows),
                "frame_count": total_frames,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
