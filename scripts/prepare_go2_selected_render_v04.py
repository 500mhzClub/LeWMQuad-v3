#!/usr/bin/env python3
"""Freeze sparse v04 render tasks from the label-independent v2 row set."""
from __future__ import annotations

import argparse
import hashlib
import json
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


def _write_stable(path: Path, payload: Any, *, jsonl: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if jsonl:
        text = "".join(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
            for value in payload
        )
    else:
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != text:
        raise FileExistsError(f"refusing to replace a different artifact: {path}")
    path.write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--expected-rows-sha256", required=True)
    parser.add_argument("--expected-role-assignments-sha256", required=True)
    parser.add_argument("--expected-g2-set-sha256", required=True)
    args = parser.parse_args()

    dataset_path = args.dataset_manifest.resolve()
    source_index_path = args.source_index.resolve()
    if _sha256_file(dataset_path) != args.expected_dataset_sha256:
        raise ValueError("dataset manifest differs from the frozen input")
    dataset = _read_json(dataset_path)
    if dataset.get("schema") != "lewm_go2_paired_navigation_dataset_v2":
        raise ValueError("selected v04 rendering requires the frozen dataset v2")
    rows_path = Path(str(dataset["index"]["path"])).resolve()
    if (
        _sha256_file(rows_path) != args.expected_rows_sha256
        or str(dataset["index"]["sha256"]) != args.expected_rows_sha256
    ):
        raise ValueError("row index differs from the frozen input")
    roles = dataset.get("scene_roles")
    if not isinstance(roles, dict):
        raise ValueError("dataset has no direct scene-role contract")
    if str(roles.get("assignments_sha256")) != args.expected_role_assignments_sha256:
        raise ValueError("role assignments differ from the frozen input")
    if (
        str(roles.get("scene_id_sha256_commitments", {}).get("g2_evaluation"))
        != args.expected_g2_set_sha256
    ):
        raise ValueError("G2 scene-set commitment differs from the frozen input")

    sources = _read_jsonl(source_index_path)
    source_by_scene = {str(row["scene_id"]): row for row in sources}
    if len(source_by_scene) != len(sources):
        raise ValueError("source index contains duplicate scene IDs")
    assignments = {str(key): str(value) for key, value in roles["assignments"].items()}
    if set(source_by_scene) != set(assignments):
        raise ValueError("source index and role scene sets differ")

    frame_keys_by_scene: dict[str, set[tuple[int, int]]] = {
        scene_id: set() for scene_id in source_by_scene
    }
    row_counts = {scene_id: 0 for scene_id in source_by_scene}
    rows = _read_jsonl(rows_path)
    for row in rows:
        scene_id = str(row["scene_id"])
        if scene_id not in frame_keys_by_scene:
            raise ValueError(f"row references an unknown scene: {scene_id}")
        if str(row.get("dataset_role")) != assignments[scene_id]:
            raise ValueError("row role differs from the frozen assignment")
        env_index = int(row["env_index"])
        frame_keys_by_scene[scene_id].add(
            (int(row["current_frame_index"]), env_index)
        )
        frame_keys_by_scene[scene_id].add(
            (int(row["next_frame_index"]), env_index)
        )
        row_counts[scene_id] += 1

    output_root = args.output_root.resolve()
    selections_dir = output_root / "frame_selections"
    renders_dir = output_root / "scenes"
    tasks = []
    frame_commitments = []
    for scene_id in sorted(source_by_scene):
        source = source_by_scene[scene_id]
        keys = sorted(frame_keys_by_scene[scene_id])
        if not keys:
            raise ValueError(f"scene has no selected frame keys: {scene_id}")
        scene_digest = hashlib.sha256(scene_id.encode()).hexdigest()
        selection_core = {
            "schema": "lewm_go2_selected_render_frames_v1",
            "scene_id": scene_id,
            "scene_id_sha256": scene_digest,
            "dataset_role": assignments[scene_id],
            "row_count": row_counts[scene_id],
            "frame_count": len(keys),
            "frame_keys": [list(key) for key in keys],
            "frame_key_set_sha256": _canonical_sha256(
                [list(key) for key in keys]
            ),
            "source_rows": {
                "path": str(rows_path),
                "sha256": _sha256_file(rows_path),
            },
            "g2_images_opened": False,
            "g2_label_shards_opened": False,
        }
        selection = {
            **selection_core,
            "content_sha256": _canonical_sha256(selection_core),
        }
        selection_path = selections_dir / f"scene_{scene_digest[:16]}.json"
        _write_stable(selection_path, selection)
        render_dir = renders_dir / f"scene_{scene_digest[:16]}"
        task = {
            "scene_id": scene_id,
            "scene_id_sha256": scene_digest,
            "dataset_role": assignments[scene_id],
            "plan_path": str(Path(str(source["render_plan_path"])).resolve()),
            "scene_corpus": str(Path(str(source["origin_scene_corpus"])).resolve()),
            "frame_selection_path": str(selection_path),
            "render_output_dir": str(render_dir),
            "expected_frame_count": len(keys),
        }
        tasks.append(task)
        frame_commitments.append(
            [scene_digest, selection["frame_key_set_sha256"]]
        )

    tasks_path = output_root / "render_tasks.jsonl"
    _write_stable(tasks_path, tasks, jsonl=True)
    core = {
        "schema": "lewm_go2_selected_render_plan_v1",
        "dataset_manifest": {
            "path": str(dataset_path),
            "sha256": _sha256_file(dataset_path),
        },
        "dataset_rows": {"path": str(rows_path), "sha256": _sha256_file(rows_path)},
        "source_index": {
            "path": str(source_index_path),
            "sha256": _sha256_file(source_index_path),
        },
        "scene_count": len(tasks),
        "row_count": len(rows),
        "selected_frame_count": sum(item["expected_frame_count"] for item in tasks),
        "role_scene_counts": {
            role: sum(value == role for value in assignments.values())
            for role in sorted(set(assignments.values()))
        },
        "role_assignments_sha256": args.expected_role_assignments_sha256,
        "g2_scene_set_sha256": args.expected_g2_set_sha256,
        "frame_selection_set_sha256": _canonical_sha256(frame_commitments),
        "render_contract": {
            "schema": "lewm_rendered_vision_v04",
            "resolution_wh": [224, 168],
            "horizontal_fov_deg": 78.323,
            "vertical_fov_deg": 62.8370386364,
            "collision_distractors_rendered": True,
            "full_box_roll_pitch_yaw_rendered": True,
            "textures_enabled": True,
        },
        "tasks": {"path": str(tasks_path), "sha256": _sha256_file(tasks_path)},
        "g2_row_metadata_read": True,
        "g2_images_opened": False,
        "g2_label_shards_opened": False,
        "g2_model_outputs_opened": False,
    }
    plan = {**core, "content_sha256": _canonical_sha256(core)}
    plan_path = output_root / "render_plan.json"
    _write_stable(plan_path, plan)
    print(
        json.dumps(
            {
                "output": str(plan_path),
                "sha256": _sha256_file(plan_path),
                "content_sha256": plan["content_sha256"],
                "scene_count": len(tasks),
                "selected_frame_count": core["selected_frame_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
