from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import types

import numpy as np
import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import finalize_go2_selected_render_v04 as finalize_v04  # noqa: E402
import prepare_go2_selected_render_v04 as prepare_v04  # noqa: E402
import render_replay_selected_v04 as render_v04  # noqa: E402


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
    )


def _box(
    object_id: str,
    *,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
) -> dict:
    return {
        "object_id": object_id,
        "kind": "box",
        "center_xyz_m": [1.0, 2.0, 0.5],
        "size_xyz_m": [0.4, 0.6, 1.0],
        "roll_rad": roll,
        "pitch_rad": pitch,
        "yaw_rad": yaw,
        "material_id": f"material_{object_id}",
    }


def test_prepare_sparse_v04_plan_is_identity_and_provenance_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene_id = "synthetic_scene"
    rows_path = tmp_path / "rows.jsonl"
    rows = [
        {
            "scene_id": scene_id,
            "dataset_role": "train",
            "env_index": 1,
            "current_frame_index": 4,
            "next_frame_index": 7,
        },
        {
            "scene_id": scene_id,
            "dataset_role": "train",
            "env_index": 1,
            "current_frame_index": 2,
            "next_frame_index": 4,
        },
    ]
    _write_jsonl(rows_path, rows)
    rows_sha = _sha256_file(rows_path)
    assignments_sha = "a" * 64
    g2_set_sha = "b" * 64
    dataset_path = tmp_path / "dataset_manifest.json"
    _write_json(
        dataset_path,
        {
            "schema": "lewm_go2_paired_navigation_dataset_v2",
            "index": {"path": str(rows_path), "sha256": rows_sha},
            "scene_roles": {
                "assignments": {scene_id: "train"},
                "assignments_sha256": assignments_sha,
                "scene_id_sha256_commitments": {
                    "g2_evaluation": g2_set_sha,
                },
            },
        },
    )
    dataset_sha = _sha256_file(dataset_path)
    source_index_path = tmp_path / "source_index.jsonl"
    _write_jsonl(
        source_index_path,
        [
            {
                "scene_id": scene_id,
                "render_plan_path": str(tmp_path / "source_plan.json"),
                "origin_scene_corpus": str(tmp_path / "scene_corpus"),
            }
        ],
    )
    output_root = tmp_path / "selected_v04"
    argv = [
        "prepare_go2_selected_render_v04.py",
        "--dataset-manifest",
        str(dataset_path),
        "--source-index",
        str(source_index_path),
        "--output-root",
        str(output_root),
        "--expected-dataset-sha256",
        dataset_sha,
        "--expected-rows-sha256",
        rows_sha,
        "--expected-role-assignments-sha256",
        assignments_sha,
        "--expected-g2-set-sha256",
        g2_set_sha,
    ]
    monkeypatch.setattr(sys, "argv", argv)

    assert prepare_v04.main() == 0
    first_plan_bytes = (output_root / "render_plan.json").read_bytes()
    assert prepare_v04.main() == 0
    assert (output_root / "render_plan.json").read_bytes() == first_plan_bytes

    scene_digest = hashlib.sha256(scene_id.encode()).hexdigest()
    selection_path = (
        output_root / "frame_selections" / f"scene_{scene_digest[:16]}.json"
    )
    selection = json.loads(selection_path.read_text())
    assert selection["scene_id_sha256"] == scene_digest
    assert selection["frame_keys"] == [[2, 1], [4, 1], [7, 1]]
    assert selection["frame_key_set_sha256"] == _canonical_sha256(
        selection["frame_keys"]
    )
    selection_core = dict(selection)
    declared_selection_sha = selection_core.pop("content_sha256")
    assert declared_selection_sha == _canonical_sha256(selection_core)

    tasks = [json.loads(line) for line in (output_root / "render_tasks.jsonl").read_text().splitlines()]
    assert tasks == [
        {
            "dataset_role": "train",
            "expected_frame_count": 3,
            "frame_selection_path": str(selection_path.resolve()),
            "plan_path": str((tmp_path / "source_plan.json").resolve()),
            "render_output_dir": str(
                (output_root / "scenes" / f"scene_{scene_digest[:16]}").resolve()
            ),
            "scene_corpus": str((tmp_path / "scene_corpus").resolve()),
            "scene_id": scene_id,
            "scene_id_sha256": scene_digest,
        }
    ]
    plan = json.loads((output_root / "render_plan.json").read_text())
    plan_core = dict(plan)
    declared_plan_sha = plan_core.pop("content_sha256")
    assert declared_plan_sha == _canonical_sha256(plan_core)
    assert plan["dataset_manifest"]["sha256"] == dataset_sha
    assert plan["dataset_rows"]["sha256"] == rows_sha
    assert plan["source_index"]["sha256"] == _sha256_file(source_index_path)
    assert plan["role_assignments_sha256"] == assignments_sha
    assert plan["g2_scene_set_sha256"] == g2_set_sha
    assert plan["selected_frame_count"] == 3
    assert plan["g2_images_opened"] is False

    bad_argv = list(argv)
    bad_argv[bad_argv.index(dataset_sha)] = "0" * 64
    monkeypatch.setattr(sys, "argv", bad_argv)
    with pytest.raises(ValueError, match="dataset manifest differs"):
        prepare_v04.main()


def test_renderer_object_records_and_collision_distractor_merge() -> None:
    manifest = {
        "walls": [_box("wall", roll=0.1)],
        "obstacles": [_box("obstacle", pitch=0.2)],
        "landmarks": [_box("landmark", yaw=0.3)],
        "visual_randomization": {
            "distractor_objects": [_box("distractor", roll=0.4, yaw=0.5)],
            "material_overrides": [{"material_id": "floor"}],
        },
    }

    records = render_v04._render_object_records(manifest)
    by_id = {record["object_id"]: record for record in records}
    assert set(by_id) == {"wall", "obstacle", "landmark", "distractor"}
    assert by_id["distractor"]["group"] == "distractor"
    assert by_id["distractor"]["rpy_rad"] == pytest.approx([0.4, 0.0, 0.5])
    assert by_id["obstacle"]["material_id"] == "material_obstacle"

    merged = render_v04._manifest_with_collision_distractors(manifest)
    assert [obj["object_id"] for obj in merged["obstacles"]] == [
        "obstacle",
        "distractor",
    ]
    assert [obj["object_id"] for obj in manifest["obstacles"]] == ["obstacle"]
    assert merged["visual_randomization"] == manifest["visual_randomization"]


def test_renderer_main_converts_horizontal_fov_without_loading_genesis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene_id = "projection_scene"
    corpus = tmp_path / "corpus"
    manifest_path = corpus / "train" / "unit" / scene_id / "manifest.json"
    manifest = {
        "scene_id": scene_id,
        "walls": [_box("wall")],
        "obstacles": [_box("obstacle")],
        "landmarks": [_box("landmark")],
        "visual_randomization": {
            "distractor_objects": [_box("distractor")],
        },
    }
    _write_json(manifest_path, manifest)
    frames_path = tmp_path / "frames.jsonl"
    _write_jsonl(
        frames_path,
        [
            {
                "frame_index": 3,
                "env_index": 0,
                "timestamp_ns": 123,
                "camera_pose_world": {
                    "position": [0.0, 0.0, 1.0],
                    "lookat": [1.0, 0.0, 1.0],
                    "up": [0.0, 0.0, 1.0],
                },
            }
        ],
    )
    plan_path = tmp_path / "source_plan.json"
    _write_json(
        plan_path,
        {
            "scene_id": scene_id,
            "split": "train",
            "scene_family": "unit",
            "frames_jsonl": str(frames_path),
            "camera": {
                "fov_axis": "horizontal",
                "fov_deg": 78.323,
                "near_m": 0.05,
                "far_m": 200.0,
            },
        },
    )
    frame_keys = [[3, 0]]
    selection_path = tmp_path / "selection.json"
    _write_json(
        selection_path,
        {
            "schema": "lewm_go2_selected_render_frames_v1",
            "scene_id": scene_id,
            "frame_keys": frame_keys,
            "frame_key_set_sha256": _canonical_sha256(frame_keys),
        },
    )

    captured: dict = {}

    class FakeCamera:
        def set_pose(self, **kwargs) -> None:
            captured["pose"] = kwargs

        def render(self, *, rgb: bool, depth: bool) -> np.ndarray:
            assert rgb is True and depth is False
            return np.zeros((168, 224, 3), dtype=np.uint8)

    def fake_build_scene(_gs, passed_manifest, **kwargs):
        captured["manifest"] = passed_manifest
        captured["build_kwargs"] = kwargs
        return object(), FakeCamera()

    fake_genesis = types.ModuleType("genesis")
    fake_genesis.vulkan = object()
    fake_genesis.init = lambda **kwargs: captured.setdefault("init", kwargs)
    monkeypatch.setitem(sys.modules, "genesis", fake_genesis)
    monkeypatch.setattr(render_v04.legacy, "build_scene", fake_build_scene)
    out = tmp_path / "render"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render_replay_selected_v04.py",
            "--plan",
            str(plan_path),
            "--scene-corpus",
            str(corpus),
            "--frame-selection",
            str(selection_path),
            "--out",
            str(out),
            "--width",
            "224",
            "--height",
            "168",
            "--textures",
        ],
    )

    assert render_v04.main() == 0
    expected_vertical_fov = math.degrees(
        2.0 * math.atan(math.tan(math.radians(78.323) * 0.5) * 168.0 / 224.0)
    )
    assert captured["build_kwargs"]["fov"] == pytest.approx(expected_vertical_fov)
    assert captured["build_kwargs"]["res"] == (224, 168)
    assert captured["build_kwargs"]["near"] == pytest.approx(0.05)
    assert captured["build_kwargs"]["textures"] is True
    assert [
        obj["object_id"] for obj in captured["manifest"]["obstacles"]
    ] == ["obstacle", "distractor"]
    summary = json.loads((out / "summary.json").read_text())
    assert summary["camera_projection"] == {
        "far_m": 200.0,
        "horizontal_fov_deg": 78.323,
        "model": "pinhole",
        "near_m": 0.05,
        "renderer_fov_axis": "vertical",
        "runtime_rectification_required": False,
        "vertical_fov_deg": expected_vertical_fov,
    }
    assert summary["resolution_wh"] == [224, 168]
    assert summary["object_parity"]["rendered_object_count"] == 4
    assert (out / ".render_done").read_text().strip() == _sha256_file(
        out / "summary.json"
    )


def _make_finalizer_fixture(root: Path) -> dict[str, Path]:
    scene_id = "finalizer_scene"
    manifest_path = root / "manifest.json"
    _write_json(
        manifest_path,
        {
            "scene_id": scene_id,
            "walls": [{"object_id": "wall"}],
            "obstacles": [{"object_id": "obstacle"}],
            "landmarks": [{"object_id": "landmark"}],
            "visual_randomization": {
                "distractor_objects": [{"object_id": "distractor"}],
            },
        },
    )
    frame_keys = [[3, 0]]
    selection_core = {
        "schema": "lewm_go2_selected_render_frames_v1",
        "scene_id": scene_id,
        "scene_id_sha256": hashlib.sha256(scene_id.encode()).hexdigest(),
        "dataset_role": "train",
        "row_count": 1,
        "frame_count": 1,
        "frame_keys": frame_keys,
        "frame_key_set_sha256": _canonical_sha256(frame_keys),
    }
    selection_path = root / "selection.json"
    _write_json(
        selection_path,
        {**selection_core, "content_sha256": _canonical_sha256(selection_core)},
    )
    render_dir = root / "render"
    image_path = render_dir / "rgb" / "frame_000003_env_00.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"committed sparse RGB bytes")
    rendered_frames = [
        {
            "frame_index": 3,
            "env_index": 0,
            "timestamp_ns": 123,
            "image_sha256": _sha256_file(image_path),
        }
    ]
    object_ids = ["distractor", "landmark", "obstacle", "wall"]
    renderer_source_path = root / "frozen_renderer.py"
    renderer_source_path.write_text("# frozen renderer fixture\n")
    vertical_fov = math.degrees(
        2.0 * math.atan(math.tan(math.radians(78.323) * 0.5) * 168.0 / 224.0)
    )
    summary = {
        "schema": "lewm_rendered_vision_v04",
        "render_status": "complete",
        "scene_id": scene_id,
        "frame_count": 1,
        "rendered_frames": rendered_frames,
        "rendered_image_set_sha256": _canonical_sha256(rendered_frames),
        "resolution_wh": [224, 168],
        "camera_projection": {
            "model": "pinhole",
            "renderer_fov_axis": "vertical",
            "horizontal_fov_deg": 78.323,
            "vertical_fov_deg": vertical_fov,
            "near_m": 0.05,
            "far_m": 200.0,
            "runtime_rectification_required": False,
        },
        "object_parity": {
            "schema": "lewm_render_object_parity_v1",
            "collision_distractors_rendered": True,
            "full_box_roll_pitch_yaw_rendered": True,
            "rendered_object_count": len(object_ids),
            "rendered_object_ids": object_ids,
            "rendered_object_ids_sha256": _canonical_sha256(object_ids),
        },
        "source": {
            "scene_manifest": {
                "path": str(manifest_path),
                "sha256": _sha256_file(manifest_path),
            },
            "renderer_source": {
                "path": str(renderer_source_path),
                "sha256": _sha256_file(renderer_source_path),
            },
        },
    }
    summary_path = render_dir / "summary.json"
    _write_json(summary_path, summary)
    done_path = render_dir / ".render_done"
    done_path.write_text(_sha256_file(summary_path) + "\n")
    task = {
        "scene_id": scene_id,
        "scene_id_sha256": hashlib.sha256(scene_id.encode()).hexdigest(),
        "dataset_role": "train",
        "frame_selection_path": str(selection_path),
        "render_output_dir": str(render_dir),
        "expected_frame_count": 1,
    }
    tasks_path = root / "tasks.jsonl"
    _write_jsonl(tasks_path, [task])
    plan_core = {
        "schema": "lewm_go2_selected_render_plan_v1",
        "scene_count": 1,
        "tasks": {"path": str(tasks_path), "sha256": _sha256_file(tasks_path)},
    }
    plan_path = root / "render_plan.json"
    _write_json(
        plan_path,
        {**plan_core, "content_sha256": _canonical_sha256(plan_core)},
    )
    legacy_index_path = root / "legacy_sources.jsonl"
    _write_jsonl(
        legacy_index_path,
        [
            {
                "schema": "lewm_go2_navigation_source_v1",
                "scene_id": scene_id,
                "family": "unit",
                "hashes": {"legacy_summary_sha256": "c" * 64},
            }
        ],
    )
    return {
        "plan": plan_path,
        "legacy_index": legacy_index_path,
        "output_index": root / "sparse_sources.jsonl",
        "report": root / "audit.json",
        "summary": summary_path,
        "done": done_path,
        "image": image_path,
        "render_dir": render_dir,
    }


def _run_finalizer(
    fixture: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> int:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finalize_go2_selected_render_v04.py",
            "--render-plan",
            str(fixture["plan"]),
            "--legacy-source-index",
            str(fixture["legacy_index"]),
            "--output-source-index",
            str(fixture["output_index"]),
            "--output-report",
            str(fixture["report"]),
        ],
    )
    return finalize_v04.main()


def test_finalizer_emits_provenance_bound_sparse_source_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_finalizer_fixture(tmp_path)

    assert _run_finalizer(fixture, monkeypatch) == 0
    rows = [json.loads(line) for line in fixture["output_index"].read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["schema"] == "lewm_go2_navigation_source_v2_sparse_rgb"
    assert row["rendered_frame_count"] == 1
    assert row["rgb_dir"] == str(fixture["render_dir"] / "rgb")
    assert row["render_summary_path"] == str(fixture["summary"])
    assert row["image_validation"] == "all_committed_sparse_frames_sha256"
    assert row["hashes"]["legacy_summary_sha256"] == "c" * 64
    assert row["hashes"]["render_summary_file_sha256"] == _sha256_file(
        fixture["summary"]
    )
    assert row["render_contract"]["schema"] == "lewm_rendered_vision_v04"
    assert row["render_contract"]["resolution_wh"] == [224, 168]
    report = json.loads(fixture["report"].read_text())
    report_core = dict(report)
    declared = report_core.pop("content_sha256")
    assert declared == _canonical_sha256(report_core)
    assert report["output_source_index"]["sha256"] == _sha256_file(
        fixture["output_index"]
    )
    assert report["scene_count"] == 1
    assert report["frame_count"] == 1
    assert report["g2_images_decoded_or_inspected"] is False


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("object_hash", "rendered object parity mismatch"),
        ("camera", "camera projection mismatch"),
        ("image", "rendered image hash mismatch"),
    ],
)
def test_finalizer_rejects_render_contract_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
    message: str,
) -> None:
    fixture = _make_finalizer_fixture(tmp_path)
    if tamper == "image":
        fixture["image"].write_bytes(b"tampered sparse RGB bytes")
    else:
        summary = json.loads(fixture["summary"].read_text())
        if tamper == "object_hash":
            summary["object_parity"]["rendered_object_ids_sha256"] = "0" * 64
        elif tamper == "camera":
            summary["camera_projection"]["horizontal_fov_deg"] = 70.0
        _write_json(fixture["summary"], summary)
        fixture["done"].write_text(_sha256_file(fixture["summary"]) + "\n")

    with pytest.raises(ValueError, match=message):
        _run_finalizer(fixture, monkeypatch)
