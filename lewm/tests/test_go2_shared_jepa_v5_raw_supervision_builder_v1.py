from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from lewm.datasets import go2_shared_jepa_v5_raw_supervision_builder_v1 as builder
from lewm_worlds.manifest import (
    CameraValidityConstraints,
    SceneManifest,
    SpawnSpec,
    manifest_sha256,
)


def _hashed(core: dict[str, object]) -> dict[str, object]:
    return {**core, "content_sha256": builder.canonical_json_sha256(core)}


def _synthetic_inputs(scene_count: int = 2):
    source_jobs = builder.v4_builder.synthetic_scene_jobs(2 * scene_count)
    jobs = []
    pairs = []
    for scene_index in range(scene_count):
        scene_id = f"synthetic_shared_scene_{scene_index:02d}"
        family = f"synthetic_family_{scene_index % 2}"
        prepared = []
        endpoint_hashes = []
        for side_index, side in enumerate(("current", "next")):
            source_frame = source_jobs[2 * scene_index + side_index].frames[0]
            identity = {
                "dataset_role": "train",
                "scene_id": scene_id,
                "episode_id": f"episode_{scene_index}",
                "env_index": 0,
                "episode_step": side_index,
                "frame_index": 2 * scene_index + side_index,
                "timestamp_ns": 1_000_000 + 2 * scene_index + side_index,
                "image_sha256": source_frame.image_sha256,
            }
            identity_hash = builder.canonical_json_sha256(identity)
            endpoint = _hashed(
                {
                    "schema": builder.plan_v5.ENDPOINT_SCHEMA,
                    "identity": identity,
                    "identity_sha256": identity_hash,
                    "image_path_metadata_only": source_frame.image_path_metadata_only,
                    "frames_jsonl_sha256": hashlib.sha256(
                        f"frames:{scene_id}".encode()
                    ).hexdigest(),
                    "scene_manifest_sha256": hashlib.sha256(
                        f"manifest:{scene_id}".encode()
                    ).hexdigest(),
                    "base_quat_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "stored_base_yaw_rad": 0.0,
                }
            )
            prepared.append(
                builder.PreparedEndpointV1(
                    plan_endpoint=endpoint,
                    family=family,
                    frame=source_frame,
                )
            )
            endpoint_hashes.append(identity_hash)
        jobs.append(
            builder.PreparedSceneJobV1(
                scene_id=scene_id,
                role="train",
                family=family,
                endpoints=tuple(prepared),
            )
        )
        pairs.append(
            _hashed(
                {
                    "schema": builder.plan_v5.PAIR_SCHEMA,
                    "dataset_role": "train",
                    "global_row": scene_index,
                    "scene_id": scene_id,
                    "family": family,
                    "current_endpoint_sha256": endpoint_hashes[0],
                    "next_endpoint_sha256": endpoint_hashes[1],
                }
            )
        )
    return tuple(jobs), tuple(pairs)


def _file_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): builder._sha256_file(path)
        for path in sorted(root.rglob("*"), key=str)
        if path.is_file()
    }


def test_one_and_six_workers_publish_byte_identical_full_artifacts(
    tmp_path: Path,
) -> None:
    jobs, pairs = _synthetic_inputs(6)
    one = tmp_path / "one"
    six = tmp_path / "six"
    first = builder.build_prepared_dataset_v1(
        jobs,
        pairs,
        output_directory=one,
        workers=1,
        input_provenance={"schema": "synthetic_input_v1"},
        access_ledger={"schema": "synthetic_access_v1", "rgb_byte_opens": 0},
    )
    second = builder.build_prepared_dataset_v1(
        jobs,
        pairs,
        output_directory=six,
        workers=6,
        input_provenance={"schema": "synthetic_input_v1"},
        access_ledger={"schema": "synthetic_access_v1", "rgb_byte_opens": 0},
    )

    assert first == second
    assert _file_hashes(one) == _file_hashes(six)
    assert first["parallel_contract"]["worker_start_method"] == "spawn"
    assert first["parallel_contract"]["gpu_visible_to_workers"] is False
    assert all(value is False for value in first["licenses"].values())


def test_manifest_and_shards_publish_raster_bytes_and_direct_endpoint_joins(
    tmp_path: Path,
) -> None:
    jobs, pairs = _synthetic_inputs(1)
    output = tmp_path / "dataset"
    manifest = builder.build_prepared_dataset_v1(
        jobs,
        pairs,
        output_directory=output,
        workers=1,
        input_provenance={"schema": "synthetic_input_v1"},
        access_ledger={"schema": "synthetic_access_v1"},
    )
    shard_path = output / manifest["shards"][0]["path"]
    shard = json.loads(shard_path.read_text())
    by_name = {item["path"]: item for item in shard["files"]}
    assert by_name["raster_labels.u1"]["dtype"] == "|u1"
    assert by_name["raster_labels.u1"]["shape"] == [2, 64, 64]
    endpoints = [json.loads(line) for line in (output / "endpoints.jsonl").read_text().splitlines()]
    assert len(endpoints) == 2
    assert all(item["scene_shard"] == manifest["shards"][0]["path"] for item in endpoints)
    assert [item["shard_row"] for item in endpoints] == [0, 1]
    assert all(builder._is_sha256(item["evidence_content_sha256"]) for item in endpoints)
    assert all(builder._is_sha256(item["raster_content_sha256"]) for item in endpoints)
    assert set(manifest["files"][0]) == {"path", "byte_count", "file_sha256"}


def test_atomic_noreplace_preserves_an_occupied_destination(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    sentinel = destination / "sentinel"
    sentinel.write_text("foreign")
    with pytest.raises(FileExistsError):
        builder._rename_noreplace(source, destination)
    assert source.is_dir()
    assert sentinel.read_text() == "foreign"


def test_dataset_build_does_not_replace_an_existing_destination(tmp_path: Path) -> None:
    jobs, pairs = _synthetic_inputs(1)
    output = tmp_path / "dataset"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_text("foreign")
    with pytest.raises(FileExistsError, match="immutable"):
        builder.build_prepared_dataset_v1(
            jobs,
            pairs,
            output_directory=output,
            workers=1,
            input_provenance={},
            access_ledger={},
        )
    assert sentinel.read_text() == "foreign"


@pytest.mark.parametrize("workers", [0, 7])
def test_worker_count_is_bounded_before_publication(
    workers: int, tmp_path: Path
) -> None:
    jobs, pairs = _synthetic_inputs(1)
    with pytest.raises(ValueError, match="workers"):
        builder.build_prepared_dataset_v1(
            jobs,
            pairs,
            output_directory=tmp_path / "dataset",
            workers=workers,
            input_provenance={},
            access_ledger={},
        )


def test_bound_reader_rejects_symlinks_and_postread_parent_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "value.json"
    source.write_bytes(b"{}\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    alias = source_dir / "alias.json"
    alias.symlink_to(source)
    with pytest.raises(PermissionError):
        builder._read_bound_regular_file(
            repository_root=tmp_path,
            path=alias,
            expected_sha256=digest,
        )

    original_read = builder.os.read
    changed = False

    def mutating_read(descriptor: int, count: int) -> bytes:
        nonlocal changed
        payload = original_read(descriptor, count)
        if payload and not changed:
            changed = True
            os.utime(source_dir, None)
        return payload

    monkeypatch.setattr(builder.os, "read", mutating_read)
    with pytest.raises(builder.RawSupervisionBuildError, match="component changed"):
        builder._read_bound_regular_file(
            repository_root=tmp_path,
            path=source,
            expected_sha256=digest,
        )


def test_retained_parent_rejects_alias_replacement(tmp_path: Path) -> None:
    container = tmp_path / "container"
    container.mkdir()
    retained = builder._open_publication_parent(container)
    moved = tmp_path / "moved"
    try:
        container.rename(moved)
        container.mkdir()
        with pytest.raises(builder.RawSupervisionBuildError):
            retained.validate()
    finally:
        retained.close()


def test_owned_cleanup_preserves_replacement_staging(tmp_path: Path) -> None:
    container = tmp_path / "container"
    container.mkdir()
    retained = builder._open_publication_parent(container)
    try:
        os.mkdir("staging", dir_fd=retained.parent_fd)
        retained.refresh_after_owned_mutation()
        identity = builder._named_directory_identity(retained.parent_fd, "staging")
        (container / "staging").rename(container / "moved_owned")
        (container / "staging").mkdir()
        (container / "staging" / "foreign").write_text("keep")
        assert not builder._cleanup_owned_directory(retained, "staging", identity)
        assert (container / "staging" / "foreign").read_text() == "keep"
        assert (container / "moved_owned").is_dir()
    finally:
        retained.close()


def test_foreign_failure_receipt_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = tmp_path / "failure.json"
    receipt.write_text("foreign\n")
    monkeypatch.setattr(builder, "FAILURE_RECEIPT", receipt)
    builder._write_failure_receipt(
        authorization_sha256="0" * 64,
        error=RuntimeError("synthetic"),
    )
    assert receipt.read_text() == "foreign\n"


def test_failure_receipt_is_exclusive_self_hashed_and_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = tmp_path / "failure.json"
    output = tmp_path / "dataset"
    monkeypatch.setattr(builder, "FAILURE_RECEIPT", receipt)
    monkeypatch.setattr(builder, "CANONICAL_OUTPUT", output)
    builder._write_failure_receipt(
        authorization_sha256="0" * 64,
        error=RuntimeError("synthetic"),
    )
    value = json.loads(receipt.read_text())
    core = dict(value)
    declared = core.pop("content_sha256")
    assert declared == builder.canonical_json_sha256(core)
    assert value["retry_authorized"] is False
    assert value["canonical_output_present"] is False


def test_prepublish_failure_removes_only_owned_staging(tmp_path: Path) -> None:
    jobs, pairs = _synthetic_inputs(1)
    output = tmp_path / "dataset"

    def fail_validation() -> None:
        raise RuntimeError("synthetic prepublication failure")

    with pytest.raises(RuntimeError, match="prepublication"):
        builder.build_prepared_dataset_v1(
            jobs,
            pairs,
            output_directory=output,
            workers=1,
            input_provenance={},
            access_ledger={},
            prepublication_validator=fail_validation,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".dataset.staging.*"))


def test_absent_authorization_rejects_before_metadata_or_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(builder, "AUTHORIZATION_PATH", tmp_path / "absent.json")
    monkeypatch.setattr(builder, "CANONICAL_OUTPUT", tmp_path / "output")
    monkeypatch.setattr(builder, "FAILURE_RECEIPT", tmp_path / "failed.json")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("metadata/payload opener was reached")

    monkeypatch.setattr(builder.plan_v5, "load_frozen_development_metadata", forbidden)
    with pytest.raises(PermissionError, match="authorization is absent"):
        builder.execute_exact_build_v1(authorization_sha256="0" * 64, workers=1)
    assert not (tmp_path / "failed.json").exists()


def test_exact_access_ledger_schema_has_all_forbidden_counts() -> None:
    required_zero_fields = {
        "g2_sidecar_byte_opens",
        "g2_source_payload_opens",
        "g2_label_payload_opens",
        "g2_rgb_byte_opens",
        "rgb_byte_opens",
        "rgb_decodes",
        "parent_label_shard_payload_opens",
        "checkpoint_or_model_output_opens",
        "runtime_or_navigation_result_opens",
        "heldout_or_sealed_opens",
        "hardware_or_production_opens",
        "writes_outside_output_or_failure_namespace",
        "denied_or_unexpected_accesses",
    }
    assert required_zero_fields < builder.EXACT_ACCESS_LEDGER_KEYS


def test_exact_source_loader_uses_reviewed_camera_and_object_pipeline_without_rgb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_v4, _manifest_semantics = builder._reviewed_v4_source_semantics()
    scene_id = "synthetic_exact_source_scene"
    family = source_v4.FAMILIES[0]
    manifest = SceneManifest(
        scene_id=scene_id,
        family=family,
        difficulty_tier="unit_test",
        topology_seed=1,
        visual_seed=2,
        physics_seed=3,
        world_bounds_xy_m=((-4.0, -4.0), (7.0, 4.0)),
        spawn=SpawnSpec(
            xyz_m=(0.0, 0.0, 0.35),
            quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        ),
        graph_nodes=(),
        graph_edges=(),
        obstacles=(),
        landmarks=(),
        camera_constraints=CameraValidityConstraints(
            min_wall_thickness_m=0.08,
            near_m=0.05,
            far_m=200.0,
            min_camera_clearance_m=0.10,
        ),
        split="train",
    )
    manifest_payload = builder.canonical_json_bytes(manifest.to_dict())
    frames_path = tmp_path / "frames.jsonl"
    manifest_path = tmp_path / "manifest.json"
    plan_path = tmp_path / "plan.json"
    summary_root = tmp_path / "render"
    image_path = summary_root / "rgb/frame.png"
    summary_path = summary_root / "summary.json"
    renderer_path = tmp_path / "renderer.py"
    source_frame = {
        "frame_index": 7,
        "env_index": 0,
        "timestamp_ns": 11,
        "episode": {
            "episode_id": 3,
            "reset_count": 0,
            "episode_step": 4,
            "split": "synthetic",
        },
        "base_pose_world": {
            "position": {"x": 0.0, "y": 0.0, "z": 0.35},
        },
        "base_rpy_rad": {"yaw": 0.0},
        "base_quat_world_xyzw": [0.0, 0.0, 0.0, 1.0],
        "camera_mount_body": dict(source_v4.NOMINAL_CAMERA_MOUNT_BODY),
        "camera_pose_world": {
            "position": [0.326, 0.0, 0.393],
            "lookat": [1.326, 0.0, 0.393],
            "up": [0.0, 0.0, 1.0],
        },
    }
    frames_payload = builder.canonical_json_bytes(source_frame) + b"\n"
    plan = {
        "schema": "lewm_render_replay_plan_v0",
        "scene_id": scene_id,
        "frames_jsonl": str(frames_path),
        "camera": {
            "native_resolution": [640, 480],
            "training_resolution": [224, 168],
            "fov_axis": "horizontal",
            "fov_deg": 78.323,
            "near_m": 0.05,
            "far_m": 200.0,
            "encoding": "rgb8",
            "mount_body": dict(source_v4.NOMINAL_CAMERA_MOUNT_BODY),
        },
    }
    plan_payload = builder.canonical_json_bytes(plan)
    image_sha = "a" * 64
    rendered_frames = [
        {
            "frame_index": 7,
            "env_index": 0,
            "timestamp_ns": 11,
            "image_sha256": image_sha,
        }
    ]
    render_records = source_v4.labels_v3._render_object_records(manifest)
    object_ids = sorted(str(item["object_id"]) for item in render_records)
    summary = {
        "schema": "lewm_rendered_vision_v04",
        "render_status": "complete",
        "scene_id": scene_id,
        "family": family,
        "g2_model_outputs_opened": False,
        "frame_count": 1,
        "resolution_wh": [224, 168],
        "camera_projection": {
            "model": "pinhole",
            "renderer_fov_axis": "vertical",
            "horizontal_fov_deg": 78.323,
            "vertical_fov_deg": 62.837038636424516,
            "near_m": 0.05,
            "far_m": 200.0,
            "runtime_rectification_required": False,
        },
        "rendered_frames": rendered_frames,
        "rendered_image_set_sha256": builder.canonical_json_sha256(rendered_frames),
        "object_parity": {
            "schema": "lewm_render_object_parity_v1",
            "rendered_groups": ["wall", "obstacle", "landmark", "distractor"],
            "rendered_object_count": len(render_records),
            "rendered_object_ids": object_ids,
            "rendered_object_ids_sha256": builder.canonical_json_sha256(object_ids),
            "rendered_object_records_sha256": builder.canonical_json_sha256(
                render_records
            ),
            "collision_distractors_rendered": True,
            "full_box_roll_pitch_yaw_rendered": True,
        },
        "source": {
            "plan": {
                "path": str(plan_path),
                "sha256": hashlib.sha256(plan_payload).hexdigest(),
            },
            "frames_jsonl": {
                "path": str(frames_path),
                "sha256": hashlib.sha256(frames_payload).hexdigest(),
            },
            "scene_manifest": {
                "path": str(manifest_path),
                "sha256": hashlib.sha256(manifest_payload).hexdigest(),
            },
            "renderer_source": {
                "path": str(renderer_path),
                "sha256": "b" * 64,
            },
        },
    }
    summary_payload = builder.canonical_json_bytes(summary)
    identity = {
        "dataset_role": "train",
        "scene_id": scene_id,
        "episode_id": "3",
        "env_index": 0,
        "episode_step": 4,
        "frame_index": 7,
        "timestamp_ns": 11,
        "image_sha256": image_sha,
    }
    endpoint = _hashed(
        {
            "schema": builder.plan_v5.ENDPOINT_SCHEMA,
            "identity": identity,
            "identity_sha256": builder.canonical_json_sha256(identity),
            "image_path_metadata_only": str(image_path),
            "frames_jsonl_sha256": hashlib.sha256(frames_payload).hexdigest(),
            "scene_manifest_sha256": manifest_sha256(manifest),
            "base_quat_world_xyzw": [0.0, 0.0, 0.0, 1.0],
            "stored_base_yaw_rad": 0.0,
        }
    )
    source_record = {
        "scene_id": scene_id,
        "role": "train",
        "family": family,
        "source_split": "train",
        "frames": {
            "scene_id": scene_id,
            "path": str(frames_path),
            "sha256": hashlib.sha256(frames_payload).hexdigest(),
        },
        "scene_manifest": {
            "scene_id": scene_id,
            "path": str(manifest_path),
            "file_sha256": hashlib.sha256(manifest_payload).hexdigest(),
            "content_sha256": manifest_sha256(manifest),
        },
        "render_plan": {
            "scene_id": scene_id,
            "path": str(plan_path),
            "sha256": hashlib.sha256(plan_payload).hexdigest(),
        },
        "render_summary": {
            "scene_id": scene_id,
            "path": str(summary_path),
            "sha256": hashlib.sha256(summary_payload).hexdigest(),
        },
    }
    payload_by_path = {
        frames_path: frames_payload,
        manifest_path: manifest_payload,
        plan_path: plan_payload,
        summary_path: summary_payload,
    }
    opened: list[Path] = []

    monkeypatch.setattr(builder, "_require_exact_authority", lambda _value: {})

    def synthetic_read(*, path: Path, expected_sha256: str, authorization_sha256: str):
        assert authorization_sha256 == "c" * 64
        payload = payload_by_path[path]
        assert hashlib.sha256(payload).hexdigest() == expected_sha256
        opened.append(path)
        return payload

    monkeypatch.setattr(builder, "_read_exact_source", synthetic_read)
    context = {
        endpoint["identity_sha256"]: {
            "scene_id": scene_id,
            "family": family,
            "episode_id": "3",
            "reset_count": 0,
            "episode_step": 4,
            "frame_index": 7,
            "env_index": 0,
            "timestamp_ns": 11,
            "image_sha256": image_sha,
            "image_path_metadata_only": str(image_path),
        }
    }
    result = builder._load_exact_scene_job(
        source_record,
        (endpoint,),
        context,
        "c" * 64,
    )

    assert opened == [frames_path, manifest_path, plan_path, summary_path]
    assert result["source_frames_jsonl_records_scanned"] == 1
    assert result["source_frames_selected_records"] == 1
    assert result["job"].scene_id == scene_id
    assert len(result["job"].endpoints) == 1
    evidence = builder.v4_builder.build_frame_evidence_v4(
        result["job"].endpoints[0].frame
    )
    assert evidence.pixel_hit_mask.sum() == 0
    assert not image_path.exists()
