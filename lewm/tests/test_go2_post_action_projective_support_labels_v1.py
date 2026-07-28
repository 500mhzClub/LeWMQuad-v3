from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from lewm.benchmarks import go2_post_action_projective_support_labels_v1 as labels
from lewm.benchmarks import (
    go2_post_action_projective_support_corridor_contract_v1 as contract,
)
from lewm.planning.oriented_footprint import Pose2D
from lewm_worlds.manifest import (
    BoxObject,
    CameraValidityConstraints,
    SceneManifest,
    SpawnSpec,
    manifest_sha256,
)


ROOT = Path(__file__).resolve().parents[2]
POLICY = (
    ROOT
    / "config/go2_geometry_v2_artifacts/"
    "go2_directional_footprint_policy_v1_"
    "c57650326e8b7d302498bbfe93b9e3d15c36d56d55ae9e1f339507ece0a9f1fc.json"
)


@pytest.fixture(scope="module")
def geometry() -> labels.GeometryInputsV1:
    return labels.load_geometry_inputs_v1(
        repository_root=ROOT,
        geometry_path=ROOT / "config/go2_generalization_geometry_v2.json",
        directional_policy_path=POLICY,
        primitive_registry_path=ROOT / "config/go2_primitive_registry.yaml",
    )


def _scene(*, obstacle: BoxObject | None = None) -> SceneManifest:
    return SceneManifest(
        scene_id="synthetic_scene",
        family="open_obstacle_field",
        difficulty_tier="synthetic",
        topology_seed=1,
        visual_seed=2,
        physics_seed=3,
        world_bounds_xy_m=((-10.0, -10.0), (10.0, 10.0)),
        spawn=SpawnSpec(xyz_m=(0.0, 0.0, 0.33), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
        graph_nodes=(),
        graph_edges=(),
        obstacles=(() if obstacle is None else (obstacle,)),
        landmarks=(),
        camera_constraints=CameraValidityConstraints(0.08, 0.05, 200.0, 0.1),
    )


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value: object) -> bytes:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path.write_bytes(raw)
    return raw


def _authority_artifact(path: str, marker: str) -> dict[str, object]:
    return {
        "path": path,
        "byte_count": 1,
        "file_sha256": marker * 64,
        "content_sha256": marker * 64,
    }


def _source_artifacts() -> tuple[dict[str, object], dict[str, object]]:
    return (
        _authority_artifact(labels.SOURCE_MANIFEST_RELATIVE_PATH, "a"),
        _authority_artifact(labels.SOURCE_REVIEW_RELATIVE_PATH, "b"),
    )


def _execution_binding() -> dict[str, object]:
    hashes = {
        "raw_manifest": (
            labels.RAW_MANIFEST_FILE_SHA256,
            labels.RAW_MANIFEST_CONTENT_SHA256,
        ),
        "raw_pairs": (labels.RAW_PAIRS_FILE_SHA256, None),
        "raw_endpoints": (labels.RAW_ENDPOINTS_FILE_SHA256, None),
        "raw_audit": (
            labels.RAW_AUDIT_FILE_SHA256,
            labels.RAW_AUDIT_CONTENT_SHA256,
        ),
        "schedule": (labels.SCHEDULE_FILE_SHA256, labels.SCHEDULE_CONTENT_SHA256),
        "geometry_contract": (
            labels.GEOMETRY_CONTRACT_FILE_SHA256,
            labels.GEOMETRY_CONTRACT_CONTENT_SHA256,
        ),
        "directional_policy": (
            labels.DIRECTIONAL_POLICY_FILE_SHA256,
            labels.DIRECTIONAL_POLICY_CONTENT_SHA256,
        ),
        "primitive_registry": (labels.PRIMITIVE_REGISTRY_FILE_SHA256, None),
    }
    inputs: dict[str, dict[str, object]] = {}
    for name, (file_sha256, content_sha256) in hashes.items():
        record: dict[str, object] = {
            "path": labels.INPUT_RELATIVE_PATHS[name],
            "byte_count": 1,
            "file_sha256": file_sha256,
        }
        if content_sha256 is not None:
            record["content_sha256"] = content_sha256
        inputs[name] = record
    source_manifest, source_review = _source_artifacts()
    return labels.with_content_sha256(
        {
            "schema": labels.EXECUTION_BINDING_SCHEMA,
            "status": "AUTHORIZED_ONE_EXACT_DEVELOPMENT_LABEL_PREFLIGHT",
            "preregistration_commit": labels.PREREGISTRATION_COMMIT,
            "integrity_adapter_amendment": (
                contract.integrity_adapter_amendment_binding()
            ),
            "label_v1_terminal_predecessor_bindings": {
                name: dict(binding)
                for name, binding in contract.LABEL_V1_TERMINAL_PREDECESSOR_BINDINGS.items()
            },
            "schedule_schema_adapter_amendment": (
                contract.schedule_schema_adapter_amendment_binding()
            ),
            "label_v2_terminal_predecessor_bindings": {
                name: dict(binding)
                for name, binding in contract.LABEL_V2_TERMINAL_PREDECESSOR_BINDINGS.items()
            },
            "authority": {
                "development_label_preflight_authorized": True,
                "training_authorized": False,
            },
            "source_manifest": source_manifest,
            "independent_source_review": source_review,
            "inputs": inputs,
            "output_directory": labels.LABEL_OUTPUT_RELATIVE_PATH,
            "source_records": [],
        }
    )


def _rehash(value: dict[str, object]) -> dict[str, object]:
    core = copy.deepcopy(value)
    core.pop("content_sha256")
    return labels.with_content_sha256(core)


def _builder_module():
    path = ROOT / "scripts/build_go2_post_action_projective_support_labels_v1.py"
    spec = importlib.util.spec_from_file_location("_label_builder_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_raw_endpoint_index_rejects_the_distinct_metadata_plan_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert labels.RAW_ENDPOINT_INDEX_ORDER_SHA256 == (
        "ab21c1a89b37ef60a056de390d59d3983705ab2e40de061d0cb163d1837e850f"
    )
    assert labels.RAW_METADATA_PLAN_ENDPOINT_ORDER_SHA256 == (
        "8130e961b7b5c04944b178fa4f73c1fa157776f7702ab5cdc213cf16c922f698"
    )
    labels._validate_raw_manifest_endpoint_order_v1(
        labels.RAW_ENDPOINT_INDEX_ORDER_SHA256
    )
    with pytest.raises(labels.LabelContractError, match="raw manifest identity"):
        labels._validate_raw_manifest_endpoint_order_v1(
            labels.RAW_METADATA_PLAN_ENDPOINT_ORDER_SHA256
        )

    endpoints = ({"content_sha256": "1" * 64}, {"content_sha256": "2" * 64})
    monkeypatch.setattr(
        labels,
        "RAW_ENDPOINT_INDEX_ORDER_SHA256",
        labels.canonical_json_sha256([row["content_sha256"] for row in endpoints]),
    )
    labels._validate_raw_endpoint_content_order_v1(endpoints)
    with pytest.raises(labels.LabelContractError, match="raw endpoint ordering"):
        labels._validate_raw_endpoint_content_order_v1(tuple(reversed(endpoints)))


def test_matched_training_v4_schedule_is_exact_and_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert labels.MATCHED_TRAINING_V4_SCHEDULE_SCHEMA == (
        "lewm_go2_shared_jepa_v5_matched_training_v4_schedule_v1"
    )
    assert labels.MATCHED_TRAINING_V4_SCHEDULE_IDENTITY_SHA256 == {
        "ordered_pair_ids_sha256": (
            "74b90f10347a89d2151c4f65f76d6fc3c6a94fb3e8caa350d2a92e934e80840a"
        ),
        "indices_sha256": (
            "a6f4fda5eb570336fb360631af3629832cccbe4cba21bdbb325dcb8a21963663"
        ),
        "presentation_pair_ids_sha256": (
            "1534dcdd85feb8421639a0dc433473913f6674556e22e0fa9f515be455b7b79a"
        ),
        "per_update_pair_ids_sha256": (
            "fe4aab82bd05b5e3438e8623319211ae75220f8bf3143223f6b6e375d91d46f0"
        ),
    }
    assert labels.LABEL_OUTPUT_RELATIVE_PATH.endswith("labels_v3")
    assert labels.SOURCE_MANIFEST_RELATIVE_PATH.endswith(
        "source_manifest_v3_2026-07-28.json"
    )
    assert labels.SOURCE_REVIEW_RELATIVE_PATH.endswith(
        "source_review_v3_2026-07-28.json"
    )

    pair_ids = [
        _sha(f"synthetic-pair-{index}".encode())
        for index in range(
            labels.MATCHED_TRAINING_V4_SCHEDULE_DIMENSIONS["train_pair_count"]
        )
    ]
    indices = [
        index % len(pair_ids)
        for index in range(
            labels.MATCHED_TRAINING_V4_SCHEDULE_DIMENSIONS["presentation_count"]
        )
    ]
    identities = dict(labels.MATCHED_TRAINING_V4_SCHEDULE_IDENTITY_SHA256)
    identities["ordered_pair_ids_sha256"] = labels.canonical_json_sha256(pair_ids)
    identities["indices_sha256"] = labels.canonical_json_sha256(indices)
    monkeypatch.setattr(
        labels,
        "MATCHED_TRAINING_V4_SCHEDULE_IDENTITY_SHA256",
        identities,
    )
    monkeypatch.setattr(
        labels,
        "SCHEDULE_PREFIX_SHA256",
        labels.canonical_json_sha256(indices[:16_000]),
    )
    schedule = labels.with_content_sha256(
        {
            "schema": labels.MATCHED_TRAINING_V4_SCHEDULE_SCHEMA,
            **labels.MATCHED_TRAINING_V4_SCHEDULE_DIMENSIONS,
            **identities,
            "presentation_indices": indices,
        }
    )
    monkeypatch.setattr(
        labels,
        "SCHEDULE_CONTENT_SHA256",
        schedule["content_sha256"],
    )
    schedule_raw = labels.canonical_json_bytes(schedule) + b"\n"
    schedule_path = tmp_path / "schedule.json"
    schedule_path.write_bytes(schedule_raw)
    monkeypatch.setattr(labels, "SCHEDULE_FILE_SHA256", _sha(schedule_raw))
    raw_indexes = labels.RawIndexesV1(
        manifest={},
        pairs=tuple(
            {"dataset_role": "train", "content_sha256": pair_id}
            for pair_id in pair_ids
        ),
        endpoints=(),
        endpoint_by_sha256={},
        shard_by_scene={},
    )
    assert labels.load_schedule_indices_v1(
        schedule_path,
        raw_indexes=raw_indexes,
    ) == tuple(indices[:16_000])

    def rehashed(changes: dict[str, object]) -> dict[str, object]:
        core = copy.deepcopy(schedule)
        core.pop("content_sha256")
        core.update(changes)
        return labels.with_content_sha256(core)

    extra_field = rehashed({"ordered_train_pair_ids": pair_ids})
    with pytest.raises(labels.LabelContractError, match="fields changed"):
        labels._validate_matched_training_v4_schedule_v1(
            extra_field,
            raw_indexes=raw_indexes,
        )

    wrong_schema = rehashed(
        {"schema": "lewm_go2_shared_jepa_v5_full_training_v4_schedule_v1"}
    )
    monkeypatch.setattr(
        labels,
        "SCHEDULE_CONTENT_SHA256",
        wrong_schema["content_sha256"],
    )
    with pytest.raises(labels.LabelContractError, match="identity changed"):
        labels._validate_matched_training_v4_schedule_v1(
            wrong_schema,
            raw_indexes=raw_indexes,
        )

    wrong_dimensions = rehashed({"update_count": 7_999})
    monkeypatch.setattr(
        labels,
        "SCHEDULE_CONTENT_SHA256",
        wrong_dimensions["content_sha256"],
    )
    with pytest.raises(labels.LabelContractError, match="dimensions changed"):
        labels._validate_matched_training_v4_schedule_v1(
            wrong_dimensions,
            raw_indexes=raw_indexes,
        )

    wrong_hashes = rehashed({"presentation_pair_ids_sha256": "0" * 64})
    monkeypatch.setattr(
        labels,
        "SCHEDULE_CONTENT_SHA256",
        wrong_hashes["content_sha256"],
    )
    with pytest.raises(labels.LabelContractError, match="hashes changed"):
        labels._validate_matched_training_v4_schedule_v1(
            wrong_hashes,
            raw_indexes=raw_indexes,
        )

    changed_indices = list(indices)
    changed_indices[-1] = (changed_indices[-1] + 1) % len(pair_ids)
    wrong_indices = rehashed({"presentation_indices": changed_indices})
    monkeypatch.setattr(
        labels,
        "SCHEDULE_CONTENT_SHA256",
        wrong_indices["content_sha256"],
    )
    with pytest.raises(labels.LabelContractError, match="index hash changed"):
        labels._validate_matched_training_v4_schedule_v1(
            wrong_indices,
            raw_indexes=raw_indexes,
        )

    wrong_pairs = list(raw_indexes.pairs)
    wrong_pairs[0] = {"dataset_role": "train", "content_sha256": "f" * 64}
    mismatched_raw_indexes = labels.RawIndexesV1(
        manifest={},
        pairs=tuple(wrong_pairs),
        endpoints=(),
        endpoint_by_sha256={},
        shard_by_scene={},
    )
    monkeypatch.setattr(
        labels,
        "SCHEDULE_CONTENT_SHA256",
        schedule["content_sha256"],
    )
    with pytest.raises(labels.LabelContractError, match="train-pair identity"):
        labels._validate_matched_training_v4_schedule_v1(
            schedule,
            raw_indexes=mismatched_raw_indexes,
        )


def test_exact_integrator_and_remote_sampler() -> None:
    poses = labels.integrate_action_v1(((0.2, 0.0, 0.45),) * 5)
    assert len(poses) == 6
    expected = Pose2D(0.0, 0.0, 0.0)
    for command_index in range(5):
        expected = Pose2D(
            expected.x_m + 0.2 * math.cos(expected.yaw_rad) * 0.1,
            expected.y_m + 0.2 * math.sin(expected.yaw_rad) * 0.1,
            labels.wrap_angle_pi(expected.yaw_rad + 0.45 * 0.1),
        )
        assert poses[command_index + 1] == expected

    groups = labels.remote_corridor_pose_samples_v1()
    assert [len(group) for group in groups] == [1, *([9] * 10)]
    assert groups[0][0] == Pose2D(1.45, 0.0, 0.0)
    assert groups[1][0] == groups[0][0]
    assert groups[-1][-1] == Pose2D(3.45, 0.0, 0.0)
    values = np.asarray(
        [(pose.x_m, pose.y_m, pose.yaw_rad) for group in groups for pose in group],
        dtype="<f8",
    )
    offsets = np.asarray(labels.REMOTE_SAMPLE_OFFSETS, dtype="<i8")
    assert _sha(offsets.tobytes() + values.tobytes()) == labels.REMOTE_SAMPLE_SHA256


def test_frozen_polygon_masks_match_preregistration(geometry: labels.GeometryInputsV1) -> None:
    predicted = labels.predicted_next_corridor_masks_v1(geometry.footprint)
    persistence = labels.persistence_corridor_masks_v1(
        geometry.footprint, geometry.commands_by_action
    )
    support = labels.projective_support_mask_v1()
    assert predicted.dtype == np.uint8 and predicted.flags.c_contiguous
    assert predicted.shape == (11, 64, 64)
    assert predicted.sum(axis=(1, 2)).tolist() == list(labels.PREDICTED_NEXT_MASK_COUNTS)
    assert _sha(predicted.tobytes()) == labels.PREDICTED_NEXT_MASK_SHA256
    assert persistence.shape == (9, 11, 64, 64)
    assert int(persistence.sum()) == 6_040
    assert _sha(persistence.tobytes()) == labels.PERSISTENCE_STACK_SHA256
    assert int(support.sum()) == 1_964
    assert not np.any(predicted & (1 - support)[None])
    assert not np.any(persistence & (1 - support)[None, None])


def test_state_labels_report_polygon_collisions_and_flat_schema(
    geometry: labels.GeometryInputsV1,
) -> None:
    obstacle = BoxObject(
        object_id="remote_box",
        kind="obstacle",
        center_xyz_m=(1.55, 0.0, 0.5),
        size_xyz_m=(0.10, 0.10, 1.0),
        yaw_rad=0.0,
        material_id="wall",
    )
    pair = labels.with_content_sha256(
        {
            "dataset_role": "train",
            "global_row": 7,
            "scene_id": "synthetic_scene",
            "family": "open_obstacle_field",
            "primitive": "hold",
            "current_endpoint_sha256": "2" * 64,
            "frames_jsonl_sha256": "5" * 64,
            "scene_manifest_sha256": "6" * 64,
        }
    )
    endpoint = {
        "endpoint_identity_sha256": "2" * 64,
        "content_sha256": "3" * 64,
    }
    rows = labels.label_state_v1(
        pair=pair,
        endpoint=endpoint,
        source_pose_world=Pose2D(0.0, 0.0, 0.0),
        source_line_number=11,
        scene_manifest=_scene(obstacle=obstacle),
        footprint=geometry.footprint,
        commands_by_action=geometry.commands_by_action,
        source_bindings={"source_frames_jsonl": {"file_sha256": "4" * 64}},
        role_state_index=0,
    )
    checked = labels.validate_label_rows_v1(
        rows, role="train", enforce_frozen_count=False
    )
    assert [row["action"] for row in checked] == list(labels.ACTION_ORDER)
    hold = checked[labels.ACTION_ORDER.index("hold")]
    assert hold["station_safe"][0] is False
    assert "remote_box" in hold["station_colliding_object_ids"][0]
    assert hold["remote_safe_prefix_length"] == 0
    assert hold["station_sample_counts"] == [1, *([9] * 10)]
    for row in checked:
        core = dict(row)
        declared = core.pop("content_sha256")
        assert labels.canonical_json_sha256(core) == declared


def test_narrow_scene_join_reconstructs_current_endpoint_without_rgb(tmp_path: Path) -> None:
    scene = _scene()
    manifest_path = tmp_path / "manifest.json"
    manifest_raw = _write_json(manifest_path, scene.to_dict())
    manifest_content = manifest_sha256(scene)

    frames_path = tmp_path / "frames.jsonl"
    frame = {
        "frame_index": 12,
        "env_index": 2,
        "timestamp_ns": 1234,
        "episode": {"episode_id": "episode-a", "reset_count": 3, "episode_step": 9},
        "base_pose_world": {"position": {"x": 1.0, "y": -2.0, "z": 0.33}},
        "base_rpy_rad": {"roll": 0.0, "pitch": 0.0, "yaw": 0.25},
        "forbidden_but_unread_rgb_payload": "metadata_only",
    }
    frames_raw = _write_json(frames_path, frame)
    image_path = tmp_path / "rgb/frame_000012_env_02.png"
    identity = {
        "dataset_role": "train",
        "scene_id": scene.scene_id,
        "episode_id": "episode-a",
        "env_index": 2,
        "episode_step": 9,
        "frame_index": 12,
        "timestamp_ns": 1234,
        "image_sha256": "a" * 64,
    }
    identity_sha = labels.canonical_json_sha256(identity)
    endpoint = {
        "dataset_role": "train",
        "scene_id": scene.scene_id,
        "family": scene.family,
        "endpoint_identity_sha256": identity_sha,
        "image_path_metadata_only": str(image_path),
        "image_sha256_commitment_only": "a" * 64,
        "content_sha256": "b" * 64,
    }
    pair = {
        "dataset_role": "train",
        "scene_id": scene.scene_id,
        "family": scene.family,
        "episode_id": "episode-a",
        "env_index": 2,
        "reset_count": 3,
        "frames_jsonl_sha256": _sha(frames_raw),
        "scene_manifest_sha256": manifest_content,
        "current_endpoint_sha256": identity_sha,
        "content_sha256": "c" * 64,
    }
    plan_path = tmp_path / "plan.json"
    plan_raw = b"plan metadata is not opened"
    plan_path.write_bytes(plan_raw)
    summary_path = tmp_path / "summary.json"
    summary = {
        "schema": "lewm_rendered_vision_v04",
        "render_status": "complete",
        "scene_id": scene.scene_id,
        "family": scene.family,
        "g2_model_outputs_opened": False,
        "source": {
            "frames_jsonl": {"path": str(frames_path), "sha256": _sha(frames_raw)},
            "scene_manifest": {"path": str(manifest_path), "sha256": _sha(manifest_raw)},
            "plan": {"path": str(plan_path), "sha256": _sha(plan_raw)},
        },
        "rendered_frames": [
            {
                "frame_index": 12,
                "env_index": 2,
                "timestamp_ns": 1234,
                "image_sha256": "a" * 64,
            }
        ],
    }
    summary_raw = _write_json(summary_path, summary)
    source_records = {
        "source_frames_jsonl": {
            "path": str(frames_path),
            "byte_count": len(frames_raw),
            "file_sha256": _sha(frames_raw),
            "purpose": "source_frames_jsonl",
            "scene_id": scene.scene_id,
            "dataset_role": "train",
            "family": scene.family,
        },
        "render_summary": {
            "path": str(summary_path),
            "byte_count": len(summary_raw),
            "file_sha256": _sha(summary_raw),
            "purpose": "render_summary",
            "scene_id": scene.scene_id,
            "dataset_role": "train",
            "family": scene.family,
        },
        "source_scene_manifest": {
            "path": str(manifest_path),
            "byte_count": len(manifest_raw),
            "file_sha256": _sha(manifest_raw),
            "purpose": "source_scene_manifest",
            "scene_id": scene.scene_id,
            "dataset_role": "train",
            "family": scene.family,
        },
    }
    raw = labels.RawIndexesV1(
        manifest={
            "input_provenance": {
                "source_payload_inventory": [
                    {
                        "path": str(plan_path),
                        "byte_count": len(plan_raw),
                        "file_sha256": _sha(plan_raw),
                        "purpose": "render_plan",
                        "scene_id": scene.scene_id,
                    }
                ]
            }
        },
        pairs=(pair,),
        endpoints=(endpoint,),
        endpoint_by_sha256={identity_sha: endpoint},
        shard_by_scene={
            scene.scene_id: {
                "dataset_role": "train",
                "family": scene.family,
                "scene_id": scene.scene_id,
            }
        },
    )
    parsed, joined = labels.load_joined_scene_v1(
        raw_indexes=raw,
        scene_id=scene.scene_id,
        source_records=source_records,
        repository_root=tmp_path,
    )
    assert parsed == scene
    assert len(joined) == 1
    assert joined[0].source_pose_world == Pose2D(1.0, -2.0, 0.25)
    assert joined[0].source_line_number == 1
    assert not image_path.exists()


def test_schedule_gate_is_a_separate_pure_check(geometry: labels.GeometryInputsV1) -> None:
    pair = labels.with_content_sha256(
        {
            "dataset_role": "train",
            "global_row": 1,
            "scene_id": "synthetic_scene",
            "family": "open_obstacle_field",
            "primitive": "hold",
            "current_endpoint_sha256": "d" * 64,
            "frames_jsonl_sha256": "f" * 64,
            "scene_manifest_sha256": "1" * 64,
        }
    )
    rows = list(
        labels.label_state_v1(
            pair=pair,
            endpoint={"endpoint_identity_sha256": "d" * 64, "content_sha256": "e" * 64},
            source_pose_world=Pose2D(0.0, 0.0, 0.0),
            source_line_number=1,
            scene_manifest=_scene(),
            footprint=geometry.footprint,
            commands_by_action=geometry.commands_by_action,
            source_bindings={},
            role_state_index=0,
        )
    )
    # An empty scene has tied prefixes, so make a canonical synthetic informative
    # state to exercise only the schedule accounting surface.
    for index, row in enumerate(rows):
        core = dict(row)
        core.pop("content_sha256")
        core["informative_state"] = True
        core["action_participates_in_ranking_pair"] = core["action"] != "hold"
        rows[index] = labels.with_content_sha256(core)
    report = labels.scheduled_preflight_v1(
        rows, [0] * 16_000, enforce_frozen_count=False
    )
    assert report["informative_presentation_count"] == 16_000
    assert set(report["ranking_participation_presentations_by_action"].values()) == {16_000}
    assert report["presentation_indices_sha256"] == labels.canonical_json_sha256(
        [0] * 16_000
    )


def test_binding_envelope_requires_full_commit_and_every_exact_path() -> None:
    binding = _execution_binding()
    checked = labels.validate_execution_binding_envelope_v1(binding)
    assert checked["source_manifest"] == binding["source_manifest"]

    abbreviated = copy.deepcopy(binding)
    abbreviated["preregistration_commit"] = labels.PREREGISTRATION_COMMIT[:7]
    with pytest.raises(labels.LabelContractError, match="identity changed"):
        labels.validate_execution_binding_envelope_v1(_rehash(abbreviated))

    wrong_input = copy.deepcopy(binding)
    wrong_input["inputs"]["raw_pairs"]["path"] += ".copy"
    with pytest.raises(labels.LabelContractError, match="raw_pairs"):
        labels.validate_execution_binding_envelope_v1(_rehash(wrong_input))

    wrong_output = copy.deepcopy(binding)
    wrong_output["output_directory"] += "_replacement"
    with pytest.raises(labels.LabelContractError, match="output_directory"):
        labels.validate_execution_binding_envelope_v1(_rehash(wrong_output))

    wrong_review = copy.deepcopy(binding)
    wrong_review["independent_source_review"]["path"] += ".copy"
    with pytest.raises(labels.LabelContractError, match="source_review"):
        labels.validate_execution_binding_envelope_v1(_rehash(wrong_review))

    wrong_amendment = copy.deepcopy(binding)
    wrong_amendment["integrity_adapter_amendment"]["file_sha256"] = "0" * 64
    with pytest.raises(labels.LabelContractError, match="integrity_adapter_amendment"):
        labels.validate_execution_binding_envelope_v1(_rehash(wrong_amendment))

    wrong_predecessor = copy.deepcopy(binding)
    wrong_predecessor["label_v1_terminal_predecessor_bindings"]["failure"][
        "file_sha256"
    ] = "0" * 64
    with pytest.raises(
        labels.LabelContractError, match="label_v1_terminal_predecessor_bindings"
    ):
        labels.validate_execution_binding_envelope_v1(_rehash(wrong_predecessor))

    wrong_schedule_amendment = copy.deepcopy(binding)
    wrong_schedule_amendment["schedule_schema_adapter_amendment"][
        "file_sha256"
    ] = "0" * 64
    with pytest.raises(
        labels.LabelContractError, match="schedule_schema_adapter_amendment"
    ):
        labels.validate_execution_binding_envelope_v1(
            _rehash(wrong_schedule_amendment)
        )

    wrong_v2_predecessor = copy.deepcopy(binding)
    wrong_v2_predecessor["label_v2_terminal_predecessor_bindings"]["failure"][
        "file_sha256"
    ] = "0" * 64
    with pytest.raises(
        labels.LabelContractError, match="label_v2_terminal_predecessor_bindings"
    ):
        labels.validate_execution_binding_envelope_v1(
            _rehash(wrong_v2_predecessor)
        )


def test_reservation_is_atomic_exact_and_failure_is_canonical(tmp_path: Path) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    source_manifest, source_review = _source_artifacts()
    reservation = labels.reserve_label_root_v1(
        repository_root,
        source_manifest=source_manifest,
        independent_source_review=source_review,
    )
    output = repository_root / labels.LABEL_OUTPUT_RELATIVE_PATH
    reservation_path = output / "reservation.json"
    assert reservation_path == repository_root / labels.LABEL_RESERVATION_RELATIVE_PATH
    assert reservation_path.read_bytes() == labels.canonical_json_bytes(reservation) + b"\n"
    assert not (repository_root / labels.SOURCE_MANIFEST_RELATIVE_PATH).exists()
    with pytest.raises(FileExistsError):
        labels.reserve_label_root_v1(
            repository_root,
            source_manifest=source_manifest,
            independent_source_review=source_review,
        )

    failure = labels.write_label_failure_v1(
        repository_root,
        phase="prepare_source_authority",
        error=RuntimeError("synthetic failure"),
        source_manifest=source_manifest,
        independent_source_review=source_review,
        access_ledger=labels.new_access_ledger_v1(),
    )
    failure_path = output / "failure.json"
    assert failure_path.read_bytes() == labels.canonical_json_bytes(failure) + b"\n"
    assert failure["status"] == "FAILED_TERMINAL_NO_RETRY"
    assert failure["terminal"] == {
        "retry_authorized": False,
        "resume_authorized": False,
        "second_invocation_authorized": False,
        "same_root_replacement_authorized": False,
    }
    with pytest.raises(FileExistsError):
        labels.write_label_failure_v1(
            repository_root,
            phase="retry",
            error=RuntimeError("forbidden"),
            source_manifest=source_manifest,
            independent_source_review=source_review,
        )


def test_builder_failure_is_terminal_before_any_raw_open(tmp_path: Path) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    source_manifest, source_review = _source_artifacts()
    labels.reserve_label_root_v1(
        repository_root,
        source_manifest=source_manifest,
        independent_source_review=source_review,
    )
    binding = _execution_binding()
    binding_path = repository_root / labels.LABEL_EXECUTION_BINDING_RELATIVE_PATH
    binding_path.parent.mkdir(parents=True)
    binding_path.write_bytes(labels.canonical_json_bytes(binding) + b"\n")

    builder = _builder_module()
    with pytest.raises(labels.LabelContractError, match="authority artifact"):
        builder.build_from_binding(binding_path, repository_root=repository_root)
    output = repository_root / labels.LABEL_OUTPUT_RELATIVE_PATH
    claim_path = repository_root / labels.LABEL_BUILDER_CLAIM_RELATIVE_PATH
    claim = json.loads(claim_path.read_bytes())
    reservation = json.loads(
        (repository_root / labels.LABEL_RESERVATION_RELATIVE_PATH).read_bytes()
    )
    assert claim["reservation_content_sha256"] == reservation["content_sha256"]
    assert claim["execution_binding_content_sha256"] == binding["content_sha256"]
    assert claim["retry_authorized"] is False
    assert claim["resume_authorized"] is False
    assert claim["second_invocation_authorized"] is False
    failure_raw = (output / "failure.json").read_bytes()
    failure = json.loads(failure_raw)
    assert failure_raw == labels.canonical_json_bytes(failure) + b"\n"
    assert failure["phase"] == "validate_source_manifest_and_review"
    assert failure["access_ledger"]["execution_binding_opens"] == 1
    assert failure["access_ledger"]["raw_manifest_opens"] == 0
    assert failure["access_ledger"]["geometry_contract_opens"] == 0
    assert failure["access_ledger"]["schedule_opens"] == 0
    assert failure["content_sha256"] == labels.canonical_json_sha256(
        {key: value for key, value in failure.items() if key != "content_sha256"}
    )
    with pytest.raises(PermissionError, match="already consumed"):
        builder.build_from_binding(binding_path, repository_root=repository_root)
    assert (output / "failure.json").read_bytes() == failure_raw


def test_builder_rejects_noncanonical_binding_path_before_access(tmp_path: Path) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    source_manifest, source_review = _source_artifacts()
    labels.reserve_label_root_v1(
        repository_root,
        source_manifest=source_manifest,
        independent_source_review=source_review,
    )
    with pytest.raises(PermissionError, match="exact label execution-binding path"):
        _builder_module().build_from_binding(
            repository_root / "docs/not-the-binding.json",
            repository_root=repository_root,
        )
    output = repository_root / labels.LABEL_OUTPUT_RELATIVE_PATH
    failure = json.loads((output / "failure.json").read_bytes())
    assert failure["phase"] == "load_execution_binding"
    assert set(failure["access_ledger"].values()) == {0}


def test_schedule_prefix_and_registered_family_names_fail_closed() -> None:
    arbitrary_prefix = [0] * 16_000
    with pytest.raises(labels.LabelContractError, match="schedule prefix"):
        labels._schedule_prefix_identity_v1(arbitrary_prefix, require_frozen=True)
    prefix, digest = labels._schedule_prefix_identity_v1(
        arbitrary_prefix, require_frozen=False
    )
    assert len(prefix) == 16_000
    assert digest == labels.canonical_json_sha256(arbitrary_prefix)

    labels._validate_registered_selection_family_counts_v1(
        {family: 8 for family in labels.REGISTERED_SELECTION_FAMILIES}
    )
    counterfeit = {f"counterfeit_{index}": 8 for index in range(8)}
    with pytest.raises(labels.LabelContractError, match="exact registered families"):
        labels._validate_registered_selection_family_counts_v1(counterfeit)


def test_reserved_root_publishes_manifest_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "reserved"
    staging = output / "staging"
    staging.mkdir(parents=True)
    for name in ("a.bin", "b.bin", "manifest.json"):
        (staging / name).write_bytes(name.encode())
    rename_order: list[str] = []
    real_rename = os.rename

    def tracked_rename(source: Path, destination: Path) -> None:
        rename_order.append(Path(destination).name)
        real_rename(source, destination)

    monkeypatch.setattr(labels.os, "rename", tracked_rename)
    labels._publish_staging_manifest_last_v1(
        staging, output, ("a.bin", "b.bin")
    )
    assert rename_order == ["a.bin", "b.bin", "manifest.json"]
    assert not staging.exists()
