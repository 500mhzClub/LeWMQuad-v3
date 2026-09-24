from __future__ import annotations

import copy
from collections import Counter
import hashlib
import io
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import zipfile

import numpy as np
import pytest

from lewm.benchmarks.go2_n32_camera_frustum_observability import (
    analyze_frame_labels,
    audit_camera_centered_mapping,
    old_body_column_span_audit,
)
from scripts import finalize_go2_n32_camera_frustum_observability as finalizer


def _digest(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _npy_bytes(
    data: bytes,
    *,
    rows: int,
    descr: str,
    version: tuple[int, int] = (1, 0),
    duplicate_descr: bool = False,
) -> bytes:
    shape = f"({rows}, 64, 64)"
    if duplicate_descr:
        dictionary = (
            "{'descr': '" + descr + "', 'descr': '" + descr
            + "', 'fortran_order': False, 'shape': " + shape + ", }"
        )
    else:
        dictionary = (
            "{'descr': '" + descr
            + "', 'fortran_order': False, 'shape': " + shape + ", }"
        )
    length_size = 2 if version == (1, 0) else 4
    encoding = "utf-8" if version == (3, 0) else "latin1"
    prefix_size = 6 + 2 + length_size
    encoded = dictionary.encode(encoding)
    padding = (-((prefix_size + len(encoded) + 1) % 64)) % 64
    header = encoded + b" " * padding + b"\n"
    length = len(header).to_bytes(length_size, "little")
    return b"\x93NUMPY" + bytes(version) + length + header + data


def _npz_bytes(
    *,
    rows: int = 2,
    label_override: dict[str, bytes] | None = None,
    mask_override: dict[str, bytes] | None = None,
    version: tuple[int, int] = (1, 0),
    extra_name: str | None = None,
) -> bytes:
    labels = bytes([0]) * (rows * 4096)
    masks = bytes([1]) * (rows * 4096)
    members: dict[str, bytes] = {}
    for side in finalizer.ENDPOINT_SIDES:
        label_data = (label_override or {}).get(side, labels)
        mask_data = (mask_override or {}).get(side, masks)
        members[f"{side}_labels.npy"] = _npy_bytes(
            label_data, rows=rows, descr="|u1", version=version
        )
        members[f"{side}_supervision_mask.npy"] = _npy_bytes(
            mask_data, rows=rows, descr="|b1", version=version
        )
    for name in finalizer.REGISTERED_AUX_ARRAYS:
        members[f"{name}.npy"] = b"auxiliary bytes are never decompressed"
    if extra_name is not None:
        members[extra_name] = b"unexpected"
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(members):
            archive.writestr(name, members[name])
    return stream.getvalue()


def _frame_key(index: int, family: str, side: str, shard_hash: str) -> dict:
    return {
        "family": family,
        "scene_id": f"scene_{finalizer.FAMILIES.index(family):02d}",
        "global_row": index // 2,
        "side": side,
        "image_sha256": _digest(f"image:{index}"),
        "label_shard_sha256": shard_hash,
        "label_row": index % 16,
    }


def _empty_label_support(key: dict) -> dict:
    del key
    return {
        "schema": finalizer.LABEL_SUPPORT_SCHEMA,
        "total_supervised_label_count": 4096,
        "supported_label_count": 1990,
        "unsupported_label_count": 2106,
        "class_counts": {"unknown": 4096, "free": 0, "occupied": 0},
        "by_class": {
            "unknown": {"total": 4096, "supported": 1990, "unsupported": 2106},
            "free": {"total": 0, "supported": 0, "unsupported": 0},
            "occupied": {"total": 0, "supported": 0, "unsupported": 0},
        },
        "unsupported_free_count": 0,
        "unsupported_occupied_count": 0,
        "unsupported_unknown_count": 2106,
        "unsupported_targets_are_all_unknown": True,
        "violations": [],
        "passes": True,
    }


def _empty_ray_summary() -> dict:
    return {
        "sequence_count": 256,
        "length_histogram": dict(finalizer.PER_FRAME_LENGTH_HISTOGRAM),
        "sequences_with_fewer_than_two_cells_count": 4,
        "transition_rate_eligible_sequence_count": 252,
        "class_transition_histogram": {"0": 256},
        "maximum_transitions_per_sequence": 0,
        "directed_unequal_transition_counts": {
            name: 0 for name in finalizer.TRANSITION_NAMES
        },
        "transition_bucket_counts": {"0": 256, "1": 0, "2": 0, "3_plus": 0},
        "transition_event_count": 0,
        "transition_events_per_eligible_sequence": 0.0,
        "contains_known_after_unknown_count": 0,
        "contains_free_after_occupied_count": 0,
        "scalar_first_hit_irregular_count": 0,
        "scalar_first_hit_regular_count": 256,
    }


def _empty_partitions() -> dict:
    return {
        name: {
            "count": 0,
            "identities": [],
            "identities_sha256": finalizer.canonical_json_sha256([]),
        }
        for name in finalizer.ATTRIBUTION_PARTITION_NAMES
    }


def _valid_camera_frame() -> dict:
    return {
        "base_pose_world": {"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
        "base_quat_world_xyzw": [0.0, 0.0, 0.0, 1.0],
        "base_rpy_rad": {"yaw": 0.0},
        "camera_mount_body": copy.deepcopy(finalizer.NOMINAL_CAMERA_MOUNT_BODY),
        "camera_pose_world": {
            "position": [0.326, 0.0, 0.043],
            "lookat": [1.326, 0.0, 0.043],
            "up": [0.0, 0.0, 1.0],
        },
    }


def _valid_camera_evidence() -> dict:
    return finalizer._camera_mount_composition_evidence(
        _valid_camera_frame(),
        plan_mount_value=copy.deepcopy(finalizer.NOMINAL_CAMERA_MOUNT_BODY),
        context="synthetic camera",
    )


def _report(key: dict) -> dict:
    summary = _empty_ray_summary()
    return {
        "record_key": key,
        "camera_mount_composition": _valid_camera_evidence(),
        "label_support": _empty_label_support(key),
        "ray_sequences": {
            "schema": finalizer.RAY_SEQUENCE_SCHEMA,
            "summary": summary,
            "sequence_summary_records_sha256": _digest(key),
            "transition_table_sha256": finalizer.canonical_json_sha256(summary),
        },
        "collision_veto_only_unknown_count": 0,
        "collision_veto_only_unknown_identities": [],
        "collision_veto_only_unknown_distance_bin_counts": {
            name: 0 for name in finalizer.DISTANCE_BIN_NAMES
        },
        "attributed_to_matched_box_count": 0,
        "depends_on_unmatched_collision_box_count": 0,
        "attribution_partitions": _empty_partitions(),
        "reconstruction_mismatch_cell_count": 0,
        "reconstruction_mismatch_identities": [],
        "rendered_collision_overlap_xor_cell_count": 0,
        "geometry_ambiguity_cell_count": 0,
    }


def _panel_records_and_manifests() -> tuple[list[dict], dict, list[dict]]:
    shard_entries = []
    shard_hashes = []
    for index in range(20):
        digest = _digest(f"shard:{index}")
        shard_hashes.append(digest)
        shard_entries.append(
            {
                "path": f"/synthetic/shard_{index:02d}.npz",
                "sha256": digest,
                "selected_tuples": [],
                "selected_row_count": 0,
                "family_side_counts": {
                    family: {side: 0 for side in finalizer.ENDPOINT_SIDES}
                    for family in finalizer.FAMILIES
                },
            }
        )
    records = []
    index = 0
    for family in finalizer.FAMILIES:
        for family_index in range(64):
            side = finalizer.ENDPOINT_SIDES[family_index % 2]
            shard_index = index // 16
            key = _frame_key(index, family, side, shard_hashes[shard_index])
            record = {
                **key,
                "label_shard_path": shard_entries[shard_index]["path"],
            }
            records.append(record)
            selected_tuple = [
                key["family"], key["scene_id"], key["global_row"], key["side"], key["label_row"]
            ]
            shard_entries[shard_index]["selected_tuples"].append(selected_tuple)
            shard_entries[shard_index]["selected_row_count"] += 1
            shard_entries[shard_index]["family_side_counts"][family][side] += 1
            index += 1
    manifest = {
        "entry_count": 20,
        "entries": shard_entries,
        "manifest_sha256": finalizer.canonical_json_sha256(shard_entries),
    }
    reports = [_report({field: record[field] for field in finalizer.FRAME_IDENTITY_FIELDS}) for record in records]
    return records, manifest, reports


def _small_panel(*, duplicate_order_key: bool = False, reuse_storage_row: bool = False) -> tuple[dict, str, str]:
    rows = []
    global_row = 0
    for family in finalizer.FAMILIES:
        for offset in range(2 if duplicate_order_key else 1):
            row = {
                "family": family,
                "scene_id": f"scene_{family}",
                "dataset_role": "train",
                "global_row": global_row,
                "label_shard_path": "/synthetic/shared.npz",
                "label_shard_sha256": _digest("shared"),
                "label_shard_row": 0 if reuse_storage_row else global_row,
                "env_index": 0,
                "episode_id": str(global_row),
                "reset_count": 0,
            }
            if duplicate_order_key and offset == 1:
                row["global_row"] = global_row - 1
            for side_index, side in enumerate(finalizer.ENDPOINT_SIDES):
                row[f"{side}_image_path"] = f"/synthetic/{family}/{offset}/{side}.png"
                row[f"{side}_image_sha256"] = _digest(f"{family}:{offset}:{side}")
                row[f"{side}_frame_index"] = global_row * 2 + side_index
                row[f"{side}_timestamp_ns"] = global_row * 10 + side_index
                row[f"{side}_episode_step"] = side_index
            rows.append(row)
            global_row += 1
    rows_sha = finalizer.canonical_json_sha256(rows)
    core = {
        "schema": "lewm_go2_physical_micro_overfit_panel_v1",
        "families": list(finalizer.FAMILIES),
        "rows_per_family_panel": 2 if duplicate_order_key else 1,
        "local_grid": {
            "shape": [64, 64],
            "cell_size_m": 0.1,
            "forward_edge_range_m": [-1.0, 5.4],
            "left_edge_range_m": [-3.2, 3.2],
        },
        "source_camera_projection": {"horizontal_fov_deg": 78.323, "near_m": 0.05},
        "inputs": {
            "geometry_contract": {
                "path": "/synthetic/geometry.json",
                "file_sha256": _digest("geometry"),
                "semantic_sha256": _digest("geometry-semantic"),
            },
            "render_audit_contract": {
                "path": "/synthetic/render-audit.json",
                "file_sha256": _digest("render-audit"),
                "content_sha256": _digest("render-audit-content"),
            },
        },
        "panels": {
            "fit": {
                "row_count": len(rows),
                "frame_count": len(rows) * 2,
                "rows_sha256": rows_sha,
                "rows": rows,
            }
        },
    }
    return {**core, "content_sha256": finalizer.canonical_json_sha256(core)}, finalizer.canonical_json_sha256(core), rows_sha


def test_stdlib_geometry_recomputation_exactly_matches_primary() -> None:
    assert finalizer._stdlib_mapping_audit() == audit_camera_centered_mapping()
    assert finalizer._stdlib_old_body_column_span_audit() == old_body_column_span_audit()
    assert finalizer._stdlib_mapping_audit()["mapping_sha256"] == finalizer.MAPPING_SHA256
    assert finalizer._stdlib_mapping_audit()["support_mask_sha256"] == finalizer.SUPPORT_MASK_SHA256


def _zero_origin_camera_frame() -> dict:
    frame = _valid_camera_frame()
    frame["base_pose_world"]["position"] = {
        "x": -0.326,
        "y": 0.0,
        "z": -0.043,
    }
    frame["camera_pose_world"] = {
        "position": [0.0, 0.0, 0.0],
        "lookat": [1.0, 0.0, 0.0],
        "up": [0.0, 0.0, 1.0],
    }
    return frame


def test_full_quaternion_camera_mount_composition_preserves_roll_pitch() -> None:
    roll = 0.2
    sine = math.sin(roll)
    cosine = math.cos(roll)
    frame = _valid_camera_frame()
    frame["base_quat_world_xyzw"] = [
        math.sin(roll * 0.5),
        0.0,
        0.0,
        math.cos(roll * 0.5),
    ]
    frame["camera_pose_world"] = {
        "position": [0.326, -sine * 0.043, cosine * 0.043],
        "lookat": [1.326, -sine * 0.043, cosine * 0.043],
        "up": [0.0, -sine, cosine],
    }
    evidence = finalizer._camera_mount_composition_evidence(
        frame,
        plan_mount_value=copy.deepcopy(finalizer.NOMINAL_CAMERA_MOUNT_BODY),
        context="rolled body",
    )
    assert evidence["passes"] is True
    assert evidence["position_max_abs_residual_m"] <= 1e-15
    assert evidence["lookat_max_abs_residual_m"] <= 1e-15
    assert evidence["up_max_abs_residual"] <= 1e-15

    pitch = 0.15
    pitch_sine = math.sin(pitch)
    pitch_cosine = math.cos(pitch)
    pitched = _valid_camera_frame()
    pitched["base_quat_world_xyzw"] = [
        0.0,
        math.sin(pitch * 0.5),
        0.0,
        math.cos(pitch * 0.5),
    ]
    pitched_position = [
        pitch_cosine * 0.326 + pitch_sine * 0.043,
        0.0,
        -pitch_sine * 0.326 + pitch_cosine * 0.043,
    ]
    pitched_forward = [pitch_cosine, 0.0, -pitch_sine]
    pitched["camera_pose_world"] = {
        "position": pitched_position,
        "lookat": [
            pitched_position[index] + pitched_forward[index] for index in range(3)
        ],
        "up": [pitch_sine, 0.0, pitch_cosine],
    }
    assert finalizer._camera_mount_composition_evidence(
        pitched,
        plan_mount_value=copy.deepcopy(finalizer.NOMINAL_CAMERA_MOUNT_BODY),
        context="pitched body",
    )["passes"]

    yaw_only = copy.deepcopy(frame)
    yaw_only["camera_pose_world"] = _valid_camera_frame()["camera_pose_world"]
    rejected = finalizer._camera_mount_composition_evidence(
        yaw_only,
        plan_mount_value=copy.deepcopy(finalizer.NOMINAL_CAMERA_MOUNT_BODY),
        context="yaw-only substitution",
    )
    assert rejected["passes"] is False
    assert rejected["up_angular_error_rad"] > 1e-5


def test_camera_evidence_is_exactly_independent_runner_compatible() -> None:
    from scripts import audit_go2_n32_camera_frustum_observability as runner

    complex_frame = _valid_camera_frame()
    complex_frame["base_pose_world"]["position"] = {
        "x": -0.7,
        "y": 1.2,
        "z": 0.31,
    }
    complex_frame["base_quat_world_xyzw"] = [0.1, -0.2, 0.3, 0.9273618495]
    complex_frame["base_rpy_rad"]["yaw"] = 0.4
    complex_frame["camera_pose_world"] = {
        "position": [-0.2, 1.0, 0.5],
        "lookat": [0.6, 1.3, 0.2],
        "up": [0.2, -0.1, 0.9],
    }
    ulp_sensitive_frame = _valid_camera_frame()
    ulp_sensitive_frame["base_quat_world_xyzw"] = [
        -0.02480493113398552,
        0.01778911054134369,
        -0.5497533679008484,
        0.8347691893577576,
    ]
    for frame in (_valid_camera_frame(), complex_frame, ulp_sensitive_frame):
        finalizer_evidence = finalizer._camera_mount_composition_evidence(
            frame,
            plan_mount_value=copy.deepcopy(finalizer.NOMINAL_CAMERA_MOUNT_BODY),
            context="runner compatibility",
        )
        position = frame["base_pose_world"]["position"]
        runner_evidence = runner._camera_mount_composition_evidence(
            base_position_world=[position[axis] for axis in ("x", "y", "z")],
            base_quat_world_xyzw=frame["base_quat_world_xyzw"],
            stored_base_yaw_rad=frame["base_rpy_rad"]["yaw"],
            plan_camera_mount_body=copy.deepcopy(finalizer.NOMINAL_CAMERA_MOUNT_BODY),
            frame_camera_mount_body=frame["camera_mount_body"],
            recorded_camera_pose_world=frame["camera_pose_world"],
        )
        assert finalizer_evidence == runner_evidence
        assert finalizer.canonical_json_sha256(finalizer_evidence) == (
            finalizer.canonical_json_sha256(runner_evidence)
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "translation",
        "rotation",
        "retraction",
        "frame_mount_jitter",
        "plan_mount_jitter",
        "look_distance",
        "quaternion_norm",
        "quaternion_yaw_disagreement",
    ),
)
def test_camera_mount_composition_rejects_registered_mutations(mutation: str) -> None:
    frame = _zero_origin_camera_frame()
    plan_mount = copy.deepcopy(finalizer.NOMINAL_CAMERA_MOUNT_BODY)
    outside = 2.0e-5
    if mutation == "translation":
        frame["camera_pose_world"]["position"][1] += outside
        frame["camera_pose_world"]["lookat"][1] += outside
    elif mutation == "rotation":
        frame["camera_pose_world"]["lookat"] = [
            math.cos(outside),
            math.sin(outside),
            0.0,
        ]
    elif mutation == "retraction":
        frame["camera_pose_world"]["position"][0] -= 0.01
        frame["camera_pose_world"]["lookat"][0] -= 0.01
    elif mutation == "frame_mount_jitter":
        frame["camera_mount_body"]["xyz_body_m"][0] += outside
    elif mutation == "plan_mount_jitter":
        plan_mount["xyz_body_m"][0] += outside
    elif mutation == "look_distance":
        frame["camera_pose_world"]["lookat"][0] = 1.0 + outside
    elif mutation == "quaternion_norm":
        frame["base_quat_world_xyzw"][3] = 1.0 + outside
    else:
        frame["base_rpy_rad"]["yaw"] = outside
    evidence = finalizer._camera_mount_composition_evidence(
        frame,
        plan_mount_value=plan_mount,
        context=f"mutation {mutation}",
    )
    assert evidence["passes"] is False


def test_camera_mount_tolerance_is_inclusive_for_every_residual_gate() -> None:
    tolerance = finalizer.CAMERA_MOUNT_COMPOSITION_TOLERANCE
    assert finalizer._within_camera_mount_tolerance(tolerance)
    assert finalizer._within_camera_mount_tolerance(math.nextafter(tolerance, 0.0))
    assert not finalizer._within_camera_mount_tolerance(
        math.nextafter(tolerance, math.inf)
    )

    component = _zero_origin_camera_frame()
    component["camera_pose_world"]["position"][1] = tolerance
    component["camera_pose_world"]["lookat"][1] = tolerance
    assert finalizer._camera_mount_composition_evidence(
        component,
        plan_mount_value=copy.deepcopy(finalizer.NOMINAL_CAMERA_MOUNT_BODY),
        context="component boundary",
    )["passes"]

    quaternion = _zero_origin_camera_frame()
    quaternion["base_quat_world_xyzw"][3] = math.nextafter(
        1.0 + tolerance, 1.0
    )
    assert finalizer._camera_mount_composition_evidence(
        quaternion,
        plan_mount_value=copy.deepcopy(finalizer.NOMINAL_CAMERA_MOUNT_BODY),
        context="quaternion boundary",
    )["passes"]

    yaw = _zero_origin_camera_frame()
    yaw["base_rpy_rad"]["yaw"] = tolerance
    assert finalizer._camera_mount_composition_evidence(
        yaw,
        plan_mount_value=copy.deepcopy(finalizer.NOMINAL_CAMERA_MOUNT_BODY),
        context="yaw boundary",
    )["passes"]

    look = _zero_origin_camera_frame()
    look["camera_pose_world"]["lookat"][0] = math.nextafter(
        1.0 + tolerance, 1.0
    )
    assert finalizer._camera_mount_composition_evidence(
        look,
        plan_mount_value=copy.deepcopy(finalizer.NOMINAL_CAMERA_MOUNT_BODY),
        context="look-distance boundary",
    )["passes"]


def test_compact_camera_evidence_must_equal_independent_recomputation() -> None:
    expected = _valid_camera_evidence()
    finalizer._validate_camera_mount_evidence(
        expected,
        context="camera evidence",
        expected=expected,
    )
    changed = copy.deepcopy(expected)
    changed["quaternion_norm_abs_residual"] = 1e-12
    with pytest.raises(finalizer.FinalizationError, match="independent"):
        finalizer._validate_camera_mount_evidence(
            changed,
            context="camera evidence",
            expected=expected,
        )


def _synthetic_source_provenance() -> tuple[list[dict], dict[Path, object], list[dict], dict[str, Path]]:
    scene_id = "synthetic_scene"
    family = finalizer.FAMILIES[0]
    paths = {
        "physical_geometry_contract": Path("/synthetic/geometry.json"),
        "render_audit_contract": Path("/synthetic/render_audit.json"),
        "fit_render_summary": Path("/synthetic/summary.json"),
        "fit_frame_selection": Path("/synthetic/selection.json"),
        "render_source_plan": Path("/synthetic/plan.json"),
        "source_frames_jsonl": Path("/synthetic/frames.jsonl"),
        "source_scene_manifest": Path("/synthetic/manifest.json"),
        "renderer_source": Path("/synthetic/renderer.py"),
    }
    entries = [
        {
            "path": str(path),
            "sha256": _digest(f"source:{role}"),
            "semantic_role": role,
            "scene_id": scene_id,
        }
        for role, path in paths.items()
    ]
    by_role = {entry["semantic_role"]: entry for entry in entries}
    panel_record = {
        "family": family,
        "scene_id": scene_id,
        "global_row": 3,
        "side": "current",
        "image_sha256": _digest("selected image"),
        "label_shard_sha256": _digest("selected shard"),
        "label_row": 2,
        "frame_index": 7,
        "env_index": 0,
        "timestamp_ns": 11,
        "episode_id": "episode-1",
        "reset_count": 0,
        "episode_step": 4,
    }
    frame_keys = [[7, 0], [99, 0]]
    selection_core = {
        "schema": "lewm_go2_selected_render_frames_v1",
        "scene_id": scene_id,
        "scene_id_sha256": hashlib.sha256(scene_id.encode("utf-8")).hexdigest(),
        "dataset_role": "train",
        "row_count": 1,
        "frame_count": 2,
        "frame_keys": frame_keys,
        "frame_key_set_sha256": finalizer.canonical_json_sha256(frame_keys),
        "source_rows": {
            "path": "/synthetic/source_rows.jsonl",
            "sha256": _digest("source rows"),
        },
        "g2_images_opened": False,
        "g2_label_shards_opened": False,
    }
    selection = {
        **selection_core,
        "content_sha256": finalizer.canonical_json_sha256(selection_core),
    }
    rendered = [
        {
            "frame_index": 7,
            "env_index": 0,
            "timestamp_ns": 11,
            "image_sha256": panel_record["image_sha256"],
        },
        {
            "frame_index": 99,
            "env_index": 0,
            "timestamp_ns": 99,
            "image_sha256": _digest("unrelated selected image"),
        },
    ]
    source = {
        name: {
            "path": by_role[role]["path"],
            "sha256": by_role[role]["sha256"],
        }
        for name, role in {
            "plan": "render_source_plan",
            "frames_jsonl": "source_frames_jsonl",
            "scene_manifest": "source_scene_manifest",
            "renderer_source": "renderer_source",
        }.items()
    }
    summary = {
        "schema": "lewm_rendered_vision_v04",
        "render_status": "complete",
        "scene_id": scene_id,
        "family": family,
        "g2_model_outputs_opened": False,
        "frame_count": 2,
        "frame_selection": {
            "path": by_role["fit_frame_selection"]["path"],
            "sha256": by_role["fit_frame_selection"]["sha256"],
            "frame_key_set_sha256": finalizer.canonical_json_sha256(frame_keys),
        },
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
        "rendered_frames": rendered,
        "rendered_image_set_sha256": finalizer.canonical_json_sha256(rendered),
        "object_parity": {
            "schema": "lewm_render_object_parity_v1",
            "rendered_groups": ["wall", "obstacle", "landmark", "distractor"],
            "collision_distractors_rendered": True,
            "full_box_roll_pitch_yaw_rendered": True,
            "rendered_object_count": 0,
            "rendered_object_ids": [],
            "rendered_object_ids_sha256": finalizer.canonical_json_sha256([]),
            "rendered_object_records_sha256": finalizer.canonical_json_sha256([]),
        },
        "source": source,
    }
    selected_frame = _valid_camera_frame()
    selected_frame.update(
        {
            "frame_index": 7,
            "env_index": 0,
            "timestamp_ns": 11,
            "episode": {
                "episode_id": "episode-1",
                "reset_count": 0,
                "episode_step": 4,
                "split": "train",
            },
        }
    )
    selected_nonfit_frame = {
        "frame_index": 99,
        "env_index": 0,
        "timestamp_ns": 99,
    }
    outside_selection_frame = {
        "frame_index": 123,
        "env_index": 0,
        "timestamp_ns": 123,
    }
    plan = {
        "schema": "lewm_render_replay_plan_v0",
        "scene_id": scene_id,
        "frames_jsonl": by_role["source_frames_jsonl"]["path"],
        "camera": {
            "native_resolution": [640, 480],
            "training_resolution": [224, 168],
            "fov_axis": "horizontal",
            "fov_deg": 78.323,
            "near_m": 0.05,
            "far_m": 200.0,
            "encoding": "rgb8",
            "mount_body": copy.deepcopy(finalizer.NOMINAL_CAMERA_MOUNT_BODY),
        },
    }
    scene_manifest = {
        "scene_id": scene_id,
        "family": family,
        "walls": [],
        "obstacles": [],
        "landmarks": [],
        "visual_randomization": {"distractor_objects": []},
    }
    parsed: dict[Path, object] = {
        paths["physical_geometry_contract"]: {},
        paths["render_audit_contract"]: {},
        paths["fit_render_summary"]: summary,
        paths["fit_frame_selection"]: selection,
        paths["render_source_plan"]: plan,
        paths["source_frames_jsonl"]: [
            selected_nonfit_frame,
            selected_frame,
            outside_selection_frame,
        ],
        paths["source_scene_manifest"]: scene_manifest,
    }
    return entries, parsed, [panel_record], paths


def _recommit_selection(selection: dict) -> None:
    core = dict(selection)
    core.pop("content_sha256", None)
    selection["content_sha256"] = finalizer.canonical_json_sha256(core)


def test_independent_source_provenance_rejects_every_schema_and_scope_gap() -> None:
    entries, parsed, panel, paths = _synthetic_source_provenance()
    evidence = finalizer._validate_source_scene_provenance(
        machine_entries=entries,
        parsed_by_path=parsed,
        panel_records=panel,
        selected_scene_families={"synthetic_scene": finalizer.FAMILIES[0]},
    )
    assert len(evidence) == 1

    mutations = (
        "source_inventory_extra",
        "source_leaf_extra",
        "rendered_frame_extra",
        "selection_commitment_extra",
        "selection_extra",
        "selection_scene_hash",
        "selection_row_count",
        "selection_source_rows",
        "selection_render_mismatch",
        "plan_schema",
        "plan_camera_extra",
        "plan_fov_axis",
        "summary_projection_extra",
        "object_parity_extra",
        "duplicate_selected_jsonl",
        "missing_selected_jsonl",
        "duplicate_nonfit_selected_jsonl",
        "missing_nonfit_selected_jsonl",
        "retimestamp_nonfit_selected_jsonl",
        "malformed_outside_selection_jsonl",
    )
    for mutation in mutations:
        changed = copy.deepcopy(parsed)
        summary = changed[paths["fit_render_summary"]]
        selection = changed[paths["fit_frame_selection"]]
        plan = changed[paths["render_source_plan"]]
        frames = changed[paths["source_frames_jsonl"]]
        assert isinstance(summary, dict)
        assert isinstance(selection, dict)
        assert isinstance(plan, dict)
        assert isinstance(frames, list)
        if mutation == "source_inventory_extra":
            summary["source"]["extra"] = copy.deepcopy(summary["source"]["plan"])
        elif mutation == "source_leaf_extra":
            summary["source"]["plan"]["extra"] = True
        elif mutation == "rendered_frame_extra":
            summary["rendered_frames"][0]["extra"] = True
        elif mutation == "selection_commitment_extra":
            summary["frame_selection"]["extra"] = True
        elif mutation == "selection_extra":
            selection["extra"] = True
        elif mutation == "selection_scene_hash":
            selection["scene_id_sha256"] = "0" * 64
            _recommit_selection(selection)
        elif mutation == "selection_row_count":
            selection["row_count"] = 0
            _recommit_selection(selection)
        elif mutation == "selection_source_rows":
            selection["source_rows"]["extra"] = True
        elif mutation == "selection_render_mismatch":
            selection["frame_keys"].append([100, 0])
            selection["frame_count"] = 3
            selection["frame_key_set_sha256"] = finalizer.canonical_json_sha256(
                selection["frame_keys"]
            )
            _recommit_selection(selection)
            summary["frame_selection"]["frame_key_set_sha256"] = selection[
                "frame_key_set_sha256"
            ]
        elif mutation == "plan_schema":
            plan["schema"] = "wrong"
        elif mutation == "plan_camera_extra":
            plan["camera"]["extra"] = True
        elif mutation == "plan_fov_axis":
            plan["camera"]["fov_axis"] = "vertical"
        elif mutation == "summary_projection_extra":
            summary["camera_projection"]["extra"] = True
        elif mutation == "object_parity_extra":
            summary["object_parity"]["extra"] = True
        elif mutation == "duplicate_selected_jsonl":
            frames.append(copy.deepcopy(frames[1]))
        elif mutation == "missing_selected_jsonl":
            frames.pop(1)
        elif mutation == "duplicate_nonfit_selected_jsonl":
            frames.append(copy.deepcopy(frames[0]))
        elif mutation == "missing_nonfit_selected_jsonl":
            frames.pop(0)
        elif mutation == "retimestamp_nonfit_selected_jsonl":
            frames[0]["timestamp_ns"] = 100
        elif mutation == "malformed_outside_selection_jsonl":
            frames[2]["frame_index"] = False
        else:  # pragma: no cover - keeps the mutation registry exhaustive.
            raise AssertionError(mutation)
        with pytest.raises(finalizer.FinalizationError):
            finalizer._validate_source_scene_provenance(
                machine_entries=entries,
                parsed_by_path=changed,
                panel_records=panel,
                selected_scene_families={
                    "synthetic_scene": finalizer.FAMILIES[0]
                },
            )


@pytest.mark.parametrize(
    ("duplicate_order_key", "reuse_storage_row", "message"),
    (
        (True, False, "frame-order tie"),
        (False, True, "reuses one selected shard"),
    ),
)
def test_panel_derivation_rejects_order_ties_and_reused_selected_rows(
    duplicate_order_key: bool, reuse_storage_row: bool, message: str
) -> None:
    panel, content_hash, rows_hash = _small_panel(
        duplicate_order_key=duplicate_order_key,
        reuse_storage_row=reuse_storage_row,
    )
    with pytest.raises(finalizer.FinalizationError, match=message):
        finalizer._derive_panel_records(
            panel,
            expected_content_sha256=content_hash,
            expected_rows_sha256=rows_hash,
            expected_transitions=10 if duplicate_order_key else 5,
        )


@pytest.mark.parametrize("seed", range(5))
def test_stdlib_frame_evidence_matches_primary(seed: int) -> None:
    generator = np.random.default_rng(seed)
    target = generator.integers(0, 3, size=(64, 64), dtype=np.uint8)
    key = {
        "family": finalizer.FAMILIES[seed],
        "scene_id": f"scene_{seed}",
        "global_row": seed,
        "side": finalizer.ENDPOINT_SIDES[seed % 2],
        "image_sha256": _digest(f"image:{seed}"),
        "label_shard_sha256": _digest(f"shard:{seed}"),
        "label_row": seed,
    }
    actual = finalizer._stdlib_analyze_frame_labels(
        target.tobytes(), bytes([1]) * 4096, frame_key=key
    )
    primary = analyze_frame_labels(
        target,
        np.ones((64, 64), dtype=bool),
        frame_key=key,
        family=key["family"],
        endpoint_side=key["side"],
    )
    assert actual["label_support"] == primary["label_support"]
    assert actual["ray_sequences"] == {
        name: primary["ray_sequences"][name]
        for name in (
            "schema",
            "summary",
            "sequence_summary_records_sha256",
            "transition_table_sha256",
        )
    }


@pytest.mark.parametrize("version", ((1, 0), (2, 0), (3, 0)))
def test_npy_v1_v2_v3_and_npz_selected_rows_are_decoded(version: tuple[int, int]) -> None:
    payload = _npz_bytes(version=version)
    rows, counts = finalizer._decode_fit_label_npz(
        payload,
        selections=(("current", 0), ("next", 1)),
        context="synthetic shard",
    )
    assert len(rows) == 2
    assert all(target == bytes(4096) for target, _mask in rows)
    assert all(mask == bytes([1]) * 4096 for _target, mask in rows)
    assert counts["registered_arrays_decompressed"] == 4
    assert counts["materialized_label_rows"] == 4
    assert counts["materialized_supervision_rows"] == 4
    assert {tuple(item["npy_version"]) for item in counts["arrays"]} == {version}


def test_npz_only_decompresses_four_registered_arrays(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    original = zipfile.ZipFile.read

    def spy(self: zipfile.ZipFile, name: object, *args: object, **kwargs: object) -> bytes:
        opened.append(name.filename if isinstance(name, zipfile.ZipInfo) else str(name))
        return original(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "read", spy)
    finalizer._decode_fit_label_npz(
        _npz_bytes(), selections=(("current", 0),), context="synthetic shard"
    )
    assert set(opened) == {f"{name}.npy" for name in finalizer.REGISTERED_LABEL_ARRAYS}


def test_unselected_values_are_not_inspected_but_selected_noncanonical_bool_rejects() -> None:
    masks = bytearray(bytes([1]) * (2 * 4096))
    masks[4096] = 2
    payload = _npz_bytes(mask_override={"current": bytes(masks)})
    finalizer._decode_fit_label_npz(
        payload, selections=(("current", 0),), context="synthetic shard"
    )
    with pytest.raises(finalizer.FinalizationError, match="noncanonical bool"):
        finalizer._decode_fit_label_npz(
            payload, selections=(("current", 1),), context="synthetic shard"
        )


def test_npz_rejects_extra_member_and_local_central_name_mismatch() -> None:
    with pytest.raises(finalizer.FinalizationError, match="inventory"):
        finalizer._decode_fit_label_npz(
            _npz_bytes(extra_name="extra.npy"),
            selections=(("current", 0),),
            context="synthetic shard",
        )
    payload = bytearray(_npz_bytes())
    with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
        info = archive.getinfo("current_labels.npy")
        name_start = info.header_offset + 30
    payload[name_start] = ord("X")
    with pytest.raises(finalizer.FinalizationError, match="central/local names"):
        finalizer._decode_fit_label_npz(
            bytes(payload), selections=(("current", 0),), context="synthetic shard"
        )


def test_npy_duplicate_header_key_object_dtype_and_trailing_payload_reject() -> None:
    duplicate = _npy_bytes(
        bytes(4096), rows=1, descr="|u1", duplicate_descr=True
    )
    with pytest.raises(finalizer.FinalizationError, match="duplicate key"):
        finalizer._decode_npy_array(
            duplicate, expected_kind="labels", context="duplicate"
        )
    object_dtype = _npy_bytes(bytes(4096), rows=1, descr="|O")
    with pytest.raises(finalizer.FinalizationError, match="unsupported or object"):
        finalizer._decode_npy_array(
            object_dtype, expected_kind="labels", context="object"
        )
    valid = _npy_bytes(bytes(4096), rows=1, descr="|u1")
    with pytest.raises(finalizer.FinalizationError, match="trailing bytes"):
        finalizer._decode_npy_array(
            valid + b"x", expected_kind="labels", context="trailing"
        )
    float_shape = _npy_bytes(bytes(4096), rows=1.0, descr="|u1")
    with pytest.raises(finalizer.FinalizationError, match=r"shape \[N,64,64\]"):
        finalizer._decode_npy_array(
            float_shape, expected_kind="labels", context="float-shape"
        )


def test_npz_storage_rows_are_shape_bound_not_artificially_capped() -> None:
    selected, counts = finalizer._decode_fit_label_npz(
        _npz_bytes(rows=161),
        selections=[("current", 160)],
        context="large-valid-shard",
    )
    assert len(selected) == 1
    assert counts["storage_rows_per_array"] == 161
    assert counts["materialized_label_rows"] == 322


@pytest.mark.parametrize(
    ("path", "role", "declared", "modality", "reason"),
    (
        ("/tmp/sealed/a.json", "fit_panel", "train", "json", "sealed"),
        ("/tmp/g2/a.json", "fit_panel", "train", "json", "g2"),
        ("/tmp/seed_20260711.json", "fit_panel", "train", "json", "seed_20260711"),
        ("/tmp/seed_20260710_result.json", "fit_panel", "train", "json", "generated_v4_result"),
        ("/tmp/a.pt", "fit_panel", "train", "json", "model"),
        ("/tmp/probability.json", "fit_panel", "train", "json", "model"),
        (
            "/tmp/calibration/selection.json",
            "fit_frame_selection",
            "train",
            "json",
            "selection_or_calibration",
        ),
        (
            "/tmp/physical_nontrain/summary.json",
            "fit_render_summary",
            "train",
            "json",
            "physical_nontrain",
        ),
        ("/tmp/rgb/frame.json", "fit_panel", "train", "json", "image_or_depth"),
        ("/tmp/a.png", "fit_panel", "train", "image", "image_or_depth"),
        ("/tmp/a.json", "unknown", "unknown", "json", "unregistered_role"),
    ),
)
def test_semantic_denial_precedence_is_lexical_and_ledgered(
    path: str, role: str, declared: str, modality: str, reason: str
) -> None:
    ledger = finalizer.new_finalizer_access_ledger()
    with pytest.raises(finalizer.FinalizationError, match=reason):
        finalizer._authorize_path(
            Path(path),
            requested_role=role,
            declared_role=declared,
            modality=modality,
            allowlist={},
            ledger=ledger,
        )
    assert ledger["denied_attempts_total"] == 1
    assert ledger["unexpected_path_attempts"] == 1
    assert ledger["denied_primary_reasons"][reason] == 1


def test_semantic_denial_compact_spellings_match_runner_precedence() -> None:
    from scripts import audit_go2_n32_camera_frustum_observability as runner

    for role in (
        "audit_core_test",
        "audit_runner_test",
        "audit_finalizer_test",
    ):
        path = Path(f"/repo/lewm/tests/{role}.py")
        assert runner._lexical_primary_denial(
            path,
            requested_role="implementation_source",
            declared_role="implementation_source",
            modality="python_source",
        ) is None
        assert finalizer._lexical_denial_reason(
            path,
            requested_role=role,
            declared_role=role,
            modality="python_source",
        ) is None

    dataset_path = Path("/repo/physical.json")
    assert runner._lexical_primary_denial(
        dataset_path,
        requested_role="fit_panel",
        declared_role="test",
        modality="json",
    ) == "physical_nontrain"
    assert finalizer._lexical_denial_reason(
        dataset_path,
        requested_role="fit_panel",
        declared_role="test",
        modality="json",
    ) == "physical_nontrain"

    cases = {
        "/tmp/sealedpayload/x.json": "sealed",
        "/tmp/g2payload/x.json": "g2",
        "/tmp/seed20260711/x.json": "seed_20260711",
        "/tmp/generatedv4result/x.json": "generated_v4_result",
        "/tmp/seed_20260710_result.json": "generated_v4_result",
        "/tmp/model_output/x.json": "model",
        "/tmp/models/x.json": "model",
        "/tmp/parameters/x.json": "model",
        "/tmp/closedloop/x.json": "runtime",
        "/tmp/physical_nontrain/x.json": "physical_nontrain",
        "/tmp/calib/x.json": "selection_or_calibration",
        "/tmp/heldout/x.json": "holdout",
        "/tmp/pixels/x.json": "image_or_depth",
        "/tmp/images/x.json": "image_or_depth",
    }
    for raw_path, expected in cases.items():
        path = Path(raw_path)
        runner_modality = runner._infer_modality(path)
        finalizer_modality = finalizer._infer_lexical_modality(path)
        assert runner_modality == finalizer_modality
        runner_reason = runner._lexical_primary_denial(
            path,
            requested_role="fit_panel",
            declared_role="train",
            modality=runner_modality,
        )
        finalizer_reason = finalizer._lexical_denial_reason(
            path,
            requested_role="fit_panel",
            declared_role="train",
            modality=finalizer_modality,
        )
        assert runner_reason == finalizer_reason == expected
    for suffix in (
        ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff",
        ".exr", ".hdr", ".gif", ".mp4", ".avi", ".mov", ".mkv", ".webm",
    ):
        path = Path(f"/tmp/payload{suffix}")
        runner_modality = runner._infer_modality(path)
        finalizer_modality = finalizer._infer_lexical_modality(path)
        assert runner_modality == finalizer_modality
        assert runner._lexical_primary_denial(
            path,
            requested_role="fit_panel",
            declared_role="train",
            modality=runner_modality,
        ) == finalizer._lexical_denial_reason(
            path,
            requested_role="fit_panel",
            declared_role="train",
            modality=finalizer_modality,
        ) == "image_or_depth"
    requested_role_cases = {
        "sealed_payload": "sealed",
        "g2_evaluation": "g2",
        "seed20260711": "seed_20260711",
        "generated_v4_result": "generated_v4_result",
        "model_output": "model",
        "development_runtime": "runtime",
        "physical_nontrain": "physical_nontrain",
        "probability_calibration": "model",
        "heldout": "holdout",
        "rgb_image": "image_or_depth",
    }
    neutral = Path("/tmp/payload.json")
    for requested_role, expected in requested_role_cases.items():
        runner_reason = runner._lexical_primary_denial(
            neutral,
            requested_role=requested_role,
            declared_role="train",
            modality="json",
        )
        finalizer_reason = finalizer._lexical_denial_reason(
            neutral,
            requested_role=requested_role,
            declared_role="train",
            modality="json",
        )
        assert runner_reason == finalizer_reason == expected
    assert finalizer._lexical_denial_reason(
        finalizer.RESULT_PATH,
        requested_role="audit_output",
        declared_role="audit_output",
        modality="json",
    ) is None
    declared_role_cases = {
        "same_scene_holdout": "holdout",
        "unknown_role": "unregistered_role",
    }
    for declared_role, expected in declared_role_cases.items():
        runner_reason = runner._lexical_primary_denial(
            neutral,
            requested_role="fit_panel",
            declared_role=declared_role,
            modality="json",
        )
        finalizer_reason = finalizer._lexical_denial_reason(
            neutral,
            requested_role="fit_panel",
            declared_role=declared_role,
            modality="json",
        )
        assert runner_reason == finalizer_reason == expected
    calibration = Path("/tmp/calibration/x.json")
    assert runner._lexical_primary_denial(
        calibration,
        requested_role="fit_frame_selection",
        declared_role="train",
        modality="json",
    ) == finalizer._lexical_denial_reason(
        calibration,
        requested_role="fit_frame_selection",
        declared_role="train",
        modality="json",
    ) == "selection_or_calibration"


def test_symlink_alias_is_denied_after_lexical_eligibility(tmp_path: Path) -> None:
    target = tmp_path / "panel.json"
    target.write_text("{}", encoding="utf-8")
    alias = tmp_path / "alias.json"
    alias.symlink_to(target)
    ledger = finalizer.new_finalizer_access_ledger()
    with pytest.raises(finalizer.FinalizationError, match="symlink or alias"):
        finalizer._authorize_path(
            alias,
            requested_role="fit_panel",
            declared_role="train",
            modality="json",
            allowlist={alias.absolute(): _digest("content")},
            ledger=ledger,
            repository_root=tmp_path,
        )
    assert ledger["denied_primary_reasons"]["path_alias_or_escape"] == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("phantom_scene", "identity/order"),
        ("reversed_order", "identity/order"),
        ("impossible_label_row", "identity/order"),
        ("arbitrary_label_hash", "panel-derived"),
    ),
)
def test_panel_bound_identity_and_shard_fabrications_reject(
    mutation: str, message: str
) -> None:
    records, manifest, reports = _panel_records_and_manifests()
    changed_reports = copy.deepcopy(reports)
    changed_manifest = copy.deepcopy(manifest)
    if mutation == "phantom_scene":
        changed_reports[0]["record_key"]["scene_id"] = "phantom"
    elif mutation == "reversed_order":
        changed_reports[0], changed_reports[1] = changed_reports[1], changed_reports[0]
    elif mutation == "impossible_label_row":
        changed_reports[0]["record_key"]["label_row"] = 999
    else:
        changed_manifest["entries"][0]["sha256"] = "0" * 64
        changed_manifest["manifest_sha256"] = finalizer.canonical_json_sha256(
            changed_manifest["entries"]
        )
    if mutation == "arbitrary_label_hash":
        with pytest.raises(finalizer.FinalizationError, match=message):
            finalizer._validate_label_shard_manifest(
                changed_manifest,
                expected_entries=manifest["entries"],
            )
    else:
        label_manifest = finalizer._validate_label_shard_manifest(
            manifest, expected_entries=manifest["entries"]
        )
        with pytest.raises(finalizer.FinalizationError, match=message):
            finalizer._validate_frame_reports(
                changed_reports,
                label_manifest=label_manifest,
                expected_panel_records=records,
            )


def test_camera_mount_aggregate_hash_and_authorization_are_recomputed() -> None:
    records, _manifest, reports = _panel_records_and_manifests()
    expected = {
        tuple(record[field] for field in finalizer.FRAME_IDENTITY_FIELDS): report[
            "camera_mount_composition"
        ]
        for record, report in zip(records, reports)
    }
    ordered = [
        {
            "record_key": report["record_key"],
            "camera_mount_composition": report["camera_mount_composition"],
        }
        for report in reports
    ]
    aggregate = {
        "frame_count": 320,
        "pass_count": 320,
        "failure_count": 0,
        "passes": True,
        "ordered_frame_evidence_sha256": finalizer.canonical_json_sha256(ordered),
    }
    validated = finalizer._validate_camera_mount_composition_aggregate(
        aggregate,
        reports=reports,
        expected_camera_evidence=expected,
    )
    assert validated["passes"] is True
    changed = copy.deepcopy(aggregate)
    changed["ordered_frame_evidence_sha256"] = "0" * 64
    with pytest.raises(finalizer.FinalizationError, match="evidence hash"):
        finalizer._validate_camera_mount_composition_aggregate(
            changed,
            reports=reports,
            expected_camera_evidence=expected,
        )

    box_parity = {
        "aggregate": {
            "required_provenance_missing_count": 0,
            "required_provenance_nonunique_count": 0,
        }
    }
    provenance = {
        "passes": False,
        "fit_panel_file_hash_pass": True,
        "fit_panel_content_hash_pass": True,
        "current_physical_dataset_role_train_only": True,
        "fit_frame_identity_unique": True,
        "one_to_one_frame_match": True,
        "source_hashes_pass": True,
        "source_geometry_allowlisted_before_parse": True,
        "source_geometry_rehashed_after_parse": True,
        "rendered_collision_provenance_complete": True,
        "fixed_camera_mount_composition_complete": False,
        "legacy_source_split_used_for_selection": False,
    }
    camera_failed = {**aggregate, "pass_count": 319, "failure_count": 1, "passes": False}
    finalizer._validate_provenance(
        provenance,
        box_parity=box_parity,
        camera_mount_composition=camera_failed,
    )
    decision = finalizer._authorization_decision(
        provenance_passes=False,
        source_hashes_pass=True,
        reconstruction_passes=True,
        access_reconciliation_passes=True,
        mapping_passes=True,
        coverage_passes=True,
        ambiguity=True,
    )
    assert decision["camera_frustum_representation_implementation_authorized"] is False
    assert decision["target_amendment_required_before_model_output"] is True


def test_arbitrary_source_hash_and_scene_assignment_reject() -> None:
    expected = [
        {
            "path": "/synthetic/summary.json",
            "sha256": _digest("summary"),
            "semantic_role": "fit_render_summary",
            "scene_id": "scene_00",
        }
    ]
    changed = copy.deepcopy(expected)
    changed[0]["sha256"] = "0" * 64
    manifest = {
        "entry_count": 1,
        "entries": changed,
        "manifest_sha256": finalizer.canonical_json_sha256(changed),
    }
    with pytest.raises(finalizer.FinalizationError, match="machine-authorized"):
        finalizer._validate_source_geometry_manifest(
            manifest,
            expected_entries=expected,
            selected_scene_ids={"scene_00"},
        )


def test_collision_attribution_partitions_reject_contradictory_counts() -> None:
    report = _report(
        _frame_key(0, finalizer.FAMILIES[0], "current", _digest("shard"))
    )
    report["collision_veto_only_unknown_count"] = 1
    report["collision_veto_only_unknown_identities"] = [[1, 2]]
    report["attribution_partitions"]["matched_true_unmatched_false"] = {
        "count": 1,
        "identities": [[1, 2]],
        "identities_sha256": finalizer.canonical_json_sha256([[1, 2]]),
    }
    report["attributed_to_matched_box_count"] = 0
    with pytest.raises(finalizer.FinalizationError, match="contradicts"):
        finalizer._validate_frame_attribution(
            report, context="synthetic", veto_cells=[[1, 2]]
        )
    geometry = [float(index) for index in range(15)]
    unmatched = [{"index": 0, "canonical_geometry": geometry}]
    assert finalizer._validate_unmatched_box_records(
        unmatched, context="synthetic unmatched", box_count=1
    ) == unmatched
    duplicated = [*unmatched, copy.deepcopy(unmatched[0])]
    with pytest.raises(finalizer.FinalizationError, match="unique"):
        finalizer._validate_unmatched_box_records(
            duplicated, context="synthetic unmatched", box_count=1
        )


def _independent_label_fixture() -> tuple[list[dict], dict, dict, dict]:
    records = []
    reports = []
    selected = {}
    all_ray_records = []
    for family_index, family in enumerate(finalizer.FAMILIES):
        for side_index, side in enumerate(finalizer.ENDPOINT_SIDES):
            index = family_index * 2 + side_index
            key = _frame_key(index, family, side, _digest(f"shard:{family_index}"))
            record = {**key, "label_shard_path": f"/synthetic/{family}.npz"}
            target = bytes(4096)
            mask = bytes([1]) * 4096
            evidence = finalizer._stdlib_analyze_frame_labels(
                target, mask, frame_key=key
            )
            report = {
                **_report(key),
                "label_support": evidence["label_support"],
                "ray_sequences": evidence["ray_sequences"],
            }
            records.append(record)
            reports.append(report)
            selected[tuple(key[field] for field in finalizer.FRAME_IDENTITY_FIELDS)] = (
                target,
                mask,
            )
            all_ray_records.append(evidence["ray_records"])

    def scope(indices: list[int]) -> dict:
        selected_reports = [reports[index] for index in indices]
        rays = [ray for index in indices for ray in all_ray_records[index]]
        by_class = {
            name: {
                field: sum(
                    report["label_support"]["by_class"][name][field]
                    for report in selected_reports
                )
                for field in ("total", "supported", "unsupported")
            }
            for name in finalizer.CLASS_NAMES
        }
        unsupported = sum(value["unsupported"] for value in by_class.values())
        support = {
            "schema": finalizer.LABEL_SUPPORT_SCHEMA,
            "frame_count": len(indices),
            "total_supervised_label_count": len(indices) * 4096,
            "supported_label_count": len(indices) * 1990,
            "unsupported_label_count": len(indices) * 2106,
            "class_counts": {name: by_class[name]["total"] for name in finalizer.CLASS_NAMES},
            "by_class": by_class,
            "unsupported_free_count": by_class["free"]["unsupported"],
            "unsupported_occupied_count": by_class["occupied"]["unsupported"],
            "unsupported_unknown_count": by_class["unknown"]["unsupported"],
            "unsupported_targets_are_all_unknown": by_class["unknown"]["unsupported"] == unsupported,
            "violations": [],
            "passes": True,
        }
        return {
            "frame_count": len(indices),
            "label_support": support,
            "ray_sequences": {
                "schema": finalizer.RAY_SEQUENCE_SCHEMA,
                "summary": finalizer._stdlib_ray_summary(rays),
                "sequence_summary_records_sha256": finalizer.canonical_json_sha256(rays),
            },
        }

    aggregate = scope(list(range(10)))
    families = {
        family: scope([family_index * 2, family_index * 2 + 1])
        for family_index, family in enumerate(finalizer.FAMILIES)
    }
    sides = {"current": scope([0, 2, 4, 6, 8]), "next": scope([1, 3, 5, 7, 9])}
    transition_tables = {
        "aggregate": aggregate["ray_sequences"]["summary"],
        "families": {family: families[family]["ray_sequences"]["summary"] for family in finalizer.FAMILIES},
        "endpoint_sides": {side: sides[side]["ray_sequences"]["summary"] for side in finalizer.ENDPOINT_SIDES},
    }
    observability = {
        "aggregate": aggregate,
        "families": families,
        "endpoint_sides": sides,
        "ordered_sequence_summary_records_sha256": aggregate["ray_sequences"]["sequence_summary_records_sha256"],
        "aggregate_transition_tables_sha256": finalizer.canonical_json_sha256(transition_tables),
        "representative_scalar_first_hit_violations": {
            "limit": 32,
            "total_violation_count": 0,
            "records": [],
            "records_sha256": finalizer.canonical_json_sha256([]),
        },
    }
    class_rows = finalizer._expected_family_class_count_rows(reports)
    class_table = {
        "family_order": list(finalizer.FAMILIES),
        "rows": class_rows,
        "table_sha256": finalizer.canonical_json_sha256(class_rows),
    }
    return records, reports, selected, {
        "observability": observability,
        "class_table": class_table,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("selected_label_hash", "selected_label_bytes"),
        ("frame_class_counts", "label/support evidence"),
        ("ray_hash", "ray evidence"),
        ("family_class_table", "class-count table"),
    ),
)
def test_independent_selected_bytes_reject_gate_critical_fabrications(
    mutation: str, message: str
) -> None:
    records, reports, selected, summaries = _independent_label_fixture()
    selected_digest = hashlib.sha256(bytes(4096) * len(records)).hexdigest()
    selected_record = {"sha256": selected_digest}
    if mutation == "selected_label_hash":
        selected_record["sha256"] = "0" * 64
    elif mutation == "frame_class_counts":
        reports[0]["label_support"]["class_counts"]["unknown"] -= 1
    elif mutation == "ray_hash":
        reports[0]["ray_sequences"]["sequence_summary_records_sha256"] = "0" * 64
    else:
        summaries["class_table"]["rows"][0]["unknown"] -= 1
    with pytest.raises(finalizer.FinalizationError, match=message):
        finalizer._reconcile_selected_label_evidence(
            panel_records=records,
            selected_rows=selected,
            reports=reports,
            selected_label_bytes=selected_record,
            label_observability=summaries["observability"],
            family_class_count_table=summaries["class_table"],
        )


def test_runtime_version_mismatch_and_nonfinite_duplicate_json_reject() -> None:
    runtime = finalizer._current_runtime_environment()
    finalizer._validate_runtime_environment(runtime, context="synthetic runtime")
    changed = copy.deepcopy(runtime)
    changed["numpy_version"] = "0.0"
    with pytest.raises(finalizer.FinalizationError, match="active frozen runtime"):
        finalizer._validate_runtime_environment(changed, context="synthetic runtime")
    with pytest.raises(finalizer.FinalizationError, match="duplicate JSON key"):
        finalizer._strict_json_loads(b'{"a":1,"a":2}', context="synthetic")
    with pytest.raises(finalizer.FinalizationError, match="nonfinite JSON number"):
        finalizer._strict_json_loads(b'{"a":1e999}', context="synthetic")
    with pytest.raises(finalizer.FinalizationError, match="strict canonical JSON"):
        finalizer.canonical_json_sha256({"bad": float("nan")})
    assert finalizer._strict_jsonl_loads(b'{"frame_index":1}\n', context="jsonl") == [
        {"frame_index": 1}
    ]
    with pytest.raises(finalizer.FinalizationError, match="end with a newline"):
        finalizer._strict_jsonl_loads(b'{"frame_index":1}', context="jsonl")
    with pytest.raises(finalizer.FinalizationError, match="blank JSONL record"):
        finalizer._strict_jsonl_loads(
            b'{"frame_index":1}\n\n', context="jsonl"
        )


def _canonical_manifest(entries: list[dict]) -> dict:
    return {
        "entry_count": len(entries),
        "entries": entries,
        "manifest_sha256": finalizer.canonical_json_sha256(entries),
    }


def _synthetic_machine_manifest() -> tuple[dict, dict[str, dict[str, str]]]:
    source_hashes = {
        role: {"path": str(path), "sha256": _digest(f"source:{role}")}
        for role, path in sorted(finalizer._source_paths().items())
    }
    source_map_entries = [
        {"role": role, **record} for role, record in source_hashes.items()
    ]
    label_entries = [
        {"path": f"/synthetic/shard_{index}.npz", "sha256": _digest(index)}
        for index in range(20)
    ]
    summary_entries = [
        {
            "path": f"/synthetic/scene_{index}/summary.json",
            "sha256": _digest(f"summary:{index}"),
            "semantic_role": "fit_render_summary",
            "scene_id": f"scene_{index}",
        }
        for index in range(20)
    ]
    summary_entries.sort(
        key=lambda entry: (
            entry["path"], entry["semantic_role"], entry["scene_id"]
        )
    )
    preparation = finalizer.new_finalizer_access_ledger()
    preparation.update(
        {
            "passes": True,
            "forbidden_counters_zero": True,
            "panel_metadata_byte_opens": 1,
            "implementation_source_hash_byte_opens": 22,
            "document_hash_byte_opens": 4,
            "source_geometry_hash_byte_opens": 40,
            "source_geometry_json_parses": 20,
            "source_geometry_jsonl_records": 320,
            "source_frame_records_selected": 320,
            "per_shard_materialization": [],
        }
    )
    core = {
        "schema": finalizer.MACHINE_MANIFEST_SCHEMA,
        "created_at_utc": "2026-07-11T12:00:00+00:00",
        "binding": {
            "path": str(finalizer.BINDING_PATH),
            "file_sha256": finalizer.EXECUTION_BINDING_SHA256,
        },
        "preflight_access_incident": finalizer._known_input_records()[
            "preflight_access_incident"
        ],
        "human_implementation_manifest": {
            "path": str(finalizer.IMPLEMENTATION_MANIFEST_PATH),
            "file_sha256": _digest("human"),
        },
        "authorized_inputs": {
            "fit_panel": {
                "semantic_role": "fit_panel",
                "path": str(finalizer.FIT_PANEL_PATH),
                "file_sha256": finalizer.FIT_PANEL_FILE_SHA256,
                "content_sha256": finalizer.FIT_PANEL_CONTENT_SHA256,
                "fit_rows_sha256": finalizer.FIT_ROWS_SHA256,
                "schema": "lewm_go2_physical_micro_overfit_panel_v1",
            },
            "v4_adjudication_report": {
                "semantic_role": "v4_adjudication_report",
                "path": str(finalizer.V4_ADJUDICATION_REPORT_PATH),
                "file_sha256": finalizer.V4_ADJUDICATION_REPORT_SHA256,
            },
            "known_bias_proof": {
                "semantic_role": "known_bias_proof",
                "path": str(finalizer.KNOWN_BIAS_PROOF_PATH),
                "file_sha256": finalizer.KNOWN_BIAS_PROOF_SHA256,
            },
            "physical_geometry_contract": {
                "semantic_role": "physical_geometry_contract",
                "path": "/synthetic/geometry.json",
                "file_sha256": _digest("geometry"),
                "semantic_sha256": _digest("geometry-semantic"),
                "schema": "lewm_go2_generalization_geometry_v2",
            },
            "label_shards": _canonical_manifest(label_entries),
            "render_summaries": _canonical_manifest(summary_entries),
            "source_geometry": _canonical_manifest(summary_entries),
        },
        "source_map": {
            "entry_count": 11,
            "entries": source_map_entries,
            "source_map_sha256": finalizer.canonical_json_sha256(source_map_entries),
        },
        "runtime_environment": finalizer._current_runtime_environment(),
        "verification_evidence": {
            "all_passed": True,
            "commands": [
                {
                    **copy.deepcopy(expected),
                    "exit_code": 0,
                    "captured_output_sha256": _digest(
                        f"verification:{expected['category']}"
                    ),
                }
                for expected in finalizer.REQUIRED_VERIFICATION_COMMANDS
            ],
        },
        "exclusive_output": {
            "path": str(finalizer.RESULT_PATH),
            "schema": finalizer.RESULT_SCHEMA,
            "absent_before_authorization": True,
            "zero_output_state": True,
        },
        "preparation_access_ledger": preparation,
        "review": {
            "reviewer_identity": "synthetic-reviewer",
            "status": "reviewed_and_authorized",
        },
        "authoritative_fit_audit_authorized": True,
    }
    return {**core, "content_sha256": finalizer.canonical_json_sha256(core)}, source_hashes


def test_machine_manifest_schema_runtime_sources_and_incident_are_strict() -> None:
    manifest, source_hashes = _synthetic_machine_manifest()
    finalizer._validate_machine_manifest(
        manifest,
        raw_file_sha256=_digest("machine"),
        expected_source_hashes=source_hashes,
    )
    changed = copy.deepcopy(manifest)
    changed["preflight_access_incident"]["status"] = "erased"
    core = dict(changed)
    core.pop("content_sha256")
    changed["content_sha256"] = finalizer.canonical_json_sha256(core)
    with pytest.raises(finalizer.FinalizationError, match="incident"):
        finalizer._validate_machine_manifest(
            changed,
            raw_file_sha256=_digest("machine"),
            expected_source_hashes=source_hashes,
        )
    changed = copy.deepcopy(manifest)
    changed["source_map"]["entries"][0]["path"] = "/synthetic/substitute.py"
    changed["source_map"]["source_map_sha256"] = finalizer.canonical_json_sha256(
        changed["source_map"]["entries"]
    )
    core = dict(changed)
    core.pop("content_sha256")
    changed["content_sha256"] = finalizer.canonical_json_sha256(core)
    substituted = copy.deepcopy(source_hashes)
    substituted[changed["source_map"]["entries"][0]["role"]]["path"] = (
        "/synthetic/substitute.py"
    )
    with pytest.raises(finalizer.FinalizationError, match="substituted"):
        finalizer._validate_machine_manifest(
            changed,
            raw_file_sha256=_digest("machine"),
            expected_source_hashes=substituted,
        )
    verification_mutations = []
    changed = copy.deepcopy(manifest)
    changed["verification_evidence"]["commands"][0]["command"] += " ; echo forged"
    verification_mutations.append(changed)
    changed = copy.deepcopy(manifest)
    changed["verification_evidence"]["commands"][0:2] = reversed(
        changed["verification_evidence"]["commands"][0:2]
    )
    verification_mutations.append(changed)
    changed = copy.deepcopy(manifest)
    changed["verification_evidence"]["commands"].append(
        copy.deepcopy(changed["verification_evidence"]["commands"][-1])
    )
    verification_mutations.append(changed)
    for changed in verification_mutations:
        core = dict(changed)
        core.pop("content_sha256")
        changed["content_sha256"] = finalizer.canonical_json_sha256(core)
        with pytest.raises(
            finalizer.FinalizationError,
            match="frozen contract|exact four commands",
        ):
            finalizer._validate_machine_manifest(
                changed,
                raw_file_sha256=_digest("machine"),
                expected_source_hashes=source_hashes,
            )


def _synthetic_phase_contract() -> tuple[dict, list[dict], dict, dict]:
    manifest, _source_hashes = _synthetic_machine_manifest()
    shard_paths = sorted(f"/synthetic/shard_{index}.npz" for index in range(20))
    label_entries = [
        {
            "path": path,
            "selected_row_count": 16,
        }
        for path in shard_paths
    ]
    runner = finalizer.new_finalizer_access_ledger()
    runner.update(
        {
            "panel_metadata_byte_opens": 1,
            "label_shard_hash_byte_opens": 20,
            "label_shard_npz_opens": 20,
            "registered_arrays_decompressed": 80,
            "materialized_label_rows": 4000,
            "materialized_supervision_rows": 4000,
            "selected_label_rows_read": 320,
            "selected_supervision_rows_read": 320,
            "source_geometry_hash_byte_opens": 40,
            "source_geometry_json_parses": 20,
            "source_geometry_jsonl_records": 320,
            "source_frame_records_selected": 320,
            "implementation_source_hash_byte_opens": 22,
            "document_hash_byte_opens": 5,
            "per_shard_materialization": [
                {
                    "path": path,
                    "storage_rows_per_array": 100,
                    "materialized_label_rows": 200,
                    "materialized_supervision_rows": 200,
                    "selected_endpoint_rows": 16,
                }
                for path in shard_paths
            ],
        }
    )
    expected_finalizer = {
        field: 0 for field in finalizer.FINALIZER_LEDGER_SCALAR_FIELDS
    }
    expected_finalizer.update(
        {
            "panel_metadata_byte_opens": 1,
            "document_hash_byte_opens": 16,
            "label_shard_hash_byte_opens": 20,
            "label_shard_npz_opens": 20,
            "registered_arrays_decompressed": 80,
            "materialized_label_rows": 4000,
            "materialized_supervision_rows": 4000,
            "selected_label_rows_read": 320,
            "selected_supervision_rows_read": 320,
            "source_geometry_hash_byte_opens": 20,
            "source_geometry_json_parses": 20,
            "source_geometry_jsonl_records": 320,
        }
    )
    reconciliation = {
        "phase_names": ["preparation", "runner"],
        "passes": True,
        "forbidden_counters_zero": True,
        "unexpected_paths_zero": True,
        "incident_separate": True,
        "expected_distinct_label_shards": 20,
        "selected_label_rows_each": 320,
        "selected_supervision_rows_each": 320,
        "source_geometry_unique_path_count": 20,
    }
    phases = {
        "preparation": manifest["preparation_access_ledger"],
        "runner": runner,
    }
    return manifest, label_entries, phases, {
        "expected_finalizer": expected_finalizer,
        "reconciliation": reconciliation,
    }


def test_phase_ledgers_reconcile_every_shard_and_authorized_read() -> None:
    manifest, label_entries, phases, contract = _synthetic_phase_contract()
    finalizer._validate_result_phase_ledgers(
        phase_ledgers_value=phases,
        expected_finalizer_value=contract["expected_finalizer"],
        reconciliation_value=contract["reconciliation"],
        machine_manifest=manifest,
        expected_label_shard_entries=label_entries,
    )

    changed = copy.deepcopy(phases)
    changed["runner"]["document_hash_byte_opens"] = 4
    with pytest.raises(finalizer.FinalizationError, match="document_hash_byte_opens"):
        finalizer._validate_result_phase_ledgers(
            phase_ledgers_value=changed,
            expected_finalizer_value=contract["expected_finalizer"],
            reconciliation_value=contract["reconciliation"],
            machine_manifest=manifest,
            expected_label_shard_entries=label_entries,
        )

    changed = copy.deepcopy(phases)
    changed["runner"]["per_shard_materialization"][0][
        "selected_endpoint_rows"
    ] = 15
    with pytest.raises(finalizer.FinalizationError, match="panel commitment"):
        finalizer._validate_result_phase_ledgers(
            phase_ledgers_value=changed,
            expected_finalizer_value=contract["expected_finalizer"],
            reconciliation_value=contract["reconciliation"],
            machine_manifest=manifest,
            expected_label_shard_entries=label_entries,
        )

    changed = copy.deepcopy(phases)
    changed["preparation"]["source_geometry_jsonl_records"] = 321
    changed_manifest = copy.deepcopy(manifest)
    changed_manifest["preparation_access_ledger"][
        "source_geometry_jsonl_records"
    ] = 321
    with pytest.raises(finalizer.FinalizationError, match="JSONL record counts differ"):
        finalizer._validate_result_phase_ledgers(
            phase_ledgers_value=changed,
            expected_finalizer_value=contract["expected_finalizer"],
            reconciliation_value=contract["reconciliation"],
            machine_manifest=changed_manifest,
            expected_label_shard_entries=label_entries,
        )


def test_runner_and_finalizer_per_shard_shapes_must_match_exactly() -> None:
    _manifest, _label_entries, phases, _contract = _synthetic_phase_contract()
    measured = finalizer.new_finalizer_access_ledger()
    measured["label_shards"] = [
        {
            "path": record["path"],
            "storage_rows_per_array": 100,
            "materialized_label_rows": 200,
            "materialized_supervision_rows": 200,
            "selected_label_rows_read": 16,
            "selected_supervision_rows_read": 16,
        }
        for record in phases["runner"]["per_shard_materialization"]
    ]
    finalizer._validate_measured_per_shard_materialization(
        phases=phases,
        measured_finalizer_ledger=measured,
    )
    changed = copy.deepcopy(phases)
    changed["runner"]["per_shard_materialization"][0].update(
        {
            "storage_rows_per_array": 99,
            "materialized_label_rows": 198,
            "materialized_supervision_rows": 198,
        }
    )
    changed["runner"]["per_shard_materialization"][1].update(
        {
            "storage_rows_per_array": 101,
            "materialized_label_rows": 202,
            "materialized_supervision_rows": 202,
        }
    )
    with pytest.raises(finalizer.FinalizationError, match="per-shard materialization"):
        finalizer._validate_measured_per_shard_materialization(
            phases=changed,
            measured_finalizer_ledger=measured,
        )


def test_zero_denial_ledger_requires_an_empty_record_list() -> None:
    ledger = finalizer.new_finalizer_access_ledger()
    ledger["denied_attempt_records"].append({"attempted_path": "/synthetic"})
    with pytest.raises(finalizer.FinalizationError, match="must remain empty"):
        finalizer._validate_fresh_ledger_zero_denials(ledger, context="synthetic")


def test_finalizer_requires_fixed_path_and_explicit_authorizations(tmp_path: Path) -> None:
    with pytest.raises(finalizer.FinalizationError, match="exclusive immutable"):
        finalizer._validate_result_path(tmp_path / "result.json")
    with pytest.raises(SystemExit):
        finalizer._parse_args([])
    with pytest.raises(SystemExit):
        finalizer._parse_args(
            [
                "--binding-sha256",
                "0" * 64,
                "--machine-manifest-sha256",
                "1" * 64,
            ]
        )


def test_finalizer_imports_under_system_python_without_numpy_or_torch() -> None:
    command = (
        "import sys; "
        "from scripts import finalize_go2_n32_camera_frustum_observability as f; "
        "assert 'numpy' not in sys.modules; assert 'torch' not in sys.modules; "
        "f._activate_import_isolation(); "
        "assert all(x not in sys.modules for x in ('numpy','torch')); "
        "exec(\"for name in ('numpy','torch'):\\n try:\\n  __import__(name)\\n  raise AssertionError(name)\\n except ModuleNotFoundError:\\n  pass\")"
    )
    completed = subprocess.run(
        ["/usr/bin/python3", "-c", command],
        cwd=finalizer.REPOSITORY_ROOT,
        env={
            **os.environ,
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(finalizer.REPOSITORY_ROOT),
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_finalizer_source_map_is_exact_eleven_roles() -> None:
    assert set(finalizer._source_paths()) == finalizer.SOURCE_HASH_KEYS
    assert len(finalizer._source_paths()) == 11
    assert all(path.is_absolute() for path in finalizer._source_paths().values())
