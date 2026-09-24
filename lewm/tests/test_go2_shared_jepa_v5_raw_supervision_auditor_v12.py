from __future__ import annotations

import ast
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import inspect
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import subprocess
import sys
from types import MappingProxyType, SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pytest

from lewm.datasets import go2_shared_jepa_v5_raw_supervision_auditor_v10 as predecessor_v10
from lewm.datasets import go2_shared_jepa_v5_raw_supervision_auditor_v11 as predecessor
from lewm.datasets import go2_shared_jepa_v5_raw_supervision_builder_v9 as builder_oracle
from lewm.datasets import go2_shared_jepa_v5_raw_supervision_auditor_v12 as auditor
from lewm_worlds import manifest as manifest_semantics
from scripts import audit_go2_n32_camera_frustum_observability as source_v4
from scripts import build_go2_observable_camera_ray_fit_v4 as raycast_v4


def _raw_manifest() -> dict[str, object]:
    box = {
        "object_id": "wall-0",
        "kind": "wall",
        "center_xyz_m": [1.0, 0.0, 0.5],
        "size_xyz_m": [0.2, 2.0, 1.0],
        "yaw_rad": 0.0,
        "roll_rad": 0.0,
        "pitch_rad": 0.0,
        "material_id": "wall",
    }
    return {
        "scene_id": "synthetic_scene",
        "family": "open_obstacle_field",
        "difficulty_tier": "development",
        "topology_seed": 1,
        "visual_seed": 2,
        "physics_seed": 3,
        "world_bounds_xy_m": [[-2.0, -2.0], [2.0, 2.0]],
        "spawn": {"xyz_m": [0.0, 0.0, 0.4], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]},
        "graph_nodes": [],
        "graph_edges": [],
        "walls": [box],
        "obstacles": [],
        "landmarks": [],
        "camera_constraints": {
            "min_wall_thickness_m": 0.08,
            "near_m": 0.05,
            "far_m": 20.0,
            "min_camera_clearance_m": 0.1,
        },
        "visual_randomization": None,
        "physics_randomization": None,
        "camera_extrinsic_jitter": None,
    }


def _render_contract_inputs() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    horizontal = 78.323
    vertical = math.degrees(
        2.0 * math.atan(math.tan(math.radians(horizontal) * 0.5) * (168.0 / 224.0))
    )
    frames = "/tmp/v12-author-test-frames.jsonl"
    render_plan = {
        "frames_jsonl": frames,
        "camera": {
            "native_resolution": [224, 168],
            "training_resolution": [112, 84],
            "fov_axis": "horizontal",
            "fov_deg": horizontal,
            "near_m": 0.05,
            "far_m": 20.0,
            "encoding": "rgb8",
            "mount_body": {},
        },
    }
    empty_hash = source_v4.canonical_json_sha256([])
    summary = {
        "resolution_wh": [224, 168],
        "camera_projection": {
            "model": "pinhole",
            "renderer_fov_axis": "vertical",
            "horizontal_fov_deg": horizontal,
            "vertical_fov_deg": vertical,
            "near_m": 0.05,
            "far_m": 20.0,
            "runtime_rectification_required": False,
        },
        "object_parity": {
            "schema": "lewm_render_object_parity_v1",
            "rendered_groups": ["wall", "obstacle", "landmark", "distractor"],
            "rendered_object_count": 0,
            "rendered_object_ids": [],
            "rendered_object_ids_sha256": empty_hash,
            "rendered_object_records_sha256": empty_hash,
            "collision_distractors_rendered": True,
            "full_box_roll_pitch_yaw_rendered": True,
        },
    }
    source_record = {"frames": {"path": frames}}
    return render_plan, summary, source_record


def _patch_render_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auditor, "_require_exact_authority", lambda _digest: object())
    monkeypatch.setattr(
        auditor, "_install_reviewed_source_semantics", lambda _digest: None
    )
    monkeypatch.setattr(source_v4, "_rendered_boxes", lambda _manifest: ())
    monkeypatch.setattr(
        source_v4,
        "labels_v3",
        SimpleNamespace(_render_object_records=lambda _manifest: []),
    )


def _call_render_contract(
    raw: object,
    parsed: object,
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, ...]:
    _patch_render_dependencies(monkeypatch)
    render_plan, summary, source_record = _render_contract_inputs()
    return auditor._validate_sample_render_contract(
        render_plan,
        summary,
        raw,
        parsed,
        source_record,
        authorization_sha256="a" * 64,
    )


def _authorization_payload() -> dict[str, object]:
    source_map = []
    for role, path in auditor.SOURCE_ROLE_PATHS:
        digest = auditor.FROZEN_AUTHORITY_ROLE_SHA256.get(
            role, hashlib.sha256(role.encode("ascii")).hexdigest()
        )
        source_map.append({"role": role, "path": path, "sha256": digest})
    source_by_role = {str(row["role"]): row for row in source_map}
    candidate = [source_by_role[role] for role in auditor.AUDITOR_CANDIDATE_ROLES]
    review = source_by_role["auditor_review"]
    core: dict[str, object] = {
        "schema": auditor.AUTHORIZATION_SCHEMA,
        "exact_audit_v12_authorized": True,
        **{field: False for field in auditor.AUTHORIZATION_FALSE_FIELDS},
        "input_dataset_path": auditor.AUTHORIZED_DATASET_PATH,
        "success_report_path": auditor.AUTHORIZED_SUCCESS_REPORT_PATH,
        "failure_report_path": auditor.AUTHORIZED_FAILURE_REPORT_PATH,
        "v9_build_authorization_file_sha256": auditor.FROZEN_AUTHORITY_ROLE_SHA256[
            "v9_build_authorization"
        ],
        "v9_build_authorization_content_sha256": (
            auditor.FROZEN_V9_BUILD_AUTHORIZATION_CONTENT_SHA256
        ),
        "v9_build_authorization_source_map_sha256": (
            auditor.FROZEN_V9_BUILD_AUTHORIZATION_SOURCE_MAP_SHA256
        ),
        "v9_dataset_manifest_file_sha256": auditor.FROZEN_AUTHORITY_ROLE_SHA256[
            "v9_dataset_manifest"
        ],
        "v9_dataset_manifest_content_sha256": (
            auditor.FROZEN_V9_DATASET_MANIFEST_CONTENT_SHA256
        ),
        "v9_audit_failure_file_sha256": auditor.FROZEN_AUTHORITY_ROLE_SHA256[
            "v9_terminal_failure"
        ],
        "v9_audit_failure_content_sha256": (
            auditor.FROZEN_V9_AUDIT_FAILURE_CONTENT_SHA256
        ),
        "v10_audit_authorization_file_sha256": auditor.FROZEN_AUTHORITY_ROLE_SHA256[
            "v10_audit_authorization"
        ],
        "v10_audit_authorization_content_sha256": (
            auditor.FROZEN_V10_AUDIT_AUTHORIZATION_CONTENT_SHA256
        ),
        "v10_audit_authorization_source_map_sha256": (
            auditor.FROZEN_V10_AUDIT_AUTHORIZATION_SOURCE_MAP_SHA256
        ),
        "v10_audit_failure_file_sha256": auditor.FROZEN_AUTHORITY_ROLE_SHA256[
            "v10_terminal_failure"
        ],
        "v10_audit_failure_content_sha256": (
            auditor.FROZEN_V10_AUDIT_FAILURE_CONTENT_SHA256
        ),
        "v11_auditor_block_file_sha256": auditor.FROZEN_AUTHORITY_ROLE_SHA256[
            "v11_auditor_block"
        ],
        "v11_auditor_block_content_sha256": (
            auditor.FROZEN_V11_AUDITOR_BLOCK_CONTENT_SHA256
        ),
        "auditor_review": {
            "schema": auditor.REVIEW_BINDING_SCHEMA,
            "review_schema": auditor.AUDITOR_REVIEW_SCHEMA,
            "verdict": "PASS",
            "reviewer": "/root/raw_v12_independent_reviewer",
            "implementation_author": auditor.V12_IMPLEMENTATION_AUTHOR,
            "path": review["path"],
            "file_sha256": review["sha256"],
            "content_sha256": "b" * 64,
            "candidate": candidate,
        },
        "source_map": source_map,
    }
    return {**core, "content_sha256": auditor.canonical_json_sha256(core)}


def _v11_block_payload() -> dict[str, object]:
    core: dict[str, object] = {
        "authority": {
            "auditor_source_approved": False,
            "calibration_authorized": False,
            "dataset_use_authorized": False,
            "deployment_authorized": False,
            "exact_audit_v10_authorized": False,
            "exact_audit_v11_authorized": False,
            "exact_audit_v9_authorized": False,
            "exact_build_authorized": False,
            "exact_rebuild_authorized": False,
            "g2_authorized": False,
            "hardware_authorized": False,
            "heldout_authorized": False,
            "navigation_authorized": False,
            "production_authorized": False,
            "promotion_authorized": False,
            "retry_authorized": False,
            "rgb_decode_authorized": False,
            "runtime_authorized": False,
            "selection_authorized": False,
            "training_authorized": False,
        },
        "blocker": {
            "amendment_clause": "required_author_and_reviewer_proof_15",
            "consequence": (
                "a spawned audit worker can inherit a nonempty "
                "HSA_VISIBLE_DEVICES selector instead of independently hiding "
                "every accelerator"
            ),
            "function": "_set_worker_environment",
            "observed_environment_after_worker_setup": {
                "CUDA_VISIBLE_DEVICES": "",
                "GPU_DEVICE_ORDINAL": "",
                "HIP_VISIBLE_DEVICES": "",
                "HSA_VISIBLE_DEVICES": "0",
                "ROCR_VISIBLE_DEVICES": "",
            },
            "required_remediation": (
                "add HSA_VISIBLE_DEVICES to the closed accelerator environment "
                "and add a hostile-inheritance regression proving every "
                "initializer and task clears it"
            ),
            "source_line": 510,
        },
        "candidate": [
            {
                "path": auditor.SOURCE_PATH_BY_ROLE[f"v11_{role}"],
                "role": role,
                "sha256": auditor.FROZEN_AUTHORITY_ROLE_SHA256[f"v11_{role}"],
            }
            for role in auditor.AUDITOR_CANDIDATE_ROLES
        ],
        "implementation_author": auditor.V12_IMPLEMENTATION_AUTHOR,
        "passing_findings": {
            "builder_v9_context_ast_and_behavior_parity": True,
            "closed_v10_to_v11_ast_delta": True,
            "duplicate_absent_orphan_and_all_ten_conflicts_rejected": True,
            "eight_array_builder_oracle_bytes_dtype_shape_and_hash_parity": True,
            "endpoint_content_and_one_field_frame_key_provenance": True,
            "one_and_six_spawn_worker_science_bytes_identical": True,
            "original_decoded_raw_manifest_boundary_retained": True,
            "production_surface_standalone_and_audit_only": True,
            "real_synthetic_jsonl_geometry_raycast_raster_replay": True,
            "success_and_failure_publication_predecessor_preservation": True,
            "v9_and_v10_authority_and_terminal_closure": True,
        },
        "review_completed": True,
        "reviewer": auditor.V11_REVIEWER,
        "schema": (
            "lewm_go2_shared_jepa_v5_raw_supervision_auditor_v11_"
            "independent_review_block_v1"
        ),
        "verdict": "BLOCK",
        "verification": {
            "accelerators_visible_during_suites": False,
            "candidate_hashes_reverified": True,
            "canonical_dataset_or_generated_payload_opened": False,
            "compilation": "passed",
            "exact_audit_run": False,
            "focused_v11_suite": "46 passed in 1.13s",
            "hostile_hsa_inheritance_regression": "failed_as_expected_exit_1",
            "native_threads_per_process": 1,
            "retained_v9_v10_builder_auditor_suites": "181 passed in 2.15s",
            "total_passing_tests": 227,
        },
    }
    return {**core, "content_sha256": auditor.canonical_json_sha256(core)}


def _endpoint_record(
    *,
    scene_id: str = "synthetic_scene",
    episode_id: str = "episode-1",
    episode_step: int = 7,
    frame_index: int = 11,
    env_index: int = 2,
    timestamp_ns: int = 123456,
    image_sha256: str = "1" * 64,
    image_path: str = "/tmp/v12-synthetic/rgb/frame.png",
) -> dict[str, object]:
    identity = {
        "dataset_role": "train",
        "scene_id": scene_id,
        "episode_id": episode_id,
        "env_index": env_index,
        "episode_step": episode_step,
        "frame_index": frame_index,
        "timestamp_ns": timestamp_ns,
        "image_sha256": image_sha256,
    }
    core: dict[str, object] = {
        "schema": "lewm_go2_shared_jepa_v5_raw_supervision_endpoint_v1",
        "identity": identity,
        "identity_sha256": auditor.canonical_json_sha256(identity),
        "image_path_metadata_only": image_path,
        "frames_jsonl_sha256": "2" * 64,
        "scene_manifest_sha256": "3" * 64,
        "base_quat_world_xyzw": [0.0, 0.0, 0.0, 1.0],
        "stored_base_yaw_rad": 0.0,
    }
    return {**core, "content_sha256": auditor.canonical_json_sha256(core)}


def _pair_record(
    current: Mapping[str, object],
    next_endpoint: Mapping[str, object],
    *,
    global_row: int = 0,
    family: str = "open_obstacle_field",
    reset_count: int = 1,
) -> dict[str, object]:
    return {
        "global_row": global_row,
        "family": family,
        "reset_count": reset_count,
        "current_endpoint_sha256": current["identity_sha256"],
        "next_endpoint_sha256": next_endpoint["identity_sha256"],
        "sidecar_row_identity_sha256": "4" * 64,
        "label_shard_sha256": "5" * 64,
        "label_shard_row": global_row,
    }


def test_v12_contexts_are_functionally_identical_to_frozen_builder_v9() -> None:
    assert hashlib.sha256(Path(builder_oracle.__file__).read_bytes()).hexdigest() == (
        "2388c1138d9b03ea6e385cc0250c81a1869a40cab62507d02f709ef39197c664"
    )
    first = _endpoint_record()
    second = _endpoint_record(
        episode_step=8,
        frame_index=12,
        timestamp_ns=123457,
        image_sha256="6" * 64,
        image_path="/tmp/v12-synthetic/rgb/frame-next.png",
    )
    pairs = (
        _pair_record(first, second, global_row=0),
        _pair_record(first, second, global_row=1),
    )
    endpoints = (first, second)
    expected = builder_oracle._pair_endpoint_contexts(
        SimpleNamespace(endpoints=endpoints, pairs=pairs)
    )
    assert auditor._pair_endpoint_contexts(
        SimpleNamespace(endpoints=endpoints, pairs=pairs)
    ) == expected
    assert set(next(iter(expected.values()))) == {
        "scene_id",
        "family",
        "episode_id",
        "reset_count",
        "episode_step",
        "frame_index",
        "env_index",
        "timestamp_ns",
        "image_sha256",
        "image_path_metadata_only",
    }


def test_v12_context_constructor_is_mechanically_builder_v9_ast() -> None:
    observed = _function_ast(auditor, "_pair_endpoint_contexts").replace(
        "RawSupervisionAuditError", "RawSupervisionBuildError"
    )
    assert observed == _function_ast(builder_oracle, "_pair_endpoint_contexts")


@pytest.mark.parametrize(
    "field",
    [
        "scene_id",
        "family",
        "episode_id",
        "reset_count",
        "episode_step",
        "frame_index",
        "env_index",
        "timestamp_ns",
        "image_sha256",
        "image_path_metadata_only",
    ],
)
def test_v12_rejects_every_later_occurrence_context_conflict(field: str) -> None:
    first = _endpoint_record()
    second = _endpoint_record(
        episode_step=8,
        frame_index=12,
        timestamp_ns=123457,
        image_sha256="6" * 64,
        image_path="/tmp/v12-synthetic/rgb/frame-next.png",
    )
    identity = deepcopy(first["identity"])
    assert isinstance(identity, dict)

    class OccurrenceEndpoint(dict[str, object]):
        occurrence = 0

        def __getitem__(self, key: str) -> object:
            if key == "identity":
                value = deepcopy(identity)
                if self.occurrence > 0 and field in value:
                    value[field] = (
                        int(value[field]) + 1
                        if isinstance(value[field], int)
                        else "mutated"
                    )
                self.occurrence += 1
                return value
            if key == "image_path_metadata_only" and self.occurrence > 1 and field == key:
                return "/tmp/v12-synthetic/rgb/mutated.png"
            return super().__getitem__(key)

    changing = OccurrenceEndpoint(first)
    pair_one = _pair_record(changing, second, global_row=0)
    pair_two = _pair_record(changing, second, global_row=1)
    if field == "family":
        pair_two["family"] = "mutated_family"
    if field == "reset_count":
        pair_two["reset_count"] = 2
    with pytest.raises(auditor.RawSupervisionAuditError, match="conflicting"):
        auditor._pair_endpoint_contexts(
            SimpleNamespace(
                endpoints=(changing, second), pairs=(pair_one, pair_two)
            )
        )


def test_v12_context_rejects_duplicate_absent_and_orphan_endpoints() -> None:
    first = _endpoint_record()
    second = _endpoint_record(
        episode_step=8,
        frame_index=12,
        timestamp_ns=123457,
        image_sha256="6" * 64,
    )
    pair = _pair_record(first, second)
    with pytest.raises(auditor.RawSupervisionAuditError, match="repeats"):
        auditor._pair_endpoint_contexts(
            SimpleNamespace(endpoints=(first, first), pairs=(pair,))
        )
    absent = dict(pair)
    absent["next_endpoint_sha256"] = "f" * 64
    with pytest.raises(auditor.RawSupervisionAuditError, match="absent"):
        auditor._pair_endpoint_contexts(
            SimpleNamespace(endpoints=(first, second), pairs=(absent,))
        )
    third = _endpoint_record(
        episode_step=9,
        frame_index=13,
        timestamp_ns=123458,
        image_sha256="7" * 64,
    )
    with pytest.raises(auditor.RawSupervisionAuditError, match="orphan"):
        auditor._pair_endpoint_contexts(
            SimpleNamespace(endpoints=(first, second, third), pairs=(pair,))
        )


def test_v12_ignores_pair_occurrence_only_provenance() -> None:
    first = _endpoint_record()
    second = _endpoint_record(
        episode_step=8,
        frame_index=12,
        timestamp_ns=123457,
        image_sha256="6" * 64,
    )
    early = _pair_record(first, second, global_row=100)
    late = _pair_record(first, second, global_row=1)
    late["sidecar_row_identity_sha256"] = "9" * 64
    late["label_shard_sha256"] = "a" * 64
    contexts = auditor._pair_endpoint_contexts(
        SimpleNamespace(endpoints=(first, second), pairs=(early, late))
    )
    assert all(
        key not in contexts[str(first["identity_sha256"])]
        for key in (
            "global_row",
            "side",
            "sidecar_row_identity_sha256",
            "label_shard_sha256",
            "label_row",
        )
    )


def test_v12_frame_input_uses_only_builder_endpoint_provenance() -> None:
    endpoint = _endpoint_record()
    endpoint_digest = str(endpoint["identity_sha256"])
    pair_sidecar_hash = "e" * 64
    assert pair_sidecar_hash != endpoint["content_sha256"]
    camera = SimpleNamespace(origin_xyz=(0.326, 0.0, 0.043))
    basis = ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0))
    frame = auditor._builder_v9_frame_input(
        endpoint_digest=endpoint_digest,
        endpoint=endpoint,
        camera=camera,
        camera_basis_body_fru=basis,
        base_position=(0.0, 0.0, 0.4),
        rendered_boxes=(),
    )
    assert frame.frame_key == {"endpoint_identity_sha256": endpoint_digest}
    assert frame.sidecar_row_identity_sha256 == endpoint["content_sha256"]
    assert frame.sidecar_row_identity_sha256 != pair_sidecar_hash
    assert frame.image_sha256 == endpoint["identity"]["image_sha256"]


def _payload(path: Path) -> tuple[bytes, str]:
    raw = path.read_bytes()
    return raw, hashlib.sha256(raw).hexdigest()


def _safe_synthetic_read(
    path: Path,
    expected_sha256: str,
    *,
    repository_root: Path,
    name: str,
) -> bytes:
    del repository_root, name
    candidate = Path(path)
    if ".generated" in candidate.parts:
        raise AssertionError("synthetic proof attempted canonical generated data")
    raw = candidate.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise AssertionError("synthetic proof hash changed")
    return raw


def _synthetic_authority(_digest: str) -> SimpleNamespace:
    return SimpleNamespace(authorized=True)


def _spawn_authority(_digest: str) -> SimpleNamespace:
    assert tuple(
        os.environ.get(name) for name in auditor.THREAD_ENVIRONMENT
    ) == ("1",) * len(auditor.THREAD_ENVIRONMENT)
    assert tuple(
        os.environ.get(name) for name in auditor.ACCELERATOR_ENVIRONMENT
    ) == ("",) * len(auditor.ACCELERATOR_ENVIRONMENT)
    return SimpleNamespace(authorized=True)


def _spawn_synthetic_initializer() -> None:
    auditor._require_exact_authority = _spawn_authority  # type: ignore[assignment]
    auditor._read_absolute_bound_payload = _safe_synthetic_read  # type: ignore[assignment]
    auditor._initialize_exact_worker("a" * 64)


def _spawn_synthetic_task(
    task: tuple[str, str, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]],
) -> tuple[str, tuple[np.ndarray, ...], tuple[str | None, ...], bytes]:
    digest, arrays = auditor._recompute_exact_sample_task(task)
    environment = tuple(
        os.environ.get(name)
        for name in (*auditor.THREAD_ENVIRONMENT, *auditor.ACCELERATOR_ENVIRONMENT)
    )
    science = auditor.canonical_json_bytes(
        {
            "endpoint_identity_sha256": digest,
            "array_byte_sha256": [
                hashlib.sha256(array.tobytes(order="C")).hexdigest()
                for array in arrays
            ],
        }
    )
    return digest, arrays, environment, science


def _spawn_task_order_probe(task_name: str) -> tuple[str | None, ...]:
    for name in auditor.ACCELERATOR_ENVIRONMENT:
        os.environ[name] = f"hostile-child-{name}"

    def forbidden_open(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("production task opened payload before rejecting authority")

    def rejecting_authority(_digest: str) -> object:
        assert tuple(
            os.environ.get(name) for name in auditor.ACCELERATOR_ENVIRONMENT
        ) == ("",) * len(auditor.ACCELERATOR_ENVIRONMENT)
        raise PermissionError("task-order-probe")

    auditor._require_exact_authority = rejecting_authority  # type: ignore[assignment]
    auditor._read_absolute_bound_payload = forbidden_open  # type: ignore[assignment]
    auditor._validate_shards = forbidden_open  # type: ignore[assignment]
    auditor._recompute_one_exact_sample = forbidden_open  # type: ignore[assignment]
    tasks: dict[str, object] = {
        "_validate_one_shard_task": ("a" * 64,),
        "_hash_source_file": ("a" * 64, {}),
        "_recompute_exact_sample_task": ("a" * 64,),
    }
    functions = {
        "_validate_one_shard_task": auditor._validate_one_shard_task,
        "_hash_source_file": auditor._hash_source_file,
        "_recompute_exact_sample_task": auditor._recompute_exact_sample_task,
    }
    try:
        functions[task_name](tasks[task_name])  # type: ignore[arg-type]
    except PermissionError as error:
        assert str(error) == "task-order-probe"
    else:
        raise AssertionError("production task did not independently authorize")
    return tuple(
        os.environ.get(name) for name in auditor.ACCELERATOR_ENVIRONMENT
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> str:
    raw = auditor.canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _synthetic_replay_fixture(
    root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    scene_id = "synthetic_scene"
    scene_root = root / "synthetic_scene"
    frames_path = scene_root / "frames.jsonl"
    manifest_path = scene_root / "scene_manifest.json"
    plan_path = scene_root / "render_plan.json"
    summary_path = scene_root / "summary.json"
    image_path = scene_root / "rgb" / "frame_000011_env_02.png"
    raw_manifest = _raw_manifest()
    parsed = manifest_semantics.parse_scene_manifest_dict(raw_manifest)
    manifest_sha = _write_json(manifest_path, raw_manifest)

    frame = {
        "frame_index": 11,
        "env_index": 2,
        "timestamp_ns": 123456,
        "episode": {
            "episode_id": "episode-1",
            "reset_count": 1,
            "episode_step": 7,
        },
        "base_pose_world": {"position": {"x": 0.0, "y": 0.0, "z": 0.4}},
        "base_rpy_rad": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        "base_quat_world_xyzw": [0.0, 0.0, 0.0, 1.0],
        "camera_mount_body": deepcopy(source_v4.NOMINAL_CAMERA_MOUNT_BODY),
        "camera_pose_world": {
            "position": [0.326, 0.0, 0.443],
            "lookat": [1.326, 0.0, 0.443],
            "up": [0.0, 0.0, 1.0],
        },
    }
    frames_path.parent.mkdir(parents=True, exist_ok=True)
    frames_path.write_bytes(auditor.canonical_json_bytes(frame) + b"\n")
    frames_raw, frames_sha = _payload(frames_path)
    image_sha = "1" * 64
    horizontal = 78.323
    vertical = math.degrees(
        2.0
        * math.atan(math.tan(math.radians(horizontal) * 0.5) * (168.0 / 224.0))
    )
    render_plan = {
        "schema": "lewm_render_replay_plan_v0",
        "scene_id": scene_id,
        "frames_jsonl": str(frames_path),
        "camera": {
            "native_resolution": [224, 168],
            "training_resolution": [112, 84],
            "fov_axis": "horizontal",
            "fov_deg": horizontal,
            "near_m": 0.05,
            "far_m": 20.0,
            "encoding": "rgb8",
            "mount_body": deepcopy(source_v4.NOMINAL_CAMERA_MOUNT_BODY),
        },
    }
    plan_sha = _write_json(plan_path, render_plan)
    if not source_v4._SEMANTICS_LOADED:
        from lewm.benchmarks import go2_n32_camera_frustum_observability as core
        from lewm.datasets import go2_paired_navigation
        from lewm_worlds import planning_grid

        source_v4._install_semantic_modules(
            core,
            go2_paired_navigation,
            manifest_semantics,
            planning_grid,
        )
    object_records = source_v4.labels_v3._render_object_records(parsed)
    object_ids = sorted(str(item["object_id"]) for item in object_records)
    renderer_source = scene_root / "renderer.py"
    renderer_source.write_bytes(b"# synthetic metadata-only renderer\n")
    _, renderer_sha = _payload(renderer_source)
    summary = {
        "schema": "lewm_rendered_vision_v04",
        "scene_id": scene_id,
        "family": "open_obstacle_field",
        "render_status": "complete",
        "resolution_wh": [224, 168],
        "g2_model_outputs_opened": False,
        "source": {
            "plan": {"path": str(plan_path), "sha256": plan_sha},
            "frames_jsonl": {"path": str(frames_path), "sha256": frames_sha},
            "scene_manifest": {"path": str(manifest_path), "sha256": manifest_sha},
            "renderer_source": {
                "path": str(renderer_source),
                "sha256": renderer_sha,
            },
        },
        "rendered_frames": [
            {
                "frame_index": 11,
                "env_index": 2,
                "timestamp_ns": 123456,
                "image_sha256": image_sha,
            }
        ],
        "camera_projection": {
            "model": "pinhole",
            "renderer_fov_axis": "vertical",
            "horizontal_fov_deg": horizontal,
            "vertical_fov_deg": vertical,
            "near_m": 0.05,
            "far_m": 20.0,
            "runtime_rectification_required": False,
        },
        "object_parity": {
            "schema": "lewm_render_object_parity_v1",
            "rendered_groups": ["wall", "obstacle", "landmark", "distractor"],
            "rendered_object_count": len(object_records),
            "rendered_object_ids": object_ids,
            "rendered_object_ids_sha256": source_v4.canonical_json_sha256(object_ids),
            "rendered_object_records_sha256": source_v4.canonical_json_sha256(
                object_records
            ),
            "collision_distractors_rendered": True,
            "full_box_roll_pitch_yaw_rendered": True,
        },
    }
    summary_sha = _write_json(summary_path, summary)
    endpoint = _endpoint_record(image_sha256=image_sha, image_path=str(image_path))
    endpoint["frames_jsonl_sha256"] = frames_sha
    endpoint["scene_manifest_sha256"] = manifest_semantics.manifest_sha256(parsed)
    endpoint_core = dict(endpoint)
    endpoint_core.pop("content_sha256")
    endpoint["content_sha256"] = auditor.canonical_json_sha256(endpoint_core)
    pair = _pair_record(endpoint, endpoint)
    context = auditor._pair_endpoint_contexts(
        SimpleNamespace(endpoints=(endpoint,), pairs=(pair,))
    )[
        str(endpoint["identity_sha256"])
    ]
    source_record = {
        "scene_id": scene_id,
        "role": "train",
        "family": "open_obstacle_field",
        "frames": {
            "path": str(frames_path),
            "sha256": frames_sha,
            "jsonl_record_count": 1,
        },
        "scene_manifest": {
            "path": str(manifest_path),
            "file_sha256": manifest_sha,
            "content_sha256": manifest_semantics.manifest_sha256(parsed),
        },
        "render_plan": {"path": str(plan_path), "sha256": plan_sha},
        "render_summary": {"path": str(summary_path), "sha256": summary_sha},
    }
    assert frames_raw.endswith(b"\n")
    return endpoint, pair, context, source_record


def _evidence_and_raster_hashes(arrays: tuple[np.ndarray, ...]) -> tuple[str, str]:
    evidence = auditor.ObservableCameraRayEvidenceV4(
        camera_origin_body_m=arrays[0],
        camera_basis_body_fru=arrays[1],
        ground_plane_z_body_m=float(arrays[2]),
        ground_support_in_frustum=arrays[3].astype(bool),
        ground_support_clear_to_target=arrays[4].astype(bool),
        pixel_hit_mask=arrays[5].astype(bool),
        pixel_first_hit_distance_m=arrays[6],
    )
    raster = auditor.rasterize_observable_camera_ray_evidence_v4(evidence)
    assert np.array_equal(raster.output_labels, arrays[7])
    return evidence.content_sha256(), raster.content_sha256()


def test_v12_real_synthetic_replay_matches_frozen_builder_v9(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    endpoint, pair, context, source_record = _synthetic_replay_fixture(tmp_path)
    digest = str(endpoint["identity_sha256"])
    monkeypatch.setattr(auditor, "_require_exact_authority", _synthetic_authority)
    monkeypatch.setattr(
        auditor, "_read_absolute_bound_payload", _safe_synthetic_read
    )
    replay = auditor._recompute_one_exact_sample(
        endpoint_digest=digest,
        endpoint=endpoint,
        endpoint_context=context,
        source_record=source_record,
        authorization_sha256="a" * 64,
    )
    monkeypatch.setattr(builder_oracle, "_require_exact_authority", _synthetic_authority)

    def builder_read(
        *, path: Path, expected_sha256: str, authorization_sha256: str
    ) -> bytes:
        del authorization_sha256
        return _safe_synthetic_read(
            path,
            expected_sha256,
            repository_root=tmp_path,
            name="builder oracle",
        )

    monkeypatch.setattr(builder_oracle, "_read_exact_source", builder_read)
    prepared = builder_oracle._load_exact_scene_job(
        source_record,
        (endpoint,),
        builder_oracle._pair_endpoint_contexts(
            SimpleNamespace(endpoints=(endpoint,), pairs=(pair,))
        ),
        "a" * 64,
    )["job"].endpoints[0]
    expected_evidence = builder_oracle.build_frame_evidence_v4(prepared.frame)
    expected_raster = builder_oracle.rasterize_observable_camera_ray_evidence_v4(
        expected_evidence
    )
    expected = builder_oracle._endpoint_arrays(expected_evidence, expected_raster)
    assert len(replay) == len(expected) == len(auditor.ARRAY_LAYOUT) == 8
    for (name, dtype, shape), actual, wanted in zip(
        auditor.ARRAY_LAYOUT, replay, expected
    ):
        assert actual.dtype == np.dtype(dtype), name
        assert actual.shape == shape, name
        assert actual.tobytes(order="C") == wanted.tobytes(order="C"), name
    assert _evidence_and_raster_hashes(replay) == (
        expected_evidence.content_sha256(),
        expected_raster.content_sha256(),
    )


def test_v10_real_dataflow_reproduces_missing_key_before_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    endpoint, pair, _context, source_record = _synthetic_replay_fixture(tmp_path)
    digest = str(endpoint["identity_sha256"])
    predecessor_context = predecessor_v10._source_record_for_endpoint(
        digest, endpoint, (pair,)
    )
    assert "sidecar_row_identity_sha256" not in predecessor_context
    monkeypatch.setattr(
        predecessor_v10, "_require_exact_authority", _synthetic_authority
    )
    monkeypatch.setattr(
        predecessor_v10, "_read_absolute_bound_payload", _safe_synthetic_read
    )
    with pytest.raises(KeyError, match="sidecar_row_identity_sha256"):
        predecessor_v10._recompute_one_exact_sample(
            endpoint_digest=digest,
            endpoint=endpoint,
            pair_record=predecessor_context,
            source_record=source_record,
            authorization_sha256="a" * 64,
        )


def test_v12_real_synthetic_replay_is_identical_with_one_and_six_spawn_workers(
    tmp_path: Path,
) -> None:
    endpoint, _pair, context, source_record = _synthetic_replay_fixture(tmp_path)
    digest = str(endpoint["identity_sha256"])
    task = ("a" * 64, digest, endpoint, context, source_record)
    outputs: dict[
        int,
        list[tuple[str, tuple[np.ndarray, ...], tuple[str | None, ...], bytes]],
    ] = {}
    spawn = multiprocessing.get_context("spawn")
    old = {
        name: os.environ.get(name) for name in auditor.ACCELERATOR_ENVIRONMENT
    }
    try:
        for name in auditor.ACCELERATOR_ENVIRONMENT:
            os.environ[name] = f"hostile-parent-{name}"
        for workers in (1, 6):
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=spawn,
                initializer=_spawn_synthetic_initializer,
            ) as executor:
                outputs[workers] = list(
                    executor.map(_spawn_synthetic_task, [task] * workers)
                )
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    reference = outputs[1][0][1]
    reference_science = outputs[1][0][3]
    for records in outputs.values():
        for observed_digest, arrays, environment, science in records:
            assert observed_digest == digest
            assert science == reference_science
            assert all(
                actual.tobytes(order="C") == wanted.tobytes(order="C")
                for actual, wanted in zip(arrays, reference)
            )
            assert environment[: len(auditor.THREAD_ENVIRONMENT)] == ("1",) * len(
                auditor.THREAD_ENVIRONMENT
            )
            assert environment[len(auditor.THREAD_ENVIRONMENT) :] == ("",) * len(
                auditor.ACCELERATOR_ENVIRONMENT
            )


def test_v9_failure_reproduces_with_an_actual_parsed_manifest() -> None:
    raw = _raw_manifest()
    parsed = manifest_semantics.parse_scene_manifest_dict(raw)
    assert isinstance(parsed.walls, tuple)
    source_v4._validate_raw_scene_object_records(raw)
    with pytest.raises(ValueError, match="wall boxes are not a list"):
        source_v4._validate_raw_scene_object_records(parsed.to_dict())


def test_v12_passes_the_original_decoded_mapping_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _raw_manifest()
    parsed = manifest_semantics.parse_scene_manifest_dict(raw)
    before = auditor.canonical_json_sha256(raw)
    original = source_v4._validate_raw_scene_object_records
    seen: list[object] = []

    def spy(value: object) -> None:
        seen.append(value)
        original(value)  # type: ignore[arg-type]

    monkeypatch.setattr(source_v4, "_validate_raw_scene_object_records", spy)
    assert _call_render_contract(raw, parsed, monkeypatch=monkeypatch) == ()
    assert seen == [raw]
    assert seen[0] is raw
    assert auditor.canonical_json_sha256(raw) == before


@pytest.mark.parametrize(
    "raw_factory",
    [
        lambda raw, _parsed: tuple(raw.items()),
        lambda raw, _parsed: MappingProxyType(raw),
        lambda raw, _parsed: (item for item in raw.items()),
    ],
)
def test_v12_rejects_nondecoded_top_level_representations(
    raw_factory: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _raw_manifest()
    parsed = manifest_semantics.parse_scene_manifest_dict(raw)
    value = raw_factory(raw, parsed)  # type: ignore[operator]
    with pytest.raises(auditor.RawSupervisionAuditError, match="decoded JSON object"):
        _call_render_contract(value, parsed, monkeypatch=monkeypatch)


@pytest.mark.parametrize("kind", ["tuple", "generator", "reconstructed"])
def test_v12_rejects_normalized_or_reconstructed_sequences(
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _raw_manifest()
    parsed = manifest_semantics.parse_scene_manifest_dict(raw)
    if kind == "reconstructed":
        invalid = parsed.to_dict()
    else:
        invalid = deepcopy(raw)
        walls = invalid["walls"]
        assert isinstance(walls, list)
        invalid["walls"] = tuple(walls) if kind == "tuple" else (item for item in walls)
    with pytest.raises((ValueError, auditor.RawSupervisionAuditError)):
        _call_render_contract(invalid, parsed, monkeypatch=monkeypatch)


def test_v12_rejects_mutation_during_raw_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _raw_manifest()
    parsed = manifest_semantics.parse_scene_manifest_dict(raw)
    original = source_v4._validate_raw_scene_object_records

    def mutate(value: dict[str, object]) -> None:
        original(value)
        value["unexpected_mutation"] = True

    monkeypatch.setattr(source_v4, "_validate_raw_scene_object_records", mutate)
    with pytest.raises(auditor.RawSupervisionAuditError, match="changed during raw"):
        _call_render_contract(raw, parsed, monkeypatch=monkeypatch)


def test_parsed_semantic_hash_and_rendered_geometry_are_v9_identical() -> None:
    raw = _raw_manifest()
    before = auditor.canonical_json_sha256(raw)
    first = manifest_semantics.parse_scene_manifest_dict(raw)
    second = manifest_semantics.parse_scene_manifest_dict(deepcopy(raw))
    assert manifest_semantics.manifest_sha256(first) == manifest_semantics.manifest_sha256(
        second
    )
    assert source_v4._rendered_boxes(first) == source_v4._rendered_boxes(second)
    assert auditor.canonical_json_sha256(raw) == before


def _function_ast(module: object, name: str) -> str:
    tree = ast.parse(Path(str(module.__file__)).read_text(encoding="utf-8"))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    return ast.dump(node, include_attributes=False)


def test_v12_unchanged_science_functions_are_exactly_v11_ast() -> None:
    names = {
        "_validate_integer_fields",
        "_validate_access_ledger_integers",
        "_parse_manifest",
        "_validate_pair_and_endpoint_indexes",
        "_derive_population",
        "_validate_frozen_population",
        "_sample_records",
        "_validate_sample_precommit",
        "_stored_arrays_from_evidence",
        "_validate_shards",
        "_validate_shards_parallel",
        "_compare_source_replay",
        "_validate_sample_render_contract",
        "_find_source_frame",
    }
    for name in names:
        observed = _function_ast(auditor, name).replace("V12", "V11").replace(
            "v12", "v11"
        )
        assert observed == _function_ast(predecessor, name)


def test_v12_unmodified_replay_helpers_are_exactly_v11_ast() -> None:
    names = {
        "_read_exact_source_json",
        "_summary_source_entry",
        "_source_path",
        "_install_reviewed_source_semantics",
        "_find_source_frame",
    }

    def definitions(module: object) -> dict[str, str]:
        source = Path(str(module.__file__)).read_text(encoding="utf-8")
        tree = ast.parse(source)
        return {
            node.name: ast.dump(node, include_attributes=False)
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        }

    expected = definitions(predecessor)
    observed = definitions(auditor)
    assert set(expected) == set(observed) == names
    for name in names:
        normalized = observed[name].replace("V12", "V11").replace("v12", "v11")
        assert normalized == expected[name]


def test_v12_top_level_ast_delta_from_v11_is_closed() -> None:
    def definitions(module: object) -> dict[str, str]:
        source = Path(str(module.__file__)).read_text(encoding="utf-8")
        tree = ast.parse(source)
        return {
            node.name: ast.dump(node, include_attributes=False)
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }

    expected = definitions(predecessor)
    observed = definitions(auditor)
    assert set(expected) - set(observed) == {
        "AcceptedAuthorizationV11",
        "PhaseOneAuthorizationV11",
        "ReviewBindingV11",
        "SourceBindingV11",
        "_AuditPublicationContextV11",
        "_PreparedAuditV11",
        "_final_revalidate_authorized_audit_v11",
        "_prepare_authorized_audit_v11",
        "_require_v10_success_absent",
        "execute_exact_audit_v11",
    }
    assert set(observed) - set(expected) == {
        "AcceptedAuthorizationV12",
        "PhaseOneAuthorizationV12",
        "ReviewBindingV12",
        "SourceBindingV12",
        "_AuditPublicationContextV12",
        "_PreparedAuditV12",
        "_final_revalidate_authorized_audit_v12",
        "_prepare_authorized_audit_v12",
        "_require_predecessor_success_absent",
        "_validate_frozen_v11_block_artifacts",
        "execute_exact_audit_v12",
    }
    changed = {
        name
        for name in set(expected) & set(observed)
        if observed[name].replace("V12", "V11").replace("v12", "v11")
        != expected[name]
    }
    assert changed == {
        "_ClosedAuditPublicationTransaction",
        "_FailureReceiptPublisher",
        "_publish_terminal_audit_failure",
        "_review_binding",
        "_validate_authorization_phase_one",
        "_validate_authorization_phase_two",
        "_validate_frozen_v10_authority_artifacts",
    }


def test_v12_transaction_ast_delta_from_v11_is_only_success_protection() -> None:
    def methods(module: object) -> dict[str, str]:
        source = Path(str(module.__file__)).read_text(encoding="utf-8")
        tree = ast.parse(source)
        transaction = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "_ClosedAuditPublicationTransaction"
        )
        return {
            node.name: ast.dump(node, include_attributes=False)
            for node in transaction.body
            if isinstance(node, ast.FunctionDef)
        }

    expected = methods(predecessor)
    observed = methods(auditor)
    assert set(expected) == set(observed)
    changed = set()
    for name, value in observed.items():
        normalized = value.replace("V12", "V11").replace("v12", "v11")
        if normalized != expected[name]:
            changed.add(name)
    assert changed == {"_strict_events", "_validate_bound_inventory"}


def test_v12_retains_v11_raw_manifest_boundary_exactly() -> None:
    observed = _function_ast(auditor, "_validate_sample_render_contract")
    normalized = observed.replace("V12", "V11").replace("v12", "v11")
    assert normalized == _function_ast(predecessor, "_validate_sample_render_contract")


def test_v12_boundary_source_has_only_the_permitted_raw_delta() -> None:
    source = Path(auditor.__file__).read_text(encoding="utf-8")
    render_source = inspect.getsource(auditor._validate_sample_render_contract)
    replay_source = inspect.getsource(auditor._recompute_one_exact_sample)
    assert "scene_manifest.to_dict()" not in source
    assert "_validate_raw_scene_object_records(raw_scene_manifest)" in render_source
    assert "type(raw_scene_manifest) is not dict" in render_source
    assert "parse_scene_manifest_dict(raw_scene_manifest)" in replay_source
    assert replay_source.count("canonical_json_sha256(raw_scene_manifest)") == 3
    assert "manifest_sha256(scene_manifest)" in replay_source


def test_v12_phase_one_accepts_exact_closure_without_opening_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = False

    def forbidden(*_args: object, **_kwargs: object) -> bytes:
        nonlocal opened
        opened = True
        raise AssertionError("phase one opened a mapped target")

    monkeypatch.setattr(auditor, "_read_absolute_bound_payload", forbidden)
    payload = _authorization_payload()
    phase_one = auditor._validate_authorization_phase_one(
        payload, authorization_file_sha256="c" * 64
    )
    assert opened is False
    assert len(phase_one.sources) == 25
    assert phase_one.auditor_review.implementation_author == auditor.V12_IMPLEMENTATION_AUTHOR


def test_v12_source_map_is_the_exact_ordered_25_role_closure() -> None:
    assert len(auditor.SOURCE_ROLE_PATHS) == 25
    assert auditor.SOURCE_ROLES == (
        "amendment",
        "v9_build_authorization",
        "v9_builder_source",
        "v9_builder_review",
        "v9_dataset_manifest",
        "v9_terminal_failure",
        "v10_amendment",
        "v10_auditor_source",
        "v10_auditor_cli",
        "v10_auditor_test",
        "v10_auditor_handoff",
        "v10_auditor_review",
        "v10_audit_authorization",
        "v10_terminal_failure",
        "v11_amendment",
        "v11_auditor_source",
        "v11_auditor_cli",
        "v11_auditor_test",
        "v11_auditor_handoff",
        "v11_auditor_block",
        "auditor_source",
        "auditor_cli",
        "auditor_test",
        "auditor_handoff",
        "auditor_review",
    )
    assert set(auditor.FROZEN_AUTHORITY_ROLE_SHA256) == set(
        auditor.SOURCE_ROLES[:20]
    )
    assert auditor.FROZEN_AUTHORITY_ROLE_SHA256["amendment"] == (
        "f4892405cf0fd97f9096f99d840b5590810fd8640822ed2e8c4c254c0c3e6adf"
    )


def test_v12_deep_validates_the_exact_terminal_v11_block(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for version in (9, 10, 11):
        monkeypatch.setattr(
            auditor,
            f"CANONICAL_V{version}_AUDIT_REPORT",
            tmp_path / f"absent-v{version}.json",
        )
    block = _v11_block_payload()
    assert block["content_sha256"] == (
        auditor.FROZEN_V11_AUDITOR_BLOCK_CONTENT_SHA256
    )
    raw = auditor.canonical_json_bytes(block) + b"\n"
    auditor._validate_frozen_v11_block_artifacts({"v11_auditor_block": raw})
    changed = deepcopy(block)
    changed["verdict"] = "PASS"
    with pytest.raises(PermissionError, match="V11 auditor BLOCK"):
        auditor._validate_frozen_v11_block_artifacts(
            {"v11_auditor_block": auditor.canonical_json_bytes(changed) + b"\n"}
        )


@pytest.mark.parametrize(
    "mutation",
    ("incomplete", "repeated", "reordered", "noncanonical", "changed"),
)
def test_v12_phase_one_structural_drift_fails_with_zero_target_opens(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    payload = _authorization_payload()
    source_map = payload["source_map"]
    assert isinstance(source_map, list)
    if mutation == "incomplete":
        source_map.pop()
    elif mutation == "repeated":
        source_map[1] = deepcopy(source_map[0])
    elif mutation == "reordered":
        source_map[0], source_map[1] = source_map[1], source_map[0]
    elif mutation == "noncanonical":
        source_map[0]["path"] = "./" + source_map[0]["path"]
    elif mutation == "changed":
        source_map[0]["sha256"] = "0" * 64
    core = dict(payload)
    core.pop("content_sha256")
    payload["content_sha256"] = auditor.canonical_json_sha256(core)
    opened = False

    def forbidden(*_args: object, **_kwargs: object) -> bytes:
        nonlocal opened
        opened = True
        raise AssertionError("phase one opened a mapped target")

    monkeypatch.setattr(auditor, "_read_absolute_bound_payload", forbidden)
    with pytest.raises((auditor.RawSupervisionAuditError, PermissionError)):
        auditor._validate_authorization_phase_one(
            payload, authorization_file_sha256="c" * 64
        )
    assert opened is False


def test_v12_phase_one_rejects_incomplete_and_noncanonical_authorization_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _authorization_payload()
    payload.pop("auditor_review")
    core = dict(payload)
    core.pop("content_sha256")
    payload["content_sha256"] = auditor.canonical_json_sha256(core)
    monkeypatch.setattr(
        auditor,
        "_read_absolute_bound_payload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("mapped target opened")
        ),
    )
    with pytest.raises(auditor.RawSupervisionAuditError):
        auditor._validate_authorization_phase_one(
            payload, authorization_file_sha256="c" * 64
        )
    exact = _authorization_payload()
    pretty = json.dumps(exact, indent=2).encode("utf-8") + b"\n"
    with pytest.raises(auditor.RawSupervisionAuditError, match="not canonical"):
        auditor._strict_canonical_json_object(pretty, name="synthetic authorization")


@pytest.mark.parametrize("alias_kind", ("symlink", "hardlink"))
def test_v12_phase_two_rejects_authority_alias_before_dataset_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    alias_kind: str,
) -> None:
    root = (tmp_path / "synthetic-repository").absolute()
    source_rows: list[dict[str, str]] = []
    frozen: dict[str, str] = {}
    for role, relative in auditor.SOURCE_ROLE_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = f"synthetic authority {role}\n".encode("ascii")
        path.write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        source_rows.append({"role": role, "path": relative, "sha256": digest})
        if role in auditor.SOURCE_ROLES[:20]:
            frozen[role] = digest
    source_by_role = {row["role"]: row for row in source_rows}
    review = source_by_role["auditor_review"]
    payload = _authorization_payload()
    payload["source_map"] = source_rows
    payload["auditor_review"] = {
        **payload["auditor_review"],
        "path": review["path"],
        "file_sha256": review["sha256"],
        "candidate": [
            source_by_role[role] for role in auditor.AUDITOR_CANDIDATE_ROLES
        ],
    }
    for field, role in (
        ("v9_build_authorization_file_sha256", "v9_build_authorization"),
        ("v9_dataset_manifest_file_sha256", "v9_dataset_manifest"),
        ("v9_audit_failure_file_sha256", "v9_terminal_failure"),
        ("v10_audit_authorization_file_sha256", "v10_audit_authorization"),
        ("v10_audit_failure_file_sha256", "v10_terminal_failure"),
        ("v11_auditor_block_file_sha256", "v11_auditor_block"),
    ):
        payload[field] = frozen[role]
    core = dict(payload)
    core.pop("content_sha256")
    payload["content_sha256"] = auditor.canonical_json_sha256(core)
    monkeypatch.setattr(auditor, "ROOT", root)
    monkeypatch.setattr(auditor, "FROZEN_AUTHORITY_ROLE_SHA256", frozen)
    monkeypatch.setattr(auditor, "FROZEN_V9_PREDECESSOR_SHA256", {})
    monkeypatch.setattr(auditor, "REVIEWED_V4_SOURCE_SHA256", {})
    for version in (9, 10, 11):
        monkeypatch.setattr(
            auditor,
            f"CANONICAL_V{version}_AUDIT_REPORT",
            root / f"absent-v{version}.json",
        )
    reached_deep_validation = False

    def forbidden_deep_validation(*_args: object, **_kwargs: object) -> None:
        nonlocal reached_deep_validation
        reached_deep_validation = True

    monkeypatch.setattr(auditor, "_validate_review_record", forbidden_deep_validation)
    monkeypatch.setattr(
        auditor, "_validate_frozen_v9_authority_artifacts", forbidden_deep_validation
    )
    monkeypatch.setattr(
        auditor, "_validate_frozen_v10_authority_artifacts", forbidden_deep_validation
    )
    monkeypatch.setattr(
        auditor, "_validate_frozen_v11_block_artifacts", forbidden_deep_validation
    )
    phase_one = auditor._validate_authorization_phase_one(
        payload, authorization_file_sha256="c" * 64
    )
    first = root / auditor.SOURCE_ROLE_PATHS[0][1]
    if alias_kind == "symlink":
        backing = first.with_name("amendment-backing.md")
        first.rename(backing)
        first.symlink_to(backing.name)
    else:
        os.link(first, first.with_name("amendment-hardlink.md"))
    with pytest.raises(PermissionError, match="unaliased regular file"):
        auditor._validate_authorization_phase_two(phase_one)
    assert reached_deep_validation is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("exact_build_authorized", True),
        ("retry_authorized", True),
        ("dataset_use_authorized", True),
        ("input_dataset_path", ".generated/other"),
        ("v9_audit_failure_content_sha256", "0" * 64),
    ],
)
def test_v12_phase_one_rejects_authority_or_terminal_binding_drift(
    field: str,
    value: object,
) -> None:
    payload = _authorization_payload()
    payload[field] = value
    core = dict(payload)
    core.pop("content_sha256")
    payload["content_sha256"] = auditor.canonical_json_sha256(core)
    with pytest.raises(PermissionError):
        auditor._validate_authorization_phase_one(
            payload, authorization_file_sha256="c" * 64
        )


def test_v12_one_and_six_worker_science_bytes_are_identical(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    digest = "d" * 64
    arrays = tuple(
        np.zeros(shape, dtype=np.dtype(dtype))
        for _name, dtype, shape in auditor.ARRAY_LAYOUT
    )
    observed = auditor.StoredEndpointEvidence(
        endpoint_identity_sha256=digest,
        arrays=arrays,
        evidence_content_sha256="e" * 64,
        raster_content_sha256="f" * 64,
    )
    manifest = {"content_sha256": "c" * 64, "shards": [{"path": "one"}]}
    sample = [{"endpoint_identity_sha256": digest}]
    population = {"synthetic_population": 1}
    inputs = auditor.AuditInputs(
        plan=SimpleNamespace(pairs=(object(),)), inventory=SimpleNamespace()
    )
    worker_calls: list[int] = []
    monkeypatch.setattr(auditor, "_require_exact_authority", lambda _digest: object())
    monkeypatch.setattr(
        auditor, "_require_real_directory", lambda *_args, **_kwargs: tmp_path
    )
    monkeypatch.setattr(auditor, "_parse_manifest", lambda *_args, **_kwargs: manifest)
    monkeypatch.setattr(
        auditor, "_validate_root_file_inventory", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        auditor,
        "_validate_pair_and_endpoint_indexes",
        lambda *_args, **_kwargs: ((), {digest: {}}, {}),
    )
    monkeypatch.setattr(auditor, "_derive_population", lambda *_args: population)
    monkeypatch.setattr(auditor, "_validate_frozen_population", lambda *_args: None)
    monkeypatch.setattr(auditor, "_validate_sample_precommit", lambda *_args: sample)
    monkeypatch.setattr(
        auditor,
        "_validate_shards_parallel",
        lambda *_args, **_kwargs: {digest: observed},
    )

    def recompute(
        _sample: object,
        _endpoints: object,
        _published_endpoints: object,
        _inputs: object,
        workers: int,
        **_kwargs: object,
    ) -> dict[str, tuple[np.ndarray, ...]]:
        worker_calls.append(workers)
        return {digest: arrays}

    monkeypatch.setattr(auditor, "_exact_sample_recomputer", recompute)
    monkeypatch.setattr(
        auditor, "_read_absolute_bound_payload", lambda *_args, **_kwargs: b""
    )
    one = auditor._audit_fixed_dataset(
        authorization_sha256="a" * 64,
        expected_manifest_file_sha256="b" * 64,
        inputs=inputs,
        workers=1,
    )
    six = auditor._audit_fixed_dataset(
        authorization_sha256="a" * 64,
        expected_manifest_file_sha256="b" * 64,
        inputs=inputs,
        workers=6,
    )
    assert auditor.canonical_json_bytes(one) == auditor.canonical_json_bytes(six)
    assert worker_calls == [1, 6]


def test_v12_worker_wiring_reauthorizes_and_hides_accelerators() -> None:
    recomputer = inspect.getsource(auditor._exact_sample_recomputer)
    task = inspect.getsource(auditor._recompute_exact_sample_task)
    initializer = inspect.getsource(auditor._initialize_exact_worker)
    assert 'multiprocessing.get_context("spawn")' in recomputer
    assert "initializer=_initialize_exact_worker" in recomputer
    assert "_require_exact_authority(authorization_sha256)" in task
    assert "_require_exact_authority(authorization_sha256)" in initializer
    old = {name: os.environ.get(name) for name in (*auditor.THREAD_ENVIRONMENT, *auditor.ACCELERATOR_ENVIRONMENT)}
    try:
        auditor._set_worker_environment()
        assert all(os.environ[name] == "1" for name in auditor.THREAD_ENVIRONMENT)
        assert all(os.environ[name] == "" for name in auditor.ACCELERATOR_ENVIRONMENT)
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_v12_selector_delta_from_v11_is_only_hsa() -> None:
    assert auditor.THREAD_ENVIRONMENT == predecessor.THREAD_ENVIRONMENT
    assert auditor.ACCELERATOR_ENVIRONMENT == (
        *predecessor.ACCELERATOR_ENVIRONMENT,
        "HSA_VISIBLE_DEVICES",
    )
    assert auditor.ACCELERATOR_ENVIRONMENT == (
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
        "HSA_VISIBLE_DEVICES",
    )
    assert "HSA_OVERRIDE_GFX_VERSION" not in auditor.ACCELERATOR_ENVIRONMENT
    assert _function_ast(auditor, "_set_worker_environment") == _function_ast(
        predecessor, "_set_worker_environment"
    )


def test_v12_all_real_spawn_tasks_clear_hostile_selectors_before_authority() -> None:
    task_names = (
        "_validate_one_shard_task",
        "_hash_source_file",
        "_recompute_exact_sample_task",
    )
    old = {
        name: os.environ.get(name) for name in auditor.ACCELERATOR_ENVIRONMENT
    }
    try:
        for name in auditor.ACCELERATOR_ENVIRONMENT:
            os.environ[name] = f"hostile-parent-{name}"
        with ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_spawn_synthetic_initializer,
        ) as executor:
            observed = list(executor.map(_spawn_task_order_probe, task_names))
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    assert observed == [("",) * 5] * 3


def test_v12_ast_enumerates_every_production_spawn_target_and_initializer() -> None:
    tree = ast.parse(Path(auditor.__file__).read_text(encoding="utf-8"))
    targets = {
        call.args[0].id
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "map"
        and call.args
        and isinstance(call.args[0], ast.Name)
    }
    pool_calls = [
        call
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "ProcessPoolExecutor"
    ]
    assert targets == {
        "_validate_one_shard_task",
        "_hash_source_file",
        "_recompute_exact_sample_task",
    }
    assert len(pool_calls) == 3
    assert all(
        any(
            keyword.arg == "initializer"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "_initialize_exact_worker"
            for keyword in call.keywords
        )
        for call in pool_calls
    )


def test_v12_cli_sanitizes_before_import_under_hostile_environment() -> None:
    program = """
import json
import os
from scripts import audit_go2_shared_jepa_v5_raw_supervision_v12
names = (
    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES',
    'ROCR_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL', 'HSA_VISIBLE_DEVICES',
    'HSA_OVERRIDE_GFX_VERSION',
)
print(json.dumps({name: os.environ.get(name) for name in names}, sort_keys=True))
"""
    environment = dict(os.environ)
    for name in (*auditor.THREAD_ENVIRONMENT, *auditor.ACCELERATOR_ENVIRONMENT):
        environment[name] = f"hostile-{name}"
    environment["HSA_OVERRIDE_GFX_VERSION"] = "hostile-override"
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(completed.stdout)
    assert tuple(observed[name] for name in auditor.THREAD_ENVIRONMENT) == (
        "1",
    ) * 4
    assert tuple(observed[name] for name in auditor.ACCELERATOR_ENVIRONMENT) == (
        "",
    ) * 5
    assert observed["HSA_OVERRIDE_GFX_VERSION"] is None


def test_v12_real_success_publication_preserves_complete_predecessor_chain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = (tmp_path / "repository").absolute()
    publication_parent = root / "generated"
    dataset = publication_parent / "development_raw_supervision_v1"
    dataset.mkdir(parents=True)

    def write(path: Path, payload: bytes) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return hashlib.sha256(payload).hexdigest()

    data_payload = b"immutable synthetic dataset bytes\n"
    data_path = dataset / "data.bin"
    data_sha256 = write(data_path, data_payload)
    manifest_payload = b'{"synthetic":true}\n'
    manifest_path = dataset / "manifest.json"
    manifest_sha256 = write(manifest_path, manifest_payload)
    v9_failure = publication_parent / f"{dataset.name}.audit_v9.failed.json"
    v10_authorization = root / "docs" / "audit_v10_authorization.json"
    v10_failure = publication_parent / f"{dataset.name}.audit_v10.failed.json"
    v11_block = root / "docs" / "audit_v11_block.json"
    v12_authorization = root / "docs" / "audit_v12_authorization.json"
    predecessor_paths = (v9_failure, v10_authorization, v10_failure, v11_block)
    predecessor_digests = tuple(
        write(path, f"{path.name}\n".encode("ascii"))
        for path in predecessor_paths
    )
    v9_success = publication_parent / f"{dataset.name}.audit_v9.json"
    v10_success = publication_parent / f"{dataset.name}.audit_v10.json"
    v11_success = publication_parent / f"{dataset.name}.audit_v11.json"
    v12_report = publication_parent / f"{dataset.name}.audit_v12.json"
    v12_failure = publication_parent / f"{dataset.name}.audit_v12.failed.json"
    v12_authorization_sha256 = write(v12_authorization, b"v12 authority\n")
    metadata_paths = (
        root / "metadata" / "manifest.json",
        root / "metadata" / "rows.jsonl",
        root / "metadata" / "source.json",
    )
    metadata_digests = tuple(
        write(path, f"{path.name}\n".encode("ascii"))
        for path in metadata_paths
    )
    protected = {
        path: path.read_bytes()
        for path in (manifest_path, data_path, *predecessor_paths)
    }
    monkeypatch.setattr(auditor, "ROOT", root)
    monkeypatch.setattr(auditor, "CANONICAL_DATASET", dataset)
    monkeypatch.setattr(auditor, "CANONICAL_AUDIT_REPORT", v12_report)
    monkeypatch.setattr(auditor, "CANONICAL_AUDIT_FAILURE", v12_failure)
    monkeypatch.setattr(auditor, "CANONICAL_V9_AUDIT_REPORT", v9_success)
    monkeypatch.setattr(auditor, "CANONICAL_V10_AUDIT_REPORT", v10_success)
    monkeypatch.setattr(auditor, "CANONICAL_V11_AUDIT_REPORT", v11_success)
    monkeypatch.setattr(auditor, "AUDIT_AUTHORIZATION_PATH", v12_authorization)
    monkeypatch.setattr(auditor, "FROZEN_V9_PREDECESSOR_SHA256", {})
    monkeypatch.setattr(auditor, "REVIEWED_V4_SOURCE_SHA256", {})
    for name, path, digest in zip(
        (
            "DATASET_MANIFEST_RELATIVE_PATH",
            "DATASET_ROWS_RELATIVE_PATH",
            "SOURCE_INDEX_RELATIVE_PATH",
        ),
        metadata_paths,
        metadata_digests,
    ):
        monkeypatch.setattr(auditor, name, str(path.relative_to(root)))
        monkeypatch.setattr(
            auditor,
            name.replace("RELATIVE_PATH", "FILE_SHA256"),
            digest,
        )
    sources = tuple(
        auditor.SourceBindingV12(
            role=f"predecessor_{index}",
            path=str(path.relative_to(root)),
            sha256=digest,
        )
        for index, (path, digest) in enumerate(
            zip(predecessor_paths, predecessor_digests)
        )
    )
    authorization = auditor.AcceptedAuthorizationV12(
        authorization_file_sha256="1" * 64,
        authorization_content_sha256="2" * 64,
        source_map_sha256="3" * 64,
        execution_authorization_file_sha256=v12_authorization_sha256,
        execution_authorization_content_sha256="4" * 64,
        execution_source_map_sha256="5" * 64,
        sources=sources,
    )
    context = auditor._AuditPublicationContextV12(
        authorization=authorization,
        manifest={
            "files": [
                {
                    "path": "data.bin",
                    "byte_count": len(data_payload),
                    "file_sha256": data_sha256,
                }
            ]
        },
        manifest_file_sha256=manifest_sha256,
        hashed_sources=(),
        parent_contracts=(),
    )
    authority_fields = (
        "rgb_decode_authorized",
        "dataset_use_authorized",
        "training_authorized",
        "selection_authorized",
        "calibration_authorized",
        "g2_authorized",
        "heldout_authorized",
        "runtime_authorized",
        "navigation_authorized",
        "hardware_authorized",
        "production_authorized",
        "promotion_authorized",
        "deployment_authorized",
    )
    result_core = {
        "schema": auditor.AUDIT_SCHEMA,
        "verdict": "PASS",
        **{field: False for field in authority_fields},
        "frozen_v9_terminal_artifacts": {
            "dataset_manifest_file_sha256": auditor.FROZEN_AUTHORITY_ROLE_SHA256[
                "v9_dataset_manifest"
            ],
            "terminal_failure_file_sha256": auditor.FROZEN_AUTHORITY_ROLE_SHA256[
                "v9_terminal_failure"
            ],
            "success_report_absent": True,
        },
        "frozen_v11_terminal_artifacts": {
            "auditor_block_file_sha256": auditor.FROZEN_AUTHORITY_ROLE_SHA256[
                "v11_auditor_block"
            ],
            "success_report_absent": True,
        },
        "frozen_v10_terminal_artifacts": {
            "audit_authorization_file_sha256": auditor.FROZEN_AUTHORITY_ROLE_SHA256[
                "v10_audit_authorization"
            ],
            "terminal_failure_file_sha256": auditor.FROZEN_AUTHORITY_ROLE_SHA256[
                "v10_terminal_failure"
            ],
            "success_report_absent": True,
        },
    }
    result = {
        **result_core,
        "content_sha256": auditor.canonical_json_sha256(result_core),
    }
    probe_retained = auditor._open_retained_directory_chain(publication_parent)
    probe_name, probe_descriptor, probe_fingerprint, probe_digest = (
        auditor._stage_owned_audit_candidate(probe_retained, result)
    )
    probe_transaction = auditor._ClosedAuditPublicationTransaction(
        context=context,
        retained=probe_retained,
        candidate_name=probe_name,
        candidate_descriptor=probe_descriptor,
        candidate_fingerprint=probe_fingerprint,
        candidate_sha256=probe_digest,
    )
    try:
        v11_success.write_bytes(b'{"verdict":"PASS"}\n')
        with pytest.raises(PermissionError, match="forbidden V11"):
            probe_transaction.validate_before_rename()
    finally:
        probe_transaction.close()
        v11_success.unlink()
        assert auditor._cleanup_owned_audit_candidate(
            probe_retained,
            candidate_name=probe_name,
            candidate_descriptor=probe_descriptor,
            renamed=False,
        )
        os.close(probe_descriptor)
        probe_retained.close()
    retained = auditor._open_retained_directory_chain(publication_parent)
    candidate_name, descriptor, fingerprint, digest = (
        auditor._stage_owned_audit_candidate(retained, result)
    )
    transaction = auditor._ClosedAuditPublicationTransaction(
        context=context,
        retained=retained,
        candidate_name=candidate_name,
        candidate_descriptor=descriptor,
        candidate_fingerprint=fingerprint,
        candidate_sha256=digest,
    )
    try:
        transaction.validate_before_rename()
        transaction.rename_owned()
        transaction.validate_after_rename()
        os.fsync(retained.directory_fd)
        transaction.require_final_quiet()
        assert v12_report.read_bytes() == auditor.canonical_json_bytes(result) + b"\n"
        assert not v12_failure.exists()
        assert not v9_success.exists()
        assert not v10_success.exists()
        assert not v11_success.exists()
        assert all(path.read_bytes() == payload for path, payload in protected.items())
    finally:
        transaction.close()
        os.close(descriptor)
        retained.close()

    second_retained = auditor._open_retained_directory_chain(publication_parent)
    second_name, second_descriptor, _fingerprint, _digest = (
        auditor._stage_owned_audit_candidate(second_retained, result)
    )
    try:
        with pytest.raises(FileExistsError):
            auditor._rename_noreplace_at(
                second_retained.directory_fd, second_name, v12_report.name
            )
        assert auditor._cleanup_owned_audit_candidate(
            second_retained,
            candidate_name=second_name,
            candidate_descriptor=second_descriptor,
            renamed=False,
        )
    finally:
        os.close(second_descriptor)
        second_retained.close()
    assert all(path.read_bytes() == payload for path, payload in protected.items())


def test_v12_failure_publication_is_additive_and_preserves_predecessors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "development_raw_supervision_v1"
    dataset.mkdir()
    dataset_payload = b"immutable dataset payload\n"
    (dataset / "data.bin").write_bytes(dataset_payload)
    v9_failure = tmp_path / "development_raw_supervision_v1.audit_v9.failed.json"
    v9_bytes = b'{"status":"terminal_failed_no_dataset_authority"}\n'
    v9_failure.write_bytes(v9_bytes)
    v10_authorization = tmp_path / "audit_v10_authorization.json"
    v10_authorization_bytes = b'{"exact_audit_v10_authorized":true}\n'
    v10_authorization.write_bytes(v10_authorization_bytes)
    v10_failure = tmp_path / "development_raw_supervision_v1.audit_v10.failed.json"
    v10_bytes = b'{"status":"terminal_failed_no_dataset_authority"}\n'
    v10_failure.write_bytes(v10_bytes)
    v11_block = tmp_path / "audit_v11_block.json"
    v11_block_bytes = b'{"verdict":"BLOCK"}\n'
    v11_block.write_bytes(v11_block_bytes)
    v9_success = tmp_path / "development_raw_supervision_v1.audit_v9.json"
    v10_success = tmp_path / "development_raw_supervision_v1.audit_v10.json"
    v11_success = tmp_path / "development_raw_supervision_v1.audit_v11.json"
    v12_report = tmp_path / "development_raw_supervision_v1.audit_v12.json"
    v12_failure = tmp_path / "development_raw_supervision_v1.audit_v12.failed.json"
    monkeypatch.setattr(auditor, "CANONICAL_DATASET", dataset)
    monkeypatch.setattr(auditor, "CANONICAL_AUDIT_REPORT", v12_report)
    monkeypatch.setattr(auditor, "CANONICAL_AUDIT_FAILURE", v12_failure)
    monkeypatch.setattr(auditor, "CANONICAL_V9_AUDIT_REPORT", v9_success)
    monkeypatch.setattr(auditor, "CANONICAL_V10_AUDIT_REPORT", v10_success)
    monkeypatch.setattr(auditor, "CANONICAL_V11_AUDIT_REPORT", v11_success)
    auditor._publish_terminal_audit_failure(
        authorization_sha256="a" * 64,
        error=RuntimeError("synthetic author failure"),
    )
    assert v9_failure.read_bytes() == v9_bytes
    assert v10_authorization.read_bytes() == v10_authorization_bytes
    assert v10_failure.read_bytes() == v10_bytes
    assert v11_block.read_bytes() == v11_block_bytes
    assert (dataset / "data.bin").read_bytes() == dataset_payload
    assert not v9_success.exists()
    assert not v10_success.exists()
    assert not v11_success.exists()
    assert not v12_report.exists()
    value = json.loads(v12_failure.read_bytes())
    assert value["schema"] == auditor.AUDIT_FAILURE_SCHEMA
    assert value["status"] == "terminal_failed_no_dataset_authority"
    assert value["retry_authorized"] is False
    for field in (
        "rgb_decode_authorized",
        "dataset_use_authorized",
        "training_authorized",
        "selection_authorized",
        "calibration_authorized",
        "g2_authorized",
        "heldout_authorized",
        "runtime_authorized",
        "navigation_authorized",
        "hardware_authorized",
        "production_authorized",
        "promotion_authorized",
        "deployment_authorized",
    ):
        assert value[field] is False
    assert value["v10_audit_authorization_file_sha256"] == (
        auditor.FROZEN_AUTHORITY_ROLE_SHA256["v10_audit_authorization"]
    )
    assert value["v10_terminal_failure_file_sha256"] == (
        auditor.FROZEN_AUTHORITY_ROLE_SHA256["v10_terminal_failure"]
    )
    assert value["v10_success_report_absent"] is True
    assert value["v9_success_report_absent"] is True
    assert value["v11_success_report_absent"] is True
    assert value["v11_auditor_block_file_sha256"] == (
        auditor.FROZEN_AUTHORITY_ROLE_SHA256["v11_auditor_block"]
    )
    core = dict(value)
    declared = core.pop("content_sha256")
    assert declared == auditor.canonical_json_sha256(core)
    assert v12_failure.read_bytes() == auditor.canonical_json_bytes(value) + b"\n"
    with pytest.raises(FileExistsError):
        auditor._publish_terminal_audit_failure(
            authorization_sha256="a" * 64,
            error=RuntimeError("second attempt"),
        )
    assert v9_failure.read_bytes() == v9_bytes
    assert v10_authorization.read_bytes() == v10_authorization_bytes
    assert v10_failure.read_bytes() == v10_bytes
    assert v11_block.read_bytes() == v11_block_bytes
    assert (dataset / "data.bin").read_bytes() == dataset_payload
    assert not v9_success.exists()
    assert not v10_success.exists()
    assert not v11_success.exists()


@pytest.mark.parametrize(
    ("version", "attribute"),
    [
        ("V9", "CANONICAL_V9_AUDIT_REPORT"),
        ("V10", "CANONICAL_V10_AUDIT_REPORT"),
        ("V11", "CANONICAL_V11_AUDIT_REPORT"),
    ],
)
def test_v12_terminal_failure_refuses_every_forbidden_predecessor_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    version: str,
    attribute: str,
) -> None:
    dataset = tmp_path / "development_raw_supervision_v1"
    dataset.mkdir()
    success = tmp_path / f"{dataset.name}.{version.lower()}.json"
    success.write_bytes(b'{"verdict":"PASS"}\n')
    for predecessor_version, predecessor_attribute in (
        ("V9", "CANONICAL_V9_AUDIT_REPORT"),
        ("V10", "CANONICAL_V10_AUDIT_REPORT"),
        ("V11", "CANONICAL_V11_AUDIT_REPORT"),
    ):
        path = (
            success
            if predecessor_version == version
            else tmp_path / f"absent-{predecessor_version}.json"
        )
        monkeypatch.setattr(auditor, predecessor_attribute, path)
    v12_report = tmp_path / f"{dataset.name}.audit_v12.json"
    v12_failure = tmp_path / f"{dataset.name}.audit_v12.failed.json"
    monkeypatch.setattr(auditor, "CANONICAL_DATASET", dataset)
    monkeypatch.setattr(auditor, attribute, success)
    monkeypatch.setattr(auditor, "CANONICAL_AUDIT_REPORT", v12_report)
    monkeypatch.setattr(auditor, "CANONICAL_AUDIT_FAILURE", v12_failure)
    with pytest.raises(PermissionError, match=f"forbidden {version}"):
        auditor._publish_terminal_audit_failure(
            authorization_sha256="a" * 64,
            error=RuntimeError("synthetic author failure"),
        )
    assert not v12_report.exists()
    assert not v12_failure.exists()
    assert success.read_bytes() == b'{"verdict":"PASS"}\n'


def test_v12_production_and_cli_surface_is_fixed_and_audit_only() -> None:
    signature = inspect.signature(auditor.execute_exact_audit_v12)
    assert tuple(signature.parameters) == ("authorization_sha256", "workers")
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
    source = Path(auditor.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    ]
    assert not any(
        re.search(
            r"go2_shared_jepa_v5_raw_supervision_auditor_v(?:9|10|11)(?:$|\.)",
            module,
        )
        or "go2_shared_jepa_v5_raw_supervision_builder" in module
        for module in imports
    )
    assert "test_hook" not in source
    assert "importlib" not in source
    function_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    forbidden_entry_fragments = (
        "skip",
        "retry",
        "fallback",
        "rebuild",
        "trainer",
        "gpu",
        "g2",
        "heldout",
        "navigation",
        "runtime",
    )
    assert not any(
        fragment in name.lower()
        for name in function_names
        for fragment in forbidden_entry_fragments
    )
    cli = Path("scripts/audit_go2_shared_jepa_v5_raw_supervision_v12.py").read_text(
        encoding="utf-8"
    )
    assert "raw_supervision_auditor_v12" in cli
    assert "execute_exact_audit_v12" in cli
    assert "--authorization-sha256" in cli and "--workers" in cli
    for forbidden in ("--path", "--retry", "--fallback", "--mode", "trainer"):
        assert forbidden not in cli
