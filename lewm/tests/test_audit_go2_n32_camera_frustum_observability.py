from __future__ import annotations

import copy
from collections import Counter
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import weakref

import numpy as np
import pytest

from lewm.benchmarks import go2_n32_camera_frustum_observability as audit_core
from lewm.benchmarks.go2_n32_camera_frustum_observability import (
    ENDPOINT_SIDES,
    FAMILIES,
    canonical_json_sha256,
)
from lewm.datasets import go2_paired_navigation as labels_v3
from lewm_worlds.manifest import (
    BoxObject,
    CameraValidityConstraints,
    SceneManifest,
    SpawnSpec,
)
from lewm_worlds import manifest as scene_manifest_module
from lewm_worlds import planning_grid as planning_grid_module
from lewm_worlds.planning_grid import InflatedOccupancyGrid
from scripts import audit_go2_n32_camera_frustum_observability as audit


audit._install_semantic_modules(
    audit_core,
    labels_v3,
    scene_manifest_module,
    planning_grid_module,
)


def _digest(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _panel_row(index: int, family: str, root: Path) -> dict:
    shard_index = index // 8
    shard_row = index % 8
    return {
        "scene_id": f"{family}_scene_{index:03d}",
        "family": family,
        "dataset_role": "train",
        "global_row": index,
        "env_index": index % 4,
        "episode_id": str(index),
        "reset_count": 0,
        "current_episode_step": 2,
        "next_episode_step": 3,
        "current_frame_index": index * 2,
        "next_frame_index": index * 2 + 1,
        "current_timestamp_ns": index * 100 + 1,
        "next_timestamp_ns": index * 100 + 2,
        "primitive": "forward",
        "relative_se2_current_frame": [0.1, 0.0, 0.0],
        "label_shard_path": str(root / f"shard_{shard_index:02d}.npz"),
        "label_shard_sha256": _digest(f"shard:{shard_index}"),
        "label_shard_row": shard_row,
        "current_image_path": str(root / f"scene_{index:03d}/rgb/current.png"),
        "current_image_sha256": _digest(f"current:{index}"),
        "next_image_path": str(root / f"scene_{index:03d}/rgb/next.png"),
        "next_image_sha256": _digest(f"next:{index}"),
    }


def _synthetic_panel(root: Path) -> tuple[dict, str, str]:
    rows = [
        _panel_row(family_index * 32 + offset, family, root)
        for family_index, family in enumerate(FAMILIES)
        for offset in range(32)
    ]
    rows_sha = canonical_json_sha256(rows)
    core = {
        "schema": "lewm_go2_physical_micro_overfit_panel_v1",
        "families": list(FAMILIES),
        "rows_per_family_panel": 32,
        "local_grid": {
            "shape": [64, 64],
            "cell_size_m": 0.10,
            "forward_edge_range_m": [-1.0, 5.4],
            "left_edge_range_m": [-3.2, 3.2],
        },
        "source_camera_projection": {
            "horizontal_fov_deg": 78.323,
            "near_m": 0.05,
        },
        "inputs": {
            "geometry_contract": {
                "path": str(root / "geometry.json"),
                "file_sha256": "a" * 64,
                "semantic_sha256": "b" * 64,
            },
            "render_audit_contract": {
                "path": str(root / "render_audit.json"),
                "file_sha256": "c" * 64,
                "content_sha256": "d" * 64,
            },
        },
        "panels": {
            "fit": {
                "row_count": 160,
                "frame_count": 320,
                "rows_sha256": rows_sha,
                "rows": rows,
            },
            # These strings must never be interpreted or dereferenced by the audit.
            "same_scene_holdout": {"rows": [{"label_shard_path": "/sealed/no"}]},
            "cross_scene_holdout": {"rows": [{"label_shard_path": "/g2/no"}]},
        },
    }
    panel = {**core, "content_sha256": canonical_json_sha256(core)}
    return panel, panel["content_sha256"], rows_sha


def _record(
    root: Path,
    *,
    index: int,
    shard: Path,
    shard_sha256: str,
    row: int,
    side: str,
) -> dict:
    family = FAMILIES[(index // 64) % len(FAMILIES)]
    return {
        "family": family,
        "scene_id": f"scene_{index:03d}",
        "global_row": index // 2,
        "side": side,
        "image_path_metadata_only": str(root / f"render_{index:03d}/rgb/frame.png"),
        "image_sha256": _digest(f"image:{index}"),
        "label_shard_path": str(shard),
        "label_shard_sha256": shard_sha256,
        "label_row": row,
        "frame_index": index,
        "env_index": 0,
        "timestamp_ns": index + 1,
        "episode_id": str(index),
        "reset_count": 0,
        "episode_step": 1,
    }


def _synthetic_shard_arrays(
    current: np.ndarray,
    next_target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, np.ndarray]:
    rows = int(current.shape[0])
    text = np.asarray(["synthetic"] * rows)
    hashes = np.asarray(["a" * 64] * rows)
    return {
        "current_labels": current,
        "next_labels": next_target,
        "current_supervision_mask": mask,
        "next_supervision_mask": mask,
        "current_observed_mask": current != 0,
        "next_observed_mask": next_target != 0,
        "relative_se2_current_frame": np.zeros((rows, 3), dtype=np.float32),
        "primitive": text,
        "current_image_path": text,
        "next_image_path": text,
        "current_image_sha256": hashes,
        "next_image_sha256": hashes,
    }


def _synthetic_machine_manifest(
    source_hashes: dict[str, dict[str, str]],
) -> dict:
    selected_tuples = []
    for family_index, family in enumerate(FAMILIES):
        for offset in range(32):
            global_row = family_index * 32 + offset
            scene_id = f"{family}_scene_{offset:03d}"
            for side in ENDPOINT_SIDES:
                selected_tuples.append([family, scene_id, global_row, side, -1])
    label_entries = []
    for shard_index in range(20):
        shard_tuples = copy.deepcopy(selected_tuples[shard_index * 16 : (shard_index + 1) * 16])
        for endpoint_index, value in enumerate(shard_tuples):
            value[4] = endpoint_index // 2
        counts = {
            family: {
                side: sum(value[0] == family and value[3] == side for value in shard_tuples)
                for side in ENDPOINT_SIDES
            }
            for family in FAMILIES
        }
        label_entries.append(
            {
                "path": f"/synthetic/shard_{shard_index:02d}.npz",
                "sha256": _digest(f"shard:{shard_index}"),
                "selected_tuples": shard_tuples,
                "selected_row_count": len(shard_tuples),
                "family_side_counts": counts,
            }
        )
    source_entries = []
    shared_paths = {
        "physical_geometry_contract": "/synthetic/geometry.json",
        "render_audit_contract": "/synthetic/render_audit.json",
        "renderer_source": "/synthetic/renderer.py",
    }
    scene_roles = {
        "fit_render_summary": "summary.json",
        "fit_frame_selection": "frame_selection.json",
        "render_source_plan": "plan.json",
        "source_frames_jsonl": "frames.jsonl",
        "source_scene_manifest": "manifest.json",
    }
    for scene_index in range(20):
        scene_id = f"scene_{scene_index:02d}"
        for role, path in shared_paths.items():
            source_entries.append(
                {
                    "path": path,
                    "sha256": _digest(path),
                    "semantic_role": role,
                    "scene_id": scene_id,
                }
            )
        for role, filename in scene_roles.items():
            path = f"/synthetic/{scene_id}/{filename}"
            source_entries.append(
                {
                    "path": path,
                    "sha256": _digest(path),
                    "semantic_role": role,
                    "scene_id": scene_id,
                }
            )
    source_entries.sort(
        key=lambda entry: (
            entry["path"],
            entry["sha256"],
            entry["semantic_role"],
            entry["scene_id"],
        )
    )
    summary_entries = [
        dict(entry)
        for entry in source_entries
        if entry["semantic_role"] == "fit_render_summary"
    ]
    source_map_entries = [
        {"role": role, "path": record["path"], "sha256": record["sha256"]}
        for role, record in sorted(source_hashes.items())
    ]
    core = {
        "schema": audit.MACHINE_MANIFEST_SCHEMA,
        "created_at_utc": "2026-07-11T12:00:00+00:00",
        "binding": {
            "path": str(audit.BINDING_PATH),
            "file_sha256": audit.EXECUTION_BINDING_SHA256,
        },
        "preflight_access_incident": {
            "path": str(audit.PREFLIGHT_INCIDENT_PATH),
            "file_sha256": audit.PREFLIGHT_INCIDENT_SHA256,
            "status": audit.PREFLIGHT_INCIDENT_STATUS,
        },
        "human_implementation_manifest": {
            "path": str(audit.IMPLEMENTATION_MANIFEST_PATH),
            "file_sha256": "a" * 64,
        },
        "authorized_inputs": {
            "fit_panel": {
                "semantic_role": "fit_panel",
                "path": str(audit.PANEL_PATH),
                "file_sha256": audit.PANEL_FILE_SHA256,
                "content_sha256": audit.PANEL_CONTENT_SHA256,
                "fit_rows_sha256": audit.FIT_ROWS_SHA256,
                "schema": "lewm_go2_physical_micro_overfit_panel_v1",
            },
            "v4_adjudication_report": {
                "semantic_role": "v4_adjudication_report",
                "path": str(audit.V4_REPORT_PATH),
                "file_sha256": audit.V4_REPORT_SHA256,
            },
            "known_bias_proof": {
                "semantic_role": "known_bias_proof",
                "path": str(audit.KNOWN_BIAS_PROOF_PATH),
                "file_sha256": audit.KNOWN_BIAS_PROOF_SHA256,
            },
            "physical_geometry_contract": {
                "semantic_role": "physical_geometry_contract",
                "path": "/synthetic/geometry.json",
                "file_sha256": _digest("/synthetic/geometry.json"),
                "semantic_sha256": "e" * 64,
                "schema": "lewm_go2_generalization_geometry_v2",
            },
            "label_shards": audit._canonical_manifest(label_entries),
            "render_summaries": audit._canonical_manifest(summary_entries),
            "source_geometry": audit._canonical_manifest(source_entries),
        },
        "source_map": {
            "entry_count": len(source_map_entries),
            "entries": source_map_entries,
            "source_map_sha256": canonical_json_sha256(source_map_entries),
        },
        "runtime_environment": audit._runtime_environment(),
        "verification_evidence": {
            "all_passed": True,
            "commands": [
                {
                    **copy.deepcopy(expected),
                    "exit_code": 0,
                    "captured_output_sha256": "b" * 64,
                }
                for expected in audit.REQUIRED_VERIFICATION_COMMANDS
            ],
        },
        "exclusive_output": {
            "path": str(audit.OUTPUT_PATH),
            "schema": audit.RESULT_SCHEMA,
            "absent_before_authorization": True,
            "zero_output_state": True,
        },
        "preparation_access_ledger": {},
        "review": {
            "reviewer_identity": "synthetic-reviewer",
            "status": "reviewed_and_authorized",
        },
        "authoritative_fit_audit_authorized": True,
    }
    preparation = audit.new_access_ledger()
    unique_source_paths = {entry["path"] for entry in source_entries}
    preparation.update(
        {
            "panel_metadata_byte_opens": 1,
            "implementation_source_hash_byte_opens": 2 * len(source_hashes),
            "document_hash_byte_opens": 4,
            "source_geometry_hash_byte_opens": 2 * len(unique_source_paths),
            "source_geometry_json_parses": sum(
                Path(path).suffix in {".json", ".jsonl"}
                for path in unique_source_paths
            ),
            "source_geometry_jsonl_records": 320,
            "source_frame_records_selected": 320,
        }
    )
    core["preparation_access_ledger"] = {
        **preparation,
        "passes": True,
        "forbidden_counters_zero": True,
    }
    return {**core, "content_sha256": canonical_json_sha256(core)}


def _empty_manifest(scene_id: str, family: str = FAMILIES[0]) -> SceneManifest:
    return SceneManifest(
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


def _source_frame() -> dict:
    return {
        "frame_index": 7,
        "env_index": 0,
        "timestamp_ns": 11,
        "episode": {
            "episode_id": 3,
            "reset_count": 0,
            "episode_step": 4,
            "split": "test_hard",
        },
        "base_pose_world": {
            "position": {"x": 0.0, "y": 0.0, "z": 0.35},
        },
        "base_rpy_rad": {"yaw": 0.0},
        "base_quat_world_xyzw": [0.0, 0.0, 0.0, 1.0],
        "camera_mount_body": copy.deepcopy(audit.NOMINAL_CAMERA_MOUNT_BODY),
        "camera_pose_world": {
            "position": [0.326, 0.0, 0.393],
            "lookat": [1.326, 0.0, 0.393],
            "up": [0.0, 0.0, 1.0],
        },
    }


def _projection() -> dict:
    return {
        "horizontal_fov_deg": 78.323,
        "vertical_fov_deg": 62.8370386364,
        "near_m": 0.05,
        "far_m": 200.0,
        "resolution_wh": [224, 168],
    }


def _geometry_flags() -> dict:
    return {
        "oracle_cell_size_m": 0.05,
        "landmarks_are_obstacles": True,
        "distractors_are_obstacles": True,
        "horizontal_fov_deg": 78.323,
        "near_m": 0.05,
    }


def test_wrong_authorization_fails_before_any_access(tmp_path: Path) -> None:
    ledger = audit.new_access_ledger()
    spec = audit.AuditSpec(output_path=tmp_path / "result.json")

    with pytest.raises(PermissionError, match="authorization"):
        audit.run_authoritative_audit(
            authorization_sha256="0" * 64,
            machine_manifest_sha256="1" * 64,
            spec=spec,
            ledger=ledger,
        )

    assert ledger["denied_attempt_records"] == []
    assert ledger["per_shard_materialization"] == []
    assert all(value == 0 for value in ledger["denied_primary_reasons"].values())
    assert all(value == 0 for value in ledger["denied_modality_attempts"].values())
    assert all(
        value == 0
        for key, value in ledger.items()
        if key not in {
            "denied_attempt_records",
            "denied_primary_reasons",
            "denied_modality_attempts",
            "per_shard_materialization",
        }
    )
    assert not spec.output_path.exists()


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        (b'{"value":1,"value":2}', "duplicate JSON key"),
        (b'{"value":NaN}', "forbidden JSON constant"),
        (b'{"value":Infinity}', "forbidden JSON constant"),
    ),
)
def test_strict_json_rejects_duplicate_keys_and_nonfinite_constants(
    payload: bytes, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        audit._strict_json_bytes(payload, name="synthetic JSON")
    parsed = audit._strict_json_bytes(b'{"z":2,"a":1}', name="canonical example")
    assert audit._canonical_json_bytes(parsed) == b'{"a":1,"z":2}'


@pytest.mark.parametrize(
    ("relative", "role", "declared_role", "reason"),
    (
        ("sealed/payload.json", "fit_panel", "train", "sealed"),
        ("go2_g2_dataset/payload.json", "fit_panel", "train", "g2"),
        ("seed_20260711/payload.json", "fit_panel", "train", "seed_20260711"),
        ("generated/v4/result.json", "generated_v4_result", "train", "generated_v4_result"),
        ("weights.pt", "fit_panel", "train", "model"),
        ("closed_loop/payload.json", "fit_panel", "train", "runtime"),
        ("physical.json", "fit_panel", "validation", "physical_nontrain"),
        ("selection/payload.json", "fit_panel", "train", "selection_or_calibration"),
        ("holdout/payload.json", "fit_panel", "train", "holdout"),
        ("rgb/frame.png", "fit_panel", "train", "image_or_depth"),
        ("payload.json", "not_registered", "train", "unregistered_role"),
        ("payload.zip", "fit_panel", "train", "forbidden_modality"),
    ),
)
def test_semantic_path_denial_precedence_is_exact_and_pre_resolution(
    tmp_path: Path,
    relative: str,
    role: str,
    declared_role: str,
    reason: str,
) -> None:
    ledger = audit.new_access_ledger()
    path = tmp_path / relative

    with pytest.raises(PermissionError, match=reason):
        audit._authorize_path(
            path,
            tmp_path,
            ledger=ledger,
            requested_role=role,
            declared_role=declared_role,
            label="synthetic denial",
        )

    assert ledger["denied_attempts_total"] == 1
    assert ledger["unexpected_path_attempts"] == 1
    assert ledger["denied_primary_reasons"][reason] == 1
    assert sum(ledger["denied_primary_reasons"].values()) == 1
    assert ledger["denied_attempt_records"][0]["resolved_path"] is None


def test_aliases_renamed_copies_and_declared_roles_cannot_bypass_allowlist(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "allowed.json"
    allowed.write_text("{}")
    aliases = [tmp_path / "renamed.json", tmp_path / "hardlink.json"]
    aliases[0].write_text("{}")
    aliases[1].hardlink_to(allowed)
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(allowed)
    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    nested_allowed = real_parent / "nested.json"
    nested_allowed.write_text("{}")
    symlink_parent = tmp_path / "symlink_parent"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)

    for path, expected_reason in (
        (aliases[0], "unallowlisted"),
        (aliases[1], "unallowlisted"),
        (symlink, "path_alias_or_escape"),
        (symlink_parent / "nested.json", "path_alias_or_escape"),
    ):
        ledger = audit.new_access_ledger()
        with pytest.raises(PermissionError, match=expected_reason):
            audit._authorize_path(
                path,
                tmp_path,
                ledger=ledger,
                requested_role="fit_panel",
                declared_role="train",
                expected_resolved_path=allowed,
                label="synthetic alias",
            )
        assert ledger["denied_primary_reasons"][expected_reason] == 1

    ledger = audit.new_access_ledger()
    with pytest.raises(PermissionError, match="g2"):
        audit._authorize_path(
            allowed,
            tmp_path,
            ledger=ledger,
            requested_role="fit_panel",
            declared_role="g2_evaluation",
            expected_resolved_path=allowed,
            label="semantic role conflict",
        )
    assert ledger["denied_primary_reasons"]["g2"] == 1

    for path, declared_role, reason in (
        (tmp_path / "physical_nontrain/payload.json", "train", "physical_nontrain"),
        (tmp_path / "seed_20260710_result.json", "train", "generated_v4_result"),
        (tmp_path / "model_output/payload.json", "train", "model"),
        (tmp_path / "models/payload.json", "train", "model"),
        (allowed, "model_output", "model"),
        (tmp_path / "parameters/payload.json", "train", "model"),
        (tmp_path / "images/payload.json", "train", "image_or_depth"),
        (tmp_path / "clip.webm", "train", "image_or_depth"),
        (allowed, "same_scene_holdout", "holdout"),
        (allowed, "unknown_role", "unregistered_role"),
    ):
        ledger = audit.new_access_ledger()
        with pytest.raises(PermissionError, match=reason):
            audit._authorize_path(
                path,
                tmp_path,
                ledger=ledger,
                requested_role="fit_panel",
                declared_role=declared_role,
                expected_resolved_path=allowed,
                label="semantic precedence",
        )
        assert ledger["denied_primary_reasons"][reason] == 1

    assert (
        audit._lexical_primary_denial(
            allowed,
            requested_role="fit_panel",
            declared_role=None,
            modality="json",
        )
        is None
    )

    for requested_role, reason in (
        ("g2", "g2"),
        ("seed_20260711", "seed_20260711"),
        ("generated_v4_result", "generated_v4_result"),
        ("model_output", "model"),
        ("runtime", "runtime"),
        ("physical_nontrain", "physical_nontrain"),
        ("calib", "selection_or_calibration"),
        ("heldout", "holdout"),
        ("pixels", "image_or_depth"),
    ):
        ledger = audit.new_access_ledger()
        with pytest.raises(PermissionError, match=reason):
            audit._authorize_path(
                allowed,
                tmp_path,
                ledger=ledger,
                requested_role=requested_role,
                declared_role="train",
                expected_resolved_path=allowed,
                label="requested-role semantic precedence",
            )
        assert ledger["denied_primary_reasons"][reason] == 1


def test_runner_import_is_repository_semantics_free_before_authorization() -> None:
    command = (
        "import sys; "
        "from scripts import audit_go2_n32_camera_frustum_observability as audit; "
        "assert not audit._SEMANTICS_LOADED; "
        "assert 'lewm.datasets.go2_paired_navigation' not in sys.modules; "
        "assert 'lewm_worlds.manifest' not in sys.modules; "
        "assert 'lewm_worlds.planning_grid' not in sys.modules; "
        "assert 'lewm.benchmarks.go2_n32_camera_frustum_observability' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=audit.ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_machine_manifest_is_semantically_validated_before_data_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_hashes = {
        role: {"path": f"/synthetic/{role}.py", "sha256": _digest(role)}
        for role in audit._default_source_paths()
    }
    source_hashes["binding"] = {
        "path": str(audit.BINDING_PATH),
        "sha256": audit.EXECUTION_BINDING_SHA256,
    }
    valid = _synthetic_machine_manifest(source_hashes)

    audit._validate_machine_manifest(
        valid,
        machine_file_sha256="c" * 64,
        source_hashes=source_hashes,
    )

    def duplicate_source(value: dict) -> None:
        inventory = value["authorized_inputs"]["source_geometry"]
        inventory["entries"].append(copy.deepcopy(inventory["entries"][0]))
        inventory["entry_count"] = len(inventory["entries"])
        inventory["manifest_sha256"] = canonical_json_sha256(inventory["entries"])

    def boolean_family_side_count(value: dict) -> None:
        inventory = value["authorized_inputs"]["label_shards"]
        entry = inventory["entries"][0]
        family = next(iter(entry["family_side_counts"]))
        side = next(iter(entry["family_side_counts"][family]))
        entry["family_side_counts"][family][side] = False
        inventory["manifest_sha256"] = canonical_json_sha256(inventory["entries"])

    mutations = (
        (lambda value: value.update(authoritative_fit_audit_authorized=False), "authorize"),
        (
            lambda value: value["source_map"]["entries"][0].update(sha256="0" * 64),
            "source map",
        ),
        (
            lambda value: value["preparation_access_ledger"].update(
                label_shard_npz_opens=1
            ),
            "label_shard_npz_opens",
        ),
        (
            lambda value: value["preflight_access_incident"].update(status="hidden"),
            "incident",
        ),
        (
            lambda value: value["verification_evidence"]["commands"][0].update(
                command="pytest synthetic"
            ),
            "frozen suite",
        ),
        (
            lambda value: value["verification_evidence"]["commands"][3][
                "deterministic_result"
            ].__setitem__("count", False),
            "strict integer",
        ),
        (
            lambda value: value["runtime_environment"][
                "python_implementation_version"
            ].__setitem__(4, False),
            "strict integer",
        ),
        (
            lambda value: value["exclusive_output"].__setitem__(
                "absent_before_authorization", 1
            ),
            "exclusive-output",
        ),
        (boolean_family_side_count, "strict integer"),
        (
            lambda value: value["preparation_access_ledger"].pop(
                "source_geometry_json_parses"
            ),
            "ledger fields",
        ),
        (
            lambda value: value["preparation_access_ledger"][
                "denied_primary_reasons"
            ].__setitem__("sealed", False),
            "strict integer",
        ),
        (
            lambda value: value["preparation_access_ledger"][
                "denied_modality_attempts"
            ].__setitem__("video", False),
            "strict integer",
        ),
        (duplicate_source, "duplicate"),
    )
    for mutate, message in mutations:
        changed = copy.deepcopy(valid)
        mutate(changed)
        core = dict(changed)
        core.pop("content_sha256")
        changed["content_sha256"] = canonical_json_sha256(core)
        with pytest.raises((ValueError, PermissionError), match=message):
            audit._validate_machine_manifest(
                changed,
                machine_file_sha256="c" * 64,
                source_hashes=source_hashes,
            )

    pretty_path = tmp_path / "machine.json"
    pretty_path.write_text(json.dumps(valid, indent=2, sort_keys=True) + "\n")
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "MACHINE_IMPLEMENTATION_MANIFEST_PATH", pretty_path)
    with pytest.raises(ValueError, match="canonical compact"):
        audit._load_machine_manifest(
            audit._hash_file(pretty_path),
            source_hashes=source_hashes,
            ledger=audit.new_access_ledger(),
        )


def test_fit_panel_canonicalizes_exact_160_transitions_and_320_frames(
    tmp_path: Path,
) -> None:
    panel, content_sha, rows_sha = _synthetic_panel(tmp_path)
    spec = audit.AuditSpec(
        panel_content_sha256=content_sha,
        fit_rows_sha256=rows_sha,
    )

    records, metadata = audit._canonicalize_fit_panel(panel, spec=spec)

    assert len(records) == 320
    assert Counter(record["family"] for record in records) == Counter(
        {family: 64 for family in FAMILIES}
    )
    assert {record["side"] for record in records} == set(ENDPOINT_SIDES)
    assert len({tuple(audit._frame_identity_values(record)) for record in records}) == 320
    assert metadata["geometry_contract"]["path"].endswith("geometry.json")

    duplicate = copy.deepcopy(panel)
    duplicate["panels"]["fit"]["rows"][1]["global_row"] = duplicate["panels"][
        "fit"
    ]["rows"][0]["global_row"]
    duplicate["panels"]["fit"]["rows"][1]["scene_id"] = duplicate["panels"][
        "fit"
    ]["rows"][0]["scene_id"]
    duplicate["panels"]["fit"]["rows_sha256"] = canonical_json_sha256(
        duplicate["panels"]["fit"]["rows"]
    )
    duplicate_core = dict(duplicate)
    duplicate_core.pop("content_sha256")
    duplicate["content_sha256"] = canonical_json_sha256(duplicate_core)
    with pytest.raises(ValueError, match="canonical frame coordinate"):
        audit._canonicalize_fit_panel(
            duplicate,
            spec=audit.AuditSpec(
                panel_content_sha256=duplicate["content_sha256"],
                fit_rows_sha256=duplicate["panels"]["fit"]["rows_sha256"],
            ),
        )

    numeric_mutations = (
        lambda value: value.__setitem__("rows_per_family_panel", False),
        lambda value: value["panels"]["fit"].__setitem__("row_count", False),
        lambda value: value["panels"]["fit"].__setitem__("frame_count", False),
        lambda value: value["panels"]["fit"]["rows"][0].__setitem__(
            "label_shard_row", False
        ),
        lambda value: value["panels"]["fit"]["rows"][0].__setitem__(
            "global_row", False
        ),
        lambda value: value["panels"]["fit"]["rows"][0].__setitem__(
            "current_frame_index", False
        ),
        lambda value: value["panels"]["fit"]["rows"][0].__setitem__(
            "env_index", False
        ),
        lambda value: value["panels"]["fit"]["rows"][0].__setitem__(
            "current_timestamp_ns", False
        ),
        lambda value: value["panels"]["fit"]["rows"][0].__setitem__(
            "reset_count", False
        ),
        lambda value: value["panels"]["fit"]["rows"][0].__setitem__(
            "current_episode_step", False
        ),
    )
    for mutate in numeric_mutations:
        changed = copy.deepcopy(panel)
        mutate(changed)
        changed_rows = changed["panels"]["fit"]["rows"]
        changed["panels"]["fit"]["rows_sha256"] = canonical_json_sha256(
            changed_rows
        )
        changed_core = dict(changed)
        changed_core.pop("content_sha256")
        changed["content_sha256"] = canonical_json_sha256(changed_core)
        with pytest.raises(ValueError, match="strict integer"):
            audit._canonicalize_fit_panel(
                changed,
                spec=audit.AuditSpec(
                    panel_content_sha256=changed["content_sha256"],
                    fit_rows_sha256=changed["panels"]["fit"]["rows_sha256"],
                ),
            )


def test_fit_panel_rejects_nontrain_role_without_touching_holdout_metadata(
    tmp_path: Path,
) -> None:
    panel, _content_sha, rows_sha = _synthetic_panel(tmp_path)
    panel["panels"]["fit"]["rows"][0]["dataset_role"] = "g2_evaluation"
    panel["panels"]["fit"]["rows_sha256"] = canonical_json_sha256(
        panel["panels"]["fit"]["rows"]
    )
    core = dict(panel)
    core.pop("content_sha256")
    panel["content_sha256"] = canonical_json_sha256(core)
    spec = audit.AuditSpec(
        panel_content_sha256=panel["content_sha256"],
        fit_rows_sha256=panel["panels"]["fit"]["rows_sha256"],
    )

    with pytest.raises(PermissionError, match="physical train role"):
        audit._canonicalize_fit_panel(panel, spec=spec)

    assert rows_sha != spec.fit_rows_sha256


def test_exact_20_shards_are_opened_once_for_exact_320_selected_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    grouped: dict[Path, list[dict]] = {}
    raw_open_count = 0
    original_read = audit._read_bytes
    index = 0
    for shard_index in range(20):
        path = tmp_path / f"shard_{shard_index:02d}.npz"
        rows = 8
        current = np.zeros((rows, 64, 64), dtype=np.uint8)
        next_target = np.ones((rows, 64, 64), dtype=np.uint8)
        mask = np.ones_like(current, dtype=bool)
        np.savez_compressed(
            path,
            **_synthetic_shard_arrays(current, next_target, mask),
        )
        digest = audit._hash_file(path)
        records = []
        for row in range(rows):
            for side in ENDPOINT_SIDES:
                records.append(
                    _record(
                        tmp_path,
                        index=index,
                        shard=path,
                        shard_sha256=digest,
                        row=row,
                        side=side,
                    )
                )
                index += 1
        grouped[path] = records

    def counted(path: Path) -> bytes:
        nonlocal raw_open_count
        raw_open_count += 1
        return original_read(path)

    monkeypatch.setattr(audit, "_read_bytes", counted)
    ledger = audit.new_access_ledger()
    selected = audit._read_selected_labels_once(grouped, ledger=ledger)

    assert len(selected) == 320
    assert raw_open_count == 20
    assert ledger["label_shard_hash_byte_opens"] == 20
    assert ledger["label_shard_npz_opens"] == 20
    assert ledger["registered_arrays_decompressed"] == 80
    assert ledger["materialized_label_rows"] == 320
    assert ledger["materialized_supervision_rows"] == 320
    assert ledger["selected_label_rows_read"] == 320
    assert ledger["selected_supervision_rows_read"] == 320
    assert len(ledger["per_shard_materialization"]) == 20
    assert ledger["unselected_row_values_inspected"] == 0
    assert ledger["unselected_row_metrics_computed"] == 0
    assert ledger["unselected_rows_retained"] == 0
    assert ledger["derivative_shard_or_cache_writes"] == 0
    assert all(ledger[name] == 0 for name in audit.FORBIDDEN_ACCESS_FIELDS)

    flattened = [record for records in grouped.values() for record in records]
    repeated = copy.deepcopy(flattened[0])
    repeated["global_row"] = 9999
    repeated["scene_id"] = "different_identity"
    with pytest.raises(ValueError, match="shard row/side"):
        audit._label_shard_manifest(
            [*flattened, repeated],
            spec=audit.AuditSpec(expected_shards=20, expected_frames=321),
            ledger=audit.new_access_ledger(),
        )


def test_full_shard_arrays_are_released_before_next_shard_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    paths = [tmp_path / "first.npz", tmp_path / "second.npz"]
    raw_by_path = {path: f"raw:{path.name}".encode("ascii") for path in paths}
    grouped = {}
    for index, path in enumerate(paths):
        digest = hashlib.sha256(raw_by_path[path]).hexdigest()
        grouped[path] = [
            _record(
                tmp_path,
                index=index,
                shard=path,
                shard_sha256=digest,
                row=0,
                side="current",
            )
        ]

    materialized: list[weakref.ReferenceType[np.ndarray]] = []
    opened = 0

    class FakeArchive:
        files = list(audit.REGISTERED_SHARD_ARRAY_NAMES)

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def __getitem__(self, name: str) -> np.ndarray:
            dtype = bool if name.endswith("supervision_mask") else np.uint8
            value = np.ones((1, 64, 64), dtype=dtype)
            materialized.append(weakref.ref(value))
            return value

    def fake_read(path: Path) -> bytes:
        nonlocal opened
        if opened:
            gc.collect()
            assert all(reference() is None for reference in materialized)
            materialized.clear()
        opened += 1
        return raw_by_path[path]

    monkeypatch.setattr(audit, "_read_bytes", fake_read)
    monkeypatch.setattr(audit, "_validate_npz_archive_inventory", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(audit.np, "load", lambda *_args, **_kwargs: FakeArchive())

    selected = audit._read_selected_labels_once(
        grouped,
        ledger=audit.new_access_ledger(),
    )

    assert len(selected) == 2
    assert opened == 2


def test_tampered_label_shard_fails_before_npz_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    path = tmp_path / "shard.npz"
    target = np.zeros((1, 64, 64), dtype=np.uint8)
    mask = np.ones_like(target, dtype=bool)
    np.savez_compressed(path, **_synthetic_shard_arrays(target, target, mask))
    record = _record(
        tmp_path,
        index=0,
        shard=path,
        shard_sha256="f" * 64,
        row=0,
        side="current",
    )
    ledger = audit.new_access_ledger()

    with pytest.raises(ValueError, match="SHA-256"):
        audit._read_selected_labels_once({path: [record]}, ledger=ledger)

    assert ledger["label_shard_hash_byte_opens"] == 1
    assert ledger["label_shard_npz_opens"] == 0
    assert ledger["selected_label_rows_read"] == 0

    raw = bytearray(path.read_bytes())
    local_offset = raw.index(b"PK\x03\x04")
    central_offset = raw.index(b"PK\x01\x02")
    local_flags = int.from_bytes(raw[local_offset + 6 : local_offset + 8], "little") | 1
    central_flags = int.from_bytes(raw[central_offset + 8 : central_offset + 10], "little") | 1
    raw[local_offset + 6 : local_offset + 8] = local_flags.to_bytes(2, "little")
    raw[central_offset + 8 : central_offset + 10] = central_flags.to_bytes(2, "little")
    path.write_bytes(raw)
    encrypted_digest = audit._hash_file(path)
    encrypted_record = {**record, "label_shard_sha256": encrypted_digest}
    with pytest.raises(ValueError, match="encrypted"):
        audit._read_selected_labels_once(
            {path: [encrypted_record]}, ledger=audit.new_access_ledger()
        )


def test_unallowlisted_geometry_fails_before_byte_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    path = tmp_path / "geometry.json"
    path.write_text("{}")
    ledger = audit.new_access_ledger()
    monkeypatch.setattr(
        audit,
        "_hash_file",
        lambda _path: pytest.fail("unallowlisted path reached byte hashing"),
    )

    with pytest.raises(PermissionError, match="not allowlisted"):
        audit._read_allowlisted_json(
            path,
            audit._hash_file.__name__.ljust(64, "0")[:64],
            allowlist={},
            ledger=ledger,
            label="synthetic geometry",
            requested_role="physical_geometry_contract",
        )

    assert ledger["unexpected_path_attempts"] == 1
    assert ledger["source_geometry_hash_byte_opens"] == 0
    assert ledger["source_geometry_json_parses"] == 0


def test_allowlisted_json_has_two_hash_opens_and_one_separate_parse_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    path = tmp_path / "geometry.json"
    path.write_text(json.dumps({"value": 3}))
    digest = audit._hash_file(path)
    ledger = audit.new_access_ledger()

    value = audit._read_allowlisted_json(
        path,
        digest,
        allowlist={path.resolve(): digest},
        ledger=ledger,
        label="synthetic geometry",
        requested_role="physical_geometry_contract",
    )

    assert value == {"value": 3}
    assert ledger["source_geometry_hash_byte_opens"] == 2
    assert ledger["source_geometry_json_parses"] == 1


def test_source_frame_scanner_selects_only_requested_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    path = tmp_path / "frames.jsonl"
    requested = _source_frame()
    unrequested = {"frame_index": 99, "env_index": 0, "timestamp_ns": 99}
    path.write_text("".join(json.dumps(value) + "\n" for value in (unrequested, requested)))
    digest = audit._hash_file(path)
    record = {
        "family": FAMILIES[0],
        "scene_id": "scene",
        "global_row": 1,
        "side": "current",
        "image_sha256": "a" * 64,
        "label_shard_sha256": "b" * 64,
        "label_row": 0,
        "frame_index": 7,
        "env_index": 0,
        "timestamp_ns": 11,
        "episode_id": "3",
        "reset_count": 0,
        "episode_step": 4,
    }
    ledger = audit.new_access_ledger()

    found = audit._scan_allowlisted_frames(
        path,
        digest,
        [record],
        allowlist={path.resolve(): digest},
        ledger=ledger,
        expected_rendered_timestamps={(7, 0): 11},
        plan_camera_mount_body=audit.NOMINAL_CAMERA_MOUNT_BODY,
    )

    assert len(found) == 1
    assert ledger["source_geometry_jsonl_records"] == 2
    assert ledger["source_frame_records_selected"] == 1
    assert found[tuple(audit._frame_identity_values(record))]["camera_pose_world"][
        "position"
    ] == requested["camera_pose_world"]["position"]

    for invalid_text, message in (
        ("\n" + json.dumps(requested) + "\n", "blank record"),
        (json.dumps(requested), "terminal newline"),
        (
            '{"frame_index":99,"frame_index":98,"env_index":0,'
            '"timestamp_ns":99}\n' + json.dumps(requested) + "\n",
            "duplicate JSON key",
        ),
        (
            '{"frame_index":NaN,"env_index":0,"timestamp_ns":99}\n'
            + json.dumps(requested)
            + "\n",
            "forbidden JSON constant",
        ),
        (
            json.dumps(
                {"frame_index": False, "env_index": 0, "timestamp_ns": 99}
            )
            + "\n"
            + json.dumps(requested)
            + "\n",
            "strict integer",
        ),
        (
            json.dumps({"env_index": 0, "timestamp_ns": 99})
            + "\n"
            + json.dumps(requested)
            + "\n",
            "strict integer",
        ),
    ):
        path.write_text(invalid_text)
        invalid_digest = audit._hash_file(path)
        with pytest.raises(ValueError, match=message):
            audit._scan_allowlisted_frames(
                path,
                invalid_digest,
                [record],
                allowlist={path.resolve(): invalid_digest},
                ledger=audit.new_access_ledger(),
                expected_rendered_timestamps={(7, 0): 11},
                plan_camera_mount_body=audit.NOMINAL_CAMERA_MOUNT_BODY,
            )


def test_source_geometry_chain_is_committed_before_parse_and_never_opens_rgb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    scene_id = "synthetic_scene"
    render_root = tmp_path / "renders"
    scene_root = render_root / "scene_committed"
    rgb_root = scene_root / "rgb"
    rgb_root.mkdir(parents=True)
    image_path = rgb_root / "frame.png"  # Deliberately never created.

    geometry_payload = {
        "schema": "lewm_go2_generalization_geometry_v2",
        "camera": {"horizontal_fov_deg": 78.323, "near_m": 0.05},
        "configuration_space": {
            "oracle_cell_size_m": 0.05,
            "landmarks_are_obstacles": True,
            "distractors_are_obstacles": True,
        },
    }
    geometry_path = tmp_path / "geometry.json"
    geometry_path.write_text(json.dumps(geometry_payload))
    geometry_file_sha = audit._hash_file(geometry_path)
    geometry_semantic_sha = canonical_json_sha256(geometry_payload)

    manifest = _empty_manifest(scene_id)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict()))
    frames_path = tmp_path / "frames.jsonl"
    source_frame = _source_frame()
    extra_source_frame = copy.deepcopy(source_frame)
    extra_source_frame["frame_index"] = 99
    extra_source_frame["timestamp_ns"] = 99
    extra_source_frame["episode"]["episode_step"] = 5
    outside_selection_frame = copy.deepcopy(source_frame)
    outside_selection_frame["frame_index"] = 123
    outside_selection_frame["timestamp_ns"] = 123
    outside_selection_frame["episode"]["episode_step"] = 6
    frames_path.write_text(
        json.dumps(source_frame)
        + "\n"
        + json.dumps(extra_source_frame)
        + "\n"
        + json.dumps(outside_selection_frame)
        + "\n"
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
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
                    "mount_body": copy.deepcopy(audit.NOMINAL_CAMERA_MOUNT_BODY),
                },
            }
        )
    )
    renderer_path = tmp_path / "renderer.py"
    renderer_path.write_text("# committed synthetic renderer\n")
    render_records = labels_v3._render_object_records(manifest)
    frame_keys = [[7, 0], [99, 0]]
    selection_core = {
        "schema": "lewm_go2_selected_render_frames_v1",
        "scene_id": scene_id,
        "scene_id_sha256": hashlib.sha256(scene_id.encode()).hexdigest(),
        "dataset_role": "train",
        "row_count": 1,
        "frame_count": 2,
        "frame_keys": frame_keys,
        "frame_key_set_sha256": canonical_json_sha256(frame_keys),
        "source_rows": {"path": str(tmp_path / "rows.jsonl"), "sha256": "f" * 64},
        "g2_images_opened": False,
        "g2_label_shards_opened": False,
    }
    selection = {
        **selection_core,
        "content_sha256": canonical_json_sha256(selection_core),
    }
    selection_path = tmp_path / "frame_selection.json"
    selection_path.write_text(json.dumps(selection))
    normalized_rendered = [
        {
            "frame_index": 7,
            "env_index": 0,
            "timestamp_ns": 11,
            "image_sha256": "a" * 64,
        },
        {
            "frame_index": 99,
            "env_index": 0,
            "timestamp_ns": 99,
            "image_sha256": "c" * 64,
        },
    ]
    object_ids = sorted(str(record["object_id"]) for record in render_records)
    summary = {
        "schema": "lewm_rendered_vision_v04",
        "render_status": "complete",
        "scene_id": scene_id,
        "family": FAMILIES[0],
        "g2_model_outputs_opened": False,
        "frame_count": 2,
        "frame_selection": {
            "path": str(selection_path),
            "sha256": audit._hash_file(selection_path),
            "frame_key_set_sha256": canonical_json_sha256(frame_keys),
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
        "rendered_frames": normalized_rendered,
        "rendered_image_set_sha256": canonical_json_sha256(normalized_rendered),
        "object_parity": {
            "schema": "lewm_render_object_parity_v1",
            "rendered_groups": ["wall", "obstacle", "landmark", "distractor"],
            "rendered_object_count": len(render_records),
            "rendered_object_ids": object_ids,
            "rendered_object_ids_sha256": canonical_json_sha256(object_ids),
            "rendered_object_records_sha256": canonical_json_sha256(render_records),
            "collision_distractors_rendered": True,
            "full_box_roll_pitch_yaw_rendered": True,
        },
        "source": {
            "plan": {"path": str(plan_path), "sha256": audit._hash_file(plan_path)},
            "frames_jsonl": {
                "path": str(frames_path),
                "sha256": audit._hash_file(frames_path),
            },
            "scene_manifest": {
                "path": str(manifest_path),
                "sha256": audit._hash_file(manifest_path),
            },
            "renderer_source": {
                "path": str(renderer_path),
                "sha256": audit._hash_file(renderer_path),
            },
        },
    }
    summary_path = scene_root / "summary.json"
    summary_path.write_text(json.dumps(summary))
    summary_sha = audit._hash_file(summary_path)
    render_audit_core = {
        "schema": "lewm_go2_selected_render_audit_v1",
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
    render_audit = {
        **render_audit_core,
        "content_sha256": canonical_json_sha256(render_audit_core),
    }
    render_audit_path = tmp_path / "render_audit.json"
    render_audit_path.write_text(json.dumps(render_audit))
    record = {
        "family": FAMILIES[0],
        "scene_id": scene_id,
        "global_row": 1,
        "side": "current",
        "image_path_metadata_only": str(image_path),
        "image_sha256": "a" * 64,
        "label_shard_path": str(tmp_path / "labels.npz"),
        "label_shard_sha256": "b" * 64,
        "label_row": 0,
        "frame_index": 7,
        "env_index": 0,
        "timestamp_ns": 11,
        "episode_id": "3",
        "reset_count": 0,
        "episode_step": 4,
    }
    spec = audit.AuditSpec(
        summary_root=render_root,
        summary_sha256={"scene_committed/summary.json": summary_sha},
        expected_transitions=1,
        expected_frames=1,
        expected_shards=1,
    )
    ledger = audit.new_access_ledger()
    panel_inputs = {
        "geometry_contract": {
            "path": str(geometry_path),
            "file_sha256": geometry_file_sha,
            "semantic_sha256": geometry_semantic_sha,
        },
        "render_audit_contract": {
            "path": str(render_audit_path),
            "file_sha256": audit._hash_file(render_audit_path),
            "content_sha256": render_audit["content_sha256"],
        },
    }

    frames, scenes, geometry, source_entries = audit._read_source_geometry(
        [record],
        panel_inputs,
        spec=spec,
        ledger=ledger,
    )

    assert len(frames) == 1
    assert set(scenes) == {scene_id}
    assert geometry["semantic_sha256"] == geometry_semantic_sha
    assert len(source_entries) == 8
    assert ledger["source_frame_records_selected"] == 1
    assert ledger["source_geometry_jsonl_records"] == 3
    assert ledger["source_geometry_hash_byte_opens"] == 16
    assert ledger["source_geometry_json_parses"] == 7
    assert ledger["rgb_byte_opens"] == ledger["rgb_decodes"] == 0
    assert not image_path.exists()
    # Legacy source split is retained as provenance and never selects the row.
    assert source_frame["episode"]["split"] == "test_hard"
    assert scenes[scene_id]["required_provenance_missing_count"] == 0
    assert scenes[scene_id]["required_provenance_nonunique_count"] == 0

    synthetic_paths = {
        "geometry": geometry_path,
        "manifest": manifest_path,
        "frames": frames_path,
        "plan": plan_path,
        "renderer": renderer_path,
        "selection": selection_path,
        "summary": summary_path,
        "render_audit": render_audit_path,
    }
    frozen_bytes = {name: path.read_bytes() for name, path in synthetic_paths.items()}

    def read_object(name: str) -> dict:
        return json.loads(frozen_bytes[name].decode("utf-8"))

    def write_object(path: Path, value: dict, *, embedded_hash: bool = False) -> None:
        if embedded_hash:
            core = dict(value)
            core.pop("content_sha256", None)
            value["content_sha256"] = canonical_json_sha256(core)
        path.write_text(json.dumps(value))

    def recommit_summary(value: dict) -> audit.AuditSpec:
        write_object(summary_path, value)
        return audit.AuditSpec(
            summary_root=render_root,
            summary_sha256={"scene_committed/summary.json": audit._hash_file(summary_path)},
            expected_transitions=1,
            expected_frames=1,
            expected_shards=1,
        )

    def exercise_mutation(
        target: str,
        mutate,
        expected: str,
        *,
        outcome: str = "raise",
    ) -> None:
        for name, path in synthetic_paths.items():
            path.write_bytes(frozen_bytes[name])
        inputs = copy.deepcopy(panel_inputs)
        mutated_summary = read_object("summary")
        if target == "summary":
            mutate(mutated_summary)
        elif target == "selection":
            value = read_object("selection")
            mutate(value)
            write_object(selection_path, value, embedded_hash=True)
            mutated_summary["frame_selection"]["sha256"] = audit._hash_file(selection_path)
        elif target == "plan":
            value = read_object("plan")
            mutate(value)
            write_object(plan_path, value)
            mutated_summary["source"]["plan"]["sha256"] = audit._hash_file(plan_path)
        elif target == "frames":
            values = [json.loads(line) for line in frozen_bytes["frames"].decode().splitlines()]
            mutate(values)
            frames_path.write_text("".join(json.dumps(value) + "\n" for value in values))
            mutated_summary["source"]["frames_jsonl"]["sha256"] = audit._hash_file(frames_path)
        elif target == "manifest":
            value = read_object("manifest")
            mutate(value)
            write_object(manifest_path, value)
            mutated_summary["source"]["scene_manifest"]["sha256"] = audit._hash_file(manifest_path)
        elif target == "geometry":
            value = read_object("geometry")
            mutate(value)
            write_object(geometry_path, value)
            inputs["geometry_contract"]["file_sha256"] = audit._hash_file(geometry_path)
            inputs["geometry_contract"]["semantic_sha256"] = canonical_json_sha256(value)
        elif target == "summary_raw_duplicate":
            raw = frozen_bytes["summary"].decode("utf-8")
            summary_path.write_text('{"schema":"duplicate",' + raw[1:])
        else:
            raise AssertionError(target)
        mutated_spec = recommit_summary(mutated_summary) if target != "summary_raw_duplicate" else audit.AuditSpec(
            summary_root=render_root,
            summary_sha256={"scene_committed/summary.json": audit._hash_file(summary_path)},
            expected_transitions=1,
            expected_frames=1,
            expected_shards=1,
        )
        if outcome == "raise":
            with pytest.raises((ValueError, PermissionError, OSError), match=expected):
                audit._read_source_geometry(
                    [record], inputs, spec=mutated_spec, ledger=audit.new_access_ledger()
                )
            return
        _frames, mutated_scenes, _geometry, _entries = audit._read_source_geometry(
            [record], inputs, spec=mutated_spec, ledger=audit.new_access_ledger()
        )
        if outcome == "parity":
            assert mutated_scenes[scene_id]["required_provenance_missing_count"] == 1
        elif outcome == "camera":
            evidence = next(iter(_frames.values()))["camera_mount_composition"]
            assert evidence["passes"] is False
        else:
            raise AssertionError(outcome)

    def rendered_field(name: str, value: object):
        def mutation(payload: dict) -> None:
            payload["rendered_frames"][0][name] = value
            payload["rendered_image_set_sha256"] = canonical_json_sha256(
                payload["rendered_frames"]
            )
        return mutation

    def nonfit_rendered_field(name: str, value: object):
        def mutation(payload: dict) -> None:
            payload["rendered_frames"][1][name] = value
            payload["rendered_image_set_sha256"] = canonical_json_sha256(
                payload["rendered_frames"]
            )

        return mutation

    def parity_field(name: str, value: object):
        return lambda payload: payload["object_parity"].__setitem__(name, value)

    def manifest_box_mutation(name: str, value: object):
        def mutation(payload: dict) -> None:
            box = {
                "object_id": "synthetic_obstacle",
                "kind": "box",
                "center_xyz_m": [0.0, 0.0, 0.5],
                "size_xyz_m": [0.2, 0.2, 1.0],
                "yaw_rad": 0.0,
                "material_id": "wall",
            }
            if name in {"center_xyz_m", "size_xyz_m"}:
                box[name][0] = value
            else:
                box[name] = value
            payload["obstacles"] = [box]

        return mutation

    mutations = (
        ("summary", lambda value: value.__setitem__("schema", "wrong"), "identity/status", "raise"),
        ("summary", lambda value: value.__setitem__("render_status", "partial"), "identity/status", "raise"),
        ("summary", lambda value: value.__setitem__("scene_id", "other"), "identity/status", "raise"),
        ("summary", lambda value: value.__setitem__("family", FAMILIES[1]), "identity/status", "raise"),
        ("summary", lambda value: value.__setitem__("resolution_wh", [225, 168]), "projection", "raise"),
        ("summary", lambda value: value.__setitem__("frame_count", False), "strict integer", "raise"),
        ("summary", lambda value: value["source"].pop("plan"), "source inventory", "raise"),
        ("summary", lambda value: value["source"].__setitem__("extra", {"path": "x", "sha256": "0" * 64}), "source inventory", "raise"),
        ("summary", lambda value: value["source"]["plan"].__setitem__("path", str(tmp_path / "other.json")), "other.json", "raise"),
        ("summary", lambda value: value["source"]["plan"].__setitem__("sha256", "0" * 64), "SHA-256", "raise"),
        ("summary_raw_duplicate", lambda _value: None, "duplicate JSON key", "raise"),
        ("selection", lambda value: value.__setitem__("schema", "wrong"), "selection contract", "raise"),
        ("selection", lambda value: value["frame_keys"].append([7, 0]), "canonical and unique", "raise"),
        ("selection", lambda value: value.__setitem__("frame_count", 3), "key-set commitment", "raise"),
        ("selection", lambda value: value.__setitem__("frame_count", False), "strict integer", "raise"),
        ("selection", lambda value: value.__setitem__("row_count", False), "strict integer", "raise"),
        ("selection", lambda value: value.__setitem__("frame_key_set_sha256", "0" * 64), "key-set commitment", "raise"),
        ("plan", lambda value: value.__setitem__("schema", "wrong"), "scene identity", "raise"),
        ("plan", lambda value: value.__setitem__("scene_id", "other"), "scene identity", "raise"),
        ("plan", lambda value: value["camera"].pop("native_resolution"), "camera contract", "raise"),
        ("plan", lambda value: value["camera"].pop("training_resolution"), "camera contract", "raise"),
        ("plan", lambda value: value["camera"].pop("encoding"), "camera contract", "raise"),
        ("plan", lambda value: value["camera"].__setitem__("extra", True), "camera contract", "raise"),
        ("plan", lambda value: value["camera"].__setitem__("fov_axis", "vertical"), "projection mismatch", "raise"),
        ("plan", lambda value: value["camera"].__setitem__("fov_deg", 79.0), "projection mismatch", "raise"),
        ("plan", lambda value: value["camera"].__setitem__("near_m", 0.06), "projection mismatch", "raise"),
        ("plan", lambda value: value["camera"].__setitem__("far_m", True), "strict number", "raise"),
        ("plan", lambda value: value["camera"].__setitem__("far_m", 201.0), "projection mismatch", "raise"),
        ("summary", lambda value: value["camera_projection"].pop("far_m"), "camera projection", "raise"),
        ("summary", lambda value: value["camera_projection"].__setitem__("extra", True), "camera projection", "raise"),
        ("summary", lambda value: value["camera_projection"].__setitem__("model", "fisheye"), "projection mismatch", "raise"),
        ("summary", lambda value: value["camera_projection"].__setitem__("renderer_fov_axis", "horizontal"), "projection mismatch", "raise"),
        ("summary", lambda value: value["camera_projection"].__setitem__("runtime_rectification_required", True), "projection mismatch", "raise"),
        ("summary", lambda value: value["camera_projection"].__setitem__("horizontal_fov_deg", 79.0), "camera projection", "raise"),
        ("summary", lambda value: value["camera_projection"].__setitem__("vertical_fov_deg", 63.0), "projection", "raise"),
        ("summary", lambda value: value["camera_projection"].__setitem__("far_m", True), "strict number", "raise"),
        ("summary", rendered_field("env_index", False), "strict integer", "raise"),
        ("summary", lambda value: value["camera_projection"].__setitem__("near_m", 0.06), "camera projection", "raise"),
        ("summary", lambda value: value["camera_projection"].__setitem__("far_m", 0.05), "camera projection", "raise"),
        ("summary", rendered_field("timestamp_ns", 12), "commitment changed", "raise"),
        ("summary", rendered_field("image_sha256", "b" * 64), "commitment changed", "raise"),
        (
            "summary",
            nonfit_rendered_field("frame_index", 100),
            "rendered-frame set commitment changed",
            "raise",
        ),
        ("frames", lambda values: values.clear(), "did not match", "raise"),
        ("frames", lambda values: values.append(copy.deepcopy(values[0])), "repeats a planned", "raise"),
        ("frames", lambda values: values[0].__setitem__("timestamp_ns", 12), "timestamp disagrees", "raise"),
        ("frames", lambda values: values.append(copy.deepcopy(values[1])), "repeats a planned", "raise"),
        (
            "frames",
            lambda values: values.pop(1),
            "does not contain every selected render key",
            "raise",
        ),
        ("frames", lambda values: values[1].__setitem__("timestamp_ns", 100), "timestamp disagrees", "raise"),
        (
            "frames",
            lambda values: values[2].__setitem__("frame_index", False),
            "strict integer",
            "raise",
        ),
        ("frames", lambda values: values[0]["episode"].__setitem__("reset_count", False), "strict integer", "raise"),
        ("frames", lambda values: values[0]["episode"].__setitem__("episode_step", False), "strict integer", "raise"),
        ("manifest", lambda value: value.__setitem__("scene_id", "other"), "manifest identity", "raise"),
        ("manifest", lambda value: value.__setitem__("family", FAMILIES[1]), "manifest identity", "raise"),
        ("manifest", manifest_box_mutation("center_xyz_m", False), "strict number", "raise"),
        ("manifest", manifest_box_mutation("size_xyz_m", False), "strict number", "raise"),
        ("manifest", manifest_box_mutation("yaw_rad", False), "strict number", "raise"),
        ("manifest", manifest_box_mutation("object_id", 7), "object_id", "raise"),
        ("geometry", lambda value: value.__setitem__("schema", "wrong"), "schema changed", "raise"),
        ("summary", lambda value: value["object_parity"].pop("schema"), "", "parity"),
        ("summary", lambda value: value["object_parity"].__setitem__("extra", True), "", "parity"),
        ("summary", parity_field("schema", "wrong"), "", "parity"),
        ("summary", parity_field("rendered_groups", list(reversed(["wall", "obstacle", "landmark", "distractor"]))), "", "parity"),
        ("summary", parity_field("rendered_object_count", 1), "", "parity"),
        ("summary", parity_field("rendered_object_count", False), "strict integer", "raise"),
        ("summary", parity_field("rendered_object_ids", ["x", "x"]), "", "parity"),
        ("summary", parity_field("rendered_object_ids_sha256", "0" * 64), "", "parity"),
        ("summary", parity_field("rendered_object_records_sha256", "0" * 64), "", "parity"),
        ("summary", parity_field("collision_distractors_rendered", False), "", "parity"),
        ("summary", parity_field("full_box_roll_pitch_yaw_rendered", False), "", "parity"),
        ("plan", lambda value: value["camera"]["mount_body"]["xyz_body_m"].__setitem__(0, 0.3), "", "camera"),
        ("frames", lambda values: values[0]["camera_mount_body"]["xyz_body_m"].__setitem__(0, 0.3), "", "camera"),
        ("frames", lambda values: values[0]["camera_pose_world"]["position"].__setitem__(0, 0.3), "", "camera"),
        ("frames", lambda values: values[0].__setitem__("base_quat_world_xyzw", [0.0, 0.0, 0.0, 0.9]), "", "camera"),
        ("frames", lambda values: values[0]["base_pose_world"]["position"].pop("x"), "base position", "raise"),
        ("frames", lambda values: values[0]["base_pose_world"]["position"].__setitem__("extra", 0.0), "base position", "raise"),
        ("frames", lambda values: values[0]["base_rpy_rad"].pop("yaw"), "base_rpy_rad fields", "raise"),
        ("frames", lambda values: values[0]["base_rpy_rad"].__setitem__("heading", 0.0), "base_rpy_rad fields", "raise"),
    )
    for target, mutate, expected, outcome in mutations:
        exercise_mutation(target, mutate, expected, outcome=outcome)

    for name, path in synthetic_paths.items():
        path.write_bytes(frozen_bytes[name])


def test_box_matching_preserves_duplicates_and_reports_unmatched_geometry() -> None:
    common = BoxObject(
        object_id="a",
        kind="box",
        center_xyz_m=(1.0, 2.0, 0.5),
        size_xyz_m=(0.4, 0.5, 1.0),
        yaw_rad=0.2,
        material_id="m",
    )
    duplicate = BoxObject(**{**common.__dict__, "object_id": "b"})
    shifted = BoxObject(
        **{**common.__dict__, "object_id": "c", "center_xyz_m": (1.0 + 2e-12, 2.0, 0.5)}
    )

    report = audit._match_boxes((common, duplicate), (duplicate, shifted))

    assert len(report["matches"]) == 1
    assert len(report["unmatched_rendered_indices"]) == 1
    assert len(report["unmatched_collision_indices"]) == 1
    assert report["unmatched_rendered_boxes"] == [
        {
            "index": report["unmatched_rendered_indices"][0],
            "canonical_geometry": list(
                audit._box_geometry((common, duplicate)[report["unmatched_rendered_indices"][0]])
            ),
        }
    ]
    assert report["unmatched_collision_boxes"] == [
        {
            "index": report["unmatched_collision_indices"][0],
            "canonical_geometry": list(
                audit._box_geometry((duplicate, shifted)[report["unmatched_collision_indices"][0]])
            ),
        }
    ]
    assert report["matched_multiplicities"][0]["multiplicity"] == 1


def test_reconstruction_is_bit_exact_to_historical_v3_stages() -> None:
    manifest = _empty_manifest("scene")
    frame = _source_frame()
    grid = InflatedOccupancyGrid(
        manifest,
        cell_size_m=0.05,
        inflation_m=0.0,
        treat_landmarks_as_obstacles=True,
        treat_distractors_as_obstacles=True,
    )
    camera = labels_v3._camera_observation(
        frame,
        horizontal_fov_deg=78.323,
        near_m=0.05,
        vertical_fov_deg=62.8370386364,
        require_recorded_up=True,
        image_width_px=224,
        image_height_px=168,
        obstacle_ray_stride_px=2,
    )

    expected, supervision, _observed = labels_v3._observable_physical_raster_and_output_labels(
        grid,
        rendered_obstacle_boxes=(),
        collision_obstacle_boxes=(),
        base_xy_yaw=(0.0, 0.0, 0.0),
        camera=camera,
        local_grid=labels_v3.DEFAULT_LOCAL_GRID,
    )
    stages = audit._reconstruct_label_stages(
        manifest,
        frame,
        rendered_boxes=(),
        collision_boxes=(),
        geometry_flags=_geometry_flags(),
        camera_projection=_projection(),
    )

    assert np.array_equal(stages["final"], expected)
    assert np.array_equal(supervision, np.ones((64, 64), dtype=bool))
    assert np.array_equal(stages["collision_overlap"], np.zeros((64, 64), dtype=bool))

    roll = 0.4
    quaternion = [math.sin(roll / 2.0), 0.0, 0.0, math.cos(roll / 2.0)]
    yaw_only_pose = {
        "position": [0.326, 0.0, 0.393],
        "lookat": [1.326, 0.0, 0.393],
        "up": [0.0, 0.0, 1.0],
    }
    yaw_only = audit._camera_mount_composition_evidence(
        base_position_world=[0.0, 0.0, 0.35],
        base_quat_world_xyzw=quaternion,
        stored_base_yaw_rad=0.0,
        plan_camera_mount_body=audit.NOMINAL_CAMERA_MOUNT_BODY,
        frame_camera_mount_body=audit.NOMINAL_CAMERA_MOUNT_BODY,
        recorded_camera_pose_world=yaw_only_pose,
    )
    assert yaw_only["passes"] is False
    composed = audit._camera_mount_composition_evidence(
        base_position_world=[0.0, 0.0, 0.35],
        base_quat_world_xyzw=quaternion,
        stored_base_yaw_rad=0.0,
        plan_camera_mount_body=audit.NOMINAL_CAMERA_MOUNT_BODY,
        frame_camera_mount_body=audit.NOMINAL_CAMERA_MOUNT_BODY,
        recorded_camera_pose_world=yaw_only["expected_camera_pose_world"],
    )
    assert composed["passes"] is True
    assert composed["position_max_abs_residual_m"] == 0.0


def test_unrendered_collision_box_creates_veto_and_overlap_xor() -> None:
    manifest = _empty_manifest("scene")
    collision = BoxObject(
        object_id="hidden_collision",
        kind="box",
        center_xyz_m=(1.0, 0.0, 0.4),
        size_xyz_m=(0.4, 0.4, 0.8),
        yaw_rad=0.0,
        material_id="m",
    )
    stages = audit._reconstruct_label_stages(
        manifest,
        _source_frame(),
        rendered_boxes=(),
        collision_boxes=(collision,),
        geometry_flags=_geometry_flags(),
        camera_projection=_projection(),
    )
    veto = (
        (stages["final"] == 0)
        & (stages["pre_veto"] == 1)
        & stages["collision_overlap"]
    )

    assert np.count_nonzero(veto) > 0
    assert np.count_nonzero(stages["rendered_overlap"] ^ stages["collision_overlap"]) > 0
    assert audit._match_boxes((), (collision,))["unmatched_collision_indices"] == [0]


def test_atomic_exclusive_output_refuses_replacement(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    first = {"schema": "first", "content_sha256": "a" * 64}
    audit._atomic_write_json_exclusive(path, first)

    with pytest.raises(FileExistsError, match="already exists"):
        audit._atomic_write_json_exclusive(path, {"schema": "second"})

    assert json.loads(path.read_text()) == first


def test_synthetic_end_to_end_result_construction_and_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = []
    for family_index, family in enumerate(FAMILIES):
        for side_index, side in enumerate(ENDPOINT_SIDES):
            index = family_index * 2 + side_index
            records.append(
                {
                    "family": family,
                    "scene_id": f"scene_{family_index}",
                    "global_row": family_index,
                    "side": side,
                    "image_path_metadata_only": str(tmp_path / f"scene_{family_index}/rgb/{side}.png"),
                    "image_sha256": _digest(f"image:{index}"),
                    "label_shard_path": str(tmp_path / "labels.npz"),
                    "label_shard_sha256": "b" * 64,
                    "label_row": family_index,
                    "frame_index": index,
                    "env_index": 0,
                    "timestamp_ns": index + 1,
                    "episode_id": str(family_index),
                    "reset_count": 0,
                    "episode_step": side_index,
                }
            )
    target = np.zeros((64, 64), dtype=np.uint8)
    mask = np.ones((64, 64), dtype=bool)
    selected = {
        tuple(audit._frame_identity_values(record)): (target.copy(), mask.copy())
        for record in records
    }
    synthetic_source_frame = _source_frame()
    synthetic_source_frame["camera_mount_composition"] = (
        audit._camera_mount_composition_evidence(
            base_position_world=[0.0, 0.0, 0.35],
            base_quat_world_xyzw=synthetic_source_frame["base_quat_world_xyzw"],
            stored_base_yaw_rad=0.0,
            plan_camera_mount_body=audit.NOMINAL_CAMERA_MOUNT_BODY,
            frame_camera_mount_body=synthetic_source_frame["camera_mount_body"],
            recorded_camera_pose_world=synthetic_source_frame["camera_pose_world"],
        )
    )
    source_frames = {
        tuple(audit._frame_identity_values(record)): copy.deepcopy(synthetic_source_frame)
        for record in records
    }
    scenes = {
        f"scene_{index}": {
            "scene_id": f"scene_{index}",
            "family": family,
            "manifest": object(),
            "rendered_boxes": (),
            "collision_boxes": (),
            "physical_grid": None,
            "box_matching": {
                "matches": [],
                "unmatched_rendered_indices": [],
                "unmatched_collision_indices": [],
                "unmatched_rendered_boxes": [],
                "unmatched_collision_boxes": [],
                "matched_multiplicities": [],
            },
            "camera_projection": _projection(),
            "required_provenance_missing_count": 0,
            "required_provenance_nonunique_count": 0,
        }
        for index, family in enumerate(FAMILIES)
    }
    source_map = {
        name: {"path": str(tmp_path / name), "sha256": _digest(name)}
        for name in (
            "binding",
            "audit_core",
            "audit_core_test",
            "audit_runner",
            "audit_runner_test",
            "audit_finalizer",
            "audit_finalizer_test",
            "label_semantics",
            "geometry_contract_semantics",
            "scene_manifest_semantics",
            "planning_grid_semantics",
        )
    }
    source_map["binding"]["sha256"] = audit.EXECUTION_BINDING_SHA256
    manifest_sha = "c" * 64
    output = tmp_path / "result.json"
    spec = audit.AuditSpec(
        output_path=output,
        expected_transitions=5,
        expected_frames=10,
        expected_shards=1,
    )

    def fake_panel(_spec, ledger):
        ledger["panel_metadata_byte_opens"] += 1
        return records, {"geometry_contract": {"path": "g", "file_sha256": "d" * 64, "semantic_sha256": "e" * 64}}

    def fake_labels(_grouped, *, ledger):
        ledger["label_shard_hash_byte_opens"] += 1
        ledger["label_shard_npz_opens"] += 1
        ledger["registered_arrays_decompressed"] += 4
        ledger["materialized_label_rows"] += 10
        ledger["materialized_supervision_rows"] += 10
        ledger["selected_label_rows_read"] += 10
        ledger["selected_supervision_rows_read"] += 10
        return selected

    def fake_sources(_records, _geometry, *, spec, ledger, **_kwargs):
        del spec
        ledger["source_frame_records_selected"] += 10
        return (
            source_frames,
            scenes,
            {
                "path": "g",
                "file_sha256": "d" * 64,
                "semantic_sha256": "e" * 64,
                "flags": _geometry_flags(),
            },
            [
                {
                    "path": "geometry.json",
                    "sha256": "d" * 64,
                    "semantic_role": "physical_geometry_contract",
                    "scene_id": "scene_0",
                }
            ],
        )

    def fake_stages(_manifest, _frame, **_kwargs):
        zeros = np.zeros((64, 64), dtype=bool)
        coordinates = np.zeros((64, 64), dtype=np.float64)
        return {
            "pre_veto": target.copy(),
            "collision_overlap": zeros.copy(),
            "rendered_overlap": zeros.copy(),
            "final": target.copy(),
            "output_x": coordinates.copy(),
            "output_y": coordinates.copy(),
        }

    monkeypatch.setattr(audit, "_hash_file", lambda path: {
        audit.BINDING_PATH: audit.EXECUTION_BINDING_SHA256,
        audit.V4_REPORT_PATH: audit.V4_REPORT_SHA256,
        audit.KNOWN_BIAS_PROOF_PATH: audit.KNOWN_BIAS_PROOF_SHA256,
        audit.IMPLEMENTATION_MANIFEST_PATH: manifest_sha,
    }[Path(path)])
    monkeypatch.setattr(
        audit,
        "_source_hashes",
        lambda _paths, **_kwargs: copy.deepcopy(source_map),
    )
    monkeypatch.setattr(
        audit,
        "_load_machine_manifest",
        lambda *_args, **_kwargs: {
            "schema": audit.MACHINE_MANIFEST_SCHEMA,
            "content_sha256": "d" * 64,
            "human_implementation_manifest": {
                "path": str(audit.IMPLEMENTATION_MANIFEST_PATH),
                "file_sha256": manifest_sha,
            },
            "preparation_access_ledger": {"passes": True},
        },
    )
    monkeypatch.setattr(audit, "_verify_bound_document", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(audit, "_load_panel", fake_panel)
    monkeypatch.setattr(
        audit,
        "_label_shard_manifest",
        lambda _records, *, spec, ledger: ([{"path": "labels.npz", "sha256": "b" * 64}], {Path("labels.npz"): records}),
    )
    monkeypatch.setattr(audit, "_read_selected_labels_once", fake_labels)
    monkeypatch.setattr(audit, "_read_source_geometry", fake_sources)
    monkeypatch.setattr(audit, "_reconstruct_label_stages", fake_stages)
    monkeypatch.setattr(
        audit,
        "_overlap_for_boxes",
        lambda _stages, _frame, _boxes: np.zeros((64, 64), dtype=bool),
    )

    result = audit.run_authoritative_audit(
        authorization_sha256=audit.EXECUTION_BINDING_SHA256,
        machine_manifest_sha256=manifest_sha,
        spec=spec,
        synthetic_test_only=True,
    )

    assert output.exists()
    assert json.loads(output.read_text()) == result
    assert result["content_sha256"] == canonical_json_sha256(
        {key: value for key, value in result.items() if key != "content_sha256"}
    )
    assert result["scope"]["frame_count"] == 10
    assert len(result["frame_reports"]) == 10
    assert all("records" not in report["ray_sequences"] for report in result["frame_reports"])
    assert result["reconstruction"]["passes"] is True
    assert result["rendered_collision_target_ambiguity"] is False
    assert result["two_phase_access_reconciliation"]["passes"] is True
    assert result["authorization_decision"][
        "camera_frustum_representation_implementation_authorized"
    ] is True
    assert set(result["source_hashes"]) == set(source_map)
    assert all(value is False for value in result["licenses"].values())
