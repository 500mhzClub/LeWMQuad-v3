from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

import pytest

from lewm.datasets import go2_attitude_sidecar as sidecar


def _sha(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: Any) -> str:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")
    return _sha(encoded)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    encoded = b"".join(sidecar.canonical_json_bytes(row) + b"\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return _sha(encoded)


def _yaw_quaternion(yaw: float) -> list[float]:
    return [0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5)]


def _frame(
    *,
    source_scene_id: int,
    scene_manifest_sha256: str,
    source_split: str,
    frame_index: int,
    timestamp_ns: int,
    episode_step: int,
    yaw: float,
) -> dict[str, Any]:
    return {
        "frame_index": frame_index,
        "env_index": 0,
        "timestamp_ns": timestamp_ns,
        "episode": {
            "scene_id": source_scene_id,
            "episode_id": 1,
            "manifest_sha256": scene_manifest_sha256,
            "split": source_split,
            "reset_count": 0,
            "episode_step": episode_step,
        },
        "base_quat_world_xyzw": _yaw_quaternion(yaw),
        "base_rpy_rad": {"roll": 0.0, "pitch": 0.0, "yaw": yaw},
    }


def _rehash(value: dict[str, Any]) -> dict[str, Any]:
    core = dict(value)
    core.pop("content_sha256", None)
    return {**core, "content_sha256": sidecar.canonical_json_sha256(core)}


def _synthetic_inputs(
    root: Path,
    *,
    row_mutator: Callable[[list[dict[str, Any]]], None] | None = None,
    source_index_mutator: Callable[[list[dict[str, Any]]], None] | None = None,
    dataset_source_mutator: Callable[[list[dict[str, Any]]], None] | None = None,
) -> dict[str, Any]:
    roles = (
        "train",
        "train",
        "checkpoint_selection",
        "probability_calibration",
        "g2_evaluation",
    )
    binding_path = root / "binding.md"
    geometry_path = root / "dynamic_geometry.py"
    builder_path = root / "build_sidecar.py"
    library_path = root / "sidecar_library.py"
    test_path = root / "test_sidecar.py"
    implementation_manifest_path = root / "implementation_manifest.json"
    binding_sha = _write_json(binding_path, {"binding": "synthetic"})
    geometry_sha = _write_json(geometry_path, {"geometry": "synthetic"})
    _write_json(builder_path, {"builder": "synthetic"})
    _write_json(library_path, {"library": "synthetic"})
    _write_json(test_path, {"test": "synthetic"})
    source_map = {
        "binding": binding_path,
        "builder": builder_path,
        "dynamic_geometry": geometry_path,
        "implementation_manifest": implementation_manifest_path,
        "sidecar_library": library_path,
        "sidecar_test": test_path,
    }
    source_entries = [
        {
            "role": role,
            "path": str(source_map[role]),
            "sha256": sidecar.sha256_file(source_map[role]),
        }
        for role in sorted(sidecar.SIDECAR_PRECOMMITTED_SOURCE_ROLES)
    ]
    implementation_core = {
        "schema": sidecar.SIDECAR_IMPLEMENTATION_MANIFEST_SCHEMA,
        "binding": {"path": str(binding_path), "sha256": binding_sha},
        "sources": {
            "entries": source_entries,
            "entry_count": len(source_entries),
            "source_map_sha256": sidecar.canonical_json_sha256(source_entries),
        },
        "tests": {
            "command": sidecar.SIDECAR_IMPLEMENTATION_TEST_COMMAND,
            "passed": sidecar.SIDECAR_IMPLEMENTATION_TEST_COUNT,
        },
        "resource_policy": {
            "cpu_workers_max": 6,
            "thread_environment": {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            },
            "neural_device": "none_metadata_only",
            "igpu": "forbidden",
        },
    }
    implementation_manifest_sha = _write_json(
        implementation_manifest_path,
        {
            **implementation_core,
            "content_sha256": sidecar.canonical_json_sha256(implementation_core),
        },
    )

    assignments: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    indexed_sources: list[dict[str, Any]] = []
    for global_row, role in enumerate(roles):
        scene_id = f"scene-{global_row:02d}"
        source_scene_id = 1000 + global_row
        scene_manifest_sha = _sha(f"scene-manifest-{global_row}")
        source_split = "synthetic_source"
        assignments[scene_id] = role
        frames_path = root / "frames" / f"{scene_id}.jsonl"
        frames = [
            _frame(
                source_scene_id=source_scene_id,
                scene_manifest_sha256=scene_manifest_sha,
                source_split=source_split,
                frame_index=0,
                timestamp_ns=1000 + global_row * 10,
                episode_step=0,
                yaw=0.0,
            ),
            _frame(
                source_scene_id=source_scene_id,
                scene_manifest_sha256=scene_manifest_sha,
                source_split=source_split,
                frame_index=1,
                timestamp_ns=2000 + global_row * 10,
                episode_step=1,
                yaw=0.1,
            ),
        ]
        frames_sha = _write_jsonl(frames_path, frames)
        indexed_sources.append(
            {
                "schema": sidecar.SOURCE_SCHEMA,
                "scene_id": scene_id,
                "scene_id_sha256": _sha(scene_id),
                "split": source_split,
                "frames_jsonl_path": str(frames_path.resolve()),
                "hashes": {
                    "frames_jsonl_file_sha256": frames_sha,
                    "scene_manifest_sha256": scene_manifest_sha,
                },
            }
        )
        sources.append(
            {
                "scene_id": scene_id,
                "dataset_role": role,
                "paths": {"frames_jsonl": str(frames_path.resolve())},
                "hashes": {
                    "frames_jsonl_sha256": frames_sha,
                    "scene_manifest_sha256": scene_manifest_sha,
                },
            }
        )
        rows.append(
            {
                "schema": sidecar.DATASET_ROW_SCHEMA,
                "global_row": global_row,
                "scene_id": scene_id,
                "dataset_role": role,
                "env_index": 0,
                "episode_id": "1",
                "reset_count": 0,
                "current_episode_step": 0,
                "next_episode_step": 1,
                "current_frame_index": 0,
                "next_frame_index": 1,
                "current_timestamp_ns": 1000 + global_row * 10,
                "next_timestamp_ns": 2000 + global_row * 10,
                "frames_jsonl_sha256": frames_sha,
                "source_split": source_split,
                "scene_manifest_sha256": scene_manifest_sha,
                "label_shard_row": global_row,
                "label_shard_sha256": _sha(f"label-{global_row}"),
                "current_image_sha256": _sha(f"current-{global_row}"),
                "next_image_sha256": _sha(f"next-{global_row}"),
            }
        )
    if row_mutator is not None:
        row_mutator(rows)
    if source_index_mutator is not None:
        source_index_mutator(indexed_sources)
    if dataset_source_mutator is not None:
        dataset_source_mutator(sources)

    rows_path = root / "dataset" / "rows.jsonl"
    rows_sha = _write_jsonl(rows_path, rows)
    source_index_path = root / "source_index.jsonl"
    source_index_sha = _write_jsonl(source_index_path, indexed_sources)
    audit_core = {
        "schema": sidecar.RENDER_AUDIT_SCHEMA,
        "output_source_index": {
            "path": str(source_index_path.resolve()),
            "sha256": source_index_sha,
        },
    }
    render_audit_path = root / "render_audit.json"
    render_audit_sha = _write_json(
        render_audit_path,
        {**audit_core, "content_sha256": sidecar.canonical_json_sha256(audit_core)},
    )
    role_counts = {role: roles.count(role) for role in sidecar.DATASET_ROLES}
    assignment_sha = sidecar.canonical_json_sha256(assignments)
    dataset_manifest = {
        "schema": sidecar.DATASET_SCHEMA,
        "row_count": len(rows),
        "index": {"path": str(rows_path.resolve()), "sha256": rows_sha},
        "scene_roles": {
            "assignments": assignments,
            "assignments_sha256": assignment_sha,
            "row_counts": role_counts,
        },
        "sources": sources,
    }
    dataset_manifest_path = root / "dataset" / "dataset_manifest.json"
    dataset_manifest_sha = _write_json(dataset_manifest_path, dataset_manifest)
    contract = sidecar.AttitudeSidecarBuildContract(
        dataset_manifest_sha256=dataset_manifest_sha,
        dataset_rows_sha256=rows_sha,
        source_index_sha256=source_index_sha,
        render_audit_sha256=render_audit_sha,
        dynamic_geometry_sha256=geometry_sha,
        role_assignment_sha256=assignment_sha,
        role_counts=role_counts,
        binding_path=binding_path,
        binding_sha256=binding_sha,
        source_map_paths=source_map,
    )
    return {
        "dataset_manifest_path": dataset_manifest_path,
        "source_index_path": source_index_path,
        "render_audit_path": render_audit_path,
        "dynamic_geometry_path": geometry_path,
        "source_map": source_map,
        "contract": contract,
        "implementation_manifest_path": implementation_manifest_path,
        "implementation_manifest_sha256": implementation_manifest_sha,
        "rows": rows,
        "frames": [Path(row["frames_jsonl_path"]) for row in indexed_sources],
    }


def _build(inputs: dict[str, Any], output: Path, *, workers: int = 1) -> dict[str, Any]:
    return sidecar.build_attitude_sidecar(
        dataset_manifest_path=inputs["dataset_manifest_path"],
        source_index_path=inputs["source_index_path"],
        render_audit_path=inputs["render_audit_path"],
        dynamic_geometry_path=inputs["dynamic_geometry_path"],
        output_dir=output,
        source_map=inputs["source_map"],
        implementation_manifest_path=inputs["implementation_manifest_path"],
        expected_implementation_manifest_sha256=inputs[
            "implementation_manifest_sha256"
        ],
        contract=inputs["contract"],
        workers=workers,
    )


def _mutated_scene_task(
    tmp_path: Path,
    *,
    mutation: str,
) -> sidecar._SceneTask:
    inputs = _synthetic_inputs(tmp_path / "inputs")
    row = copy.deepcopy(inputs["rows"][0])
    frames_path = inputs["frames"][0]
    frames = [json.loads(line) for line in frames_path.read_text().splitlines()]
    if mutation == "bool_frame_scene":
        frames[0]["episode"]["scene_id"] = True
    elif mutation == "string_frame_scene":
        frames[0]["episode"]["scene_id"] = "1000"
    elif mutation == "mixed_frame_scene":
        frames[1]["episode"]["scene_id"] += 1
    elif mutation == "negative_frame_scene":
        for frame in frames:
            frame["episode"]["scene_id"] = -1
    elif mutation == "string_frame_episode":
        frames[0]["episode"]["episode_id"] = "1"
    elif mutation == "integer_row_episode":
        row["episode_id"] = 1
    elif mutation == "row_manifest_mismatch":
        row["scene_manifest_sha256"] = "0" * 64
    elif mutation == "row_split_mismatch":
        row["source_split"] = "foreign_split"
    else:
        raise AssertionError(f"unsupported mutation: {mutation}")
    frames_sha = _write_jsonl(frames_path, frames)
    row["frames_jsonl_sha256"] = frames_sha
    return sidecar._SceneTask(
        scene_id=row["scene_id"],
        dataset_role=row["dataset_role"],
        source_split="synthetic_source",
        scene_manifest_sha256=frames[0]["episode"]["manifest_sha256"],
        frames_path=str(frames_path),
        frames_sha256=frames_sha,
        rows=(row,),
    )


def _manifest_sha(output: Path) -> str:
    return sidecar.sha256_file(output / "manifest.json")


def _write_g2_attempt(
    output: Path,
    root: Path,
) -> tuple[Path, str, Path, str]:
    manifest_path = output / "manifest.json"
    manifest_sha = _manifest_sha(output)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint_path = root / "checkpoint.pt"
    checkpoint_path.write_bytes(b"frozen synthetic checkpoint")
    checkpoint_sha = sidecar.sha256_file(checkpoint_path)
    g2_entry = manifest["roles"]["g2_evaluation"]
    attempt_id = sidecar.g2_sidecar_attempt_id(
        sidecar_manifest_sha256=manifest_sha,
        dataset_manifest_sha256=manifest["dataset"]["manifest_sha256"],
        source_checkpoint_sha256=checkpoint_sha,
        g2_role_file_sha256=g2_entry["file_sha256"],
    )
    marker_path = sidecar.g2_sidecar_attempt_path(checkpoint_path, attempt_id)
    core = {
        "schema": sidecar.SIDECAR_G2_ATTEMPT_SCHEMA,
        "attempt_id_sha256": attempt_id,
        "attempt_marker_path": str(marker_path),
        "created_at_utc": "2026-07-11T12:00:00+00:00",
        "intent": "open_exact_untouched_g2_sidecar_once",
        "sidecar_manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha,
        },
        "dataset_manifest": {
            "path": manifest["dataset"]["manifest_path"],
            "sha256": manifest["dataset"]["manifest_sha256"],
        },
        "source_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha,
        },
        "g2_role": {
            "path": g2_entry["path"],
            "file_sha256": g2_entry["file_sha256"],
            "row_count": g2_entry["row_count"],
            "ordered_global_rows_sha256": g2_entry[
                "ordered_global_rows_sha256"
            ],
        },
        "status": "committed_before_g2_sidecar_open",
    }
    marker = {**core, "content_sha256": sidecar.canonical_json_sha256(core)}
    marker_sha = _write_json(marker_path, marker)
    return marker_path, marker_sha, checkpoint_path, checkpoint_sha


def _rewrite_role_and_manifest(
    output: Path, role: str, rows: list[dict[str, Any]]
) -> None:
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    role_path = output / sidecar.ROLE_FILE_NAMES[role]
    file_sha = _write_jsonl(role_path, rows)
    entry = manifest["roles"][role]
    entry.update(
        {
            "file_sha256": file_sha,
            "content_sha256": sidecar.canonical_json_sha256(rows),
            "row_count": len(rows),
            "ordered_identity_sha256": sidecar.canonical_json_sha256(
                [row["row_identity_sha256"] for row in rows]
            ),
            "ordered_global_rows_sha256": sidecar.canonical_json_sha256(
                [row["global_row"] for row in rows]
            ),
        }
    )
    manifest["role_assignment"]["row_counts"][role] = len(rows)
    manifest["dataset"]["row_count"] = sum(
        manifest["role_assignment"]["row_counts"].values()
    )
    _write_json(manifest_path, _rehash(manifest))


def test_builds_exact_role_sidecars_and_role_loader_isolated(tmp_path: Path) -> None:
    inputs = _synthetic_inputs(tmp_path / "inputs")
    output = tmp_path / "sidecar"
    manifest = _build(inputs, output)

    assert manifest["role_assignment"]["row_counts"] == {
        "train": 2,
        "checkpoint_selection": 1,
        "probability_calibration": 1,
        "g2_evaluation": 1,
    }
    assert manifest["access_ledger"]["image_byte_opens"] == 0
    assert manifest["access_ledger"]["label_shard_byte_opens"] == 0
    assert manifest["access_ledger"]["g2_distribution_summaries"] == 0
    for role in sidecar.DATASET_ROLES:
        assert manifest["roles"][role]["distribution_summary_emitted"] is False

    loaded = sidecar.load_attitude_sidecar_roles(
        output / "manifest.json",
        roles=("train", "checkpoint_selection", "probability_calibration"),
        expected_manifest_sha256=_manifest_sha(output),
        contract=inputs["contract"],
    )
    assert {role: len(rows) for role, rows in loaded.items()} == {
        "train": 2,
        "checkpoint_selection": 1,
        "probability_calibration": 1,
    }
    assert loaded["train"][0]["current"]["base_quat_world_xyzw"] == [
        0.0, 0.0, 0.0, 1.0
    ]
    with pytest.raises(sidecar.AttitudeSidecarAccessError, match="attempt marker"):
        sidecar.load_attitude_sidecar_roles(
            output / "manifest.json",
            roles=("g2_evaluation",),
            expected_manifest_sha256=_manifest_sha(output),
            contract=inputs["contract"],
        )


def test_g2_role_opens_only_after_precommitted_marker(tmp_path: Path) -> None:
    inputs = _synthetic_inputs(tmp_path / "inputs")
    output = tmp_path / "sidecar"
    _build(inputs, output)
    marker, marker_sha, checkpoint, checkpoint_sha = _write_g2_attempt(
        output, tmp_path
    )
    with pytest.raises(sidecar.AttitudeSidecarContractError, match="SHA-256 mismatch"):
        sidecar.load_attitude_sidecar_roles(
            output / "manifest.json",
            roles=("g2_evaluation",),
            expected_manifest_sha256=_manifest_sha(output),
            contract=inputs["contract"],
            g2_attempt_marker_path=marker,
            expected_g2_attempt_marker_sha256="0" * 64,
            g2_source_checkpoint_path=checkpoint,
            expected_g2_source_checkpoint_sha256=checkpoint_sha,
        )
    loaded = sidecar.load_attitude_sidecar_roles(
        output / "manifest.json",
        roles=("g2_evaluation",),
        expected_manifest_sha256=_manifest_sha(output),
        contract=inputs["contract"],
        g2_attempt_marker_path=marker,
        expected_g2_attempt_marker_sha256=marker_sha,
        g2_source_checkpoint_path=checkpoint,
        expected_g2_source_checkpoint_sha256=checkpoint_sha,
    )
    assert len(loaded["g2_evaluation"]) == 1
    attempt_id = json.loads(marker.read_text(encoding="utf-8"))[
        "attempt_id_sha256"
    ]
    assert sidecar.g2_sidecar_receipt_path(checkpoint, attempt_id).is_file()
    with pytest.raises(sidecar.AttitudeSidecarAccessError, match="already consumed"):
        sidecar.load_attitude_sidecar_roles(
            output / "manifest.json",
            roles=("g2_evaluation",),
            expected_manifest_sha256=_manifest_sha(output),
            contract=inputs["contract"],
            g2_attempt_marker_path=marker,
            expected_g2_attempt_marker_sha256=marker_sha,
            g2_source_checkpoint_path=checkpoint,
            expected_g2_source_checkpoint_sha256=checkpoint_sha,
        )


def test_g2_loader_rejects_arbitrary_hashed_file_as_attempt(tmp_path: Path) -> None:
    inputs = _synthetic_inputs(tmp_path / "inputs")
    output = tmp_path / "sidecar"
    _build(inputs, output)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    arbitrary = tmp_path / "arbitrary.json"
    arbitrary_sha = _write_json(arbitrary, {"status": "not-an-attempt"})

    with pytest.raises(sidecar.AttitudeSidecarContractError):
        sidecar.load_attitude_sidecar_roles(
            output / "manifest.json",
            roles=("g2_evaluation",),
            expected_manifest_sha256=_manifest_sha(output),
            contract=inputs["contract"],
            g2_attempt_marker_path=arbitrary,
            expected_g2_attempt_marker_sha256=arbitrary_sha,
            g2_source_checkpoint_path=checkpoint,
            expected_g2_source_checkpoint_sha256=sidecar.sha256_file(checkpoint),
        )


def test_g2_loader_rejects_copied_valid_marker_at_second_path(tmp_path: Path) -> None:
    inputs = _synthetic_inputs(tmp_path / "inputs")
    output = tmp_path / "sidecar"
    _build(inputs, output)
    marker, marker_sha, checkpoint, checkpoint_sha = _write_g2_attempt(
        output, tmp_path
    )
    copied = tmp_path / "copied_attempt.json"
    copied.write_bytes(marker.read_bytes())

    with pytest.raises(sidecar.AttitudeSidecarContractError, match="noncanonical"):
        sidecar.load_attitude_sidecar_roles(
            output / "manifest.json",
            roles=("g2_evaluation",),
            expected_manifest_sha256=_manifest_sha(output),
            contract=inputs["contract"],
            g2_attempt_marker_path=copied,
            expected_g2_attempt_marker_sha256=marker_sha,
            g2_source_checkpoint_path=checkpoint,
            expected_g2_source_checkpoint_sha256=checkpoint_sha,
        )


def test_g2_marker_nested_numeric_types_are_exact(tmp_path: Path) -> None:
    inputs = _synthetic_inputs(tmp_path / "inputs")
    output = tmp_path / "sidecar"
    _build(inputs, output)
    marker, _marker_sha, checkpoint, checkpoint_sha = _write_g2_attempt(
        output, tmp_path
    )
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["g2_role"]["row_count"] = 1.0
    marker_sha = _write_json(marker, _rehash(payload))

    with pytest.raises(sidecar.AttitudeSidecarContractError, match="exact integer"):
        sidecar.load_attitude_sidecar_roles(
            output / "manifest.json",
            roles=("g2_evaluation",),
            expected_manifest_sha256=_manifest_sha(output),
            contract=inputs["contract"],
            g2_attempt_marker_path=marker,
            expected_g2_attempt_marker_sha256=marker_sha,
            g2_source_checkpoint_path=checkpoint,
            expected_g2_source_checkpoint_sha256=checkpoint_sha,
        )


def test_development_loader_never_opens_corrupt_g2_file(tmp_path: Path) -> None:
    inputs = _synthetic_inputs(tmp_path / "inputs")
    output = tmp_path / "sidecar"
    _build(inputs, output)
    (output / sidecar.ROLE_FILE_NAMES["g2_evaluation"]).write_bytes(b"corrupt")
    loaded = sidecar.load_attitude_sidecar_roles(
        output / "manifest.json",
        roles=("train",),
        expected_manifest_sha256=_manifest_sha(output),
        contract=inputs["contract"],
    )
    assert len(loaded["train"]) == 2


def test_serial_and_six_worker_role_bytes_are_identical(tmp_path: Path) -> None:
    inputs = _synthetic_inputs(tmp_path / "inputs")
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    _build(inputs, serial, workers=1)
    _build(inputs, parallel, workers=6)
    for role in sidecar.DATASET_ROLES:
        assert (serial / sidecar.ROLE_FILE_NAMES[role]).read_bytes() == (
            parallel / sidecar.ROLE_FILE_NAMES[role]
        ).read_bytes()


def test_join_rejects_timestamp_mismatch(tmp_path: Path) -> None:
    def mutate(rows: list[dict[str, Any]]) -> None:
        rows[0]["current_timestamp_ns"] += 1

    inputs = _synthetic_inputs(tmp_path / "inputs", row_mutator=mutate)
    with pytest.raises(sidecar.AttitudeSidecarContractError, match="key/timestamp"):
        _build(inputs, tmp_path / "sidecar")


def test_join_rejects_changed_source_frames(tmp_path: Path) -> None:
    inputs = _synthetic_inputs(tmp_path / "inputs")
    inputs["frames"][0].write_text(
        inputs["frames"][0].read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(sidecar.AttitudeSidecarContractError, match="SHA-256 mismatch"):
        _build(inputs, tmp_path / "sidecar")


def test_join_rejects_reordered_dataset_rows(tmp_path: Path) -> None:
    def mutate(rows: list[dict[str, Any]]) -> None:
        rows[0], rows[1] = rows[1], rows[0]

    inputs = _synthetic_inputs(tmp_path / "inputs", row_mutator=mutate)
    with pytest.raises(sidecar.AttitudeSidecarContractError, match="reordered"):
        _build(inputs, tmp_path / "sidecar")


def test_join_rejects_noninjective_transition(tmp_path: Path) -> None:
    def mutate(rows: list[dict[str, Any]]) -> None:
        duplicate = dict(rows[0])
        duplicate["global_row"] = rows[1]["global_row"]
        duplicate["label_shard_row"] = rows[1]["label_shard_row"]
        duplicate["label_shard_sha256"] = rows[1]["label_shard_sha256"]
        duplicate["current_image_sha256"] = rows[1]["current_image_sha256"]
        duplicate["next_image_sha256"] = rows[1]["next_image_sha256"]
        duplicate["dataset_role"] = rows[1]["dataset_role"]
        rows[1] = duplicate

    inputs = _synthetic_inputs(tmp_path / "inputs", row_mutator=mutate)
    # The duplicated row now names scene-00 twice and leaves scene-01 empty.
    with pytest.raises(sidecar.AttitudeSidecarContractError):
        _build(inputs, tmp_path / "sidecar")


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("bool_frame_scene", "exact integer"),
        ("string_frame_scene", "exact integer"),
        ("mixed_frame_scene", "multiple numeric scene identities"),
        ("negative_frame_scene", "negative numeric scene identity"),
        ("string_frame_episode", "exact integer"),
        ("integer_row_episode", "nonempty string"),
        ("row_manifest_mismatch", "scene manifest mismatch"),
        ("row_split_mismatch", "source split mismatch"),
    ),
)
def test_production_join_types_and_provenance_fail_closed(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    task = _mutated_scene_task(tmp_path, mutation=mutation)
    with pytest.raises(sidecar.AttitudeSidecarContractError, match=message):
        sidecar._build_scene(task)


def test_source_index_scene_id_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    def mutate(rows: list[dict[str, Any]]) -> None:
        rows[0]["scene_id_sha256"] = "0" * 64

    inputs = _synthetic_inputs(
        tmp_path / "inputs", source_index_mutator=mutate
    )
    with pytest.raises(
        sidecar.AttitudeSidecarContractError,
        match="source index scene hash mismatch",
    ):
        _build(inputs, tmp_path / "sidecar")


def test_dataset_source_role_mismatch_fails_closed(tmp_path: Path) -> None:
    def mutate(sources: list[dict[str, Any]]) -> None:
        sources[0]["dataset_role"] = "g2_evaluation"

    inputs = _synthetic_inputs(
        tmp_path / "inputs", dataset_source_mutator=mutate
    )
    with pytest.raises(
        sidecar.AttitudeSidecarContractError,
        match="dataset source role assignment mismatch",
    ):
        _build(inputs, tmp_path / "sidecar")


def test_manifest_rejects_source_frame_scene_count_mismatch(tmp_path: Path) -> None:
    inputs = _synthetic_inputs(tmp_path / "inputs")
    output = tmp_path / "sidecar"
    manifest = _build(inputs, output)
    changed = copy.deepcopy(manifest)
    changed["source_index"]["scene_count"] += 1
    changed = _rehash(changed)

    with pytest.raises(
        sidecar.AttitudeSidecarContractError,
        match="source frame record count",
    ):
        sidecar.validate_attitude_sidecar_manifest(
            changed,
            manifest_path=output / "manifest.json",
            contract=inputs["contract"],
        )


def test_sidecar_row_rejects_bool_float_and_schema_mutations(tmp_path: Path) -> None:
    inputs = _synthetic_inputs(tmp_path / "inputs")
    output = tmp_path / "sidecar"
    _build(inputs, output)
    row = json.loads(
        (output / sidecar.ROLE_FILE_NAMES["train"])
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    for field, value in (
        ("global_row", True),
        ("global_row", 0.0),
        ("env_index", False),
        ("current_frame_index", 0.0),
    ):
        mutated = copy.deepcopy(row)
        mutated[field] = value
        mutated = _rehash(mutated)
        with pytest.raises(sidecar.AttitudeSidecarContractError):
            sidecar.validate_sidecar_row(mutated)
    mutated = copy.deepcopy(row)
    mutated["current"]["base_quat_world_xyzw"][0] = True
    mutated = _rehash(mutated)
    with pytest.raises(sidecar.AttitudeSidecarContractError, match="not bool"):
        sidecar.validate_sidecar_row(mutated)
    for mutation in (
        lambda value: value.update(schema="wrong"),
        lambda value: value.update(extra="field"),
        lambda value: value.pop("next_timestamp_ns"),
    ):
        mutated = copy.deepcopy(row)
        mutation(mutated)
        mutated = _rehash(mutated)
        with pytest.raises(sidecar.AttitudeSidecarContractError):
            sidecar.validate_sidecar_row(mutated)


def test_quaternion_norm_and_yaw_tolerance_boundaries(tmp_path: Path) -> None:
    inputs = _synthetic_inputs(tmp_path / "inputs")
    output = tmp_path / "sidecar"
    _build(inputs, output)
    row = json.loads(
        (output / sidecar.ROLE_FILE_NAMES["train"])
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    norm_pass = copy.deepcopy(row)
    norm_pass["current"]["base_quat_world_xyzw"] = [0, 0, 0, 1.000009]
    sidecar.validate_sidecar_row(_rehash(norm_pass))
    norm_fail = copy.deepcopy(row)
    norm_fail["current"]["base_quat_world_xyzw"] = [0, 0, 0, 1.000011]
    with pytest.raises(sidecar.AttitudeSidecarContractError, match="norm"):
        sidecar.validate_sidecar_row(_rehash(norm_fail))
    yaw_pass = copy.deepcopy(row)
    yaw_pass["current"]["stored_base_yaw_rad"] = 0.000009
    sidecar.validate_sidecar_row(_rehash(yaw_pass))
    yaw_fail = copy.deepcopy(row)
    yaw_fail["current"]["stored_base_yaw_rad"] = 0.000011
    with pytest.raises(sidecar.AttitudeSidecarContractError, match="yaw"):
        sidecar.validate_sidecar_row(_rehash(yaw_fail))


@pytest.mark.parametrize("mode", ["reordered", "duplicate"])
def test_role_loader_rejects_reorder_and_duplicate(tmp_path: Path, mode: str) -> None:
    inputs = _synthetic_inputs(tmp_path / "inputs")
    output = tmp_path / "sidecar"
    _build(inputs, output)
    rows = [
        json.loads(line)
        for line in (output / sidecar.ROLE_FILE_NAMES["train"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    mutated = list(reversed(rows)) if mode == "reordered" else [rows[0], rows[0]]
    _rewrite_role_and_manifest(output, "train", mutated)
    with pytest.raises(sidecar.AttitudeSidecarContractError):
        sidecar.load_attitude_sidecar_roles(
            output / "manifest.json",
            roles=("train",),
            expected_manifest_sha256=_manifest_sha(output),
            contract=inputs["contract"],
        )


def test_role_loader_rejects_noncanonical_jsonl(tmp_path: Path) -> None:
    inputs = _synthetic_inputs(tmp_path / "inputs")
    output = tmp_path / "sidecar"
    _build(inputs, output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    role_path = output / sidecar.ROLE_FILE_NAMES["train"]
    rows = [json.loads(line) for line in role_path.read_text().splitlines()]
    noncanonical = "\n".join(json.dumps(row, sort_keys=False) for row in rows) + "\n"
    role_path.write_text(noncanonical, encoding="utf-8")
    manifest["roles"]["train"]["file_sha256"] = _sha(noncanonical)
    for event in manifest["access_ledger"]["completed_role_write_events"]:
        if event["dataset_role"] == "train":
            event["file_sha256"] = _sha(noncanonical)
            event["byte_count"] = len(noncanonical.encode("utf-8"))
    _write_json(manifest_path, _rehash(manifest))
    with pytest.raises(sidecar.AttitudeSidecarContractError, match="not canonical"):
        sidecar.load_attitude_sidecar_roles(
            manifest_path,
            roles=("train",),
            expected_manifest_sha256=_manifest_sha(output),
            contract=inputs["contract"],
        )


def test_manifest_rejects_bool_and_float_mutations(tmp_path: Path) -> None:
    inputs = _synthetic_inputs(tmp_path / "inputs")
    output = tmp_path / "sidecar"
    _build(inputs, output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutations = (
        lambda value: value["construction"].update(workers=True),
        lambda value: value["dataset"].update(row_count=5.0),
        lambda value: value["roles"]["train"].update(
            distribution_summary_emitted=1
        ),
        lambda value: value["access_ledger"].update(image_byte_opens=False),
    )
    for mutation in mutations:
        changed = copy.deepcopy(manifest)
        mutation(changed)
        changed = _rehash(changed)
        with pytest.raises(sidecar.AttitudeSidecarContractError):
            sidecar.validate_attitude_sidecar_manifest(
                changed,
                manifest_path=manifest_path,
                contract=inputs["contract"],
            )


def test_manifest_rejects_tampered_completed_read_purpose(tmp_path: Path) -> None:
    inputs = _synthetic_inputs(tmp_path / "inputs")
    output = tmp_path / "sidecar"
    _build(inputs, output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed = copy.deepcopy(manifest)
    changed["access_ledger"]["completed_read_events"][0]["purposes"] = [
        "arbitrary"
    ]

    with pytest.raises(sidecar.AttitudeSidecarContractError, match="purposes"):
        sidecar.validate_attitude_sidecar_manifest(
            _rehash(changed),
            manifest_path=manifest_path,
            contract=inputs["contract"],
        )


def test_manifest_rejects_tampered_source_frame_byte_count(tmp_path: Path) -> None:
    inputs = _synthetic_inputs(tmp_path / "inputs")
    output = tmp_path / "sidecar"
    _build(inputs, output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_path = manifest["source_frames"][0]["path"]
    changed = copy.deepcopy(manifest)
    event = next(
        value
        for value in changed["access_ledger"]["completed_read_events"]
        if value["path"] == source_path
    )
    event["byte_count"] += 1
    event["total_bytes_read"] = event["byte_count"] * event["open_count"]

    with pytest.raises(sidecar.AttitudeSidecarContractError, match="byte count"):
        sidecar.validate_attitude_sidecar_manifest(
            _rehash(changed),
            manifest_path=manifest_path,
            contract=inputs["contract"],
        )


def test_existing_output_is_never_replaced(tmp_path: Path) -> None:
    inputs = _synthetic_inputs(tmp_path / "inputs")
    output = tmp_path / "sidecar"
    _build(inputs, output)
    original = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        _build(inputs, output)
    assert (output / "manifest.json").read_bytes() == original


def test_publication_never_overwrites_raced_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _synthetic_inputs(tmp_path / "inputs")
    output = tmp_path / "sidecar"
    original_link = sidecar.os.link
    injected = False

    def racing_link(source: Path, destination: str, **kwargs: Any) -> None:
        nonlocal injected
        if not injected:
            injected = True
            (output / destination).write_bytes(b"foreign raced content")
        original_link(source, destination, **kwargs)

    monkeypatch.setattr(sidecar.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        _build(inputs, output)
    assert (output / sidecar.ROLE_FILE_NAMES["train"]).read_bytes() == (
        b"foreign raced content"
    )
    assert not (output / "manifest.json").exists()


def test_publication_path_swap_preserves_replacement_and_cleans_only_owned_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _synthetic_inputs(tmp_path / "inputs")
    output = tmp_path / "sidecar"
    moved = tmp_path / "moved_original_directory"
    original_link = sidecar.os.link
    swapped = False

    def swapping_link(source: Path, destination: str, **kwargs: Any) -> None:
        nonlocal swapped
        if not swapped:
            swapped = True
            output.rename(moved)
            output.mkdir()
            (output / "foreign.txt").write_bytes(b"foreign replacement")
        original_link(source, destination, **kwargs)

    monkeypatch.setattr(sidecar.os, "link", swapping_link)
    with pytest.raises(sidecar.AttitudeSidecarContractError, match="identity changed"):
        _build(inputs, output)
    assert (output / "foreign.txt").read_bytes() == b"foreign replacement"
    assert list(moved.iterdir()) == []


def test_publication_rejects_staging_directory_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _synthetic_inputs(tmp_path / "inputs")
    output = tmp_path / "sidecar"
    original_publish = sidecar._publish_staged_no_replace
    moved_staging: Path | None = None

    def swapping_publish(**kwargs: Any) -> None:
        nonlocal moved_staging
        staging = kwargs["staging"]
        moved_staging = staging.with_name(staging.name + ".moved")
        staging.rename(moved_staging)
        staging.mkdir()
        (staging / "foreign.txt").write_bytes(b"foreign staging replacement")
        original_publish(**kwargs)

    monkeypatch.setattr(sidecar, "_publish_staged_no_replace", swapping_publish)
    with pytest.raises(sidecar.AttitudeSidecarContractError, match="identity changed"):
        _build(inputs, output)
    assert moved_staging is not None and moved_staging.is_dir()
    replacement_candidates = list(output.parent.glob(f".{output.name}.tmp.*"))
    assert any(
        (candidate / "foreign.txt").read_bytes() == b"foreign staging replacement"
        for candidate in replacement_candidates
        if (candidate / "foreign.txt").is_file()
    )
    assert not output.exists()


def test_builder_rejects_symlinked_authoritative_input(tmp_path: Path) -> None:
    inputs = _synthetic_inputs(tmp_path / "inputs")
    manifest_path = inputs["dataset_manifest_path"]
    alias = tmp_path / "dataset_manifest_alias.json"
    alias.symlink_to(manifest_path)
    inputs["dataset_manifest_path"] = alias

    with pytest.raises(
        sidecar.AttitudeSidecarContractError,
        match="aliased",
    ):
        _build(inputs, tmp_path / "sidecar")


def test_builder_rejects_incomplete_or_substituted_source_map(tmp_path: Path) -> None:
    inputs = _synthetic_inputs(tmp_path / "inputs")
    incomplete = dict(inputs["source_map"])
    incomplete.pop("sidecar_test")
    inputs["source_map"] = incomplete
    with pytest.raises(sidecar.AttitudeSidecarContractError, match="exactly"):
        _build(inputs, tmp_path / "incomplete")

    inputs = _synthetic_inputs(tmp_path / "second_inputs")
    substitute = tmp_path / "substitute.py"
    substitute.write_text("substitute", encoding="utf-8")
    inputs["source_map"]["sidecar_test"] = substitute
    with pytest.raises(sidecar.AttitudeSidecarContractError, match="differs"):
        _build(inputs, tmp_path / "substituted")


def test_implementation_manifest_rejects_source_tamper_before_data_access(
    tmp_path: Path,
) -> None:
    inputs = _synthetic_inputs(tmp_path / "inputs")
    inputs["source_map"]["sidecar_library"].write_text(
        "tampered after precommit", encoding="utf-8"
    )

    with pytest.raises(sidecar.AttitudeSidecarContractError, match="SHA-256 mismatch"):
        _build(inputs, tmp_path / "sidecar")


def test_implementation_manifest_is_rechecked_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _synthetic_inputs(tmp_path / "inputs")
    output = tmp_path / "sidecar"
    original_write = sidecar._write_json_exclusive

    def mutating_write(path: Path, value: dict[str, Any]) -> str:
        digest = original_write(path, value)
        if path.name == "manifest.json":
            inputs["source_map"]["sidecar_test"].write_text(
                "changed during construction", encoding="utf-8"
            )
        return digest

    monkeypatch.setattr(sidecar, "_write_json_exclusive", mutating_write)
    with pytest.raises(sidecar.AttitudeSidecarContractError, match="SHA-256 mismatch"):
        _build(inputs, output)
    assert not output.exists()


def test_loader_rejects_symlinked_role_file(tmp_path: Path) -> None:
    inputs = _synthetic_inputs(tmp_path / "inputs")
    output = tmp_path / "sidecar"
    _build(inputs, output)
    role_path = output / sidecar.ROLE_FILE_NAMES["train"]
    target = tmp_path / "copied_train.jsonl"
    target.write_bytes(role_path.read_bytes())
    role_path.unlink()
    role_path.symlink_to(target)

    with pytest.raises(
        sidecar.AttitudeSidecarContractError,
        match="aliased",
    ):
        sidecar.load_attitude_sidecar_roles(
            output / "manifest.json",
            roles=("train",),
            expected_manifest_sha256=_manifest_sha(output),
            contract=inputs["contract"],
        )
