from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

from lewm.datasets import go2_shared_jepa_v5_raw_supervision_auditor_v1 as auditor
from lewm.datasets import go2_shared_jepa_v5_raw_supervision_builder_v1 as frozen_builder
from lewm.datasets import go2_shared_jepa_v5_raw_supervision_plan_v5 as plan_v5
from scripts import audit_go2_shared_jepa_v5_raw_supervision_v1 as audit_cli


def _with_hash(core: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(auditor.canonical_json_bytes(core))
    return {**normalized, "content_sha256": auditor.canonical_json_sha256(normalized)}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(auditor.canonical_json_bytes(value) + b"\n")


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": str(path.relative_to(root)),
        "byte_count": len(payload),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _synthetic_fixture(tmp_path: Path) -> tuple[
    Path,
    str,
    auditor.AuditInputs,
    dict[str, tuple[np.ndarray, ...]],
]:
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "shards").mkdir()
    scene_id = "synthetic_scene"
    family = "synthetic_family"
    role = "train"
    scene_directory = hashlib.sha256(scene_id.encode()).hexdigest()[:16]
    shard_directory = root / "shards" / scene_directory
    shard_directory.mkdir()

    frame = auditor.raycast_v4.synthetic_scene_jobs(1)[0].frames[0]
    evidence = auditor.raycast_v4.build_frame_evidence_v4(frame)
    raster = auditor.raycast_v4.rasterize_observable_camera_ray_evidence_v4(evidence)
    source_arrays = auditor._stored_arrays_from_evidence(evidence, raster)
    image_sha = hashlib.sha256(b"synthetic-image").hexdigest()
    identity = {
        "dataset_role": role,
        "scene_id": scene_id,
        "episode_id": "episode_0",
        "env_index": 0,
        "episode_step": 0,
        "frame_index": 0,
        "timestamp_ns": 100,
        "image_sha256": image_sha,
    }
    identity_sha = auditor.canonical_json_sha256(identity)
    plan_endpoint = _with_hash(
        {
            "schema": plan_v5.ENDPOINT_SCHEMA,
            "identity": identity,
            "identity_sha256": identity_sha,
            "image_path_metadata_only": "/synthetic/rgb/frame.png",
            "frames_jsonl_sha256": hashlib.sha256(b"frames").hexdigest(),
            "scene_manifest_sha256": hashlib.sha256(b"manifest").hexdigest(),
            "base_quat_world_xyzw": [0.0, 0.0, 0.0, 1.0],
            "stored_base_yaw_rad": 0.0,
        }
    )
    pair = _with_hash(
        {
            "schema": plan_v5.PAIR_SCHEMA,
            "dataset_role": role,
            "global_row": 0,
            "scene_id": scene_id,
            "family": family,
            "episode_id": "episode_0",
            "env_index": 0,
            "reset_count": 0,
            "source_split": "synthetic",
            "frames_jsonl_sha256": hashlib.sha256(b"frames").hexdigest(),
            "scene_manifest_sha256": hashlib.sha256(b"manifest").hexdigest(),
            "primitive": "hold",
            "relative_se2_current_frame": [0.0, 0.0, 0.0],
            "current_endpoint_sha256": identity_sha,
            "next_endpoint_sha256": identity_sha,
            "label_shard_path_metadata_only": "/synthetic/labels.npz",
            "label_shard_sha256": hashlib.sha256(b"labels").hexdigest(),
            "label_shard_row": 0,
            "sidecar_row_identity_sha256": hashlib.sha256(b"sidecar").hexdigest(),
        }
    )
    plan = plan_v5.DevelopmentRawSupervisionPlan(
        value={}, pairs=(pair,), endpoints=(plan_endpoint,)
    )
    inventory = plan_v5.DevelopmentSourceInventory(
        records=(), hashes={}, access_ledger={}
    )

    shard_files: list[dict[str, Any]] = []
    for (name, dtype, shape), array in zip(auditor.ARRAY_LAYOUT, source_arrays):
        normalized = np.ascontiguousarray(array, dtype=np.dtype(dtype)).reshape((1, *shape))
        path = shard_directory / name
        path.write_bytes(normalized.tobytes(order="C"))
        shard_files.append(
            {
                **_file_record(path, shard_directory),
                "dtype": np.dtype(dtype).str,
                "shape": [1, *shape],
            }
        )
    index_row = _with_hash(
        {
            "schema": auditor.ENDPOINT_INDEX_SCHEMA,
            "dataset_role": role,
            "family": family,
            "scene_id": scene_id,
            "endpoint_identity_sha256": identity_sha,
            "plan_endpoint_content_sha256": plan_endpoint["content_sha256"],
            "shard_row": 0,
            "image_path_metadata_only": plan_endpoint["image_path_metadata_only"],
            "image_sha256_commitment_only": image_sha,
            "evidence_content_sha256": evidence.content_sha256(),
            "raster_content_sha256": raster.content_sha256(),
        }
    )
    index_path = shard_directory / "index.jsonl"
    index_payload = auditor.canonical_json_bytes(index_row) + b"\n"
    index_path.write_bytes(index_payload)
    shard_files.append(
        {
            **_file_record(index_path, shard_directory),
            "dtype": "canonical_jsonl",
            "shape": [1],
        }
    )
    shard_files.sort(key=lambda item: item["path"])
    shard = _with_hash(
        {
            "schema": auditor.SHARD_SCHEMA,
            "dataset_role": role,
            "family": family,
            "scene_id": scene_id,
            "scene_id_sha256": hashlib.sha256(scene_id.encode()).hexdigest(),
            "endpoint_count": 1,
            "ordered_endpoint_identity_sha256": auditor.canonical_json_sha256([identity_sha]),
            "ordered_evidence_sha256": auditor.canonical_json_sha256([evidence.content_sha256()]),
            "ordered_raster_sha256": auditor.canonical_json_sha256([raster.content_sha256()]),
            "files": shard_files,
        }
    )
    _write_json(shard_directory / "shard.json", shard)

    top_core = dict(index_row)
    top_core.pop("content_sha256")
    top_core["scene_shard"] = f"shards/{scene_directory}/shard.json"
    top_endpoint = _with_hash(top_core)
    pair_path = root / "pairs.jsonl"
    endpoint_path = root / "endpoints.jsonl"
    pair_payload = auditor.canonical_json_bytes(pair) + b"\n"
    endpoint_payload = auditor.canonical_json_bytes(top_endpoint) + b"\n"
    pair_path.write_bytes(pair_payload)
    endpoint_path.write_bytes(endpoint_payload)
    files = [
        _file_record(path, root)
        for path in sorted(root.rglob("*"), key=lambda item: str(item.relative_to(root)))
        if path.is_file()
    ]
    files.sort(key=lambda item: item["path"])
    sample_records = auditor._sample_records([top_endpoint])
    manifest = _with_hash(
        {
            "schema": auditor.DATASET_SCHEMA,
            "status": "complete_pending_independent_audit",
            "evidence_schema": auditor.evidence_v4.EVIDENCE_SCHEMA,
            "raster_schema": auditor.evidence_v4.RASTER_SCHEMA,
            "roles": list(plan_v5.DEVELOPMENT_ROLES),
            "pair_counts": {role: 1, "checkpoint_selection": 0, "probability_calibration": 0},
            "endpoint_instance_count": 2,
            "unique_endpoint_counts": {role: 1, "checkpoint_selection": 0, "probability_calibration": 0},
            "scene_shard_count": 1,
            "ordered_pair_sha256": auditor.canonical_json_sha256([pair["content_sha256"]]),
            "ordered_endpoint_sha256": auditor.canonical_json_sha256([top_endpoint["content_sha256"]]),
            "pair_index": {
                "path": "pairs.jsonl",
                "row_count": 1,
                "file_sha256": hashlib.sha256(pair_payload).hexdigest(),
            },
            "endpoint_index": {
                "path": "endpoints.jsonl",
                "row_count": 1,
                "file_sha256": hashlib.sha256(endpoint_payload).hexdigest(),
            },
            "array_layout": [
                {"path": name, "dtype": np.dtype(dtype).str, "trailing_shape": list(shape)}
                for name, dtype, shape in auditor.ARRAY_LAYOUT
            ],
            "shards": [
                {
                    "path": f"shards/{scene_directory}/shard.json",
                    "dataset_role": role,
                    "family": family,
                    "scene_id": scene_id,
                    "endpoint_count": 1,
                    "content_sha256": shard["content_sha256"],
                }
            ],
            "files": files,
            "input_provenance": {"fixture": "independent_synthetic"},
            "access_ledger": {"rgb_byte_opens": 0, "g2_payload_opens": 0},
            "independent_audit_precommit": {
                "scheme": "minimum_sha256_role_nul_family_nul_endpoint_identity_v1",
                "one_endpoint_per_observed_role_family": True,
                "expected_exact_record_count": 24,
                "records": sample_records,
                "records_sha256": auditor.canonical_json_sha256(sample_records),
            },
            "parallel_contract": {
                "worker_start_method": "spawn",
                "maximum_workers": 6,
                "native_threads_per_worker": 1,
                "gpu_visible_to_workers": False,
                "merge_order": "role_then_scene_then_endpoint_identity",
                "worker_count_does_not_change_artifact_bytes": True,
            },
            "publication": {
                "staging": "private_sibling_directory_mode_0700",
                "commit": "single_renameat2_RENAME_NOREPLACE",
                "manifest_self_inventory": "canonical_content_sha256",
                "file_inventory": "every_regular_file_except_manifest_self",
            },
            "licenses": {name: False for name in auditor.FALSE_LICENSE_FIELDS},
        }
    )
    _write_json(root / "manifest.json", manifest)
    digest = hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
    return root, digest, auditor.AuditInputs(plan=plan, inventory=inventory), {identity_sha: source_arrays}


def test_independent_synthetic_artifact_passes_full_audit(tmp_path: Path) -> None:
    root, digest, inputs, replay = _synthetic_fixture(tmp_path)

    result = auditor.audit_dataset_v1(
        root,
        expected_manifest_file_sha256=digest,
        inputs=inputs,
        sample_recomputer=lambda *_args: replay,
        workers=2,
    )

    assert result["verdict"] == "PASS"
    assert result["pair_count"] == 1
    assert result["unique_endpoint_count"] == 1
    assert result["sample_count"] == 1
    assert result["sample_original_geometry_recomputed"] is True


def test_frozen_builder_artifact_matches_independent_literal_auditor(
    tmp_path: Path,
) -> None:
    _manual_root, _digest, inputs, replay = _synthetic_fixture(tmp_path)
    endpoint = inputs.plan.endpoints[0]
    pair = inputs.plan.pairs[0]
    frame = auditor.raycast_v4.synthetic_scene_jobs(1)[0].frames[0]
    prepared = frozen_builder.PreparedEndpointV1(
        plan_endpoint=endpoint,
        family=str(pair["family"]),
        frame=frame,
    )
    job = frozen_builder.PreparedSceneJobV1(
        scene_id=str(pair["scene_id"]),
        role=str(pair["dataset_role"]),
        family=str(pair["family"]),
        endpoints=(prepared,),
    )
    output = tmp_path / "frozen_builder_output"
    frozen_builder.build_prepared_dataset_v1(
        (job,),
        (pair,),
        output_directory=output,
        workers=1,
        input_provenance={"fixture": "frozen_builder_cross_contract"},
        access_ledger={"rgb_byte_opens": 0, "g2_payload_opens": 0},
    )
    manifest_sha = hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()

    result = auditor.audit_dataset_v1(
        output,
        expected_manifest_file_sha256=manifest_sha,
        inputs=inputs,
        sample_recomputer=lambda *_args: replay,
        workers=2,
    )

    assert result["verdict"] == "PASS"


def test_array_byte_mutation_is_rejected(tmp_path: Path) -> None:
    root, digest, inputs, replay = _synthetic_fixture(tmp_path)
    path = next((root / "shards").glob("*/raster_labels.u1"))
    payload = bytearray(path.read_bytes())
    payload[0] ^= 1
    path.write_bytes(payload)

    with pytest.raises(auditor.RawSupervisionAuditError, match="bytes changed"):
        auditor.audit_dataset_v1(
            root,
            expected_manifest_file_sha256=digest,
            inputs=inputs,
            sample_recomputer=lambda *_args: replay,
        )


def test_source_replay_must_match_every_array_byte(tmp_path: Path) -> None:
    root, digest, inputs, replay = _synthetic_fixture(tmp_path)
    endpoint_digest = next(iter(replay))
    changed = list(replay[endpoint_digest])
    changed[0] = changed[0].copy()
    changed[0][0] += np.float32(0.25)

    with pytest.raises(auditor.RawSupervisionAuditError, match="differs byte-for-byte"):
        auditor.audit_dataset_v1(
            root,
            expected_manifest_file_sha256=digest,
            inputs=inputs,
            sample_recomputer=lambda *_args: {endpoint_digest: tuple(changed)},
        )


def test_nonzero_forbidden_access_ledger_is_rejected(tmp_path: Path) -> None:
    root, _digest, inputs, replay = _synthetic_fixture(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["access_ledger"]["g2_payload_opens"] = 1
    core = dict(manifest)
    core.pop("content_sha256")
    manifest["content_sha256"] = auditor.canonical_json_sha256(core)
    _write_json(root / "manifest.json", manifest)
    digest = hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()

    with pytest.raises(PermissionError, match="forbidden access"):
        auditor.audit_dataset_v1(
            root,
            expected_manifest_file_sha256=digest,
            inputs=inputs,
            sample_recomputer=lambda *_args: replay,
        )


def test_publisher_refuses_occupied_result_and_failure_leaves(tmp_path: Path) -> None:
    parent = tmp_path / "publication"
    parent.mkdir()
    (parent / "result.json").write_bytes(b"foreign-result")
    (parent / "failure.json").write_bytes(b"foreign-failure")

    with auditor._ExclusiveAuditPublisher(parent) as publisher:
        with pytest.raises(FileExistsError):
            publisher.require_absent("result.json", "failure.json")

    assert (parent / "result.json").read_bytes() == b"foreign-result"
    assert (parent / "failure.json").read_bytes() == b"foreign-failure"


def test_publisher_uses_true_noreplace_and_preserves_late_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "publication"
    parent.mkdir()
    original = auditor._rename_noreplace_at

    def insert_destination_then_rename(parent_fd: int, source: str, destination: str) -> None:
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=parent_fd,
        )
        try:
            os.write(descriptor, b"late-foreign-result")
        finally:
            os.close(descriptor)
        original(parent_fd, source, destination)

    monkeypatch.setattr(auditor, "_rename_noreplace_at", insert_destination_then_rename)
    with auditor._ExclusiveAuditPublisher(parent) as publisher:
        with pytest.raises(FileExistsError):
            publisher.publish("result.json", {"verdict": "PASS"})

    assert (parent / "result.json").read_bytes() == b"late-foreign-result"
    assert list(parent.iterdir()) == [parent / "result.json"]


def test_publisher_detects_parent_swap_without_writing_replacement(tmp_path: Path) -> None:
    parent = tmp_path / "publication"
    parent.mkdir()
    detached = tmp_path / "detached"

    with auditor._ExclusiveAuditPublisher(parent) as publisher:
        parent.rename(detached)
        parent.mkdir()
        (parent / "foreign.txt").write_bytes(b"preserve")
        with pytest.raises(auditor.RawSupervisionAuditError, match="directory chain changed"):
            publisher.publish("result.json", {"verdict": "PASS"})

    assert (parent / "foreign.txt").read_bytes() == b"preserve"
    assert not (parent / "result.json").exists()
    assert not (detached / "result.json").exists()


def test_publisher_cleanup_preserves_replaced_foreign_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "publication"
    parent.mkdir()

    def replace_then_fail(parent_fd: int, source: str, _destination: str) -> None:
        os.unlink(source, dir_fd=parent_fd)
        descriptor = os.open(source, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=parent_fd)
        try:
            os.write(descriptor, b"foreign-temporary")
        finally:
            os.close(descriptor)
        raise RuntimeError("injected publication failure")

    monkeypatch.setattr(auditor, "_rename_noreplace_at", replace_then_fail)
    with auditor._ExclusiveAuditPublisher(parent) as publisher:
        with pytest.raises(RuntimeError, match="injected"):
            publisher.publish("result.json", {"verdict": "PASS"})

    remaining = list(parent.iterdir())
    assert len(remaining) == 1
    assert remaining[0].read_bytes() == b"foreign-temporary"
    assert not (parent / "result.json").exists()


def test_cli_exposes_no_dataset_or_output_path_override() -> None:
    parsed = audit_cli._parse_args(["--manifest-sha256", "a" * 64, "--workers", "3"])
    assert parsed.workers == 3
    with pytest.raises(SystemExit):
        audit_cli._parse_args(
            ["--manifest-sha256", "a" * 64, "--output", "/tmp/escape"]
        )


def test_exact_access_ledger_is_literal_and_forbidden_counts_stay_zero() -> None:
    plan = plan_v5.DevelopmentRawSupervisionPlan(
        value={"access_ledger": {"metadata": "plan"}},
        pairs=(),
        endpoints=(),
    )
    inventory = plan_v5.DevelopmentSourceInventory(
        records=(),
        hashes={},
        access_ledger={"metadata": "inventory"},
    )
    inputs = auditor.AuditInputs(plan=plan, inventory=inventory)
    ledger: dict[str, Any] = {name: 0 for name in auditor.EXACT_ACCESS_LEDGER_KEYS}
    ledger.update(
        {
            "schema": auditor.ACCESS_LEDGER_SCHEMA,
            "measurement_scope": "controlled_data_opens_excluding_import_and_reviewed_source_hash_reads",
            "metadata_plan_first_pass": plan.value["access_ledger"],
            "metadata_source_inventory_first_pass": inventory.access_ledger,
            "metadata_plan_second_pass": plan.value["access_ledger"],
            "metadata_source_inventory_second_pass": inventory.access_ledger,
            "development_scene_workers": 88,
            "unique_endpoint_raycasts": 9460,
            "pair_endpoint_references": 10344,
            "source_frames_jsonl_records_scanned": 123,
            "source_frames_selected_records": 9460,
            "source_frames_byte_opens": 176,
            "source_scene_manifest_byte_opens": 176,
            "render_plan_byte_opens": 176,
            "render_summary_byte_opens": 176,
            "geometry_contract_byte_opens": 2,
            "render_audit_byte_opens": 2,
            "source_payload_first_pass_file_count": 354,
            "source_payload_second_pass_file_count": 354,
            "source_payload_total_byte_opens": 708,
            "g2_source_index_rows_read_for_exclusion": 8,
        }
    )
    auditor._validate_exact_access_ledger(ledger, inputs=inputs, frames_scanned=123)
    ledger["g2_source_payload_opens"] = 1
    with pytest.raises(auditor.RawSupervisionAuditError, match="g2_source_payload_opens"):
        auditor._validate_exact_access_ledger(
            ledger, inputs=inputs, frames_scanned=123
        )


def test_bound_source_reader_rejects_leaf_alias(tmp_path: Path) -> None:
    root = tmp_path / "source_root"
    root.mkdir()
    payload = b"allowed-source\n"
    real = root / "real.jsonl"
    real.write_bytes(payload)
    alias = root / "alias.jsonl"
    alias.symlink_to(real.name)

    assert auditor._read_absolute_bound_payload(
        real,
        hashlib.sha256(payload).hexdigest(),
        repository_root=root,
        name="synthetic allowed source",
    ) == payload
    with pytest.raises(PermissionError, match="unaliased regular file"):
        auditor._read_absolute_bound_payload(
            alias,
            hashlib.sha256(payload).hexdigest(),
            repository_root=root,
            name="synthetic allowed source",
        )
