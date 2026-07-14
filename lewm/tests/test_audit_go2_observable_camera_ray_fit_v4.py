from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import audit_go2_observable_camera_ray_fit_v4 as audit
from scripts import build_go2_observable_camera_ray_fit_v4 as builder


def _build(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    output = tmp_path / "dataset"
    manifest = builder.build_dataset_from_jobs(
        builder.synthetic_scene_jobs(2),
        output_directory=output,
        workers=1,
        input_provenance={"schema": "synthetic_audit_test", "dataset_role": "train"},
        access_ledger={
            "rgb_byte_opens": 0,
            "fit_label_payload_byte_opens": 0,
            "nontrain_role_byte_opens": 0,
            "g2_byte_opens": 0,
        },
    )
    return output / "manifest.json", manifest


def _load(path: Path):
    return audit.load_and_verify_dataset(
        path, expected_manifest_file_sha256=audit._sha256_file(path)
    )


def _references(frames):
    return {
        audit._canonical_json_bytes(index_row["frame_key"]): (
            audit.rasterize_observable_camera_ray_evidence_v4(
                evidence
            ).output_labels.copy(),
            np.ones((64, 64), dtype=bool),
        )
        for index_row, evidence in frames
    }


def test_loader_reconstructs_and_verifies_every_synthetic_frame(
    tmp_path: Path,
) -> None:
    path, expected = _build(tmp_path)
    manifest, frames = _load(path)
    assert manifest == expected
    assert len(frames) == 2
    assert all(row["frame_key"]["dataset_role"] == "train" for row, _ in frames)


def test_binary_tamper_fails_closed(tmp_path: Path) -> None:
    path, manifest = _build(tmp_path)
    shard_path = path.parent / manifest["shards"][0]["path"]
    shard = json.loads(shard_path.read_text())
    target = shard_path.parent / shard["files"][0]["path"]
    raw = bytearray(target.read_bytes())
    raw[0] ^= 1
    target.write_bytes(raw)

    with pytest.raises(ValueError, match="bytes changed"):
        _load(path)


def test_self_consistent_duplicate_shard_file_declaration_fails_closed(
    tmp_path: Path,
) -> None:
    path, manifest = _build(tmp_path)
    shard_record = manifest["shards"][0]
    shard_path = path.parent / shard_record["path"]
    shard = json.loads(shard_path.read_text())
    shard["files"].append(dict(shard["files"][0]))
    shard_core = dict(shard)
    shard_core.pop("content_sha256")
    shard["content_sha256"] = audit.canonical_json_sha256(shard_core)
    shard_raw = audit._canonical_json_bytes(shard) + b"\n"
    shard_path.write_bytes(shard_raw)

    shard_record["content_sha256"] = shard["content_sha256"]
    shard_record["file_sha256"] = hashlib.sha256(shard_raw).hexdigest()
    manifest_core = dict(manifest)
    manifest_core.pop("content_sha256")
    manifest["content_sha256"] = audit.canonical_json_sha256(manifest_core)
    manifest_raw = audit._canonical_json_bytes(manifest) + b"\n"
    path.write_bytes(manifest_raw)

    with pytest.raises(ValueError, match="repeats a declared file"):
        audit.load_and_verify_dataset(
            path,
            expected_manifest_file_sha256=hashlib.sha256(
                manifest_raw
            ).hexdigest(),
        )


def test_manifest_path_escape_fails_closed(tmp_path: Path) -> None:
    path, _manifest = _build(tmp_path)
    payload = json.loads(path.read_text())
    payload["shards"][0]["path"] = "../outside/shard.json"
    core = dict(payload)
    core.pop("content_sha256")
    payload["content_sha256"] = audit.canonical_json_sha256(core)
    raw = audit._canonical_json_bytes(payload) + b"\n"
    path.write_bytes(raw)

    with pytest.raises((PermissionError, FileNotFoundError)):
        audit.load_and_verify_dataset(
            path,
            expected_manifest_file_sha256=hashlib.sha256(raw).hexdigest(),
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "root_extra",
        "shard_extra",
        "shards_root_symlink",
        "shards_root_nonregular",
        "shard_symlink",
        "shard_nonregular",
    ],
)
def test_complete_filesystem_inventory_rejects_unregistered_entries(
    tmp_path: Path,
    mutation: str,
) -> None:
    path, manifest = _build(tmp_path)
    shard_manifest = path.parent / manifest["shards"][0]["path"]
    shard = json.loads(shard_manifest.read_text())
    if mutation == "root_extra":
        (path.parent / "unexpected.bin").write_bytes(b"unexpected")
    elif mutation == "shard_extra":
        (shard_manifest.parent / "unexpected.bin").write_bytes(b"unexpected")
    elif mutation in {"shards_root_symlink", "shards_root_nonregular"}:
        shards_root = path.parent / "shards"
        saved = tmp_path / f"saved_{mutation}"
        shards_root.rename(saved)
        if mutation == "shards_root_symlink":
            shards_root.symlink_to(saved, target_is_directory=True)
        else:
            shards_root.write_bytes(b"not a directory")
    else:
        target = shard_manifest.parent / shard["files"][0]["path"]
        target.unlink()
        if mutation == "shard_symlink":
            target.symlink_to(shard["files"][1]["path"])
        else:
            target.mkdir()
    with pytest.raises((ValueError, PermissionError), match="inventory|regular"):
        _load(path)


def test_rgb_receipt_hash_must_join_to_every_shard_index(tmp_path: Path) -> None:
    path, _manifest = _build(tmp_path)
    payload = json.loads(path.read_text())
    receipt = payload["rgb_receipt"]
    receipt["entries"][0]["rgb_file_sha256"] = "0" * 64
    receipt["entries_sha256"] = audit.canonical_json_sha256(receipt["entries"])
    receipt_core = dict(receipt)
    receipt_core.pop("content_sha256")
    receipt["content_sha256"] = audit.canonical_json_sha256(receipt_core)
    core = dict(payload)
    core.pop("content_sha256")
    payload["content_sha256"] = audit.canonical_json_sha256(core)
    raw = audit._canonical_json_bytes(payload) + b"\n"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="does not join"):
        audit.load_and_verify_dataset(
            path,
            expected_manifest_file_sha256=hashlib.sha256(raw).hexdigest(),
        )


def test_nontrain_manifest_role_fails_closed(tmp_path: Path) -> None:
    path, _manifest = _build(tmp_path)
    payload = json.loads(path.read_text())
    payload["dataset_role"] = "g2_evaluation"
    core = dict(payload)
    core.pop("content_sha256")
    payload["content_sha256"] = audit.canonical_json_sha256(core)
    raw = audit._canonical_json_bytes(payload) + b"\n"
    path.write_bytes(raw)

    with pytest.raises(ValueError, match="identity changed"):
        audit.load_and_verify_dataset(
            path,
            expected_manifest_file_sha256=hashlib.sha256(raw).hexdigest(),
        )


def test_reference_comparison_reports_mismatch_without_privileged_repair(
    tmp_path: Path,
) -> None:
    path, _manifest = _build(tmp_path)
    manifest, frames = _load(path)
    references = _references(frames)
    first_key = next(iter(references))
    target, supervision = references[first_key]
    target[0, 0] = (int(target[0, 0]) + 1) % 3
    result = audit.build_audit_result(
        dataset_manifest=manifest,
        frames=frames,
        reference_labels=references,
        access_ledger={"nontrain_role_byte_opens": 0},
        exact_fit=False,
    )
    comparison = result["audit"]["legacy_physical_v3_comparison"]
    assert not comparison["exact"]
    assert comparison["mismatch_cell_count"] == 1
    assert "may not be repaired" in comparison["interpretation"]
    assert supervision.all()


def test_missing_reference_frame_fails_closed(tmp_path: Path) -> None:
    path, _manifest = _build(tmp_path)
    _manifest_value, frames = _load(path)
    references = _references(frames)
    references.pop(next(iter(references)))
    with pytest.raises(ValueError, match="lacks an exact reference"):
        audit.audit_dataset_frames(frames, reference_labels=references)


def test_exact_n32_mode_requires_320_balanced_frames(tmp_path: Path) -> None:
    path, _manifest = _build(tmp_path)
    manifest, frames = _load(path)
    with pytest.raises(ValueError, match="balanced 320-frame"):
        audit.build_audit_result(
            dataset_manifest=manifest,
            frames=frames,
            reference_labels=_references(frames),
            access_ledger={},
            exact_fit=True,
        )


def test_audit_result_keeps_all_promotion_licenses_false(tmp_path: Path) -> None:
    path, _manifest = _build(tmp_path)
    manifest, frames = _load(path)
    result = audit.build_audit_result(
        dataset_manifest=manifest,
        frames=frames,
        reference_labels=_references(frames),
        access_ledger={"nontrain_role_byte_opens": 0},
    )
    assert not any(result["licenses"].values())
    assert result["scope"]["gpu_used"] is False
    assert result["scope"]["rgb_opened"] is False


def test_cli_separates_dry_run_and_exact_authorization() -> None:
    assert audit._parse_args(["--dry-run"]).dry_run
    with pytest.raises(SystemExit):
        audit._parse_args(["--run-exact-fit"])
    with pytest.raises(SystemExit):
        audit._parse_args(
            ["--dry-run", "--dataset-manifest-sha256", "0" * 64]
        )


def test_dry_run_is_cpu_only_and_exact_against_synthetic_reference() -> None:
    result = audit.run_dry_run()
    assert result["internal_verification_passes"]
    assert result["synthetic_reference_exact"]
    assert result["fit_payload_opened"] is False
    assert result["gpu_used"] is False


def _exact_rgb_receipt() -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for index in range(320):
        digest = hashlib.sha256(f"exact-rgb-{index}".encode()).hexdigest()
        entries.append(
            {
                "frame_key": {
                    "dataset_role": "train",
                    "family": f"family_{index // 64}",
                    "global_row": index,
                    "image_sha256": digest,
                },
                "canonical_rgb_path": str(
                    builder.ROOT
                    / ".synthetic/exact_receipt/rgb"
                    / f"frame_{index:06d}.png"
                ),
                "rgb_file_sha256": digest,
            }
        )
    entries.sort(key=lambda entry: audit._canonical_json_bytes(entry["frame_key"]))
    core: dict[str, object] = {
        "schema": builder.RGB_RECEIPT_SCHEMA,
        "dataset_role": "train",
        "frame_count": len(entries),
        "ordered_frame_keys_sha256": audit.canonical_json_sha256(
            [entry["frame_key"] for entry in entries]
        ),
        "entries_sha256": audit.canonical_json_sha256(entries),
        "rgb_byte_opens": 0,
        "entries": entries,
    }
    return {**core, "content_sha256": audit.canonical_json_sha256(core)}


def _exact_build_ledger() -> dict[str, object]:
    ledger: dict[str, object] = {
        name: 0
        for name in builder.EXACT_BUILD_LEDGER_FIELDS
        if name
        not in {
            "per_shard_materialization",
            "denied_primary_reasons",
            "denied_modality_attempts",
            "denied_attempt_records",
        }
    }
    ledger["per_shard_materialization"] = []
    ledger["denied_primary_reasons"] = {
        name: 0 for name in builder.EXACT_DENIAL_PRIMARY_REASONS
    }
    ledger["denied_modality_attempts"] = {
        name: 0 for name in builder.EXACT_DENIAL_MODALITIES
    }
    ledger["denied_attempt_records"] = []
    ledger.update(
        {
            "sidecar_manifest_byte_opens": 1,
            "sidecar_train_role_byte_opens": 1,
            "panel_metadata_byte_opens": 1,
            "source_frame_records_selected": 320,
            "source_geometry_hash_byte_opens": 2,
            "source_geometry_json_parses": 1,
            "source_geometry_jsonl_records": 320,
            "implementation_source_hash_byte_opens": 1,
            "document_hash_byte_opens": 1,
        }
    )
    return ledger


def _exact_receipt() -> tuple[dict[str, object], str, str]:
    implementation = json.loads(builder.IMPLEMENTATION_MANIFEST_PATH.read_text())
    implementation_file = audit._sha256_file(builder.IMPLEMENTATION_MANIFEST_PATH)
    source_file = builder.SOURCE_AUTHORIZATION_MANIFEST_FILE_SHA256
    manifest: dict[str, object] = {
        "schema": builder.DATASET_SCHEMA,
        "evidence_schema": builder.EVIDENCE_SCHEMA,
        "dataset_role": "train",
        "frame_count": 320,
        "scene_shard_count": 20,
        "rgb_receipt": _exact_rgb_receipt(),
        "input_provenance": {
            "implementation_manifest_file_sha256": implementation_file,
            "implementation_manifest_content_sha256": implementation[
                "content_sha256"
            ],
            "source_authorization_manifest_file_sha256": source_file,
            "source_authorization_manifest_content_sha256": (
                builder.SOURCE_AUTHORIZATION_MANIFEST_CONTENT_SHA256
            ),
            "source_hashes": {
                "binding": {
                    "path": "unused",
                    "sha256": builder.SOURCE_AUTHORIZATION_BINDING_SHA256,
                }
            },
            "fit_panel_file_sha256": builder.PANEL_FILE_SHA256,
            "fit_panel_content_sha256": builder.PANEL_CONTENT_SHA256,
            "fit_rows_sha256": builder.FIT_ROWS_SHA256,
            "source_geometry_manifest_sha256": (
                builder.SOURCE_GEOMETRY_MANIFEST_SHA256
            ),
            "render_summaries_manifest_sha256": (
                builder.RENDER_SUMMARIES_MANIFEST_SHA256
            ),
            "sidecar_manifest_file_sha256": builder.SIDECAR_MANIFEST_FILE_SHA256,
            "sidecar_manifest_content_sha256": (
                builder.SIDECAR_MANIFEST_CONTENT_SHA256
            ),
            "sidecar_train_file_sha256": builder.SIDECAR_TRAIN_FILE_SHA256,
            "sidecar_train_content_sha256": builder.SIDECAR_TRAIN_CONTENT_SHA256,
            "sidecar_train_ordered_global_sha256": (
                builder.SIDECAR_TRAIN_ORDERED_GLOBAL_SHA256
            ),
            "sidecar_train_ordered_identity_sha256": (
                builder.SIDECAR_TRAIN_ORDERED_IDENTITY_SHA256
            ),
        },
        "access_ledger": _exact_build_ledger(),
        "parallel_contract": {
            "worker_start_method": "spawn",
            "maximum_workers": 6,
            "native_threads_per_worker": 1,
            "canonical_merge": "scene_hash_then_canonical_frame_key",
            "worker_count_does_not_change_artifact_bytes": True,
            "per_worker_source_revalidation": True,
            "parent_source_revalidation_before_manifest": True,
        },
        "publication": "private_staging_hardlink_no_replace_manifest_last",
        "array_layout": [
            {
                "path": name,
                "dtype": np.dtype(dtype).str,
                "trailing_shape": list(shape),
            }
            for name, dtype, shape in builder.ARRAY_LAYOUT
        ],
        "licenses": {
            "model_output_authorized": False,
            "holdout_authorized": False,
            "g2_authorized": False,
            "runtime_authorized": False,
            "promotion_authorized": False,
        },
    }
    return manifest, implementation_file, source_file


def test_exact_receipt_binds_reviewed_producer_and_zero_access() -> None:
    manifest, implementation_file, source_file = _exact_receipt()
    implementation = json.loads(builder.IMPLEMENTATION_MANIFEST_PATH.read_text())
    audit._validate_exact_dataset_receipt(
        manifest,
        implementation_manifest_file_sha256=implementation_file,
        implementation_manifest_content_sha256=implementation["content_sha256"],
        source_authorization_manifest_file_sha256=source_file,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "producer",
        "source",
        "ledger",
        "ledger_field",
        "denial_key",
        "license",
    ],
)
def test_exact_receipt_mutations_fail_before_label_access(mutation: str) -> None:
    manifest, implementation_file, source_file = _exact_receipt()
    implementation = json.loads(builder.IMPLEMENTATION_MANIFEST_PATH.read_text())
    if mutation == "producer":
        manifest["input_provenance"][
            "implementation_manifest_file_sha256"
        ] = "0" * 64
    elif mutation == "source":
        manifest["input_provenance"][
            "source_authorization_manifest_file_sha256"
        ] = "0" * 64
    elif mutation == "ledger":
        manifest["access_ledger"]["g2_opens"] = 1
    elif mutation == "ledger_field":
        manifest["access_ledger"].pop("document_hash_byte_opens")
    elif mutation == "denial_key":
        manifest["access_ledger"]["denied_primary_reasons"].pop("g2")
    else:
        manifest["licenses"]["g2_authorized"] = True
    with pytest.raises((ValueError, PermissionError)):
        audit._validate_exact_dataset_receipt(
            manifest,
            implementation_manifest_file_sha256=implementation_file,
            implementation_manifest_content_sha256=implementation[
                "content_sha256"
            ],
            source_authorization_manifest_file_sha256=source_file,
        )


def test_direct_label_opener_requires_reviewed_audit_authorization() -> None:
    manifest, implementation_file, source_file = _exact_receipt()
    with pytest.raises(PermissionError, match="not authorized"):
        audit.load_exact_fit_reference_labels(
            machine_manifest_sha256=source_file,
            implementation_manifest_sha256=implementation_file,
            dataset_manifest=manifest,
        )


def test_authorized_label_opener_validates_build_receipt_before_source_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, implementation_file, source_file = _exact_receipt()
    events: list[str] = []

    def authorize(*_args, **_kwargs):
        events.append("authorization")
        return {"content_sha256": "1" * 64}

    def reject_receipt(*_args, **_kwargs):
        events.append("dataset_receipt")
        raise RuntimeError("RECEIPT_REJECTED_BEFORE_SOURCE")

    def forbidden_source_load(*_args, **_kwargs):
        events.append("source_access")
        raise AssertionError("source access preceded receipt validation")

    monkeypatch.setattr(
        builder, "_load_reviewed_implementation_manifest", authorize
    )
    monkeypatch.setattr(audit, "_validate_exact_dataset_receipt", reject_receipt)
    monkeypatch.setattr(builder, "_load_neutral_module", forbidden_source_load)
    with pytest.raises(RuntimeError, match="RECEIPT_REJECTED_BEFORE_SOURCE"):
        audit.load_exact_fit_reference_labels(
            machine_manifest_sha256=source_file,
            implementation_manifest_sha256=implementation_file,
            dataset_manifest=manifest,
        )
    assert events == ["authorization", "dataset_receipt"]
