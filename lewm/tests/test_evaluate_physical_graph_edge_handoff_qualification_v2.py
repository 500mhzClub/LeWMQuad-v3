from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import copy
import shutil
import zipfile

import pytest

from scripts import evaluate_physical_graph_edge_handoff_qualification_v2 as E


def test_system_python_help_has_no_scientific_runtime_imports() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = ""
    completed = subprocess.run(
        ["/usr/bin/python3", str(E.REPO_ROOT / E.__file__), "--help"],
        cwd=E.REPO_ROOT,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "emit-v1-custody" in completed.stdout
    imported = subprocess.run(
        [
            "/usr/bin/python3",
            "-c",
            (
                "import sys; sys.path.insert(0, %r); "
                "import scripts.evaluate_physical_graph_edge_handoff_qualification_v2; "
                "assert 'torch' not in sys.modules and 'genesis' not in sys.modules"
            )
            % str(E.REPO_ROOT),
        ],
        cwd=E.REPO_ROOT,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert imported.returncode == 0, imported.stderr


def test_raw_c_array_digest_domain_is_exact_payload_only() -> None:
    payload = bytes.fromhex("00000000000000000000000000000000")
    assert E.raw_c_array_sha256(payload) == hashlib.sha256(payload).hexdigest()
    # Shape and dtype are separately persisted facts and are deliberately not
    # folded into this digest domain.
    assert E.raw_c_array_sha256(payload) == E.raw_c_array_sha256(payload)
    assert E.raw_c_array_sha256(payload + b"\x00") != E.raw_c_array_sha256(payload)


@pytest.mark.skipif(
    not E.DEFAULT_V1_CUSTODY_RECEIPT.exists(),
    reason="immutable V1 custody roots are not installed",
)
def test_real_v1_custody_receipt_exactly_rebuilds_without_mutation() -> None:
    official_before = E._root_inventory(E.DEFAULT_V1_OUTPUT_ROOT, "V1 official root")
    material_before = E._root_inventory(E.DEFAULT_V1_MATERIAL_ROOT, "V1 material root")
    rebuilt = E.validate_existing_v1_custody_receipt()
    assert rebuilt["schema"] == E.V1_CUSTODY_SCHEMA
    assert rebuilt["disposition"] == "TECHNICALLY_INVALID_PERSISTED_ARRAY_HASH_METADATA"
    assert rebuilt["official_root"]["file_count"] == 1
    assert rebuilt["official_root"]["regular_file_apparent_bytes"] == 66_418
    assert rebuilt["material_root"]["file_count"] == 34
    assert rebuilt["material_root"]["regular_file_apparent_bytes"] == 4_299_695
    assert len(rebuilt["qualification_pairs"]) == 8
    assert sum(row["qualified"] for row in rebuilt["qualification_pairs"]) == 7
    rejected = [row for row in rebuilt["qualification_pairs"] if not row["qualified"]]
    assert [(row["pool_index"], row["rejection_reason"]) for row in rejected] == [
        (2, "TEACHER_PHYSICS_CONTACT")
    ]
    assert rebuilt["defect_evidence"]["other_embedded_numeric_hash_mismatch_count"] == 0
    raw = E.DEFAULT_V1_CUSTODY_RECEIPT.read_bytes()
    assert raw == E.canonical_document_bytes(rebuilt)
    assert len(raw) == 18_403
    assert hashlib.sha256(raw).hexdigest() == (
        "bb4950d2bde0bf1971e643c746b15a28a1949b1ef3d70736bdaae9da45282d32"
    )
    assert E._root_inventory(E.DEFAULT_V1_OUTPUT_ROOT, "V1 official root") == official_before
    assert E._root_inventory(E.DEFAULT_V1_MATERIAL_ROOT, "V1 material root") == material_before


def test_external_receipt_rejects_self_digest(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    output = tmp_path / "receipt.json"
    with pytest.raises(E.RegenerationError, match="self digest"):
        E._emit_external((root,), output, {"schema": "fixture", "content_digest": "0" * 64})
    assert not output.exists()


def test_root_inventory_rejects_hardlinks(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    first = root / "first"
    second = root / "second"
    first.write_bytes(b"evidence")
    os.link(first, second)
    with pytest.raises(E.RegenerationError, match="non-single-link"):
        E._root_inventory(root, "fixture")


def test_v2_success_and_terminal_inventories_are_distinct() -> None:
    assert len(E.V2_SUCCESS_FILES) == 26
    assert set(E.V2_ADDITIONAL_FILES) <= set(E.V2_SUCCESS_FILES)
    assert E.V2_REPRODUCTION_FAILURE_FILES == (
        "contract.json",
        "v1_custody_and_nonreuse.json",
        "scientific_invariance_receipt.json",
        "v1_v2_first_eight_reproduction.json",
    )
    assert not ({"metrics.json", "result.json", "file_hashes.json"} & set(E.V2_REPRODUCTION_FAILURE_FILES))


def test_source_freeze_observation_requires_complete_71_row_closure() -> None:
    source_commit = "a" * 40
    observation = {
        "head_commit": source_commit,
        "parent_commit": E.SOURCE_PARENT_COMMIT,
        "freeze_subject": "Freeze corrected physical graph edge handoff qualification V2",
        "worktree_clean": True,
        "tracked_source_count": 15,
        "tracked_sources_sha256": "b" * 64,
        "source_closure_path": "docs/source_closure.json",
        "source_closure_bytes": 1,
        "source_closure_sha256": "c" * 64,
        "source_closure_row_count": 71,
        "source_closure_live_bytes_exact": True,
        "metrics_module": E.METRICS_MODULE,
    }
    assert E._validate_source_freeze_observation(
        observation, expected_commit=source_commit
    ) == observation
    stale = dict(observation, source_closure_row_count=70)
    with pytest.raises(E.RegenerationError, match="source-freeze observation failed"):
        E._validate_source_freeze_observation(stale, expected_commit=source_commit)


def test_first_eight_metadata_keeps_observed_stage_runtime() -> None:
    baseline = {
        "experiment_id": E.V1_EXPERIMENT_ID,
        "snapshot": {"previous_applied_command_sha256": "0" * 64},
        "stage_runtime": {"device": "cpu", "fake_runtime": False},
        "payload": {"path": "qualification/pool-000/payload.npz"},
    }
    corrected = copy.deepcopy(baseline)
    corrected["experiment_id"] = E.EXPERIMENT_ID
    corrected["stage_runtime"]["device"] = "cuda"
    assert (
        E._comparable_first_eight_metadata(baseline)
        != E._comparable_first_eight_metadata(corrected)
    )


@pytest.mark.skipif(
    not E.DEFAULT_V1_MATERIAL_ROOT.exists(),
    reason="immutable V1 material custody root is not installed",
)
def test_first_eight_reproduction_is_rebuilt_from_reopened_arrays(
    tmp_path: Path,
) -> None:
    from scripts import run_physical_graph_edge_handoff_qualification_v1 as V1_RUNNER
    from scripts import run_physical_graph_edge_handoff_qualification_v2 as V2_RUNNER
    from lewm.safety import physical_graph_edge_handoff_qualification_v2_metrics as METRICS

    material = tmp_path / "physical_graph_edge_handoff_qualification_v2_material"
    qualification = material / "qualification"
    qualification.mkdir(parents=True)
    for index in range(8):
        source = E.DEFAULT_V1_MATERIAL_ROOT / "qualification" / f"pool-{index:03d}"
        metadata, arrays = V2_RUNNER._ORIGINAL_V1_LOADER(source)
        projected = V2_RUNNER._v2_identity(copy.deepcopy(metadata))
        projected.pop("content_digest", None)
        projected.pop("payload", None)
        projected["snapshot"]["previous_applied_command_sha256"] = (
            V2_RUNNER.persisted_array_sha256(
                arrays["snapshot__previous_applied_command"]
            )
        )
        V2_RUNNER._write_material_shard_impl(
            qualification / f"pool-{index:03d}",
            projected,
            arrays,
            root=material,
        )
    rows, custody = E._first_eight_reproduction_rows(material)
    assert len(rows) == 8
    assert all(row["pass"] for row in rows)
    assert custody["shared_inode_count"] == 0
    assert custody["exact_v1_shard_file_copy_count"] == 0
    receipt = METRICS.build_first_eight_reproduction(rows)
    assert METRICS.validate_first_eight_reproduction(receipt) == receipt
    assert METRICS.authorizes_full_v2_collection(receipt) is True
    expected_paths = [
        f"qualification/pool-{index:03d}/metadata.json" for index in range(8)
    ]
    assert E._validate_all_v2_material_payloads(
        material,
        expected_shard_count=8,
        expected_metadata_paths=expected_paths,
    )["validated_shard_count"] == 8
    with pytest.raises(E.RegenerationError, match="stage/shard path inventory"):
        E._validate_all_v2_material_payloads(
            material,
            expected_shard_count=8,
            expected_metadata_paths=[*expected_paths[:-1], "qualification/wrong/metadata.json"],
        )

    # Independent persisted-evidence validation fails on a dtype declaration
    # that no longer describes the reopened NPY payload.
    metadata_path = qualification / "pool-000" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    members = E._read_npz_members(
        qualification / "pool-000" / "payload.npz",
        "tamper fixture",
    )
    binding = dict(metadata["persisted_array_evidence"]["payload_file"])
    tampered = copy.deepcopy(metadata)
    tampered["persisted_array_evidence"]["arrays"][0]["dtype_str"] = "<f4"
    with pytest.raises(E.RegenerationError, match="raw-byte inventory drift"):
        E._validate_persisted_array_evidence(
            metadata=tampered,
            members=members,
            payload_binding=binding,
            label="tamper fixture",
        )
    legacy_domain = copy.deepcopy(metadata)
    legacy_domain["snapshot"]["previous_applied_command_sha256"] = (
        E.V1_PERSISTED_PREVIOUS_COMMAND_CANONICAL_SHA256
    )
    with pytest.raises(E.RegenerationError, match="previous-command raw persisted-byte"):
        E._validate_persisted_array_evidence(
            metadata=legacy_domain,
            members=members,
            payload_binding=binding,
            label="legacy digest-domain fixture",
        )

    # A genuine logical-member mismatch is a scientific reproduction mismatch,
    # not a reducer crash: it must be representable by the four-leaf terminal.
    source = E.DEFAULT_V1_MATERIAL_ROOT / "qualification" / "pool-000"
    source_metadata, source_arrays = V2_RUNNER._ORIGINAL_V1_LOADER(source)
    missing_metadata = V2_RUNNER._v2_identity(copy.deepcopy(source_metadata))
    missing_metadata.pop("content_digest", None)
    missing_metadata.pop("payload", None)
    missing_metadata["snapshot"]["previous_applied_command_sha256"] = (
        V2_RUNNER.persisted_array_sha256(
            source_arrays["snapshot__previous_applied_command"]
        )
    )
    missing_arrays = dict(source_arrays)
    missing_arrays.pop("teacher__physics_contact")
    shutil.rmtree(qualification / "pool-000")
    V2_RUNNER._write_material_shard_impl(
        qualification / "pool-000",
        missing_metadata,
        missing_arrays,
        root=material,
    )
    mismatch_rows, _mismatch_custody = E._first_eight_reproduction_rows(material)
    assert mismatch_rows[0]["pass"] is False
    assert mismatch_rows[0]["teacher_trace_member_inventory_equal"] is False
    assert mismatch_rows[0]["contact_sequence_equal"] is False
    mismatch_receipt = METRICS.build_first_eight_reproduction(mismatch_rows)
    assert mismatch_receipt["status"] == "V1_V2_PHYSICAL_REPRODUCTION_MISMATCH"
    assert mismatch_receipt["full_collection_authorized"] is False
    assert METRICS.validate_first_eight_reproduction(mismatch_receipt) == mismatch_receipt

    # Container provenance is independently required on every V2 material
    # payload, even when all logical NPY members are otherwise unchanged.
    payload_path = qualification / "pool-000" / "payload.npz"
    with zipfile.ZipFile(payload_path, mode="a") as archive:
        archive.comment = b"WRONG"
    with pytest.raises(E.RegenerationError, match="archive comment drift"):
        E._read_npz_members(
            payload_path,
            "comment tamper fixture",
            expected_comment=E.V2_NPZ_ARCHIVE_COMMENT,
        )
