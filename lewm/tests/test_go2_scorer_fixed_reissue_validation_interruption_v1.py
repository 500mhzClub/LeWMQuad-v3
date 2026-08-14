"""Pure custody tests for the fixed-reissue validation interruption."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from lewm.oracle import \
    go2_scorer_fixed_reissue_validation_interruption_v1 as I


NEW_SOURCE_COMMIT = "b" * 40
NEW_CLEAN_DIGEST = "c" * 64
NEW_IMPLEMENTATION_DIGEST = "d" * 64


def test_frozen_production_bindings_have_exact_digest_shapes():
    assert len(I.INTERRUPTED_SOURCE_REPOSITORY_COMMIT) == 40
    for value in (
        I.INTERRUPTED_CLEAN_SOURCE_BINDING_DIGEST,
        I.INTERRUPTED_BOUND_IMPLEMENTATIONS_DIGEST,
        I.INTERRUPTED_SCORER_CONTRACT_DIGEST,
        I.RETAINED_PREIDENTITY_ARTIFACT["self_digest"],
        I.RETAINED_PREIDENTITY_ARTIFACT["raw_sha256"],
        *(row["self_digest"] for row in I.INTERRUPTED_AUTHORITIES.values()),
        *(row["raw_sha256"] for row in I.INTERRUPTED_AUTHORITIES.values()),
    ):
        assert len(value) == 64
        assert set(value) <= set("0123456789abcdef")


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _bound(payload: dict, key: str) -> dict:
    result = copy.deepcopy(payload)
    result[key] = I._digest(result)
    return result


def _zero() -> dict:
    return dict(I.ZERO_SCIENCE_FIELDS)


def _file_binding(
        *, label: str, path: Path, active: Path, archive: Path,
        managed: Path, key: str, payload: dict) -> dict:
    raw = path.read_bytes()
    return {
        "label": label,
        "managed_root": str(managed),
        "active_path": str(active),
        "archive_path": str(archive),
        "self_digest_key": key,
        "self_digest": payload[key],
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
    }


def _receipt_style(binding: dict, status: str) -> dict:
    return {
        "path": binding["active_path"],
        "receipt_digest": binding["self_digest"],
        "raw_sha256": binding["raw_sha256"],
        "byte_count": binding["byte_count"],
        "status": status,
    }


def _fixture(tmp_path: Path, monkeypatch) -> dict:
    (tmp_path / I.CORPUS_ROOT_RELATIVE).mkdir(parents=True)
    (tmp_path / I.SCORER_ROOT_RELATIVE).mkdir(parents=True)

    proof_bindings = []
    proof_paths = []
    for label, relative, raw in (
        ("candidate_allocator_source", "proof/allocator.py", b"allocator-v1\n"),
        ("candidate_allocation_amendment", "proof/amendment.json", b"{}\n"),
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        proof_paths.append(path)
        proof_bindings.append({
            "label": label,
            "path": relative,
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "byte_count": len(raw),
        })
    monkeypatch.setattr(
        I, "RETAINED_PREIDENTITY_PROOF_SOURCE_BINDINGS",
        tuple(proof_bindings))

    preidentity = {
        "schema": I.PREIDENTITY_SCHEMA,
        "status": "PASS_PRE_IDENTITY_STRUCTURAL_VALIDATION",
        "global": {"state_slot_count": 120, "candidate_slot_count": 720},
        "synthetic": True,
    }
    preidentity["pre_identity_validation_digest"] = I._allocator_digest(
        preidentity)
    preidentity_path = tmp_path / I.PREIDENTITY_RELATIVE_PATH
    _write(preidentity_path, preidentity)
    preidentity_raw = preidentity_path.read_bytes()
    preidentity_binding = {
        "managed_root": str(I.CORPUS_ROOT_RELATIVE),
        "path": str(I.PREIDENTITY_RELATIVE_PATH),
        "self_digest_key": "pre_identity_validation_digest",
        "self_digest": preidentity["pre_identity_validation_digest"],
        "raw_sha256": hashlib.sha256(preidentity_raw).hexdigest(),
        "byte_count": len(preidentity_raw),
    }
    monkeypatch.setattr(I, "RETAINED_PREIDENTITY_ARTIFACT",
                        preidentity_binding)

    authorities = {}
    payloads = {}

    def add(label, active, managed, key, payload):
        payload = _bound(payload, key)
        path = tmp_path / active
        _write(path, payload)
        archive_root = (I.SCORER_ARCHIVE_ROOT_RELATIVE
                        if managed == I.SCORER_ROOT_RELATIVE
                        else I.ARCHIVE_ROOT_RELATIVE)
        archive = archive_root / "authorities" / f"{label}.json"
        authorities[label] = _file_binding(
            label=label, path=path, active=active, archive=archive,
            managed=managed, key=key, payload=payload)
        payloads[label] = payload

    add(
        "performance_interruption", I.PERFORMANCE_RECEIPT_RELATIVE_PATH,
        I.CORPUS_ROOT_RELATIVE,
        "preoutcome_small_search_performance_interruption_receipt_digest",
        {
            "schema": I.PERFORMANCE_SCHEMA, "status": I.PERFORMANCE_STATUS,
            "superseding_source_repository_commit":
                I.INTERRUPTED_SOURCE_REPOSITORY_COMMIT,
            "superseding_clean_source_binding_digest":
                I.INTERRUPTED_CLEAN_SOURCE_BINDING_DIGEST,
            "superseding_bound_implementations_digest":
                I.INTERRUPTED_BOUND_IMPLEMENTATIONS_DIGEST,
            **_zero(),
        },
    )
    add(
        "projection_interruption", I.PROJECTION_RECEIPT_RELATIVE_PATH,
        I.CORPUS_ROOT_RELATIVE,
        "preoutcome_projection_fix_interruption_receipt_digest",
        {
            "schema": I.PROJECTION_SCHEMA, "status": I.PROJECTION_STATUS,
            "superseding_source_repository_commit":
                I.INTERRUPTED_SOURCE_REPOSITORY_COMMIT,
            "superseding_clean_source_binding_digest":
                I.INTERRUPTED_CLEAN_SOURCE_BINDING_DIGEST,
            "superseding_bound_implementations_digest":
                I.INTERRUPTED_BOUND_IMPLEMENTATIONS_DIGEST,
            **_zero(),
        },
    )
    add(
        "mixed_disposition", I.MIXED_DISPOSITION_RELATIVE_PATH,
        I.CORPUS_ROOT_RELATIVE,
        "mixed_precontract_disposition_receipt_digest",
        {
            "schema": I.MIXED_DISPOSITION_SCHEMA,
            "status": I.MIXED_DISPOSITION_STATUS, "complete": True,
            "source_repository_commit": I.INTERRUPTED_SOURCE_REPOSITORY_COMMIT,
            "clean_source_binding_digest":
                I.INTERRUPTED_CLEAN_SOURCE_BINDING_DIGEST,
            "bound_implementations_digest":
                I.INTERRUPTED_BOUND_IMPLEMENTATIONS_DIGEST,
            "retained_predecessor_state_count": 37,
            "rejected_predecessor_state_count": 8,
            "replacement_slot_count": 8,
            **_zero(),
        },
    )
    performance_binding = _receipt_style(
        authorities["performance_interruption"], I.PERFORMANCE_STATUS)
    projection_binding = _receipt_style(
        authorities["projection_interruption"], I.PROJECTION_STATUS)
    add(
        "scorer_contract", I.SCORER_CONTRACT_RELATIVE_PATH,
        I.SCORER_ROOT_RELATIVE, "contract_artifact_digest",
        {
            "schema": I.SCORER_CONTRACT_SCHEMA,
            "status": I.DEVELOPMENT_STATUS, "complete": True,
            "source_repository_commit": I.INTERRUPTED_SOURCE_REPOSITORY_COMMIT,
            "clean_source_binding_digest":
                I.INTERRUPTED_CLEAN_SOURCE_BINDING_DIGEST,
            "clean_source_binding": {
                "source_repository_commit":
                    I.INTERRUPTED_SOURCE_REPOSITORY_COMMIT,
                "bound_implementations_digest":
                    I.INTERRUPTED_BOUND_IMPLEMENTATIONS_DIGEST,
            },
            "scorer_contract_v1_2_digest":
                I.INTERRUPTED_SCORER_CONTRACT_DIGEST,
            "mixed_precontract_disposition_receipt_digest":
                authorities["mixed_disposition"]["self_digest"],
            "preoutcome_small_search_performance_interruption":
                performance_binding,
            "preoutcome_projection_fix_interruption": projection_binding,
        },
    )
    add(
        "clean_launch", I.CLEAN_LAUNCH_RELATIVE_PATH,
        I.CORPUS_ROOT_RELATIVE, "clean_source_launch_receipt_digest",
        {
            "schema": I.CLEAN_LAUNCH_SCHEMA,
            "status": I.DEVELOPMENT_STATUS, "complete": True,
            "source_repository_commit": I.INTERRUPTED_SOURCE_REPOSITORY_COMMIT,
            "clean_source_binding_digest":
                I.INTERRUPTED_CLEAN_SOURCE_BINDING_DIGEST,
            "bound_implementations_digest":
                I.INTERRUPTED_BOUND_IMPLEMENTATIONS_DIGEST,
            "scorer_contract_v1_2_digest":
                I.INTERRUPTED_SCORER_CONTRACT_DIGEST,
            "scorer_contract_artifact_digest":
                authorities["scorer_contract"]["self_digest"],
            "scorer_contract_artifact_sha256":
                authorities["scorer_contract"]["raw_sha256"],
            "mixed_precontract_disposition_receipt_digest":
                authorities["mixed_disposition"]["self_digest"],
            "pre_identity_allocation_validation_digest":
                preidentity_binding["self_digest"],
            "preoutcome_small_search_performance_interruption":
                performance_binding,
            "preoutcome_projection_fix_interruption": projection_binding,
        },
    )
    monkeypatch.setattr(I, "INTERRUPTED_AUTHORITIES", authorities)

    attestation = {"schema": "synthetic-zero-outcome-attestation"}
    attestation["attestation_digest"] = I._digest(attestation)
    attestation_calls = []

    def validate_attestation(value):
        attestation_calls.append(copy.deepcopy(dict(value)))
        if dict(value) != attestation:
            raise RuntimeError("synthetic attestation changed")

    monkeypatch.setattr(
        I.SELECTOR, "validate_phase1_outcome_surface_absence_attestation",
        validate_attestation)
    return {
        "root": tmp_path,
        "authorities": authorities,
        "payloads": payloads,
        "preidentity": preidentity,
        "preidentity_path": preidentity_path,
        "proof_paths": proof_paths,
        "attestation": attestation,
        "attestation_calls": attestation_calls,
    }


def _issue(fixture):
    return I.issue_and_archive_interruption_receipt(
        source_repository_commit=NEW_SOURCE_COMMIT,
        clean_source_binding_digest=NEW_CLEAN_DIGEST,
        bound_implementations_digest=NEW_IMPLEMENTATION_DIGEST,
        outcome_surface_absent=lambda: fixture["attestation"],
        root=fixture["root"],
    )


def test_issue_archives_exactly_five_and_exposes_read_only_bindings(
        tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    receipt = _issue(fixture)
    assert receipt["status"] == I.STATUS
    assert receipt["interrupted_authority_count"] == 5
    assert receipt["execution"] == I._execution_record(None, None)
    assert receipt["fixed_wrapper_count_issued"] == 0
    assert len(fixture["attestation_calls"]) >= 2

    for label, binding in fixture["authorities"].items():
        active, archive = I._binding_paths(binding, root=tmp_path)
        assert not active.exists()
        assert archive.is_file() and not archive.is_symlink()
        assert json.loads(archive.read_text()) == fixture["payloads"][label]
    assert fixture["preidentity_path"].is_file()

    loaded = I.load_and_validate_interruption_receipt(
        expected_source_repository_commit=NEW_SOURCE_COMMIT,
        expected_clean_source_binding_digest=NEW_CLEAN_DIGEST,
        expected_bound_implementations_digest=NEW_IMPLEMENTATION_DIGEST,
        root=tmp_path,
    )
    assert loaded == receipt
    assert set(I.receipt_binding(receipt, root=tmp_path)) == {
        "path", "receipt_digest", "raw_sha256", "byte_count", "status",
    }
    assert I.load_archived_performance_receipt_v1(
        receipt, root=tmp_path) == fixture["payloads"][
            "performance_interruption"]
    archived_binding = I.archived_performance_receipt_binding_v1(
        receipt, root=tmp_path)
    assert archived_binding == {
        "path": fixture["authorities"]["performance_interruption"][
            "archive_path"],
        "receipt_digest": fixture["authorities"][
            "performance_interruption"]["self_digest"],
        "raw_sha256": fixture["authorities"][
            "performance_interruption"]["raw_sha256"],
        "byte_count": fixture["authorities"][
            "performance_interruption"]["byte_count"],
        "status": I.PERFORMANCE_STATUS,
    }

    first = I.validate_retained_preidentity_artifact(receipt, root=tmp_path)
    first["synthetic"] = False
    assert I.validate_retained_preidentity_artifact(
        receipt, root=tmp_path) == fixture["preidentity"]

    # Absence is an issuance-time fact.  A later legitimate wrapper does not
    # invalidate the immutable transition or make its validator mutate state.
    wrapper = tmp_path / I.FIXED_WRAPPER_ACTIVE_PATHS[0]
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text("later-successor-wrapper\n")
    assert I.load_and_validate_interruption_receipt(
        expected_source_repository_commit=NEW_SOURCE_COMMIT,
        expected_clean_source_binding_digest=NEW_CLEAN_DIGEST,
        expected_bound_implementations_digest=NEW_IMPLEMENTATION_DIGEST,
        root=tmp_path,
    ) == receipt

    # Projection, disposition, contract, and launch deliberately reuse their
    # canonical active paths under the successor source.  Historical
    # validation must keep reading the exact archives without treating those
    # independently validated successor bytes as an old-authority collision.
    for label in I.SUCCESSOR_REUSABLE_ACTIVE_AUTHORITY_LABELS:
        active, _archive = I._binding_paths(
            fixture["authorities"][label], root=tmp_path)
        _write(active, {"successor_authority": label})
    assert I.load_and_validate_interruption_receipt(
        expected_source_repository_commit=NEW_SOURCE_COMMIT,
        expected_clean_source_binding_digest=NEW_CLEAN_DIGEST,
        expected_bound_implementations_digest=NEW_IMPLEMENTATION_DIGEST,
        root=tmp_path,
    ) == receipt

    # V2 performance uses a distinct path.  Reoccupation of the archived V1
    # active name therefore remains an unexplained collision and fails closed.
    performance_active, _archive = I._binding_paths(
        fixture["authorities"]["performance_interruption"], root=tmp_path)
    _write(performance_active, {"unexpected": "V1 active replacement"})
    with pytest.raises(
            I.FixedReissueValidationInterruptionError,
            match="performance_interruption active path is a collision"):
        I.load_and_validate_interruption_receipt(
            expected_source_repository_commit=NEW_SOURCE_COMMIT,
            expected_clean_source_binding_digest=NEW_CLEAN_DIGEST,
            expected_bound_implementations_digest=NEW_IMPLEMENTATION_DIGEST,
            root=tmp_path,
        )


def test_all_inputs_are_validated_before_first_archive(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    final = fixture["authorities"]["clean_launch"]
    final_path = tmp_path / final["active_path"]
    final_path.write_bytes(final_path.read_bytes() + b"tamper")
    with pytest.raises(
            I.FixedReissueValidationInterruptionError, match="collision"):
        _issue(fixture)
    for binding in fixture["authorities"].values():
        active, archive = I._binding_paths(binding, root=tmp_path)
        assert active.exists()
        assert not archive.exists()
    assert not (tmp_path / I.RECEIPT_RELATIVE_PATH).exists()


def test_outcome_surface_is_rechecked_after_archives_before_receipt_install(
        tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    changed = {
        "schema": "synthetic-zero-outcome-attestation",
        "concurrent_forbidden_outcome": True,
    }
    changed["attestation_digest"] = I._digest(changed)
    observations = iter([fixture["attestation"], changed])
    monkeypatch.setattr(
        I.SELECTOR, "validate_phase1_outcome_surface_absence_attestation",
        lambda _value: None)

    with pytest.raises(
            I.FixedReissueValidationInterruptionError,
            match="outcome-surface absence changed"):
        I.issue_and_archive_interruption_receipt(
            source_repository_commit=NEW_SOURCE_COMMIT,
            clean_source_binding_digest=NEW_CLEAN_DIGEST,
            bound_implementations_digest=NEW_IMPLEMENTATION_DIGEST,
            outcome_surface_absent=lambda: next(observations),
            root=tmp_path,
        )
    assert not (tmp_path / I.RECEIPT_RELATIVE_PATH).exists()
    for binding in fixture["authorities"].values():
        active, archive = I._binding_paths(binding, root=tmp_path)
        assert not active.exists()
        assert archive.is_file() and not archive.is_symlink()


def test_existing_wrapper_blocks_issuance_without_archiving(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    wrapper = tmp_path / I.FIXED_WRAPPER_ACTIVE_PATHS[2]
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text("collision\n")
    with pytest.raises(
            I.FixedReissueValidationInterruptionError,
            match="pre-issuance output already exists"):
        _issue(fixture)
    for binding in fixture["authorities"].values():
        active, archive = I._binding_paths(binding, root=tmp_path)
        assert active.exists()
        assert not archive.exists()


def test_issuer_recovers_hardlink_then_unlink_crash(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    first = fixture["authorities"][I.AUTHORITY_LABELS[0]]
    active, archive = I._binding_paths(first, root=tmp_path)
    archive.parent.mkdir(parents=True)
    os.link(active, archive)
    assert os.path.samefile(active, archive)
    receipt = _issue(fixture)
    assert not active.exists()
    assert archive.is_file()
    assert I.load_and_validate_interruption_receipt(
        expected_source_repository_commit=NEW_SOURCE_COMMIT,
        expected_clean_source_binding_digest=NEW_CLEAN_DIGEST,
        expected_bound_implementations_digest=NEW_IMPLEMENTATION_DIGEST,
        root=tmp_path,
    ) == receipt


def test_public_validation_is_read_only_on_active_archive_collision(
        tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    receipt = _issue(fixture)
    first = fixture["authorities"][I.AUTHORITY_LABELS[0]]
    active, archive = I._binding_paths(first, root=tmp_path)
    active.parent.mkdir(parents=True, exist_ok=True)
    os.link(archive, active)
    with pytest.raises(
            I.FixedReissueValidationInterruptionError, match="collision"):
        I.validate_interruption_receipt(
            receipt,
            expected_source_repository_commit=NEW_SOURCE_COMMIT,
            expected_clean_source_binding_digest=NEW_CLEAN_DIGEST,
            expected_bound_implementations_digest=NEW_IMPLEMENTATION_DIGEST,
            root=tmp_path,
        )
    assert active.is_file() and archive.is_file()
    assert os.path.samefile(active, archive)


def test_receipt_collision_is_never_overwritten(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    path = tmp_path / I.RECEIPT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    original = b"not-json\n"
    path.write_bytes(original)
    with pytest.raises(I.FixedReissueValidationInterruptionError):
        _issue(fixture)
    assert path.read_bytes() == original
    for binding in fixture["authorities"].values():
        assert (tmp_path / binding["active_path"]).is_file()


@pytest.mark.parametrize("target", ("preidentity", "allocator", "amendment"))
def test_solve_free_preidentity_reuse_rejects_each_changed_binding(
        target, tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    receipt = _issue(fixture)
    if target == "preidentity":
        path = fixture["preidentity_path"]
    elif target == "allocator":
        path = fixture["proof_paths"][0]
    else:
        path = fixture["proof_paths"][1]
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(
            I.FixedReissueValidationInterruptionError,
            match="pre-identity"):
        I.validate_retained_preidentity_artifact(receipt, root=tmp_path)


def test_execution_evidence_is_frozen_not_self_resignable(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    with pytest.raises(
            I.FixedReissueValidationInterruptionError, match="argv"):
        I.issue_and_archive_interruption_receipt(
            source_repository_commit=NEW_SOURCE_COMMIT,
            clean_source_binding_digest=NEW_CLEAN_DIGEST,
            bound_implementations_digest=NEW_IMPLEMENTATION_DIGEST,
            outcome_surface_absent=lambda: fixture["attestation"],
            execution_argv=["invented"], root=tmp_path)
    with pytest.raises(
            I.FixedReissueValidationInterruptionError,
            match="interpreter versions"):
        I.issue_and_archive_interruption_receipt(
            source_repository_commit=NEW_SOURCE_COMMIT,
            clean_source_binding_digest=NEW_CLEAN_DIGEST,
            bound_implementations_digest=NEW_IMPLEMENTATION_DIGEST,
            outcome_surface_absent=lambda: fixture["attestation"],
            interpreter_versions={
                "python": "3.12.3", "numpy": "wrong", "scipy": "1.17.1",
            }, root=tmp_path)
    assert all((tmp_path / binding["active_path"]).is_file()
               for binding in fixture["authorities"].values())


def test_lineage_contract_is_non_scientific_and_exact():
    contract = I.lineage_contract()
    assert contract["interrupted_authority_count"] == 5
    assert contract["fixed_wrapper_active_path_count"] == 7
    assert contract["wrapper_loop_entered"] is False
    assert contract["execution_exit_code"] == 130
    assert contract["pid_recorded"] is False
    assert contract["preidentity_milp_validation_rerun"] is False
    assert contract["new_candidate_allocation_performed"] is False
    assert contract["scientific_gate_input"] is False
