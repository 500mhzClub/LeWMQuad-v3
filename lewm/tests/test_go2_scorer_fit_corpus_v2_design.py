from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import stat

import pytest

from lewm.oracle import go2_scorer_fit_corpus_v2_design as design


COMMIT = "a" * 40


def _sources() -> list[dict[str, object]]:
    return [
        {
            "path": path,
            "role": role,
            "byte_count": index + 1,
            "sha256": f"{index + 1:064x}",
        }
        for index, (path, role) in enumerate(design.SOURCE_SPECS)
    ]


def _classification() -> dict[str, object]:
    return design.build_rotation_mask_classification(
        source_repository_commit=COMMIT,
        source_bindings=_sources(),
        predecessor_validation=design.PREDECESSOR_VALIDATION_PROJECTION,
    )


def _classification_binding() -> dict[str, object]:
    payload = _classification()
    raw = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    return design.rotation_mask_classification_artifact_binding(payload, raw)


def _amendment() -> dict[str, object]:
    return design.build_design_amendment(
        source_repository_commit=COMMIT,
        source_bindings=_sources(),
        rotation_mask_classification_binding=_classification_binding(),
        predecessor_validation=design.PREDECESSOR_VALIDATION_PROJECTION,
    )


def _issued_design_authority() -> dict[str, object]:
    sources = _sources()
    classification = design.build_rotation_mask_classification(
        source_repository_commit=
            design.ISSUED_FULL_BANK_V2_SOURCE_REPOSITORY_COMMIT,
        source_bindings=sources,
        predecessor_validation=design.PREDECESSOR_VALIDATION_PROJECTION,
    )
    classification_raw = (
        json.dumps(classification, sort_keys=True, indent=2) + "\n").encode()
    classification_binding = (
        design.rotation_mask_classification_artifact_binding(
            classification, classification_raw))
    amendment = design.build_design_amendment(
        source_repository_commit=
            design.ISSUED_FULL_BANK_V2_SOURCE_REPOSITORY_COMMIT,
        source_bindings=sources,
        rotation_mask_classification_binding=classification_binding,
        predecessor_validation=design.PREDECESSOR_VALIDATION_PROJECTION,
    )
    amendment_raw = (
        json.dumps(amendment, sort_keys=True, indent=2) + "\n").encode()
    return design.validate_immutable_issued_design_authority({
        "rotation_mask_classification_payload": classification,
        "rotation_mask_classification_binding": classification_binding,
        "design_amendment_payload": amendment,
        "design_amendment_binding": design.design_amendment_artifact_binding(
            amendment, amendment_raw),
    })


def _corrected_sources_v1() -> list[dict[str, object]]:
    rows = _sources()
    changed = set(design.SOURCE_CORRECTION_V1_ALLOWED_CHANGED_SOURCE_PATHS)
    for index, row in enumerate(rows):
        if row["path"] in changed:
            row["byte_count"] = int(row["byte_count"]) + 10_000
            row["sha256"] = f"{index + 10_000:064x}"
    return rows


def _source_correction_v1() -> dict[str, object]:
    return design.build_preselection_source_correction_v1(
        source_repository_commit=
            design.IMMUTABLE_SOURCE_CORRECTION_V1_SOURCE_REPOSITORY_COMMIT,
        source_bindings=_corrected_sources_v1(),
        immutable_issued_design_authority=_issued_design_authority(),
        runtime_outputs_absent_at_issue=design._expected_absence_rows(
            phase="design"),
    )


def _immutable_source_correction_v1(
        monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    payload = _source_correction_v1()
    monkeypatch.setattr(
        design, "IMMUTABLE_SOURCE_CORRECTION_V1_DIGEST",
        payload[design.SOURCE_CORRECTION_SELF_KEY])
    raw = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    return design.validate_immutable_preselection_source_correction_v1({
        "payload": payload,
        "binding": design.preselection_source_correction_v1_artifact_binding(
            payload, raw),
    })


def _corrected_sources_v2() -> list[dict[str, object]]:
    rows = _corrected_sources_v1()
    changed = set(design.SOURCE_CORRECTION_V2_ALLOWED_CHANGED_SOURCE_PATHS)
    for index, row in enumerate(rows):
        if row["path"] in changed:
            row["byte_count"] = int(row["byte_count"]) + 20_000
            row["sha256"] = f"{index + 20_000:064x}"
    return rows


def _source_correction_v2(
        monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    immutable_v1 = _immutable_source_correction_v1(monkeypatch)
    return design.build_preselection_source_correction_v2(
        source_repository_commit=
            design.IMMUTABLE_SOURCE_CORRECTION_V2_SOURCE_REPOSITORY_COMMIT,
        source_bindings=_corrected_sources_v2(),
        immutable_preselection_source_correction_v1=immutable_v1,
        runtime_outputs_absent_at_issue=design._expected_absence_rows(
            phase="design"),
    )


def _immutable_source_correction_v2(
        monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    payload = _source_correction_v2(monkeypatch)
    monkeypatch.setattr(
        design, "IMMUTABLE_SOURCE_CORRECTION_V2_DIGEST",
        payload[design.SOURCE_CORRECTION_SELF_KEY])
    raw = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    return design.validate_immutable_preselection_source_correction_v2({
        "payload": payload,
        "binding": design.preselection_source_correction_v2_artifact_binding(
            payload, raw),
    })


def _corrected_sources_final() -> list[dict[str, object]]:
    rows = _corrected_sources_v2()
    changed = set(design.SOURCE_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS)
    for index, row in enumerate(rows):
        if row["path"] in changed:
            row["byte_count"] = int(row["byte_count"]) + 30_000
            row["sha256"] = f"{index + 30_000:064x}"
    return rows


def _source_correction_final(
        monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    immutable_v2 = _immutable_source_correction_v2(monkeypatch)
    return design.build_preselection_source_correction(
        source_repository_commit=
            design.IMMUTABLE_ACTIVE_PRESELECTION_SOURCE_REPOSITORY_COMMIT,
        source_bindings=_corrected_sources_final(),
        immutable_preselection_source_correction_v2=immutable_v2,
        runtime_outputs_absent_at_issue=design._expected_absence_rows(
            phase="design"),
    )


def _immutable_active_preselection_source_correction(
        monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    payload = _source_correction_final(monkeypatch)
    raw = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    binding = design.preselection_source_correction_artifact_binding(
        payload, raw)
    monkeypatch.setattr(
        design, "IMMUTABLE_ACTIVE_PRESELECTION_SOURCE_CORRECTION_DIGEST",
        payload[design.SOURCE_CORRECTION_SELF_KEY])
    monkeypatch.setattr(
        design, "IMMUTABLE_ACTIVE_PRESELECTION_SOURCE_CORRECTION_BINDING",
        copy.deepcopy(binding))
    failure = copy.deepcopy(design.MANIFEST_REPLAY_FAILURE_BOUNDARY)
    failure["active_preselection_correction_digest"] = payload[
        design.SOURCE_CORRECTION_SELF_KEY]
    monkeypatch.setattr(
        design, "MANIFEST_REPLAY_FAILURE_BOUNDARY", failure)
    return design.validate_immutable_active_preselection_source_correction({
        "payload": payload,
        "binding": binding,
    })


def _corrected_sources_replay() -> list[dict[str, object]]:
    rows = _corrected_sources_final()
    changed = set(
        design.MANIFEST_REPLAY_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS)
    for index, row in enumerate(rows):
        if row["path"] in changed:
            row["byte_count"] = int(row["byte_count"]) + 40_000
            row["sha256"] = f"{index + 40_000:064x}"
    return rows


def _manifest_replay_correction(
        monkeypatch: pytest.MonkeyPatch, *, source_commit: str = "e" * 40,
        ) -> dict[str, object]:
    immutable = _immutable_active_preselection_source_correction(monkeypatch)
    return design.build_manifest_replay_correction(
        source_repository_commit=source_commit,
        source_bindings=_corrected_sources_replay(),
        immutable_active_preselection_source_correction=immutable,
        installed_preoutcome_artifact_bindings=
            design.INSTALLED_FULL_BANK_V2_PREOUTCOME_ARTIFACT_BINDINGS,
        successor_and_runtime_outputs_absent_at_issue=
            design._expected_absence_rows(phase="successor_contract"),
    )


def _encoder_import_correction(
        monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    replay = _manifest_replay_correction(
        monkeypatch,
        source_commit=
            design.ENCODER_IMPORT_CORRECTION_HISTORICAL_SOURCE_REPOSITORY_COMMIT)
    monkeypatch.setattr(
        design, "IMMUTABLE_MANIFEST_REPLAY_CORRECTION_DIGEST",
        replay[design.MANIFEST_REPLAY_CORRECTION_SELF_KEY])
    replay_raw = (json.dumps(replay, sort_keys=True, indent=2) + "\n").encode()
    immutable_replay = {
        "payload": replay,
        "binding": design.manifest_replay_correction_artifact_binding(
            replay, replay_raw),
    }
    sources = copy.deepcopy(replay["source_bindings"])
    changed_base = set(
        design.ENCODER_IMPORT_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS
    ).intersection(design.EXPECTED_SOURCE_PATHS)
    for index, row in enumerate(sources):
        if row["path"] in changed_base:
            row["byte_count"] = int(row["byte_count"]) + 50_000
            row["sha256"] = f"{index + 50_000:064x}"
    old_dev = copy.deepcopy(
        design.ENCODER_IMPORT_CORRECTION_DEV_ENCODER_HISTORICAL_BINDING)
    new_dev = copy.deepcopy(old_dev)
    new_dev["byte_count"] = int(new_dev["byte_count"]) + 1
    new_dev["sha256"] = "8" * 64
    dev_transition = {
        "path": old_dev["path"], "role": old_dev["role"],
        "historical": old_dev, "current": new_dev,
    }
    tests = []
    for index, (path, role) in enumerate(
            design.ENCODER_IMPORT_CORRECTION_FOCUSED_TEST_SPECS):
        historical = {
            "path": path, "role": role, "exists": index != 3,
            "byte_count": 0 if index == 3 else index + 1,
            "sha256": None if index == 3 else f"{index + 1:064x}",
        }
        current = {
            "path": path, "role": role, "exists": True,
            "byte_count": index + 101, "sha256": f"{index + 101:064x}",
        }
        tests.append({
            "path": path, "role": role,
            "historical": historical, "current": current,
        })
    return design.build_encoder_import_correction(
        source_repository_commit="f" * 40,
        source_bindings=sources,
        immutable_manifest_replay_correction=immutable_replay,
        immutable_successor_scorer_contract_binding=
            design.IMMUTABLE_SUCCESSOR_SCORER_CONTRACT_BINDING,
        dev_encoder_source_transition=dev_transition,
        focused_test_source_transitions=tests,
        branch_smoke_binding=
            design.IMMUTABLE_ENCODER_IMPORT_FAILURE_BRANCH_SMOKE_BINDING,
        branch_corpus_binding=
            design.IMMUTABLE_ENCODER_IMPORT_FAILURE_CORPUS_RECEIPT_BINDING,
        prelatent_outputs_absent_at_issue=
            design._expected_encoder_import_correction_absence_rows(),
    )


def _encoder_compute_dtype_correction(
        monkeypatch: pytest.MonkeyPatch, *,
        encoder_import: dict[str, object] | None = None,
        ) -> dict[str, object]:
    encoder_import = (
        _encoder_import_correction(monkeypatch)
        if encoder_import is None else copy.deepcopy(encoder_import))
    historical_commit = str(encoder_import["source_repository_commit"])
    monkeypatch.setattr(
        design,
        "ENCODER_COMPUTE_DTYPE_CORRECTION_HISTORICAL_SOURCE_REPOSITORY_COMMIT",
        historical_commit)
    import_digest = encoder_import[design.ENCODER_IMPORT_CORRECTION_SELF_KEY]
    import_raw = (
        json.dumps(encoder_import, sort_keys=True, indent=2) + "\n").encode()
    import_binding = design.encoder_import_correction_artifact_binding(
        encoder_import, import_raw)
    monkeypatch.setattr(
        design, "IMMUTABLE_ENCODER_IMPORT_CORRECTION_DIGEST", import_digest)
    monkeypatch.setattr(
        design, "IMMUTABLE_ENCODER_IMPORT_CORRECTION_BINDING",
        copy.deepcopy(import_binding))
    science = copy.deepcopy(
        design.ENCODER_COMPUTE_DTYPE_CORRECTION_PRESERVED_SCIENCE)
    science["encoder_import_correction_digest"] = import_digest
    monkeypatch.setattr(
        design, "ENCODER_COMPUTE_DTYPE_CORRECTION_PRESERVED_SCIENCE", science)
    failure = copy.deepcopy(design.ENCODER_COMPUTE_DTYPE_FAILURE_BOUNDARY)
    failure["historical_source_repository_commit"] = historical_commit
    monkeypatch.setattr(
        design, "ENCODER_COMPUTE_DTYPE_FAILURE_BOUNDARY", failure)

    historical_sources = encoder_import["source_bindings"]
    failed_row = next(
        row for row in historical_sources
        if row["path"] == "scripts/encode_go2_branch_corpus_v1_2.py")
    failed_binding = {
        "path": failed_row["path"],
        "role": "failed_full_bank_v2_bfloat16_encoder_route",
        "exists": True,
        "byte_count": failed_row["byte_count"],
        "sha256": failed_row["sha256"],
    }
    monkeypatch.setattr(
        design, "ENCODER_COMPUTE_DTYPE_FAILURE_ENCODER_SOURCE_BINDING",
        failed_binding)
    dev_binding = copy.deepcopy(
        encoder_import["dev_encoder_source_transition"]["current"])
    monkeypatch.setattr(
        design, "ENCODER_COMPUTE_DTYPE_UNCHANGED_DEV_ENCODER_BINDING",
        dev_binding)

    sources = copy.deepcopy(historical_sources)
    changed = set(
        design.ENCODER_COMPUTE_DTYPE_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS)
    for index, row in enumerate(sources):
        if row["path"] in changed:
            row["byte_count"] = int(row["byte_count"]) + 60_000
            row["sha256"] = f"{index + 60_000:064x}"
    tests = []
    for index, (path, role) in enumerate(
            design.ENCODER_COMPUTE_DTYPE_CORRECTION_FOCUSED_TEST_SPECS):
        historical = {
            "path": path, "role": role, "exists": True,
            "byte_count": index + 201, "sha256": f"{index + 201:064x}",
        }
        current = {
            "path": path, "role": role, "exists": True,
            "byte_count": index + 301, "sha256": f"{index + 301:064x}",
        }
        tests.append({
            "path": path, "role": role,
            "historical": historical, "current": current,
        })
    return design.build_encoder_compute_dtype_correction(
        source_repository_commit="1" * 40,
        source_bindings=sources,
        immutable_encoder_import_correction={
            "payload": encoder_import, "binding": import_binding,
        },
        immutable_successor_scorer_contract_binding=
            design.IMMUTABLE_SUCCESSOR_SCORER_CONTRACT_BINDING,
        focused_test_source_transitions=tests,
        branch_smoke_binding=
            design.IMMUTABLE_ENCODER_IMPORT_FAILURE_BRANCH_SMOKE_BINDING,
        branch_corpus_binding=
            design.IMMUTABLE_ENCODER_IMPORT_FAILURE_CORPUS_RECEIPT_BINDING,
        failed_encoder_source_binding=failed_binding,
        unchanged_dev_encoder_source_binding=dev_binding,
        unchanged_stage_a_fp32_source_binding=
            design.ENCODER_COMPUTE_DTYPE_STAGE_A_FP32_SOURCE_BINDING,
        upstream_rope_source_binding=
            design.ENCODER_COMPUTE_DTYPE_UPSTREAM_ROPE_SOURCE_BINDING,
        prelatent_outputs_absent_at_issue=
            design._expected_encoder_compute_dtype_correction_absence_rows(),
    )


def _encoder_path_projection_correction(
        monkeypatch: pytest.MonkeyPatch, *,
        dtype_correction: dict[str, object] | None = None,
        ) -> dict[str, object]:
    dtype_correction = (
        _encoder_compute_dtype_correction(monkeypatch)
        if dtype_correction is None else copy.deepcopy(dtype_correction))
    historical_commit = str(dtype_correction["source_repository_commit"])
    monkeypatch.setattr(
        design,
        "ENCODER_PATH_PROJECTION_CORRECTION_HISTORICAL_SOURCE_REPOSITORY_COMMIT",
        historical_commit)
    dtype_digest = dtype_correction[
        design.ENCODER_COMPUTE_DTYPE_CORRECTION_SELF_KEY]
    dtype_raw = (
        json.dumps(dtype_correction, sort_keys=True, indent=2) + "\n").encode()
    dtype_binding = design.encoder_compute_dtype_correction_artifact_binding(
        dtype_correction, dtype_raw)
    monkeypatch.setattr(
        design, "IMMUTABLE_ENCODER_COMPUTE_DTYPE_CORRECTION_DIGEST",
        dtype_digest)
    monkeypatch.setattr(
        design, "IMMUTABLE_ENCODER_COMPUTE_DTYPE_CORRECTION_BINDING",
        copy.deepcopy(dtype_binding))
    science = copy.deepcopy(
        design.ENCODER_PATH_PROJECTION_CORRECTION_PRESERVED_SCIENCE)
    science["encoder_compute_dtype_correction_digest"] = dtype_digest
    monkeypatch.setattr(
        design, "ENCODER_PATH_PROJECTION_CORRECTION_PRESERVED_SCIENCE",
        science)
    failure = copy.deepcopy(design.ENCODER_PATH_PROJECTION_FAILURE_BOUNDARY)
    failure["historical_source_repository_commit"] = historical_commit
    monkeypatch.setattr(
        design, "ENCODER_PATH_PROJECTION_FAILURE_BOUNDARY", failure)

    historical_sources = dtype_correction["source_bindings"]
    failed_row = next(
        row for row in historical_sources
        if row["path"] == "scripts/encode_go2_branch_corpus_v1_2.py")
    failed_binding = {
        "path": failed_row["path"],
        "role": "failed_physical_to_repository_root_path_projection_route",
        "exists": True,
        "byte_count": failed_row["byte_count"],
        "sha256": failed_row["sha256"],
    }
    monkeypatch.setattr(
        design, "ENCODER_PATH_PROJECTION_FAILURE_ENCODER_SOURCE_BINDING",
        failed_binding)
    sources = copy.deepcopy(historical_sources)
    changed = set(
        design.ENCODER_PATH_PROJECTION_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS)
    for index, row in enumerate(sources):
        if row["path"] in changed:
            row["byte_count"] = int(row["byte_count"]) + 70_000
            row["sha256"] = f"{index + 70_000:064x}"
    tests = []
    for index, (path, role) in enumerate(
            design.ENCODER_PATH_PROJECTION_CORRECTION_FOCUSED_TEST_SPECS):
        historical = {
            "path": path, "role": role, "exists": True,
            "byte_count": index + 401, "sha256": f"{index + 401:064x}",
        }
        current = {
            "path": path, "role": role, "exists": True,
            "byte_count": index + 501, "sha256": f"{index + 501:064x}",
        }
        tests.append({
            "path": path, "role": role,
            "historical": historical, "current": current,
        })
    return design.build_encoder_path_projection_correction(
        source_repository_commit="2" * 40,
        source_bindings=sources,
        immutable_encoder_compute_dtype_correction={
            "payload": dtype_correction, "binding": dtype_binding,
        },
        immutable_successor_scorer_contract_binding=
            design.IMMUTABLE_SUCCESSOR_SCORER_CONTRACT_BINDING,
        focused_test_source_transitions=tests,
        failed_encoder_source_binding=failed_binding,
        base_smoke_artifact_bundle=
            design.IMMUTABLE_ENCODER_PATH_PROJECTION_BASE_ARTIFACT_BUNDLE,
        downstream_outputs_absent_at_issue=
            design._expected_encoder_path_projection_correction_absence_rows(),
        single_shard_regeneration_transaction_artifacts_absent_at_issue=
            design._expected_encoder_path_projection_transaction_absence_rows(),
    )


def _branch_redrive_projection_correction(
        monkeypatch: pytest.MonkeyPatch, *,
        path_correction: dict[str, object] | None = None,
        ) -> dict[str, object]:
    path_correction = (
        _encoder_path_projection_correction(monkeypatch)
        if path_correction is None else copy.deepcopy(path_correction))
    historical_commit = str(path_correction["source_repository_commit"])
    path_digest = path_correction[
        design.ENCODER_PATH_PROJECTION_CORRECTION_SELF_KEY]
    path_raw = (
        json.dumps(path_correction, sort_keys=True, indent=2) + "\n").encode()
    path_binding = design.encoder_path_projection_correction_artifact_binding(
        path_correction, path_raw)
    monkeypatch.setattr(
        design,
        "BRANCH_REDRIVE_PROJECTION_CORRECTION_HISTORICAL_SOURCE_REPOSITORY_COMMIT",
        historical_commit)
    monkeypatch.setattr(
        design, "IMMUTABLE_ENCODER_PATH_PROJECTION_CORRECTION_DIGEST",
        path_digest)
    monkeypatch.setattr(
        design, "IMMUTABLE_ENCODER_PATH_PROJECTION_CORRECTION_BINDING",
        copy.deepcopy(path_binding))
    failure = copy.deepcopy(design.BRANCH_REDRIVE_PROJECTION_FAILURE_BOUNDARY)
    failure["historical_source_repository_commit"] = historical_commit
    monkeypatch.setattr(
        design, "BRANCH_REDRIVE_PROJECTION_FAILURE_BOUNDARY", failure)
    science = copy.deepcopy(
        design.BRANCH_REDRIVE_PROJECTION_CORRECTION_PRESERVED_SCIENCE)
    science["immutable_encoder_path_projection_correction_digest"] = path_digest
    monkeypatch.setattr(
        design, "BRANCH_REDRIVE_PROJECTION_CORRECTION_PRESERVED_SCIENCE",
        science)
    sources = copy.deepcopy(path_correction["source_bindings"])
    changed = set(
        design.BRANCH_REDRIVE_PROJECTION_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS)
    for index, row in enumerate(sources):
        if row["path"] in changed:
            row["byte_count"] = int(row["byte_count"]) + 80_000
            row["sha256"] = f"{index + 80_000:064x}"
    return design.build_branch_redrive_projection_correction(
        source_repository_commit="3" * 40,
        source_bindings=sources,
        immutable_encoder_path_projection_correction={
            "payload": path_correction, "binding": path_binding,
        },
        partial_corpus_failure_boundary=
            design.IMMUTABLE_BRANCH_REDRIVE_PARTIAL_CORPUS_BINDING,
        invalid_attempt_receipt_bindings=
            design.IMMUTABLE_BRANCH_REDRIVE_INVALID_ATTEMPT_RECEIPT_BINDINGS,
        completed_smoke_boundary=
            design.IMMUTABLE_BRANCH_REDRIVE_COMPLETED_SMOKE_BUNDLE,
        downstream_outputs_absent_at_issue=
            design._expected_branch_redrive_projection_correction_absence_rows(),
    )


def _optional_smoke_partial_corpus_resume_correction(
        monkeypatch: pytest.MonkeyPatch, *,
        redrive_correction: dict[str, object] | None = None,
        ) -> dict[str, object]:
    redrive_correction = (
        _branch_redrive_projection_correction(monkeypatch)
        if redrive_correction is None else copy.deepcopy(redrive_correction))
    historical_commit = str(redrive_correction["source_repository_commit"])
    redrive_digest = redrive_correction[
        design.BRANCH_REDRIVE_PROJECTION_CORRECTION_SELF_KEY]
    redrive_raw = design._pretty_json_bytes(redrive_correction)
    redrive_binding = (
        design.branch_redrive_projection_correction_artifact_binding(
            redrive_correction, redrive_raw))
    monkeypatch.setattr(
        design,
        "OPTIONAL_SMOKE_PARTIAL_CORPUS_RESUME_CORRECTION_"
        "HISTORICAL_SOURCE_REPOSITORY_COMMIT",
        historical_commit)
    monkeypatch.setattr(
        design, "IMMUTABLE_BRANCH_REDRIVE_PROJECTION_CORRECTION_DIGEST",
        redrive_digest)
    monkeypatch.setattr(
        design, "IMMUTABLE_BRANCH_REDRIVE_PROJECTION_CORRECTION_BINDING",
        copy.deepcopy(redrive_binding))
    failure = copy.deepcopy(
        design.OPTIONAL_SMOKE_PARTIAL_CORPUS_RESUME_FAILURE_BOUNDARY)
    failure["historical_source_repository_commit"] = historical_commit
    monkeypatch.setattr(
        design, "OPTIONAL_SMOKE_PARTIAL_CORPUS_RESUME_FAILURE_BOUNDARY",
        failure)
    science = copy.deepcopy(
        design.OPTIONAL_SMOKE_PARTIAL_CORPUS_RESUME_CORRECTION_PRESERVED_SCIENCE)
    science["immutable_branch_redrive_projection_correction_digest"] = (
        redrive_digest)
    monkeypatch.setattr(
        design,
        "OPTIONAL_SMOKE_PARTIAL_CORPUS_RESUME_CORRECTION_PRESERVED_SCIENCE",
        science)
    sources = copy.deepcopy(redrive_correction["source_bindings"])
    changed = set(
        design.OPTIONAL_SMOKE_PARTIAL_CORPUS_RESUME_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS)
    for index, row in enumerate(sources):
        if row["path"] in changed:
            row["byte_count"] = int(row["byte_count"]) + 90_000
            row["sha256"] = f"{index + 90_000:064x}"
    tests = []
    for index, (path, role) in enumerate(
            design.OPTIONAL_SMOKE_PARTIAL_CORPUS_RESUME_CORRECTION_FOCUSED_TEST_SPECS):
        tests.append({
            "path": path,
            "role": role,
            "historical": {
                "path": path, "role": role, "exists": True,
                "byte_count": 100 + index, "sha256": f"{index + 1:064x}",
            },
            "current": {
                "path": path, "role": role, "exists": True,
                "byte_count": 200 + index, "sha256": f"{index + 101:064x}",
            },
        })
    return design.build_optional_smoke_partial_corpus_resume_correction(
        source_repository_commit="4" * 40,
        source_bindings=sources,
        focused_test_source_transitions=tests,
        immutable_branch_redrive_projection_correction={
            "payload": redrive_correction, "binding": redrive_binding,
        },
        downstream_outputs_absent_at_issue=(
            design
            ._expected_optional_smoke_partial_corpus_resume_correction_absence_rows()),
    )


def _synthetic_smoke_regeneration_receipts(
        correction: dict[str, object],
        ) -> tuple[dict[str, object], dict[str, object]]:
    lineage = {
        "scorer_fit_corpus_v2_scorer_contract_digest":
            design.IMMUTABLE_SUCCESSOR_SCORER_CONTRACT_BINDING[
                "embedded_contract_self_digest"],
        "scorer_fit_corpus_v2_scorer_contract_artifact_digest":
            design.IMMUTABLE_SUCCESSOR_SCORER_CONTRACT_BINDING["self_digest"],
        "state_manifest_digest": "1" * 64,
        "full_bank_assignment_manifest_digest": "2" * 64,
        "corpus_digest": "3" * 64,
        "branch_smoke_receipt_digest": "4" * 64,
        "encoder_compute_dtype_correction_digest":
            design.IMMUTABLE_ENCODER_COMPUTE_DTYPE_CORRECTION_DIGEST,
        "encoder_path_projection_correction_digest": correction[
            design.ENCODER_PATH_PROJECTION_CORRECTION_SELF_KEY],
    }
    target = {
        "path": str(
            design.SCORER_FIT_RELATIVE_PATH / "latents_v2/horizon" /
            ("5" * 64 + ".f16")),
        "candidate_index": 0,
        "sha256": "6" * 64,
        "byte_count": 6_291_456,
        "shape": [4, 768, 1024],
        "device_id": 101,
        "inode": 202,
        "mode_octal": "0644",
        "link_count": 1,
    }
    pre = {
        "latent_index_digest": "7" * 64,
        "encoding_smoke_receipt_digest": "8" * 64,
        "registered_smoke_shard_inventory_digest": "9" * 64,
        "registered_smoke_non_target_shard_inventory_digest": "f" * 64,
        "registered_smoke_non_target_shard_custody_inventory_digest":
            "e" * 64,
        "registered_smoke_stable_artifact_inventory_digest": "a" * 64,
        "zero_new_resume_verified": True,
    }
    prepared = design.build_full_bank_v2_smoke_regeneration_prepared_receipt(
        lineage=lineage, designated_target=target,
        pretransaction_evidence=pre)
    prepared_binding = (
        design.full_bank_v2_smoke_regeneration_prepared_receipt_artifact_binding(
            prepared, design._pretty_json_bytes(prepared)))
    backup = {**prepared["expected_backup_binding"]}
    regenerated = {**target, "inode": 303}
    post = {
        "latent_index_digest": "b" * 64,
        "encoding_smoke_receipt_digest": "d" * 64,
        "registered_smoke_shard_inventory_digest": "9" * 64,
        "registered_smoke_non_target_shard_custody_inventory_digest":
            "e" * 64,
        "registered_smoke_stable_artifact_inventory_digest": "a" * 64,
        "encoder_invocation_new_context_shards": 0,
        "encoder_invocation_new_horizon_shards": 1,
        "target_restored_exact": True,
        "non_target_shards_unchanged": True,
        "complete_before_pass_smoke": True,
    }
    final_smoke_binding = {
        "path": str(
            design.SCORER_FIT_RELATIVE_PATH /
            "smoke_encoding_receipt_v2.json"),
        "schema": "go2_scorer_fit_corpus_v2_end_to_end_smoke_receipt_v1",
        "self_digest_key": "smoke_receipt_digest",
        "self_digest": "d" * 64,
        "raw_sha256": "e" * 64,
        "byte_count": 8_192,
    }
    complete = design.build_full_bank_v2_smoke_regeneration_complete_receipt(
        prepared_receipt_binding=prepared_binding,
        lineage=lineage,
        designated_target=target,
        retained_backup_binding=backup,
        regenerated_target_binding=regenerated,
        non_target_shard_inventory_digest="f" * 64,
        posttransaction_evidence=post,
        final_smoke_receipt_binding=final_smoke_binding,
    )
    return prepared, complete


def _synthetic_installed_artifacts(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *,
        compact_first: bool = False) -> list[dict[str, object]]:
    source_digest = "9" * 64
    monkeypatch.setattr(
        design, "IMMUTABLE_ACTIVE_PRESELECTION_SOURCE_CORRECTION_DIGEST",
        source_digest)
    specs = [
        ("small_completion_selection", "selection.json", "selection_v1",
         "selection_digest", {
             "ordered_candidate_count": 17,
             "selected_scene_ids": [f"scene-{index}" for index in range(5)],
             "branch_data_consumed": False,
             "scientific_outcomes_accessed": False,
             "downstream_metric_used": False,
             "optimisation_or_solver_used": False,
         }),
        ("preoutcome_state_revalidation", "revalidation.json",
         "revalidation_v1", "revalidation_digest", {
             "fixed_state_count": 115,
             "selected_small_completion_state_count": 5,
             "revalidated_state_count": 120,
             "completion_state_count": 40,
             "full_bank_candidate_indices": list(range(12)),
             "branch_data_created": False,
             "frames_or_latents_accessed": False,
             "scientific_outcomes_accessed": False,
             "scorer_or_predictor_accessed": False,
             "true_branch_execution_requirement_count": 0,
         }),
        ("small_family_state_shard", "small.json", "small_v1",
         "small_digest", {
             "states": [{"state": index} for index in range(15)],
             "branch_data_created": False,
             "scientific_outcomes_accessed": False,
             "solver_or_optimisation_used": False,
         }),
        ("assignment_manifest", "assignment.json", "assignment_v1",
         "assignment_digest", {
             "state_count": 120,
             "assignment_count": 1_440,
             "candidate_indices": list(range(12)),
             "branch_execution_used": False,
         }),
        ("state_manifest", "state.json", "state_v1", "state_digest", {
             "states": [{"state": index} for index in range(120)],
             "attempted_branch_count_registered": 1_440,
             "candidate_indices_per_state": list(range(12)),
             "branch_data_created": False,
             "frames_or_latents_accessed": False,
             "scientific_outcomes_accessed": False,
             "scorer_or_predictor_accessed": False,
         }),
    ]
    bindings: list[dict[str, object]] = []
    for index, (role, name, schema, self_key, fields) in enumerate(specs):
        relative = design.SCORER_FIT_RELATIVE_PATH / name
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        body = {
            "schema": schema,
            "complete": True,
            design.SOURCE_CORRECTION_SELF_KEY: source_digest,
            "candidate_outcomes_consumed": False,
            **fields,
        }
        self_digest = (
            design.canonical_digest(body)
            if compact_first and index == 0
            else design.builder_default_canonical_digest(body))
        payload = {**body, self_key: self_digest}
        raw = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
        path.write_bytes(raw)
        path.chmod(0o444)
        bindings.append({
            "role": role,
            "path": str(relative),
            "schema": schema,
            "self_digest_key": self_key,
            "self_digest": self_digest,
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "byte_count": len(raw),
            "mode": "0444",
        })
    monkeypatch.setattr(
        design, "INSTALLED_FULL_BANK_V2_PREOUTCOME_ARTIFACT_BINDINGS",
        tuple(copy.deepcopy(bindings)))
    return bindings


def test_rotation_inventory_is_closed_and_allocation_only() -> None:
    payload = _classification()
    assert design.validate_rotation_mask_classification(payload) == payload
    assert [row["constraint_id"] for row in payload["conditions"]] == list(
        design.EXPECTED_ROTATION_CONSTRAINT_IDS)
    assert {row["classification"] for row in payload["conditions"]} == {
        "PARTIAL_SUBSET_ALLOCATION_ONLY"}
    assert payload["counts"] == {
        "old_rotation_related_condition_count": 18,
        "partial_subset_allocation_only_count": 18,
        "true_branch_execution_requirement_count": 0,
    }
    assert payload["true_branch_execution_test"][
        "matching_old_rotation_condition_ids"] == []


def test_subset_lmax_is_retired_but_completion_science_is_retained() -> None:
    payload = _classification()
    rows = {row["constraint_id"]: row for row in payload["conditions"]}
    for key in (
        "COMPLETION_ASSIGNED_ROTATION_ELIGIBILITY",
        "ALL_40_COMPLETION_MASKS_PASS",
    ):
        assert rows[key]["v2_disposition"] == (
            "REPLACED_BY_FULL_BANK_L_MAX_STATE_REVALIDATION")
        assert "does not establish branch executability" in rows[key]["rationale"]
    retained = payload["retained_non_rotation_completion_requirements"]
    assert retained["full_bank_l_max_candidate_indices"] == list(range(12))
    assert retained["completion_radius_m"] == 0.75
    assert retained["horizon_ticks"] == 20
    assert retained["branch_execution_used_for_revalidation"] is False


def test_classification_is_self_bound_and_tamper_evident() -> None:
    payload = _classification()
    assert payload[design.MASK_CLASSIFICATION_SELF_KEY] == design.canonical_digest({
        key: value for key, value in payload.items()
        if key != design.MASK_CLASSIFICATION_SELF_KEY
    })
    tampered = copy.deepcopy(payload)
    tampered["conditions"][0]["classification"] = (
        "TRUE_BRANCH_EXECUTION_REQUIREMENT")
    with pytest.raises(design.ScorerFitCorpusV2DesignError):
        design.validate_rotation_mask_classification(tampered)


def test_full_bank_design_freezes_exact_algebra_and_only_one_supersession() -> None:
    payload = _amendment()
    assert design.validate_design_amendment(payload) == payload
    counts = payload["count_contract"]
    assert counts["state_count"] == 120
    assert counts["candidate_indices"] == list(range(12))
    assert counts["assignments_total"] == 1_440
    assert counts["per_candidate"] == {
        "overall": 120,
        "fit": 96,
        "calibration": 24,
        "per_stratum": 40,
        "per_family": 15,
        "fit_per_family": 12,
        "calibration_per_family": 3,
        "per_family_stratum": 5,
    }
    assert counts["unordered_candidate_pair_cooccurrence"] == 120
    assert payload["supersession"]["status"] == design.SIX_OF_TWELVE_SUPERSESSION
    assert payload["supersession"]["selector_superseded"] is False
    assert payload["supersession"]["oracle_superseded"] is False
    assert payload["issuance_boundary"]["milp_or_cp_sat_run"] is False


def test_design_binds_exact_terminal_and_all_prior_failure_lineage() -> None:
    lineage = _amendment()["preoutcome_lineage"]
    assert lineage["terminal_source_repository_commit"] == (
        design.TERMINAL_SOURCE_REPOSITORY_COMMIT)
    assert lineage["active_global_amendment_digest"] == (
        design.ACTIVE_GLOBAL_AMENDMENT_DIGEST)
    assert lineage["global_exact_model_digest"] == design.GLOBAL_EXACT_MODEL_DIGEST
    assert lineage["exact_infeasibility_digest"] == design.EXACT_INFEASIBILITY_DIGEST
    assert lineage["terminal_receipt_digest"] == design.TERMINAL_RECEIPT_DIGEST
    assert lineage["candidate_outcomes_consumed_at_proof"] is False
    assert len(lineage["immutable_v1_v2_failure_bindings"]) == 4
    assert len(lineage["prior_preoutcome_failure_bindings"]) == len(
        design.PRIOR_PREOUTCOME_FAILURE_BINDINGS)
    assert lineage["frozen_predictor_qualification"]["modified_or_rerun"] is False


def test_order_key_is_canonical_deterministic_and_domain_separated() -> None:
    structural_a = {"scene_id": "scene-a", "source_step": 20, "split": "fit"}
    structural_b = {"split": "fit", "source_step": 20, "scene_id": "scene-a"}
    goal = {"landmark_id": "g", "landmark_cell": 7, "material_id": 2}
    key_a = design.completion_order_key(structural_a, goal)
    key_b = design.completion_order_key(structural_b, goal)
    assert key_a == key_b
    assert len(key_a[0]) == 64
    assert key_a != design.completion_order_key(
        structural_a, {**goal, "landmark_cell": 8})
    with pytest.raises(design.ScorerFitCorpusV2DesignError):
        design.completion_order_key(
            structural_a, goal, active_selector_digest="b" * 64)


def test_design_preserves_prospective_final_eval_disjointness_not_reservation() -> None:
    science = _amendment()["preserved_nonallocation_science"]
    assert science["final_200_state_corpus_authorized_in_this_pass"] is False
    assert science["preexisting_reserved_final_evaluation_scene_set"] is False
    assert science["final_evaluation_manifest_absent_at_issue"] is True
    assert "excludes all 120 scenes" in science["future_final_evaluation_rule"]


def test_phase_aware_absence_audit(tmp_path: Path) -> None:
    scorer_fit = tmp_path / design.SCORER_FIT_RELATIVE_PATH
    scorer_fit.mkdir(parents=True)
    assert design.audit_v2_runtime_outputs_absent(
        root=tmp_path, phase="design")
    preoutcome = tmp_path / design.V2_PREOUTCOME_ARTIFACT_PATHS[0]
    preoutcome.write_text("source-only")
    with pytest.raises(design.ScorerFitCorpusV2DesignError):
        design.audit_v2_runtime_outputs_absent(root=tmp_path, phase="design")
    assert design.audit_v2_runtime_outputs_absent(
        root=tmp_path, phase="successor_contract")
    runtime = tmp_path / design.V2_RUNTIME_OUTPUT_PATHS[0]
    runtime.write_text("outcome")
    with pytest.raises(design.ScorerFitCorpusV2DesignError):
        design.audit_v2_runtime_outputs_absent(
            root=tmp_path, phase="successor_contract")


def test_issue_is_exclusive_read_only_and_validates_predecessors_twice(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scorer_fit = tmp_path / design.SCORER_FIT_RELATIVE_PATH
    scorer_fit.mkdir(parents=True)
    calls: list[str] = []

    monkeypatch.setattr(
        design, "clean_source_authority", lambda *, root: (COMMIT, _sources()))

    def predecessors(*, root: Path) -> dict[str, object]:
        calls.append(str(root))
        return copy.deepcopy(design.PREDECESSOR_VALIDATION_PROJECTION)

    monkeypatch.setattr(
        design, "validate_historical_predecessor_artifacts", predecessors)
    classification = design.issue_rotation_mask_classification(root=tmp_path)
    assert len(calls) == 2
    class_path = tmp_path / design.MASK_CLASSIFICATION_RELATIVE_PATH
    assert stat.S_IMODE(class_path.stat().st_mode) == 0o444
    assert design.issue_rotation_mask_classification(root=tmp_path) == classification
    assert len(calls) == 2  # reopening an issued classification is source-only.

    amendment = design.issue_design_amendment(root=tmp_path)
    assert len(calls) == 4
    design_path = tmp_path / design.DESIGN_RELATIVE_PATH
    assert stat.S_IMODE(design_path.stat().st_mode) == 0o444
    assert design.load_design_amendment(root=tmp_path) == amendment


def test_artifact_binding_rejects_noncanonical_bytes() -> None:
    payload = _classification()
    compact = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(design.ScorerFitCorpusV2DesignError):
        design.rotation_mask_classification_artifact_binding(payload, compact)


def test_builders_are_pure_and_do_not_require_repository_or_generated_files(
        tmp_path: Path) -> None:
    # Both builders operate entirely on supplied source identities and literal
    # frozen lineage.  The empty path proves no repository fixture is needed.
    assert not list(tmp_path.iterdir())
    assert _classification()["outcome_access"][
        "historical_receipts_used_for_classification"] is False
    assert _amendment()["selection_field_policy"][
        "historical_receipts_used_for_selection"] is False
    assert not list(tmp_path.iterdir())


def test_preselection_source_correction_v1_preserves_first_failure() -> None:
    correction = _source_correction_v1()
    assert design.validate_preselection_source_correction_v1(
        correction, validate_live_authorities=False) == correction
    issued = _issued_design_authority()
    assert correction["preserved_scientific_design_digest"] == issued[
        "design_amendment_payload"][design.DESIGN_SELF_KEY]
    assert correction["preserved_rotation_mask_classification_digest"] == issued[
        "rotation_mask_classification_payload"][
            design.MASK_CLASSIFICATION_SELF_KEY]
    assert correction["source_correction"][
        "observed_changed_source_paths"] == sorted(
            design.SOURCE_CORRECTION_V1_ALLOWED_CHANGED_SOURCE_PATHS)
    failure = correction["preselection_alias_failure_boundary"]
    assert failure == design.PRESELECTION_ALIAS_FAILURE_BOUNDARY_V1
    assert failure["predecessor_fixed_state_count_validated"] == 115
    assert failure["eligible_small_completion_scene_count_validated"] == 17
    assert failure["exclusion_authority_returned"] is False
    assert failure["small_completion_selection_started"] is False
    assert failure["preoutcome_manifest_or_selection_artifact_issued"] is False
    assert failure["candidate_outcome_or_branch_label_read"] is False
    assert failure["solver_or_optimisation_invoked"] is False


def test_preselection_source_correction_v1_is_tamper_evident() -> None:
    correction = _source_correction_v1()
    raw = (json.dumps(correction, sort_keys=True, indent=2) + "\n").encode()
    binding = design.preselection_source_correction_v1_artifact_binding(
        correction, raw)
    assert binding["self_digest"] == correction[
        design.SOURCE_CORRECTION_SELF_KEY]
    assert set(binding) == {
        "path", "schema", "self_digest_key", "self_digest", "raw_sha256",
        "byte_count", "source_repository_commit",
    }
    tampered = copy.deepcopy(correction)
    tampered["preselection_alias_failure_boundary"][
        "exclusion_authority_returned"] = True
    with pytest.raises(design.ScorerFitCorpusV2DesignError):
        design.validate_preselection_source_correction_v1(
            tampered, validate_live_authorities=False)
    extra = copy.deepcopy(correction)
    extra["unregistered"] = True
    with pytest.raises(design.ScorerFitCorpusV2DesignError):
        design.validate_preselection_source_correction_v1(
            extra, validate_live_authorities=False)


def test_preselection_source_correction_v1_rejects_wrong_source_delta() -> None:
    sources = _corrected_sources_v1()
    unchanged_path = next(
        row for row in sources
        if row["path"]
        not in design.SOURCE_CORRECTION_V1_ALLOWED_CHANGED_SOURCE_PATHS)
    unchanged_path["byte_count"] = int(unchanged_path["byte_count"]) + 1
    unchanged_path["sha256"] = "f" * 64
    with pytest.raises(design.ScorerFitCorpusV2DesignError):
        design.build_preselection_source_correction_v1(
            source_repository_commit=
                design.IMMUTABLE_SOURCE_CORRECTION_V1_SOURCE_REPOSITORY_COMMIT,
            source_bindings=sources,
            immutable_issued_design_authority=_issued_design_authority(),
            runtime_outputs_absent_at_issue=design._expected_absence_rows(
                phase="design"),
        )


def test_source_correction_v1_cannot_be_reissued_by_chained_source(
        tmp_path: Path) -> None:
    scorer_fit = tmp_path / design.SCORER_FIT_RELATIVE_PATH
    scorer_fit.mkdir(parents=True)
    path = tmp_path / design.SOURCE_CORRECTION_V1_RELATIVE_PATH
    with pytest.raises(
            design.ScorerFitCorpusV2DesignError,
            match="cannot be reissued"):
        design.issue_preselection_source_correction_v1(root=tmp_path)
    assert not path.exists()


def test_chained_source_correction_v2_preserves_v1_and_second_failure(
        monkeypatch: pytest.MonkeyPatch) -> None:
    correction = _source_correction_v2(monkeypatch)
    assert design.validate_preselection_source_correction_v2(
        correction, validate_live_authorities=False) == correction
    immutable_v1 = correction[
        "immutable_preselection_source_correction_v1"]
    assert immutable_v1["payload"][design.SOURCE_CORRECTION_SELF_KEY] == (
        correction["immutable_preselection_source_correction_v1_digest"])
    issued = immutable_v1["payload"]["immutable_issued_design_authority"]
    assert correction["preserved_scientific_design_digest"] == issued[
        "design_amendment_payload"][design.DESIGN_SELF_KEY]
    assert correction["source_correction"][
        "observed_changed_source_paths"] == sorted(
            design.SOURCE_CORRECTION_V2_ALLOWED_CHANGED_SOURCE_PATHS)
    failure = correction["preselection_alias_failure_boundary"]
    assert failure == design.PRESELECTION_ALIAS_FAILURE_BOUNDARY_V2
    assert failure[
        "development_stage_a_identity_manifest_json_read_and_validated"] is True
    assert failure[
        "registered_development_manifest_alias_resolved_and_validated"] is True
    assert failure["failure_cause"] == (
        "OUT_ROOT_IS_A_REGISTERED_GENERATED_ROOT_SYMLINK")
    assert failure["prospective_final_eval_absence_verdict_returned"] is False
    assert failure["exclusion_authority_returned"] is False
    assert failure["candidate_revalidation_started"] is False
    assert failure["preoutcome_manifest_or_selection_artifact_issued"] is False
    assert failure["candidate_outcome_or_branch_label_read"] is False


def test_chained_source_correction_v2_is_closed_and_tamper_evident(
        monkeypatch: pytest.MonkeyPatch) -> None:
    correction = _source_correction_v2(monkeypatch)
    raw = (json.dumps(correction, sort_keys=True, indent=2) + "\n").encode()
    binding = design.preselection_source_correction_v2_artifact_binding(
        correction, raw)
    assert set(binding) == {
        "path", "schema", "self_digest_key", "self_digest", "raw_sha256",
        "byte_count", "source_repository_commit",
    }
    assert binding["path"] == str(design.SOURCE_CORRECTION_V2_RELATIVE_PATH)
    tampered = copy.deepcopy(correction)
    tampered["preselection_alias_failure_boundary"][
        "exclusion_authority_returned"] = True
    with pytest.raises(design.ScorerFitCorpusV2DesignError):
        design.validate_preselection_source_correction_v2(
            tampered, validate_live_authorities=False)


def test_chained_source_correction_v2_rejects_extra_source_change(
        monkeypatch: pytest.MonkeyPatch) -> None:
    immutable_v1 = _immutable_source_correction_v1(monkeypatch)
    sources = _corrected_sources_v2()
    unchanged = next(
        row for row in sources
        if row["path"]
        not in design.SOURCE_CORRECTION_V2_ALLOWED_CHANGED_SOURCE_PATHS)
    unchanged["byte_count"] = int(unchanged["byte_count"]) + 1
    unchanged["sha256"] = "f" * 64
    with pytest.raises(design.ScorerFitCorpusV2DesignError):
        design.build_preselection_source_correction_v2(
            source_repository_commit=
                design.IMMUTABLE_SOURCE_CORRECTION_V2_SOURCE_REPOSITORY_COMMIT,
            source_bindings=sources,
            immutable_preselection_source_correction_v1=immutable_v1,
            runtime_outputs_absent_at_issue=design._expected_absence_rows(
                phase="design"),
        )


def test_source_correction_v2_cannot_be_reissued_by_final_source(
        tmp_path: Path) -> None:
    scorer_fit = tmp_path / design.SCORER_FIT_RELATIVE_PATH
    scorer_fit.mkdir(parents=True)
    path = tmp_path / design.SOURCE_CORRECTION_V2_RELATIVE_PATH
    with pytest.raises(
            design.ScorerFitCorpusV2DesignError,
            match="cannot be reissued"):
        design.issue_preselection_source_correction_v2(root=tmp_path)
    assert not path.exists()


def test_final_structural_validation_correction_preserves_full_chain(
        monkeypatch: pytest.MonkeyPatch) -> None:
    correction = _source_correction_final(monkeypatch)
    assert design.validate_preselection_source_correction(
        correction, validate_live_authorities=False) == correction
    immutable_v2 = correction[
        "immutable_preselection_source_correction_v2"]
    assert immutable_v2["payload"][design.SOURCE_CORRECTION_SELF_KEY] == (
        correction["immutable_preselection_source_correction_v2_digest"])
    immutable_v1 = immutable_v2["payload"][
        "immutable_preselection_source_correction_v1"]
    assert immutable_v1["payload"][design.SOURCE_CORRECTION_SELF_KEY] == (
        correction[
            "transitive_immutable_preselection_source_correction_v1_digest"])
    assert correction["structural_validation_correction"][
        "observed_changed_source_paths"] == sorted(
            design.SOURCE_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS)
    material = correction["structural_validation_correction"]
    assert material["body_clearance_m_domain"] == "FINITE_SIGNED_REAL"
    assert material["clearance_m_domain"] == "FINITE_REAL_GTE_0"
    assert material["safety_enriched_body_clearance_upper_bound_m"] == 0.10
    failure = correction[
        "preselection_structural_validation_failure_boundary"]
    assert failure == design.PRESELECTION_STRUCTURAL_VALIDATION_FAILURE_BOUNDARY
    assert failure["exclusion_authority_returned"] is True
    assert failure[
        "eligible_small_completion_candidate_revalidation_count"] == 17
    assert failure[
        "deterministic_five_scene_selection_computed_in_memory"] is True
    assert failure["first_rejected_value_relation"] == (
        "body_clearance_m < 0.0")
    assert failure["preoutcome_manifest_or_selection_artifact_issued"] is False
    assert failure["candidate_outcome_or_branch_label_read"] is False
    dry_run = correction["post_fix_production_bundle_dry_run"]
    assert dry_run["state_count"] == 120
    assert dry_run["assignment_count"] == 1_440
    assert dry_run["verify_scene_files"] is True
    assert len(dry_run["payload_digests"]) == 5
    assert dry_run[
        "live_clean_source_equality_check_substituted_for_diagnostic"] is True
    assert dry_run["scientific_constraint_validator_bypassed"] is False
    assert dry_run["payload_or_digest_validator_bypassed"] is False
    assert dry_run["generated_artifact_written"] is False


def test_final_structural_validation_correction_is_tamper_evident(
        monkeypatch: pytest.MonkeyPatch) -> None:
    correction = _source_correction_final(monkeypatch)
    raw = (json.dumps(correction, sort_keys=True, indent=2) + "\n").encode()
    binding = design.preselection_source_correction_artifact_binding(
        correction, raw)
    assert set(binding) == {
        "path", "schema", "self_digest_key", "self_digest", "raw_sha256",
        "byte_count", "source_repository_commit",
    }
    assert binding["path"] == str(design.SOURCE_CORRECTION_RELATIVE_PATH)
    tampered = copy.deepcopy(correction)
    tampered["structural_validation_correction"][
        "body_clearance_m_domain"] = "FINITE_NONNEGATIVE_REAL"
    with pytest.raises(design.ScorerFitCorpusV2DesignError):
        design.validate_preselection_source_correction(
            tampered, validate_live_authorities=False)


def test_final_structural_validation_correction_rejects_extra_source_change(
        monkeypatch: pytest.MonkeyPatch) -> None:
    immutable_v2 = _immutable_source_correction_v2(monkeypatch)
    sources = _corrected_sources_final()
    unchanged = next(
        row for row in sources
        if row["path"] not in design.SOURCE_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS)
    unchanged["byte_count"] = int(unchanged["byte_count"]) + 1
    unchanged["sha256"] = "f" * 64
    with pytest.raises(design.ScorerFitCorpusV2DesignError):
        design.build_preselection_source_correction(
            source_repository_commit="d" * 40,
            source_bindings=sources,
            immutable_preselection_source_correction_v2=immutable_v2,
            runtime_outputs_absent_at_issue=design._expected_absence_rows(
                phase="design"),
        )


def test_active_preselection_correction_cannot_be_reissued(
        tmp_path: Path) -> None:
    scorer_fit = tmp_path / design.SCORER_FIT_RELATIVE_PATH
    scorer_fit.mkdir(parents=True)
    path = tmp_path / design.SOURCE_CORRECTION_RELATIVE_PATH
    with pytest.raises(
            design.ScorerFitCorpusV2DesignError,
            match="cannot be reissued"):
        design.issue_preselection_source_correction(root=tmp_path)
    assert not path.exists()


def test_manifest_replay_correction_preserves_5206_and_exact_failure(
        monkeypatch: pytest.MonkeyPatch) -> None:
    correction = _manifest_replay_correction(monkeypatch)
    assert design.validate_manifest_replay_correction(
        correction, validate_live_authorities=False) == correction
    immutable = correction[
        "immutable_active_preselection_source_correction"]
    assert immutable["payload"][design.SOURCE_CORRECTION_SELF_KEY] == (
        correction[
            "immutable_active_preselection_source_correction_digest"])
    assert correction["preserved_scientific_manifest_lineage_digest"] == (
        immutable["payload"][design.SOURCE_CORRECTION_SELF_KEY])
    assert correction["installed_preoutcome_artifact_bindings"] == list(
        design.INSTALLED_FULL_BANK_V2_PREOUTCOME_ARTIFACT_BINDINGS)
    assert [row["role"] for row in correction[
            "installed_preoutcome_artifact_bindings"]] == [
        "small_completion_selection", "preoutcome_state_revalidation",
        "small_family_state_shard", "assignment_manifest", "state_manifest",
    ]
    assert {row["mode"] for row in correction[
        "installed_preoutcome_artifact_bindings"]} == {"0444"}
    failure = correction["manifest_replay_failure_boundary"]
    assert failure == design.MANIFEST_REPLAY_FAILURE_BOUNDARY
    assert failure["all_five_preoutcome_artifacts_installed"] is True
    assert failure["state_manifest_installed_last_as_terminal_marker"] is True
    assert failure["first_replay_role"] == "small_completion_selection"
    assert failure["post_install_replay_completed"] is False
    assert failure["successor_scorer_contract_issued"] is False
    assert failure["candidate_outcome_or_branch_label_read"] is False
    material = correction["manifest_replay_correction"]
    assert material["full_bank_v2_self_digest_canonicalization"] == (
        "JSON_DUMPS_SORT_KEYS_DEFAULT_SEPARATORS")
    assert material["installed_manifest_payload_or_digest_changed"] is False
    assert material["scientific_manifest_lineage_digest_preserved"] == (
        correction["preserved_scientific_manifest_lineage_digest"])


def test_builder_default_digest_is_not_parallel_compact() -> None:
    body = {"schema": "synthetic", "nested": {"value": 3}}
    assert design.builder_default_canonical_digest(body) == (
        "4588d100adc9cc3ba1a554c8800400fc0d474660261a9d43fff8ef8726f3c8de")
    assert design.builder_default_canonical_digest(body) != (
        design.canonical_digest(body))


def test_installed_manifest_validator_requires_builder_default_digest(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bindings = _synthetic_installed_artifacts(tmp_path, monkeypatch)
    assert design.validate_installed_full_bank_v2_preoutcome_artifacts(
        root=tmp_path) == bindings

    bad_root = tmp_path / "bad"
    bad_bindings = _synthetic_installed_artifacts(
        bad_root, monkeypatch, compact_first=True)
    assert bad_bindings[0]["self_digest"] != (
        design.builder_default_canonical_digest({
            "schema": "selection_v1",
            "complete": True,
            design.SOURCE_CORRECTION_SELF_KEY: "9" * 64,
            "candidate_outcomes_consumed": False,
            "ordered_candidate_count": 17,
            "selected_scene_ids": [f"scene-{index}" for index in range(5)],
            "branch_data_consumed": False,
            "scientific_outcomes_accessed": False,
            "downstream_metric_used": False,
            "optimisation_or_solver_used": False,
        }))
    with pytest.raises(
            design.ScorerFitCorpusV2DesignError,
            match="small_completion_selection artifact changed"):
        design.validate_installed_full_bank_v2_preoutcome_artifacts(
            root=bad_root)


def test_manifest_replay_correction_is_tamper_evident(
        monkeypatch: pytest.MonkeyPatch) -> None:
    correction = _manifest_replay_correction(monkeypatch)
    raw = (json.dumps(correction, sort_keys=True, indent=2) + "\n").encode()
    binding = design.manifest_replay_correction_artifact_binding(
        correction, raw)
    assert set(binding) == {
        "path", "schema", "self_digest_key", "self_digest", "raw_sha256",
        "byte_count", "source_repository_commit",
    }
    assert binding["path"] == str(
        design.MANIFEST_REPLAY_CORRECTION_RELATIVE_PATH)
    tampered = copy.deepcopy(correction)
    tampered["installed_preoutcome_artifact_bindings"][0][
        "self_digest"] = "f" * 64
    with pytest.raises(design.ScorerFitCorpusV2DesignError):
        design.validate_manifest_replay_correction(
            tampered, validate_live_authorities=False)


def test_manifest_replay_correction_rejects_extra_source_change(
        monkeypatch: pytest.MonkeyPatch) -> None:
    immutable = _immutable_active_preselection_source_correction(monkeypatch)
    sources = _corrected_sources_replay()
    unchanged = next(
        row for row in sources
        if row["path"]
        not in design.MANIFEST_REPLAY_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS)
    unchanged["byte_count"] = int(unchanged["byte_count"]) + 1
    unchanged["sha256"] = "f" * 64
    with pytest.raises(design.ScorerFitCorpusV2DesignError):
        design.build_manifest_replay_correction(
            source_repository_commit="e" * 40,
            source_bindings=sources,
            immutable_active_preselection_source_correction=immutable,
            installed_preoutcome_artifact_bindings=
                design.INSTALLED_FULL_BANK_V2_PREOUTCOME_ARTIFACT_BINDINGS,
            successor_and_runtime_outputs_absent_at_issue=
                design._expected_absence_rows(phase="successor_contract"),
        )


def test_manifest_replay_issue_and_active_loader_keep_5206_lineage(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scorer_fit = tmp_path / design.SCORER_FIT_RELATIVE_PATH
    scorer_fit.mkdir(parents=True)
    immutable = _immutable_active_preselection_source_correction(monkeypatch)
    sources = _corrected_sources_replay()
    installed = copy.deepcopy(list(
        design.INSTALLED_FULL_BANK_V2_PREOUTCOME_ARTIFACT_BINDINGS))
    absence = design._expected_absence_rows(phase="successor_contract")
    absence_calls: list[int] = []
    installed_calls: list[int] = []
    monkeypatch.setattr(
        design, "clean_source_authority",
        lambda *, root: ("e" * 40, copy.deepcopy(sources)))
    monkeypatch.setattr(
        design, "_load_immutable_active_preselection_source_correction",
        lambda *, root: copy.deepcopy(immutable))

    def installed_validator(*, root: Path) -> list[dict[str, object]]:
        assert root == tmp_path
        installed_calls.append(1)
        return copy.deepcopy(installed)

    monkeypatch.setattr(
        design, "validate_installed_full_bank_v2_preoutcome_artifacts",
        installed_validator)

    def audit(*, root: Path, phase: str) -> list[dict[str, object]]:
        assert root == tmp_path
        assert phase == "successor_contract"
        absence_calls.append(1)
        return copy.deepcopy(absence)

    monkeypatch.setattr(design, "audit_v2_runtime_outputs_absent", audit)
    correction = design.issue_manifest_replay_correction(root=tmp_path)
    assert len(absence_calls) == 2
    assert len(installed_calls) >= 2
    path = tmp_path / design.MANIFEST_REPLAY_CORRECTION_RELATIVE_PATH
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    monkeypatch.setattr(
        design, "ENCODER_IMPORT_CORRECTION_HISTORICAL_SOURCE_REPOSITORY_COMMIT",
        "e" * 40)
    monkeypatch.setattr(
        design, "IMMUTABLE_MANIFEST_REPLAY_CORRECTION_DIGEST",
        correction[design.MANIFEST_REPLAY_CORRECTION_SELF_KEY])
    encoder_sources = copy.deepcopy(correction["source_bindings"])
    base_changed = set(
        design.ENCODER_IMPORT_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS
    ).intersection(design.EXPECTED_SOURCE_PATHS)
    for index, row in enumerate(encoder_sources):
        if row["path"] in base_changed:
            row["byte_count"] = int(row["byte_count"]) + 50_000
            row["sha256"] = f"{index + 50_000:064x}"
    old_dev = copy.deepcopy(
        design.ENCODER_IMPORT_CORRECTION_DEV_ENCODER_HISTORICAL_BINDING)
    new_dev = {**old_dev, "byte_count": old_dev["byte_count"] + 1,
               "sha256": "8" * 64}
    dev_transition = {
        "path": old_dev["path"], "role": old_dev["role"],
        "historical": old_dev, "current": new_dev,
    }
    test_transitions = []
    for index, (test_path, role) in enumerate(
            design.ENCODER_IMPORT_CORRECTION_FOCUSED_TEST_SPECS):
        historical = {
            "path": test_path, "role": role, "exists": index != 3,
            "byte_count": 0 if index == 3 else index + 1,
            "sha256": None if index == 3 else f"{index + 1:064x}",
        }
        current = {
            "path": test_path, "role": role, "exists": True,
            "byte_count": index + 101, "sha256": f"{index + 101:064x}",
        }
        test_transitions.append({
            "path": test_path, "role": role,
            "historical": historical, "current": current,
        })
    replay_raw = path.read_bytes()
    encoder_correction = design.build_encoder_import_correction(
        source_repository_commit="f" * 40,
        source_bindings=encoder_sources,
        immutable_manifest_replay_correction={
            "payload": correction,
            "binding": design.manifest_replay_correction_artifact_binding(
                correction, replay_raw),
        },
        immutable_successor_scorer_contract_binding=
            design.IMMUTABLE_SUCCESSOR_SCORER_CONTRACT_BINDING,
        dev_encoder_source_transition=dev_transition,
        focused_test_source_transitions=test_transitions,
        branch_smoke_binding=
            design.IMMUTABLE_ENCODER_IMPORT_FAILURE_BRANCH_SMOKE_BINDING,
        branch_corpus_binding=
            design.IMMUTABLE_ENCODER_IMPORT_FAILURE_CORPUS_RECEIPT_BINDING,
        prelatent_outputs_absent_at_issue=
            design._expected_encoder_import_correction_absence_rows(),
    )
    dtype_correction = _encoder_compute_dtype_correction(
        monkeypatch, encoder_import=encoder_correction)
    path_correction = _encoder_path_projection_correction(
        monkeypatch, dtype_correction=dtype_correction)
    redrive_correction = _branch_redrive_projection_correction(
        monkeypatch, path_correction=path_correction)
    resume_correction = _optional_smoke_partial_corpus_resume_correction(
        monkeypatch, redrive_correction=redrive_correction)
    monkeypatch.setattr(
        design,
        "load_optional_smoke_partial_corpus_resume_correction_for_consumption",
        lambda **_kwargs: copy.deepcopy(resume_correction))
    active = design.load_active_design_authority(root=tmp_path)
    assert active["source_correction"] == immutable["payload"]
    assert active["source_correction_binding"] == immutable["binding"]
    assert active["source_correction_digest"] == immutable[
        "payload"][design.SOURCE_CORRECTION_SELF_KEY]
    assert active["manifest_replay_correction"] == correction
    assert active["manifest_replay_correction_digest"] == correction[
        design.MANIFEST_REPLAY_CORRECTION_SELF_KEY]
    assert active["manifest_replay_correction_binding"] == (
        design.manifest_replay_correction_artifact_binding(
            correction, path.read_bytes()))
    assert active["manifest_replay_source_repository_commit"] == "e" * 40
    assert active["encoder_import_source_repository_commit"] == "f" * 40
    assert active["encoder_compute_dtype_source_repository_commit"] == "1" * 40
    assert active["encoder_path_projection_source_repository_commit"] == \
        "2" * 40
    assert active["active_source_repository_commit"] == "4" * 40
    assert active["encoder_import_correction"] == encoder_correction
    assert active["encoder_compute_dtype_correction"] == dtype_correction
    assert active["encoder_path_projection_correction"] == path_correction
    assert active["branch_redrive_projection_correction"] == \
        redrive_correction
    assert active["branch_redrive_projection_correction_digest"] == \
        redrive_correction[
            design.BRANCH_REDRIVE_PROJECTION_CORRECTION_SELF_KEY]
    assert active["optional_smoke_partial_corpus_resume_correction"] == \
        resume_correction
    assert active[
        "optional_smoke_partial_corpus_resume_correction_digest"] == \
        resume_correction[
            design.OPTIONAL_SMOKE_PARTIAL_CORPUS_RESUME_CORRECTION_SELF_KEY]
    immutable_v2 = immutable["payload"][
        "immutable_preselection_source_correction_v2"]
    immutable_v1 = immutable_v2["payload"][
        "immutable_preselection_source_correction_v1"]
    issued = immutable_v1["payload"]["immutable_issued_design_authority"]
    assert active["design_amendment"] == issued["design_amendment_payload"]
    assert design.issue_manifest_replay_correction(
        root=tmp_path) == correction
    assert len(absence_calls) == 2


def test_encoder_import_correction_is_closed_science_preserving_and_truthful(
        monkeypatch: pytest.MonkeyPatch) -> None:
    correction = _encoder_import_correction(monkeypatch)
    assert design.validate_encoder_import_correction(
        correction, validate_live_authorities=False) == correction
    assert correction["production_source_transition"][
        "observed_changed_source_paths"] == sorted(
            design.ENCODER_IMPORT_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS)
    assert [row["path"] for row in correction[
            "focused_test_source_transitions"]] == list(
                design.ENCODER_IMPORT_CORRECTION_FOCUSED_TEST_PATHS)
    assert correction["immutable_successor_scorer_contract_binding"] == (
        design.IMMUTABLE_SUCCESSOR_SCORER_CONTRACT_BINDING)
    assert correction["preserved_scientific_contract"] == (
        design.ENCODER_IMPORT_CORRECTION_PRESERVED_SCIENCE)
    failure = correction["encoder_import_failure_boundary"]
    assert failure["branch_record_count"] == 12
    assert failure["rendered_horizon_frame_count"] == 48
    assert failure["branch_outcomes_exist"] is True
    assert failure[
        "branch_outcome_or_label_value_consumed_for_correction"] is False
    assert failure[
        "checkpoint_file_read_only_for_sha256_identity_verification"] is True
    assert failure["checkpoint_torch_load_or_tensor_deserialization_started"] is False
    assert failure["encoder_or_predictor_model_constructed"] is False
    assert failure["latent_shard_written"] is False
    assert correction["encoder_import_correction"][
        "runtime_venv_package_installed_or_mutated"] is False
    assert correction["issuance_boundary"][
        "later_consumption_requires_failure_time_receipts_live"] is False


def test_encoder_import_correction_rejects_tamper_and_extra_production_change(
        monkeypatch: pytest.MonkeyPatch) -> None:
    correction = _encoder_import_correction(monkeypatch)
    raw = (json.dumps(correction, sort_keys=True, indent=2) + "\n").encode()
    binding = design.encoder_import_correction_artifact_binding(correction, raw)
    assert binding["self_digest"] == correction[
        design.ENCODER_IMPORT_CORRECTION_SELF_KEY]
    tampered = copy.deepcopy(correction)
    tampered["encoder_import_failure_boundary"][
        "checkpoint_torch_load_or_tensor_deserialization_started"] = True
    with pytest.raises(design.ScorerFitCorpusV2DesignError):
        design.validate_encoder_import_correction(
            tampered, validate_live_authorities=False)

    replay = correction["immutable_manifest_replay_correction"]["payload"]
    sources = copy.deepcopy(correction["source_bindings"])
    unchanged = next(
        row for row in sources
        if row["path"] not in set(
            design.ENCODER_IMPORT_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS))
    unchanged["byte_count"] = int(unchanged["byte_count"]) + 1
    unchanged["sha256"] = "f" * 64
    with pytest.raises(design.ScorerFitCorpusV2DesignError):
        design.build_encoder_import_correction(
            source_repository_commit=correction["source_repository_commit"],
            source_bindings=sources,
            immutable_manifest_replay_correction=
                correction["immutable_manifest_replay_correction"],
            immutable_successor_scorer_contract_binding=
                correction["immutable_successor_scorer_contract_binding"],
            dev_encoder_source_transition=correction[
                "dev_encoder_source_transition"],
            focused_test_source_transitions=correction[
                "focused_test_source_transitions"],
            branch_smoke_binding=correction["immutable_branch_smoke_binding"],
            branch_corpus_binding=correction[
                "immutable_partial_corpus_receipt_binding"],
            prelatent_outputs_absent_at_issue=correction[
                "prelatent_outputs_absent_at_issue"],
        )
    assert replay[design.MANIFEST_REPLAY_CORRECTION_SELF_KEY] == correction[
        "immutable_manifest_replay_correction_digest"]


def test_encoder_import_correction_issue_reopen_and_receipt_refresh_lifecycle(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    correction = _encoder_import_correction(monkeypatch)
    commit = correction["source_repository_commit"]
    sources = correction["source_bindings"]
    immutable_replay = correction["immutable_manifest_replay_correction"]
    successor = correction["immutable_successor_scorer_contract_binding"]
    dev_transition = correction["dev_encoder_source_transition"]
    test_transitions = correction["focused_test_source_transitions"]
    smoke = correction["immutable_branch_smoke_binding"]
    corpus = correction["immutable_partial_corpus_receipt_binding"]
    absence = correction["prelatent_outputs_absent_at_issue"]
    expected = tmp_path / design.ENCODER_IMPORT_CORRECTION_RELATIVE_PATH
    expected.parent.mkdir(parents=True)

    calls = {
        "source": 0, "replay": 0, "successor": 0, "dev": 0,
        "tests": 0, "receipts": 0, "absence": 0, "install": 0,
    }

    def clean_source(*, root: Path) -> tuple[str, list[dict[str, object]]]:
        assert root == tmp_path
        calls["source"] += 1
        return str(commit), copy.deepcopy(sources)

    def load_replay(*, root: Path) -> dict[str, object]:
        assert root == tmp_path
        calls["replay"] += 1
        return copy.deepcopy(immutable_replay)

    def load_successor(*, root: Path) -> dict[str, object]:
        assert root == tmp_path
        calls["successor"] += 1
        return copy.deepcopy(successor)

    def load_dev(*, root: Path) -> dict[str, object]:
        assert root == tmp_path
        calls["dev"] += 1
        return copy.deepcopy(dev_transition)

    def load_tests(*, root: Path) -> list[dict[str, object]]:
        assert root == tmp_path
        calls["tests"] += 1
        return copy.deepcopy(test_transitions)

    def load_receipts(
            *, root: Path,
            ) -> tuple[dict[str, object], dict[str, object]]:
        assert root == tmp_path
        calls["receipts"] += 1
        return copy.deepcopy(smoke), copy.deepcopy(corpus)

    def load_absence(*, root: Path) -> list[dict[str, object]]:
        assert root == tmp_path
        calls["absence"] += 1
        return copy.deepcopy(absence)

    monkeypatch.setattr(design, "clean_source_authority", clean_source)
    monkeypatch.setattr(
        design, "_load_immutable_manifest_replay_correction", load_replay)
    monkeypatch.setattr(
        design, "_load_immutable_successor_scorer_contract_binding",
        load_successor)
    monkeypatch.setattr(design, "_dev_encoder_source_transition", load_dev)
    monkeypatch.setattr(
        design, "_focused_test_source_transitions", load_tests)
    monkeypatch.setattr(
        design, "_validate_live_encoder_import_failure_receipts",
        load_receipts)
    monkeypatch.setattr(
        design, "audit_encoder_import_correction_prelatent_absence",
        load_absence)

    exclusive_json = design._exclusive_json

    def checked_exclusive_json(
            path: Path, payload: dict[str, object], *, label: str) -> None:
        assert path == expected
        assert label == "post-smoke encoder-import correction"
        assert not path.exists() and not path.is_symlink()
        assert {key: calls[key] for key in (
            "source", "replay", "successor", "dev", "tests", "receipts",
            "absence",
        )} == {
            "source": 2, "replay": 2, "successor": 2, "dev": 2,
            "tests": 2, "receipts": 2, "absence": 2,
        }
        calls["install"] += 1
        exclusive_json(path, payload, label=label)

    monkeypatch.setattr(design, "_exclusive_json", checked_exclusive_json)
    issued = design.issue_encoder_import_correction(
        root=tmp_path, source_repository_commit=str(commit))
    assert issued == correction
    assert calls == {
        "source": 3, "replay": 3, "successor": 3, "dev": 3,
        "tests": 3, "receipts": 3, "absence": 3, "install": 1,
    }
    assert stat.S_IMODE(expected.stat().st_mode) == 0o444
    raw = expected.read_bytes()
    assert design.encoder_import_correction_artifact_binding(
        correction, raw)["self_digest"] == correction[
            design.ENCODER_IMPORT_CORRECTION_SELF_KEY]

    refreshed_receipt_checks: list[int] = []
    refreshed_absence_checks: list[int] = []

    def refreshed_receipts(
            *, root: Path,
            ) -> tuple[dict[str, object], dict[str, object]]:
        assert root == tmp_path
        refreshed_receipt_checks.append(1)
        return {"refreshed": True}, {"refreshed": True}

    def refreshed_absence(*, root: Path) -> list[dict[str, object]]:
        assert root == tmp_path
        refreshed_absence_checks.append(1)
        return [{"refreshed": True}]

    monkeypatch.setattr(
        design, "_validate_live_encoder_import_failure_receipts",
        refreshed_receipts)
    monkeypatch.setattr(
        design, "audit_encoder_import_correction_prelatent_absence",
        refreshed_absence)
    reopened = design.load_encoder_import_correction_for_consumption(
        root=tmp_path, require_failure_boundary_live=False)
    assert reopened == correction
    assert design.issue_encoder_import_correction(
        root=tmp_path, source_repository_commit=str(commit)) == correction
    assert refreshed_receipt_checks == []
    assert refreshed_absence_checks == []
    assert calls["install"] == 1
    assert expected.read_bytes() == raw
    assert stat.S_IMODE(expected.stat().st_mode) == 0o444


def test_branch_redrive_projection_correction_is_closed_and_science_preserving(
        monkeypatch: pytest.MonkeyPatch) -> None:
    correction = _branch_redrive_projection_correction(monkeypatch)
    assert design.validate_branch_redrive_projection_correction(
        correction, validate_live_authorities=False) == correction
    assert correction["production_source_transition"][
        "observed_changed_source_paths"] == sorted(
            design.BRANCH_REDRIVE_PROJECTION_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS)
    boundary = correction["partial_corpus_failure_boundary"]["corpus_receipt"]
    assert boundary["completed_states"] == 10
    assert boundary["attempted_branches"] == 120
    assert boundary["valid_branches"] == 120
    assert boundary["invalid_branches"] == 0
    assert len(correction["invalid_attempt_receipt_bindings"]) == 12
    assert [row["candidate_index"] for row in correction[
        "invalid_attempt_receipt_bindings"]] == list(range(12))
    failure = correction["branch_redrive_projection_failure_boundary"]
    assert failure[
        "active_manifest_field_copied_into_structural_evidence"] == \
        "candidate_indices"
    assert failure[
        "broad_exception_handler_overwrote_prior_comparison_truth_values"] is True
    assert failure[
        "reported_reason_proves_full_bank_l_max_ineligibility"] is False
    science = correction["preserved_scientific_contract"]
    assert science["state_identity_or_manifest_replacement_authorised"] is False
    assert science["unchanged_source_retry_authorised"] is False
    assert science["candidate_outcome_or_label_value_read_for_correction"] is False
    assert correction["issuance_boundary"][
        "later_consumption_requires_failure_time_partial_corpus_live"] is False
    raw = design._pretty_json_bytes(correction)
    binding = design.branch_redrive_projection_correction_artifact_binding(
        correction, raw)
    assert binding["self_digest"] == correction[
        design.BRANCH_REDRIVE_PROJECTION_CORRECTION_SELF_KEY]

    tampered = copy.deepcopy(correction)
    tampered["preserved_scientific_contract"][
        "state_identity_or_manifest_replacement_authorised"] = True
    with pytest.raises(design.ScorerFitCorpusV2DesignError):
        design.validate_branch_redrive_projection_correction(
            tampered, validate_live_authorities=False)

    extra_change = copy.deepcopy(correction["source_bindings"])
    unchanged = next(
        row for row in extra_change
        if row["path"] not in
        design.BRANCH_REDRIVE_PROJECTION_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS)
    unchanged["byte_count"] = int(unchanged["byte_count"]) + 1
    unchanged["sha256"] = "f" * 64
    with pytest.raises(design.ScorerFitCorpusV2DesignError):
        design.build_branch_redrive_projection_correction(
            source_repository_commit=correction["source_repository_commit"],
            source_bindings=extra_change,
            immutable_encoder_path_projection_correction=correction[
                "immutable_encoder_path_projection_correction"],
            partial_corpus_failure_boundary=correction[
                "partial_corpus_failure_boundary"],
            invalid_attempt_receipt_bindings=correction[
                "invalid_attempt_receipt_bindings"],
            completed_smoke_boundary=correction["completed_smoke_boundary"],
            downstream_outputs_absent_at_issue=correction[
                "downstream_outputs_absent_at_issue"],
        )


def test_branch_redrive_projection_issue_is_atomic_and_reopen_is_boundary_free(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    correction = _branch_redrive_projection_correction(monkeypatch)
    commit = correction["source_repository_commit"]
    sources = correction["source_bindings"]
    immutable_path = correction[
        "immutable_encoder_path_projection_correction"]
    partial = correction["partial_corpus_failure_boundary"]
    invalid = correction["invalid_attempt_receipt_bindings"]
    smoke = correction["completed_smoke_boundary"]
    absence = correction["downstream_outputs_absent_at_issue"]
    expected = tmp_path / \
        design.BRANCH_REDRIVE_PROJECTION_CORRECTION_RELATIVE_PATH
    staged = tmp_path / \
        design.BRANCH_REDRIVE_PROJECTION_CORRECTION_STAGED_RELATIVE_PATH
    expected.parent.mkdir(parents=True)
    calls = {"source": 0, "path": 0, "boundary": 0, "absence": 0,
             "install": 0}

    def clean_source(*, root: Path):
        assert root == tmp_path
        calls["source"] += 1
        return str(commit), copy.deepcopy(sources)

    def load_path(*, root: Path):
        assert root == tmp_path
        calls["path"] += 1
        return copy.deepcopy(immutable_path)

    def load_boundary(*, root: Path):
        assert root == tmp_path
        calls["boundary"] += 1
        return (copy.deepcopy(partial), copy.deepcopy(invalid),
                copy.deepcopy(smoke))

    def load_absence(*, root: Path):
        assert root == tmp_path
        calls["absence"] += 1
        return copy.deepcopy(absence)

    monkeypatch.setattr(design, "clean_source_authority", clean_source)
    monkeypatch.setattr(
        design, "_load_immutable_encoder_path_projection_correction",
        load_path)
    monkeypatch.setattr(
        design, "_validate_live_branch_redrive_failure_boundary",
        load_boundary)
    monkeypatch.setattr(
        design, "audit_branch_redrive_projection_correction_downstream_absence",
        load_absence)
    atomic_publish = design._exclusive_json_atomic_no_overwrite

    def checked_publish(path: Path, staged_path: Path,
                        payload: dict[str, object], *, label: str,
                        recover_nonexact_staged: bool) -> bytes:
        assert path == expected and staged_path == staged
        assert label == "branch-redrive projection correction"
        if not path.exists() and not path.is_symlink():
            assert recover_nonexact_staged is True
            assert calls == {"source": 2, "path": 2, "boundary": 2,
                             "absence": 2, "install": 0}
            calls["install"] += 1
        else:
            assert recover_nonexact_staged is False
        return atomic_publish(
            path, staged_path, payload, label=label,
            recover_nonexact_staged=recover_nonexact_staged)

    monkeypatch.setattr(
        design, "_exclusive_json_atomic_no_overwrite", checked_publish)
    issued = design.issue_branch_redrive_projection_correction(
        root=tmp_path, source_repository_commit=str(commit))
    assert issued == correction
    assert calls == {"source": 3, "path": 2, "boundary": 2,
                     "absence": 2, "install": 1}
    assert stat.S_IMODE(expected.stat().st_mode) == 0o444
    assert not staged.exists() and not staged.is_symlink()
    before = expected.read_bytes()

    def forbidden_boundary(**_kwargs):
        raise AssertionError("mutable failure boundary was reopened")

    monkeypatch.setattr(
        design, "_validate_live_branch_redrive_failure_boundary",
        forbidden_boundary)
    monkeypatch.setattr(
        design, "audit_branch_redrive_projection_correction_downstream_absence",
        forbidden_boundary)
    assert design.load_branch_redrive_projection_correction_for_consumption(
        root=tmp_path) == correction
    assert design.issue_branch_redrive_projection_correction(
        root=tmp_path, source_repository_commit=str(commit)) == correction
    assert expected.read_bytes() == before
    assert stat.S_IMODE(expected.stat().st_mode) == 0o444
    assert calls["install"] == 1


def test_optional_smoke_partial_corpus_resume_correction_is_closed_and_narrow(
        monkeypatch: pytest.MonkeyPatch) -> None:
    correction = _optional_smoke_partial_corpus_resume_correction(monkeypatch)
    assert design.validate_optional_smoke_partial_corpus_resume_correction(
        correction, validate_live_authorities=False) == correction
    assert correction["production_source_transition"][
        "observed_changed_source_paths"] == sorted(
            design
            .OPTIONAL_SMOKE_PARTIAL_CORPUS_RESUME_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS)
    assert [row["path"] for row in correction[
        "focused_test_source_transitions"]] == list(
            design
            .OPTIONAL_SMOKE_PARTIAL_CORPUS_RESUME_CORRECTION_FOCUSED_TEST_PATHS)
    failure = correction[
        "optional_smoke_partial_corpus_resume_failure_boundary"]
    assert failure["observed_partial_branch_count"] == 120
    assert failure["observed_completed_state_count"] == 10
    assert failure["transaction_state_before_failure"] == "COMPLETE"
    assert failure[
        "encoding_smoke_and_branch_smoke_receipt_digests_still_match"] is True
    assert failure[
        "matching_smoke_receipt_digests_prove_no_partial_progress"] is False
    assert failure["corrected_branch_command_started"] is False
    assert failure["scientific_or_feasibility_failure_established"] is False
    material = correction[
        "optional_smoke_partial_corpus_resume_correction"]
    assert material["complete_transaction_only_fast_path"] is True
    assert material[
        "noncomplete_transaction_loader_and_live_lineage_unchanged"] is True
    assert material[
        "matching_historical_smoke_digests_sufficient_for_no_partial_lag"] \
        is False
    assert material[
        "runner_partial_lag_detection_uses_strict_builder_producer"] is True
    assert material[
        "runner_partial_lag_requires_state_aligned_complete_candidate_banks"] \
        is True
    assert material[
        "runner_partial_lag_skips_only_stale_strict_smoke_replay"] is True
    assert material[
        "normal_full_encoder_refresh_after_1440_branches_required"] is True
    assert material[
        "strict_encoded_corpus_validation_after_refresh_required"] is True
    assert material[
        "prepared_and_complete_transaction_receipts_immutable"] is True
    assert material[
        "existing_encoding_smoke_preserved_during_partial_corpus_phase"] \
        is True
    assert material[
        "encoding_smoke_refresh_before_1440_branches_authorised"] is False
    assert material[
        "frozen_one_time_complete_corpus_encoding_smoke_refresh_at_1440_"
        "authorised"] is True
    assert material[
        "frozen_one_time_complete_corpus_encoding_smoke_refresh_at_1440_"
        "required"] is True
    assert material[
        "original_branch_smoke_preserved_during_partial_advancement"] is True
    assert material[
        "branch_smoke_rebinding_before_1440_branches_authorised"] is False
    assert material[
        "frozen_complete_corpus_branch_smoke_rebinding_at_1440_authorised"] \
        is True
    assert material["encoder_smoke_partial_cardinality_rule_changed"] is False
    assert material["full_encoder_cardinality_rule_changed"] is False
    science = correction["preserved_scientific_contract"]
    assert science["retained_valid_branch_count"] == 120
    assert science["retained_invalid_attempt_receipt_count"] == 12
    assert science["completed_transaction_receipts_preserved"] is True
    assert science["existing_frame_or_latent_regeneration_authorised"] is False
    assert science["existing_valid_branch_row_rewrite_authorised"] is False
    assert science[
        "advancing_compiled_ledger_and_corpus_receipt_refresh_authorised"] \
        is True
    assert science[
        "correction_issuance_rewrites_smoke_or_transaction_receipts"] is False
    assert science[
        "prepared_and_complete_transaction_receipts_immutable"] is True
    assert science[
        "existing_encoding_smoke_preserved_during_partial_corpus_phase"] \
        is True
    assert science[
        "encoding_smoke_refresh_before_1440_branches_authorised"] is False
    assert science[
        "frozen_one_time_complete_corpus_encoding_smoke_refresh_at_1440_"
        "authorised"] is True
    assert science[
        "frozen_one_time_complete_corpus_encoding_smoke_refresh_at_1440_"
        "required"] is True
    assert science[
        "original_branch_smoke_preserved_during_partial_corpus_advancement"] \
        is True
    assert science[
        "branch_smoke_rebinding_before_1440_branches_authorised"] is False
    assert science[
        "frozen_complete_corpus_branch_smoke_rebinding_at_1440_authorised"] \
        is True
    assert science[
        "runner_partial_lag_requires_strict_builder_producer_validation"] \
        is True
    assert science[
        "strict_encoded_corpus_validation_after_full_refresh_required"] is True
    assert science["resume_scope"] == "MISSING_REGISTERED_ASSIGNMENTS_ONLY"
    immutable = correction[
        "immutable_branch_redrive_projection_correction"]
    assert immutable["payload"][
        design.BRANCH_REDRIVE_PROJECTION_CORRECTION_SELF_KEY] == correction[
            "immutable_branch_redrive_projection_correction_digest"]
    raw = design._pretty_json_bytes(correction)
    binding = (
        design
        .optional_smoke_partial_corpus_resume_correction_artifact_binding(
            correction, raw))
    assert binding["self_digest"] == correction[
        design.OPTIONAL_SMOKE_PARTIAL_CORPUS_RESUME_CORRECTION_SELF_KEY]

    tampered = copy.deepcopy(correction)
    tampered["optional_smoke_partial_corpus_resume_correction"][
        "full_encoder_cardinality_rule_changed"] = True
    with pytest.raises(design.ScorerFitCorpusV2DesignError):
        design.validate_optional_smoke_partial_corpus_resume_correction(
            tampered, validate_live_authorities=False)

    extra_change = copy.deepcopy(correction["source_bindings"])
    unchanged = next(
        row for row in extra_change
        if row["path"] not in
        design
        .OPTIONAL_SMOKE_PARTIAL_CORPUS_RESUME_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS)
    unchanged["byte_count"] = int(unchanged["byte_count"]) + 1
    unchanged["sha256"] = "f" * 64
    with pytest.raises(design.ScorerFitCorpusV2DesignError):
        design.build_optional_smoke_partial_corpus_resume_correction(
            source_repository_commit=correction["source_repository_commit"],
            source_bindings=extra_change,
            focused_test_source_transitions=correction[
                "focused_test_source_transitions"],
            immutable_branch_redrive_projection_correction=correction[
                "immutable_branch_redrive_projection_correction"],
            downstream_outputs_absent_at_issue=correction[
                "downstream_outputs_absent_at_issue"],
        )


def test_optional_smoke_partial_corpus_resume_issue_is_atomic_and_reopen_is_boundary_free(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    correction = _optional_smoke_partial_corpus_resume_correction(monkeypatch)
    commit = correction["source_repository_commit"]
    sources = correction["source_bindings"]
    tests = correction["focused_test_source_transitions"]
    immutable_redrive = correction[
        "immutable_branch_redrive_projection_correction"]
    partial = correction["partial_corpus_failure_boundary"]
    invalid = correction["invalid_attempt_receipt_bindings"]
    smoke = correction["completed_smoke_boundary"]
    absence = correction["downstream_outputs_absent_at_issue"]
    expected = tmp_path / \
        design.OPTIONAL_SMOKE_PARTIAL_CORPUS_RESUME_CORRECTION_RELATIVE_PATH
    staged = tmp_path / \
        design.OPTIONAL_SMOKE_PARTIAL_CORPUS_RESUME_CORRECTION_STAGED_RELATIVE_PATH
    expected.parent.mkdir(parents=True)
    calls = {"source": 0, "redrive": 0, "tests": 0, "boundary": 0,
             "absence": 0, "install": 0}

    def clean_source(*, root: Path):
        assert root == tmp_path
        calls["source"] += 1
        return str(commit), copy.deepcopy(sources)

    def load_redrive(*, root: Path):
        assert root == tmp_path
        calls["redrive"] += 1
        return copy.deepcopy(immutable_redrive)

    def load_tests(*, root: Path):
        assert root == tmp_path
        calls["tests"] += 1
        return copy.deepcopy(tests)

    def load_boundary(*, root: Path):
        assert root == tmp_path
        calls["boundary"] += 1
        return (copy.deepcopy(partial), copy.deepcopy(invalid),
                copy.deepcopy(smoke))

    def load_absence(*, root: Path):
        assert root == tmp_path
        calls["absence"] += 1
        return copy.deepcopy(absence)

    monkeypatch.setattr(design, "clean_source_authority", clean_source)
    monkeypatch.setattr(
        design, "_load_immutable_branch_redrive_projection_correction",
        load_redrive)
    monkeypatch.setattr(
        design,
        "_optional_smoke_partial_corpus_resume_focused_test_source_transitions",
        load_tests)
    monkeypatch.setattr(
        design, "_validate_live_branch_redrive_failure_boundary",
        load_boundary)
    monkeypatch.setattr(
        design,
        "audit_optional_smoke_partial_corpus_resume_correction_downstream_absence",
        load_absence)
    atomic_publish = design._exclusive_json_atomic_no_overwrite

    def checked_publish(path: Path, staged_path: Path,
                        payload: dict[str, object], *, label: str,
                        recover_nonexact_staged: bool) -> bytes:
        assert path == expected and staged_path == staged
        assert label == "optional-smoke partial-corpus resume correction"
        if not path.exists() and not path.is_symlink():
            assert recover_nonexact_staged is True
            assert calls["redrive"] == 2
            assert calls["boundary"] == 2
            assert calls["absence"] == 2
            calls["install"] += 1
        else:
            assert recover_nonexact_staged is False
        return atomic_publish(
            path, staged_path, payload, label=label,
            recover_nonexact_staged=recover_nonexact_staged)

    monkeypatch.setattr(
        design, "_exclusive_json_atomic_no_overwrite", checked_publish)
    issued = design.issue_optional_smoke_partial_corpus_resume_correction(
        root=tmp_path, source_repository_commit=str(commit))
    assert issued == correction
    assert calls["install"] == 1
    assert stat.S_IMODE(expected.stat().st_mode) == 0o444
    assert not staged.exists() and not staged.is_symlink()
    before = expected.read_bytes()

    def forbidden_boundary(**_kwargs):
        raise AssertionError("mutable failure boundary was reopened")

    monkeypatch.setattr(
        design, "_validate_live_branch_redrive_failure_boundary",
        forbidden_boundary)
    monkeypatch.setattr(
        design,
        "audit_optional_smoke_partial_corpus_resume_correction_downstream_absence",
        forbidden_boundary)
    assert (
        design
        .load_optional_smoke_partial_corpus_resume_correction_for_consumption(
            root=tmp_path) == correction)
    assert design.issue_optional_smoke_partial_corpus_resume_correction(
        root=tmp_path, source_repository_commit=str(commit)) == correction
    assert expected.read_bytes() == before
    assert stat.S_IMODE(expected.stat().st_mode) == 0o444
    assert calls["install"] == 1


def test_encoder_path_projection_correction_is_chained_and_base_bound(
        monkeypatch: pytest.MonkeyPatch) -> None:
    correction = _encoder_path_projection_correction(monkeypatch)
    assert design.validate_encoder_path_projection_correction(
        correction, validate_live_authorities=False) == correction
    immutable_dtype = correction[
        "immutable_encoder_compute_dtype_correction"]
    assert immutable_dtype["payload"][
        design.ENCODER_COMPUTE_DTYPE_CORRECTION_SELF_KEY] == correction[
            "immutable_encoder_compute_dtype_correction_digest"]
    assert correction["immutable_successor_scorer_contract_binding"] == (
        design.IMMUTABLE_SUCCESSOR_SCORER_CONTRACT_BINDING)
    assert correction["production_source_transition"][
        "observed_changed_source_paths"] == sorted(
            design.ENCODER_PATH_PROJECTION_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS)
    bundle = correction["immutable_base_smoke_artifact_bundle"]
    assert bundle["total_latent_shard_count"] == 13
    assert bundle["context_latent_shard_count"] == 1
    assert bundle["horizon_latent_shard_count"] == 12
    assert bundle["total_latent_storage_bytes"] == 80_216_064
    assert len(bundle["latent_shard_inventory"]) == 13
    failure = correction["encoder_path_projection_failure_boundary"]
    assert failure["base_smoke_end_to_end_pass"] is True
    assert failure["base_smoke_protocol_complete"] is False
    assert failure["validator_write_attempted"] is False
    assert failure["zero_new_resume_started"] is False
    assert failure["single_shard_deletion_started"] is False
    assert failure[
        "single_shard_transaction_prepared_receipt_written"] is False
    assert failure["single_shard_transaction_backup_created"] is False
    assert failure[
        "single_shard_transaction_complete_receipt_written"] is False
    assert failure["full_corpus_latent_encoding_started"] is False
    assert failure["branch_outcome_or_label_value_used_for_correction"] is False
    assert "exception_message" not in failure
    assert failure["exception_message_suffix_claimed"] is False
    material = correction["encoder_path_projection_correction"]
    assert material[
        "path_projection_defect_is_read_only_validator_projection"] is True
    assert material[
        "branch_row_frame_or_latent_shard_changed_during_issue_or_path_"
        "digest_migration"] is False
    assert material["preprocessing_changed"] is False
    assert material["target_normalisation_changed"] is False
    assert material[
        "target_encoder_architecture_checkpoint_or_output_layer_changed"] is False
    assert material["latent_shape_token_order_or_storage_dtype_changed"] is False
    assert material[
        "path_digest_metadata_transition_requires_all_13_shard_bindings_"
        "unchanged"] is True
    assert material[
        "path_digest_metadata_current_current_recovery_requires_exact_file_"
        "reopen_fsync_and_parent_directory_fsync"] is True
    assert material["authorised_path_digest_metadata_transition"] == (
        "ADD_ENCODER_PATH_PROJECTION_CORRECTION_DIGEST_TO_INDEX_AND_"
        "SMOKE_WITHOUT_LATENT_SHARD_WRITE")
    assert material["direct_active_target_unlink_authorised"] is False
    assert material[
        "complete_receipt_before_pass_smoke_publication_required"] is True
    transaction = correction[
        "single_shard_regeneration_transaction_contract"]
    assert transaction == (
        design.ENCODER_PATH_PROJECTION_SINGLE_SHARD_REGENERATION_TRANSACTION_CONTRACT)
    assert correction[
        "single_shard_regeneration_transaction_contract_digest"] == (
            design.canonical_digest(transaction))
    assert transaction["immutable_receipt_publication"][
        "direct_write_to_final_path_allowed"] is False
    assert transaction["backup_contract"]["retained_after_complete"] is True
    assert transaction["pass_smoke_publication"][
        "complete_receipt_durable_before_pass_smoke"] is True
    audit = correction[
        "preissue_single_shard_regeneration_transaction_audit"]
    assert audit["observed_as_a_runtime_failure"] is False
    assert audit[
        "active_completion_proof_absent_in_historical_crash_window"] is True
    assert audit[
        "resume_could_authorise_second_deliberate_target_deletion"] is True
    assert audit["second_deliberate_target_deletion_observed"] is False
    assert audit["historical_encoder_publication_operation"] == (
        "OS_REPLACE_ACTIVE_SMOKE_TO_ARCHIVE_THEN_ATOMIC_JSON_SUCCESSOR")
    assert audit[
        "path_projection_correction_artifact_issued_when_discovered"] is False
    transaction_absence = correction[
        "single_shard_regeneration_transaction_artifacts_absent_at_issue"]
    assert {row["path"] for row in transaction_absence} == {
        str(design.FULL_BANK_V2_SMOKE_REGENERATION_PREPARED_RELATIVE_PATH),
        str(design.FULL_BANK_V2_SMOKE_REGENERATION_PREPARED_STAGED_RELATIVE_PATH),
        str(design.FULL_BANK_V2_SMOKE_REGENERATION_COMPLETE_RELATIVE_PATH),
        str(design.FULL_BANK_V2_SMOKE_REGENERATION_COMPLETE_STAGED_RELATIVE_PATH),
        str(design.FULL_BANK_V2_SMOKE_REGENERATION_BACKUP_RELATIVE_PATH),
        str(design.FULL_BANK_V2_SMOKE_REGENERATION_TRANSACTION_DIRECTORY_RELATIVE_PATH),
    }


def test_smoke_regeneration_receipts_are_closed_exact_and_reloadable(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    correction = _encoder_path_projection_correction(monkeypatch)
    prepared, complete = _synthetic_smoke_regeneration_receipts(correction)
    assert design.validate_full_bank_v2_smoke_regeneration_prepared_receipt(
        prepared) == prepared
    assert design.validate_full_bank_v2_smoke_regeneration_complete_receipt(
        complete) == complete
    target = prepared["designated_target"]
    backup = complete["retained_backup_binding"]
    regenerated = complete["regenerated_target_binding"]
    assert backup["device_id"] == target["device_id"]
    assert backup["inode"] == target["inode"]
    assert backup["mode_octal"] == target["mode_octal"]
    assert backup["link_count"] == target["link_count"] == 1
    assert regenerated["device_id"] == target["device_id"]
    assert regenerated["inode"] != target["inode"]
    assert {key: regenerated[key] for key in (
        "path", "candidate_index", "sha256", "byte_count", "shape",
    )} == {key: target[key] for key in (
        "path", "candidate_index", "sha256", "byte_count", "shape",
    )}
    contract = correction[
        "single_shard_regeneration_transaction_contract"]
    staged_recovery = contract["immutable_receipt_publication"][
        "partial_or_nonexact_staged_file_recovery"]
    assert staged_recovery[
        "active_target_or_backup_mutation_during_staged_rebuild_allowed"] is False
    assert staged_recovery["final_receipt_unlink_or_overwrite_allowed"] is False
    publication = contract["immutable_receipt_publication"]
    assert publication[
        "parent_directory_fsync_immediately_after_final_link_required"] is True
    assert publication[
        "parent_directory_fsync_immediately_after_staged_unlink_required"] is True
    exact_link_recovery = publication[
        "exact_final_and_exact_staged_link_recovery"]
    assert exact_link_recovery[
        "same_device_and_inode_hard_link_proof_required"] is True
    assert exact_link_recovery[
        "parent_fsync_before_staged_unlink_required"] is True
    assert exact_link_recovery[
        "staged_file_only_unlink_then_parent_fsync_required"] is True
    assert exact_link_recovery[
        "parent_fsync_after_staged_unlink_required"] is True
    assert exact_link_recovery[
        "final_receipt_target_or_backup_mutation_allowed"] is False
    backup_contract = contract["backup_contract"]
    assert backup_contract["atomic_move_primitive"] == "RENAME_NOREPLACE"
    assert backup_contract["absence_precheck_alone_is_not_no_overwrite"] is True
    assert backup_contract[
        "retained_backup_exact_reopen_before_durability_fsync_required"] is True
    assert backup_contract[
        "retained_backup_file_fsync_after_move_required"] is True
    assert backup_contract[
        "destination_directory_fsync_before_source_directory_required"] is True
    assert backup_contract[
        "source_directory_fsync_after_destination_directory_required"] is True
    assert backup_contract[
        "moved_resume_must_reestablish_backup_file_destination_directory_and_"
        "source_directory_durability_before_regeneration"] is True
    custody_contract = contract["non_target_custody_contract"]
    assert custody_contract["required_row_count"] == 12
    assert custody_contract["sha256_read_mode"] == "O_NOATIME_O_NOFOLLOW"
    assert custody_contract[
        "pretransaction_and_precomplete_canonical_digest_must_match"] is True
    assert contract["pass_smoke_publication"][
        "complete_receipt_binds_exact_final_smoke_bytes"] is True
    assert contract["pass_smoke_publication"][
        "original_protocol_pass_omits_complete_digest_to_avoid_cyclic_"
        "self_binding"] is True
    assert contract["pass_smoke_publication"][
        "original_protocol_pass_must_be_parsed_self_validated_and_cross_bound_"
        "to_prepared_and_complete_lineage"] is True
    assert contract["pass_smoke_publication"][
        "original_protocol_pass_is_stable_historical_witness_after_full_"
        "corpus_receipts_advance"] is True
    assert contract["authorised_mutation"][
        "registered_stable_artifact_inventory_must_be_recomputed_live_before_"
        "complete"] is True
    assert contract["authorised_mutation"][
        "all_non_target_shard_bytes_device_inode_mode_link_size_and_times_"
        "must_remain_unchanged"] is True
    assert contract["authorised_mutation"][
        "prepared_lineage_must_equal_live_zero_new_manifest_assignment_corpus_"
        "branch_smoke_contract_and_corrections_before_every_precomplete_"
        "mutation"] is True
    assert contract["authorised_mutation"][
        "complete_lineage_must_equal_prepared_and_original_protocol_pass_"
        "lineage_before_downstream_acceptance"] is True
    assert contract["authorised_mutation"][
        "restored_target_exact_reopen_and_file_fsync_before_complete_required"] \
        is True
    assert contract["authorised_mutation"][
        "restored_target_parent_directory_fsync_before_complete_required"] \
        is True
    assert contract["pass_smoke_publication"][
        "refreshed_smoke_must_be_fully_replayed_against_current_index_and_"
        "current_lineage"] is True
    assert contract["pass_smoke_publication"][
        "refreshed_smoke_current_corpus_and_branch_smoke_may_advance_from_"
        "prepared_partial_smoke_lineage"] is True
    assert contract["pass_smoke_publication"][
        "refreshed_smoke_state_assignment_scorer_and_correction_lineage_must_"
        "equal_prepared"] is True
    assert contract["pass_smoke_publication"][
        "exact_successor_active_replay_requires_file_fsync_and_parent_"
        "directory_fsync_before_acceptance"] is True
    assert set(contract["optional_validation_projection"]["required_fields"]) == {
        "transaction_state", "prepared_present", "prepared_receipt_digest",
        "target_state", "backup_state", "complete_present",
        "complete_receipt_digest", "pass_smoke_state", "next_action",
        "prepared_staged_state", "complete_staged_state", "target_exact",
        "backup_exact", "target_backup_custody_exact",
        "regenerated_target_custody_exact",
        "encoder_path_projection_correction_digest",
        "single_shard_regeneration_transaction_contract_digest",
        "candidate_outcomes_used_for_selection",
        "final_200_state_corpus_generated",
    }
    assert contract["optional_validation_projection"][
        "staged_receipt_states"] == ["ABSENT", "EXACT", "PARTIAL_REGULAR"]
    assert contract["optional_validation_projection"][
        "partial_or_nonexact_staged_receipt_recovery"] == {
            "prepared_allowed_only_in_unstarted_state": True,
            "complete_allowed_only_in_restored_complete_pending_state": True,
            "all_other_states": "FAIL_CLOSED",
        }

    for relative, payload in (
            (design.FULL_BANK_V2_SMOKE_REGENERATION_PREPARED_RELATIVE_PATH,
             prepared),
            (design.FULL_BANK_V2_SMOKE_REGENERATION_COMPLETE_RELATIVE_PATH,
             complete)):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(design._pretty_json_bytes(payload))
        path.chmod(0o444)
    assert design.load_full_bank_v2_smoke_regeneration_prepared_receipt(
        root=tmp_path) == prepared
    assert design.load_full_bank_v2_smoke_regeneration_complete_receipt(
        root=tmp_path) == complete


def test_smoke_regeneration_receipts_reject_stat_and_publication_tamper(
        monkeypatch: pytest.MonkeyPatch) -> None:
    correction = _encoder_path_projection_correction(monkeypatch)
    prepared, complete = _synthetic_smoke_regeneration_receipts(correction)

    bad_prepared = copy.deepcopy(prepared)
    bad_prepared["designated_target"]["device_id"] = True
    with pytest.raises(design.ScorerFitCorpusV2DesignError):
        design.validate_full_bank_v2_smoke_regeneration_prepared_receipt(
            bad_prepared)

    bad_publication = copy.deepcopy(prepared)
    bad_publication["receipt_publication_contract"][
        "direct_write_to_final_path_allowed"] = True
    with pytest.raises(design.ScorerFitCorpusV2DesignError):
        design.validate_full_bank_v2_smoke_regeneration_prepared_receipt(
            bad_publication)

    same_inode = copy.deepcopy(complete)
    same_inode["regenerated_target_binding"]["inode"] = same_inode[
        "designated_target"]["inode"]
    with pytest.raises(design.ScorerFitCorpusV2DesignError):
        design.validate_full_bank_v2_smoke_regeneration_complete_receipt(
            same_inode)

    changed_backup = copy.deepcopy(complete)
    changed_backup["retained_backup_binding"]["inode"] += 1
    with pytest.raises(design.ScorerFitCorpusV2DesignError):
        design.validate_full_bank_v2_smoke_regeneration_complete_receipt(
            changed_backup)

    changed_smoke = copy.deepcopy(complete)
    changed_smoke["final_smoke_receipt_binding"]["raw_sha256"] = "0" * 64
    with pytest.raises(design.ScorerFitCorpusV2DesignError):
        design.validate_full_bank_v2_smoke_regeneration_complete_receipt(
            changed_smoke)

    contradictory_post = copy.deepcopy(complete["posttransaction_evidence"])
    contradictory_post["encoding_smoke_receipt_digest"] = "c" * 64
    with pytest.raises(
            design.ScorerFitCorpusV2DesignError,
            match="contradictory smoke digests"):
        design.build_full_bank_v2_smoke_regeneration_complete_receipt(
            prepared_receipt_binding=complete["prepared_receipt_binding"],
            lineage=complete["lineage"],
            designated_target=complete["designated_target"],
            retained_backup_binding=complete["retained_backup_binding"],
            regenerated_target_binding=complete["regenerated_target_binding"],
            non_target_shard_inventory_digest=complete[
                "non_target_shard_inventory_digest"],
            posttransaction_evidence=contradictory_post,
            final_smoke_receipt_binding=complete[
                "final_smoke_receipt_binding"],
        )

    changed_custody = copy.deepcopy(complete["posttransaction_evidence"])
    changed_custody[
        "registered_smoke_non_target_shard_custody_inventory_digest"] = (
            "0" * 64)
    with pytest.raises(
            design.ScorerFitCorpusV2DesignError,
            match="registered stable artifact"):
        design.build_full_bank_v2_smoke_regeneration_complete_receipt(
            prepared_receipt_binding=complete["prepared_receipt_binding"],
            lineage=complete["lineage"],
            designated_target=complete["designated_target"],
            retained_backup_binding=complete["retained_backup_binding"],
            regenerated_target_binding=complete[
                "regenerated_target_binding"],
            non_target_shard_inventory_digest=complete[
                "non_target_shard_inventory_digest"],
            posttransaction_evidence=changed_custody,
            final_smoke_receipt_binding=complete[
                "final_smoke_receipt_binding"],
        )

    changed_lineage = copy.deepcopy(complete["lineage"])
    changed_lineage["state_manifest_digest"] = "0" * 64
    with pytest.raises(
            design.ScorerFitCorpusV2DesignError,
            match="changed PREPARED lineage"):
        design.build_full_bank_v2_smoke_regeneration_complete_receipt(
            prepared_receipt_binding=complete["prepared_receipt_binding"],
            lineage=changed_lineage,
            designated_target=complete["designated_target"],
            retained_backup_binding=complete["retained_backup_binding"],
            regenerated_target_binding=complete["regenerated_target_binding"],
            non_target_shard_inventory_digest=complete[
                "non_target_shard_inventory_digest"],
            posttransaction_evidence=complete["posttransaction_evidence"],
            final_smoke_receipt_binding=complete[
                "final_smoke_receipt_binding"],
        )


def test_smoke_regeneration_transaction_absence_audit_is_exact(
        tmp_path: Path) -> None:
    expected = design._expected_encoder_path_projection_transaction_absence_rows()
    assert design.audit_encoder_path_projection_transaction_artifacts_absent(
        root=tmp_path) == expected
    transaction_directory = (
        tmp_path /
        design.FULL_BANK_V2_SMOKE_REGENERATION_TRANSACTION_DIRECTORY_RELATIVE_PATH)
    transaction_directory.mkdir(parents=True)
    with pytest.raises(design.ScorerFitCorpusV2DesignError):
        design.audit_encoder_path_projection_transaction_artifacts_absent(
            root=tmp_path)


def test_encoder_path_projection_correction_rejects_tamper_and_extra_change(
        monkeypatch: pytest.MonkeyPatch) -> None:
    correction = _encoder_path_projection_correction(monkeypatch)
    raw = (json.dumps(correction, sort_keys=True, indent=2) + "\n").encode()
    assert design.encoder_path_projection_correction_artifact_binding(
        correction, raw)["self_digest"] == correction[
            design.ENCODER_PATH_PROJECTION_CORRECTION_SELF_KEY]
    tampered = copy.deepcopy(correction)
    tampered["immutable_base_smoke_artifact_bundle"][
        "latent_shard_inventory"][0]["sha256"] = "f" * 64
    with pytest.raises(design.ScorerFitCorpusV2DesignError):
        design.validate_encoder_path_projection_correction(
            tampered, validate_live_authorities=False)
    transaction_tamper = copy.deepcopy(correction)
    transaction_tamper["single_shard_regeneration_transaction_contract"][
        "backup_contract"]["retained_after_complete"] = False
    with pytest.raises(design.ScorerFitCorpusV2DesignError):
        design.validate_encoder_path_projection_correction(
            transaction_tamper, validate_live_authorities=False)

    sources = copy.deepcopy(correction["source_bindings"])
    unchanged = next(
        row for row in sources
        if row["path"] not in set(
            design.ENCODER_PATH_PROJECTION_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS))
    unchanged["byte_count"] = int(unchanged["byte_count"]) + 1
    unchanged["sha256"] = "f" * 64
    with pytest.raises(design.ScorerFitCorpusV2DesignError):
        design.build_encoder_path_projection_correction(
            source_repository_commit=correction["source_repository_commit"],
            source_bindings=sources,
            immutable_encoder_compute_dtype_correction=correction[
                "immutable_encoder_compute_dtype_correction"],
            immutable_successor_scorer_contract_binding=correction[
                "immutable_successor_scorer_contract_binding"],
            focused_test_source_transitions=correction[
                "focused_test_source_transitions"],
            failed_encoder_source_binding=correction[
                "failed_encoder_source_binding"],
            base_smoke_artifact_bundle=correction[
                "immutable_base_smoke_artifact_bundle"],
            downstream_outputs_absent_at_issue=correction[
                "downstream_outputs_absent_at_issue"],
            single_shard_regeneration_transaction_artifacts_absent_at_issue=
                correction[
                    "single_shard_regeneration_transaction_artifacts_"
                    "absent_at_issue"],
        )


def test_encoder_path_projection_issue_reopen_and_refresh_lifecycle(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    correction = _encoder_path_projection_correction(monkeypatch)
    commit = correction["source_repository_commit"]
    sources = correction["source_bindings"]
    immutable_dtype = correction[
        "immutable_encoder_compute_dtype_correction"]
    successor = correction["immutable_successor_scorer_contract_binding"]
    tests = correction["focused_test_source_transitions"]
    failed = correction["failed_encoder_source_binding"]
    bundle = correction["immutable_base_smoke_artifact_bundle"]
    absence = correction["downstream_outputs_absent_at_issue"]
    transaction_absence = correction[
        "single_shard_regeneration_transaction_artifacts_absent_at_issue"]
    expected = tmp_path / design.ENCODER_PATH_PROJECTION_CORRECTION_RELATIVE_PATH
    staged = (
        tmp_path / design.ENCODER_PATH_PROJECTION_CORRECTION_STAGED_RELATIVE_PATH)
    expected.parent.mkdir(parents=True)
    calls = {
        "source": 0, "dtype": 0, "successor": 0, "tests": 0,
        "failed": 0, "bundle": 0, "absence": 0,
        "transaction_absence": 0, "install": 0,
    }

    def clean_source(*, root: Path) -> tuple[str, list[dict[str, object]]]:
        assert root == tmp_path
        calls["source"] += 1
        return str(commit), copy.deepcopy(sources)

    def load_dtype(*, root: Path) -> dict[str, object]:
        assert root == tmp_path
        calls["dtype"] += 1
        return copy.deepcopy(immutable_dtype)

    def load_successor(*, root: Path) -> dict[str, object]:
        assert root == tmp_path
        calls["successor"] += 1
        return copy.deepcopy(successor)

    def load_tests(*, root: Path) -> list[dict[str, object]]:
        assert root == tmp_path
        calls["tests"] += 1
        return copy.deepcopy(tests)

    def load_failed(*, root: Path) -> dict[str, object]:
        assert root == tmp_path
        calls["failed"] += 1
        return copy.deepcopy(failed)

    def load_bundle(*, root: Path) -> dict[str, object]:
        assert root == tmp_path
        calls["bundle"] += 1
        return copy.deepcopy(bundle)

    def load_absence(*, root: Path) -> list[dict[str, object]]:
        assert root == tmp_path
        calls["absence"] += 1
        return copy.deepcopy(absence)

    def load_transaction_absence(
            *, root: Path) -> list[dict[str, object]]:
        assert root == tmp_path
        calls["transaction_absence"] += 1
        return copy.deepcopy(transaction_absence)

    monkeypatch.setattr(design, "clean_source_authority", clean_source)
    monkeypatch.setattr(
        design, "_load_immutable_encoder_compute_dtype_correction", load_dtype)
    monkeypatch.setattr(
        design, "_load_immutable_successor_scorer_contract_binding",
        load_successor)
    monkeypatch.setattr(
        design, "_encoder_path_projection_focused_test_source_transitions",
        load_tests)
    monkeypatch.setattr(
        design, "_validate_live_encoder_path_projection_failure_source",
        load_failed)
    monkeypatch.setattr(
        design, "_validate_live_encoder_path_projection_base_bundle",
        load_bundle)
    monkeypatch.setattr(
        design, "audit_encoder_path_projection_correction_downstream_absence",
        load_absence)
    monkeypatch.setattr(
        design, "audit_encoder_path_projection_transaction_artifacts_absent",
        load_transaction_absence)
    atomic_publish = design._exclusive_json_atomic_no_overwrite

    def checked_atomic_publish(
            path: Path, staged_path: Path, payload: dict[str, object], *,
            label: str, recover_nonexact_staged: bool) -> bytes:
        assert path == expected
        assert staged_path == staged
        assert label == "encoder-path-projection correction"
        if not path.exists() and not path.is_symlink():
            assert recover_nonexact_staged is True
            assert {key: calls[key] for key in (
                "source", "dtype", "successor", "tests", "failed", "bundle",
                "absence", "transaction_absence",
            )} == {
                "source": 2, "dtype": 2, "successor": 2, "tests": 2,
                "failed": 2, "bundle": 2, "absence": 2,
                "transaction_absence": 2,
            }
            calls["install"] += 1
        else:
            assert recover_nonexact_staged is False
        return atomic_publish(
            path, staged_path, payload, label=label,
            recover_nonexact_staged=recover_nonexact_staged)

    monkeypatch.setattr(
        design, "_exclusive_json_atomic_no_overwrite", checked_atomic_publish)
    issued = design.issue_encoder_path_projection_correction(
        root=tmp_path, source_repository_commit=str(commit))
    assert issued == correction
    assert calls == {
        "source": 3, "dtype": 3, "successor": 3, "tests": 3,
        "failed": 3, "bundle": 3, "absence": 3,
        "transaction_absence": 3, "install": 1,
    }
    assert stat.S_IMODE(expected.stat().st_mode) == 0o444
    assert not staged.exists() and not staged.is_symlink()
    raw = expected.read_bytes()

    refreshed_bundle_checks: list[int] = []
    refreshed_absence_checks: list[int] = []
    refreshed_transaction_absence_checks: list[int] = []

    def refreshed_bundle(*, root: Path) -> dict[str, object]:
        assert root == tmp_path
        refreshed_bundle_checks.append(1)
        return {"refreshed": True}

    def refreshed_absence(*, root: Path) -> list[dict[str, object]]:
        assert root == tmp_path
        refreshed_absence_checks.append(1)
        return [{"refreshed": True}]

    def refreshed_transaction_absence(
            *, root: Path) -> list[dict[str, object]]:
        assert root == tmp_path
        refreshed_transaction_absence_checks.append(1)
        return [{"refreshed": True}]

    monkeypatch.setattr(
        design, "_validate_live_encoder_path_projection_base_bundle",
        refreshed_bundle)
    monkeypatch.setattr(
        design, "audit_encoder_path_projection_correction_downstream_absence",
        refreshed_absence)
    monkeypatch.setattr(
        design, "audit_encoder_path_projection_transaction_artifacts_absent",
        refreshed_transaction_absence)
    assert design.load_encoder_path_projection_correction_for_consumption(
        root=tmp_path, require_failure_boundary_live=False) == correction
    assert design.issue_encoder_path_projection_correction(
        root=tmp_path, source_repository_commit=str(commit)) == correction
    assert refreshed_bundle_checks == []
    assert refreshed_absence_checks == []
    assert refreshed_transaction_absence_checks == []
    assert calls["install"] == 1
    assert expected.read_bytes() == raw
    assert stat.S_IMODE(expected.stat().st_mode) == 0o444


def test_encoder_path_projection_atomic_authority_publication_recovers_staging(
        tmp_path: Path) -> None:
    payload = {"schema": "synthetic_authority", "value": 7}
    raw = design._pretty_json_bytes(payload)

    partial_case = tmp_path / "partial"
    partial_case.mkdir()
    final_path = partial_case / "authority.json"
    staged_path = partial_case / "authority.json.staged"
    staged_path.write_bytes(raw[:11])
    staged_path.chmod(0o444)
    assert design._exclusive_json_atomic_no_overwrite(
        final_path, staged_path, payload, label="synthetic authority",
        recover_nonexact_staged=True) == raw
    assert final_path.read_bytes() == raw
    assert stat.S_IMODE(final_path.stat().st_mode) == 0o444
    assert not staged_path.exists() and not staged_path.is_symlink()

    linked_case = tmp_path / "linked"
    linked_case.mkdir()
    linked_final = linked_case / "authority.json"
    linked_staged = linked_case / "authority.json.staged"
    design._exclusive_json_atomic_no_overwrite(
        linked_final, linked_staged, payload, label="synthetic authority",
        recover_nonexact_staged=True)
    os.link(linked_final, linked_staged)
    final_stat = linked_final.stat()
    assert linked_staged.stat().st_ino == final_stat.st_ino
    design._exclusive_json_atomic_no_overwrite(
        linked_final, linked_staged, payload, label="synthetic authority",
        recover_nonexact_staged=False)
    assert linked_final.stat().st_ino == final_stat.st_ino
    assert not linked_staged.exists() and not linked_staged.is_symlink()

    collision_case = tmp_path / "collision"
    collision_case.mkdir()
    collision_final = collision_case / "authority.json"
    collision_staged = collision_case / "authority.json.staged"
    collision_final.write_bytes(b"different\n")
    collision_final.chmod(0o444)
    with pytest.raises(
            design.ScorerFitCorpusV2DesignError,
            match="immutable final collision"):
        design._exclusive_json_atomic_no_overwrite(
            collision_final, collision_staged, payload,
            label="synthetic authority", recover_nonexact_staged=True)


def test_path_projection_authority_hashes_shards_without_atime_change(
        tmp_path: Path) -> None:
    shard = tmp_path / "latent.f16"
    raw = b"synthetic-latent-shard" * 257
    shard.write_bytes(raw)
    atime_ns = 1_600_000_000_123_456_789
    mtime_ns = 1_600_000_001_987_654_321
    os.utime(shard, ns=(atime_ns, mtime_ns))
    before = shard.stat()
    digest, byte_count = design._sha256_regular_file(
        shard, label="synthetic latent shard")
    second_digest, second_byte_count = design._sha256_regular_file(
        shard, label="synthetic latent shard second pass")
    after = shard.stat()
    assert digest == hashlib.sha256(raw).hexdigest()
    assert byte_count == len(raw)
    assert (second_digest, second_byte_count) == (digest, byte_count)
    assert after.st_atime_ns == before.st_atime_ns == atime_ns
    assert after.st_mtime_ns == before.st_mtime_ns == mtime_ns


def test_encoder_compute_dtype_correction_is_chained_truthful_and_fp32_bound(
        monkeypatch: pytest.MonkeyPatch) -> None:
    correction = _encoder_compute_dtype_correction(monkeypatch)
    assert design.validate_encoder_compute_dtype_correction(
        correction, validate_live_authorities=False) == correction
    immutable_import = correction["immutable_encoder_import_correction"]
    assert immutable_import["payload"][
        design.ENCODER_IMPORT_CORRECTION_SELF_KEY] == correction[
            "immutable_encoder_import_correction_digest"]
    assert correction["immutable_successor_scorer_contract_binding"] == (
        design.IMMUTABLE_SUCCESSOR_SCORER_CONTRACT_BINDING)
    assert correction["production_source_transition"][
        "observed_changed_source_paths"] == sorted(
            design.ENCODER_COMPUTE_DTYPE_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS)
    assert correction["unchanged_stage_a_fp32_source_binding"][
        "sha256"] == design.ENCODER_COMPUTE_DTYPE_STAGE_A_FP32_SOURCE_BINDING[
            "sha256"]
    material = correction["encoder_compute_dtype_correction"]
    assert material["failed_compute_dtype"] == "bfloat16"
    assert material["corrected_compute_dtype"] == "float32"
    assert material["latent_storage_dtype"] == "float16"
    assert material["automatic_mixed_precision_or_autocast_enabled"] is False
    assert material["runtime_compute_dtype_restored_to_frozen_stage_a"] is True
    assert material["preprocessing_changed"] is False
    assert material["target_normalisation_changed"] is False
    assert material["target_encoder_architecture_changed"] is False
    assert material["target_encoder_checkpoint_changed"] is False
    assert material["target_encoder_output_layer_changed"] is False
    assert material["scientific_target_encoder_contract_changed"] is False
    assert "preprocessing_normalisation_or_target_encoding_changed" not in material
    failure = correction["encoder_compute_dtype_failure_boundary"]
    assert failure["target_encoder_constructor_completed"] is True
    assert failure["checkpoint_torch_load_map_location_cpu_completed"] is True
    assert failure["strict_encoder_state_dict_load_completed"] is True
    assert failure["first_encoder_forward_entered"] is True
    assert failure["rope_query_dtype_at_failure"] == "float32"
    assert failure["attention_value_dtype_at_failure"] == "bfloat16"
    assert failure["atomic_f16_reached"] is False
    assert failure["context_latent_shard_written"] is False
    assert failure["horizon_latent_shard_written"] is False
    assert failure[
        "branch_outcome_or_label_value_consumed_for_correction"] is False
    assert failure["branch_frame_value_opened_by_correction_issuer"] is False


def test_encoder_compute_dtype_correction_rejects_tamper_and_extra_change(
        monkeypatch: pytest.MonkeyPatch) -> None:
    correction = _encoder_compute_dtype_correction(monkeypatch)
    raw = (json.dumps(correction, sort_keys=True, indent=2) + "\n").encode()
    binding = design.encoder_compute_dtype_correction_artifact_binding(
        correction, raw)
    assert binding["self_digest"] == correction[
        design.ENCODER_COMPUTE_DTYPE_CORRECTION_SELF_KEY]
    tampered = copy.deepcopy(correction)
    tampered["encoder_compute_dtype_failure_boundary"][
        "atomic_f16_reached"] = True
    with pytest.raises(design.ScorerFitCorpusV2DesignError):
        design.validate_encoder_compute_dtype_correction(
            tampered, validate_live_authorities=False)

    sources = copy.deepcopy(correction["source_bindings"])
    unchanged = next(
        row for row in sources
        if row["path"] not in set(
            design.ENCODER_COMPUTE_DTYPE_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS))
    unchanged["byte_count"] = int(unchanged["byte_count"]) + 1
    unchanged["sha256"] = "f" * 64
    with pytest.raises(design.ScorerFitCorpusV2DesignError):
        design.build_encoder_compute_dtype_correction(
            source_repository_commit=correction["source_repository_commit"],
            source_bindings=sources,
            immutable_encoder_import_correction=correction[
                "immutable_encoder_import_correction"],
            immutable_successor_scorer_contract_binding=correction[
                "immutable_successor_scorer_contract_binding"],
            focused_test_source_transitions=correction[
                "focused_test_source_transitions"],
            branch_smoke_binding=correction["immutable_branch_smoke_binding"],
            branch_corpus_binding=correction[
                "immutable_partial_corpus_receipt_binding"],
            failed_encoder_source_binding=correction[
                "failed_encoder_source_binding"],
            unchanged_dev_encoder_source_binding=correction[
                "unchanged_dev_encoder_source_binding"],
            unchanged_stage_a_fp32_source_binding=correction[
                "unchanged_stage_a_fp32_source_binding"],
            upstream_rope_source_binding=correction[
                "upstream_rope_source_binding"],
            prelatent_outputs_absent_at_issue=correction[
                "prelatent_outputs_absent_at_issue"],
        )


def test_encoder_compute_dtype_issue_reopen_and_receipt_refresh_lifecycle(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    correction = _encoder_compute_dtype_correction(monkeypatch)
    commit = correction["source_repository_commit"]
    sources = correction["source_bindings"]
    immutable_import = correction["immutable_encoder_import_correction"]
    successor = correction["immutable_successor_scorer_contract_binding"]
    tests = correction["focused_test_source_transitions"]
    evidence = (
        correction["failed_encoder_source_binding"],
        correction["unchanged_dev_encoder_source_binding"],
        correction["unchanged_stage_a_fp32_source_binding"],
        correction["upstream_rope_source_binding"],
    )
    smoke = correction["immutable_branch_smoke_binding"]
    corpus = correction["immutable_partial_corpus_receipt_binding"]
    absence = correction["prelatent_outputs_absent_at_issue"]
    expected = tmp_path / design.ENCODER_COMPUTE_DTYPE_CORRECTION_RELATIVE_PATH
    expected.parent.mkdir(parents=True)

    calls = {
        "source": 0, "import": 0, "successor": 0, "tests": 0,
        "evidence": 0, "receipts": 0, "absence": 0, "install": 0,
    }

    def clean_source(*, root: Path) -> tuple[str, list[dict[str, object]]]:
        assert root == tmp_path
        calls["source"] += 1
        return str(commit), copy.deepcopy(sources)

    def load_import(*, root: Path) -> dict[str, object]:
        assert root == tmp_path
        calls["import"] += 1
        return copy.deepcopy(immutable_import)

    def load_successor(*, root: Path) -> dict[str, object]:
        assert root == tmp_path
        calls["successor"] += 1
        return copy.deepcopy(successor)

    def load_tests(*, root: Path) -> list[dict[str, object]]:
        assert root == tmp_path
        calls["tests"] += 1
        return copy.deepcopy(tests)

    def load_evidence(
            *, root: Path,
            ) -> tuple[dict[str, object], dict[str, object],
                       dict[str, object], dict[str, object]]:
        assert root == tmp_path
        calls["evidence"] += 1
        return copy.deepcopy(evidence)

    def load_receipts(
            *, root: Path,
            ) -> tuple[dict[str, object], dict[str, object]]:
        assert root == tmp_path
        calls["receipts"] += 1
        return copy.deepcopy(smoke), copy.deepcopy(corpus)

    def load_absence(*, root: Path) -> list[dict[str, object]]:
        assert root == tmp_path
        calls["absence"] += 1
        return copy.deepcopy(absence)

    monkeypatch.setattr(design, "clean_source_authority", clean_source)
    monkeypatch.setattr(
        design, "_load_immutable_encoder_import_correction", load_import)
    monkeypatch.setattr(
        design, "_load_immutable_successor_scorer_contract_binding",
        load_successor)
    monkeypatch.setattr(
        design, "_encoder_compute_dtype_focused_test_source_transitions",
        load_tests)
    monkeypatch.setattr(
        design, "_validate_live_encoder_compute_dtype_source_evidence",
        load_evidence)
    monkeypatch.setattr(
        design, "_validate_live_encoder_import_failure_receipts",
        load_receipts)
    monkeypatch.setattr(
        design, "audit_encoder_compute_dtype_correction_prelatent_absence",
        load_absence)

    exclusive_json = design._exclusive_json

    def checked_exclusive_json(
            path: Path, payload: dict[str, object], *, label: str) -> None:
        assert path == expected
        assert label == "encoder-compute-dtype correction"
        assert not path.exists() and not path.is_symlink()
        assert {key: calls[key] for key in (
            "source", "import", "successor", "tests", "evidence",
            "receipts", "absence",
        )} == {
            "source": 2, "import": 2, "successor": 2, "tests": 2,
            "evidence": 2, "receipts": 2, "absence": 2,
        }
        calls["install"] += 1
        exclusive_json(path, payload, label=label)

    monkeypatch.setattr(design, "_exclusive_json", checked_exclusive_json)
    issued = design.issue_encoder_compute_dtype_correction(
        root=tmp_path, source_repository_commit=str(commit))
    assert issued == correction
    assert calls == {
        "source": 3, "import": 3, "successor": 3, "tests": 3,
        "evidence": 3, "receipts": 3, "absence": 3, "install": 1,
    }
    assert stat.S_IMODE(expected.stat().st_mode) == 0o444
    raw = expected.read_bytes()
    assert design.encoder_compute_dtype_correction_artifact_binding(
        correction, raw)["self_digest"] == correction[
            design.ENCODER_COMPUTE_DTYPE_CORRECTION_SELF_KEY]

    refreshed_receipt_checks: list[int] = []
    refreshed_absence_checks: list[int] = []

    def refreshed_receipts(
            *, root: Path,
            ) -> tuple[dict[str, object], dict[str, object]]:
        assert root == tmp_path
        refreshed_receipt_checks.append(1)
        return {"refreshed": True}, {"refreshed": True}

    def refreshed_absence(*, root: Path) -> list[dict[str, object]]:
        assert root == tmp_path
        refreshed_absence_checks.append(1)
        return [{"refreshed": True}]

    monkeypatch.setattr(
        design, "_validate_live_encoder_import_failure_receipts",
        refreshed_receipts)
    monkeypatch.setattr(
        design, "audit_encoder_compute_dtype_correction_prelatent_absence",
        refreshed_absence)
    reopened = design.load_encoder_compute_dtype_correction_for_consumption(
        root=tmp_path, require_failure_boundary_live=False)
    assert reopened == correction
    assert design.issue_encoder_compute_dtype_correction(
        root=tmp_path, source_repository_commit=str(commit)) == correction
    assert refreshed_receipt_checks == []
    assert refreshed_absence_checks == []
    assert calls["install"] == 1
    assert expected.read_bytes() == raw
    assert stat.S_IMODE(expected.stat().st_mode) == 0o444
