#!/usr/bin/env python3
"""Materialize and load the exact metadata-only bounded-branch admission.

The source attempt is a consumed terminal failure, but its physics result and
receipt check are complete.  This module never changes that attempt and never
calls a collector, renderer, simulator, or the legacy mutating join entrypoint.
It reuses only the frozen pixel verifier and pure joined-document builder.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from types import MappingProxyType
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as pilot  # noqa: E402
from lewm.datasets import go2_world_model_counterfactual_pilot_v1 as consumer  # noqa: E402
from scripts import analyze_go2_world_model_counterfactual_calibration_v1 as calibration  # noqa: E402
from scripts import join_go2_world_model_counterfactual_pilot_v1 as joiner  # noqa: E402


PREREGISTRATION = (
    REPO_ROOT
    / "docs/lewm_go2_world_model_bounded_branch_posthoc_join_admission_v1_"
    "preregistration_2026-08-02.json"
)
PREREGISTRATION_SHA256 = (
    "ee2cbd54459760c23107e961f9ae4c860a5cfccfeb3cdb332a5449394031fdbf"
)
PREREGISTRATION_BYTE_COUNT = 6_320
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / ".generated/dev/lewm-go2-wm-bounded-branch-posthoc-join-admission-v1"
)

AUTHORITY_SCHEMA = (
    "lewm_go2_world_model_bounded_branch_posthoc_join_admission_authority_v1"
)
AUTHORITY_STATUS = "AUTHORIZED_ONE_EXACT_POSTHOC_JOIN_ADMISSION_ONLY"
MANIFEST_SCHEMA = (
    "lewm_go2_world_model_bounded_branch_posthoc_join_admission_manifest_v1"
)
MANIFEST_STATUS = "COMPLETE_POSTHOC_METADATA_DERIVATION_PENDING_REVIEW"
TERMINAL_SCHEMA = (
    "lewm_go2_world_model_bounded_branch_posthoc_join_admission_terminal_v1"
)
TERMINAL_SUCCESS = "COMPLETE_PENDING_INDEPENDENT_REVIEW"
TERMINAL_FAILURE = "CONSUMED_TERMINAL_FAILURE"

EXPECTED_COUNTS = {
    "actions": 9,
    "candidate_branches": 2_304,
    "context_frames": 768,
    "roles": {"eval": 128, "train": 128},
    "scenes": 32,
    "sentinel_branches": 0,
    "states": 256,
    "target_frames": 2_304,
    "total_branches": 2_304,
}
EXPECTED_LEAVES = {
    "rgb_manifest": {
        "name": "rgb_manifest.json",
        "byte_count": 1_880_307,
        "sha256": "5e03afa7665ffef54a1cab5e37135a18d42761bc844ecefacaa433f75a1b1f7e",
    },
    "train": {
        "name": "train.jsonl",
        "byte_count": 30_432_624,
        "sha256": "edc6f88bb105c39575477fbfbb0224bf0312cf5ee3e90551f86a9c11c2ebb447",
    },
    "eval": {
        "name": "eval.jsonl",
        "byte_count": 30_411_588,
        "sha256": "531debbc431f2f8afc83a491b491b8822134c831b16ca4d283fe1e7f4ba07768",
    },
}
CONSUMER_COMPATIBILITY_PROJECTION = (
    "validate_and_omit_redundant_all_zero_per_lane_and_rms_sync_diagnostics_"
    "for_frozen_three_field_group_parser_only"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PosthocJoinAdmissionError(RuntimeError):
    """Raised when the exact split-root admission changes."""


@dataclass(frozen=True)
class DerivedDocumentsV1:
    collection: Mapping[str, Any]
    calibration_receipt: Mapping[str, Any]
    rgb_manifest: Mapping[str, Any]
    rows: Mapping[str, Sequence[Mapping[str, Any]]]
    metadata: Mapping[str, Any]
    raw_by_leaf: Mapping[str, bytes]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _protected_name(name: str) -> bool:
    lowered = name.lower()
    return (
        lowered == "sealed_test.json"
        or lowered == "sealed"
        or lowered.startswith("sealed_")
        or lowered in {"heldout", "held_out", "held-out"}
        or lowered.startswith("heldout_")
        or lowered.startswith("held_out_")
        or lowered.startswith("held-out-")
    )


def _reject_protected(path: Path, *, label: str) -> None:
    if any(_protected_name(part) for part in Path(path).parts):
        raise PosthocJoinAdmissionError(f"{label} names protected material")


def _absolute_nofollow(path: Path, *, label: str, must_exist: bool) -> Path:
    selected = Path(os.path.abspath(os.fspath(path)))
    _reject_protected(selected, label=label)
    cursor = Path(selected.anchor)
    for part in selected.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise PosthocJoinAdmissionError(f"{label} traverses a symlink")
        if not cursor.exists():
            if must_exist:
                raise PosthocJoinAdmissionError(f"{label} is absent")
            break
    if must_exist and not selected.exists():
        raise PosthocJoinAdmissionError(f"{label} is absent")
    return selected


def _require_binding(value: object, *, label: str) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"path", "file_sha256", "byte_count"}
        or not isinstance(value.get("path"), str)
        or not value["path"]
        or _SHA256.fullmatch(str(value.get("file_sha256"))) is None
        or type(value.get("byte_count")) is not int
        or int(value["byte_count"]) <= 0
    ):
        raise PosthocJoinAdmissionError(f"{label} binding is malformed")
    _reject_protected(Path(str(value["path"])), label=label)
    return dict(value)


def _file_binding(path: Path) -> dict[str, Any]:
    selected = _absolute_nofollow(path, label="bound file", must_exist=True)
    if not selected.is_file():
        raise PosthocJoinAdmissionError(f"bound file is not regular: {selected}")
    return dict(pilot.file_binding(selected))


def _read_bound_json(binding: object, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = _require_binding(binding, label=label)
    document, actual = pilot.read_bound_json(
        Path(str(expected["path"])),
        expected_sha256=str(expected["file_sha256"]),
        expected_byte_count=int(expected["byte_count"]),
        label=label,
    )
    if not isinstance(document, Mapping) or actual != expected:
        raise PosthocJoinAdmissionError(f"{label} is not an exact JSON object")
    return dict(document), dict(actual)


def _write_exclusive(path: Path, raw: bytes) -> None:
    selected = Path(path)
    if selected.exists() or selected.is_symlink():
        raise FileExistsError(f"refusing to overwrite posthoc output: {selected}")
    selected.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        selected,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o644,
    )
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    _write_exclusive(path, _canonical_json_bytes(value) + b"\n")


def _source_inventory(root: Path) -> dict[str, Any]:
    selected = _absolute_nofollow(root, label="source receipt root", must_exist=True)
    if not selected.is_dir():
        raise PosthocJoinAdmissionError("source receipt root is not a directory")
    entries: list[dict[str, Any]] = []

    def walk(directory: Path) -> None:
        with os.scandir(directory) as iterator:
            children = sorted(iterator, key=lambda item: item.name)
        for child in children:
            if _protected_name(child.name):
                raise PosthocJoinAdmissionError(
                    "source receipt root contains protected material"
                )
            child_path = Path(child.path)
            if child.is_symlink():
                raise PosthocJoinAdmissionError("source receipt tree contains a symlink")
            if child.is_dir(follow_symlinks=False):
                walk(child_path)
                continue
            if not child.is_file(follow_symlinks=False):
                raise PosthocJoinAdmissionError("source receipt tree has a non-file leaf")
            raw = child_path.read_bytes()
            relative = child_path.relative_to(selected).as_posix()
            entries.append({
                "path": relative,
                "byte_count": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            })

    walk(selected)
    entries.sort(key=lambda item: str(item["path"]))
    return {
        "file_count": len(entries),
        "byte_count": sum(int(item["byte_count"]) for item in entries),
        "inventory_sha256": hashlib.sha256(_canonical_json_bytes(entries)).hexdigest(),
    }


def _validate_original_attempt(
    *,
    terminal_binding: Mapping[str, Any],
    physics_binding: Mapping[str, Any],
    physics_check_binding: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    terminal, _ = _read_bound_json(terminal_binding, label="consumed terminal")
    plan, _ = _read_bound_json(plan_binding, label="frozen collection plan")
    check, _ = _read_bound_json(physics_check_binding, label="physics receipt check")
    source_root = _absolute_nofollow(
        Path(str(physics_binding["path"])).parent,
        label="source receipt root",
        must_exist=True,
    )
    phases = terminal.get("phase_receipts")
    if (
        terminal.get("status") != "CONSUMED_TERMINAL_FAILURE"
        or terminal.get("citable_as_scientific_evidence") is not False
        or terminal.get("authorizes_retry_or_resume") is not False
        or terminal.get("authorizes_refill_or_screening") is not False
        or terminal.get("physics_result_binding") != dict(physics_binding)
        or terminal.get("physics_receipt_check_binding") != dict(physics_check_binding)
        or terminal.get("joined_manifest_binding") is not None
        or terminal.get("joined_receipt_check_binding") is not None
        or not isinstance(phases, list)
        or len(phases) != 3
        or phases[1].get("exit_code") != 0
        or phases[2].get("exit_code") != 0
        or terminal.get("failure")
        != "CalibrationSupervisionError: supervised command exited with status 1"
        or plan.get("output_root") != str(source_root)
        or plan.get("expected_counts") != EXPECTED_COUNTS
        or check.get("status") != "PASS"
        or check.get("phase") != "physics_collection"
        or check.get("counts") != EXPECTED_COUNTS
        or check.get("manifest_binding") != dict(physics_binding)
        or check.get("can_freeze_pilot_contract") is not False
    ):
        raise PosthocJoinAdmissionError("consumed source attempt identity changed")
    return terminal, plan, source_root


def derive_documents_v1(
    *,
    terminal_binding: Mapping[str, Any],
    physics_binding: Mapping[str, Any],
    physics_check_binding: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    calibration_receipt_binding: Mapping[str, Any],
    verify_textured_pixels: bool,
) -> DerivedDocumentsV1:
    """Recompute the exact pure joined metadata from immutable receipts."""

    terminal_binding = _require_binding(terminal_binding, label="consumed terminal")
    physics_binding = _require_binding(physics_binding, label="physics result")
    physics_check_binding = _require_binding(
        physics_check_binding, label="physics receipt check"
    )
    plan_binding = _require_binding(plan_binding, label="frozen collection plan")
    calibration_receipt_binding = _require_binding(
        calibration_receipt_binding, label="calibration receipt"
    )
    _terminal, _plan, source_root = _validate_original_attempt(
        terminal_binding=terminal_binding,
        physics_binding=physics_binding,
        physics_check_binding=physics_check_binding,
        plan_binding=plan_binding,
    )
    try:
        collection = joiner.checker.load_bound_collection_receipts(
            Path(str(physics_binding["path"])),
            expected_file_sha256=str(physics_binding["file_sha256"]),
            expected_byte_count=int(physics_binding["byte_count"]),
            verify_textured_pixels=verify_textured_pixels,
        )
    except joiner.checker.PilotReceiptError as exc:
        raise PosthocJoinAdmissionError(str(exc)) from exc
    document = collection.get("document")
    if (
        not isinstance(document, Mapping)
        or document.get("schema") != pilot.PHYSICS_RESULT_SCHEMA
        or document.get("status") != "PHYSICS_COMPLETE"
        or document.get("failure") is not None
        or document.get("expected_counts") != EXPECTED_COUNTS
        or document.get("observed_counts") != EXPECTED_COUNTS
        or document.get("authorizes_retry_or_resume") is not False
        or len(document.get("state_receipt_bindings", ())) != 256
        or len(document.get("render_receipt_bindings", ())) != 32
        or Path(str(physics_binding["path"])).resolve(strict=True).parent
        != source_root
    ):
        raise PosthocJoinAdmissionError("complete physics collection changed")

    receipt, actual_calibration, _raw = calibration.load_bound_calibration_receipt_v1(
        Path(str(calibration_receipt_binding["path"])),
        expected_sha256=str(calibration_receipt_binding["file_sha256"]),
        expected_byte_count=int(calibration_receipt_binding["byte_count"]),
    )
    if actual_calibration != dict(calibration_receipt_binding):
        raise PosthocJoinAdmissionError("calibration receipt binding changed")
    calibration_collection_binding = joiner._inert_binding(  # noqa: SLF001
        receipt["calibration_collection_receipt"],
        label="calibration collection receipt",
    )
    try:
        calibration_collection = joiner.checker.load_bound_collection_receipts(
            Path(str(calibration_collection_binding["path"])),
            expected_file_sha256=str(calibration_collection_binding["file_sha256"]),
            expected_byte_count=int(calibration_collection_binding["byte_count"]),
            verify_textured_pixels=verify_textured_pixels,
        )
    except joiner.checker.PilotReceiptError as exc:
        raise PosthocJoinAdmissionError(str(exc)) from exc
    _, parity_result, parity_terminal, parity_review = joiner._collection_render_lineage(  # noqa: SLF001
        calibration_collection, label="calibration collection"
    )
    rgb_manifest, rows, metadata = joiner.build_joined_documents_v1(
        collection,
        receipt,
        calibration_visual_domain_parity_result_binding=parity_result,
        calibration_visual_domain_parity_terminal_binding=parity_terminal,
        calibration_visual_domain_parity_review_binding=parity_review,
    )
    raw_by_leaf = {
        "rgb_manifest": joiner._canonical_json_bytes(rgb_manifest),  # noqa: SLF001
        "train": b"".join(
            joiner._canonical_json_bytes(row) + b"\n"  # noqa: SLF001
            for row in rows["train"]
        ),
        "eval": b"".join(
            joiner._canonical_json_bytes(row) + b"\n"  # noqa: SLF001
            for row in rows["eval"]
        ),
    }
    for name, expected in EXPECTED_LEAVES.items():
        raw = raw_by_leaf[name]
        if (
            len(raw) != int(expected["byte_count"])
            or hashlib.sha256(raw).hexdigest() != expected["sha256"]
        ):
            raise PosthocJoinAdmissionError(f"derived {name} bytes changed")
    if (
        len(rgb_manifest.get("artifacts", ())) != 3_072
        or len(rows.get("train", ())) != 128
        or len(rows.get("eval", ())) != 128
        or len(metadata.get("scene_ids", {}).get("train", ())) != 16
        or len(metadata.get("scene_ids", {}).get("eval", ())) != 16
    ):
        raise PosthocJoinAdmissionError("derived posthoc counts changed")
    return DerivedDocumentsV1(
        collection=collection,
        calibration_receipt=receipt,
        rgb_manifest=rgb_manifest,
        rows=rows,
        metadata=metadata,
        raw_by_leaf=raw_by_leaf,
    )


def validate_authority_v1(document: object) -> dict[str, Any]:
    required = {
        "schema",
        "status",
        "authority_granted_by_this_document",
        "scientific_claim_authorized",
        "issued_at",
        "authorizer",
        "preregistration_binding",
        "failure_audit_binding",
        "source_commit",
        "source_review_binding",
        "source_bindings",
        "input_bindings",
        "attempt",
        "permissions",
        "expected_outputs",
    }
    if not isinstance(document, Mapping) or set(document) != required:
        raise PosthocJoinAdmissionError("posthoc authority fields changed")
    preregistration = _require_binding(
        document["preregistration_binding"], label="posthoc preregistration"
    )
    expected_preregistration = {
        "path": str(PREREGISTRATION.resolve()),
        "file_sha256": PREREGISTRATION_SHA256,
        "byte_count": PREREGISTRATION_BYTE_COUNT,
    }
    inputs = document.get("input_bindings")
    attempt = document.get("attempt")
    permissions = document.get("permissions")
    if (
        document.get("schema") != AUTHORITY_SCHEMA
        or document.get("status") != AUTHORITY_STATUS
        or document.get("authority_granted_by_this_document") is not True
        or document.get("scientific_claim_authorized") is not False
        or preregistration != expected_preregistration
        or not isinstance(document.get("issued_at"), str)
        or not document["issued_at"]
        or not isinstance(document.get("authorizer"), str)
        or not document["authorizer"]
        or not isinstance(document.get("source_commit"), str)
        or re.fullmatch(r"[0-9a-f]{40}", document["source_commit"]) is None
        or not isinstance(inputs, Mapping)
        or set(inputs) != {
            "consumed_terminal",
            "physics_result",
            "physics_receipt_check",
            "collection_plan",
            "calibration_gate",
            "collection_source_review",
            "collection_execution_authority",
            "calibration_receipt",
        }
        or not isinstance(attempt, Mapping)
        or attempt != {
            "id": "lewm-go2-wm-bounded-branch-posthoc-join-admission-v1",
            "root": str(DEFAULT_OUTPUT_ROOT.resolve()),
            "maximum_attempts": 1,
            "root_creation_consumes_attempt": True,
            "must_be_absent": True,
            "retry": False,
            "resume": False,
            "overwrite": False,
        }
        or permissions != {
            "source_receipt_reads": True,
            "decoded_pixel_verification": True,
            "metadata_only_derivation": True,
            "write_only_fresh_output_root": True,
            "collector_or_renderer": False,
            "physics_or_gpu": False,
            "training_or_checkpoint_access": False,
            "retry_resume_refill_or_overwrite": False,
            "protected_material": False,
            "scientific_verdict": False,
        }
        or document.get("expected_outputs")
        != {
            name: {
                "name": value["name"],
                "byte_count": value["byte_count"],
                "sha256": value["sha256"],
            }
            for name, value in EXPECTED_LEAVES.items()
        }
    ):
        raise PosthocJoinAdmissionError("posthoc authority contract changed")
    normalized_inputs = {
        name: _require_binding(value, label=f"authority input {name}")
        for name, value in inputs.items()
    }
    source_bindings = document.get("source_bindings")
    if not isinstance(source_bindings, list) or not source_bindings:
        raise PosthocJoinAdmissionError("posthoc authority source closure is absent")
    normalized_sources = [
        _require_binding(value, label="posthoc authority source")
        for value in source_bindings
    ]
    if len({str(value["path"]) for value in normalized_sources}) != len(
        normalized_sources
    ):
        raise PosthocJoinAdmissionError("posthoc authority source paths repeat")
    normalized = dict(document)
    normalized["input_bindings"] = normalized_inputs
    normalized["source_bindings"] = normalized_sources
    normalized["preregistration_binding"] = preregistration
    normalized["failure_audit_binding"] = _require_binding(
        document["failure_audit_binding"], label="failure admissibility audit"
    )
    normalized["source_review_binding"] = _require_binding(
        document["source_review_binding"], label="posthoc source review"
    )
    return normalized


def _rehash_authority_inputs(authority: Mapping[str, Any]) -> None:
    for binding in (
        authority["preregistration_binding"],
        authority["failure_audit_binding"],
        authority["source_review_binding"],
        *authority["input_bindings"].values(),
        *authority["source_bindings"],
    ):
        if _file_binding(Path(str(binding["path"]))) != dict(binding):
            raise PosthocJoinAdmissionError("posthoc authority input changed")


def _validate_collection_lineage_inputs(authority: Mapping[str, Any]) -> None:
    inputs = authority["input_bindings"]
    terminal, _ = _read_bound_json(
        inputs["consumed_terminal"], label="consumed terminal lineage"
    )
    physics, _ = _read_bound_json(
        inputs["physics_result"], label="physics result lineage"
    )
    collection_authority, _ = _read_bound_json(
        inputs["collection_execution_authority"],
        label="collection execution authority lineage",
    )
    gate, _ = _read_bound_json(
        inputs["calibration_gate"], label="calibration gate lineage"
    )
    if (
        terminal.get("authority_binding")
        != inputs["collection_execution_authority"]
        or terminal.get("plan_binding") != inputs["collection_plan"]
        or terminal.get("calibration_gate_binding") != inputs["calibration_gate"]
        or physics.get("authority_binding")
        != inputs["collection_execution_authority"]
        or physics.get("plan_binding") != inputs["collection_plan"]
        or collection_authority.get("plan_binding") != inputs["collection_plan"]
        or collection_authority.get("calibration_gate_binding")
        != inputs["calibration_gate"]
        or collection_authority.get("review_binding")
        != inputs["collection_source_review"]
        or gate.get("calibration_receipt_binding")
        != inputs["calibration_receipt"]
    ):
        raise PosthocJoinAdmissionError("collection lineage input changed")


def materialize_v1(
    *,
    authority_path: Path,
    expected_authority_sha256: str,
    expected_authority_byte_count: int,
) -> dict[str, Any]:
    authority_document, authority_binding = pilot.read_bound_json(
        authority_path,
        expected_sha256=expected_authority_sha256,
        expected_byte_count=expected_authority_byte_count,
        label="posthoc join admission authority",
    )
    authority = validate_authority_v1(authority_document)
    _rehash_authority_inputs(authority)
    _validate_collection_lineage_inputs(authority)
    output_root = _absolute_nofollow(
        Path(str(authority["attempt"]["root"])),
        label="posthoc output root",
        must_exist=False,
    )
    if output_root != DEFAULT_OUTPUT_ROOT.resolve() or output_root.exists():
        raise PosthocJoinAdmissionError("posthoc output root is not fresh and exact")
    source_root = Path(str(authority["input_bindings"]["physics_result"]["path"])).parent
    before = _source_inventory(source_root)
    output_root.mkdir(parents=False)
    manifest_binding: dict[str, Any] | None = None
    try:
        derived = derive_documents_v1(
            terminal_binding=authority["input_bindings"]["consumed_terminal"],
            physics_binding=authority["input_bindings"]["physics_result"],
            physics_check_binding=authority["input_bindings"]["physics_receipt_check"],
            plan_binding=authority["input_bindings"]["collection_plan"],
            calibration_receipt_binding=authority["input_bindings"]["calibration_receipt"],
            verify_textured_pixels=True,
        )
        leaf_bindings: dict[str, dict[str, Any]] = {}
        for name, expected in EXPECTED_LEAVES.items():
            leaf_path = output_root / str(expected["name"])
            _write_exclusive(leaf_path, derived.raw_by_leaf[name])
            leaf_bindings[name] = _file_binding(leaf_path)
        after = _source_inventory(source_root)
        if before != after:
            raise PosthocJoinAdmissionError("immutable source receipt tree changed")
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "status": MANIFEST_STATUS,
            "citable_as_scientific_evidence": False,
            "original_attempt_completed_successfully": False,
            "authorizes_retry_or_resume": False,
            "source_receipt_root": str(source_root.resolve()),
            "derived_output_root": str(output_root),
            "authority_binding": authority_binding,
            "preregistration_binding": authority["preregistration_binding"],
            "failure_audit_binding": authority["failure_audit_binding"],
            "source_review_binding": authority["source_review_binding"],
            "input_bindings": authority["input_bindings"],
            "source_bindings": authority["source_bindings"],
            "source_inventory_before": before,
            "source_inventory_after": after,
            "counts": EXPECTED_COUNTS,
            "rgb_artifacts": 3_072,
            "role_scene_counts": {"train": 16, "eval": 16},
            "render_profile": derived.metadata["render_profile"],
            "visual_domain_parity_result_binding": derived.metadata[
                "visual_domain_parity_result_binding"
            ],
            "visual_domain_parity_terminal_binding": derived.metadata[
                "visual_domain_parity_terminal_binding"
            ],
            "visual_domain_parity_review_binding": derived.metadata[
                "visual_domain_parity_review_binding"
            ],
            "calibration_contract": derived.metadata["calibration_contract"],
            "scene_ids": derived.metadata["scene_ids"],
            "action_catalog": derived.metadata["action_catalog"],
            "derived_leaf_bindings": leaf_bindings,
            "derivation": "frozen_pixel_verifier_plus_pure_build_joined_documents_v1",
            "rgb_storage": "immutable_source_receipt_root_only",
            "consumer_compatibility_projection": CONSUMER_COMPATIBILITY_PROJECTION,
        }
        manifest_path = output_root / "manifest.json"
        _write_json_exclusive(manifest_path, manifest)
        manifest_binding = _file_binding(manifest_path)
        _rehash_authority_inputs(authority)
        _validate_collection_lineage_inputs(authority)
        if _source_inventory(source_root) != before:
            raise PosthocJoinAdmissionError("source changed before terminal publication")
        terminal = {
            "schema": TERMINAL_SCHEMA,
            "status": TERMINAL_SUCCESS,
            "citable_as_scientific_evidence": False,
            "scientific_claim_emitted": False,
            "authorizes_retry_or_resume": False,
            "original_terminal_remains_failure": True,
            "authority_binding": authority_binding,
            "manifest_binding": manifest_binding,
            "source_inventory_before": before,
            "source_inventory_after": after,
            "generation_or_rendering_performed": False,
            "independent_review_required": True,
            "failure": None,
        }
        terminal_path = output_root / "terminal.json"
        _write_json_exclusive(terminal_path, terminal)
        terminal_binding = _file_binding(terminal_path)
        return {
            "output_root": str(output_root),
            "manifest_binding": manifest_binding,
            "terminal_binding": terminal_binding,
        }
    except Exception as exc:
        terminal_path = output_root / "terminal.json"
        if not terminal_path.exists() and not terminal_path.is_symlink():
            failure = {
                "schema": TERMINAL_SCHEMA,
                "status": TERMINAL_FAILURE,
                "citable_as_scientific_evidence": False,
                "scientific_claim_emitted": False,
                "authorizes_retry_or_resume": False,
                "original_terminal_remains_failure": True,
                "authority_binding": authority_binding,
                "manifest_binding": manifest_binding,
                "source_inventory_before": before,
                "source_inventory_after": _source_inventory(source_root),
                "generation_or_rendering_performed": False,
                "independent_review_required": True,
                "failure": f"{type(exc).__name__}: {exc}",
            }
            _write_json_exclusive(terminal_path, failure)
        raise


def _load_leaf(path: Path, binding: Mapping[str, Any], *, label: str) -> bytes:
    expected = _require_binding(binding, label=label)
    selected = _absolute_nofollow(path, label=label, must_exist=True)
    if _file_binding(selected) != expected:
        raise PosthocJoinAdmissionError(f"{label} binding changed")
    return selected.read_bytes()


def _consumer_compatible_sync_document(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project the richer receipt audit onto the frozen parser's exact keys."""

    result = copy.deepcopy(dict(value))
    sync = result.get("synchronization_audit")
    components = sync.get("components") if isinstance(sync, Mapping) else None
    if not isinstance(components, Mapping):
        raise PosthocJoinAdmissionError("synchronization components are absent")
    projected: dict[str, Any] = {}
    for name, component in components.items():
        if (
            not isinstance(component, Mapping)
            or set(component)
            != {
                "exact_equal",
                "max_abs_difference",
                "per_lane_max_abs_difference",
                "rms_difference",
                "shape_per_lane",
            }
            or component.get("exact_equal") is not True
            or float(component.get("max_abs_difference", -1.0)) != 0.0
            or float(component.get("rms_difference", -1.0)) != 0.0
            or not isinstance(component.get("per_lane_max_abs_difference"), list)
            or len(component["per_lane_max_abs_difference"]) != 9
            or any(
                float(item) != 0.0
                for item in component["per_lane_max_abs_difference"]
            )
        ):
            raise PosthocJoinAdmissionError(
                "richer synchronization diagnostic is not redundant and exact"
            )
        projected[str(name)] = {
            "exact_equal": True,
            "max_abs_difference": 0.0,
            "shape_per_lane": list(component["shape_per_lane"]),
        }
    sync["components"] = projected
    return result


def load_posthoc_bundle_v1(
    pilot_root: Path,
    *,
    expected_manifest_byte_count: int,
    expected_manifest_sha256: str,
) -> consumer.CounterfactualPilotBundleV1:
    """Load the reviewed split-root metadata without opening an RGB leaf."""

    derived_root = _absolute_nofollow(
        pilot_root, label="posthoc pilot root", must_exist=True
    )
    if derived_root != DEFAULT_OUTPUT_ROOT.resolve() or not derived_root.is_dir():
        raise PosthocJoinAdmissionError("posthoc pilot root changed")
    manifest_path = derived_root / "manifest.json"
    manifest, manifest_binding = pilot.read_bound_json(
        manifest_path,
        expected_sha256=expected_manifest_sha256,
        expected_byte_count=expected_manifest_byte_count,
        label="posthoc manifest",
    )
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("status") != MANIFEST_STATUS
        or manifest.get("citable_as_scientific_evidence") is not False
        or manifest.get("original_attempt_completed_successfully") is not False
        or manifest.get("authorizes_retry_or_resume") is not False
        or manifest.get("derived_output_root") != str(derived_root)
        or manifest.get("counts") != EXPECTED_COUNTS
        or manifest.get("rgb_artifacts") != 3_072
        or manifest.get("role_scene_counts") != {"train": 16, "eval": 16}
        or manifest.get("source_inventory_before")
        != manifest.get("source_inventory_after")
        or manifest.get("consumer_compatibility_projection")
        != CONSUMER_COMPATIBILITY_PROJECTION
    ):
        raise PosthocJoinAdmissionError("posthoc manifest changed")
    inputs = manifest.get("input_bindings")
    leaves = manifest.get("derived_leaf_bindings")
    if not isinstance(inputs, Mapping) or not isinstance(leaves, Mapping):
        raise PosthocJoinAdmissionError("posthoc manifest bindings are absent")
    derived = derive_documents_v1(
        terminal_binding=inputs["consumed_terminal"],
        physics_binding=inputs["physics_result"],
        physics_check_binding=inputs["physics_receipt_check"],
        plan_binding=inputs["collection_plan"],
        calibration_receipt_binding=inputs["calibration_receipt"],
        verify_textured_pixels=False,
    )
    for name, expected in EXPECTED_LEAVES.items():
        binding = _require_binding(leaves.get(name), label=f"derived {name}")
        expected_path = derived_root / str(expected["name"])
        if Path(str(binding["path"])).resolve(strict=True) != expected_path:
            raise PosthocJoinAdmissionError(f"derived {name} path changed")
        raw = _load_leaf(expected_path, binding, label=f"derived {name}")
        if raw != derived.raw_by_leaf[name]:
            raise PosthocJoinAdmissionError(f"derived {name} is not reproducible")
    source_root = _absolute_nofollow(
        Path(str(manifest["source_receipt_root"])),
        label="source receipt root",
        must_exist=True,
    )
    if _source_inventory(source_root) != manifest["source_inventory_before"]:
        raise PosthocJoinAdmissionError("source receipt inventory changed")

    artifacts: dict[str, consumer.RGBArtifactV1] = {}
    for item in derived.rgb_manifest["artifacts"]:
        artifact_id = str(item["artifact_id"])
        if artifact_id in artifacts:
            raise PosthocJoinAdmissionError("posthoc RGB artifact identity repeats")
        artifacts[artifact_id] = consumer.RGBArtifactV1(
            artifact_id=artifact_id,
            frame_identity=str(item["frame_identity"]),
            relative_path=str(item["path"]),
            byte_count=int(item["byte_count"]),
            file_sha256=str(item["file_sha256"]),
            pixel_sha256=str(item["pixel_sha256"]),
            low_information=bool(item["low_information"]),
            low_info_reasons=tuple(str(value) for value in item["low_info_reasons"]),
        )
    _excluded, tolerances = consumer._validate_calibration_contract(  # noqa: SLF001
        derived.metadata["calibration_contract"], textured_v03=True
    )
    requested_blocks = tuple(
        consumer._validate_tape(entry["requested_block"], name="requested action")  # noqa: SLF001
        for entry in derived.metadata["action_catalog"]
    )
    state_documents = {
        str(state["document"]["state"]["state_id"]): (
            _consumer_compatible_sync_document(state["document"])
        )
        for state in derived.collection["states"]
    }
    groups_by_role: dict[str, tuple[consumer.CounterfactualGroupV1, ...]] = {}
    role_bindings: dict[str, Mapping[str, Any]] = {}
    for role in ("train", "eval"):
        groups = tuple(
            consumer._parse_group(  # noqa: SLF001
                _consumer_compatible_sync_document(row),
                role=role,
                artifacts=artifacts,
                tolerances=tolerances,
                requested_blocks=requested_blocks,
                collection_state=state_documents[str(row["state_id"])],
            )
            for row in derived.rows[role]
        )
        if len(groups) != 128:
            raise PosthocJoinAdmissionError(f"posthoc {role} group count changed")
        groups_by_role[role] = groups
        role_bindings[role] = MappingProxyType(dict(leaves[role]))
    return consumer.CounterfactualPilotBundleV1(
        root=source_root,
        manifest_binding=MappingProxyType(dict(manifest_binding)),
        manifest=MappingProxyType(dict(manifest)),
        rgb_manifest_binding=MappingProxyType(dict(leaves["rgb_manifest"])),
        artifacts=MappingProxyType(artifacts),
        groups_by_role=MappingProxyType(groups_by_role),
        role_bindings=MappingProxyType(role_bindings),
        calibration_receipt=MappingProxyType(dict(derived.calibration_receipt)),
        calibration_tolerances=MappingProxyType(dict(tolerances)),
        access_audit=MappingProxyType({
            "manifest_open_count": 1,
            "role_index_open_count": 2,
            "rgb_manifest_open_count": 1,
            "state_receipt_open_count": 256,
            "render_receipt_open_count": 32,
            "rgb_leaf_open_count": 0,
            "split_root_posthoc_admission": True,
        }),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", required=True, type=Path)
    parser.add_argument("--expected-authority-sha256", required=True)
    parser.add_argument("--expected-authority-byte-count", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = materialize_v1(
        authority_path=args.authority,
        expected_authority_sha256=args.expected_authority_sha256,
        expected_authority_byte_count=args.expected_authority_byte_count,
    )
    print(json.dumps({"status": TERMINAL_SUCCESS, **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORITY_SCHEMA",
    "AUTHORITY_STATUS",
    "DEFAULT_OUTPUT_ROOT",
    "MANIFEST_SCHEMA",
    "MANIFEST_STATUS",
    "PosthocJoinAdmissionError",
    "TERMINAL_SCHEMA",
    "TERMINAL_SUCCESS",
    "derive_documents_v1",
    "load_posthoc_bundle_v1",
    "materialize_v1",
    "validate_authority_v1",
]
