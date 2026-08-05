#!/usr/bin/env python3
"""Independently recompute and finalize dynamic cell-square evidence."""
from __future__ import annotations

# Deliberately stdlib-only until the reviewed allowlist has been hashed.
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
import types
from typing import Any, Sequence
import zipfile


ROOT = Path(__file__).resolve().parents[1]
BINDING_PATH = "docs/lewm_go2_n32_dynamic_cell_square_geometry_binding_2026-07-11.md"
BINDING_SHA256 = "211043ee3c3200d1fc93febbae73059341aea19560c83f53f3b3bb231bf06e66"
MACHINE_PATH = (
    "docs/lewm_go2_n32_dynamic_cell_square_geometry_implementation_manifest_2026-07-11.json"
)
CANDIDATE_PATH = (
    ".generated/go2_dynamic_cell_square_projection_diagnostic/v1/candidate.json"
)
RESULT_PATH = ".generated/go2_dynamic_cell_square_projection_diagnostic/v1/result.json"
FAILURE_PATH = (
    ".generated/go2_dynamic_cell_square_projection_diagnostic/v1/failure_diagnostic.json"
)
PREDECESSOR_RESULT_PATH = (
    ".generated/go2_n32_camera_frustum_observability_audit/v2/result.json"
)
PREDECESSOR_REPORT_PATH = (
    "docs/lewm_go2_n32_camera_frustum_observability_audit_v2_result_2026-07-11.md"
)
PREDECESSOR_REPORT_SHA256 = (
    "8bfb4c9a8b69f67b3b9e4d6e3b21e9ff89ecaff89a2bab3eb83d759ca4fe6d22"
)
DYNAMIC_GEOMETRY_SHA256 = (
    "ce2bb0d38ed1436635cdd1468ba1dfe1a935fdafdd6dda5adcf37b97a32a74bf"
)
PREDECESSOR_FILE_SHA256 = (
    "7725ecddf2fa77bb762733fd35df2efd2fb60d4f9aa8ab6fdf2bee660522909e"
)
PREDECESSOR_CONTENT_SHA256 = (
    "11420607d2c4f8e79af9214d43bbc6259669ee84c9ccc0aaefd4167cc1d809a1"
)
MACHINE_SCHEMA = "lewm_go2_dynamic_cell_square_projection_implementation_manifest_v1"
CANDIDATE_SCHEMA = "lewm_go2_dynamic_cell_square_projection_candidate_v1"
FINAL_SCHEMA = "lewm_go2_dynamic_cell_square_projection_diagnostic_v1"
LEDGER_SCHEMA = "lewm_go2_dynamic_projection_access_ledger_v1"
SELF_TEMPLATE_ENTRY = {
    "path": MACHINE_PATH,
    "role": "machine_manifest",
    "sha256_source": "implementation_manifest_sha256_argument",
}
CANDIDATE_TEMPLATE_ENTRY = {
    "path": CANDIDATE_PATH,
    "role": "candidate",
    "sha256_source": "candidate_sha256_argument",
}
FAMILIES = (
    "open_obstacle_field",
    "rough_local_dynamics",
    "small_enclosed_maze",
    "medium_enclosed_maze",
    "large_enclosed_maze",
)
SIDES = ("current", "next")
CLASSES = ("unknown", "free", "occupied")
KNOWN_CLASSES = ("free", "occupied")
CLASS_IDS = {"unknown": 0, "free": 1, "occupied": 2}
FORBIDDEN_ROLES = (
    "g2",
    "heldout",
    "image",
    "model_output",
    "physical_nontrain",
    "runtime_result",
    "sealed",
    "selection_calibration",
    "source_geometry",
)
DENIED_REASONS = (
    "path_alias_or_escape",
    "symlink_component",
    "outside_repository",
    "unallowlisted",
    "forbidden_role",
    "modality_mismatch",
    "hash_mismatch",
)
FRAME_FIELDS = (
    "family",
    "scene_id",
    "global_row",
    "side",
    "image_sha256",
    "label_shard_sha256",
    "label_row",
)
EXPECTED_LABEL_MANIFEST_SHA256 = (
    "998ce5a768029c23c931fbbec730c1fe31b9ed1fe155494fc68f34a0c23d3d1b"
)
EXPECTED_TARGET_BYTES = 1_310_720
TARGET_ROW_BYTES = 64 * 64
EXPECTED_TARGET_ROWS = EXPECTED_TARGET_BYTES // TARGET_ROW_BYTES
EXPECTED_TARGET_SHA256 = (
    "6952c1f9604da1d9fd4c94a3f33deb142451836609b7059970ff6c459737ce05"
)
EXPECTED_CLASS_TOTALS = {
    "unknown": 1_181_699,
    "free": 118_793,
    "occupied": 10_228,
    "all": 1_310_720,
}
EXPECTED_KNOWN_TOTAL = 129_021
EXPECTED_CENTER_COUNT = 1_990
EXPECTED_CENTER_HASH = (
    "026d7654864bea7ae0545bd6448f6def64519a3bedcbc7ea747e7b4b95f82b3a"
)
EXPECTED_CENTER_FREE = 118_792
EXPECTED_CENTER_OCCUPIED = 9_856
EXPECTED_CENTER_VIOLATIONS = 373
EXPECTED_CENTER_IDENTITIES_HASH = (
    "f85a9ece8f4a34fe0f175de900934780a750d076f70a7e672be8337cffb64bcc"
)
EXPECTED_SQUARE_COUNT = 2_062
EXPECTED_SQUARE_HASH = (
    "4ebbafb6d4dd5fb13b96df978abfa7b81bc2f879b2ba6dec2fcda38dec54e60b"
)
EXPECTED_STATIC_IDENTITIES_HASH = (
    "c574f35890ef68114fb36ebf701eec7552262d03c49cf4d1c07b47740fc505f0"
)
EMPTY_HASH = "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
REGISTERED_MEMBERS = {
    "current_labels.npy",
    "current_supervision_mask.npy",
    "next_labels.npy",
    "next_supervision_mask.npy",
    "current_observed_mask.npy",
    "next_observed_mask.npy",
    "relative_se2_current_frame.npy",
    "primitive.npy",
    "current_image_path.npy",
    "next_image_path.npy",
    "current_image_sha256.npy",
    "next_image_sha256.npy",
}
CANDIDATE_KEYS = {
    "schema",
    "created_at_utc",
    "execution_binding",
    "implementation_manifests",
    "inputs",
    "source_map",
    "scope",
    "preparation_access_ledger",
    "runner_access_ledger",
    "label_reconciliation",
    "support",
    "family_class_rows",
    "frame_summary_records_sha256",
    "scientific_core_sha256",
    "gates",
    "content_sha256",
}
FINAL_KEYS = {
    "schema",
    "created_at_utc",
    "execution_binding",
    "implementation_manifests",
    "candidate",
    "inputs",
    "source_map",
    "scope",
    "preparation_access_ledger",
    "runner_access_ledger",
    "finalizer_access_ledger",
    "label_reconciliation",
    "support",
    "family_class_rows",
    "frame_summary_records_sha256",
    "scientific_core_sha256",
    "independent_recomputation",
    "gates",
    "content_sha256",
}
GATE_KEYS = (
    "binding_and_source_hashes_pass",
    "predecessor_authority_pass",
    "label_manifest_and_bytes_pass",
    "label_count_reconciliation_pass",
    "level_center_parity_pass",
    "level_cell_square_frozen_pass",
    "static_all_known_scored_pass",
    "dynamic_all_known_scored_pass",
    "dynamic_zero_known_unsupported_pass",
    "access_reconciliation_pass",
    "independent_recomputation_pass",
    "all_passed",
)
LEDGER_KEYS = {
    "schema", "phase", "authorized_read_paths", "authorized_read_path_set_sha256",
    "authorized_write_paths", "authorized_write_path_set_sha256", "role_byte_open_counts",
    "label_shard_pre_hash_byte_opens", "label_shard_post_hash_byte_opens",
    "label_shard_npz_parses", "array_decompression_counts", "selected_label_rows_read",
    "unselected_rows_scored", "unselected_rows_retained", "metadata_only_shard_stats",
    "denied_attempt_records", "denied_reason_counts", "unexpected_path_attempts",
    "forbidden_role_open_counts", "all_counts_reconcile",
}
SOURCE_MAP_CONTRACT = (
    ("dynamic_geometry", "lewm/benchmarks/go2_dynamic_cell_square_projection.py"),
    ("diagnostic_core", "lewm/benchmarks/go2_dynamic_cell_square_projection_diagnostic.py"),
    ("preparation", "scripts/prepare_go2_dynamic_cell_square_projection.py"),
    ("runner", "scripts/diagnose_go2_dynamic_cell_square_projection.py"),
    ("finalizer", "scripts/finalize_go2_dynamic_cell_square_projection.py"),
    ("geometry_test", "lewm/tests/test_go2_dynamic_cell_square_projection.py"),
    ("diagnostic_test", "lewm/tests/test_go2_dynamic_cell_square_projection_diagnostic.py"),
    ("preparation_test", "lewm/tests/test_prepare_go2_dynamic_cell_square_projection.py"),
    ("finalizer_test", "lewm/tests/test_finalize_go2_dynamic_cell_square_projection.py"),
)
EXPECTED_LABEL_ACCESS = {
    "label_shard_pre_hash_byte_opens": 20,
    "label_shard_post_hash_byte_opens": 20,
    "label_shard_npz_parses": 20,
    "array_decompression_counts": {"current_labels": 20, "next_labels": 20},
    "selected_label_rows_read": 320,
    "metadata_only_shard_stats": 0,
}


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def exact_equal(first: object, second: object) -> bool:
    """Compare JSON values without Python bool/int/float equality coercions."""

    return canonical_bytes(first) == canonical_bytes(second)


def validate_utc_timestamp(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be an exact string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} is not ISO-8601") from exc
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() is None
        or parsed.utcoffset().total_seconds() != 0
        or parsed.isoformat() != value
        or not value.endswith("+00:00")
    ):
        raise ValueError(f"{name} is not canonical UTC")
    return value


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def is_sha(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def strict_json(raw: bytes, *, name: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if type(key) is not str or key in result:
                raise ValueError(f"{name} duplicate/non-string key")
            result[key] = value
        return result

    def constant(value: str) -> None:
        raise ValueError(f"{name} nonfinite constant {value}")

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} invalid UTF-8 JSON") from exc
    if type(value) is not dict:
        raise ValueError(f"{name} root is not an exact object")
    return value


def validate_content(record: dict[str, Any], *, name: str) -> str:
    declared = record.get("content_sha256")
    if not is_sha(declared):
        raise ValueError(f"{name} content hash malformed")
    core = dict(record)
    del core["content_sha256"]
    if canonical_hash(core) != declared:
        raise ValueError(f"{name} content hash mismatch")
    return str(declared)


def instantiate_template(
    template: object,
    *,
    manifest_sha256: str,
    candidate_sha256: str,
    manifest_verified: bool,
    candidate_verified: bool,
) -> list[dict[str, str]]:
    if not manifest_verified or not candidate_verified:
        raise ValueError("bootstrap bytes must be verified before substitution")
    if not is_sha(manifest_sha256) or not is_sha(candidate_sha256) or type(template) is not list:
        raise ValueError("bootstrap hashes/template malformed")
    placeholders = [
        item for item in template if type(item) is dict and "sha256_source" in item
    ]
    if len(placeholders) != 2 or SELF_TEMPLATE_ENTRY not in placeholders or CANDIDATE_TEMPLATE_ENTRY not in placeholders:
        raise ValueError("finalizer placeholders are not the exact registered pair")
    result: list[dict[str, str]] = []
    for item in template:
        if item == SELF_TEMPLATE_ENTRY:
            result.append({"path": MACHINE_PATH, "role": "machine_manifest", "sha256": manifest_sha256})
        elif item == CANDIDATE_TEMPLATE_ENTRY:
            result.append({"path": CANDIDATE_PATH, "role": "candidate", "sha256": candidate_sha256})
        else:
            if type(item) is not dict or set(item) != {"path", "role", "sha256"} or not is_sha(item["sha256"]):
                raise ValueError("non-placeholder finalizer allowlist entry malformed")
            result.append(dict(item))
    return result


def instantiate_runner_template(
    template: object, *, manifest_sha256: str
) -> list[dict[str, str]]:
    if type(template) is not list or not is_sha(manifest_sha256):
        raise ValueError("runner template/bootstrap hash malformed")
    placeholders = [
        item for item in template if type(item) is dict and "sha256_source" in item
    ]
    if placeholders != [SELF_TEMPLATE_ENTRY]:
        raise ValueError("runner template self placeholder is not exact")
    result: list[dict[str, str]] = []
    for item in template:
        if item == SELF_TEMPLATE_ENTRY:
            result.append(
                {"path": MACHINE_PATH, "role": "machine_manifest", "sha256": manifest_sha256}
            )
        else:
            if type(item) is not dict or set(item) != {"path", "role", "sha256"} or not is_sha(item["sha256"]):
                raise ValueError("runner non-placeholder allowlist entry malformed")
            result.append(dict(item))
    return result


def _validate_source_map(value: object) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    if type(value) is not dict or set(value) != {"entries", "entry_count", "source_map_sha256"}:
        raise ValueError("machine source map schema mismatch")
    entries = value["entries"]
    if (
        type(entries) is not list
        or len(entries) != 9
        or type(value["entry_count"]) is not int
        or value["entry_count"] != 9
    ):
        raise ValueError("machine source map count mismatch")
    if canonical_hash(entries) != value["source_map_sha256"]:
        raise ValueError("machine source map hash mismatch")
    for item, (role, path) in zip(entries, SOURCE_MAP_CONTRACT):
        if type(item) is not dict or set(item) != {"path", "role", "sha256"} or item["role"] != role or item["path"] != path or not is_sha(item["sha256"]):
            raise ValueError("machine source map role/path/hash contract mismatch")
    by_role = {item["role"]: item for item in entries}
    if len(by_role) != 9 or by_role["dynamic_geometry"]["sha256"] != DYNAMIC_GEOMETRY_SHA256:
        raise ValueError("machine source map duplicates or geometry hash mismatch")
    return value, by_role


def validate_machine_manifest(
    manifest: object,
    *,
    manifest_sha256: str,
    candidate_sha256: str,
) -> tuple[
    dict[str, Any],
    list[dict[str, str]],
    dict[str, Any],
    list[dict[str, str]],
]:
    if type(manifest) is not dict or set(manifest) != {
        "schema", "created_at_utc", "execution_binding", "human_manifest", "inputs",
        "source_map", "phase_contracts", "preparation_access_ledger",
        "output_absence", "runtime_environment", "content_sha256",
    } or manifest.get("schema") != MACHINE_SCHEMA:
        raise ValueError("machine manifest schema/key set mismatch")
    validate_content(manifest, name="machine manifest")
    validate_utc_timestamp(manifest["created_at_utc"], name="machine timestamp")
    if not exact_equal(
        manifest["execution_binding"],
        {"path": BINDING_PATH, "file_sha256": BINDING_SHA256},
    ):
        raise ValueError("machine manifest execution binding mismatch")
    human = manifest["human_manifest"]
    if type(human) is not dict or set(human) != {"path", "file_sha256"} or human["path"] != (
        "docs/lewm_go2_n32_dynamic_cell_square_geometry_implementation_manifest_2026-07-11.md"
    ) or not is_sha(human["file_sha256"]):
        raise ValueError("machine human-manifest commitment mismatch")
    inputs = manifest["inputs"]
    if type(inputs) is not dict or set(inputs) != {"predecessor_report", "predecessor_result", "label_shard_manifest", "selected_targets"}:
        raise ValueError("machine input key set mismatch")
    if not exact_equal(inputs["predecessor_report"], {"path": PREDECESSOR_REPORT_PATH, "file_sha256": PREDECESSOR_REPORT_SHA256}):
        raise ValueError("machine predecessor-report commitment mismatch")
    if not exact_equal(inputs["predecessor_result"], {"path": PREDECESSOR_RESULT_PATH, "file_sha256": PREDECESSOR_FILE_SHA256, "content_sha256": PREDECESSOR_CONTENT_SHA256}):
        raise ValueError("machine predecessor-result commitment mismatch")
    if not exact_equal(inputs["label_shard_manifest"], {"entry_count": 20, "manifest_sha256": EXPECTED_LABEL_MANIFEST_SHA256}) or not exact_equal(inputs["selected_targets"], {"frame_count": 320, "byte_count": EXPECTED_TARGET_BYTES, "sha256": EXPECTED_TARGET_SHA256}):
        raise ValueError("machine label commitments mismatch")
    source_map, source_by_role = _validate_source_map(manifest["source_map"])
    del source_map
    expected_runtime = {
        "python_implementation": sys.implementation.name,
        "python_version": list(sys.version_info[:3]),
        "numpy_version": importlib.metadata.version("numpy"),
    }
    if not exact_equal(manifest["runtime_environment"], expected_runtime):
        raise ValueError("machine runtime environment does not match execution")
    expected_absence = {
        "paths": [
            {"path": str(ROOT / relative), "exists": False}
            for relative in (CANDIDATE_PATH, RESULT_PATH, FAILURE_PATH)
        ],
        "all_absent": True,
    }
    if not exact_equal(manifest["output_absence"], expected_absence):
        raise ValueError("machine output-absence proof is not exact")
    phases = manifest["phase_contracts"]
    if type(phases) is not dict or set(phases) != {"runner", "finalizer"}:
        raise ValueError("machine phase contract key set mismatch")
    phase = phases["finalizer"]
    phase_keys = {
        "authorized_read_path_template", "authorized_read_path_template_sha256",
        "authorized_write_paths", "authorized_write_path_set_sha256", "expected_roles",
        "expected_role_byte_open_counts", "expected_label_access",
    }
    if type(phase) is not dict or set(phase) != phase_keys:
        raise ValueError("finalizer phase contract key set mismatch")
    template = phase["authorized_read_path_template"]
    if canonical_hash(template) != phase["authorized_read_path_template_sha256"]:
        raise ValueError("finalizer template hash mismatch")
    actual = instantiate_template(
        template,
        manifest_sha256=manifest_sha256,
        candidate_sha256=candidate_sha256,
        manifest_verified=True,
        candidate_verified=True,
    )
    if actual != sorted(actual, key=lambda item: (item["path"], item["role"])):
        raise ValueError("instantiated finalizer template not sorted")
    expected_nonshard = [
        {"path": BINDING_PATH, "role": "binding", "sha256": BINDING_SHA256},
        {"path": human["path"], "role": "human_manifest", "sha256": human["file_sha256"]},
        {"path": MACHINE_PATH, "role": "machine_manifest", "sha256": manifest_sha256},
        {"path": CANDIDATE_PATH, "role": "candidate", "sha256": candidate_sha256},
        {"path": PREDECESSOR_RESULT_PATH, "role": "predecessor_result", "sha256": PREDECESSOR_FILE_SHA256},
        *[
            dict(source_by_role[role])
            for role in (
                "dynamic_geometry", "finalizer", "geometry_test", "diagnostic_test",
                "preparation_test", "finalizer_test",
            )
        ],
    ]
    expected_nonshard.sort(key=lambda item: (item["path"], item["role"]))
    actual_nonshard = [item for item in actual if item["role"] != "label_shard"]
    if actual_nonshard != expected_nonshard or len([item for item in actual if item["role"] == "label_shard"]) != 20:
        raise ValueError("finalizer allowlist role/path graph is not exact")
    roles = sorted({item["role"] for item in actual})
    if not exact_equal(phase["expected_roles"], roles):
        raise ValueError("finalizer expected roles are not exact")
    expected_counts = {role: 1 for role in roles if role != "label_shard"}
    expected_counts["label_shard"] = 40
    if not exact_equal(phase["expected_role_byte_open_counts"], expected_counts):
        raise ValueError("finalizer role-open counts are not exact")
    if not exact_equal(phase["expected_label_access"], EXPECTED_LABEL_ACCESS):
        raise ValueError("finalizer numeric label-access contract mismatch")
    expected_writes = sorted(
        [
            {"path": RESULT_PATH, "role": "finalizer_output", "sha256": None},
            {"path": FAILURE_PATH, "role": "failure_diagnostic_output", "sha256": None},
        ],
        key=lambda item: (item["path"], item["role"]),
    )
    if not exact_equal(phase["authorized_write_paths"], expected_writes) or phase["authorized_write_path_set_sha256"] != canonical_hash(expected_writes):
        raise ValueError("finalizer write allowlist is not exact")

    runner_phase = phases["runner"]
    if type(runner_phase) is not dict or set(runner_phase) != phase_keys:
        raise ValueError("runner phase contract key set mismatch")
    runner_template = runner_phase["authorized_read_path_template"]
    if canonical_hash(runner_template) != runner_phase["authorized_read_path_template_sha256"]:
        raise ValueError("runner template hash mismatch")
    runner_actual = instantiate_runner_template(
        runner_template, manifest_sha256=manifest_sha256
    )
    if runner_actual != sorted(runner_actual, key=lambda item: (item["path"], item["role"])):
        raise ValueError("instantiated runner template not sorted")
    runner_expected_nonshard = [
        {"path": BINDING_PATH, "role": "binding", "sha256": BINDING_SHA256},
        {"path": human["path"], "role": "human_manifest", "sha256": human["file_sha256"]},
        {"path": MACHINE_PATH, "role": "machine_manifest", "sha256": manifest_sha256},
        {"path": PREDECESSOR_RESULT_PATH, "role": "predecessor_result", "sha256": PREDECESSOR_FILE_SHA256},
        *[
            dict(source_by_role[role])
            for role in (
                "dynamic_geometry", "diagnostic_core", "runner", "geometry_test",
                "diagnostic_test", "preparation_test", "finalizer_test",
            )
        ],
    ]
    runner_expected_nonshard.sort(key=lambda item: (item["path"], item["role"]))
    if [item for item in runner_actual if item["role"] != "label_shard"] != runner_expected_nonshard or len(
        [item for item in runner_actual if item["role"] == "label_shard"]
    ) != 20:
        raise ValueError("runner allowlist role/path graph is not exact")
    runner_roles = sorted({item["role"] for item in runner_actual})
    runner_counts = {
        role: (40 if role == "label_shard" else 1) for role in runner_roles
    }
    runner_writes = [
        {"path": CANDIDATE_PATH, "role": "runner_output", "sha256": None}
    ]
    if (
        not exact_equal(runner_phase["expected_roles"], runner_roles)
        or not exact_equal(runner_phase["expected_role_byte_open_counts"], runner_counts)
        or not exact_equal(runner_phase["expected_label_access"], EXPECTED_LABEL_ACCESS)
        or not exact_equal(runner_phase["authorized_write_paths"], runner_writes)
        or runner_phase["authorized_write_path_set_sha256"] != canonical_hash(runner_writes)
    ):
        raise ValueError("runner phase numeric/write contract is not exact")
    validate_exact_preparation_ledger(
        manifest["preparation_access_ledger"],
        source_by_role=source_by_role,
        human_manifest=human,
        label_records=[item for item in actual if item["role"] == "label_shard"],
    )
    return phase, actual, runner_phase, runner_actual


def safe_path(text: str) -> Path:
    if (
        type(text) is not str
        or not text
        or "\\" in text
        or os.path.normpath(text) != text
        or Path(text).as_posix() != text
    ):
        raise PermissionError("path_alias_or_escape")
    lexical = Path(text)
    if any(part in ("", ".", "..") for part in lexical.parts):
        raise PermissionError("path_alias_or_escape")
    path = lexical if lexical.is_absolute() else ROOT / lexical
    if os.path.normpath(str(path)) != str(path):
        raise PermissionError("path_alias_or_escape")
    try:
        relative = path.relative_to(ROOT)
    except ValueError as exc:
        raise PermissionError("outside_repository") from exc
    current = ROOT
    for part in relative.parts:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            break
        if stat.S_ISLNK(mode):
            raise PermissionError("symlink_component")
    return path


def anchor_records(records: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    actual: list[dict[str, str]] = []
    lookup: dict[str, dict[str, str]] = {}
    for item in records:
        path = str(safe_path(item["path"]))
        if path in lookup:
            raise ValueError("finalizer allowlist path duplicated")
        record = {"path": path, "role": item["role"], "sha256": item["sha256"]}
        actual.append(record)
        lookup[path] = record
    actual.sort(key=lambda item: (item["path"], item["role"]))
    return actual, lookup


def new_ledger(
    phase: dict[str, Any], records: list[dict[str, str]]
) -> dict[str, Any]:
    writes = [
        {"path": str(safe_path(RESULT_PATH)), "role": "finalizer_output", "sha256": None},
        {"path": str(safe_path(FAILURE_PATH)), "role": "failure_diagnostic_output", "sha256": None},
    ]
    writes.sort(key=lambda item: (item["path"], item["role"]))
    return {
        "schema": LEDGER_SCHEMA,
        "phase": "finalizer",
        "authorized_read_paths": records,
        "authorized_read_path_set_sha256": canonical_hash(records),
        "authorized_write_paths": writes,
        "authorized_write_path_set_sha256": canonical_hash(writes),
        "role_byte_open_counts": {role: 0 for role in phase["expected_roles"]},
        "label_shard_pre_hash_byte_opens": 0,
        "label_shard_post_hash_byte_opens": 0,
        "label_shard_npz_parses": 0,
        "array_decompression_counts": {"current_labels": 0, "next_labels": 0},
        "selected_label_rows_read": 0,
        "unselected_rows_scored": 0,
        "unselected_rows_retained": 0,
        "metadata_only_shard_stats": 0,
        "denied_attempt_records": [],
        "denied_reason_counts": {reason: 0 for reason in DENIED_REASONS},
        "unexpected_path_attempts": 0,
        "forbidden_role_open_counts": {role: 0 for role in FORBIDDEN_ROLES},
        "all_counts_reconcile": False,
    }


def validate_phase_ledger(value: object, *, phase: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != LEDGER_KEYS:
        raise ValueError(f"{phase} ledger key set mismatch")
    if value["schema"] != LEDGER_SCHEMA or value["phase"] != phase:
        raise ValueError(f"{phase} ledger schema/phase mismatch")
    reads, writes = value["authorized_read_paths"], value["authorized_write_paths"]
    if type(reads) is not list or type(writes) is not list:
        raise ValueError(f"{phase} ledger allowlists malformed")
    if reads != sorted(reads, key=lambda item: (item["path"], item["role"])) or writes != sorted(writes, key=lambda item: (item["path"], item["role"])):
        raise ValueError(f"{phase} ledger allowlists not sorted")
    if canonical_hash(reads) != value["authorized_read_path_set_sha256"] or canonical_hash(writes) != value["authorized_write_path_set_sha256"]:
        raise ValueError(f"{phase} ledger allowlist hash mismatch")
    for item in reads:
        if type(item) is not dict or set(item) != {"path", "role", "sha256"} or not is_sha(item["sha256"]):
            raise ValueError(f"{phase} ledger read record malformed")
    for item in writes:
        if type(item) is not dict or set(item) != {"path", "role", "sha256"} or item["sha256"] is not None:
            raise ValueError(f"{phase} ledger write record malformed")
    label_reads = [item for item in reads if item["role"] == "label_shard"]
    if len(label_reads) != 20:
        raise ValueError(f"{phase} ledger label allowlist count mismatch")
    roles = {item["role"] for item in reads}
    if type(value["role_byte_open_counts"]) is not dict or set(value["role_byte_open_counts"]) != roles:
        raise ValueError(f"{phase} ledger role map mismatch")
    expected_role_counts = {role: 1 for role in roles if role != "label_shard"}
    expected_role_counts["label_shard"] = 0 if phase == "preparation" else 40
    if not exact_equal(value["role_byte_open_counts"], expected_role_counts):
        raise ValueError(f"{phase} ledger role counts mismatch")
    if phase == "preparation":
        expected_numeric = (0, 0, 0, 0, 20)
        expected_arrays = {}
        expected_outputs = [(MACHINE_PATH, "machine_manifest_output")]
    elif phase == "runner":
        expected_numeric = (20, 20, 20, 320, 0)
        expected_arrays = {"current_labels": 20, "next_labels": 20}
        expected_outputs = [(CANDIDATE_PATH, "runner_output")]
    elif phase == "finalizer":
        expected_numeric = (20, 20, 20, 320, 0)
        expected_arrays = {"current_labels": 20, "next_labels": 20}
        expected_outputs = [
            (RESULT_PATH, "finalizer_output"),
            (FAILURE_PATH, "failure_diagnostic_output"),
        ]
    else:
        raise ValueError("unregistered ledger phase")
    actual_numeric = (
        value["label_shard_pre_hash_byte_opens"],
        value["label_shard_post_hash_byte_opens"],
        value["label_shard_npz_parses"],
        value["selected_label_rows_read"],
        value["metadata_only_shard_stats"],
    )
    if not exact_equal(actual_numeric, expected_numeric) or not exact_equal(
        value["array_decompression_counts"], expected_arrays
    ):
        raise ValueError(f"{phase} ledger numeric label access mismatch")
    expected_write_records = sorted(
        [
            {"path": str(safe_path(path)), "role": role, "sha256": None}
            for path, role in expected_outputs
        ],
        key=lambda item: (item["path"], item["role"]),
    )
    if not exact_equal(writes, expected_write_records):
        raise ValueError(f"{phase} ledger write graph mismatch")
    if (
        type(value["denied_attempt_records"]) is not list
        or value["denied_attempt_records"]
        or type(value["denied_reason_counts"]) is not dict
        or set(value["denied_reason_counts"]) != set(DENIED_REASONS)
        or any(type(count) is not int or count != 0 for count in value["denied_reason_counts"].values())
        or type(value["unexpected_path_attempts"]) is not int
        or value["unexpected_path_attempts"] != 0
        or type(value["forbidden_role_open_counts"]) is not dict
        or set(value["forbidden_role_open_counts"]) != set(FORBIDDEN_ROLES)
        or any(type(count) is not int or count != 0 for count in value["forbidden_role_open_counts"].values())
        or type(value["unselected_rows_scored"]) is not int
        or value["unselected_rows_scored"] != 0
        or type(value["unselected_rows_retained"]) is not int
        or value["unselected_rows_retained"] != 0
        or type(value["all_counts_reconcile"]) is not bool
        or value["all_counts_reconcile"] is not True
    ):
        raise ValueError(f"{phase} ledger denied/forbidden reconciliation failed")
    return value


def validate_exact_preparation_ledger(
    value: object,
    *,
    source_by_role: dict[str, dict[str, str]],
    human_manifest: dict[str, str],
    label_records: list[dict[str, str]],
) -> dict[str, Any]:
    ledger = validate_phase_ledger(value, phase="preparation")

    def record(path: str, role: str, digest: str) -> dict[str, str]:
        return {"path": str(safe_path(path)), "role": role, "sha256": digest}

    expected = [
        record(BINDING_PATH, "binding", BINDING_SHA256),
        record(
            PREDECESSOR_REPORT_PATH,
            "predecessor_report",
            PREDECESSOR_REPORT_SHA256,
        ),
        record(
            PREDECESSOR_RESULT_PATH,
            "predecessor_result",
            PREDECESSOR_FILE_SHA256,
        ),
        record(
            human_manifest["path"],
            "human_manifest",
            human_manifest["file_sha256"],
        ),
        *[
            record(item["path"], role, item["sha256"])
            for role, item in (
                (role, source_by_role[role]) for role, _path in SOURCE_MAP_CONTRACT
            )
        ],
        *[
            record(item["path"], "label_shard", item["sha256"])
            for item in label_records
        ],
    ]
    expected.sort(key=lambda item: (item["path"], item["role"]))
    if not exact_equal(ledger["authorized_read_paths"], expected):
        raise ValueError("preparation ledger read graph is not exact")
    if ledger["authorized_read_path_set_sha256"] != canonical_hash(expected):
        raise ValueError("preparation ledger read graph hash mismatch")
    return ledger


def primary_denial_reason(reasons: Sequence[str]) -> str:
    if type(reasons) not in (list, tuple) or not reasons:
        raise ValueError("denial reasons must be a nonempty exact sequence")
    if any(type(reason) is not str or reason not in DENIED_REASONS for reason in reasons):
        raise ValueError("denial reason is unregistered")
    for reason in DENIED_REASONS:
        if reason in reasons:
            return reason
    raise AssertionError("registered denial sequence has no primary reason")


def record_denial(
    ledger: dict[str, Any],
    *,
    requested_role: str,
    declared_role: str,
    lexical_path: str,
    resolved_path: str | None,
    reasons: Sequence[str],
    modality: str = "bytes",
) -> str:
    primary = primary_denial_reason(reasons)
    ledger["denied_attempt_records"].append({
        "requested_role": requested_role,
        "declared_role": declared_role,
        "modality": modality,
        "lexical_path": lexical_path,
        "resolved_path": resolved_path,
        "primary_reason": primary,
    })
    ledger["denied_reason_counts"][primary] += 1
    ledger["unexpected_path_attempts"] += 1
    return primary


def read_allowed(
    path: Path,
    *,
    role: str,
    allowlist: dict[str, dict[str, str]],
    ledger: dict[str, Any],
) -> bytes:
    lexical = str(path)
    try:
        authorized_path = safe_path(lexical)
        absolute = str(authorized_path)
    except PermissionError as exc:
        reason = str(exc)
        if reason not in DENIED_REASONS:
            reason = "path_alias_or_escape"
        record_denial(
            ledger,
            requested_role=role,
            declared_role="",
            lexical_path=lexical,
            resolved_path=None,
            reasons=[reason],
        )
        raise
    record = allowlist.get(absolute)
    if record is None or record["role"] != role:
        declared = "" if record is None else str(record["role"])
        reasons = ["unallowlisted"] if record is None else ["modality_mismatch"]
        if role in FORBIDDEN_ROLES or declared in FORBIDDEN_ROLES:
            reasons.append("forbidden_role")
        primary = record_denial(
            ledger,
            requested_role=role,
            declared_role=declared,
            lexical_path=lexical,
            resolved_path=absolute,
            reasons=reasons,
        )
        raise PermissionError(primary)
    raw = authorized_path.read_bytes()
    ledger["role_byte_open_counts"][role] += 1
    if sha_bytes(raw) != record["sha256"]:
        record_denial(
            ledger,
            requested_role=role,
            declared_role=str(record["role"]),
            lexical_path=lexical,
            resolved_path=absolute,
            reasons=["hash_mismatch"],
        )
        raise ValueError("hash_mismatch")
    return raw


def validate_inventory(raw: bytes) -> None:
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw), "r")
    except (zipfile.BadZipFile, OSError) as exc:
        raise ValueError("label shard is not NPZ") from exc
    with archive:
        infos = archive.infolist()
        names = [str(info.filename) for info in infos]
        if len(names) != len(set(names)) or set(names) != REGISTERED_MEMBERS:
            raise ValueError("label shard ZIP inventory mismatch")
        for info in infos:
            path = Path(info.filename)
            if info.is_dir() or path.name != info.filename or path.is_absolute() or ".." in path.parts or info.flag_bits & 1:
                raise ValueError("unsafe label shard member")
            offset = int(info.header_offset)
            if offset < 0 or offset + 30 > len(raw) or raw[offset : offset + 4] != b"PK\x03\x04":
                raise ValueError("invalid local ZIP header")
            flags = int(struct.unpack_from("<H", raw, offset + 6)[0])
            name_len = int(struct.unpack_from("<H", raw, offset + 26)[0])
            extra_len = int(struct.unpack_from("<H", raw, offset + 28)[0])
            start, end = offset + 30, offset + 30 + name_len
            encoding = "utf-8" if flags & 0x800 else "cp437"
            if end + extra_len > len(raw) or raw[start:end].decode(encoding) != info.filename or flags & 1:
                raise ValueError("central/local ZIP member mismatch")


def canonical_target_ranks(predecessor: dict[str, Any]) -> dict[tuple[Any, ...], int]:
    frames = predecessor.get("frame_reports")
    if type(frames) is not list or len(frames) != EXPECTED_TARGET_ROWS:
        raise ValueError("canonical frame report count mismatch")
    ranks: dict[tuple[Any, ...], int] = {}
    for rank, frame in enumerate(frames):
        if type(frame) is not dict or type(frame.get("record_key")) is not dict:
            raise ValueError("canonical frame record key malformed")
        key = frame["record_key"]
        required = {
            "family", "scene_id", "global_row", "side", "label_row",
            "label_shard_sha256",
        }
        if not required.issubset(key):
            raise ValueError("canonical frame record key incomplete")
        identity = (
            key["family"], key["scene_id"], key["global_row"], key["side"],
            key["label_row"], key["label_shard_sha256"],
        )
        if identity in ranks:
            raise ValueError("canonical frame identity duplicated")
        ranks[identity] = rank
    return ranks


def decode_rows(
    raw: bytes,
    entry: dict[str, Any],
    np: Any,
    *,
    rank_by_identity: dict[tuple[Any, ...], int],
    target_buffer: bytearray,
    rank_filled: bytearray,
    lifetime_events: list[dict[str, Any]] | None = None,
) -> tuple[int, dict[str, int]]:
    validate_inventory(raw)
    expected_hash = entry.get("sha256")
    if type(expected_hash) is not str or len(expected_hash) != 64:
        raise ValueError("label shard hash malformed")
    if (
        type(target_buffer) is not bytearray
        or len(target_buffer) != EXPECTED_TARGET_BYTES
        or type(rank_filled) is not bytearray
        or len(rank_filled) != EXPECTED_TARGET_ROWS
    ):
        raise ValueError("canonical target buffer contract mismatch")
    requests = {side: [] for side in SIDES}
    storage: set[tuple[int, str]] = set()
    for selected in entry["selected_tuples"]:
        if type(selected) is not list or len(selected) != 5:
            raise ValueError("selected tuple malformed")
        family, scene, global_row, side, row = selected
        if side not in SIDES or type(row) is not int or row < 0 or (row, side) in storage:
            raise ValueError("selected storage identity malformed/duplicated")
        storage.add((row, side))
        requests[side].append((family, scene, global_row, side, row))
    selected_count = 0
    decompressions = {"current_labels": 0, "next_labels": 0}
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        if set(map(str, archive.files)) != {name[:-4] for name in REGISTERED_MEMBERS}:
            raise ValueError("NPZ/ZIP inventories disagree")
        storage_rows: int | None = None
        for side in SIDES:
            name = f"{side}_labels"
            labels = np.asarray(archive[name])
            decompressions[name] += 1
            if labels.dtype != np.dtype("uint8") or labels.ndim != 3 or labels.shape[1:] != (64, 64) or not labels.flags.c_contiguous or not np.isin(labels, (0, 1, 2)).all():
                raise ValueError("label array contract mismatch")
            if storage_rows is not None and storage_rows != labels.shape[0]:
                raise ValueError("label array row counts disagree")
            storage_rows = int(labels.shape[0])
            for family, scene, global_row, tuple_side, row in requests[side]:
                if row >= labels.shape[0]:
                    raise ValueError("selected label row out of range")
                identity = (
                    family, scene, global_row, tuple_side, row, expected_hash,
                )
                rank = rank_by_identity.get(identity)
                if rank is None:
                    raise ValueError("selected tuple outside canonical frame order")
                if rank_filled[rank]:
                    raise ValueError("selected tuple duplicated")
                start = rank * TARGET_ROW_BYTES
                target_view = memoryview(labels[row]).cast("B")
                try:
                    target_buffer[start : start + TARGET_ROW_BYTES] = target_view
                finally:
                    target_view.release()
                    del target_view
                rank_filled[rank] = 1
                selected_count += 1
            del labels
            if lifetime_events is not None:
                lifetime_events.append({
                    "event": "array_released",
                    "array": name,
                    "selected_row_copies_retained": 0,
                })
    if lifetime_events is not None:
        lifetime_events.append({
            "event": "archive_released",
            "selected_row_copies_retained": 0,
        })
    return selected_count, decompressions


def manifest_entries(predecessor: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = predecessor.get("label_shard_manifest")
    if type(manifest) is not dict or set(manifest) != {"entry_count", "entries", "manifest_sha256"}:
        raise ValueError("label manifest schema mismatch")
    entries = manifest["entries"]
    if (
        type(entries) is not list
        or len(entries) != 20
        or type(manifest["entry_count"]) is not int
        or manifest["entry_count"] != 20
        or manifest["manifest_sha256"] != EXPECTED_LABEL_MANIFEST_SHA256
        or canonical_hash(entries) != EXPECTED_LABEL_MANIFEST_SHA256
    ):
        raise ValueError("label manifest commitment mismatch")
    return entries


def validate_template_shards(
    records: list[dict[str, str]], predecessor: dict[str, Any]
) -> None:
    expected = sorted(
        [
            {"path": str(entry["path"]), "role": "label_shard", "sha256": str(entry["sha256"])}
            for entry in manifest_entries(predecessor)
        ],
        key=lambda item: (item["path"], item["role"]),
    )
    actual = [item for item in records if item["role"] == "label_shard"]
    if actual != expected:
        raise ValueError("finalizer shard allowlist differs from predecessor commitment")


def load_targets(
    predecessor: dict[str, Any],
    *,
    np: Any,
    allowlist: dict[str, dict[str, str]],
    ledger: dict[str, Any],
    lifetime_events: list[dict[str, Any]] | None = None,
) -> bytes:
    rank_by_identity = canonical_target_ranks(predecessor)
    target_buffer = bytearray(EXPECTED_TARGET_BYTES)
    rank_filled = bytearray(EXPECTED_TARGET_ROWS)
    for entry in manifest_entries(predecessor):
        path = safe_path(str(entry["path"]))
        expected = str(entry["sha256"])
        record = allowlist.get(str(path))
        expected_record = {
            "path": str(path),
            "role": "label_shard",
            "sha256": expected,
        }
        if record != expected_record:
            reasons = ["unallowlisted"] if record is None else ["hash_mismatch"]
            if record is not None and record.get("role") != "label_shard":
                reasons.append("modality_mismatch")
            primary = record_denial(
                ledger,
                requested_role="label_shard",
                declared_role=("" if record is None else str(record.get("role", ""))),
                lexical_path=str(entry["path"]),
                resolved_path=str(path),
                reasons=reasons,
            )
            raise PermissionError(primary)
        if lifetime_events is not None:
            lifetime_events.append({"event": "shard_open", "path": str(path)})
        ledger["label_shard_pre_hash_byte_opens"] += 1
        raw = read_allowed(path, role="label_shard", allowlist=allowlist, ledger=ledger)
        ledger["label_shard_npz_parses"] += 1
        selected_count, decompressions = decode_rows(
            raw,
            entry,
            np,
            rank_by_identity=rank_by_identity,
            target_buffer=target_buffer,
            rank_filled=rank_filled,
            lifetime_events=lifetime_events,
        )
        for name, count in decompressions.items():
            ledger["array_decompression_counts"][name] += count
        ledger["selected_label_rows_read"] += selected_count
        ledger["label_shard_post_hash_byte_opens"] += 1
        post = read_allowed(path, role="label_shard", allowlist=allowlist, ledger=ledger)
        if post != raw:
            raise ValueError("label shard changed during finalization")
        del post, raw, selected_count, decompressions
        if lifetime_events is not None:
            lifetime_events.append({
                "event": "shard_released",
                "path": str(path),
                "selected_row_copies_retained": 0,
            })
    if rank_filled != bytearray([1]) * EXPECTED_TARGET_ROWS:
        raise ValueError("canonical frame missing selected label")
    targets = bytes(target_buffer)
    if len(targets) != EXPECTED_TARGET_BYTES or sha_bytes(targets) != EXPECTED_TARGET_SHA256:
        raise ValueError("selected target byte commitment mismatch")
    return targets


def load_geometry(payload: bytes) -> Any:
    lewm = sys.modules.setdefault("lewm", types.ModuleType("lewm"))
    lewm.__path__ = [str(ROOT / "lewm")]
    benchmarks = sys.modules.setdefault("lewm.benchmarks", types.ModuleType("lewm.benchmarks"))
    benchmarks.__path__ = [str(ROOT / "lewm/benchmarks")]
    name = "lewm.benchmarks.go2_dynamic_cell_square_projection"
    module = types.ModuleType(name)
    module.__file__ = str(ROOT / "lewm/benchmarks/go2_dynamic_cell_square_projection.py")
    module.__package__ = "lewm.benchmarks"
    sys.modules[name] = module
    exec(compile(payload, module.__file__, "exec", dont_inherit=True), module.__dict__)
    return module


def center_mask(geometry: Any) -> tuple[tuple[bool, ...], ...]:
    tan_h = math.tan(math.radians(geometry.HORIZONTAL_FOV_DEG) * 0.5)
    tan_v = math.tan(math.radians(geometry.VERTICAL_FOV_DEG) * 0.5)
    camera_f, camera_l, camera_u = geometry.CAMERA_XYZ_BODY_M
    return tuple(
        tuple(
            any(
                (point_f := geometry.cell_center(row, column)[0] - camera_f) >= geometry.CAMERA_NEAR_M
                and -point_f * tan_h <= geometry.cell_center(row, column)[1] - camera_l <= point_f * tan_h
                and -point_f * tan_v <= anchor - camera_u <= point_f * tan_v
                for anchor in geometry.VERTICAL_ANCHOR_Z_M
            )
            for column in range(64)
        )
        for row in range(64)
    )


def identity(key: dict[str, Any], class_name: str, row: int, column: int) -> dict[str, Any]:
    return {"class_id": CLASS_IDS[class_name], "class_name": class_name, "column": column, "frame_key": key, "row": row}


def compute_science(predecessor: dict[str, Any], targets: bytes, geometry: Any) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    if validate_content(predecessor, name="predecessor") != PREDECESSOR_CONTENT_SHA256:
        raise ValueError("predecessor content is not frozen")
    if len(predecessor.get("frame_reports", [])) != 320:
        raise ValueError("predecessor frame count mismatch")
    centre = center_mask(geometry)
    static = geometry.build_dynamic_cell_square_support_mask((0.0, 0.0, 0.0, 1.0), 0.0)
    counts: Counter[tuple[str, str, str]] = Counter()
    totals = Counter({name: 0 for name in CLASSES})
    remaining = {"center": [], "static": [], "dynamic": []}
    per_frame: list[dict[str, int]] = []
    summaries: list[dict[str, int]] = []
    for frame_rank, frame in enumerate(predecessor["frame_reports"]):
        key = frame["record_key"]
        if set(key) != set(FRAME_FIELDS) or key["family"] not in FAMILIES or key["side"] not in SIDES:
            raise ValueError("predecessor frame identity malformed")
        camera = frame["camera_mount_composition"]
        dynamic = geometry.build_dynamic_cell_square_support_mask(
            camera["base_quat_world_xyzw"], camera["stored_base_yaw_rad"]
        )
        target = targets[frame_rank * 4096 : (frame_rank + 1) * 4096]
        frame_counts = Counter({name: 0 for name in CLASSES})
        supported: Counter[tuple[str, str]] = Counter()
        for flat, class_id in enumerate(target):
            class_name = CLASSES[class_id]
            totals[class_name] += 1
            frame_counts[class_name] += 1
            if class_id == 0:
                continue
            row, column = divmod(flat, 64)
            support = {"center": centre[row][column], "static": static[row][column], "dynamic": dynamic[row][column]}
            counts[(key["family"], class_name, "total")] += 1
            for name, passed in support.items():
                counts[(key["family"], class_name, name)] += int(passed)
                supported[(name, class_name)] += int(passed)
                if not passed:
                    remaining[name].append(identity(key, class_name, row, column))
        per_frame.append({"frame_rank": frame_rank, "unknown": frame_counts["unknown"], "free": frame_counts["free"], "occupied": frame_counts["occupied"], "all": sum(frame_counts.values())})
        summaries.append({
            "family_rank": FAMILIES.index(key["family"]), "frame_rank": frame_rank,
            "unknown_total": frame_counts["unknown"], "free_total": frame_counts["free"], "occupied_total": frame_counts["occupied"],
            "level_center_free_supported": supported[("center", "free")], "level_center_occupied_supported": supported[("center", "occupied")],
            "static_free_supported": supported[("static", "free")], "static_occupied_supported": supported[("static", "occupied")],
            "dynamic_free_supported": supported[("dynamic", "free")], "dynamic_occupied_supported": supported[("dynamic", "occupied")],
        })
    class_totals = {"unknown": totals["unknown"], "free": totals["free"], "occupied": totals["occupied"], "all": sum(totals.values())}
    known = totals["free"] + totals["occupied"]
    rows = [
        {"family": family, "class_id": CLASS_IDS[class_name], "class_name": class_name,
         "total": counts[(family, class_name, "total")], "level_center_supported": counts[(family, class_name, "center")],
         "static_cell_square_supported": counts[(family, class_name, "static")], "dynamic_cell_square_supported": counts[(family, class_name, "dynamic")]}
        for family in FAMILIES for class_name in KNOWN_CLASSES
    ]
    def support_record(name: str) -> dict[str, Any]:
        missing = remaining[name]
        free = sum(item["class_name"] == "free" for item in missing)
        return {"known_total": known, "supported_count": known - len(missing), "unsupported_count": len(missing),
                "unsupported_free_count": free, "unsupported_occupied_count": len(missing) - free,
                "unsupported_frame_count": len({tuple(item["frame_key"][field] for field in FRAME_FIELDS) for item in missing}),
                "unsupported_identities_sha256": canonical_hash(missing)}
    scientific = {
        "label_reconciliation": {"byte_count": len(targets), "byte_sha256": sha_bytes(targets), "class_totals": class_totals,
            "known_total": known, "per_frame_cell_count": 4096, "per_frame_count": len(per_frame),
            "per_frame_totals_sha256": canonical_hash(per_frame),
            "all_counts_reconcile": class_totals == EXPECTED_CLASS_TOTALS and known == EXPECTED_KNOWN_TOTAL and all(item["all"] == 4096 for item in per_frame)},
        "support": {
            "level_center": {"support_cell_count": sum(map(sum, centre)), "support_mask_sha256": geometry.support_mask_sha256(centre),
                "free_total": totals["free"], "free_supported": sum(counts[(family, "free", "center")] for family in FAMILIES),
                "occupied_total": totals["occupied"], "occupied_supported": sum(counts[(family, "occupied", "center")] for family in FAMILIES),
                "known_violation_count": len(remaining["center"]), "known_violation_identities_sha256": canonical_hash(remaining["center"])},
            "level_cell_square": {"support_cell_count": sum(map(sum, static)), "support_mask_sha256": geometry.support_mask_sha256(static)},
            "static_cell_square_known": support_record("static"), "dynamic_cell_square_known": support_record("dynamic")},
        "family_class_rows": rows, "frame_summary_records_sha256": canonical_hash(summaries),
    }
    return scientific, remaining


def gates(scientific: dict[str, Any], access: bool, independent: bool) -> dict[str, bool]:
    label, support = scientific["label_reconciliation"], scientific["support"]
    center, square = support["level_center"], support["level_cell_square"]
    static, dynamic = support["static_cell_square_known"], support["dynamic_cell_square_known"]
    result = {
        "binding_and_source_hashes_pass": True, "predecessor_authority_pass": True,
        "label_manifest_and_bytes_pass": label["byte_count"] == EXPECTED_TARGET_BYTES and label["byte_sha256"] == EXPECTED_TARGET_SHA256,
        "label_count_reconciliation_pass": label["all_counts_reconcile"] is True,
        "level_center_parity_pass": exact_equal(center, {"support_cell_count": EXPECTED_CENTER_COUNT, "support_mask_sha256": EXPECTED_CENTER_HASH,
            "free_total": EXPECTED_CLASS_TOTALS["free"], "free_supported": EXPECTED_CENTER_FREE,
            "occupied_total": EXPECTED_CLASS_TOTALS["occupied"], "occupied_supported": EXPECTED_CENTER_OCCUPIED,
            "known_violation_count": EXPECTED_CENTER_VIOLATIONS, "known_violation_identities_sha256": EXPECTED_CENTER_IDENTITIES_HASH}),
        "level_cell_square_frozen_pass": exact_equal(square, {"support_cell_count": EXPECTED_SQUARE_COUNT, "support_mask_sha256": EXPECTED_SQUARE_HASH}),
        "static_all_known_scored_pass": exact_equal(static, {"known_total": EXPECTED_KNOWN_TOTAL, "supported_count": EXPECTED_KNOWN_TOTAL - 4,
            "unsupported_count": 4, "unsupported_free_count": 0, "unsupported_occupied_count": 4, "unsupported_frame_count": 4,
            "unsupported_identities_sha256": EXPECTED_STATIC_IDENTITIES_HASH}),
        "dynamic_all_known_scored_pass": dynamic["known_total"] == EXPECTED_KNOWN_TOTAL,
        "dynamic_zero_known_unsupported_pass": exact_equal(dynamic, {"known_total": EXPECTED_KNOWN_TOTAL, "supported_count": EXPECTED_KNOWN_TOTAL,
            "unsupported_count": 0, "unsupported_free_count": 0, "unsupported_occupied_count": 0, "unsupported_frame_count": 0,
            "unsupported_identities_sha256": EMPTY_HASH}),
        "access_reconciliation_pass": access, "independent_recomputation_pass": independent, "all_passed": False,
    }
    result["all_passed"] = all(result[key] for key in GATE_KEYS if key != "all_passed")
    return result


def _validate_scientific_shape(scientific: object) -> dict[str, Any]:
    if type(scientific) is not dict or set(scientific) != {
        "label_reconciliation", "support", "family_class_rows", "frame_summary_records_sha256"
    }:
        raise ValueError("candidate scientific core key set mismatch")
    label = scientific["label_reconciliation"]
    if type(label) is not dict or set(label) != {
        "byte_count", "byte_sha256", "class_totals", "known_total", "per_frame_cell_count",
        "per_frame_count", "per_frame_totals_sha256", "all_counts_reconcile",
    } or type(label["class_totals"]) is not dict or set(label["class_totals"]) != {"unknown", "free", "occupied", "all"}:
        raise ValueError("candidate label reconciliation schema mismatch")
    for field in ("byte_count", "known_total", "per_frame_cell_count", "per_frame_count"):
        if type(label[field]) is not int or label[field] < 0:
            raise ValueError("candidate label count type mismatch")
    if type(label["all_counts_reconcile"]) is not bool or not is_sha(label["byte_sha256"]) or not is_sha(label["per_frame_totals_sha256"]):
        raise ValueError("candidate label scalar/hash type mismatch")
    if any(type(value) is not int or value < 0 for value in label["class_totals"].values()):
        raise ValueError("candidate class total type mismatch")
    support = scientific["support"]
    if type(support) is not dict or set(support) != {"level_center", "level_cell_square", "static_cell_square_known", "dynamic_cell_square_known"}:
        raise ValueError("candidate support schema mismatch")
    if set(support["level_center"]) != {"support_cell_count", "support_mask_sha256", "free_total", "free_supported", "occupied_total", "occupied_supported", "known_violation_count", "known_violation_identities_sha256"}:
        raise ValueError("candidate level-center schema mismatch")
    if set(support["level_cell_square"]) != {"support_cell_count", "support_mask_sha256"}:
        raise ValueError("candidate level-square schema mismatch")
    known_keys = {"known_total", "supported_count", "unsupported_count", "unsupported_free_count", "unsupported_occupied_count", "unsupported_frame_count", "unsupported_identities_sha256"}
    if set(support["static_cell_square_known"]) != known_keys or set(support["dynamic_cell_square_known"]) != known_keys:
        raise ValueError("candidate known-support schema mismatch")
    for record in support.values():
        for key, value in record.items():
            if key.endswith("sha256"):
                if not is_sha(value):
                    raise ValueError("candidate support hash malformed")
            elif type(value) is not int or value < 0:
                raise ValueError("candidate support count type mismatch")
    rows = scientific["family_class_rows"]
    expected_pairs = [(family, class_name) for family in FAMILIES for class_name in KNOWN_CLASSES]
    row_keys = {"family", "class_id", "class_name", "total", "level_center_supported", "static_cell_square_supported", "dynamic_cell_square_supported"}
    if type(rows) is not list or len(rows) != 10:
        raise ValueError("candidate family-class row count mismatch")
    for row, pair in zip(rows, expected_pairs):
        if type(row) is not dict or set(row) != row_keys or (row["family"], row["class_name"]) != pair or row["class_id"] != CLASS_IDS[pair[1]]:
            raise ValueError("candidate family-class row schema/order mismatch")
        if any(
            type(row[field]) is not int or row[field] < 0
            for field in row_keys - {"family", "class_name"}
        ):
            raise ValueError("candidate family-class count type mismatch")
    if not is_sha(scientific["frame_summary_records_sha256"]):
        raise ValueError("candidate frame-summary hash malformed")
    return scientific


def validate_candidate(
    candidate: dict[str, Any],
    *,
    candidate_hash: str,
    manifest_hash: str,
    manifest: dict[str, Any],
    runner_instantiated: list[dict[str, str]],
) -> dict[str, Any]:
    if set(candidate) != CANDIDATE_KEYS or candidate.get("schema") != CANDIDATE_SCHEMA:
        raise ValueError("candidate schema/key set mismatch")
    validate_content(candidate, name="candidate")
    validate_utc_timestamp(candidate.get("created_at_utc"), name="candidate timestamp")
    if not is_sha(candidate_hash):
        raise ValueError("candidate file hash malformed")
    if not exact_equal(candidate["execution_binding"], {"path": BINDING_PATH, "file_sha256": BINDING_SHA256}):
        raise ValueError("candidate execution binding mismatch")
    expected_manifests = {
        "human": {
            "path": manifest["human_manifest"]["path"],
            "file_sha256": manifest["human_manifest"]["file_sha256"],
        },
        "machine": {
            "path": MACHINE_PATH,
            "file_sha256": manifest_hash,
            "content_sha256": manifest["content_sha256"],
        },
    }
    if not exact_equal(candidate["implementation_manifests"], expected_manifests):
        raise ValueError("candidate implementation-manifest commitments mismatch")
    expected_inputs = {
        "predecessor_report": {"path": PREDECESSOR_REPORT_PATH, "file_sha256": PREDECESSOR_REPORT_SHA256},
        "predecessor_result": {"path": PREDECESSOR_RESULT_PATH, "file_sha256": PREDECESSOR_FILE_SHA256, "content_sha256": PREDECESSOR_CONTENT_SHA256},
        "dynamic_geometry": {"path": SOURCE_MAP_CONTRACT[0][1], "file_sha256": DYNAMIC_GEOMETRY_SHA256},
        "label_shard_manifest": {"entry_count": 20, "manifest_sha256": EXPECTED_LABEL_MANIFEST_SHA256},
        "selected_targets": {"frame_count": 320, "byte_count": EXPECTED_TARGET_BYTES, "sha256": EXPECTED_TARGET_SHA256},
    }
    if not exact_equal(candidate["inputs"], expected_inputs):
        raise ValueError("candidate input commitments mismatch")
    if not exact_equal(candidate["source_map"], manifest["source_map"]):
        raise ValueError("candidate source map differs from machine manifest")
    expected_scope = {
        "dataset_role": "train", "learning_performed": False, "frame_count": 320,
        "transition_count": 160, "families": list(FAMILIES), "endpoint_sides": list(SIDES),
        "class_order": list(CLASSES), "forbidden_roles": list(FORBIDDEN_ROLES),
    }
    if not exact_equal(candidate["scope"], expected_scope):
        raise ValueError("candidate scope mismatch")
    if not exact_equal(candidate["preparation_access_ledger"], manifest["preparation_access_ledger"]):
        raise ValueError("candidate preparation ledger differs from machine manifest")
    source_by_role = {
        item["role"]: item for item in manifest["source_map"]["entries"]
    }
    validate_exact_preparation_ledger(
        candidate["preparation_access_ledger"],
        source_by_role=source_by_role,
        human_manifest=manifest["human_manifest"],
        label_records=[
            item for item in runner_instantiated if item["role"] == "label_shard"
        ],
    )
    runner_ledger = validate_phase_ledger(candidate["runner_access_ledger"], phase="runner")
    expected_runner_reads, _lookup = anchor_records(runner_instantiated)
    if (
        not exact_equal(runner_ledger["authorized_read_paths"], expected_runner_reads)
        or runner_ledger["authorized_read_path_set_sha256"] != canonical_hash(expected_runner_reads)
    ):
        raise ValueError("candidate runner ledger differs from instantiated allowlist")
    scientific = {key: candidate[key] for key in ("label_reconciliation", "support", "family_class_rows", "frame_summary_records_sha256")}
    _validate_scientific_shape(scientific)
    if canonical_hash(scientific) != candidate.get("scientific_core_sha256"):
        raise ValueError("candidate scientific-core hash mismatch")
    forbidden = (b'"scene_id"', b'"global_row"', b'"label_row"', b'"image_sha256"', b'"base_quat', b'"stored_base_yaw', b'"remaining_identities"')
    payload = canonical_bytes(candidate)
    if any(token in payload for token in forbidden):
        raise ValueError("candidate leaks a forbidden passing identity")
    expected_gates = gates(scientific, True, False)
    if set(candidate["gates"]) != set(GATE_KEYS) or not exact_equal(candidate["gates"], expected_gates):
        raise ValueError("candidate gates do not equal recomputed candidate gates")
    return scientific


def candidate_unchanged(path: Path, initial: os.stat_result) -> bool:
    current = path.lstat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    return all(getattr(initial, field) == getattr(current, field) for field in fields)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    values = list(sys.argv[1:] if argv is None else argv)
    if (
        len(values) != 4
        or values[0] != "--implementation-manifest-sha256"
        or values[2] != "--candidate-sha256"
    ):
        raise SystemExit(
            "finalizer requires ordered manifest-hash then candidate-hash arguments"
        )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation-manifest-sha256", required=True)
    parser.add_argument("--candidate-sha256", required=True)
    args = parser.parse_args(values)
    if not is_sha(args.implementation_manifest_sha256) or not is_sha(args.candidate_sha256):
        parser.error("both arguments must be lowercase SHA-256 values")
    return args


def write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_hash, candidate_hash = str(args.implementation_manifest_sha256), str(args.candidate_sha256)
    result_path, failure_path = safe_path(RESULT_PATH), safe_path(FAILURE_PATH)
    if result_path.exists() or failure_path.exists():
        raise FileExistsError("immutable finalizer output already exists")
    machine_path, candidate_path = safe_path(MACHINE_PATH), safe_path(CANDIDATE_PATH)
    machine_raw = machine_path.read_bytes()
    candidate_initial_stat = candidate_path.lstat()
    candidate_raw = candidate_path.read_bytes()
    if sha_bytes(machine_raw) != manifest_hash or sha_bytes(candidate_raw) != candidate_hash:
        raise ValueError("bootstrap manifest/candidate hash mismatch")
    manifest = strict_json(machine_raw, name="machine manifest")
    candidate = strict_json(candidate_raw, name="candidate")
    if machine_raw != canonical_bytes(manifest) + b"\n":
        raise ValueError("machine manifest bytes are not canonical")
    if candidate_raw != canonical_bytes(candidate) + b"\n":
        raise ValueError("candidate bytes are not canonical")
    phase, instantiated, runner_phase, runner_instantiated = validate_machine_manifest(
        manifest,
        manifest_sha256=manifest_hash,
        candidate_sha256=candidate_hash,
    )
    del runner_phase
    candidate_science = validate_candidate(
        candidate,
        candidate_hash=candidate_hash,
        manifest_hash=manifest_hash,
        manifest=manifest,
        runner_instantiated=runner_instantiated,
    )
    records, allowlist = anchor_records(instantiated)
    ledger = new_ledger(phase, records)
    ledger["role_byte_open_counts"]["machine_manifest"] = 1
    ledger["role_byte_open_counts"]["candidate"] = 1
    geometry_raw: bytes | None = None
    predecessor: dict[str, Any] | None = None
    for record in records:
        role = record["role"]
        if role in ("machine_manifest", "candidate", "label_shard"):
            continue
        raw = read_allowed(Path(record["path"]), role=role, allowlist=allowlist, ledger=ledger)
        if role == "dynamic_geometry":
            geometry_raw = raw
        elif role == "predecessor_result":
            if sha_bytes(raw) != PREDECESSOR_FILE_SHA256:
                raise ValueError("predecessor file hash mismatch")
            predecessor = strict_json(raw, name="predecessor result")
    if geometry_raw is None or predecessor is None:
        raise ValueError("finalizer allowlist omitted required inputs")
    validate_template_shards(instantiated, predecessor)
    validate_template_shards(runner_instantiated, predecessor)
    geometry = load_geometry(geometry_raw)
    import numpy as np
    targets = load_targets(predecessor, np=np, allowlist=allowlist, ledger=ledger)
    expected_label = phase["expected_label_access"]
    ledger["all_counts_reconcile"] = (
        ledger["role_byte_open_counts"] == phase["expected_role_byte_open_counts"]
        and all(ledger[key] == expected_label[key] for key in expected_label if key != "array_decompression_counts")
        and ledger["array_decompression_counts"] == expected_label["array_decompression_counts"]
        and not ledger["denied_attempt_records"] and ledger["unexpected_path_attempts"] == 0
        and not any(ledger["forbidden_role_open_counts"].values())
        and ledger["unselected_rows_scored"] == ledger["unselected_rows_retained"] == 0
    )
    validate_phase_ledger(ledger, phase="finalizer")
    final_science, remaining = compute_science(predecessor, targets, geometry)
    del targets
    candidate_core_hash = candidate["scientific_core_sha256"]
    final_core_hash = canonical_hash(final_science)
    exactly_equal = (
        exact_equal(candidate_science, final_science)
        and candidate_core_hash == final_core_hash
    )
    final_gates = gates(final_science, ledger["all_counts_reconcile"], exactly_equal)
    if not candidate_unchanged(candidate_path, candidate_initial_stat):
        raise RuntimeError("candidate changed between bootstrap and publication")
    if not final_gates["all_passed"]:
        failure_core = {
            "schema": "lewm_go2_dynamic_cell_square_projection_failure_diagnostic_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "execution_binding": {"path": BINDING_PATH, "file_sha256": BINDING_SHA256},
            "candidate": {"path": CANDIDATE_PATH, "file_sha256": candidate_hash, "content_sha256": candidate["content_sha256"]},
            "gates": final_gates,
            "remaining_identities": remaining,
            "licenses": {"model": False, "runtime": False, "promotion": False},
        }
        failure = {**failure_core, "content_sha256": canonical_hash(failure_core)}
        write_exclusive(failure_path, canonical_bytes(failure) + b"\n")
        return 2
    final_core = {
        "schema": FINAL_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_binding": candidate["execution_binding"],
        "implementation_manifests": candidate["implementation_manifests"],
        "candidate": {"path": CANDIDATE_PATH, "file_sha256": candidate_hash, "content_sha256": candidate["content_sha256"]},
        "inputs": candidate["inputs"], "source_map": candidate["source_map"], "scope": candidate["scope"],
        "preparation_access_ledger": candidate["preparation_access_ledger"], "runner_access_ledger": candidate["runner_access_ledger"],
        "finalizer_access_ledger": ledger, **final_science, "scientific_core_sha256": final_core_hash,
        "independent_recomputation": {"candidate_scientific_core_sha256": candidate_core_hash, "finalizer_scientific_core_sha256": final_core_hash, "exactly_equal": exactly_equal},
        "gates": final_gates,
    }
    final = {**final_core, "content_sha256": canonical_hash(final_core)}
    if set(final) != FINAL_KEYS:
        raise ValueError("final result exact key set mismatch")
    payload = canonical_bytes(final) + b"\n"
    write_exclusive(result_path, payload)
    print(canonical_bytes({"candidate_file_sha256": candidate_hash, "content_sha256": final["content_sha256"], "file_sha256": sha_bytes(payload), "result_path": str(result_path)}).decode("utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
