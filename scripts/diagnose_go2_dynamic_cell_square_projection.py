#!/usr/bin/env python3
"""Create the immutable all-cell dynamic-projection candidate."""
from __future__ import annotations

# Bootstrap imports are stdlib-only. Repository modules are executed from
# already-hashed source bytes after the reviewed allowlist passes.
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import stat
import struct
import sys
import types
from typing import Any, Sequence
import zipfile


ROOT = Path(__file__).resolve().parents[1]
BINDING_RELATIVE_PATH = (
    "docs/lewm_go2_n32_dynamic_cell_square_geometry_binding_2026-07-11.md"
)
BINDING_SHA256 = "211043ee3c3200d1fc93febbae73059341aea19560c83f53f3b3bb231bf06e66"
MACHINE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_n32_dynamic_cell_square_geometry_implementation_manifest_2026-07-11.json"
)
HUMAN_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_n32_dynamic_cell_square_geometry_implementation_manifest_2026-07-11.md"
)
PREDECESSOR_RESULT_RELATIVE_PATH = (
    ".generated/go2_n32_camera_frustum_observability_audit/v2/result.json"
)
PREDECESSOR_RESULT_FILE_SHA256 = (
    "7725ecddf2fa77bb762733fd35df2efd2fb60d4f9aa8ab6fdf2bee660522909e"
)
PREDECESSOR_REPORT_RELATIVE_PATH = (
    "docs/lewm_go2_n32_camera_frustum_observability_audit_v2_result_2026-07-11.md"
)
PREDECESSOR_REPORT_SHA256 = (
    "8bfb4c9a8b69f67b3b9e4d6e3b21e9ff89ecaff89a2bab3eb83d759ca4fe6d22"
)
DYNAMIC_GEOMETRY_SHA256 = (
    "ce2bb0d38ed1436635cdd1468ba1dfe1a935fdafdd6dda5adcf37b97a32a74bf"
)
CANDIDATE_RELATIVE_PATH = (
    ".generated/go2_dynamic_cell_square_projection_diagnostic/v1/candidate.json"
)
FINAL_RESULT_RELATIVE_PATH = (
    ".generated/go2_dynamic_cell_square_projection_diagnostic/v1/result.json"
)
FAILURE_RESULT_RELATIVE_PATH = (
    ".generated/go2_dynamic_cell_square_projection_diagnostic/v1/"
    "failure_diagnostic.json"
)
MACHINE_SCHEMA = "lewm_go2_dynamic_cell_square_projection_implementation_manifest_v1"
LEDGER_SCHEMA = "lewm_go2_dynamic_projection_access_ledger_v1"
EXPECTED_LABEL_MANIFEST_SHA256 = (
    "998ce5a768029c23c931fbbec730c1fe31b9ed1fe155494fc68f34a0c23d3d1b"
)
EXPECTED_TARGET_BYTES = 1_310_720
TARGET_ROW_BYTES = 64 * 64
EXPECTED_TARGET_ROWS = EXPECTED_TARGET_BYTES // TARGET_ROW_BYTES
EXPECTED_TARGET_SHA256 = (
    "6952c1f9604da1d9fd4c94a3f33deb142451836609b7059970ff6c459737ce05"
)
SELF_TEMPLATE_ENTRY = {
    "path": MACHINE_MANIFEST_RELATIVE_PATH,
    "role": "machine_manifest",
    "sha256_source": "implementation_manifest_sha256_argument",
}
CANDIDATE_TEMPLATE_ENTRY = {
    "path": CANDIDATE_RELATIVE_PATH,
    "role": "candidate",
    "sha256_source": "candidate_sha256_argument",
}
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
REGISTERED_NPZ_MEMBERS = {
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
MACHINE_KEYS = {
    "schema",
    "created_at_utc",
    "execution_binding",
    "human_manifest",
    "inputs",
    "source_map",
    "phase_contracts",
    "preparation_access_ledger",
    "output_absence",
    "runtime_environment",
    "content_sha256",
}
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


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def exact_equal(first: object, second: object) -> bool:
    return canonical_json_bytes(first) == canonical_json_bytes(second)


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


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def strict_json_bytes(payload: bytes, *, name: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if type(key) is not str or key in result:
                raise ValueError(f"{name} has duplicate or non-string keys")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"{name} contains nonfinite constant {value}")

    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise ValueError(f"{name} root must be an exact object")
    return value


def _validate_content_hash(record: dict[str, Any], *, name: str) -> str:
    declared = record.get("content_sha256")
    if not _is_sha256(declared):
        raise ValueError(f"{name} content hash is malformed")
    core = dict(record)
    del core["content_sha256"]
    if canonical_json_sha256(core) != declared:
        raise ValueError(f"{name} content hash mismatch")
    return str(declared)


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def instantiate_read_template(
    template: object,
    *,
    verified_manifest_sha256: str,
    manifest_bytes_verified: bool,
) -> list[dict[str, str]]:
    if not manifest_bytes_verified:
        raise ValueError("machine manifest bytes must be verified before substitution")
    if not _is_sha256(verified_manifest_sha256) or type(template) is not list:
        raise ValueError("allowlist template or manifest SHA-256 is malformed")
    placeholders = [
        item
        for item in template
        if type(item) is dict and "sha256_source" in item
    ]
    if len(placeholders) != 1 or placeholders[0] != SELF_TEMPLATE_ENTRY:
        raise ValueError("machine-manifest self placeholder is not exact")
    result: list[dict[str, str]] = []
    for item in template:
        if item == SELF_TEMPLATE_ENTRY:
            result.append(
                {
                    "path": MACHINE_MANIFEST_RELATIVE_PATH,
                    "role": "machine_manifest",
                    "sha256": verified_manifest_sha256,
                }
            )
            continue
        if type(item) is not dict or set(item) != {"path", "role", "sha256"}:
            raise ValueError("non-self allowlist template entry is malformed")
        if not _is_sha256(item["sha256"]):
            raise ValueError("allowlist template SHA-256 is malformed")
        result.append(dict(item))
    return result


def _validate_machine_manifest(
    manifest: dict[str, Any], *, manifest_sha256: str
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    if set(manifest) != MACHINE_KEYS or manifest.get("schema") != MACHINE_SCHEMA:
        raise ValueError("machine manifest schema or key set mismatch")
    _validate_content_hash(manifest, name="machine manifest")
    validate_utc_timestamp(manifest.get("created_at_utc"), name="machine timestamp")
    if not exact_equal(manifest.get("execution_binding"), {
        "path": BINDING_RELATIVE_PATH,
        "file_sha256": BINDING_SHA256,
    }):
        raise ValueError("machine manifest binding mismatch")
    human = manifest.get("human_manifest")
    if (
        type(human) is not dict
        or set(human) != {"path", "file_sha256"}
        or human.get("path") != HUMAN_MANIFEST_RELATIVE_PATH
        or not _is_sha256(human.get("file_sha256"))
    ):
        raise ValueError("machine manifest human-manifest commitment mismatch")
    inputs = manifest.get("inputs")
    if type(inputs) is not dict or set(inputs) != {
        "predecessor_report",
        "predecessor_result",
        "label_shard_manifest",
        "selected_targets",
    }:
        raise ValueError("machine manifest input key set mismatch")
    if not exact_equal(inputs["predecessor_result"], {
        "path": PREDECESSOR_RESULT_RELATIVE_PATH,
        "file_sha256": PREDECESSOR_RESULT_FILE_SHA256,
        "content_sha256": (
            "11420607d2c4f8e79af9214d43bbc6259669ee84c9ccc0aaefd4167cc1d809a1"
        ),
    }):
        raise ValueError("machine manifest predecessor commitment mismatch")
    if not exact_equal(inputs["predecessor_report"], {
        "path": PREDECESSOR_REPORT_RELATIVE_PATH,
        "file_sha256": PREDECESSOR_REPORT_SHA256,
    }):
        raise ValueError("machine manifest predecessor-report commitment mismatch")
    if not exact_equal(inputs["label_shard_manifest"], {
        "entry_count": 20,
        "manifest_sha256": EXPECTED_LABEL_MANIFEST_SHA256,
    }) or not exact_equal(inputs["selected_targets"], {
        "frame_count": 320,
        "byte_count": EXPECTED_TARGET_BYTES,
        "sha256": EXPECTED_TARGET_SHA256,
    }):
        raise ValueError("machine manifest label commitments mismatch")
    source_map = manifest.get("source_map")
    if type(source_map) is not dict or set(source_map) != {
        "entries",
        "entry_count",
        "source_map_sha256",
    }:
        raise ValueError("machine manifest source map schema mismatch")
    entries = source_map["entries"]
    if (
        type(entries) is not list
        or len(entries) != 9
        or type(source_map["entry_count"]) is not int
        or source_map["entry_count"] != 9
    ):
        raise ValueError("machine manifest source map count mismatch")
    if canonical_json_sha256(entries) != source_map["source_map_sha256"]:
        raise ValueError("machine manifest source map hash mismatch")
    for item, (role, path) in zip(entries, SOURCE_MAP_CONTRACT):
        if (
            type(item) is not dict
            or set(item) != {"path", "role", "sha256"}
            or item["role"] != role
            or item["path"] != path
            or not _is_sha256(item["sha256"])
        ):
            raise ValueError("machine manifest source map contract mismatch")
    source_by_role = {item["role"]: item for item in entries}
    if source_by_role["dynamic_geometry"]["sha256"] != DYNAMIC_GEOMETRY_SHA256:
        raise ValueError("machine manifest dynamic-geometry hash mismatch")
    expected_runtime = {
        "python_implementation": sys.implementation.name,
        "python_version": list(sys.version_info[:3]),
        "numpy_version": importlib.metadata.version("numpy"),
    }
    if not exact_equal(manifest.get("runtime_environment"), expected_runtime):
        raise ValueError("machine runtime environment does not match execution")
    expected_absence = {
        "paths": [
            {"path": str(ROOT / relative), "exists": False}
            for relative in (
                CANDIDATE_RELATIVE_PATH,
                ".generated/go2_dynamic_cell_square_projection_diagnostic/v1/result.json",
                ".generated/go2_dynamic_cell_square_projection_diagnostic/v1/failure_diagnostic.json",
            )
        ],
        "all_absent": True,
    }
    if not exact_equal(manifest.get("output_absence"), expected_absence):
        raise ValueError("machine output-absence proof is not exact")

    phases = manifest.get("phase_contracts")
    if type(phases) is not dict or set(phases) != {"runner", "finalizer"}:
        raise ValueError("machine manifest phase contracts mismatch")
    phase = phases["runner"]
    expected_phase_keys = {
        "authorized_read_path_template",
        "authorized_read_path_template_sha256",
        "authorized_write_paths",
        "authorized_write_path_set_sha256",
        "expected_roles",
        "expected_role_byte_open_counts",
        "expected_label_access",
    }
    if type(phase) is not dict or set(phase) != expected_phase_keys:
        raise ValueError("runner phase contract key set mismatch")
    template = phase["authorized_read_path_template"]
    if canonical_json_sha256(template) != phase["authorized_read_path_template_sha256"]:
        raise ValueError("runner allowlist template hash mismatch")
    actual = instantiate_read_template(
        template,
        verified_manifest_sha256=manifest_sha256,
        manifest_bytes_verified=True,
    )
    if actual != sorted(actual, key=lambda item: (item["path"], item["role"])):
        raise ValueError("instantiated runner allowlist is not sorted")
    roles = sorted(set(item["role"] for item in actual))
    if not exact_equal(phase["expected_roles"], roles):
        raise ValueError("runner expected role set mismatch")
    if set(phase["expected_role_byte_open_counts"]) != set(roles):
        raise ValueError("runner role-open map differs from its roles")
    expected_counts = {role: 1 for role in roles if role != "label_shard"}
    expected_counts["label_shard"] = 40
    if not exact_equal(phase["expected_role_byte_open_counts"], expected_counts):
        raise ValueError("runner role-open counts are not exact")
    if not exact_equal(phase["expected_label_access"], EXPECTED_LABEL_ACCESS):
        raise ValueError("runner numeric label-access contract mismatch")
    expected_writes = [
        {"path": CANDIDATE_RELATIVE_PATH, "role": "runner_output", "sha256": None}
    ]
    if (
        not exact_equal(phase["authorized_write_paths"], expected_writes)
        or phase["authorized_write_path_set_sha256"]
        != canonical_json_sha256(expected_writes)
    ):
        raise ValueError("runner write allowlist mismatch")
    expected_nonshard = [
        {"path": BINDING_RELATIVE_PATH, "role": "binding", "sha256": BINDING_SHA256},
        {"path": HUMAN_MANIFEST_RELATIVE_PATH, "role": "human_manifest", "sha256": human["file_sha256"]},
        {"path": MACHINE_MANIFEST_RELATIVE_PATH, "role": "machine_manifest", "sha256": manifest_sha256},
        {"path": PREDECESSOR_RESULT_RELATIVE_PATH, "role": "predecessor_result", "sha256": PREDECESSOR_RESULT_FILE_SHA256},
        *[
            dict(source_by_role[role])
            for role in (
                "dynamic_geometry",
                "diagnostic_core",
                "runner",
                "geometry_test",
                "diagnostic_test",
                "preparation_test",
                "finalizer_test",
            )
        ],
    ]
    expected_nonshard.sort(key=lambda item: (item["path"], item["role"]))
    actual_nonshard = [item for item in actual if item["role"] != "label_shard"]
    if actual_nonshard != expected_nonshard:
        raise ValueError("runner allowlist role/path graph is not exact")
    if len([item for item in actual if item["role"] == "label_shard"]) != 20:
        raise ValueError("runner allowlist does not contain exactly 20 shards")

    finalizer_phase = phases["finalizer"]
    if type(finalizer_phase) is not dict or set(finalizer_phase) != expected_phase_keys:
        raise ValueError("finalizer phase contract key set mismatch")
    expected_finalizer_template = [
        {
            "path": BINDING_RELATIVE_PATH,
            "role": "binding",
            "sha256": BINDING_SHA256,
        },
        {
            "path": HUMAN_MANIFEST_RELATIVE_PATH,
            "role": "human_manifest",
            "sha256": human["file_sha256"],
        },
        dict(SELF_TEMPLATE_ENTRY),
        dict(CANDIDATE_TEMPLATE_ENTRY),
        {
            "path": PREDECESSOR_RESULT_RELATIVE_PATH,
            "role": "predecessor_result",
            "sha256": PREDECESSOR_RESULT_FILE_SHA256,
        },
        *[
            dict(source_by_role[role])
            for role in (
                "dynamic_geometry",
                "finalizer",
                "geometry_test",
                "diagnostic_test",
                "preparation_test",
                "finalizer_test",
            )
        ],
        *[
            {
                "path": item["path"],
                "role": "label_shard",
                "sha256": item["sha256"],
            }
            for item in actual
            if item["role"] == "label_shard"
        ],
    ]
    expected_finalizer_template.sort(key=lambda item: (item["path"], item["role"]))
    expected_finalizer_roles = sorted(
        {item["role"] for item in expected_finalizer_template}
    )
    expected_finalizer_counts = {
        role: (40 if role == "label_shard" else 1)
        for role in expected_finalizer_roles
    }
    expected_finalizer_writes = sorted(
        [
            {
                "path": FINAL_RESULT_RELATIVE_PATH,
                "role": "finalizer_output",
                "sha256": None,
            },
            {
                "path": FAILURE_RESULT_RELATIVE_PATH,
                "role": "failure_diagnostic_output",
                "sha256": None,
            },
        ],
        key=lambda item: (item["path"], item["role"]),
    )
    if (
        not exact_equal(
            finalizer_phase["authorized_read_path_template"],
            expected_finalizer_template,
        )
        or finalizer_phase["authorized_read_path_template_sha256"]
        != canonical_json_sha256(expected_finalizer_template)
        or not exact_equal(
            finalizer_phase["expected_roles"], expected_finalizer_roles
        )
        or not exact_equal(
            finalizer_phase["expected_role_byte_open_counts"],
            expected_finalizer_counts,
        )
        or not exact_equal(
            finalizer_phase["expected_label_access"], EXPECTED_LABEL_ACCESS
        )
        or not exact_equal(
            finalizer_phase["authorized_write_paths"],
            expected_finalizer_writes,
        )
        or finalizer_phase["authorized_write_path_set_sha256"]
        != canonical_json_sha256(expected_finalizer_writes)
    ):
        raise ValueError("finalizer phase contract is not globally exact")
    validate_exact_preparation_ledger(
        manifest["preparation_access_ledger"],
        source_by_role=source_by_role,
        human_manifest=human,
        label_records=[item for item in actual if item["role"] == "label_shard"],
    )
    return phase, actual


def validate_template_shards_against_predecessor(
    template_records: list[dict[str, str]], predecessor: dict[str, Any]
) -> None:
    expected = sorted(
        [
            {
                "path": str(entry["path"]),
                "role": "label_shard",
                "sha256": str(entry["sha256"]),
            }
            for entry in _manifest_entries(predecessor)
        ],
        key=lambda item: (item["path"], item["role"]),
    )
    actual = [item for item in template_records if item["role"] == "label_shard"]
    if actual != expected:
        raise ValueError("runner shard allowlist differs from predecessor commitment")


def _lexically_safe_absolute(path_text: str) -> Path:
    if (
        type(path_text) is not str
        or not path_text
        or "\\" in path_text
        or os.path.normpath(path_text) != path_text
        or Path(path_text).as_posix() != path_text
    ):
        raise PermissionError("path_alias_or_escape")
    lexical = Path(path_text)
    if any(part in ("", ".", "..") for part in lexical.parts):
        raise PermissionError("path_alias_or_escape")
    anchored = lexical if lexical.is_absolute() else ROOT / lexical
    if os.path.normpath(str(anchored)) != str(anchored):
        raise PermissionError("path_alias_or_escape")
    try:
        relative = anchored.relative_to(ROOT)
    except ValueError as exc:
        raise PermissionError("outside_repository") from exc
    current = ROOT
    for part in relative.parts:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            break
        if stat.S_ISLNK(mode):
            raise PermissionError("symlink_component")
    return anchored


def _anchor_allowlist(
    records: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    anchored: list[dict[str, str]] = []
    by_path: dict[str, dict[str, str]] = {}
    for item in records:
        absolute = str(_lexically_safe_absolute(item["path"]))
        if absolute in by_path:
            raise ValueError("allowlist path is duplicated")
        record = {"path": absolute, "role": item["role"], "sha256": item["sha256"]}
        anchored.append(record)
        by_path[absolute] = record
    anchored.sort(key=lambda item: (item["path"], item["role"]))
    return anchored, by_path


def _new_ledger(
    *,
    phase_contract: dict[str, Any],
    read_records: list[dict[str, str]],
) -> dict[str, Any]:
    writes = [
        {
            "path": str(_lexically_safe_absolute(CANDIDATE_RELATIVE_PATH)),
            "role": "runner_output",
            "sha256": None,
        }
    ]
    return {
        "schema": LEDGER_SCHEMA,
        "phase": "runner",
        "authorized_read_paths": read_records,
        "authorized_read_path_set_sha256": canonical_json_sha256(read_records),
        "authorized_write_paths": writes,
        "authorized_write_path_set_sha256": canonical_json_sha256(writes),
        "role_byte_open_counts": {
            role: 0 for role in phase_contract["expected_roles"]
        },
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


def validate_exact_preparation_ledger(
    value: object,
    *,
    source_by_role: dict[str, dict[str, str]],
    human_manifest: dict[str, str],
    label_records: list[dict[str, str]],
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != LEDGER_KEYS:
        raise ValueError("preparation ledger key set mismatch")

    def record(path: str, role: str, digest: str) -> dict[str, str]:
        return {
            "path": str(_lexically_safe_absolute(path)),
            "role": role,
            "sha256": digest,
        }

    reads = [
        record(BINDING_RELATIVE_PATH, "binding", BINDING_SHA256),
        record(
            PREDECESSOR_REPORT_RELATIVE_PATH,
            "predecessor_report",
            PREDECESSOR_REPORT_SHA256,
        ),
        record(
            PREDECESSOR_RESULT_RELATIVE_PATH,
            "predecessor_result",
            PREDECESSOR_RESULT_FILE_SHA256,
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
    reads.sort(key=lambda item: (item["path"], item["role"]))
    writes = [
        {
            "path": str(_lexically_safe_absolute(MACHINE_MANIFEST_RELATIVE_PATH)),
            "role": "machine_manifest_output",
            "sha256": None,
        }
    ]
    roles = {item["role"] for item in reads}
    expected = {
        "schema": LEDGER_SCHEMA,
        "phase": "preparation",
        "authorized_read_paths": reads,
        "authorized_read_path_set_sha256": canonical_json_sha256(reads),
        "authorized_write_paths": writes,
        "authorized_write_path_set_sha256": canonical_json_sha256(writes),
        "role_byte_open_counts": {
            role: (0 if role == "label_shard" else 1) for role in roles
        },
        "label_shard_pre_hash_byte_opens": 0,
        "label_shard_post_hash_byte_opens": 0,
        "label_shard_npz_parses": 0,
        "array_decompression_counts": {},
        "selected_label_rows_read": 0,
        "unselected_rows_scored": 0,
        "unselected_rows_retained": 0,
        "metadata_only_shard_stats": 20,
        "denied_attempt_records": [],
        "denied_reason_counts": {reason: 0 for reason in DENIED_REASONS},
        "unexpected_path_attempts": 0,
        "forbidden_role_open_counts": {role: 0 for role in FORBIDDEN_ROLES},
        "all_counts_reconcile": True,
    }
    if not exact_equal(value, expected):
        raise ValueError("preparation ledger differs from the exact derived graph")
    return value


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
    ledger["denied_attempt_records"].append(
        {
            "requested_role": requested_role,
            "declared_role": declared_role,
            "modality": modality,
            "lexical_path": lexical_path,
            "resolved_path": resolved_path,
            "primary_reason": primary,
        }
    )
    ledger["denied_reason_counts"][primary] += 1
    ledger["unexpected_path_attempts"] += 1
    return primary


def _read_authorized(
    path: Path,
    *,
    role: str,
    allowlist: dict[str, dict[str, str]],
    ledger: dict[str, Any],
) -> bytes:
    lexical = str(path)
    try:
        authorized_path = _lexically_safe_absolute(lexical)
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
    payload = authorized_path.read_bytes()
    ledger["role_byte_open_counts"][role] += 1
    if _sha256_bytes(payload) != record["sha256"]:
        record_denial(
            ledger,
            requested_role=role,
            declared_role=str(record["role"]),
            lexical_path=lexical,
            resolved_path=absolute,
            reasons=["hash_mismatch"],
        )
        raise ValueError("hash_mismatch")
    return payload


def validate_npz_inventory(raw: bytes, *, name: str) -> None:
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw), mode="r")
    except (zipfile.BadZipFile, OSError) as exc:
        raise ValueError(f"{name} is not a valid NPZ") from exc
    with archive:
        infos = archive.infolist()
        names = [str(info.filename) for info in infos]
        if len(names) != len(set(names)) or set(names) != REGISTERED_NPZ_MEMBERS:
            raise ValueError(f"{name} inventory differs from the 12 registered members")
        for info in infos:
            member = Path(info.filename)
            if (
                info.is_dir()
                or member.name != info.filename
                or member.is_absolute()
                or ".." in member.parts
                or info.flag_bits & 0x1
            ):
                raise ValueError(f"{name} has an unsafe member")
            offset = int(info.header_offset)
            if offset < 0 or offset + 30 > len(raw) or raw[offset : offset + 4] != b"PK\x03\x04":
                raise ValueError(f"{name} has an invalid local ZIP header")
            flags = int(struct.unpack_from("<H", raw, offset + 6)[0])
            name_length = int(struct.unpack_from("<H", raw, offset + 26)[0])
            extra_length = int(struct.unpack_from("<H", raw, offset + 28)[0])
            start = offset + 30
            end = start + name_length
            if end + extra_length > len(raw):
                raise ValueError(f"{name} has a truncated local ZIP header")
            encoding = "utf-8" if flags & 0x800 else "cp437"
            if raw[start:end].decode(encoding) != info.filename or flags & 0x1:
                raise ValueError(f"{name} central/local member mismatch")


def canonical_target_ranks(predecessor: dict[str, Any]) -> dict[tuple[Any, ...], int]:
    """Bind every selected label identity directly to its canonical output rank."""

    frames = predecessor.get("frame_reports")
    if type(frames) is not list or len(frames) != EXPECTED_TARGET_ROWS:
        raise ValueError("canonical frame report count mismatch")
    ranks: dict[tuple[Any, ...], int] = {}
    for rank, frame in enumerate(frames):
        if type(frame) is not dict or type(frame.get("record_key")) is not dict:
            raise ValueError("canonical frame record key is malformed")
        key = frame["record_key"]
        required = {
            "family",
            "scene_id",
            "global_row",
            "side",
            "label_row",
            "label_shard_sha256",
        }
        if not required.issubset(key):
            raise ValueError("canonical frame record key is incomplete")
        identity = (
            key["family"],
            key["scene_id"],
            key["global_row"],
            key["side"],
            key["label_row"],
            key["label_shard_sha256"],
        )
        if identity in ranks:
            raise ValueError("canonical frame identity is duplicated")
        ranks[identity] = rank
    return ranks


def decode_selected_label_rows(
    raw: bytes,
    *,
    entry: dict[str, Any],
    np: Any,
    rank_by_identity: dict[tuple[Any, ...], int],
    target_buffer: bytearray,
    rank_filled: bytearray,
    lifetime_events: list[dict[str, Any]] | None = None,
) -> tuple[int, dict[str, int]]:
    """Decode one shard directly into the preallocated canonical target buffer."""

    validate_npz_inventory(raw, name="label shard")
    expected_hash = entry.get("sha256")
    if type(expected_hash) is not str or len(expected_hash) != 64:
        raise ValueError("label shard hash is malformed")
    if (
        type(target_buffer) is not bytearray
        or len(target_buffer) != EXPECTED_TARGET_BYTES
        or type(rank_filled) is not bytearray
        or len(rank_filled) != EXPECTED_TARGET_ROWS
    ):
        raise ValueError("canonical target buffer contract mismatch")
    selected_tuples = entry["selected_tuples"]
    requested_by_side: dict[str, list[list[Any]]] = {"current": [], "next": []}
    storage: set[tuple[int, str]] = set()
    for selected in selected_tuples:
        if type(selected) is not list or len(selected) != 5:
            raise ValueError("selected tuple is malformed")
        family, scene_id, global_row, side, label_row = selected
        if side not in requested_by_side or type(label_row) is not int or label_row < 0:
            raise ValueError("selected tuple side/row is malformed")
        if (label_row, side) in storage:
            raise ValueError("selected shard/row/side is duplicated")
        storage.add((label_row, side))
        requested_by_side[side].append(selected)
    selected_count = 0
    decompressions: dict[str, int] = {"current_labels": 0, "next_labels": 0}
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        if set(map(str, archive.files)) != {
            name[:-4] for name in REGISTERED_NPZ_MEMBERS
        }:
            raise ValueError("NPZ array inventory differs from ZIP inventory")
        storage_rows: int | None = None
        for side in ("current", "next"):
            array_name = f"{side}_labels"
            labels = np.asarray(archive[array_name])
            decompressions[array_name] += 1
            if (
                labels.dtype != np.dtype("uint8")
                or labels.ndim != 3
                or labels.shape[1:] != (64, 64)
                or not labels.flags.c_contiguous
                or not np.isin(labels, (0, 1, 2)).all()
            ):
                raise ValueError("label array dtype, shape, order, or classes changed")
            if storage_rows is None:
                storage_rows = int(labels.shape[0])
            elif storage_rows != int(labels.shape[0]):
                raise ValueError("current/next label storage rows disagree")
            for family, scene_id, global_row, tuple_side, row in requested_by_side[side]:
                if row >= labels.shape[0]:
                    raise ValueError("selected label row is outside its shard")
                identity = (
                    family,
                    scene_id,
                    global_row,
                    tuple_side,
                    row,
                    expected_hash,
                )
                rank = rank_by_identity.get(identity)
                if rank is None:
                    raise ValueError("selected tuple is outside canonical frame order")
                if rank_filled[rank]:
                    raise ValueError("selected tuple identity is duplicated")
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
                lifetime_events.append(
                    {
                        "event": "array_released",
                        "array": array_name,
                        "selected_row_copies_retained": 0,
                    }
                )
    if lifetime_events is not None:
        lifetime_events.append(
            {
                "event": "archive_released",
                "selected_row_copies_retained": 0,
            }
        )
    return selected_count, decompressions


def _manifest_entries(predecessor: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = predecessor.get("label_shard_manifest")
    if type(manifest) is not dict or set(manifest) != {
        "entry_count",
        "entries",
        "manifest_sha256",
    }:
        raise ValueError("predecessor label manifest schema mismatch")
    entries = manifest["entries"]
    if (
        type(entries) is not list
        or type(manifest["entry_count"]) is not int
        or manifest["entry_count"] != 20
        or len(entries) != 20
        or manifest["manifest_sha256"] != EXPECTED_LABEL_MANIFEST_SHA256
        or canonical_json_sha256(entries) != EXPECTED_LABEL_MANIFEST_SHA256
    ):
        raise ValueError("predecessor label manifest commitment mismatch")
    return entries


def load_ordered_targets(
    predecessor: dict[str, Any],
    *,
    np: Any,
    allowlist: dict[str, dict[str, str]],
    ledger: dict[str, Any],
    lifetime_events: list[dict[str, Any]] | None = None,
) -> bytes:
    entries = _manifest_entries(predecessor)
    rank_by_identity = canonical_target_ranks(predecessor)
    target_buffer = bytearray(EXPECTED_TARGET_BYTES)
    rank_filled = bytearray(EXPECTED_TARGET_ROWS)
    for entry in entries:
        path = _lexically_safe_absolute(str(entry["path"]))
        expected_hash = str(entry["sha256"])
        record = allowlist.get(str(path))
        expected_record = {
            "path": str(path),
            "role": "label_shard",
            "sha256": expected_hash,
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
        raw = _read_authorized(
            path, role="label_shard", allowlist=allowlist, ledger=ledger
        )
        ledger["label_shard_npz_parses"] += 1
        selected_count, decompressions = decode_selected_label_rows(
            raw,
            entry=entry,
            np=np,
            rank_by_identity=rank_by_identity,
            target_buffer=target_buffer,
            rank_filled=rank_filled,
            lifetime_events=lifetime_events,
        )
        for name, count in decompressions.items():
            ledger["array_decompression_counts"][name] += count
        ledger["selected_label_rows_read"] += selected_count
        ledger["label_shard_post_hash_byte_opens"] += 1
        post = _read_authorized(
            path, role="label_shard", allowlist=allowlist, ledger=ledger
        )
        if post != raw:
            raise ValueError("label shard changed between pre/post hash reads")
        del selected_count, decompressions, post, raw
        if lifetime_events is not None:
            lifetime_events.append(
                {
                    "event": "shard_released",
                    "path": str(path),
                    "selected_row_copies_retained": 0,
                }
            )

    if rank_filled != bytearray([1]) * EXPECTED_TARGET_ROWS:
        raise ValueError("canonical frame has no selected target row")
    targets = bytes(target_buffer)
    if len(targets) != EXPECTED_TARGET_BYTES or _sha256_bytes(targets) != EXPECTED_TARGET_SHA256:
        raise ValueError("canonical selected target byte commitment mismatch")
    return targets


def _load_validated_repo_modules(
    source_payloads: dict[str, bytes],
) -> tuple[Any, Any]:
    """Execute the two reviewed sources without a second source byte open."""

    package = sys.modules.setdefault("lewm", types.ModuleType("lewm"))
    package.__path__ = [str(ROOT / "lewm")]
    benchmarks = sys.modules.setdefault(
        "lewm.benchmarks", types.ModuleType("lewm.benchmarks")
    )
    benchmarks.__path__ = [str(ROOT / "lewm/benchmarks")]

    def execute(name: str, relative: str, payload: bytes) -> Any:
        module = types.ModuleType(name)
        module.__file__ = str(ROOT / relative)
        module.__package__ = name.rpartition(".")[0]
        sys.modules[name] = module
        code = compile(payload, module.__file__, "exec", dont_inherit=True)
        exec(code, module.__dict__)
        return module

    execute(
        "lewm.benchmarks.go2_dynamic_cell_square_projection",
        "lewm/benchmarks/go2_dynamic_cell_square_projection.py",
        source_payloads["dynamic_geometry"],
    )
    core = execute(
        "lewm.benchmarks.go2_dynamic_cell_square_projection_diagnostic",
        "lewm/benchmarks/go2_dynamic_cell_square_projection_diagnostic.py",
        source_payloads["diagnostic_core"],
    )
    import numpy as np

    return core, np


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    values = list(sys.argv[1:] if argv is None else argv)
    if len(values) != 2 or values[0] != "--implementation-manifest-sha256":
        raise SystemExit(
            "runner requires exactly --implementation-manifest-sha256 <hash>"
        )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation-manifest-sha256", required=True)
    args = parser.parse_args(values)
    if not _is_sha256(args.implementation_manifest_sha256):
        parser.error("implementation-manifest-sha256 must be lowercase SHA-256")
    return args


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest_sha256 = str(args.implementation_manifest_sha256)
    output = _lexically_safe_absolute(CANDIDATE_RELATIVE_PATH)
    if output.exists():
        raise FileExistsError("immutable dynamic-projection candidate already exists")

    # Fixed bootstrap authorization for the machine manifest itself.
    manifest_path = _lexically_safe_absolute(MACHINE_MANIFEST_RELATIVE_PATH)
    manifest_raw = manifest_path.read_bytes()
    if _sha256_bytes(manifest_raw) != manifest_sha256:
        raise ValueError("supplied implementation-manifest SHA-256 does not match bytes")
    manifest = strict_json_bytes(manifest_raw, name="machine manifest")
    if manifest_raw != canonical_json_bytes(manifest) + b"\n":
        raise ValueError("machine manifest bytes are not canonical")
    phase_contract, template_records = _validate_machine_manifest(
        manifest, manifest_sha256=manifest_sha256
    )
    read_records, allowlist = _anchor_allowlist(template_records)
    ledger = _new_ledger(phase_contract=phase_contract, read_records=read_records)
    ledger["role_byte_open_counts"]["machine_manifest"] = 1

    source_payloads: dict[str, bytes] = {}
    predecessor: dict[str, Any] | None = None
    for record in read_records:
        role = record["role"]
        if role in ("machine_manifest", "label_shard"):
            continue
        payload = _read_authorized(
            Path(record["path"]), role=role, allowlist=allowlist, ledger=ledger
        )
        if role in ("dynamic_geometry", "diagnostic_core"):
            source_payloads[role] = payload
        elif role == "predecessor_result":
            if _sha256_bytes(payload) != PREDECESSOR_RESULT_FILE_SHA256:
                raise ValueError("predecessor file hash mismatch")
            predecessor = strict_json_bytes(payload, name="predecessor result")
    if predecessor is None or set(source_payloads) != {"dynamic_geometry", "diagnostic_core"}:
        raise ValueError("runner allowlist omitted required reviewed sources")
    validate_template_shards_against_predecessor(template_records, predecessor)
    core, np = _load_validated_repo_modules(source_payloads)
    core.validate_source_map(manifest["source_map"])
    core.validate_phase_ledger(
        manifest["preparation_access_ledger"], expected_phase="preparation"
    )
    core.validate_predecessor_result(predecessor)
    targets = load_ordered_targets(
        predecessor, np=np, allowlist=allowlist, ledger=ledger
    )

    expected_label = phase_contract["expected_label_access"]
    expected_roles = phase_contract["expected_role_byte_open_counts"]
    ledger["all_counts_reconcile"] = (
        ledger["role_byte_open_counts"] == expected_roles
        and all(ledger[key] == expected_label[key] for key in expected_label if key != "array_decompression_counts")
        and ledger["array_decompression_counts"] == expected_label["array_decompression_counts"]
        and not ledger["denied_attempt_records"]
        and ledger["unexpected_path_attempts"] == 0
        and not any(ledger["forbidden_role_open_counts"].values())
        and ledger["unselected_rows_scored"] == 0
        and ledger["unselected_rows_retained"] == 0
    )
    if not ledger["all_counts_reconcile"]:
        raise ValueError("runner access ledger does not reconcile")
    scientific, _remaining = core.compute_scientific_evidence(predecessor, targets)
    del _remaining, targets
    human = manifest["human_manifest"]
    implementations = {
        "human": {
            "path": human["path"],
            "file_sha256": human["file_sha256"],
        },
        "machine": {
            "path": MACHINE_MANIFEST_RELATIVE_PATH,
            "file_sha256": manifest_sha256,
            "content_sha256": manifest["content_sha256"],
        },
    }
    candidate = core.build_candidate(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        implementation_manifests=implementations,
        source_map=manifest["source_map"],
        preparation_access_ledger=manifest["preparation_access_ledger"],
        runner_access_ledger=ledger,
        scientific=scientific,
        label_shard_entries=predecessor["label_shard_manifest"]["entries"],
    )
    payload = core.canonical_json_bytes(candidate) + b"\n"
    _write_exclusive(output, payload)
    print(
        canonical_json_bytes(
            {
                "candidate_content_sha256": candidate["content_sha256"],
                "candidate_file_sha256": _sha256_bytes(payload),
                "candidate_path": str(output),
                "dynamic_unsupported_count": candidate["support"][
                    "dynamic_cell_square_known"
                ]["unsupported_count"],
            }
        ).decode("utf-8"),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
