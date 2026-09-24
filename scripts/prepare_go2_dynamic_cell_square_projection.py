#!/usr/bin/env python3
"""Prepare the reviewed dynamic-projection implementation manifest.

This phase is metadata-only for label shards.  It never reads an NPZ byte.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import stat as stat_module
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
MACHINE_SCHEMA = "lewm_go2_dynamic_cell_square_projection_implementation_manifest_v1"
LEDGER_SCHEMA = "lewm_go2_dynamic_projection_access_ledger_v1"
BINDING_RELATIVE_PATH = (
    "docs/lewm_go2_n32_dynamic_cell_square_geometry_binding_2026-07-11.md"
)
BINDING_SHA256 = "211043ee3c3200d1fc93febbae73059341aea19560c83f53f3b3bb231bf06e66"
DYNAMIC_GEOMETRY_SHA256 = (
    "ce2bb0d38ed1436635cdd1468ba1dfe1a935fdafdd6dda5adcf37b97a32a74bf"
)
PREDECESSOR_REPORT_RELATIVE_PATH = (
    "docs/lewm_go2_n32_camera_frustum_observability_audit_v2_result_2026-07-11.md"
)
PREDECESSOR_REPORT_SHA256 = (
    "8bfb4c9a8b69f67b3b9e4d6e3b21e9ff89ecaff89a2bab3eb83d759ca4fe6d22"
)
PREDECESSOR_RESULT_RELATIVE_PATH = (
    ".generated/go2_n32_camera_frustum_observability_audit/v2/result.json"
)
PREDECESSOR_RESULT_FILE_SHA256 = (
    "7725ecddf2fa77bb762733fd35df2efd2fb60d4f9aa8ab6fdf2bee660522909e"
)
PREDECESSOR_RESULT_CONTENT_SHA256 = (
    "11420607d2c4f8e79af9214d43bbc6259669ee84c9ccc0aaefd4167cc1d809a1"
)
HUMAN_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_n32_dynamic_cell_square_geometry_implementation_manifest_2026-07-11.md"
)
MACHINE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_n32_dynamic_cell_square_geometry_implementation_manifest_2026-07-11.json"
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
EXPECTED_LABEL_MANIFEST_SHA256 = (
    "998ce5a768029c23c931fbbec730c1fe31b9ed1fe155494fc68f34a0c23d3d1b"
)
EXPECTED_LABEL_SHARDS = 20
EXPECTED_TARGET_SHA256 = (
    "6952c1f9604da1d9fd4c94a3f33deb142451836609b7059970ff6c459737ce05"
)
EXPECTED_TARGET_BYTES = 1_310_720
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
SOURCE_MAP = (
    ("dynamic_geometry", "lewm/benchmarks/go2_dynamic_cell_square_projection.py"),
    (
        "diagnostic_core",
        "lewm/benchmarks/go2_dynamic_cell_square_projection_diagnostic.py",
    ),
    ("preparation", "scripts/prepare_go2_dynamic_cell_square_projection.py"),
    ("runner", "scripts/diagnose_go2_dynamic_cell_square_projection.py"),
    ("finalizer", "scripts/finalize_go2_dynamic_cell_square_projection.py"),
    ("geometry_test", "lewm/tests/test_go2_dynamic_cell_square_projection.py"),
    (
        "diagnostic_test",
        "lewm/tests/test_go2_dynamic_cell_square_projection_diagnostic.py",
    ),
    (
        "preparation_test",
        "lewm/tests/test_prepare_go2_dynamic_cell_square_projection.py",
    ),
    (
        "finalizer_test",
        "lewm/tests/test_finalize_go2_dynamic_cell_square_projection.py",
    ),
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


def validate_utc_timestamp(value: object) -> str:
    if type(value) is not str:
        raise ValueError("created_at_utc must be an exact string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("created_at_utc is not ISO-8601") from exc
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() is None
        or parsed.utcoffset().total_seconds() != 0
        or parsed.isoformat() != value
        or not value.endswith("+00:00")
    ):
        raise ValueError("created_at_utc is not canonical UTC")
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _strict_json_bytes(payload: bytes, *, name: str) -> dict[str, Any]:
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


def _content_hash(record: dict[str, Any], *, name: str) -> str:
    declared = record.get("content_sha256")
    if type(declared) is not str or len(declared) != 64:
        raise ValueError(f"{name} lacks a valid content hash")
    core = dict(record)
    del core["content_sha256"]
    if canonical_json_sha256(core) != declared:
        raise ValueError(f"{name} content hash mismatch")
    return declared


def _relative_record(path: str, role: str, sha256: str) -> dict[str, str]:
    return {"path": path, "role": role, "sha256": sha256}


def _source_map_entries() -> list[dict[str, str]]:
    entries = []
    seen_paths: set[str] = set()
    seen_roles: set[str] = set()
    for role, relative in SOURCE_MAP:
        if role in seen_roles or relative in seen_paths:
            raise ValueError("source map contains duplicate roles or paths")
        seen_roles.add(role)
        seen_paths.add(relative)
        entries.append(_relative_record(relative, role, _sha256_file(ROOT / relative)))
    return entries


def _predecessor_and_shards() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = ROOT / PREDECESSOR_RESULT_RELATIVE_PATH
    raw = path.read_bytes()
    if _sha256_bytes(raw) != PREDECESSOR_RESULT_FILE_SHA256:
        raise ValueError("predecessor result file hash mismatch")
    result = _strict_json_bytes(raw, name="predecessor result")
    if _content_hash(result, name="predecessor result") != PREDECESSOR_RESULT_CONTENT_SHA256:
        raise ValueError("predecessor result content hash mismatch")
    manifest = result.get("label_shard_manifest")
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
        or manifest["entry_count"] != EXPECTED_LABEL_SHARDS
        or len(entries) != EXPECTED_LABEL_SHARDS
        or manifest["manifest_sha256"] != EXPECTED_LABEL_MANIFEST_SHA256
        or canonical_json_sha256(entries) != EXPECTED_LABEL_MANIFEST_SHA256
    ):
        raise ValueError("predecessor label manifest commitment mismatch")
    return result, entries


def _anchored_absolute(path: str) -> str:
    if (
        type(path) is not str
        or not path
        or "\\" in path
        or os.path.normpath(path) != path
        or Path(path).as_posix() != path
    ):
        raise ValueError("path is not raw canonical text")
    lexical = Path(path)
    anchored = lexical if lexical.is_absolute() else ROOT / lexical
    try:
        relative = anchored.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError("path is outside the repository") from exc
    current = ROOT
    for part in relative.parts:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            break
        if stat_module.S_ISLNK(mode):
            raise ValueError("path contains a symlink component")
    return str(anchored)


def _read_template(
    *,
    phase: str,
    source_entries: list[dict[str, str]],
    shard_entries: list[dict[str, Any]],
    human_sha256: str,
) -> list[dict[str, Any]]:
    sources = {entry["role"]: entry for entry in source_entries}
    roles = (
        ("binding", BINDING_RELATIVE_PATH, BINDING_SHA256),
        ("human_manifest", HUMAN_MANIFEST_RELATIVE_PATH, human_sha256),
    )
    records: list[dict[str, Any]] = [
        _relative_record(path, role, digest) for role, path, digest in roles
    ]
    records.append(dict(SELF_TEMPLATE_ENTRY))
    if phase == "finalizer":
        records.append(dict(CANDIDATE_TEMPLATE_ENTRY))
    records.append(
        _relative_record(
            PREDECESSOR_RESULT_RELATIVE_PATH,
            "predecessor_result",
            PREDECESSOR_RESULT_FILE_SHA256,
        )
    )
    source_roles = (
        ("dynamic_geometry", "diagnostic_core", "runner")
        if phase == "runner"
        else ("dynamic_geometry", "finalizer")
    )
    for role in source_roles:
        records.append(dict(sources[role]))
    for role in (
        "geometry_test",
        "diagnostic_test",
        "preparation_test",
        "finalizer_test",
    ):
        records.append(dict(sources[role]))
    for entry in shard_entries:
        records.append(
            _relative_record(
                str(entry["path"]), "label_shard", str(entry["sha256"])
            )
        )
    return sorted(records, key=lambda item: (item["path"], item["role"]))


def instantiate_read_template(
    template: object,
    *,
    verified_manifest_sha256: str,
    manifest_bytes_verified: bool,
) -> list[dict[str, str]]:
    if not manifest_bytes_verified:
        raise ValueError("machine manifest bytes must be verified before substitution")
    if type(template) is not list:
        raise ValueError("authorized read template must be an exact list")
    placeholders = [
        (index, item)
        for index, item in enumerate(template)
        if type(item) is dict and "sha256_source" in item
    ]
    if len(placeholders) != 1 or placeholders[0][1] != SELF_TEMPLATE_ENTRY:
        raise ValueError("authorized read template self placeholder is not exact")
    actual: list[dict[str, str]] = []
    for item in template:
        if item == SELF_TEMPLATE_ENTRY:
            actual.append(
                {
                    "path": MACHINE_MANIFEST_RELATIVE_PATH,
                    "role": "machine_manifest",
                    "sha256": verified_manifest_sha256,
                }
            )
        else:
            if type(item) is not dict or set(item) != {"path", "role", "sha256"}:
                raise ValueError("non-self allowlist template entry is malformed")
            actual.append(dict(item))
    return actual


def _phase_contract(
    *,
    phase: str,
    template: list[dict[str, Any]],
    output_relative_path: str,
) -> dict[str, Any]:
    role_counts = {str(item["role"]): 1 for item in template if item["role"] != "label_shard"}
    role_counts["label_shard"] = 40
    writes = [
        {"path": output_relative_path, "role": f"{phase}_output", "sha256": None}
    ]
    if phase == "finalizer":
        writes.append(
            {
                "path": FAILURE_RESULT_RELATIVE_PATH,
                "role": "failure_diagnostic_output",
                "sha256": None,
            }
        )
    writes.sort(key=lambda item: (item["path"], item["role"]))
    return {
        "authorized_read_path_template": template,
        "authorized_read_path_template_sha256": canonical_json_sha256(template),
        "authorized_write_paths": writes,
        "authorized_write_path_set_sha256": canonical_json_sha256(writes),
        "expected_roles": sorted(set(role_counts)),
        "expected_role_byte_open_counts": role_counts,
        "expected_label_access": {
            "label_shard_pre_hash_byte_opens": 20,
            "label_shard_post_hash_byte_opens": 20,
            "label_shard_npz_parses": 20,
            "array_decompression_counts": {"current_labels": 20, "next_labels": 20},
            "selected_label_rows_read": 320,
            "metadata_only_shard_stats": 0,
        },
    }


def _preparation_ledger(
    *,
    source_entries: list[dict[str, str]],
    shard_entries: list[dict[str, Any]],
    human_sha256: str,
) -> dict[str, Any]:
    reads = [
        _relative_record(BINDING_RELATIVE_PATH, "binding", BINDING_SHA256),
        _relative_record(
            PREDECESSOR_REPORT_RELATIVE_PATH,
            "predecessor_report",
            PREDECESSOR_REPORT_SHA256,
        ),
        _relative_record(
            PREDECESSOR_RESULT_RELATIVE_PATH,
            "predecessor_result",
            PREDECESSOR_RESULT_FILE_SHA256,
        ),
        _relative_record(
            HUMAN_MANIFEST_RELATIVE_PATH, "human_manifest", human_sha256
        ),
        *[dict(entry) for entry in source_entries],
        *[
            _relative_record(str(entry["path"]), "label_shard", str(entry["sha256"]))
            for entry in shard_entries
        ],
    ]
    reads = [
        {**record, "path": _anchored_absolute(record["path"])} for record in reads
    ]
    reads = sorted(reads, key=lambda item: (item["path"], item["role"]))
    writes = [
        {
            "path": _anchored_absolute(MACHINE_MANIFEST_RELATIVE_PATH),
            "role": "machine_manifest_output",
            "sha256": None,
        }
    ]
    role_counts = {str(item["role"]): 1 for item in reads if item["role"] != "label_shard"}
    role_counts["label_shard"] = 0
    return {
        "schema": LEDGER_SCHEMA,
        "phase": "preparation",
        "authorized_read_paths": reads,
        "authorized_read_path_set_sha256": canonical_json_sha256(reads),
        "authorized_write_paths": writes,
        "authorized_write_path_set_sha256": canonical_json_sha256(writes),
        "role_byte_open_counts": role_counts,
        "label_shard_pre_hash_byte_opens": 0,
        "label_shard_post_hash_byte_opens": 0,
        "label_shard_npz_parses": 0,
        "array_decompression_counts": {},
        "selected_label_rows_read": 0,
        "unselected_rows_scored": 0,
        "unselected_rows_retained": 0,
        "metadata_only_shard_stats": len(shard_entries),
        "denied_attempt_records": [],
        "denied_reason_counts": {reason: 0 for reason in DENIED_REASONS},
        "unexpected_path_attempts": 0,
        "forbidden_role_open_counts": {role: 0 for role in FORBIDDEN_ROLES},
        "all_counts_reconcile": True,
    }


def build_machine_manifest(
    *,
    human_manifest_sha256: str,
    created_at_utc: str,
) -> dict[str, Any]:
    validate_utc_timestamp(created_at_utc)
    if type(human_manifest_sha256) is not str or len(human_manifest_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in human_manifest_sha256
    ):
        raise ValueError("human manifest SHA-256 is malformed")
    if _sha256_file(ROOT / BINDING_RELATIVE_PATH) != BINDING_SHA256:
        raise ValueError("execution binding hash mismatch")
    if _sha256_file(ROOT / PREDECESSOR_REPORT_RELATIVE_PATH) != PREDECESSOR_REPORT_SHA256:
        raise ValueError("predecessor report hash mismatch")
    human_path = ROOT / HUMAN_MANIFEST_RELATIVE_PATH
    if _sha256_file(human_path) != human_manifest_sha256:
        raise ValueError("human manifest hash mismatch")
    absence_paths = (
        CANDIDATE_RELATIVE_PATH,
        FINAL_RESULT_RELATIVE_PATH,
        FAILURE_RESULT_RELATIVE_PATH,
    )
    absence_records = [
        {"path": _anchored_absolute(relative), "exists": False}
        for relative in absence_paths
    ]
    if any(Path(record["path"]).exists() for record in absence_records):
        raise FileExistsError("dynamic-projection evidence output already exists")
    predecessor, shard_entries = _predecessor_and_shards()
    del predecessor
    for entry in shard_entries:
        path = Path(_anchored_absolute(str(entry["path"])))
        stat_result = path.lstat()
        if not stat_module.S_ISREG(stat_result.st_mode) or stat_result.st_size <= 0:
            raise ValueError("committed label shard is absent or empty")
    source_entries = _source_map_entries()
    preparation = _preparation_ledger(
        source_entries=source_entries,
        shard_entries=shard_entries,
        human_sha256=human_manifest_sha256,
    )
    return assemble_machine_manifest(
        human_manifest_sha256=human_manifest_sha256,
        created_at_utc=created_at_utc,
        source_entries=source_entries,
        shard_entries=shard_entries,
        preparation_ledger=preparation,
        output_absence={"paths": absence_records, "all_absent": True},
    )


def assemble_machine_manifest(
    *,
    human_manifest_sha256: str,
    created_at_utc: str,
    source_entries: list[dict[str, str]],
    shard_entries: list[dict[str, Any]],
    preparation_ledger: dict[str, Any],
    output_absence: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the machine record from already-reviewed synthetic or real inputs."""

    if (
        type(human_manifest_sha256) is not str
        or len(human_manifest_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in human_manifest_sha256
        )
    ):
        raise ValueError("human manifest SHA-256 is malformed")
    if type(source_entries) is not list or len(source_entries) != len(SOURCE_MAP):
        raise ValueError("source map must contain the exact nine sources")
    for entry, (expected_role, expected_path) in zip(source_entries, SOURCE_MAP):
        if (
            type(entry) is not dict
            or set(entry) != {"path", "role", "sha256"}
            or entry["role"] != expected_role
            or entry["path"] != expected_path
            or type(entry["sha256"]) is not str
            or len(entry["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in entry["sha256"])
        ):
            raise ValueError("source map differs from the exact nine-source contract")
    if source_entries[0]["sha256"] != DYNAMIC_GEOMETRY_SHA256:
        raise ValueError("dynamic geometry source hash mismatch")
    if type(shard_entries) is not list or len(shard_entries) != EXPECTED_LABEL_SHARDS:
        raise ValueError("label shard graph must contain exactly 20 entries")
    for entry in shard_entries:
        if (
            type(entry) is not dict
            or type(entry.get("path")) is not str
            or type(entry.get("sha256")) is not str
            or len(entry["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in entry["sha256"])
        ):
            raise ValueError("label shard graph entry is malformed")
        _anchored_absolute(entry["path"])
    expected_preparation = _preparation_ledger(
        source_entries=source_entries,
        shard_entries=shard_entries,
        human_sha256=human_manifest_sha256,
    )
    if canonical_json_bytes(preparation_ledger) != canonical_json_bytes(
        expected_preparation
    ):
        raise ValueError("preparation access ledger differs from the derived graph")
    expected_absence = {
        "paths": [
            {"path": _anchored_absolute(relative), "exists": False}
            for relative in (
                CANDIDATE_RELATIVE_PATH,
                FINAL_RESULT_RELATIVE_PATH,
                FAILURE_RESULT_RELATIVE_PATH,
            )
        ],
        "all_absent": True,
    }
    if canonical_json_bytes(output_absence) != canonical_json_bytes(expected_absence):
        raise ValueError("output-absence proof differs from the exact output graph")

    runner_template = _read_template(
        phase="runner",
        source_entries=source_entries,
        shard_entries=shard_entries,
        human_sha256=human_manifest_sha256,
    )
    finalizer_template = _read_template(
        phase="finalizer",
        source_entries=source_entries,
        shard_entries=shard_entries,
        human_sha256=human_manifest_sha256,
    )
    source_map = {
        "entries": source_entries,
        "entry_count": len(source_entries),
        "source_map_sha256": canonical_json_sha256(source_entries),
    }
    core = {
        "schema": MACHINE_SCHEMA,
        "created_at_utc": validate_utc_timestamp(created_at_utc),
        "execution_binding": {
            "path": BINDING_RELATIVE_PATH,
            "file_sha256": BINDING_SHA256,
        },
        "human_manifest": {
            "path": HUMAN_MANIFEST_RELATIVE_PATH,
            "file_sha256": human_manifest_sha256,
        },
        "inputs": {
            "predecessor_report": {
                "path": PREDECESSOR_REPORT_RELATIVE_PATH,
                "file_sha256": PREDECESSOR_REPORT_SHA256,
            },
            "predecessor_result": {
                "path": PREDECESSOR_RESULT_RELATIVE_PATH,
                "file_sha256": PREDECESSOR_RESULT_FILE_SHA256,
                "content_sha256": PREDECESSOR_RESULT_CONTENT_SHA256,
            },
            "label_shard_manifest": {
                "entry_count": EXPECTED_LABEL_SHARDS,
                "manifest_sha256": EXPECTED_LABEL_MANIFEST_SHA256,
            },
            "selected_targets": {
                "frame_count": 320,
                "byte_count": EXPECTED_TARGET_BYTES,
                "sha256": EXPECTED_TARGET_SHA256,
            },
        },
        "source_map": source_map,
        "phase_contracts": {
            "runner": _phase_contract(
                phase="runner",
                template=runner_template,
                output_relative_path=CANDIDATE_RELATIVE_PATH,
            ),
            "finalizer": _phase_contract(
                phase="finalizer",
                template=finalizer_template,
                output_relative_path=FINAL_RESULT_RELATIVE_PATH,
            ),
        },
        "preparation_access_ledger": preparation_ledger,
        "output_absence": output_absence,
        "runtime_environment": {
            "python_implementation": sys.implementation.name,
            "python_version": list(sys.version_info[:3]),
            "numpy_version": importlib.metadata.version("numpy"),
        },
    }
    return {**core, "content_sha256": canonical_json_sha256(core)}


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-manifest-sha256", required=True)
    args = parser.parse_args(argv)
    value = str(args.human_manifest_sha256)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        parser.error("human-manifest-sha256 must be lowercase SHA-256")
    return args


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = build_machine_manifest(
        human_manifest_sha256=str(args.human_manifest_sha256),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    output = ROOT / MACHINE_MANIFEST_RELATIVE_PATH
    payload = canonical_json_bytes(manifest) + b"\n"
    _write_exclusive(output, payload)
    print(
        canonical_json_bytes(
            {
                "content_sha256": manifest["content_sha256"],
                "file_sha256": _sha256_bytes(payload),
                "output": str(output),
            }
        ).decode("utf-8"),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
