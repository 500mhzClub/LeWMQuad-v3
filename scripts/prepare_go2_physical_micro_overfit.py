#!/usr/bin/env python3
"""Prepare a content-addressed, train-only Go2 micro-overfit panel."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from lewm.benchmarks.go2_physical_micro_overfit import (  # noqa: E402
    PANEL_SCHEMA,
    ROWS_PER_FAMILY_PANEL,
    SELECTION_SEED,
    SELECTION_UNIT,
    canonical_json_sha256,
    select_train_only_panels,
    validate_panel_manifest,
)


DATASET_SCHEMA = "lewm_go2_paired_navigation_dataset_v3"
SOURCE_PATHS = {
    "contract": REPOSITORY_ROOT / "lewm/benchmarks/go2_physical_micro_overfit.py",
    "execution_contract": (
        REPOSITORY_ROOT
        / "docs/lewm_go2_generalization_execution_contract_2026-07-09.md"
    ),
    "micro_overfit_protocol": (
        REPOSITORY_ROOT
        / "docs/lewm_go2_physical_micro_overfit_protocol_2026-07-10.md"
    ),
    "preparer": Path(__file__).resolve(),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"row is not an object: {path}:{line_number}")
            rows.append(value)
    return rows


def _source_hashes() -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path), "sha256": _sha256_file(path)}
        for name, path in sorted(SOURCE_PATHS.items())
    }


def _git_snapshot() -> dict[str, Any]:
    head = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ("git", "status", "--short"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.rstrip()
    diff = subprocess.run(
        ("git", "diff", "--binary", "--no-ext-diff"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return {
        "head": head,
        "status_short": status,
        "tracked_dirty_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "tracked_dirty_diff_bytes": len(diff),
    }


def _validated_manifest(payload: Mapping[str, Any]) -> tuple[dict[str, str], Path]:
    if payload.get("schema") != DATASET_SCHEMA:
        raise ValueError("micro-overfit preparer requires paired dataset schema v3")
    role_contract = payload.get("scene_roles")
    if not isinstance(role_contract, Mapping) or role_contract.get("schema") != (
        "lewm_go2_family_scene_roles_v1"
    ):
        raise ValueError("dataset lacks a direct family scene-role contract")
    assignments = role_contract.get("assignments")
    if not isinstance(assignments, Mapping):
        raise ValueError("dataset scene-role contract lacks assignments")
    normalized = {str(scene): str(role) for scene, role in assignments.items()}
    if canonical_json_sha256(normalized) != str(
        role_contract.get("assignments_sha256", "")
    ):
        raise ValueError("dataset role assignment content hash mismatch")
    index = payload.get("index")
    if not isinstance(index, Mapping):
        raise ValueError("dataset manifest lacks its row index")
    index_path = Path(str(index.get("path", ""))).resolve()
    if _sha256_file(index_path) != str(index.get("sha256", "")):
        raise ValueError("dataset row-index SHA-256 mismatch")
    return normalized, index_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--expected-dataset-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-seed", default=SELECTION_SEED)
    parser.add_argument(
        "--rows-per-family-panel", type=int, default=ROWS_PER_FAMILY_PANEL
    )
    parser.add_argument("--selection-unit", default=SELECTION_UNIT)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; panel artifacts are immutable")
    if args.selection_seed != SELECTION_SEED:
        parser.error(f"selection-seed must be exactly {SELECTION_SEED}")
    if args.rows_per_family_panel != ROWS_PER_FAMILY_PANEL:
        parser.error(
            f"rows-per-family-panel must be exactly {ROWS_PER_FAMILY_PANEL}"
        )
    if args.selection_unit != SELECTION_UNIT:
        parser.error(f"selection-unit must be exactly {SELECTION_UNIT}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    invocation_argv = (
        list(sys.argv)
        if argv is None
        else [str(Path(__file__).resolve()), *(str(value) for value in argv)]
    )
    manifest_path = args.dataset_manifest.resolve()
    output_path = args.output.resolve()
    expected_manifest_sha256 = str(args.expected_dataset_manifest_sha256)
    manifest_sha256 = _sha256_file(manifest_path)
    if manifest_sha256 != expected_manifest_sha256:
        raise ValueError("dataset manifest differs from the precommitted SHA-256")

    source_start = _source_hashes()
    git_start = _git_snapshot()
    manifest = _read_json(manifest_path)
    assignments, index_path = _validated_manifest(manifest)
    index_sha256 = _sha256_file(index_path)
    rows = _read_rows(index_path)
    if int(manifest.get("row_count", -1)) != len(rows):
        raise ValueError("dataset row count does not match the row index")
    selected = select_train_only_panels(
        rows,
        assignments,
        seed=str(args.selection_seed),
        rows_per_family_panel=int(args.rows_per_family_panel),
    )

    geometry = manifest.get("geometry_contract")
    render_audit = manifest.get("render_audit_contract")
    if not isinstance(geometry, Mapping) or not isinstance(render_audit, Mapping):
        raise ValueError("dataset manifest lacks geometry or render-audit provenance")
    geometry_path = Path(str(geometry.get("path", ""))).resolve()
    render_audit_path = Path(str(render_audit.get("path", ""))).resolve()
    if _sha256_file(geometry_path) != str(geometry.get("file_sha256", "")):
        raise ValueError("geometry contract file SHA-256 mismatch")
    if _sha256_file(render_audit_path) != str(render_audit.get("file_sha256", "")):
        raise ValueError("render-audit contract file SHA-256 mismatch")

    source_end = _source_hashes()
    if source_end != source_start:
        raise RuntimeError("micro-overfit preparer source changed during execution")
    if _sha256_file(manifest_path) != manifest_sha256:
        raise RuntimeError("dataset manifest changed during panel preparation")
    if _sha256_file(index_path) != index_sha256:
        raise RuntimeError("dataset row index changed during panel preparation")
    if _sha256_file(geometry_path) != str(geometry.get("file_sha256", "")):
        raise RuntimeError("geometry contract changed during panel preparation")
    if _sha256_file(render_audit_path) != str(render_audit.get("file_sha256", "")):
        raise RuntimeError("render-audit contract changed during panel preparation")
    git_end = _git_snapshot()
    core = {
        "schema": PANEL_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "invocation": {
            "argv": invocation_argv,
            "resolved": {
                "dataset_manifest": str(manifest_path),
                "expected_dataset_manifest_sha256": expected_manifest_sha256,
                "output": str(output_path),
                "selection_seed": str(args.selection_seed),
                "rows_per_family_panel": int(args.rows_per_family_panel),
                "selection_unit": str(args.selection_unit),
            },
        },
        "local_grid": manifest["local_grid"],
        "source_camera_projection": render_audit["camera_projection"],
        "inputs": {
            "dataset_manifest": {
                "path": str(manifest_path),
                "sha256": manifest_sha256,
                "expected_sha256": expected_manifest_sha256,
                "pre_deserialization_hash_match": True,
            },
            "dataset_index": {"path": str(index_path), "sha256": index_sha256},
            "geometry_contract": {
                "path": str(geometry_path),
                "file_sha256": str(geometry["file_sha256"]),
                "semantic_sha256": str(geometry.get("sha256", "")),
            },
            "render_audit_contract": {
                "path": str(render_audit_path),
                "file_sha256": str(render_audit["file_sha256"]),
                "content_sha256": str(render_audit.get("content_sha256", "")),
            },
            "scene_role_assignments_sha256": str(
                manifest["scene_roles"]["assignments_sha256"]
            ),
        },
        **selected,
        "artifact_access_ledger": {
            "selection_is_label_independent": True,
            "global_jsonl_full_row_objects_parsed": True,
            "non_train_path_strings_temporarily_materialized_by_json_parser": True,
            "non_train_artifact_paths_emitted": False,
            "non_train_artifact_paths_dereferenced": False,
            "label_shard_byte_opens": 0,
            "image_byte_opens": 0,
            "model_outputs": 0,
            "checkpoint_selection": {
                "label_shard_byte_opens": 0,
                "image_byte_opens": 0,
                "model_outputs": 0,
            },
            "probability_calibration": {
                "label_shard_byte_opens": 0,
                "image_byte_opens": 0,
                "model_outputs": 0,
            },
            "g2_evaluation": {
                "label_shard_byte_opens": 0,
                "image_byte_opens": 0,
                "model_outputs": 0,
            },
        },
        "source_hashes": source_end,
        "git": {"start": git_start, "end": git_end},
    }
    payload = {**core, "content_sha256": canonical_json_sha256(core)}
    validate_panel_manifest(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "file_sha256": _sha256_file(output_path),
                "content_sha256": payload["content_sha256"],
                "panel_rows": {
                    name: int(record["row_count"])
                    for name, record in payload["panels"].items()
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
