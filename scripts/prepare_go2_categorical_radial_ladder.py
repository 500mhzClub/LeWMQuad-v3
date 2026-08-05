#!/usr/bin/env python3
"""Freeze the train-only categorical-radial geometry and overfit ladder."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "lewm_worlds"))

from lewm.benchmarks.go2_categorical_radial_factorization import (  # noqa: E402
    audit_exact_cartesian_roundtrip,
    audit_mapping_injectivity,
    build_cartesian_to_polar_bin_mapping,
    geometry_metadata,
)
from lewm.benchmarks.go2_categorical_radial_micro_overfit import (  # noqa: E402
    canonical_json_sha256,
    frame_identity,
    select_ladder_frames,
)
from lewm.benchmarks.go2_physical_micro_overfit import (  # noqa: E402
    PANELS,
    frame_records,
    validate_panel_manifest,
)


SCHEMA = "lewm_go2_categorical_radial_ladder_manifest_v1"
SOURCE_PATHS = {
    "factorization": (
        REPOSITORY_ROOT
        / "lewm/benchmarks/go2_categorical_radial_factorization.py"
    ),
    "ladder_contract": (
        REPOSITORY_ROOT
        / "lewm/benchmarks/go2_categorical_radial_micro_overfit.py"
    ),
    "panel_contract": (
        REPOSITORY_ROOT / "lewm/benchmarks/go2_physical_micro_overfit.py"
    ),
    "preparer": Path(__file__).resolve(),
    "protocol": (
        REPOSITORY_ROOT
        / "docs/lewm_go2_categorical_radial_microfit_protocol_2026-07-10.md"
    ),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes() -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path), "sha256": _sha256_file(path)}
        for name, path in sorted(SOURCE_PATHS.items())
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _shard_contract(
    panels: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, str]:
    shards: dict[str, str] = {}
    for rows in panels.values():
        for row in rows:
            if str(row.get("dataset_role")) != "train":
                raise ValueError("non-train row reached the radial ladder preparer")
            path = str(Path(str(row["label_shard_path"])).resolve())
            expected = str(row["label_shard_sha256"])
            previous = shards.setdefault(path, expected)
            if previous != expected:
                raise ValueError("selected train shard has conflicting hashes")
    return shards


def _audit_labels_and_presence(
    panels: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    mapping: np.ndarray,
) -> tuple[dict[str, Any], dict[tuple[int, str], tuple[bool, bool, bool]]]:
    grouped: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    for panel_name, rows in panels.items():
        for row in rows:
            grouped.setdefault(str(Path(str(row["label_shard_path"])).resolve()), []).append(
                (panel_name, row)
            )

    panel_counts = {
        name: {
            "frame_count": 0,
            "class_counts": [0, 0, 0],
            "known_cell_count": 0,
            "outside_support_known_count": 0,
            "roundtrip_mismatch_count": 0,
        }
        for name in PANELS
    }
    class_presence: dict[tuple[int, str], tuple[bool, bool, bool]] = {}
    label_frame_access_events = 0
    for path, records in sorted(grouped.items()):
        with np.load(path, allow_pickle=False) as shard:
            for panel_name, row in records:
                index = int(row["label_shard_row"])
                for side in ("current", "next"):
                    labels = np.asarray(shard[f"{side}_labels"][index])
                    mask = np.asarray(
                        shard[f"{side}_supervision_mask"][index], dtype=bool
                    )
                    audit = audit_exact_cartesian_roundtrip(labels, mapping=mapping)
                    counts = np.bincount(labels[mask].astype(np.int64), minlength=3)[:3]
                    record = panel_counts[panel_name]
                    record["frame_count"] += 1
                    record["known_cell_count"] += int(audit["known_cartesian_cell_count"])
                    record["outside_support_known_count"] += int(
                        audit["outside_support_known_count"]
                    )
                    record["roundtrip_mismatch_count"] += int(
                        audit["roundtrip_mismatch_count"]
                    )
                    record["class_counts"] = [
                        int(existing + addition)
                        for existing, addition in zip(record["class_counts"], counts)
                    ]
                    if panel_name == "fit":
                        key = (int(row["global_row"]), side)
                        if key in class_presence:
                            raise ValueError("fit frame identity occurs more than once")
                        class_presence[key] = tuple(bool(value) for value in counts)
                    label_frame_access_events += 1

    for panel_name, record in panel_counts.items():
        if int(record["frame_count"]) != 320:
            raise ValueError(f"{panel_name} geometry audit did not read 320 frames")
        if int(record["outside_support_known_count"]) != 0 or int(
            record["roundtrip_mismatch_count"]
        ) != 0:
            raise ValueError(f"{panel_name} failed the exact radial roundtrip")
        record["class_counts"] = {
            name: int(value)
            for name, value in zip(("unknown", "free", "occupied"), record["class_counts"])
        }
        record["exact_roundtrip"] = True
    return (
        {
            "schema": "lewm_go2_categorical_radial_panel_roundtrip_audit_v1",
            "panels": panel_counts,
            "label_frame_access_events": label_frame_access_events,
            "label_shard_npz_open_events": len(grouped),
            "all_960_frames_exact": label_frame_access_events == 960,
        },
        class_presence,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--expected-panel-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; ladder artifacts are immutable")
    expected = str(args.expected_panel_sha256)
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        parser.error("expected-panel-sha256 must be a lowercase SHA-256")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    panel_path = args.panel_manifest.resolve()
    output_path = args.output.resolve()
    expected_panel_sha256 = str(args.expected_panel_sha256)
    panel_file_sha256 = _sha256_file(panel_path)
    if panel_file_sha256 != expected_panel_sha256:
        raise ValueError("panel manifest differs from its precommitted SHA-256")

    source_start = _source_hashes()
    panel = _read_json(panel_path)
    panels = validate_panel_manifest(panel)
    shards = _shard_contract(panels)
    for path, expected in sorted(shards.items()):
        if _sha256_file(Path(path)) != expected:
            raise ValueError(f"selected train shard SHA-256 mismatch: {path}")

    mapping = build_cartesian_to_polar_bin_mapping()
    mapping_audit = audit_mapping_injectivity(mapping)
    roundtrip_audit, class_presence = _audit_labels_and_presence(
        panels,
        mapping=mapping,
    )
    fit_frames = frame_records(panels["fit"])
    if set(map(frame_identity, fit_frames)) != set(class_presence):
        raise ValueError("fit frame records differ from audited label identities")
    ladder = select_ladder_frames(fit_frames, class_presence=class_presence)

    source_end = _source_hashes()
    if source_end != source_start:
        raise RuntimeError("radial ladder sources changed during preparation")
    if _sha256_file(panel_path) != panel_file_sha256:
        raise RuntimeError("panel manifest changed during ladder preparation")
    for path, expected in sorted(shards.items()):
        if _sha256_file(Path(path)) != expected:
            raise RuntimeError(f"selected train shard changed during audit: {path}")

    invocation = (
        list(sys.argv)
        if argv is None
        else [str(Path(__file__).resolve()), *(str(value) for value in argv)]
    )
    core = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "invocation": invocation,
        "inputs": {
            "panel_manifest": {
                "path": str(panel_path),
                "sha256": panel_file_sha256,
                "expected_sha256": expected_panel_sha256,
                "content_sha256": str(panel["content_sha256"]),
                "pre_deserialization_hash_match": True,
            }
        },
        "factorization": geometry_metadata(),
        "mapping_audit": mapping_audit,
        "roundtrip_audit": roundtrip_audit,
        "ladder": ladder,
        "artifact_access_ledger": {
            "runner_input_contains_only_train_rows": True,
            "distinct_train_label_shards_hashed": len(shards),
            "train_label_shard_integrity_hash_passes": 2,
            "train_label_shard_hash_byte_open_events": 2 * len(shards),
            "train_label_shard_npz_opens": len(shards),
            "train_label_frames_read": 960,
            "train_image_byte_opens": 0,
            "checkpoint_selection": {
                "image_byte_opens": 0,
                "label_shard_byte_opens": 0,
                "model_outputs": 0,
            },
            "probability_calibration": {
                "image_byte_opens": 0,
                "label_shard_byte_opens": 0,
                "model_outputs": 0,
            },
            "g2_evaluation": {
                "image_byte_opens": 0,
                "label_shard_byte_opens": 0,
                "model_outputs": 0,
            },
        },
        "source_hashes": source_end,
    }
    payload = {**core, "content_sha256": canonical_json_sha256(core)}
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
                "mapping_cells": mapping_audit["mapped_cartesian_cell_count"],
                "roundtrip_frames": roundtrip_audit["label_frame_access_events"],
                "ladder_prefixes": sorted(map(int, ladder["prefixes"])),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
