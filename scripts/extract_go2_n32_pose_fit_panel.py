#!/usr/bin/env python3
"""Publish the frozen N32 fit metadata as a standalone audit input."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
AMENDMENT_SHA256 = "56f29c4f2eb05c726b0b4461352fe89da2639b86bf9341ec3072958720cf7c6d"
SOURCE_PATH = ROOT / ".generated/go2_physical_micro_overfit/patch7_v1/panel.json"
SOURCE_FILE_SHA256 = "c3f44c6b1147efbb6a5fbc2294c6431c72e25da877cab6884972d25c1ffdb16c"
SOURCE_CONTENT_SHA256 = "f3e5198b81ac48c06f6c8e4b21e8bf24d62200e3830b1d6685d949a668349d5f"
FIT_ROWS_SHA256 = "5a75e202c8f7a803aafaad093c7f474137dd2e69f50ecdb7fb4e97765afb659d"
OUTPUT_PATH = ROOT / ".generated/go2_n32_pose_projection_audit/v1/fit_panel.json"
FAMILIES = (
    "open_obstacle_field",
    "rough_local_dynamics",
    "small_enclosed_maze",
    "medium_enclosed_maze",
    "large_enclosed_maze",
)


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _exclusive_atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"fit-only panel already exists: {path}")
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def extract(*, authorization: str) -> dict[str, Any]:
    if authorization != AMENDMENT_SHA256:
        raise PermissionError("fit-panel extraction lacks the frozen amendment")
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"fit-only panel already exists: {OUTPUT_PATH}")

    before = SOURCE_PATH.read_bytes()
    before_sha256 = hashlib.sha256(before).hexdigest()
    if before_sha256 != SOURCE_FILE_SHA256:
        raise ValueError("source panel file SHA-256 mismatch")
    source = json.loads(before)
    if not isinstance(source, dict):
        raise ValueError("source panel must be a JSON object")
    declared = source.get("content_sha256")
    source_core = dict(source)
    source_core.pop("content_sha256", None)
    if (
        declared != SOURCE_CONTENT_SHA256
        or canonical_json_sha256(source_core) != SOURCE_CONTENT_SHA256
    ):
        raise ValueError("source panel canonical content mismatch")
    panels = source.get("panels")
    fit = panels.get("fit") if isinstance(panels, Mapping) else None
    if not isinstance(fit, Mapping):
        raise ValueError("source panel lacks fit metadata")
    rows = fit.get("rows")
    if (
        not isinstance(rows, list)
        or len(rows) != 160
        or int(fit.get("row_count", -1)) != 160
        or int(fit.get("frame_count", -1)) != 320
        or fit.get("rows_sha256") != FIT_ROWS_SHA256
        or canonical_json_sha256(rows) != FIT_ROWS_SHA256
    ):
        raise ValueError("source fit metadata commitment mismatch")
    family_counts = Counter(str(row.get("family")) for row in rows)
    if family_counts != Counter({family: 32 for family in FAMILIES}):
        raise ValueError("source fit family counts changed")
    if any(str(row.get("dataset_role")) != "train" for row in rows):
        raise PermissionError("source fit metadata contains a non-train row")

    after_sha256 = hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest()
    if after_sha256 != before_sha256:
        raise RuntimeError("source panel changed during fit-only extraction")
    core = {
        "schema": "lewm_go2_n32_pose_projection_fit_panel_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_panel": {
            "path": str(SOURCE_PATH.resolve()),
            "file_sha256_before_parse": before_sha256,
            "file_sha256_after_parse": after_sha256,
            "content_sha256": SOURCE_CONTENT_SHA256,
        },
        "amendment_sha256": AMENDMENT_SHA256,
        "family_order": list(FAMILIES),
        "fit": dict(fit),
        "access_ledger": {
            "source_panel_byte_opens": 2,
            "source_panel_parse_count": 1,
            "fit_rows_copied": 160,
            "fit_frames_represented": 320,
            "non_fit_rows_copied": 0,
            "rgb_byte_opens": 0,
            "label_shard_byte_opens": 0,
            "model_checkpoint_or_output_opens": 0,
            "g2_payload_opens": 0,
            "sealed_manifest_or_payload_opens": 0,
        },
        "interpretation_limits": {
            "is_research_result": False,
            "can_pass_n32": False,
            "can_pass_g2": False,
            "can_license_runtime": False,
        },
    }
    result = {**core, "content_sha256": canonical_json_sha256(core)}
    _exclusive_atomic_write(OUTPUT_PATH, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", required=True)
    args = parser.parse_args()
    result = extract(authorization=str(args.authorization))
    print(
        json.dumps(
            {
                "output": str(OUTPUT_PATH),
                "file_sha256": hashlib.sha256(OUTPUT_PATH.read_bytes()).hexdigest(),
                "content_sha256": result["content_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
