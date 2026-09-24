#!/usr/bin/env python3
"""Build the fixed train/validation schedule for the recurrent H4 JEPA probe."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.datasets.go2_recurrent_h4_rgb_sequences import (  # noqa: E402
    SCHEMA,
    build_index,
    canonical_row_bytes,
)


DEFAULT_OUTPUT = Path(".generated/go2_recurrent_h4_rgb_sequence_index_v1")
CENSUS_RECEIPT = Path(".generated/go2_recurrent_jepa_main_pool_census_v2/receipt.json")
CENSUS_RECEIPT_BYTES = 54_695
CENSUS_RECEIPT_SHA256 = "aac85f1016dca12e57e0cf612cd51a745becb2941adf361c0b4a752fe10a5408"
CENSUS_SOURCE_BINDING_SHA256 = (
    "0d5ce1c8aae3777a3e1c930959d5985817d92c28ec240ad03ed79121869d4696"
)


def _canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_fresh(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o644,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_census_source_binding(repo_root: Path) -> dict[str, Any]:
    path = repo_root / CENSUS_RECEIPT
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            raw = stream.read()
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    receipt = json.loads(raw)
    if (
        len(raw) != CENSUS_RECEIPT_BYTES
        or _sha256(raw) != CENSUS_RECEIPT_SHA256
        or receipt.get("schema") != "lewm_go2_recurrent_jepa_main_pool_census_v2"
        or receipt.get("decision") != "MAIN_POOL_H4_METADATA_FEASIBLE"
        or receipt.get("totals", {}).get("source_count") != 1_150
        or receipt.get("totals", {}).get("byte_count") != 138_549_246_020
        or receipt.get("identity", {}).get(
            "ordered_source_content_binding_sha256"
        )
        != CENSUS_SOURCE_BINDING_SHA256
    ):
        raise RuntimeError("the frozen V2 census source-content binding changed")
    return {
        "receipt_path": str(CENSUS_RECEIPT),
        "receipt_byte_count": len(raw),
        "receipt_sha256": _sha256(raw),
        "ordered_source_content_binding_sha256": CENSUS_SOURCE_BINDING_SHA256,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve(strict=True)
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir = output_dir.resolve(strict=False)
    if output_dir.exists():
        raise RuntimeError(f"fresh output directory already exists: {output_dir}")
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    started = time.monotonic()

    def progress(done: int, total: int) -> None:
        if done == total or done % 50 == 0:
            print(f"indexed_sources={done}/{total}", flush=True)

    indexes, manifest = build_index(repo_root, workers=args.workers, progress=progress)
    manifest["source"]["frozen_census_v2"] = _load_census_source_binding(repo_root)
    index_payloads = {
        role: b"".join(canonical_row_bytes(window) for window in indexes[role])
        for role in ("train", "val")
    }
    manifest["artifacts"] = {
        f"{role}.jsonl": {
            "row_count": len(indexes[role]),
            "byte_count": len(payload),
            "sha256": _sha256(payload),
        }
        for role, payload in index_payloads.items()
    }
    manifest["elapsed_seconds"] = round(time.monotonic() - started, 3)
    manifest_payload = _canonical_json(manifest)

    output_dir.mkdir(parents=True, exist_ok=False)
    for role in ("train", "val"):
        _write_fresh(output_dir / f"{role}.jsonl", index_payloads[role])
    _write_fresh(output_dir / "manifest.json", manifest_payload)

    receipt = {
        "schema": f"{SCHEMA}_build_receipt",
        "status": "PASS",
        "output_dir": str(output_dir.relative_to(repo_root)),
        "manifest_sha256": _sha256(manifest_payload),
        "train_rows": len(indexes["train"]),
        "val_rows": len(indexes["val"]),
        "elapsed_seconds": manifest["elapsed_seconds"],
    }
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
