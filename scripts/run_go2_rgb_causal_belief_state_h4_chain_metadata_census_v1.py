#!/usr/bin/env python3
"""Run the one-shot recurrent-H4 development-metadata census."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_rgb_causal_belief_state_h4_chain_metadata_census_v1.py"
)
CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/"
    "go2_rgb_causal_belief_state_h4_chain_metadata_census_v1.py"
)


def _source_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load source module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _source_module(
    "_lewm_go2_rgb_causal_belief_state_h4_chain_metadata_census_v1_contract",
    ROOT / CONTRACT_RELATIVE_PATH,
)


def _read_regular(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise PermissionError(f"required input is not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(descriptor)


def _source_bindings() -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    for relative_path in (CONTRACT_RELATIVE_PATH, RUNNER_RELATIVE_PATH):
        raw = _read_regular(ROOT / relative_path)
        bindings.append(
            {
                "path": relative_path,
                "file_sha256": hashlib.sha256(raw).hexdigest(),
                "byte_count": len(raw),
            }
        )
    return bindings


def _validate_preregistration() -> None:
    binding = contract.PREREGISTRATION
    raw = _read_regular(ROOT / binding["path"])
    if (
        len(raw) != binding["byte_count"]
        or hashlib.sha256(raw).hexdigest() != binding["file_sha256"]
    ):
        raise PermissionError("census preregistration binding changed")


def _reserve_output_root() -> Path:
    output_path = ROOT / contract.OUTPUT_PATH
    output_root = output_path.parent
    if output_path.name != "receipt.json":
        raise PermissionError("census output path changed")
    os.mkdir(output_root, 0o755)
    return output_path


def _write_all(descriptor: int, raw: bytes) -> None:
    offset = 0
    while offset < len(raw):
        written = os.write(descriptor, raw[offset:])
        if written <= 0:
            raise OSError("exclusive receipt write made no progress")
        offset += written


def _publish_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    validated = contract.validate_receipt(dict(receipt))
    raw = contract.canonical_json_bytes(validated) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    try:
        _write_all(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _access_receipt(*, manifest_open_count: int, pairs_open_count: int) -> dict[str, Any]:
    access = contract.default_access_receipt()
    access["allowed"] = {
        "manifest_open_count": manifest_open_count,
        "pairs_open_count": pairs_open_count,
    }
    return access


def run_parent(*, authorization_file_sha256: str) -> int:
    _validate_preregistration()
    source_bindings = _source_bindings()
    authorization_raw = _read_regular(ROOT / contract.AUTHORIZATION_PATH)
    contract.validate_authorization(
        authorization_raw,
        expected_file_sha256=authorization_file_sha256,
        source_bindings=source_bindings,
    )

    # The exact result namespace is consumed before either generated metadata
    # input is opened.  Any later failure is terminal and cannot be retried.
    output_path = _reserve_output_root()
    manifest_raw: bytes | None = None
    pairs_raw: bytes | None = None
    manifest_open_count = 0
    pairs_open_count = 0
    try:
        manifest_raw = _read_regular(ROOT / contract.MANIFEST_PATH)
        manifest_open_count = 1
        pairs_raw = _read_regular(ROOT / contract.PAIRS_PATH)
        pairs_open_count = 1
        census = contract.census_from_raw(manifest_raw, pairs_raw)
        receipt = contract.build_receipt(
            census,
            access=_access_receipt(
                manifest_open_count=manifest_open_count,
                pairs_open_count=pairs_open_count,
            ),
            work=contract.default_work_receipt(),
        )
    except BaseException:
        receipt = contract.build_stop_receipt(
            input_bindings=None,
            access=_access_receipt(
                manifest_open_count=manifest_open_count,
                pairs_open_count=pairs_open_count,
            ),
            work=contract.default_work_receipt(),
        )
        return_code = 2
    else:
        return_code = (
            0 if receipt["decision"] == "H4_METADATA_FEASIBLE" else 2
        )
    _publish_receipt(output_path, receipt)
    return return_code


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--authorization-sha256")
    args = parser.parse_args(argv)
    if not args.run or not contract.is_sha256(args.authorization_sha256):
        parser.error("--run and one exact authorization SHA-256 are required")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run_parent(authorization_file_sha256=args.authorization_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
