#!/usr/bin/env python3
"""Run the authorized Camera V5 environment-recovery attempt."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
_CONTRACT_PATH = (
    ROOT
    / "lewm/benchmarks/"
    "go2_shared_jepa_v5_protected_camera_adaptation_v5_environment_recovery_v1.py"
)
_CONTRACT_MODULE = (
    "lewm.benchmarks."
    "go2_shared_jepa_v5_protected_camera_adaptation_v5_environment_recovery_v1"
)
_V5_RUNNER_PATH = (
    ROOT / "scripts/run_go2_shared_jepa_v5_protected_camera_adaptation_v5.py"
)
_V5_RUNNER_SHA256 = (
    "3640ca35300ca36485487d6529dd352c76900c47018f7043cb165a1a078d72c4"
)


def _load_path(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path.relative_to(ROOT).as_posix()}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if _CONTRACT_MODULE in sys.modules:
    contract = sys.modules[_CONTRACT_MODULE]
else:
    contract = _load_path(
        "_lewm_protected_camera_adaptation_v5_environment_recovery_v1_contract",
        _CONTRACT_PATH,
    )


def _load_exact_v5_runner() -> Any:
    if _V5_RUNNER_PATH.is_symlink() or not _V5_RUNNER_PATH.is_file():
        raise PermissionError("committed protected Camera V5 runner changed")
    raw = _V5_RUNNER_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != _V5_RUNNER_SHA256:
        raise PermissionError("committed protected Camera V5 runner changed")
    return _load_path(
        "_lewm_protected_camera_adaptation_v5_runner_for_environment_recovery_v1",
        _V5_RUNNER_PATH,
    )


_v5 = _load_exact_v5_runner()
_BASE_V5_RUN_PARENT = _v5.run_parent


def run_parent(*, review_file_sha256: str, authorization_file_sha256: str) -> int:
    """Install only the recovery contract around the exact committed V5 runner."""
    original_contract = _v5.contract
    _v5.contract = contract
    try:
        return _BASE_V5_RUN_PARENT(
            review_file_sha256=review_file_sha256,
            authorization_file_sha256=authorization_file_sha256,
        )
    finally:
        _v5.contract = original_contract


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--review-sha256")
    parser.add_argument("--authorization-sha256")
    args = parser.parse_args(argv)
    if (
        not args.run
        or not contract._v1.is_sha256(args.review_sha256)
        or not contract._v1.is_sha256(args.authorization_sha256)
    ):
        parser.error("--run and both exact SHA-256 arguments are required")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run_parent(
        review_file_sha256=args.review_sha256,
        authorization_file_sha256=args.authorization_sha256,
    )


if __name__ == "__main__":
    raise SystemExit(main())
