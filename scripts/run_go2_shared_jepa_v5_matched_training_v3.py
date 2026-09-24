#!/usr/bin/env python3
"""Install V2, add the narrow V3 scalar fix, then launch frozen V1."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]


def _load_module(path: Path, name: str, *, expected_sha256: str | None = None) -> Any:
    if path.is_symlink() or not path.is_file():
        raise PermissionError(f"module is not a regular file: {path}")
    if expected_sha256 is not None and hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise PermissionError(f"frozen module changed: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _load_module(
    ROOT / "lewm/benchmarks/go2_shared_jepa_v5_matched_training_v3.py",
    "_lewm_go2_shared_jepa_v5_matched_training_v3_contract",
)
predecessor = _load_module(
    ROOT / contract.V2_RUNNER_RELATIVE_PATH,
    "_lewm_go2_shared_jepa_v5_matched_training_v3_frozen_v2_runner",
    expected_sha256=contract.V2_SOURCE_SHA256[contract.V2_RUNNER_RELATIVE_PATH],
)


def install() -> Any:
    installed_v2 = predecessor.contract.install_successor(
        predecessor.base, predecessor._BASE_NAMESPACE_SNAPSHOT
    )
    snapshot = MappingProxyType(dict(vars(installed_v2)))
    return contract.install_successor(installed_v2, snapshot, predecessor.contract)


def main(argv: Sequence[str] | None = None) -> int:
    return install().main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
