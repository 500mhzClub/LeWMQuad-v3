#!/usr/bin/env python3
"""Install and launch the narrow V2 overlay on the exact frozen V1 runner."""
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
    ROOT / "lewm/benchmarks/go2_shared_jepa_v5_matched_training_v2.py",
    "_lewm_go2_shared_jepa_v5_matched_training_v2_contract",
)
base = _load_module(
    ROOT / contract.V1_RUNNER_RELATIVE_PATH,
    "_lewm_go2_shared_jepa_v5_matched_training_v2_frozen_v1_runner",
    expected_sha256=contract.V1_SOURCE_SHA256[contract.V1_RUNNER_RELATIVE_PATH],
)
_BASE_NAMESPACE_SNAPSHOT = MappingProxyType(dict(vars(base)))


def main(argv: Sequence[str] | None = None) -> int:
    installed = contract.install_successor(base, _BASE_NAMESPACE_SNAPSHOT)
    return installed.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
