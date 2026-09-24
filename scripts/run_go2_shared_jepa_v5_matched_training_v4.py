#!/usr/bin/env python3
"""Install exact V3, add the V4 runtime policy, then launch frozen V1."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str, expected: str | None = None) -> Any:
    if path.is_symlink() or not path.is_file() or (expected is not None and hashlib.sha256(path.read_bytes()).hexdigest() != expected):
        raise PermissionError(f"frozen module changed: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _load(ROOT / "lewm/benchmarks/go2_shared_jepa_v5_matched_training_v4.py", "_lewm_matched_training_v4_contract")
predecessor = _load(ROOT / contract.V3_RUNNER_RELATIVE_PATH, "_lewm_matched_training_v4_frozen_v3_runner", contract.V3_SOURCE_SHA256[contract.V3_RUNNER_RELATIVE_PATH])


def install() -> Any:
    installed_v3 = predecessor.install()
    return contract.install_successor(installed_v3, MappingProxyType(dict(vars(installed_v3))), predecessor.contract)


def main(argv: Sequence[str] | None = None) -> int:
    return install().main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
