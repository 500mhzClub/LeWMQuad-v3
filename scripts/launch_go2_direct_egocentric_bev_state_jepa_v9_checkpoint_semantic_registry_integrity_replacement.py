#!/usr/bin/env python3
"""Authority-first launcher for the Direct-BEV V9 integrity replacement."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_STATE_JEPA_V9_"
    "CHECKPOINT_SEMANTIC_REGISTRY_INTEGRITY_REPLACEMENT_PREFLIGHT_JSON"
)


def _source_only_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path.relative_to(ROOT).as_posix()}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _source_only_module(
    "_lewm_direct_bev_v9_checkpoint_registry_integrity_launcher_contract",
    ROOT
    / "lewm/benchmarks/"
    "go2_direct_egocentric_bev_state_jepa_v9_"
    "checkpoint_semantic_registry_integrity_replacement.py",
)
if (
    ROOT / contract.LAUNCHER_RELATIVE_PATH != Path(__file__).resolve()
    or contract.PREFLIGHT_ENVIRONMENT_KEY != PREFLIGHT_ENVIRONMENT_KEY
):
    raise PermissionError("Direct-BEV V9 launcher identity changed")

_V8 = _source_only_module(
    "_lewm_direct_bev_v9_checkpoint_registry_frozen_v8_launcher",
    ROOT / contract.FROZEN_V8_LAUNCHER_RELATIVE_PATH,
)
_LEAF = _V8._LEAF


def _rebind_inherited_launcher() -> None:
    """Bind the complete frozen V8 authority stack to V9 identities."""

    wrapper = Path(__file__).resolve()
    _V8.contract = contract
    _V8.PREFLIGHT_ENVIRONMENT_KEY = PREFLIGHT_ENVIRONMENT_KEY
    _V8.__file__ = str(wrapper)
    _V8._rebind_inherited_launcher()
    owners = (
        _V8,
        _V8._V7,
        _V8._V6,
        _V8._V6._V5,
        _V8._V6._V5._V4,
        _V8._V6._V5._V4._V3,
        _V8._V6._V5._V4._V3._V2,
        _LEAF,
        _LEAF._V11,
        _LEAF._V11._BASE,
    )
    if any(owner.contract is not contract for owner in owners):
        raise RuntimeError("V9 contract did not reach authority stack")
    if any(
        owner.PREFLIGHT_ENVIRONMENT_KEY != PREFLIGHT_ENVIRONMENT_KEY
        for owner in owners
    ):
        raise RuntimeError("V9 preflight identity did not reach authority stack")
    if any(Path(owner.__file__).resolve() != wrapper for owner in owners):
        raise RuntimeError("V9 launcher path did not reach authority stack")
    if _LEAF._V11._BASE.RUNNER_PATH != ROOT / contract.RUNNER_RELATIVE_PATH:
        raise RuntimeError("V9 runner path did not reach preflight base")


_rebind_inherited_launcher()
NO_TENSOR_PREFLIGHT_PROGRAM = _LEAF.NO_TENSOR_PREFLIGHT_PROGRAM


def parse_args(argv: Sequence[str] | None = None) -> Any:
    _rebind_inherited_launcher()
    return _LEAF.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    _rebind_inherited_launcher()
    return _LEAF.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
