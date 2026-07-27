#!/usr/bin/env python3
"""Authority-first launcher for Direct-BEV multiprototype state-head V1."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_MULTIPROTOTYPE_STATE_HEAD_V1_"
    "PREFLIGHT_JSON"
)
MODEL_RELATIVE_PATH = (
    "lewm/models/direct_egocentric_bev_multiprototype_state_head_v1.py"
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
    "_lewm_direct_bev_multiprototype_v1_launcher_contract",
    ROOT
    / "lewm/benchmarks/"
    "go2_direct_egocentric_bev_multiprototype_state_head_v1.py",
)
if (
    ROOT / contract.LAUNCHER_RELATIVE_PATH != Path(__file__).resolve()
    or contract.PREFLIGHT_ENVIRONMENT_KEY != PREFLIGHT_ENVIRONMENT_KEY
    or contract.MODEL_RELATIVE_PATH != MODEL_RELATIVE_PATH
):
    raise PermissionError(
        "Direct-BEV multiprototype V1 launcher identity changed"
    )

# V10, V11, and V12 add identity wrappers but no launcher mechanism after V9.
# Reuse the complete V9 authority stack directly and bind it to the new V1
# contract so no retired public runner or launcher is called.
_V9 = _source_only_module(
    "_lewm_direct_bev_multiprototype_v1_frozen_v9_launcher",
    ROOT / contract.FROZEN_V9_LAUNCHER_RELATIVE_PATH,
)
_LEAF = _V9._LEAF


def _authority_owners() -> tuple[Any, ...]:
    return (
        _V9,
        _V9._V8,
        _V9._V8._V7,
        _V9._V8._V6,
        _V9._V8._V6._V5,
        _V9._V8._V6._V5._V4,
        _V9._V8._V6._V5._V4._V3,
        _V9._V8._V6._V5._V4._V3._V2,
        _LEAF,
        _LEAF._V11,
        _LEAF._V11._BASE,
    )


def _assert_multiprototype_bindings() -> None:
    wrapper = Path(__file__).resolve()
    owners = _authority_owners()
    if any(owner.contract is not contract for owner in owners):
        raise RuntimeError(
            "multiprototype V1 contract did not reach authority stack"
        )
    if any(
        owner.PREFLIGHT_ENVIRONMENT_KEY != PREFLIGHT_ENVIRONMENT_KEY
        for owner in owners
    ):
        raise RuntimeError(
            "multiprototype V1 preflight identity did not reach authority stack"
        )
    if any(Path(owner.__file__).resolve() != wrapper for owner in owners):
        raise RuntimeError(
            "multiprototype V1 launcher path did not reach authority stack"
        )
    if _LEAF._V11._BASE.RUNNER_PATH != ROOT / contract.RUNNER_RELATIVE_PATH:
        raise RuntimeError(
            "multiprototype V1 runner path did not reach preflight base"
        )
    if contract.MODEL_RELATIVE_PATH != MODEL_RELATIVE_PATH:
        raise RuntimeError("multiprototype V1 model source path changed")


def _rebind_inherited_launcher() -> None:
    wrapper = Path(__file__).resolve()
    _V9.contract = contract
    _V9.PREFLIGHT_ENVIRONMENT_KEY = PREFLIGHT_ENVIRONMENT_KEY
    _V9.__file__ = str(wrapper)
    _V9._rebind_inherited_launcher()
    _assert_multiprototype_bindings()


_rebind_inherited_launcher()
NO_TENSOR_PREFLIGHT_PROGRAM = _LEAF.NO_TENSOR_PREFLIGHT_PROGRAM


def parse_args(argv: Sequence[str] | None = None) -> Any:
    _rebind_inherited_launcher()
    result = _LEAF.parse_args(argv)
    _assert_multiprototype_bindings()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    _rebind_inherited_launcher()
    result = _LEAF.main(argv)
    _assert_multiprototype_bindings()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
