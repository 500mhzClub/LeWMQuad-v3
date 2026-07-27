#!/usr/bin/env python3
"""Authority-first launcher for the Direct-BEV V11 integrity replacement."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_STATE_JEPA_V11_"
    "NESTED_OBSERVER_GATE_DISPATCH_INTEGRITY_REPLACEMENT_PREFLIGHT_JSON"
)
MODEL_RELATIVE_PATH = (
    "lewm/models/direct_egocentric_bev_state_jepa_v10_"
    "final_class_macro_grounding.py"
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
    "_lewm_direct_bev_v11_nested_observer_gate_dispatch_launcher_contract",
    ROOT
    / "lewm/benchmarks/"
    "go2_direct_egocentric_bev_state_jepa_v11_"
    "nested_observer_gate_dispatch_integrity_replacement.py",
)
if (
    ROOT / contract.LAUNCHER_RELATIVE_PATH != Path(__file__).resolve()
    or contract.PREFLIGHT_ENVIRONMENT_KEY != PREFLIGHT_ENVIRONMENT_KEY
    or contract.MODEL_RELATIVE_PATH != MODEL_RELATIVE_PATH
):
    raise PermissionError("Direct-BEV V11 launcher identity changed")

_V10 = _source_only_module(
    "_lewm_direct_bev_v11_nested_observer_gate_dispatch_frozen_v10_launcher",
    ROOT / contract.FROZEN_V10_LAUNCHER_RELATIVE_PATH,
)
_V9 = _V10._V9
_LEAF = _V10._LEAF


def _authority_owners() -> tuple[Any, ...]:
    return (_V10, *_V10._authority_owners())


def _assert_v11_bindings() -> None:
    wrapper = Path(__file__).resolve()
    _V10._assert_v10_bindings()
    owners = _authority_owners()
    if any(owner.contract is not contract for owner in owners):
        raise RuntimeError("V11 contract did not reach authority stack")
    if any(
        owner.PREFLIGHT_ENVIRONMENT_KEY != PREFLIGHT_ENVIRONMENT_KEY
        for owner in owners
    ):
        raise RuntimeError("V11 preflight identity did not reach authority stack")
    if any(Path(owner.__file__).resolve() != wrapper for owner in owners):
        raise RuntimeError("V11 launcher path did not reach authority stack")
    if _LEAF._V11._BASE.RUNNER_PATH != ROOT / contract.RUNNER_RELATIVE_PATH:
        raise RuntimeError("V11 runner path did not reach preflight base")
    if (
        _V10.MODEL_RELATIVE_PATH != MODEL_RELATIVE_PATH
        or contract.MODEL_RELATIVE_PATH != MODEL_RELATIVE_PATH
    ):
        raise RuntimeError("V11 changed frozen V10 model source identity")


def _rebind_inherited_launcher() -> None:
    wrapper = Path(__file__).resolve()
    _V10.contract = contract
    _V10.PREFLIGHT_ENVIRONMENT_KEY = PREFLIGHT_ENVIRONMENT_KEY
    _V10.__file__ = str(wrapper)
    _V10._rebind_inherited_launcher()
    _assert_v11_bindings()


_rebind_inherited_launcher()
NO_TENSOR_PREFLIGHT_PROGRAM = _LEAF.NO_TENSOR_PREFLIGHT_PROGRAM


def parse_args(argv: Sequence[str] | None = None) -> Any:
    _rebind_inherited_launcher()
    result = _LEAF.parse_args(argv)
    _assert_v11_bindings()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    _rebind_inherited_launcher()
    result = _LEAF.main(argv)
    _assert_v11_bindings()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
