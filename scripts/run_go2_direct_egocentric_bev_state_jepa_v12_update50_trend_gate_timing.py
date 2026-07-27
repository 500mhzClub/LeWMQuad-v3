#!/usr/bin/env python3
"""Run the Direct-BEV V12 update-50 trend-gate probe.

V12 retains the complete frozen V11 execution stack and exact V10 model.
The V12 contract owns the update-50 gate delta; this wrapper only rebinds V12
identities and delegates directly to the deepest runner leaf.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_STATE_JEPA_V12_"
    "UPDATE50_TREND_GATE_TIMING_PREFLIGHT_JSON"
)
V10_MODEL_RUNTIME_MODULE_NAME = (
    "_lewm_direct_bev_v10_final_class_macro_grounding_model_runtime"
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
    "_lewm_direct_bev_v12_update50_trend_gate_timing_contract",
    ROOT
    / "lewm/benchmarks/"
    "go2_direct_egocentric_bev_state_jepa_v12_"
    "update50_trend_gate_timing.py",
)
if (
    ROOT / contract.RUNNER_RELATIVE_PATH != Path(__file__).resolve()
    or contract.PREFLIGHT_ENVIRONMENT_KEY != PREFLIGHT_ENVIRONMENT_KEY
    or contract.MODEL_RELATIVE_PATH != MODEL_RELATIVE_PATH
):
    raise PermissionError("Direct-BEV V12 runner identity changed")

_V11 = _source_only_module(
    "_lewm_direct_bev_v12_update50_trend_gate_frozen_v11_runner",
    ROOT / contract.FROZEN_V11_RUNNER_RELATIVE_PATH,
)
_V10 = _V11._V10
_V9 = _V11._V9
_LEAF = _V11._LEAF


def _runner_owners() -> tuple[Any, ...]:
    return (_V11, *_V11._runner_owners())


def _runtime_module_names() -> tuple[str, ...]:
    return _V11._runtime_module_names()


def _assert_v12_bindings() -> None:
    """Fail closed unless V11 mechanics and V12 identities are exact."""

    wrapper = Path(__file__).resolve()
    _V11._assert_v11_bindings()
    expected_v9 = dict(_V9._V9_SEAM_TABLE)
    for name, expected_v8 in _V9._V8._V8_SEAM_TABLE:
        if getattr(_LEAF, name) is not expected_v9.get(name, expected_v8):
            raise RuntimeError(f"V12 changed frozen V9 runner seam: {name}")
    owners = _runner_owners()
    if any(owner.contract is not contract for owner in owners):
        raise RuntimeError("V12 contract did not reach complete runner stack")
    if any(
        owner.PREFLIGHT_ENVIRONMENT_KEY != PREFLIGHT_ENVIRONMENT_KEY
        for owner in owners
    ):
        raise RuntimeError("V12 preflight identity did not reach runner stack")
    if any(Path(owner.__file__).resolve() != wrapper for owner in owners):
        raise RuntimeError("V12 runner path did not reach runner stack")
    if any(
        name != V10_MODEL_RUNTIME_MODULE_NAME
        for name in _runtime_module_names()
    ):
        raise RuntimeError("V12 changed frozen V10 model runtime identity")
    if (
        _V11.V10_MODEL_RUNTIME_MODULE_NAME != V10_MODEL_RUNTIME_MODULE_NAME
        or _V11.MODEL_RELATIVE_PATH != MODEL_RELATIVE_PATH
        or contract.MODEL_RELATIVE_PATH != MODEL_RELATIVE_PATH
        or ROOT / contract.MODEL_RELATIVE_PATH
        != ROOT / MODEL_RELATIVE_PATH
    ):
        raise RuntimeError("V12 changed frozen V10 model source identity")


def _rebind_inherited_runner() -> None:
    wrapper = Path(__file__).resolve()
    _V11.contract = contract
    _V11.PREFLIGHT_ENVIRONMENT_KEY = PREFLIGHT_ENVIRONMENT_KEY
    _V11.V10_MODEL_RUNTIME_MODULE_NAME = V10_MODEL_RUNTIME_MODULE_NAME
    _V11.__file__ = str(wrapper)
    _V11._rebind_inherited_runner()
    _assert_v12_bindings()


_rebind_inherited_runner()


def parse_args(argv: Sequence[str] | None = None) -> Any:
    _rebind_inherited_runner()
    result = _LEAF.parse_args(argv)
    _assert_v12_bindings()
    return result


def run_parent(
    *,
    review_file_sha256: str,
    authorization_file_sha256: str,
) -> int:
    _rebind_inherited_runner()
    _V9._assert_fresh_attempt_receipts()
    result = _LEAF.run_parent(
        review_file_sha256=review_file_sha256,
        authorization_file_sha256=authorization_file_sha256,
    )
    _assert_v12_bindings()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    _rebind_inherited_runner()
    _V9._assert_fresh_attempt_receipts()
    result = _LEAF.main(argv)
    _assert_v12_bindings()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
