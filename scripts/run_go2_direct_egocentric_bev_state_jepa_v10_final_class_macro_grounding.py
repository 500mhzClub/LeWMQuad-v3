#!/usr/bin/env python3
"""Run the Direct-BEV V10 final-class macro-grounding probe.

V10 changes grounding inside its model source only.  This wrapper rebinds the
complete frozen V9 execution stack to V10 identities and delegates directly
to the deepest runner leaf without adding or replacing any runner seam.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_STATE_JEPA_V10_"
    "FINAL_CLASS_MACRO_GROUNDING_PREFLIGHT_JSON"
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
    "_lewm_direct_bev_v10_final_class_macro_grounding_contract",
    ROOT
    / "lewm/benchmarks/"
    "go2_direct_egocentric_bev_state_jepa_v10_"
    "final_class_macro_grounding.py",
)
if (
    ROOT / contract.RUNNER_RELATIVE_PATH != Path(__file__).resolve()
    or contract.PREFLIGHT_ENVIRONMENT_KEY != PREFLIGHT_ENVIRONMENT_KEY
    or contract.MODEL_RELATIVE_PATH != MODEL_RELATIVE_PATH
):
    raise PermissionError("Direct-BEV V10 runner identity changed")

_V9 = _source_only_module(
    "_lewm_direct_bev_v10_final_class_macro_grounding_frozen_v9_runner",
    ROOT / contract.FROZEN_V9_RUNNER_RELATIVE_PATH,
)
_LEAF = _V9._LEAF


def _runner_owners() -> tuple[Any, ...]:
    return (_V9, *_V9._runner_contract_owners())


def _runtime_module_names() -> tuple[str, ...]:
    v8 = _V9._V8
    v6 = v8._V6
    return (
        _V9.V9_MODEL_RUNTIME_MODULE_NAME,
        v8.V8_MODEL_RUNTIME_MODULE_NAME,
        v8._V7.V7_MODEL_RUNTIME_MODULE_NAME,
        v6.V6_MODEL_RUNTIME_MODULE_NAME,
        v6._V5.V5_MODEL_RUNTIME_MODULE_NAME,
        v6._V5._V4.V4_MODEL_RUNTIME_MODULE_NAME,
        v6._V5._V4._V3.V3_MODEL_RUNTIME_MODULE_NAME,
        v6._V5._V4._V3._V2.V2_MODEL_RUNTIME_MODULE_NAME,
    )


def _assert_v10_bindings() -> None:
    """Fail closed unless V9 mechanics and all V10 identities are exact."""

    wrapper = Path(__file__).resolve()
    _V9._assert_v9_seams()
    expected_v9 = dict(_V9._V9_SEAM_TABLE)
    for name, expected_v8 in _V9._V8._V8_SEAM_TABLE:
        if getattr(_LEAF, name) is not expected_v9.get(name, expected_v8):
            raise RuntimeError(f"V10 changed frozen V9 runner seam: {name}")
    owners = _runner_owners()
    if any(owner.contract is not contract for owner in owners):
        raise RuntimeError("V10 contract did not reach complete runner stack")
    if any(
        owner.PREFLIGHT_ENVIRONMENT_KEY != PREFLIGHT_ENVIRONMENT_KEY
        for owner in owners
    ):
        raise RuntimeError("V10 preflight identity did not reach runner stack")
    if any(Path(owner.__file__).resolve() != wrapper for owner in owners):
        raise RuntimeError("V10 runner path did not reach runner stack")
    if any(
        name != V10_MODEL_RUNTIME_MODULE_NAME
        for name in _runtime_module_names()
    ):
        raise RuntimeError("V10 model runtime identity did not reach runner stack")
    if (
        contract.MODEL_RELATIVE_PATH != MODEL_RELATIVE_PATH
        or ROOT / contract.MODEL_RELATIVE_PATH
        != ROOT / MODEL_RELATIVE_PATH
    ):
        raise RuntimeError("V10 model source path changed")


def _rebind_inherited_runner() -> None:
    wrapper = Path(__file__).resolve()
    _V9.contract = contract
    _V9.PREFLIGHT_ENVIRONMENT_KEY = PREFLIGHT_ENVIRONMENT_KEY
    _V9.V9_MODEL_RUNTIME_MODULE_NAME = V10_MODEL_RUNTIME_MODULE_NAME
    _V9.__file__ = str(wrapper)
    _V9._rebind_inherited_runner()
    _assert_v10_bindings()


_rebind_inherited_runner()


def parse_args(argv: Sequence[str] | None = None) -> Any:
    _rebind_inherited_runner()
    result = _LEAF.parse_args(argv)
    _assert_v10_bindings()
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
    _assert_v10_bindings()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    _rebind_inherited_runner()
    _V9._assert_fresh_attempt_receipts()
    result = _LEAF.main(argv)
    _assert_v10_bindings()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
