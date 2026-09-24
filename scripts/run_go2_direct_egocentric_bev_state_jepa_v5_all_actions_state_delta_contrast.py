#!/usr/bin/env python3
"""Run Direct BEV V5 through the frozen V4 execution adapters."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_STATE_JEPA_V5_"
    "ALL_ACTIONS_STATE_DELTA_CONTRAST_PREFLIGHT_JSON"
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
    "_lewm_direct_bev_v5_state_delta_runner_contract",
    ROOT
    / "lewm/benchmarks/"
    "go2_direct_egocentric_bev_state_jepa_v5_"
    "all_actions_state_delta_contrast.py",
)
if (
    ROOT / contract.RUNNER_RELATIVE_PATH != Path(__file__).resolve()
    or contract.PREFLIGHT_ENVIRONMENT_KEY != PREFLIGHT_ENVIRONMENT_KEY
):
    raise PermissionError("Direct-BEV V5 runner identity changed")

_V4 = _source_only_module(
    "_lewm_direct_bev_v5_state_delta_frozen_v4_runner",
    ROOT / contract.FROZEN_V4_RUNNER_RELATIVE_PATH,
)
V5_MODEL_RUNTIME_MODULE_NAME = (
    "_lewm_direct_bev_v5_all_actions_state_delta_model_runtime"
)


def _rebind_inherited_runner() -> None:
    """Bind V4's frozen adapters and runner stack to only V5 identities."""

    wrapper_path = Path(__file__).resolve()
    _V4.contract = contract
    _V4.PREFLIGHT_ENVIRONMENT_KEY = PREFLIGHT_ENVIRONMENT_KEY
    _V4.V4_MODEL_RUNTIME_MODULE_NAME = V5_MODEL_RUNTIME_MODULE_NAME
    _V4.__file__ = str(wrapper_path)
    _V4._rebind_inherited_runner()

    owners = (_V4, _V4._V3, _V4._V3._V2, _V4._V3._V2._V1)
    if any(owner.contract is not contract for owner in owners):
        raise RuntimeError("V5 contract did not reach the complete runner stack")
    if any(
        owner.PREFLIGHT_ENVIRONMENT_KEY != PREFLIGHT_ENVIRONMENT_KEY
        for owner in owners
    ):
        raise RuntimeError("V5 preflight identity did not reach the runner stack")
    if any(Path(owner.__file__).resolve() != wrapper_path for owner in owners):
        raise RuntimeError("V5 runner path did not reach the runner stack")
    if (
        _V4.V4_MODEL_RUNTIME_MODULE_NAME != V5_MODEL_RUNTIME_MODULE_NAME
        or _V4._V3.V3_MODEL_RUNTIME_MODULE_NAME
        != V5_MODEL_RUNTIME_MODULE_NAME
        or _V4._V3._V2.V2_MODEL_RUNTIME_MODULE_NAME
        != V5_MODEL_RUNTIME_MODULE_NAME
    ):
        raise RuntimeError("V5 model runtime identity was not fully rebound")
    deepest = _V4._V3._V2._V1
    if (
        deepest._gradient_integrity_probe
        is not _V4._v4_gradient_integrity_probe
        or deepest._initialize_model is not _V4._v4_initialize_model
    ):
        raise RuntimeError("V4 integrity adapters were not preserved for V5")
    if deepest.contract.validate_failure_status_chain is not (
        contract.validate_failure_status_chain
    ):
        raise RuntimeError("V5 failure-chain validator was not rebound")


_rebind_inherited_runner()


def parse_args(argv: Sequence[str] | None = None) -> Any:
    _rebind_inherited_runner()
    return _V4.parse_args(argv)


def run_parent(
    *,
    review_file_sha256: str,
    authorization_file_sha256: str,
) -> int:
    _rebind_inherited_runner()
    return _V4.run_parent(
        review_file_sha256=review_file_sha256,
        authorization_file_sha256=authorization_file_sha256,
    )


def main(argv: Sequence[str] | None = None) -> int:
    _rebind_inherited_runner()
    return _V4.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
