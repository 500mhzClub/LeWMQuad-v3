#!/usr/bin/env python3
"""Run the science-identical Direct-BEV V7 runner-integrity replacement.

V7 changes no model, data, initialization, objective, optimizer, schedule,
gate, or cap.  It installs the complete frozen V6 seam table once and then
delegates directly to the deepest V1 runner leaf.  In particular, it never
enters the V6/V5 public wrappers whose nested rebinds invalidated V6's
initializer-to-optimizer witness.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_STATE_JEPA_V7_"
    "RUNNER_INTEGRITY_REPLACEMENT_PREFLIGHT_JSON"
)
V7_MODEL_RUNTIME_MODULE_NAME = (
    "_lewm_direct_bev_v7_runner_integrity_replacement_model_runtime"
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
    "_lewm_direct_bev_v7_runner_integrity_replacement_contract",
    ROOT
    / "lewm/benchmarks/"
    "go2_direct_egocentric_bev_state_jepa_v7_"
    "runner_integrity_replacement.py",
)
if (
    ROOT / contract.RUNNER_RELATIVE_PATH != Path(__file__).resolve()
    or contract.PREFLIGHT_ENVIRONMENT_KEY != PREFLIGHT_ENVIRONMENT_KEY
):
    raise PermissionError("Direct-BEV V7 runner identity changed")

_V6 = _source_only_module(
    "_lewm_direct_bev_v7_runner_integrity_frozen_v6_runner",
    ROOT / contract.FROZEN_V6_RUNNER_RELATIVE_PATH,
)
_LEAF = _V6._V5._V4._V3._V2._V1

_V6_SEAM_TABLE = (
    ("_initialize_model", "_v6_initialize_model"),
    ("_build_optimizer", "_v6_build_optimizer"),
    ("_gradient_integrity_probe", "_v6_gradient_integrity_probe"),
    ("_evaluate_observation_impl", "_v6_evaluate_observation_impl"),
    ("_train_probe", "_v6_train_probe"),
    ("_write_training_trace", "_v6_write_training_trace"),
    ("_snapshot_model", "_v6_snapshot_model"),
    ("_terminal_failure", "_v6_terminal_failure"),
)


def _assert_full_v6_seam_table() -> None:
    """Fail closed unless every V6 science/accounting seam is installed."""

    for leaf_name, v6_name in _V6_SEAM_TABLE:
        if getattr(_LEAF, leaf_name) is not getattr(_V6, v6_name):
            raise RuntimeError(f"V7 lost frozen V6 seam: {leaf_name}")
    if _LEAF.contract.validate_failure_status_chain is not (
        contract.validate_failure_status_chain
    ):
        raise RuntimeError("V7 failure-chain validator was not rebound")


def _rebind_inherited_runner() -> None:
    """Install V7 identities, then atomically restore all frozen V6 seams."""

    wrapper_path = Path(__file__).resolve()
    _V6.contract = contract
    _V6.PREFLIGHT_ENVIRONMENT_KEY = PREFLIGHT_ENVIRONMENT_KEY
    _V6.V6_MODEL_RUNTIME_MODULE_NAME = V7_MODEL_RUNTIME_MODULE_NAME
    _V6.__file__ = str(wrapper_path)
    _V6._rebind_inherited_runner()

    owners = (
        _V6,
        _V6._V5,
        _V6._V5._V4,
        _V6._V5._V4._V3,
        _V6._V5._V4._V3._V2,
        _LEAF,
    )
    if any(owner.contract is not contract for owner in owners):
        raise RuntimeError("V7 contract did not reach the complete runner stack")
    if any(
        owner.PREFLIGHT_ENVIRONMENT_KEY != PREFLIGHT_ENVIRONMENT_KEY
        for owner in owners
    ):
        raise RuntimeError("V7 preflight identity did not reach the runner stack")
    if any(Path(owner.__file__).resolve() != wrapper_path for owner in owners):
        raise RuntimeError("V7 runner path did not reach the runner stack")
    runtime_names = (
        _V6.V6_MODEL_RUNTIME_MODULE_NAME,
        _V6._V5.V5_MODEL_RUNTIME_MODULE_NAME,
        _V6._V5._V4.V4_MODEL_RUNTIME_MODULE_NAME,
        _V6._V5._V4._V3.V3_MODEL_RUNTIME_MODULE_NAME,
        _V6._V5._V4._V3._V2.V2_MODEL_RUNTIME_MODULE_NAME,
    )
    if any(name != V7_MODEL_RUNTIME_MODULE_NAME for name in runtime_names):
        raise RuntimeError("V7 model runtime identity was not fully rebound")
    _assert_full_v6_seam_table()


_rebind_inherited_runner()


def parse_args(argv: Sequence[str] | None = None) -> Any:
    _rebind_inherited_runner()
    result = _LEAF.parse_args(argv)
    _assert_full_v6_seam_table()
    return result


def run_parent(
    *,
    review_file_sha256: str,
    authorization_file_sha256: str,
) -> int:
    _rebind_inherited_runner()
    result = _LEAF.run_parent(
        review_file_sha256=review_file_sha256,
        authorization_file_sha256=authorization_file_sha256,
    )
    _assert_full_v6_seam_table()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    _rebind_inherited_runner()
    result = _LEAF.main(argv)
    _assert_full_v6_seam_table()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
