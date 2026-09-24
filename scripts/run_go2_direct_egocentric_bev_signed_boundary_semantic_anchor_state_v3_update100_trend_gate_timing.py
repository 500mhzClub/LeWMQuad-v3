#!/usr/bin/env python3
"""Run semantic-anchor V3 through the frozen V2 implementation."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
WRAPPER_PATH = Path(__file__).resolve()


def _source_only_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path.relative_to(ROOT).as_posix()}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _source_only_module(
    "_lewm_direct_bev_semantic_anchor_v3_trend_runner_contract",
    ROOT
    / "lewm/benchmarks/"
    "go2_direct_egocentric_bev_signed_boundary_semantic_anchor_state_v3_"
    "update100_trend_gate_timing.py",
)
if ROOT / contract.RUNNER_RELATIVE_PATH != WRAPPER_PATH:
    raise PermissionError("semantic-anchor V3 runner path changed")

PREFLIGHT_ENVIRONMENT_KEY = contract.PREFLIGHT_ENVIRONMENT_KEY
RUNTIME_INTERPRETER_PATH = contract.RUNTIME_INTERPRETER_PATH
RUNTIME_SYS_PREFIX = contract.RUNTIME_SYS_PREFIX
V3_MODEL_RUNTIME_MODULE_NAME = (
    "_lewm_direct_bev_signed_boundary_semantic_anchor_state_v3_model_runtime"
)

_V2 = _source_only_module(
    "_lewm_direct_bev_semantic_anchor_v3_frozen_v2_runner",
    ROOT / contract.FROZEN_V2_RUNNER_RELATIVE_PATH,
)
if contract.FROZEN_V2_RUNNER_RELATIVE_PATH != _V2.contract.RUNNER_RELATIVE_PATH:
    raise PermissionError("frozen semantic-anchor V2 runner identity changed")


def _rebind_inherited_runner() -> None:
    """Bind the frozen V2 runtime stack to V3 source identities."""

    _V2.contract = contract
    _V2.PREFLIGHT_ENVIRONMENT_KEY = PREFLIGHT_ENVIRONMENT_KEY
    _V2.MODEL_RELATIVE_PATH = contract.MODEL_RELATIVE_PATH
    _V2.RUNTIME_INTERPRETER_PATH = RUNTIME_INTERPRETER_PATH
    _V2.RUNTIME_SYS_PREFIX = RUNTIME_SYS_PREFIX
    _V2.V2_MODEL_RUNTIME_MODULE_NAME = V3_MODEL_RUNTIME_MODULE_NAME
    _V2.__file__ = str(WRAPPER_PATH)
    _V2._rebind_inherited_runner()


_rebind_inherited_runner()


def _runtime_interpreter_matches() -> bool:
    return bool(
        sys.executable == RUNTIME_INTERPRETER_PATH
        and sys.prefix == RUNTIME_SYS_PREFIX
        and sys.flags.isolated
        and sys.dont_write_bytecode
    )


def _require_reviewed_runtime() -> None:
    if not _runtime_interpreter_matches():
        raise PermissionError(
            "semantic-anchor V3 runner requires the exact reviewed ROCm "
            "interpreter with python -I -B before reservation"
        )


def parse_args(argv: Sequence[str] | None = None) -> Any:
    _rebind_inherited_runner()
    return _V2.parse_args(argv)


def run_parent(
    *, review_file_sha256: str, authorization_file_sha256: str
) -> int:
    _rebind_inherited_runner()
    _require_reviewed_runtime()
    return _V2.run_parent(
        review_file_sha256=review_file_sha256,
        authorization_file_sha256=authorization_file_sha256,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run_parent(
        review_file_sha256=args.review_sha256,
        authorization_file_sha256=args.authorization_sha256,
    )


if __name__ == "__main__":
    raise SystemExit(main())
