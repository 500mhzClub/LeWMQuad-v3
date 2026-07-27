#!/usr/bin/env python3
"""Run semantic-anchor V2 through the frozen V1 scientific implementation."""
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
    "_lewm_direct_bev_semantic_anchor_v2_runtime_runner_contract",
    ROOT
    / "lewm/benchmarks/"
    "go2_direct_egocentric_bev_signed_boundary_semantic_anchor_state_v2_"
    "runtime_interpreter_integrity_replacement.py",
)
if ROOT / contract.RUNNER_RELATIVE_PATH != WRAPPER_PATH:
    raise PermissionError("semantic-anchor V2 runner path changed")

PREFLIGHT_ENVIRONMENT_KEY = contract.PREFLIGHT_ENVIRONMENT_KEY
RUNTIME_INTERPRETER_PATH = contract.RUNTIME_INTERPRETER_PATH
RUNTIME_SYS_PREFIX = contract.RUNTIME_SYS_PREFIX
V2_MODEL_RUNTIME_MODULE_NAME = (
    "_lewm_direct_bev_signed_boundary_semantic_anchor_state_v2_model_runtime"
)

_V1 = _source_only_module(
    "_lewm_direct_bev_semantic_anchor_v2_frozen_v1_runner",
    ROOT / contract.FROZEN_V1_RUNNER_RELATIVE_PATH,
)
if (
    contract.FROZEN_V1_RUNNER_RELATIVE_PATH
    != _V1.contract.RUNNER_RELATIVE_PATH
):
    raise PermissionError("frozen semantic-anchor V1 runner identity changed")


def _rebind_inherited_runner() -> None:
    """Bind frozen V1 science, schedule, seams, and receipts to V2 authority."""

    _V1.contract = contract
    _V1.PREFLIGHT_ENVIRONMENT_KEY = PREFLIGHT_ENVIRONMENT_KEY
    _V1.MODEL_RELATIVE_PATH = contract.MODEL_RELATIVE_PATH
    _V1.SEMANTIC_ANCHOR_MODEL_RUNTIME_MODULE_NAME = (
        V2_MODEL_RUNTIME_MODULE_NAME
    )
    _V1.__file__ = str(WRAPPER_PATH)
    _V1._rebind_inherited_runner()


_rebind_inherited_runner()


def _runtime_interpreter_matches() -> bool:
    """Check exact venv identity without resolving its executable symlink."""

    return bool(
        sys.executable == RUNTIME_INTERPRETER_PATH
        and sys.prefix == RUNTIME_SYS_PREFIX
        and sys.flags.isolated
        and sys.dont_write_bytecode
    )


def _require_reviewed_runtime() -> None:
    if not _runtime_interpreter_matches():
        raise PermissionError(
            "semantic-anchor V2 runner requires the exact reviewed ROCm "
            "interpreter with python -I -B before reservation"
        )


def parse_args(argv: Sequence[str] | None = None) -> Any:
    _rebind_inherited_runner()
    return _V1.parse_args(argv)


def run_parent(
    *, review_file_sha256: str, authorization_file_sha256: str
) -> int:
    _rebind_inherited_runner()
    _require_reviewed_runtime()
    return _V1.run_parent(
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
