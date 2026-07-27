#!/usr/bin/env python3
"""Authority-first launcher for semantic-anchor V2 runtime integrity."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
from typing import Any, NoReturn, Sequence


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
    "_lewm_direct_bev_semantic_anchor_v2_runtime_launcher_contract",
    ROOT
    / "lewm/benchmarks/"
    "go2_direct_egocentric_bev_signed_boundary_semantic_anchor_state_v2_"
    "runtime_interpreter_integrity_replacement.py",
)
if ROOT / contract.LAUNCHER_RELATIVE_PATH != WRAPPER_PATH:
    raise PermissionError("semantic-anchor V2 launcher path changed")

PREFLIGHT_ENVIRONMENT_KEY = contract.PREFLIGHT_ENVIRONMENT_KEY
RUNTIME_INTERPRETER_PATH = contract.RUNTIME_INTERPRETER_PATH
RUNTIME_SYS_PREFIX = contract.RUNTIME_SYS_PREFIX
INTERPRETER_HANDOFF_ENVIRONMENT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_SIGNED_BOUNDARY_SEMANTIC_ANCHOR_STATE_V2_"
    "RUNTIME_INTERPRETER_HANDOFF"
)

_V1 = _source_only_module(
    "_lewm_direct_bev_semantic_anchor_v2_frozen_v1_launcher",
    ROOT / contract.FROZEN_V1_LAUNCHER_RELATIVE_PATH,
)
if (
    contract.FROZEN_V1_LAUNCHER_RELATIVE_PATH
    != _V1.contract.LAUNCHER_RELATIVE_PATH
):
    raise PermissionError("frozen semantic-anchor V1 launcher identity changed")


def _rebind_inherited_launcher() -> None:
    """Bind the complete frozen V1 authority stack to fresh V2 identities."""

    _V1.contract = contract
    _V1.PREFLIGHT_ENVIRONMENT_KEY = PREFLIGHT_ENVIRONMENT_KEY
    _V1.MODEL_RELATIVE_PATH = contract.MODEL_RELATIVE_PATH
    _V1.__file__ = str(WRAPPER_PATH)
    _V1._rebind_inherited_launcher()


_rebind_inherited_launcher()
NO_TENSOR_PREFLIGHT_PROGRAM = _V1.NO_TENSOR_PREFLIGHT_PROGRAM


def _runtime_interpreter_matches() -> bool:
    """Check the venv identity lexically; resolving its symlink is incorrect."""

    return bool(
        sys.executable == RUNTIME_INTERPRETER_PATH
        and sys.prefix == RUNTIME_SYS_PREFIX
        and sys.flags.isolated
        and sys.dont_write_bytecode
    )


def _runtime_handoff_environment() -> dict[str, str]:
    """Reuse the frozen launcher's reviewed environment isolation policy."""

    _rebind_inherited_launcher()
    environment = _V1._LEAF._V11._BASE._launch_environment()
    environment[INTERPRETER_HANDOFF_ENVIRONMENT_KEY] = "1"
    return environment


def _exec_reviewed_runtime(raw_argv: Sequence[str]) -> NoReturn:
    """Perform the sole pre-reservation handoff to the reviewed ROCm venv."""

    if os.environ.get(INTERPRETER_HANDOFF_ENVIRONMENT_KEY) == "1":
        raise PermissionError(
            "reviewed runtime-interpreter handoff did not establish its identity"
        )
    environment = _runtime_handoff_environment()
    argv = [
        RUNTIME_INTERPRETER_PATH,
        "-I",
        "-B",
        str(WRAPPER_PATH),
        *list(raw_argv),
    ]
    os.execve(RUNTIME_INTERPRETER_PATH, argv, environment)
    raise AssertionError("runtime-interpreter os.execve unexpectedly returned")


def parse_args(argv: Sequence[str] | None = None) -> Any:
    _rebind_inherited_launcher()
    return _V1.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    _rebind_inherited_launcher()
    if not _runtime_interpreter_matches():
        _exec_reviewed_runtime(raw_argv)
    os.environ.pop(INTERPRETER_HANDOFF_ENVIRONMENT_KEY, None)
    return _V1.main(raw_argv)


if __name__ == "__main__":
    raise SystemExit(main())
