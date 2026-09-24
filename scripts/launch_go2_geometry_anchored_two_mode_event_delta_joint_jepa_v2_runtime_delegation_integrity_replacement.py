#!/usr/bin/env python3
"""Authority-first launcher for the event-delta V2 delegation replacement."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = Path(__file__).resolve()
CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/"
    "go2_geometry_anchored_two_mode_event_delta_joint_jepa_v2_"
    "runtime_delegation_integrity_replacement.py"
)
FROZEN_V1_LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_geometry_anchored_two_mode_event_delta_joint_jepa_v1.py"
)


def _source_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load source-only module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _source_module(
    "_lewm_two_mode_event_delta_v2_delegation_launcher_contract",
    ROOT / CONTRACT_RELATIVE_PATH,
)
if (
    ROOT / contract.LAUNCHER_RELATIVE_PATH != LAUNCHER_PATH
    or contract.FROZEN_V1_LAUNCHER_RELATIVE_PATH
    != FROZEN_V1_LAUNCHER_RELATIVE_PATH
):
    raise PermissionError("event-delta V2 launcher identity changed")

_V1 = _source_module(
    "_lewm_two_mode_event_delta_v2_delegation_frozen_v1_launcher",
    ROOT / FROZEN_V1_LAUNCHER_RELATIVE_PATH,
)
_BASE = _V1._RIGID._V3._V2._V1


def _assert_final_launcher_bindings() -> None:
    """Fail closed unless V2 identity reached the deepest launcher."""

    if _V1.contract is not contract or _BASE.contract is not contract:
        raise RuntimeError("event-delta V2 contract did not reach final launcher")
    if (
        Path(_V1.LAUNCHER_PATH).resolve() != LAUNCHER_PATH
        or Path(_V1.__file__).resolve() != LAUNCHER_PATH
        or Path(_BASE.LAUNCHER_PATH).resolve() != LAUNCHER_PATH
        or Path(_BASE.CONTRACT_PATH).resolve()
        != ROOT / contract.CONTRACT_RELATIVE_PATH
        or Path(_BASE.RUNNER_PATH).resolve()
        != ROOT / contract.RUNNER_RELATIVE_PATH
        or Path(_BASE.OUTPUT_ROOT).resolve()
        != ROOT / contract.OUTPUT_ROOT_RELATIVE_PATH
        or Path(_BASE.__file__).resolve() != LAUNCHER_PATH
    ):
        raise RuntimeError(
            "event-delta V2 launcher paths did not reach final launcher"
        )


def _rebind_inherited_launcher() -> None:
    """Rebind the complete frozen V1 launcher once, then verify the leaf."""

    _V1.contract = contract
    _V1.LAUNCHER_PATH = LAUNCHER_PATH
    _V1.__file__ = str(LAUNCHER_PATH)
    _V1._rebind_inherited_launcher()
    _assert_final_launcher_bindings()


_rebind_inherited_launcher()


def parse_args(argv: Sequence[str] | None = None) -> Any:
    _rebind_inherited_launcher()
    _assert_final_launcher_bindings()
    return _BASE.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    _rebind_inherited_launcher()
    _assert_final_launcher_bindings()
    return _BASE.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
