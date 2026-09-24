#!/usr/bin/env python3
"""Run the science-identical event-delta V2 delegation replacement.

V2 changes only the final control transfer.  The complete frozen V1 event
runner is rebound to V2 governance, then execution is dispatched directly to
the deepest inherited runner so that no predecessor wrapper can rebind the
event hooks a second time.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = Path(__file__).resolve()
CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/"
    "go2_geometry_anchored_two_mode_event_delta_joint_jepa_v2_"
    "runtime_delegation_integrity_replacement.py"
)
FROZEN_V1_RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_geometry_anchored_two_mode_event_delta_joint_jepa_v1.py"
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
    "_lewm_two_mode_event_delta_v2_delegation_contract",
    ROOT / CONTRACT_RELATIVE_PATH,
)
if (
    ROOT / contract.RUNNER_RELATIVE_PATH != RUNNER_PATH
    or contract.FROZEN_V1_RUNNER_RELATIVE_PATH
    != FROZEN_V1_RUNNER_RELATIVE_PATH
):
    raise PermissionError("event-delta V2 runner identity changed")

_V1 = _source_module(
    "_lewm_two_mode_event_delta_v2_delegation_frozen_v1_runner",
    ROOT / FROZEN_V1_RUNNER_RELATIVE_PATH,
)
_BASE = _V1._BASE

_EVENT_BINDING_NAMES = (
    "_execute",
    "_terminal_failure",
    "_load_post_reservation_stack",
    "_parameter_receipt",
    "_evaluate_observation",
    "_train_probe",
)
_FROZEN_V1_EVENT_BINDINGS = {
    name: getattr(_V1, name) for name in _EVENT_BINDING_NAMES
}


def _assert_final_runner_bindings() -> None:
    """Fail closed unless V2 identity and every required V1 hook are final."""

    if _BASE is not _V1._RIGID._V3._V2._V1:
        raise RuntimeError("event-delta V2 deepest runner identity changed")
    if _V1.contract is not contract or _BASE.contract is not contract:
        raise RuntimeError("event-delta V2 contract did not reach final runner")
    if (
        Path(_V1.RUNNER_PATH).resolve() != RUNNER_PATH
        or Path(_V1.__file__).resolve() != RUNNER_PATH
        or Path(_BASE.RUNNER_PATH).resolve() != RUNNER_PATH
        or Path(_BASE.CONTRACT_PATH).resolve()
        != ROOT / contract.CONTRACT_RELATIVE_PATH
        or Path(_BASE.__file__).resolve() != RUNNER_PATH
    ):
        raise RuntimeError("event-delta V2 runner paths did not reach final runner")
    for name, expected in _FROZEN_V1_EVENT_BINDINGS.items():
        if getattr(_V1, name) is not expected or getattr(_BASE, name) is not expected:
            raise RuntimeError(f"frozen V1 event runner binding changed: {name}")


def _rebind_inherited_runner() -> None:
    """Rebind all frozen V1 event mechanics once, then verify the leaf."""

    _V1.contract = contract
    _V1.RUNNER_PATH = RUNNER_PATH
    _V1.__file__ = str(RUNNER_PATH)
    _V1._rebind_inherited_runner()
    _assert_final_runner_bindings()


_rebind_inherited_runner()


def run_isolated_import_preflight() -> dict[str, Any]:
    """Report the source-only final-delegation checks without runtime I/O."""

    _rebind_inherited_runner()
    return dict(contract.DELEGATION_PREFLIGHT_REQUIREMENTS)


def parse_args(argv: Sequence[str] | None = None) -> Any:
    _rebind_inherited_runner()
    _assert_final_runner_bindings()
    return _BASE.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    _rebind_inherited_runner()
    _assert_final_runner_bindings()
    return _BASE.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
