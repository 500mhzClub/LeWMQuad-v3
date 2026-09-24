#!/usr/bin/env python3
"""Authority-first launcher for Direct-BEV semantic-anchor state V1."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_SIGNED_BOUNDARY_SEMANTIC_ANCHOR_STATE_V1_"
    "PREFLIGHT_JSON"
)
MODEL_RELATIVE_PATH = (
    "lewm/models/direct_egocentric_bev_signed_boundary_semantic_anchor_state_v1.py"
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
    "_lewm_direct_bev_semantic_anchor_v1_launcher_contract",
    ROOT
    / "lewm/benchmarks/"
    "go2_direct_egocentric_bev_signed_boundary_semantic_anchor_state_v1.py",
)
if (
    ROOT / contract.LAUNCHER_RELATIVE_PATH != Path(__file__).resolve()
    or contract.PREFLIGHT_ENVIRONMENT_KEY != PREFLIGHT_ENVIRONMENT_KEY
    or contract.MODEL_RELATIVE_PATH != MODEL_RELATIVE_PATH
):
    raise PermissionError("Direct-BEV semantic-anchor V1 launcher changed")

# Preserve the reviewed signed-boundary authority stack and replace only its
# source identities.  Import and rebinding remain source-only; tensor-capable
# code is reached only by the inherited authority-first main path.
_SIGNED_BOUNDARY = _source_only_module(
    "_lewm_direct_bev_semantic_anchor_v1_frozen_signed_boundary_launcher",
    ROOT / contract.FROZEN_SIGNED_BOUNDARY_LAUNCHER_RELATIVE_PATH,
)
_FROZEN_SIGNED_BOUNDARY_CONTRACT = _SIGNED_BOUNDARY.contract
if (
    contract.FROZEN_SIGNED_BOUNDARY_CONTRACT_RELATIVE_PATH
    != _FROZEN_SIGNED_BOUNDARY_CONTRACT.CONTRACT_RELATIVE_PATH
    or contract.FROZEN_SIGNED_BOUNDARY_LAUNCHER_RELATIVE_PATH
    != _FROZEN_SIGNED_BOUNDARY_CONTRACT.LAUNCHER_RELATIVE_PATH
):
    raise PermissionError("frozen signed-boundary launcher identity changed")
_LEAF = _SIGNED_BOUNDARY._LEAF


def _authority_owners() -> tuple[Any, ...]:
    return (
        _SIGNED_BOUNDARY,
        *_SIGNED_BOUNDARY._authority_owners(),
    )


def _assert_semantic_anchor_bindings() -> None:
    wrapper = Path(__file__).resolve()
    owners = _authority_owners()
    if any(owner.contract is not contract for owner in owners):
        raise RuntimeError("semantic-anchor contract did not reach authority stack")
    if any(
        owner.PREFLIGHT_ENVIRONMENT_KEY != PREFLIGHT_ENVIRONMENT_KEY
        for owner in owners
    ):
        raise RuntimeError("semantic-anchor preflight did not reach authority stack")
    if any(Path(owner.__file__).resolve() != wrapper for owner in owners):
        raise RuntimeError("semantic-anchor launcher path did not reach stack")
    if _LEAF._V11._BASE.RUNNER_PATH != ROOT / contract.RUNNER_RELATIVE_PATH:
        raise RuntimeError("semantic-anchor runner path did not reach preflight")
    if (
        contract.MODEL_RELATIVE_PATH != MODEL_RELATIVE_PATH
        or _SIGNED_BOUNDARY.MODEL_RELATIVE_PATH != MODEL_RELATIVE_PATH
    ):
        raise RuntimeError("semantic-anchor model source path changed")


def _rebind_inherited_launcher() -> None:
    wrapper = Path(__file__).resolve()
    _SIGNED_BOUNDARY.contract = contract
    _SIGNED_BOUNDARY.PREFLIGHT_ENVIRONMENT_KEY = PREFLIGHT_ENVIRONMENT_KEY
    _SIGNED_BOUNDARY.MODEL_RELATIVE_PATH = MODEL_RELATIVE_PATH
    _SIGNED_BOUNDARY.__file__ = str(wrapper)
    _SIGNED_BOUNDARY._rebind_inherited_launcher()
    _assert_semantic_anchor_bindings()


_rebind_inherited_launcher()
NO_TENSOR_PREFLIGHT_PROGRAM = _LEAF.NO_TENSOR_PREFLIGHT_PROGRAM


def parse_args(argv: Sequence[str] | None = None) -> Any:
    _rebind_inherited_launcher()
    result = _LEAF.parse_args(argv)
    _assert_semantic_anchor_bindings()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    _rebind_inherited_launcher()
    result = _LEAF.main(argv)
    _assert_semantic_anchor_bindings()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
