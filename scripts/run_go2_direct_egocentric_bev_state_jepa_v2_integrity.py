#!/usr/bin/env python3
"""Run Direct BEV V2 through the frozen V1 scientific implementation."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_STATE_JEPA_V2_INTEGRITY_PREFLIGHT_JSON"
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
    "_lewm_direct_egocentric_bev_state_jepa_v2_integrity_runner_contract",
    ROOT
    / "lewm/benchmarks/"
    "go2_direct_egocentric_bev_state_jepa_v2_integrity.py",
)
if ROOT / contract.RUNNER_RELATIVE_PATH != Path(__file__).resolve():
    raise PermissionError("Direct-BEV V2 integrity runner path changed")
_V1 = _source_only_module(
    "_lewm_direct_egocentric_bev_state_jepa_v2_integrity_frozen_v1_runner",
    ROOT / contract.FROZEN_V1_RUNNER_RELATIVE_PATH,
)
_FROZEN_V1_SOURCE_ONLY_MODULE = _V1._source_only_module
V2_MODEL_RUNTIME_MODULE_NAME = (
    "_lewm_direct_egocentric_bev_state_jepa_v2_integrity_model_runtime"
)


def _source_only_runtime_module(name: str, path: Path) -> Any:
    """Give only the V2 model a noncolliding runtime import identity."""

    if Path(path) == ROOT / contract.MODEL_RELATIVE_PATH:
        name = V2_MODEL_RUNTIME_MODULE_NAME
    return _FROZEN_V1_SOURCE_ONLY_MODULE(name, path)


def _rebind_inherited_runner() -> None:
    """Bind frozen V1 execution logic to V2 identities and authority."""

    _V1.contract = contract
    _V1.PREFLIGHT_ENVIRONMENT_KEY = PREFLIGHT_ENVIRONMENT_KEY
    _V1.__file__ = str(Path(__file__).resolve())
    _V1._source_only_module = _source_only_runtime_module


_rebind_inherited_runner()


def parse_args(argv: Sequence[str] | None = None) -> Any:
    _rebind_inherited_runner()
    return _V1.parse_args(argv)


def run_parent(
    *,
    review_file_sha256: str,
    authorization_file_sha256: str,
) -> int:
    _rebind_inherited_runner()
    return _V1.run_parent(
        review_file_sha256=review_file_sha256,
        authorization_file_sha256=authorization_file_sha256,
    )


def main(argv: Sequence[str] | None = None) -> int:
    _rebind_inherited_runner()
    return _V1.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
