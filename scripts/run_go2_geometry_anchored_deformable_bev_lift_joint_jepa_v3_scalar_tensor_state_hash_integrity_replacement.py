#!/usr/bin/env python3
"""Run the science-identical joint-JEPA V3 scalar-state hash replacement."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = Path(__file__).resolve()
CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_geometry_anchored_deformable_bev_lift_joint_jepa_v3_"
    "scalar_tensor_state_hash_integrity_replacement.py"
)
FROZEN_V2_RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_geometry_anchored_deformable_bev_lift_joint_jepa_v2_"
    "runtime_import_integrity_replacement.py"
)


def _source_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load source module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _source_module(
    "_lewm_geometry_anchored_joint_jepa_v3_scalar_hash_contract",
    ROOT / CONTRACT_RELATIVE_PATH,
)
if ROOT / contract.RUNNER_RELATIVE_PATH != RUNNER_PATH:
    raise PermissionError("joint-JEPA V3 scalar-hash runner path changed")
_V2 = _source_module(
    "_lewm_geometry_anchored_joint_jepa_v3_scalar_hash_frozen_v2_runner",
    ROOT / FROZEN_V2_RUNNER_RELATIVE_PATH,
)


def _tensor_state_sha256(torch: Any, values: Mapping[str, Any]) -> str:
    """Hash tensor state with the sole V3 scalar-safe byte-view adapter."""

    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        tensor = value.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(
            json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(
            tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        )
    return digest.hexdigest()


def _rebind_inherited_runner() -> None:
    """Bind frozen V2 execution to V3 identity and the sole hash fix."""

    _V2.contract = contract
    _V2.RUNNER_PATH = RUNNER_PATH
    _V2.__file__ = str(RUNNER_PATH)
    _V2._rebind_inherited_runner()
    _V2._V1._tensor_state_sha256 = _tensor_state_sha256


_rebind_inherited_runner()


def run_isolated_import_preflight() -> dict[str, Any]:
    """Retain the reviewed V2 import-root preflight under V3 bindings."""

    _rebind_inherited_runner()
    return _V2.run_isolated_import_preflight()


def parse_args(argv: Sequence[str] | None = None) -> Any:
    _rebind_inherited_runner()
    return _V2.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    _rebind_inherited_runner()
    return _V2.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
