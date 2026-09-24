#!/usr/bin/env python3
"""Run the science-identical joint-JEPA V2 runtime-import replacement."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = Path(__file__).resolve()
CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_geometry_anchored_deformable_bev_lift_joint_jepa_v2_"
    "runtime_import_integrity_replacement.py"
)
FROZEN_V1_RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_geometry_anchored_deformable_bev_lift_joint_jepa_v1.py"
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
    "_lewm_geometry_anchored_joint_jepa_v2_import_contract",
    ROOT / CONTRACT_RELATIVE_PATH,
)
if ROOT / contract.RUNNER_RELATIVE_PATH != RUNNER_PATH:
    raise PermissionError("joint-JEPA V2 runtime-import runner path changed")
_V1 = _source_module(
    "_lewm_geometry_anchored_joint_jepa_v2_import_frozen_v1_runner",
    ROOT / FROZEN_V1_RUNNER_RELATIVE_PATH,
)


def _canonical_root_entry(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return Path(value).resolve() == ROOT
    except (OSError, RuntimeError):
        return False


def _load_post_reservation_stack(
    sources: Mapping[str, str],
) -> tuple[Any, ...]:
    """Load the unchanged V1 stack while keeping exactly one import root."""

    for relative, expected in sources.items():
        _V1._read_regular(ROOT / relative, expected_sha256=expected)

    original_path = list(sys.path)
    try:
        sys.path[:] = [entry for entry in sys.path if not _canonical_root_entry(entry)]
        sys.path.insert(0, str(ROOT))
        if (
            not sys.path
            or sys.path[0] != str(ROOT)
            or sum(_canonical_root_entry(entry) for entry in sys.path) != 1
        ):
            raise PermissionError("canonical repository import root is not exact")

        matched = _V1._source_module(
            "_lewm_geometry_anchored_joint_jepa_v2_import_matched_runtime",
            _V1.MATCHED_RUNNER_PATH,
        )
        runtime = matched._load_runtime()
        schedule_adapter = _V1._source_module(
            "_lewm_geometry_anchored_joint_jepa_v2_import_schedule_adapter",
            _V1.SCHEDULE_ADAPTER_PATH,
        )
        model_api = _V1._source_module(
            "lewm.models.geometry_anchored_deformable_bev_lift_joint_jepa_v1",
            ROOT / contract.MODEL_RELATIVE_PATH,
        )
    finally:
        sys.path[:] = original_path

    if sys.path != original_path:
        raise PermissionError("post-stack import did not restore sys.path")
    for relative, expected in sources.items():
        _V1._read_regular(ROOT / relative, expected_sha256=expected)
    return matched, runtime, schedule_adapter, model_api


def _rebind_inherited_runner() -> None:
    """Bind the frozen V1 execution body to V2 identity and the sole fix."""

    _V1.contract = contract
    _V1.RUNNER_PATH = RUNNER_PATH
    _V1.CONTRACT_PATH = ROOT / contract.CONTRACT_RELATIVE_PATH
    _V1.__file__ = str(RUNNER_PATH)
    _V1._load_post_reservation_stack = _load_post_reservation_stack


_rebind_inherited_runner()


def run_isolated_import_preflight() -> dict[str, Any]:
    """Exercise the exact corrected stack without data, checkpoint, or GPU I/O."""

    _rebind_inherited_runner()
    before_path = list(sys.path)
    sources = contract.current_source_bindings(ROOT)
    loaded = _load_post_reservation_stack(sources)
    if len(loaded) != 4 or sys.path != before_path:
        raise PermissionError("isolated post-stack import preflight failed")
    return dict(contract.IMPORT_PREFLIGHT_REQUIREMENTS)


def parse_args(argv: Sequence[str] | None = None) -> Any:
    _rebind_inherited_runner()
    return _V1.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    _rebind_inherited_runner()
    return _V1.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
