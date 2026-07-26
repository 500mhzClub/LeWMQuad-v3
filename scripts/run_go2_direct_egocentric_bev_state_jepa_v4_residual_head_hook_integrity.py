#!/usr/bin/env python3
"""Run the science-identical Direct BEV V4 hook-integrity successor."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_STATE_JEPA_V4_"
    "RESIDUAL_HEAD_HOOK_INTEGRITY_PREFLIGHT_JSON"
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
    "_lewm_direct_bev_v4_residual_head_hook_runner_contract",
    ROOT
    / "lewm/benchmarks/"
    "go2_direct_egocentric_bev_state_jepa_v4_residual_head_hook_integrity.py",
)
if ROOT / contract.RUNNER_RELATIVE_PATH != Path(__file__).resolve():
    raise PermissionError("Direct-BEV V4 runner path changed")

_V3 = _source_only_module(
    "_lewm_direct_bev_v4_residual_head_hook_frozen_v3_runner",
    ROOT
    / "scripts/run_go2_direct_egocentric_bev_state_jepa_v3_"
    "coordinate_aware_film_unet_predictor.py",
)
_FROZEN_GRADIENT_INTEGRITY_PROBE = (
    _V3._V2._V1._gradient_integrity_probe
)
_FROZEN_INITIALIZE_MODEL = _V3._V2._V1._initialize_model
V4_MODEL_RUNTIME_MODULE_NAME = (
    "_lewm_direct_bev_v4_residual_head_hook_model_runtime"
)


def _residual_head_hook_witness(model: Any) -> Any:
    """Return the exact once-per-all-actions V3 transition-call witness."""

    predictor = model.predictor
    head = predictor.residual_head
    if (
        predictor.net[-1] is not head
        or head.in_channels != 16
        or head.out_channels != 3
        or tuple(head.kernel_size) != (3, 3)
        or tuple(head.padding) != (1, 1)
        or head.bias is None
    ):
        raise RuntimeError("V4 residual-head hook witness topology changed")
    return head


class _ResidualHeadHookModelView:
    """Delegate all science to the real model; replace only the hook witness."""

    __slots__ = ("_real_model",)

    def __init__(self, real_model: Any) -> None:
        object.__setattr__(self, "_real_model", real_model)

    @property
    def predictor(self) -> Any:
        return _residual_head_hook_witness(self._real_model)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real_model, name)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("V4 gradient-probe model view is immutable")


def _v4_gradient_integrity_probe(
    runtime: Any,
    model: Any,
    partition: Mapping[str, Any],
    batch: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the frozen probe with only its predictor hook witness adapted."""

    view = _ResidualHeadHookModelView(model)
    result = _FROZEN_GRADIENT_INTEGRITY_PROBE(
        runtime,
        view,
        partition,
        batch,
    )
    if result.get("training_objective_call_counts") != {
        "online_state_stack": 3,
        "predictor": 1,
        "target_state_stack": 3,
    }:
        result["six_call_graph_isolation_exact"] = False
    return result


def _v4_initialize_model(
    runtime: Any,
    model_api: Any,
    fit: Any,
    device: Any,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Require the fresh V4 state to reproduce the frozen V3 initial state."""

    model, partition, receipt = _FROZEN_INITIALIZE_MODEL(
        runtime,
        model_api,
        fit,
        device,
    )
    if receipt.get("complete_initial_state_sha256") != (
        contract.FROZEN_V3_INITIAL_MODEL_STATE_SHA256
    ):
        raise RuntimeError("fresh V4 initial model differs from frozen V3")
    return model, partition, receipt


def _rebind_inherited_runner() -> None:
    """Bind frozen V3 science to V4 identities and one integrity witness."""

    _V3.contract = contract
    _V3.PREFLIGHT_ENVIRONMENT_KEY = PREFLIGHT_ENVIRONMENT_KEY
    _V3.V3_MODEL_RUNTIME_MODULE_NAME = V4_MODEL_RUNTIME_MODULE_NAME
    _V3.__file__ = str(Path(__file__).resolve())
    _V3._rebind_inherited_runner()
    _V3._V2._V1._gradient_integrity_probe = _v4_gradient_integrity_probe
    _V3._V2._V1._initialize_model = _v4_initialize_model


_rebind_inherited_runner()


def parse_args(argv: Sequence[str] | None = None) -> Any:
    _rebind_inherited_runner()
    return _V3.parse_args(argv)


def run_parent(
    *,
    review_file_sha256: str,
    authorization_file_sha256: str,
) -> int:
    _rebind_inherited_runner()
    return _V3.run_parent(
        review_file_sha256=review_file_sha256,
        authorization_file_sha256=authorization_file_sha256,
    )


def main(argv: Sequence[str] | None = None) -> int:
    _rebind_inherited_runner()
    return _V3.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
