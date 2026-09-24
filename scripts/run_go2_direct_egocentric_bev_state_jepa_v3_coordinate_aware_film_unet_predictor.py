#!/usr/bin/env python3
"""Run Direct BEV V3 through the frozen V2/V1 scientific runner."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_STATE_JEPA_V3_"
    "COORDINATE_AWARE_FILM_UNET_PREDICTOR_PREFLIGHT_JSON"
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
    "_lewm_direct_bev_v3_film_unet_runner_contract",
    ROOT
    / "lewm/benchmarks/"
    "go2_direct_egocentric_bev_state_jepa_v3_"
    "coordinate_aware_film_unet_predictor.py",
)
if ROOT / contract.RUNNER_RELATIVE_PATH != Path(__file__).resolve():
    raise PermissionError("Direct-BEV V3 runner path changed")

_V2 = _source_only_module(
    "_lewm_direct_bev_v3_film_unet_frozen_v2_runner",
    ROOT
    / "scripts/run_go2_direct_egocentric_bev_state_jepa_v2_integrity.py",
)
_FROZEN_V1_SOURCE_ONLY_MODULE = _V2._FROZEN_V1_SOURCE_ONLY_MODULE
_FROZEN_V1_EVALUATE_OBSERVATION_IMPL = _V2._V1._evaluate_observation_impl
V3_MODEL_RUNTIME_MODULE_NAME = (
    "_lewm_direct_egocentric_bev_state_jepa_v3_film_unet_model_runtime"
)


def _source_only_runtime_module(name: str, path: Path) -> Any:
    """Give only the V3 model a noncolliding runtime import identity."""

    if Path(path) == ROOT / contract.MODEL_RELATIVE_PATH:
        name = V3_MODEL_RUNTIME_MODULE_NAME
    return _FROZEN_V1_SOURCE_ONLY_MODULE(name, path)


def _v3_three_logit_bottleneck_exact(model: Any) -> bool:
    """Translate the frozen local-conv integrity check to the V3 predictor."""

    predictor = model.predictor
    return bool(
        model.state_head.out_channels == 3
        and predictor.enc64.conv1.in_channels == 5
        and predictor.residual_head.in_channels == 16
        and predictor.residual_head.out_channels == 3
        and contract.PREDICTOR_CONFIG["inputs_in_order"]
        == [
            "current_three_logit_state",
            "normalized_row_index",
            "normalized_column_index",
        ]
        and contract.PREDICTOR_CONFIG["all_actions"]
        == "encode_once_decode_nine_film_conditions"
    )


def _v3_evaluate_observation_impl(
    runtime: Any,
    model_api: Any,
    model: Any,
    partition: Mapping[str, Any],
    loader: Any,
    selection_pairs: Sequence[Mapping[str, Any]],
    selection_mapping: Mapping[str, Any],
    device: Any,
    *,
    update: int,
    update_zero: Mapping[str, Any] | None,
    prior_gates_passed: bool,
) -> dict[str, Any]:
    """Reuse the observation exactly, adapting only V3's bottleneck receipt."""

    result = _FROZEN_V1_EVALUATE_OBSERVATION_IMPL(
        runtime,
        model_api,
        model,
        partition,
        loader,
        selection_pairs,
        selection_mapping,
        device,
        update=update,
        update_zero=update_zero,
        prior_gates_passed=prior_gates_passed,
    )
    if update == 0:
        metrics = result["metrics"]
        metrics["three_logit_bottleneck_exact"] = (
            _v3_three_logit_bottleneck_exact(model)
        )
        result["gate"] = contract.evaluate_gate(
            update,
            metrics,
            update_zero=update_zero,
            prior_gates_passed=prior_gates_passed,
        )
    return result


def _rebind_inherited_runner() -> None:
    """Bind the frozen execution stack to V3 identities and two API seams."""

    _V2.contract = contract
    _V2.PREFLIGHT_ENVIRONMENT_KEY = PREFLIGHT_ENVIRONMENT_KEY
    _V2.V2_MODEL_RUNTIME_MODULE_NAME = V3_MODEL_RUNTIME_MODULE_NAME
    _V2._source_only_runtime_module = _source_only_runtime_module
    _V2.__file__ = str(Path(__file__).resolve())
    _V2._rebind_inherited_runner()
    _V2._V1._evaluate_observation_impl = _v3_evaluate_observation_impl


_rebind_inherited_runner()


def parse_args(argv: Sequence[str] | None = None) -> Any:
    _rebind_inherited_runner()
    return _V2.parse_args(argv)


def run_parent(
    *,
    review_file_sha256: str,
    authorization_file_sha256: str,
) -> int:
    _rebind_inherited_runner()
    return _V2.run_parent(
        review_file_sha256=review_file_sha256,
        authorization_file_sha256=authorization_file_sha256,
    )


def main(argv: Sequence[str] | None = None) -> int:
    _rebind_inherited_runner()
    return _V2.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
