#!/usr/bin/env python3
"""Run the one authorized Camera V4 tail-depth adaptation attempt."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
_CONTRACT_PATH = ROOT / "lewm/benchmarks/go2_shared_jepa_v5_protected_camera_adaptation_v4.py"
_SPEC = importlib.util.spec_from_file_location("_lewm_protected_camera_adaptation_v4_contract", _CONTRACT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("cannot load protected Camera adaptation V4 contract")
contract = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(contract)


def _load_exact_v3_runner() -> Any:
    relative = contract.V3_RUNNER_RELATIVE_PATH
    path = ROOT / relative
    raw = path.read_bytes()
    if path.is_symlink() or not path.is_file() or hashlib.sha256(raw).hexdigest() != contract.V3_SOURCE_SHA256[relative]:
        raise PermissionError("frozen protected Camera V3 runner changed")
    spec = importlib.util.spec_from_file_location("_lewm_protected_camera_adaptation_v3_runner_for_v4", path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load frozen protected Camera V3 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_v3 = _load_exact_v3_runner()
_v1_runner = _v3._v1_runner
_BASE_V3_RUN_PARENT = _v3.run_parent
_BASE_V3_TRAIN = _v3._train
_BASE_V3_ACCESS_RECEIPT = _v3._access_receipt


def _camera_components(loss: Any) -> dict[str, float]:
    result = {"camera_total": _v1_runner._scalar(loss.total)}
    for side in ("current", "next"):
        frame = getattr(loss, side)
        result.update({
            f"{side}_hierarchical_first_hit_nll": _v1_runner._scalar(frame.hierarchical_first_hit_nll),
            f"{side}_tail_depth_p95_cvar": _v1_runner._scalar(frame.tail_depth_p95_cvar),
            f"{side}_ground_clear_distance_state_balanced_bce": _v1_runner._scalar(frame.ground_clear_distance_state_balanced_bce),
            f"{side}_derived_raster_hierarchical_bce": _v1_runner._scalar(frame.derived_raster_hierarchical_bce.total),
            f"{side}_derived_raster_cell_nll": _v1_runner._scalar(frame.derived_raster_cell_nll),
        })
    return result


def _finite_snapshot(base: Any, runtime: Any):
    def checked(model_runtime: Any, model: Any, output_root: Path, *, update: int, frozen_sha: str) -> dict[str, Any]:
        if model_runtime is not runtime:
            raise RuntimeError("snapshot runtime changed")
        for name, value in model.state_dict().items():
            if (value.is_floating_point() or value.is_complex()) and not bool(runtime.torch.isfinite(value).all().item()):
                raise FloatingPointError(f"checkpoint state became nonfinite: {name}")
        return base(model_runtime, model, output_root, update=update, frozen_sha=frozen_sha)
    return checked


def _train(runtime: Any, trainer: Any, model: Any, head: Sequence[Any], encoder: Sequence[Any], frozen: Sequence[Any], train_pairs: Sequence[Mapping[str, Any]], selection_pairs: Sequence[Mapping[str, Any]], indices: Sequence[int], vocabulary: Sequence[str], commanded: Any, device: Any, output_root: Path):
    """Install exactly one loss-slot replacement around the exact V3 loop."""
    from lewm.models import shared_observable_camera_ray_jepa_v5_protected_camera_adaptation_v4_tail_depth as tail

    originals = (
        runtime.loss_adapter.observable_camera_ray_v4_loss_v4,
        _v1_runner._camera_components,
        _v1_runner._snapshot,
    )
    runtime.loss_adapter.observable_camera_ray_v4_loss_v4 = tail.observable_camera_ray_v4_tail_depth_loss_v4
    _v1_runner._camera_components = _camera_components
    _v1_runner._snapshot = _finite_snapshot(originals[2], runtime)
    try:
        result = _BASE_V3_TRAIN(runtime, trainer, model, head, encoder, frozen, train_pairs, selection_pairs, indices, vocabulary, commanded, device, output_root)
    finally:
        (
            runtime.loss_adapter.observable_camera_ray_v4_loss_v4,
            _v1_runner._camera_components,
            _v1_runner._snapshot,
        ) = originals
    trace = result[0]
    for row in trace:
        losses = row.get("losses", {})
        if any("target_bin_offset_smooth_l1" in name for name in losses) or not all(
            f"{side}_tail_depth_p95_cvar" in losses for side in ("current", "next")
        ):
            raise RuntimeError("V4 training trace did not truthfully name the substituted loss slot")
    return result


def _access_receipt(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return {
        **_BASE_V3_ACCESS_RECEIPT(*args, **kwargs),
        "protected_camera_v4_preregistered_evidence": contract.evidence_contract(),
    }


def run_parent(*, review_file_sha256: str, authorization_file_sha256: str) -> int:
    """Rebind only the V4 contract, loss-wrapped train loop, and evidence receipt."""
    originals = (_v3.contract, _v3._train, _v3._access_receipt)
    _v3.contract = contract
    _v3._train = _train
    _v3._access_receipt = _access_receipt
    try:
        return _BASE_V3_RUN_PARENT(
            review_file_sha256=review_file_sha256,
            authorization_file_sha256=authorization_file_sha256,
        )
    finally:
        _v3.contract, _v3._train, _v3._access_receipt = originals


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--review-sha256")
    parser.add_argument("--authorization-sha256")
    args = parser.parse_args(argv)
    if not args.run or not contract._v1.is_sha256(args.review_sha256) or not contract._v1.is_sha256(args.authorization_sha256):
        parser.error("--run and both exact SHA-256 arguments are required")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run_parent(
        review_file_sha256=args.review_sha256,
        authorization_file_sha256=args.authorization_sha256,
    )


if __name__ == "__main__":
    raise SystemExit(main())
