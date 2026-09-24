#!/usr/bin/env python3
"""Run the one authorized final fresh-update-zero Camera V6 attempt."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
_CONTRACT_PATH = (
    ROOT
    / "lewm/benchmarks/"
    "go2_shared_jepa_v5_protected_camera_adaptation_v6.py"
)
_CONTRACT_MODULE = (
    "lewm.benchmarks.go2_shared_jepa_v5_protected_camera_adaptation_v6"
)
_V5_RUNNER_PATH = (
    ROOT / "scripts/run_go2_shared_jepa_v5_protected_camera_adaptation_v5.py"
)
_V5_RUNNER_SHA256 = (
    "3640ca35300ca36485487d6529dd352c76900c47018f7043cb165a1a078d72c4"
)


def _load_path(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path.relative_to(ROOT).as_posix()}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if _CONTRACT_MODULE in sys.modules:
    contract = sys.modules[_CONTRACT_MODULE]
else:
    contract = _load_path(
        "_lewm_protected_camera_adaptation_v6_contract", _CONTRACT_PATH
    )


def _load_exact_v5_runner() -> Any:
    if _V5_RUNNER_PATH.is_symlink() or not _V5_RUNNER_PATH.is_file():
        raise PermissionError("committed protected Camera V5 runner changed")
    raw = _V5_RUNNER_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != _V5_RUNNER_SHA256:
        raise PermissionError("committed protected Camera V5 runner changed")
    return _load_path(
        "_lewm_protected_camera_adaptation_v5_runner_for_v6",
        _V5_RUNNER_PATH,
    )


_v5 = _load_exact_v5_runner()
_v3 = _v5._v3
_v1_runner = _v5._v1_runner
_BASE_V5_RUN_PARENT = _v5.run_parent
_BASE_V5_TRAIN = _v5._train
_BASE_V3_PUBLISH_METRIC_SIDECAR = _v3._publish_metric_sidecar
_BASE_V3_ACCESS_RECEIPT = _v5._BASE_V3_ACCESS_RECEIPT

_read_regular = _v5._read_regular
_write_exclusive = _v5._write_exclusive
_publish_json = _v5._publish_json
_binding = _v5._binding
_ACTIVE_SIDECARS = _v5._ACTIVE_SIDECARS
_ACTIVE_CONTROL_DECISIONS = _v5._ACTIVE_CONTROL_DECISIONS


def _load_exact_tail_loss() -> Any:
    path = ROOT / contract.TAIL_DEPTH_LOSS_RELATIVE_PATH
    raw = path.read_bytes()
    expected_sha256 = contract.FIXED_EVIDENCE_SHA256[
        contract.TAIL_DEPTH_LOSS_RELATIVE_PATH
    ]
    if (
        path.is_symlink()
        or not path.is_file()
        or hashlib.sha256(raw).hexdigest() != expected_sha256
    ):
        raise PermissionError("frozen protected Camera V4 tail-depth loss changed")

    # The retained V4 module uses package-relative imports, so preserve its
    # reviewed package-import semantics rather than file-loading it under an
    # invented top-level name. The exact lifecycle normally installs ROOT
    # while loading the matched stack; restore the caller's path exactly when
    # this guard is exercised independently.
    original_sys_path = list(sys.path)
    try:
        if not sys.path or sys.path[0] != str(ROOT):
            sys.path.insert(0, str(ROOT))
        from lewm.models import (
            shared_observable_camera_ray_jepa_v5_protected_camera_adaptation_v4_tail_depth
            as tail,
        )
    finally:
        sys.path[:] = original_sys_path
    resolved = Path(tail.__file__).resolve()
    expected = path.resolve()
    observed = resolved.read_bytes()
    if (
        tail.__package__ != "lewm.models"
        or resolved != expected
        or resolved.is_symlink()
        or not resolved.is_file()
        or hashlib.sha256(observed).hexdigest() != expected_sha256
        or observed != raw
    ):
        raise PermissionError(
            "imported protected Camera V4 tail-depth loss binding changed"
        )
    return tail


def _camera_components(loss: Any) -> dict[str, float]:
    result = {"camera_total": _v1_runner._scalar(loss.total)}
    for side in ("current", "next"):
        frame = getattr(loss, side)
        result.update(
            {
                f"{side}_hierarchical_first_hit_nll":
                    _v1_runner._scalar(frame.hierarchical_first_hit_nll),
                f"{side}_tail_depth_p95_cvar":
                    _v1_runner._scalar(frame.tail_depth_p95_cvar),
                f"{side}_ground_clear_distance_state_balanced_bce":
                    _v1_runner._scalar(
                        frame.ground_clear_distance_state_balanced_bce
                    ),
                f"{side}_derived_raster_hierarchical_bce":
                    _v1_runner._scalar(
                        frame.derived_raster_hierarchical_bce.total
                    ),
                f"{side}_derived_raster_cell_nll":
                    _v1_runner._scalar(frame.derived_raster_cell_nll),
            }
        )
    return result


def _same_run_health_baseline(
    output_root: Path, *, update: int
) -> dict[str, Any]:
    previous = {400: 100, 1_000: 400}.get(update)
    if previous is None:
        raise ValueError("same-run health baseline requested at the wrong update")
    relative = contract.metric_sidecar_path(previous)
    if not relative.endswith(".metrics.json"):
        raise PermissionError("health baseline is not a metric sidecar")
    raw = _read_regular(output_root / relative)
    value = contract.parse_canonical_json(
        raw, name=f"same-run update {previous} metric sidecar"
    )
    contract.validate_metric_sidecar(value, update=previous)
    progress = contract.checkpoint_progress(value["metric"])
    if progress["update"] != previous:
        raise PermissionError("same-run health baseline update changed")
    return {
        "update": previous,
        "path": relative,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "content_sha256": value["content_sha256"],
        "passed_margin_count": progress["passed_margin_count"],
        "total_shortfall": progress["total_shortfall"],
    }


def _publish_metric_sidecar(
    output_root: Path,
    *,
    update: int,
    checkpoint: Mapping[str, Any],
    metric: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the two health checks, then delegate atomic mode-0444 publication."""
    if type(metric) is not dict:
        raise TypeError("checkpoint metric must be mutable before publication")
    if (
        "same_run_health_baseline" in metric
        or "update_4000_control_baseline" in metric
    ):
        raise PermissionError("checkpoint metric already contains a control baseline")
    metric["same_run_health_baseline"] = (
        _same_run_health_baseline(output_root, update=update)
        if update in (400, 1_000)
        else None
    )
    return _BASE_V3_PUBLISH_METRIC_SIDECAR(
        output_root,
        update=update,
        checkpoint=checkpoint,
        metric=metric,
    )


def _train(
    runtime: Any,
    trainer: Any,
    model: Any,
    head: Sequence[Any],
    encoder: Sequence[Any],
    frozen: Sequence[Any],
    train_pairs: Sequence[Mapping[str, Any]],
    selection_pairs: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    vocabulary: Sequence[str],
    commanded: Any,
    device: Any,
    output_root: Path,
):
    """Replace one loss slot around the exact V5 8k training loop."""
    tail = _load_exact_tail_loss()
    originals = (
        runtime.loss_adapter.observable_camera_ray_v4_loss_v4,
        _v1_runner._camera_components,
    )
    runtime.loss_adapter.observable_camera_ray_v4_loss_v4 = (
        tail.observable_camera_ray_v4_tail_depth_loss_v4
    )
    _v1_runner._camera_components = _camera_components
    try:
        result = _BASE_V5_TRAIN(
            runtime,
            trainer,
            model,
            head,
            encoder,
            frozen,
            train_pairs,
            selection_pairs,
            indices,
            vocabulary,
            commanded,
            device,
            output_root,
        )
    finally:
        (
            runtime.loss_adapter.observable_camera_ray_v4_loss_v4,
            _v1_runner._camera_components,
        ) = originals
    for row in result[0]:
        losses = row.get("losses", {})
        if (
            any("target_bin_offset_smooth_l1" in name for name in losses)
            or not all(
                f"{side}_tail_depth_p95_cvar" in losses
                for side in ("current", "next")
            )
        ):
            raise RuntimeError(
                "V6 trace did not truthfully name the V4 tail-depth slot"
            )
    return result


def _publish_training(
    output_root: Path,
    trace: Sequence[Mapping[str, Any]],
    metrics: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Collate only already-published immutable V6 metric sidecars."""
    trace_raw = b"".join(
        contract.canonical_json_bytes(row) + b"\n" for row in trace
    )
    _write_exclusive(output_root / "training_trace.jsonl", trace_raw)
    trace_binding = {
        "path": "training_trace.jsonl",
        "file_sha256": hashlib.sha256(trace_raw).hexdigest(),
        "content_sha256": contract.canonical_json_sha256(list(trace)),
        "byte_count": len(trace_raw),
        "row_count": len(trace),
    }
    updates = [int(row["update"]) for row in metrics]
    contract.validate_checkpoint_prefix(updates)
    if (
        [item["update"] for item in _ACTIVE_SIDECARS] != updates
        or len(_ACTIVE_CONTROL_DECISIONS) != len(metrics)
    ):
        raise RuntimeError(
            "final metrics did not receive the published sidecar/control prefix"
        )
    for item, metric, decision in zip(
        _ACTIVE_SIDECARS,
        metrics,
        _ACTIVE_CONTROL_DECISIONS,
        strict=True,
    ):
        raw = _read_regular(
            output_root / item["path"],
            expected_sha256=item["file_sha256"],
        )
        value = contract.parse_canonical_json(
            raw, name=f"published {item['path']}"
        )
        contract.validate_metric_sidecar(
            value, update=item["update"], metric=metric
        )
        if (
            len(raw) != item["byte_count"]
            or value["content_sha256"] != item["content_sha256"]
            or value["continuation"] != decision
        ):
            raise PermissionError(
                "published checkpoint metric sidecar binding changed"
            )
    metric_value, metric_raw = _publish_json(
        output_root / "checkpoint_metrics.json",
        {
            "schema": contract.METRICS_SCHEMA,
            "status":
                "fixed_prefix_collated_from_already_computed_immutable_sidecars",
            "checkpoint_updates": updates,
            "rows": list(metrics),
            "sidecars": list(_ACTIVE_SIDECARS),
            "checkpoint_controls": list(_ACTIVE_CONTROL_DECISIONS),
            "terminal_checkpoint_control": (
                dict(_v3._ACTIVE_TERMINAL_CONTROL)
                if _v3._ACTIVE_TERMINAL_CONTROL is not None
                else None
            ),
            "inline_evaluation_count": len(metrics),
            "observer_evaluation_rerun_count": 0,
            "selection_rule":
                "earliest_all_nine_physical_pass_with_same_run_health_checks_"
                "at_updates_400_and_1000_and_exact_update_8000_maximum",
            "soft_or_closest_promotion_authorized": False,
        },
    )
    return trace_binding, _binding(
        "checkpoint_metrics.json", metric_value, metric_raw
    )


def _access_receipt(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return {
        **_BASE_V3_ACCESS_RECEIPT(*args, **kwargs),
        "protected_camera_v6_preregistered_evidence":
            contract.evidence_contract(),
        "protected_camera_v6_visibility_preflight":
            contract.visibility_preflight_contract(),
    }


def run_parent(
    *, review_file_sha256: str, authorization_file_sha256: str
) -> int:
    """Install only the V6 contract, loss, controls, and truthful collation."""
    originals = (
        _v5.contract,
        _v5._train,
        _v5._publish_metric_sidecar,
        _v5._publish_training,
        _v5._access_receipt,
    )
    _v5.contract = contract
    _v5._train = _train
    _v5._publish_metric_sidecar = _publish_metric_sidecar
    _v5._publish_training = _publish_training
    _v5._access_receipt = _access_receipt
    try:
        return _BASE_V5_RUN_PARENT(
            review_file_sha256=review_file_sha256,
            authorization_file_sha256=authorization_file_sha256,
        )
    finally:
        (
            _v5.contract,
            _v5._train,
            _v5._publish_metric_sidecar,
            _v5._publish_training,
            _v5._access_receipt,
        ) = originals


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--review-sha256")
    parser.add_argument("--authorization-sha256")
    args = parser.parse_args(argv)
    if (
        not args.run
        or not contract._v1.is_sha256(args.review_sha256)
        or not contract._v1.is_sha256(args.authorization_sha256)
    ):
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
