#!/usr/bin/env python3
"""Run the one authorized Camera V5 native-schedule completion attempt."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
_CONTRACT_PATH = ROOT / "lewm/benchmarks/go2_shared_jepa_v5_protected_camera_adaptation_v5.py"
_CONTRACT_MODULE = "lewm.benchmarks.go2_shared_jepa_v5_protected_camera_adaptation_v5"
if _CONTRACT_MODULE in sys.modules:
    contract = sys.modules[_CONTRACT_MODULE]
else:
    _SPEC = importlib.util.spec_from_file_location(
        "_lewm_protected_camera_adaptation_v5_contract", _CONTRACT_PATH
    )
    if _SPEC is None or _SPEC.loader is None:
        raise ImportError("cannot load protected Camera adaptation V5 contract")
    contract = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(contract)


def _load_exact_v3_runner() -> Any:
    relative = contract.V3_RUNNER_RELATIVE_PATH
    path = ROOT / relative
    raw = path.read_bytes()
    if (
        path.is_symlink()
        or not path.is_file()
        or hashlib.sha256(raw).hexdigest() != contract.V3_SOURCE_SHA256[relative]
    ):
        raise PermissionError("frozen protected Camera V3 runner changed")
    spec = importlib.util.spec_from_file_location(
        "_lewm_protected_camera_adaptation_v3_runner_for_v5", path
    )
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

_read_regular = _v3._read_regular
_write_exclusive = _v3._write_exclusive
_write_atomic_exclusive_read_only = _v3._write_atomic_exclusive_read_only
_publish_json = _v3._publish_json
_binding = _v3._binding
_existing_artifact_bindings = _v3._existing_artifact_bindings

_ACTIVE_SIDECARS = _v3._ACTIVE_SIDECARS
_ACTIVE_CONTROL_DECISIONS = _v3._ACTIVE_CONTROL_DECISIONS


def _finite_snapshot(base: Any, runtime: Any):
    """Reject a nonfinite model state before the inherited serializer observes it."""

    def checked(
        model_runtime: Any,
        model: Any,
        output_root: Path,
        *,
        update: int,
        frozen_sha: str,
    ) -> dict[str, Any]:
        if model_runtime is not runtime:
            raise RuntimeError("snapshot runtime changed")
        for name, value in model.state_dict().items():
            if (value.is_floating_point() or value.is_complex()) and not bool(
                runtime.torch.isfinite(value).all().item()
            ):
                raise FloatingPointError(f"checkpoint state became nonfinite: {name}")
        return base(
            model_runtime,
            model,
            output_root,
            update=update,
            frozen_sha=frozen_sha,
        )

    return checked


def _validate_schedule(indices: Sequence[int]) -> None:
    """Require the complete, exact native 8k presentation schedule."""
    required = contract.MAXIMUM_UPDATE * 16
    if len(indices) != required:
        raise PermissionError(
            f"bound V4 schedule must contain exactly {required} protected presentations"
        )
    try:
        normalized = contract._v1.validate_schedule_indices(indices)
    except (TypeError, ValueError) as error:
        raise PermissionError("bound native schedule structure or prefix changed") from error
    for update, expected in contract.CHECKPOINT_SCHEDULE_PREFIX_SHA256.items():
        observed = contract.canonical_json_sha256(list(normalized[: update * 16]))
        if observed != expected:
            raise PermissionError(
                f"bound V4 schedule prefix changed at update {update}"
            )


def _update_4000_control_baseline(output_root: Path) -> dict[str, Any]:
    """Derive the u6000 comparator only from the immutable same-run u4000 sidecar."""
    relative = contract.metric_sidecar_path(4_000)
    raw = _read_regular(output_root / relative)
    value = contract.parse_canonical_json(raw, name="same-run update 4000 metric sidecar")
    contract.validate_metric_sidecar(value, update=4_000)
    progress = contract.checkpoint_progress(value["metric"])
    if progress["update"] != 4_000:
        raise PermissionError("same-run update 4000 control baseline changed")
    return {
        "update": 4_000,
        "path": relative,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "content_sha256": value["content_sha256"],
        "passed_margin_count": progress["passed_margin_count"],
        "total_shortfall": progress["total_shortfall"],
        "worst_margin": progress["worst_margin"],
    }


def _publish_metric_sidecar(
    output_root: Path,
    *,
    update: int,
    checkpoint: Mapping[str, Any],
    metric: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish one metric and bind u6000 to the immutable same-run u4000 metric."""
    if type(metric) is not dict:
        raise TypeError("checkpoint metric must be mutable before sidecar publication")
    if "update_4000_control_baseline" in metric:
        raise PermissionError("checkpoint metric already contains a control baseline")
    metric["update_4000_control_baseline"] = (
        _update_4000_control_baseline(output_root) if update == 6_000 else None
    )
    decision = contract.checkpoint_control_decision(metric)
    core = {
        "schema": contract.METRIC_SIDECAR_SCHEMA,
        "status": "published_after_inline_nonmutating_physical_evaluation_before_control_branch",
        "update": update,
        "checkpoint": dict(checkpoint),
        "metric": dict(metric),
        "inline_evaluation_count": 1,
        "state_mutation_count": 0,
        "publication": contract.reporting_contract()["publication_order"],
        "continuation": decision,
        "authority": {
            "read_only_observation_authorized": True,
            "observer_evaluation_rerun_authorized": False,
            "only_predeclared_metric_control_authorized": True,
            "g2_navigation_or_heldout_use_authorized": False,
        },
    }
    value = contract.with_content_sha256(core)
    contract.validate_metric_sidecar(
        value, update=update, checkpoint=checkpoint, metric=metric
    )
    raw = contract.canonical_json_bytes(value) + b"\n"
    relative = contract.metric_sidecar_path(update)
    _write_atomic_exclusive_read_only(output_root / relative, raw)
    observed = _read_regular(
        output_root / relative, expected_sha256=hashlib.sha256(raw).hexdigest()
    )
    parsed = contract.parse_canonical_json(
        observed, name=f"checkpoint metric sidecar {update}"
    )
    contract.validate_metric_sidecar(
        parsed, update=update, checkpoint=checkpoint, metric=metric
    )
    return {
        **contract.artifact_binding(
            relative, observed, content_sha256=value["content_sha256"]
        ),
        "schema": contract.METRIC_SIDECAR_SCHEMA,
        "update": update,
        "control_action": decision["action"],
    }


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
    """Run the exact V3 optimizer loop with only a finite-snapshot guard added."""
    original_snapshot = _v1_runner._snapshot
    original_loss = runtime.loss_adapter.observable_camera_ray_v4_loss_v4
    _v1_runner._snapshot = _finite_snapshot(original_snapshot, runtime)
    try:
        result = _BASE_V3_TRAIN(
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
        if runtime.loss_adapter.observable_camera_ray_v4_loss_v4 is not original_loss:
            raise RuntimeError("protected Camera V3 loss slot changed during V5")
        return result
    finally:
        _v1_runner._snapshot = original_snapshot


def _publish_training(
    output_root: Path,
    trace: Sequence[Mapping[str, Any]],
    metrics: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Collate immutable sidecars and revalidate the same-run Pareto binding."""
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
        _ACTIVE_SIDECARS, metrics, _ACTIVE_CONTROL_DECISIONS, strict=True
    ):
        raw = _read_regular(
            output_root / item["path"], expected_sha256=item["file_sha256"]
        )
        value = contract.parse_canonical_json(
            raw, name=f"published {item['path']}"
        )
        contract.validate_metric_sidecar(value, update=item["update"], metric=metric)
        if (
            len(raw) != item["byte_count"]
            or value["content_sha256"] != item["content_sha256"]
            or value["continuation"] != decision
        ):
            raise PermissionError("published checkpoint metric sidecar binding changed")
    if 6_000 in updates:
        row = metrics[updates.index(6_000)]
        if row.get("update_4000_control_baseline") != _update_4000_control_baseline(
            output_root
        ):
            raise PermissionError(
                "update 6000 comparator is not the actual same-run update 4000 sidecar"
            )
    metric_value, metric_raw = _publish_json(
        output_root / "checkpoint_metrics.json",
        {
            "schema": contract.METRICS_SCHEMA,
            "status": "fixed_prefix_collated_from_already_computed_immutable_sidecars",
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
            "selection_rule": "earliest_all_nine_physical_pass_with_predeclared_update_1000_4000_and_6000_progress_controls_and_update_8000_maximum",
            "soft_or_closest_promotion_authorized": False,
        },
    )
    return trace_binding, _binding(
        "checkpoint_metrics.json", metric_value, metric_raw
    )


def _access_receipt(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return {
        **_BASE_V3_ACCESS_RECEIPT(*args, **kwargs),
        "protected_camera_v5_preregistered_evidence": contract.evidence_contract(),
    }


def _terminal_failure(
    output_root: Path,
    reservation: Mapping[str, Any],
    stage: str,
    error: BaseException,
    published: Mapping[str, Any],
    *,
    numeric: bool = False,
) -> None:
    """Publish a truthful terminal record despite the inherited 4k caller text."""
    records, directories = _existing_artifact_bindings(output_root)
    artifacts = {record["path"]: record for record in records}
    reservation_binding = artifacts.get("reservation.json")
    if (
        type(reservation_binding) is not dict
        or reservation_binding.get("content_sha256") != reservation["content_sha256"]
    ):
        raise RuntimeError("failure inventory cannot bind the committed reservation")
    paths = [record["path"] for record in records]
    decision = (
        dict(_v3._ACTIVE_TERMINAL_CONTROL)
        if _v3._ACTIVE_TERMINAL_CONTROL is not None
        else None
    )
    progress_cutoff = (
        numeric
        and decision is not None
        and decision["action"] == contract.CONTROL_ACTION_STOP_PROGRESS
    )
    maximum_no_pass = (
        numeric
        and decision is not None
        and decision["action"] == contract.CONTROL_ACTION_STOP_MAXIMUM
    )
    controlled_numeric = progress_cutoff or maximum_no_pass
    terminal_stage = decision["terminal_stage"] if controlled_numeric else stage
    error_value = (
        {
            "type": (
                "PredeclaredNumericProgressCutoff"
                if progress_cutoff
                else "MaximumUpdatePhysicalGateFailure"
            ),
            "message": decision["reason"],
        }
        if controlled_numeric
        else {"type": type(error).__name__, "message": str(error)}
    )
    if progress_cutoff:
        caller_message = (
            "base lifecycle requested numeric terminalization after the "
            f"predeclared stop at update {decision['update']}; update "
            f"{contract.MAXIMUM_UPDATE} was not reached"
        )
    elif maximum_no_pass:
        caller_message = (
            "base lifecycle requested numeric terminalization after the exact "
            f"maximum update {contract.MAXIMUM_UPDATE} did not qualify"
        )
    else:
        caller_message = str(error)
    caller_error = {
        "type": (
            "BaseLifecycleSelectedCheckpointAbsentTrigger"
            if controlled_numeric
            else type(error).__name__
        ),
        "message": caller_message,
    }
    core = {
        "schema": contract.FAILURE_SCHEMA,
        "status": (
            "failed_predeclared_numeric_progress_cutoff"
            if progress_cutoff
            else "failed_numeric_physical_gate"
            if numeric
            else "failed_protected_camera_adaptation"
        ),
        "stage": terminal_stage,
        "failure_path": "failed.json",
        "attempt_identity": reservation["attempt_identity"],
        "published_prefix": [
            "reservation.json",
            *(path for path in paths if path != "reservation.json"),
        ],
        "artifacts": artifacts,
        "caller_ledger_paths": list(published),
        "exact_terminal_files": sorted([*paths, "failed.json"]),
        "exact_terminal_directories_including_root": directories,
        "all_existing_regular_artifacts_bound": True,
        "error": error_value,
        "caller_error": caller_error,
        "checkpoint_control": decision,
        "numeric_progress_cutoff_applied": progress_cutoff,
        "maximum_update_no_pass": maximum_no_pass,
        "closest_or_soft_promotion": False,
        "extension_or_retry_authorized": False,
        "downstream_authority_granted": False,
        "g2_attempted": False,
        "navigation_attempted": False,
        "heldout_open_count": 0,
        "authority": dict(contract.DOWNSTREAM_DENIALS),
    }
    try:
        _publish_json(output_root / "failed.json", core)
    except FileExistsError:
        pass


def run_parent(*, review_file_sha256: str, authorization_file_sha256: str) -> int:
    """Install only V5 schedule/control/reporting hooks around exact V3."""
    originals = (
        _v3.contract,
        _v3._train,
        _v3._publish_metric_sidecar,
        _v3._publish_training,
        _v3._access_receipt,
        _v3._terminal_failure,
        _v1_runner._validate_schedule,
    )
    _v3.contract = contract
    _v3._train = _train
    _v3._publish_metric_sidecar = _publish_metric_sidecar
    _v3._publish_training = _publish_training
    _v3._access_receipt = _access_receipt
    _v3._terminal_failure = _terminal_failure
    _v1_runner._validate_schedule = _validate_schedule
    try:
        return _BASE_V3_RUN_PARENT(
            review_file_sha256=review_file_sha256,
            authorization_file_sha256=authorization_file_sha256,
        )
    finally:
        (
            _v3.contract,
            _v3._train,
            _v3._publish_metric_sidecar,
            _v3._publish_training,
            _v3._access_receipt,
            _v3._terminal_failure,
            _v1_runner._validate_schedule,
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
