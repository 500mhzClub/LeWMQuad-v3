#!/usr/bin/env python3
"""Run the one authorized final protected Camera adaptation V3 attempt."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
from typing import Any, Mapping, Sequence
import warnings


ROOT = Path(__file__).resolve().parents[1]
_CONTRACT_PATH = ROOT / "lewm/benchmarks/go2_shared_jepa_v5_protected_camera_adaptation_v3.py"
_SPEC = importlib.util.spec_from_file_location("_lewm_protected_camera_adaptation_v3_contract", _CONTRACT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("cannot load protected Camera adaptation V3 contract")
contract = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(contract)


def _load_exact_v2_runner() -> Any:
    relative = contract.V2_RUNNER_RELATIVE_PATH
    path = ROOT / relative
    raw = path.read_bytes()
    if path.is_symlink() or not path.is_file() or hashlib.sha256(raw).hexdigest() != contract.V2_SOURCE_SHA256[relative]:
        raise PermissionError("frozen protected Camera V2 runner changed")
    spec = importlib.util.spec_from_file_location("_lewm_protected_camera_adaptation_v2_runner_for_v3", path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load frozen protected Camera V2 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_v2 = _load_exact_v2_runner()
_v1_runner = _v2._base
_BASE_V2_RUN_PARENT = _v2.run_parent
_BASE_V2_UPDATE0_TERMINAL = _v2._update0_terminal_with_v1
_BASE_V2_ACCESS_RECEIPT = _v2._access_receipt

_read_regular = _v2._read_regular
_write_exclusive = _v2._write_exclusive
_write_atomic_exclusive_read_only = _v2._write_atomic_exclusive_read_only
_publish_json = _v2._publish_json
_binding = _v2._binding
_existing_artifact_bindings = _v2._existing_artifact_bindings

_ACTIVE_SIDECARS = _v2._ACTIVE_SIDECARS
_ACTIVE_V2_AUDIT: dict[str, Any] | None = None
_ACTIVE_V2_RECORDS: list[dict[str, Any]] = []
_ACTIVE_CONTROL_DECISIONS: list[dict[str, Any]] = []
_ACTIVE_TERMINAL_CONTROL: dict[str, Any] | None = None


def _publish_metric_sidecar(
    output_root: Path,
    *,
    update: int,
    checkpoint: Mapping[str, Any],
    metric: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish the one immutable sidecar before applying its declared control branch."""
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
    contract.validate_metric_sidecar(value, update=update, checkpoint=checkpoint, metric=metric)
    raw = contract.canonical_json_bytes(value) + b"\n"
    relative = contract.metric_sidecar_path(update)
    _write_atomic_exclusive_read_only(output_root / relative, raw)
    observed = _read_regular(output_root / relative, expected_sha256=hashlib.sha256(raw).hexdigest())
    parsed = contract.parse_canonical_json(observed, name=f"checkpoint metric sidecar {update}")
    contract.validate_metric_sidecar(parsed, update=update, checkpoint=checkpoint, metric=metric)
    return {
        **contract.artifact_binding(relative, observed, content_sha256=value["content_sha256"]),
        "schema": contract.METRIC_SIDECAR_SCHEMA,
        "update": update,
        "control_action": decision["action"],
    }


def _train(runtime: Any, trainer: Any, model: Any, head: Sequence[Any], encoder: Sequence[Any], frozen: Sequence[Any], train_pairs: Sequence[Mapping[str, Any]], selection_pairs: Sequence[Mapping[str, Any]], indices: Sequence[int], vocabulary: Sequence[str], commanded: Any, device: Any, output_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int | None, dict[str, Any], dict[str, Any]]:
    """Run the exact protected loop with only the predeclared V3 checkpoint controls."""
    global _ACTIVE_TERMINAL_CONTROL
    frozen_sha = _v1_runner._subset_sha(runtime, model, contract.FROZEN_STATE_PREFIXES)
    optimizer = runtime.torch.optim.AdamW(
        [{"params": list(head), "lr": contract.learning_rates(1)[0], "group_name": "evidence_head"}, {"params": list(encoder), "lr": contract.learning_rates(1)[1], "group_name": "encoder"}],
        betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4, amsgrad=False,
    )
    trace: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    sidecars: list[dict[str, Any]] = []
    selected: int | None = None
    collector = contract._diagnostic.CompactDeterminismWarnings()
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        original = warnings.showwarning
        warnings.showwarning = collector
        try:
            for update in range(1, contract.MAXIMUM_UPDATE + 1):
                head_lr, encoder_lr = contract.learning_rates(update)
                optimizer.param_groups[0]["lr"] = head_lr
                optimizer.param_groups[1]["lr"] = encoder_lr
                _v1_runner._assert_frozen_grads_none(frozen)
                optimizer.zero_grad(set_to_none=True)
                sums: dict[str, float] = {}
                start = (update - 1) * 16
                update_indices = indices[start : start + 16]
                for micro in range(4):
                    batch = trainer.batch(train_pairs, update_indices[micro * 4 : (micro + 1) * 4], vocabulary, commanded, device, role="train", arm="protected_camera_adaptation", stage="camera_gradient")
                    pair = _v1_runner._camera_pair(runtime, model, batch)
                    camera = runtime.loss_adapter.observable_camera_ray_v4_loss_v4(model, pair, batch["current_supervision"], batch["next_supervision"])
                    if not bool(runtime.torch.isfinite(camera.total).item()):
                        raise FloatingPointError("Camera-only backward scalar became nonfinite")
                    (camera.total / 4.0).backward()
                    for name, value in _v1_runner._camera_components(camera).items():
                        sums[name] = sums.get(name, 0.0) + value / 4.0
                _v1_runner._assert_frozen_grads_none(frozen)
                head_pre_clip = _v1_runner._gradient_group_norm(runtime, head, "evidence_head")
                encoder_pre_clip = _v1_runner._gradient_group_norm(runtime, encoder, "encoder")
                head_norm = runtime.torch.nn.utils.clip_grad_norm_(head, max_norm=1.0)
                encoder_norm = runtime.torch.nn.utils.clip_grad_norm_(encoder, max_norm=1.0)
                if not bool(runtime.torch.isfinite(head_norm).item()) or not bool(runtime.torch.isfinite(encoder_norm).item()):
                    raise FloatingPointError("protected group gradient norm became nonfinite")
                head_post_clip = _v1_runner._gradient_group_norm(runtime, head, "evidence_head", maximum=1.0)
                encoder_post_clip = _v1_runner._gradient_group_norm(runtime, encoder, "encoder", maximum=1.0)
                optimizer.step()
                _v1_runner._assert_frozen_grads_none(frozen)
                trace.append({"schema": f"{contract.SCHEMA_PREFIX}_trace_row_v1", "update": update, "presentation_indices_sha256": contract.canonical_json_sha256(list(update_indices)), "head_learning_rate": head_lr, "encoder_learning_rate": encoder_lr, "microbatch_count": 4, "camera_backward_count": 4, "jepa_objective_count": 0, "jepa_backward_count": 0, "optimizer_step_count": update, "head_clip_invocation_count": update, "encoder_clip_invocation_count": update, "head_gradient_tensor_count": len(head), "encoder_gradient_tensor_count": len(encoder), "head_gradient_norm_before_clip": head_pre_clip, "encoder_gradient_norm_before_clip": encoder_pre_clip, "head_clip_return_norm": _v1_runner._scalar(head_norm), "encoder_clip_return_norm": _v1_runner._scalar(encoder_norm), "head_gradient_norm_after_clip": head_post_clip, "encoder_gradient_norm_after_clip": encoder_post_clip, "post_clip_group_norm_maximum": 1.0, "post_clip_norm_assertion_tolerance": contract.POST_CLIP_NORM_ASSERTION_TOLERANCE, "losses": sums, "ema_update_count": 0})
                if update in contract.CHECKPOINT_UPDATES:
                    if _v1_runner._subset_sha(runtime, model, contract.FROZEN_STATE_PREFIXES) != frozen_sha:
                        raise RuntimeError("frozen state changed during protected training")
                    checkpoint = _v1_runner._snapshot(runtime, model, output_root, update=update, frozen_sha=frozen_sha)
                    snapshots.append(checkpoint)
                    metric = _v1_runner._evaluate(runtime, trainer, model, selection_pairs, device, update=update, frozen_sha=frozen_sha)
                    sidecar = _publish_metric_sidecar(output_root, update=update, checkpoint=checkpoint, metric=metric)
                    sidecars.append(sidecar)
                    metrics.append(metric)
                    decision = contract.checkpoint_control_decision(metric)
                    _ACTIVE_CONTROL_DECISIONS.append(decision)
                    if decision["action"] == contract.CONTROL_ACTION_QUALIFY:
                        selected = update
                        _ACTIVE_TERMINAL_CONTROL = decision
                        break
                    if decision["action"] in {contract.CONTROL_ACTION_STOP_PROGRESS, contract.CONTROL_ACTION_STOP_MAXIMUM}:
                        _ACTIVE_TERMINAL_CONTROL = decision
                        break
        finally:
            warnings.showwarning = original
    warning_receipt = collector.receipt()
    if warning_receipt["warning_count"] <= 0 or _v1_runner._subset_sha(runtime, model, contract.FROZEN_STATE_PREFIXES) != frozen_sha:
        raise RuntimeError("warning or frozen-state guard failed")
    if [item["update"] for item in sidecars] != [int(row["update"]) for row in metrics]:
        raise RuntimeError("checkpoint metric sidecar prefix diverged from inline evaluations")
    operation_counts = {"complete_update_count": len(trace), "camera_objective_count": len(trace) * 4, "camera_backward_count": len(trace) * 4, "optimizer_construction_count": 1, "optimizer_step_count": len(trace), "head_clip_invocation_count": len(trace), "encoder_clip_invocation_count": len(trace), "total_clip_invocation_count": len(trace) * 2, "global_clip_invocation_count": 0, "trainable_gradient_tensor_checks_per_update": sum(contract.EXPECTED_PARAMETER_TENSOR_COUNTS.values()) * 2, "all_trainable_gradients_present_and_finite_before_clip": True, "all_trainable_gradients_finite_after_clip": True, "all_post_clip_group_norms_at_most_one_with_declared_fp32_tolerance": True, "post_clip_norm_assertion_tolerance": contract.POST_CLIP_NORM_ASSERTION_TOLERANCE, "jepa_objective_count": 0, "jepa_backward_count": 0, "ema_update_count": 0, "physical_selection_count": len(metrics), "frozen_state_mutation_count": 0, "checkpoint_metric_sidecar_publication_count": len(sidecars), "read_only_observer_evaluation_rerun_count": 0, "numeric_progress_cutoff_count": int(_ACTIVE_TERMINAL_CONTROL is not None and _ACTIVE_TERMINAL_CONTROL["action"] == contract.CONTROL_ACTION_STOP_PROGRESS)}
    _ACTIVE_SIDECARS[:] = sidecars
    state = {"frozen_state_sha256": frozen_sha, "final_state_sha256": runtime.model_module.tensor_state_dict_sha256(model.state_dict()), "operation_counts": operation_counts, "terminal_checkpoint_control": dict(_ACTIVE_TERMINAL_CONTROL) if _ACTIVE_TERMINAL_CONTROL is not None else None}
    return trace, metrics, [*snapshots, *sidecars], selected, warning_receipt, state


def _publish_training(output_root: Path, trace: Sequence[Mapping[str, Any]], metrics: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Collate only the immutable sidecars already published inline."""
    trace_raw = b"".join(contract.canonical_json_bytes(row) + b"\n" for row in trace)
    _write_exclusive(output_root / "training_trace.jsonl", trace_raw)
    trace_binding = {"path": "training_trace.jsonl", "file_sha256": hashlib.sha256(trace_raw).hexdigest(), "content_sha256": contract.canonical_json_sha256(list(trace)), "byte_count": len(trace_raw), "row_count": len(trace)}
    updates = [int(row["update"]) for row in metrics]
    contract.validate_checkpoint_prefix(updates)
    if [item["update"] for item in _ACTIVE_SIDECARS] != updates or len(_ACTIVE_CONTROL_DECISIONS) != len(metrics):
        raise RuntimeError("final metrics did not receive the published sidecar/control prefix")
    for item, metric, decision in zip(_ACTIVE_SIDECARS, metrics, _ACTIVE_CONTROL_DECISIONS, strict=True):
        raw = _read_regular(output_root / item["path"], expected_sha256=item["file_sha256"])
        value = contract.parse_canonical_json(raw, name=f"published {item['path']}")
        contract.validate_metric_sidecar(value, update=item["update"], metric=metric)
        if len(raw) != item["byte_count"] or value["content_sha256"] != item["content_sha256"] or value["continuation"] != decision:
            raise PermissionError("published checkpoint metric sidecar binding changed")
    metric_value, metric_raw = _publish_json(output_root / "checkpoint_metrics.json", {
        "schema": contract.METRICS_SCHEMA,
        "status": "fixed_prefix_collated_from_already_computed_immutable_sidecars",
        "checkpoint_updates": updates,
        "rows": list(metrics),
        "sidecars": list(_ACTIVE_SIDECARS),
        "checkpoint_controls": list(_ACTIVE_CONTROL_DECISIONS),
        "terminal_checkpoint_control": dict(_ACTIVE_TERMINAL_CONTROL) if _ACTIVE_TERMINAL_CONTROL is not None else None,
        "inline_evaluation_count": len(metrics),
        "observer_evaluation_rerun_count": 0,
        "selection_rule": "earliest_all_nine_physical_pass_with_predeclared_update_1000_and_2000_progress_controls",
        "soft_or_closest_promotion_authorized": False,
    })
    return trace_binding, _binding("checkpoint_metrics.json", metric_value, metric_raw)


def _flatten_terminal_artifact_bindings(value: object) -> list[dict[str, Any]]:
    """Flatten the audit's singleton and repeated artifact groups exactly once."""
    if type(value) is not dict or not value:
        raise PermissionError("protected Camera V2 terminal artifact groups are malformed")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for kind in sorted(value):
        group = value[kind]
        if type(group) is dict:
            items = [group]
        elif type(group) is list and group:
            items = group
        else:
            raise PermissionError("protected Camera V2 terminal artifact group is malformed")
        for item in items:
            if type(item) is not dict or "kind" in item:
                raise PermissionError("protected Camera V2 terminal artifact binding is malformed")
            path = item.get("path")
            file_sha256 = item.get("file_sha256")
            byte_count = item.get("byte_count")
            if (
                type(path) is not str
                or not path
                or path.startswith("/")
                or ".." in Path(path).parts
                or not contract._v1.is_sha256(file_sha256)
                or type(byte_count) is not int
                or byte_count < 0
                or path in seen
            ):
                raise PermissionError("protected Camera V2 terminal artifact binding is malformed or duplicated")
            seen.add(path)
            records.append({"kind": kind, **item})
    return sorted(records, key=lambda item: item["path"])


def _v2_terminal() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    binding = contract.V2_TERMINAL_AUDIT_BINDING
    audit_raw = _read_regular(ROOT / binding["path"], expected_sha256=binding["file_sha256"])
    audit = contract.validate_v2_terminal_audit(audit_raw)
    root = ROOT / contract.V2_TERMINAL_ROOT_RELATIVE_PATH
    entries = list(root.rglob("*")) if root.is_dir() and not root.is_symlink() else []
    files = sorted(item.relative_to(root).as_posix() for item in entries if item.is_file() and not item.is_symlink())
    directories = [".", *sorted(item.relative_to(root).as_posix() for item in entries if item.is_dir() and not item.is_symlink())]
    if files != list(contract.V2_TERMINAL_EXACT_PATHS) or directories != list(contract.V2_TERMINAL_EXACT_DIRECTORIES) or len(entries) != len(files) + len(directories) - 1:
        raise PermissionError("protected Camera V2 terminal inventory changed")
    bindings = audit.get("bindings")
    groups = bindings.get("terminal_artifacts") if type(bindings) is dict else None
    flattened = _flatten_terminal_artifact_bindings(groups)
    by_path = {item["path"]: item for item in flattened}
    if tuple(by_path) != contract.V2_TERMINAL_EXACT_PATHS:
        raise PermissionError("protected Camera V2 terminal artifact binding paths changed")
    records = []
    for relative in files:
        expected = by_path.get(relative)
        if expected is None:
            raise PermissionError("protected Camera V2 terminal binding is absent")
        raw = _read_regular(root / relative, expected_sha256=expected["file_sha256"])
        if len(raw) != expected["byte_count"]:
            raise PermissionError("protected Camera V2 terminal byte count changed")
        records.append(expected)
    return audit, records


def _update0_terminal_with_v2() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    global _ACTIVE_V2_AUDIT
    _ACTIVE_V2_AUDIT, records = _v2_terminal()
    _ACTIVE_V2_RECORDS[:] = records
    return _BASE_V2_UPDATE0_TERMINAL()


def _access_receipt(*args: Any, **kwargs: Any) -> dict[str, Any]:
    value = _BASE_V2_ACCESS_RECEIPT(*args, **kwargs)
    if _ACTIVE_V2_AUDIT is None or len(_ACTIVE_V2_RECORDS) != len(contract.V2_TERMINAL_EXACT_PATHS):
        raise RuntimeError("protected Camera V2 terminal was not rehashed")
    return {
        **value,
        "protected_camera_v2_predecessor": {
            "terminal_audit": dict(contract.V2_TERMINAL_AUDIT_BINDING),
            "verdict": _ACTIVE_V2_AUDIT["verdict"],
            "qualified_checkpoint_exists": False,
            "terminal_records": list(_ACTIVE_V2_RECORDS),
            "all_rehashed": True,
        },
        "checkpoint_control": {
            "contract": contract.control_contract(),
            "decisions": list(_ACTIVE_CONTROL_DECISIONS),
            "terminal": dict(_ACTIVE_TERMINAL_CONTROL) if _ACTIVE_TERMINAL_CONTROL is not None else None,
            "observer_evaluation_rerun_count": 0,
        },
    }


def _terminal_failure(output_root: Path, reservation: Mapping[str, Any], stage: str, error: BaseException, published: Mapping[str, Any], *, numeric: bool = False) -> None:
    records, directories = _existing_artifact_bindings(output_root)
    artifacts = {record["path"]: record for record in records}
    reservation_binding = artifacts.get("reservation.json")
    if type(reservation_binding) is not dict or reservation_binding.get("content_sha256") != reservation["content_sha256"]:
        raise RuntimeError("failure inventory cannot bind the committed reservation")
    paths = [record["path"] for record in records]
    decision = dict(_ACTIVE_TERMINAL_CONTROL) if _ACTIVE_TERMINAL_CONTROL is not None else None
    progress_cutoff = numeric and decision is not None and decision["action"] == contract.CONTROL_ACTION_STOP_PROGRESS
    maximum_no_pass = numeric and decision is not None and decision["action"] == contract.CONTROL_ACTION_STOP_MAXIMUM
    terminal_stage = decision["terminal_stage"] if progress_cutoff or maximum_no_pass else stage
    error_value = (
        {"type": "PredeclaredNumericProgressCutoff", "message": decision["reason"]}
        if progress_cutoff
        else {"type": type(error).__name__, "message": str(error)}
    )
    caller_error = (
        {
            "type": "BaseLifecycleSelectedCheckpointAbsentTrigger",
            "message": f"base lifecycle requested numeric terminalization after the predeclared stop at update {decision['update']}; update 4000 was not reached",
        }
        if progress_cutoff
        else {"type": type(error).__name__, "message": str(error)}
    )
    core = {
        "schema": contract.FAILURE_SCHEMA,
        "status": "failed_predeclared_numeric_progress_cutoff" if progress_cutoff else "failed_numeric_physical_gate" if numeric else "failed_protected_camera_adaptation",
        "stage": terminal_stage,
        "failure_path": "failed.json",
        "attempt_identity": reservation["attempt_identity"],
        "published_prefix": ["reservation.json", *(path for path in paths if path != "reservation.json")],
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
    """Install the V3 loop/control and V2-lineage hooks around the exact lifecycle."""
    global _ACTIVE_V2_AUDIT, _ACTIVE_TERMINAL_CONTROL
    _ACTIVE_SIDECARS.clear()
    _ACTIVE_V2_RECORDS.clear()
    _ACTIVE_CONTROL_DECISIONS.clear()
    _ACTIVE_V2_AUDIT = None
    _ACTIVE_TERMINAL_CONTROL = None
    originals = (
        _v2.contract,
        _v1_runner.contract,
        _v2._train,
        _v2._publish_training,
        _v2._update0_terminal_with_v1,
        _v2._access_receipt,
        _v2._terminal_failure,
    )
    _v2.contract = contract
    _v1_runner.contract = contract
    _v2._train = _train
    _v2._publish_training = _publish_training
    _v2._update0_terminal_with_v1 = _update0_terminal_with_v2
    _v2._access_receipt = _access_receipt
    _v2._terminal_failure = _terminal_failure
    try:
        return _BASE_V2_RUN_PARENT(review_file_sha256=review_file_sha256, authorization_file_sha256=authorization_file_sha256)
    finally:
        (
            _v2.contract,
            _v1_runner.contract,
            _v2._train,
            _v2._publish_training,
            _v2._update0_terminal_with_v1,
            _v2._access_receipt,
            _v2._terminal_failure,
        ) = originals


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
    return run_parent(review_file_sha256=args.review_sha256, authorization_file_sha256=args.authorization_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
