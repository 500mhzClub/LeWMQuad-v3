#!/usr/bin/env python3
"""Run the one authorized one-knob protected Camera adaptation V2 attempt."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
_CONTRACT_PATH = ROOT / "lewm/benchmarks/go2_shared_jepa_v5_protected_camera_adaptation_v2.py"
_SPEC = importlib.util.spec_from_file_location("_lewm_protected_camera_adaptation_v2_contract", _CONTRACT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("cannot load protected Camera adaptation V2 contract")
contract = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(contract)


def _load_exact_v1_runner() -> Any:
    relative = contract.V1_RUNNER_RELATIVE_PATH
    path = ROOT / relative
    raw = path.read_bytes()
    if path.is_symlink() or not path.is_file() or hashlib.sha256(raw).hexdigest() != contract.V1_SOURCE_SHA256[relative]:
        raise PermissionError("frozen protected Camera V1 runner changed")
    spec = importlib.util.spec_from_file_location("_lewm_protected_camera_adaptation_v1_runner_for_v2", path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load frozen protected Camera V1 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.contract = contract
    return module


_base = _load_exact_v1_runner()
_BASE_TRAIN = _base._train
_BASE_UPDATE0_TERMINAL = _base._update0_terminal
_BASE_ACCESS_RECEIPT = _base._access_receipt
_BASE_RUN_PARENT = _base.run_parent

_read_regular = _base._read_regular
_write_exclusive = _base._write_exclusive
_publish_json = _base._publish_json
_binding = _base._binding
_existing_artifact_bindings = _base._existing_artifact_bindings

_ACTIVE_SIDECARS: list[dict[str, Any]] = []
_ACTIVE_V1_AUDIT: dict[str, Any] | None = None
_ACTIVE_V1_RECORDS: list[dict[str, Any]] = []


def _write_atomic_exclusive_read_only(path: Path, raw: bytes) -> None:
    """Publish a complete immutable-visible file without replacing any path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.publishing")
    _write_exclusive(temporary, raw)
    linked = False
    try:
        os.chmod(temporary, 0o444, follow_symlinks=False)
        descriptor = os.open(temporary, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.link(temporary, path, follow_symlinks=False)
        linked = True
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    if not linked:
        raise RuntimeError("checkpoint metric sidecar was not atomically linked")


def _publish_metric_sidecar(
    output_root: Path,
    *,
    update: int,
    checkpoint: Mapping[str, Any],
    metric: Mapping[str, Any],
) -> dict[str, Any]:
    core = {
        "schema": contract.METRIC_SIDECAR_SCHEMA,
        "status": "published_after_inline_nonmutating_physical_evaluation",
        "update": update,
        "checkpoint": dict(checkpoint),
        "metric": dict(metric),
        "inline_evaluation_count": 1,
        "state_mutation_count": 0,
        "publication": contract.reporting_contract()["publication_order"],
        "continuation": contract.reporting_contract()["numeric_continuation_rule"],
        "authority": {
            "read_only_observation_authorized": True,
            "observer_evaluation_rerun_authorized": False,
            "metric_controlled_stop_other_than_earliest_all_nine_pass": False,
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
    }


def _train(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
    """Run exact V1 training while publishing each already-computed metric row."""
    output_root = kwargs.get("output_root")
    if not isinstance(output_root, Path):
        if len(args) < 13 or not isinstance(args[12], Path):
            raise TypeError("protected Camera V2 output root is absent")
        output_root = args[12]
    snapshots: dict[int, dict[str, Any]] = {}
    sidecars: list[dict[str, Any]] = []
    original_snapshot = _base._snapshot
    original_evaluate = _base._evaluate

    def snapshot_then_remember(*inner_args: Any, **inner_kwargs: Any) -> dict[str, Any]:
        record = original_snapshot(*inner_args, **inner_kwargs)
        update = int(inner_kwargs["update"])
        if update in snapshots:
            raise RuntimeError("duplicate protected Camera V2 checkpoint snapshot")
        snapshots[update] = record
        return record

    def evaluate_then_publish(*inner_args: Any, **inner_kwargs: Any) -> dict[str, Any]:
        metric = original_evaluate(*inner_args, **inner_kwargs)
        update = int(inner_kwargs["update"])
        checkpoint = snapshots.get(update)
        if checkpoint is None:
            raise RuntimeError("metric evaluation preceded its checkpoint snapshot")
        sidecars.append(_publish_metric_sidecar(output_root, update=update, checkpoint=checkpoint, metric=metric))
        return metric

    _base._snapshot = snapshot_then_remember
    _base._evaluate = evaluate_then_publish
    try:
        trace, metrics, artifacts, selected, warnings, state = _BASE_TRAIN(*args, **kwargs)
    finally:
        _base._snapshot = original_snapshot
        _base._evaluate = original_evaluate
    updates = [int(row["update"]) for row in metrics]
    if [item["update"] for item in sidecars] != updates or tuple(item["path"] for item in sidecars) != contract.expected_metric_sidecar_paths(updates):
        raise RuntimeError("checkpoint metric sidecar prefix diverged from inline evaluations")
    state["operation_counts"] = {
        **state["operation_counts"],
        "checkpoint_metric_sidecar_publication_count": len(sidecars),
        "read_only_observer_evaluation_rerun_count": 0,
        "numeric_progress_cutoff_count": 0,
    }
    _ACTIVE_SIDECARS[:] = sidecars
    return trace, metrics, [*artifacts, *sidecars], selected, warnings, state


def _publish_training(output_root: Path, trace: Sequence[Mapping[str, Any]], metrics: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    trace_raw = b"".join(contract.canonical_json_bytes(row) + b"\n" for row in trace)
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
    if [item["update"] for item in _ACTIVE_SIDECARS] != updates:
        raise RuntimeError("final metrics did not receive the published sidecar prefix")
    for item, metric in zip(_ACTIVE_SIDECARS, metrics, strict=True):
        raw = _read_regular(output_root / item["path"], expected_sha256=item["file_sha256"])
        value = contract.parse_canonical_json(raw, name=f"published {item['path']}")
        contract.validate_metric_sidecar(value, update=item["update"], metric=metric)
        if len(raw) != item["byte_count"] or value["content_sha256"] != item["content_sha256"]:
            raise PermissionError("published checkpoint metric sidecar binding changed")
    metric_value, metric_raw = _publish_json(output_root / "checkpoint_metrics.json", {
        "schema": contract.METRICS_SCHEMA,
        "status": "fixed_prefix_collated_from_already_computed_immutable_sidecars",
        "checkpoint_updates": updates,
        "rows": list(metrics),
        "sidecars": list(_ACTIVE_SIDECARS),
        "inline_evaluation_count": len(metrics),
        "observer_evaluation_rerun_count": 0,
        "selection_rule": "earliest_all_nine_physical_pass",
        "numeric_progress_cutoff_at_update_400": False,
        "soft_or_closest_promotion_authorized": False,
    })
    return trace_binding, _binding("checkpoint_metrics.json", metric_value, metric_raw)


def _v1_terminal() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    binding = contract.V1_TERMINAL_AUDIT_BINDING
    audit_raw = _read_regular(ROOT / binding["path"], expected_sha256=binding["file_sha256"])
    audit = contract.validate_v1_terminal_audit(audit_raw)
    root = ROOT / contract.V1_TERMINAL_ROOT_RELATIVE_PATH
    entries = list(root.rglob("*")) if root.is_dir() and not root.is_symlink() else []
    files = sorted(item.relative_to(root).as_posix() for item in entries if item.is_file() and not item.is_symlink())
    directories = [".", *sorted(item.relative_to(root).as_posix() for item in entries if item.is_dir() and not item.is_symlink())]
    if files != list(contract.V1_TERMINAL_EXACT_PATHS) or directories != list(contract.V1_TERMINAL_EXACT_DIRECTORIES) or len(entries) != len(files) + len(directories) - 1:
        raise PermissionError("protected Camera V1 terminal inventory changed")
    by_path = {item["path"]: {"kind": kind, **item} for kind, item in audit["bindings"]["terminal_artifacts"].items()}
    records = []
    for relative in files:
        expected = by_path.get(relative)
        if expected is None:
            raise PermissionError("protected Camera V1 terminal binding is absent")
        raw = _read_regular(root / relative, expected_sha256=expected["file_sha256"])
        if len(raw) != expected["byte_count"]:
            raise PermissionError("protected Camera V1 terminal byte count changed")
        records.append(expected)
    return audit, records


def _update0_terminal_with_v1() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    global _ACTIVE_V1_AUDIT
    _ACTIVE_V1_AUDIT, records = _v1_terminal()
    _ACTIVE_V1_RECORDS[:] = records
    return _BASE_UPDATE0_TERMINAL()


def _access_receipt(*args: Any, **kwargs: Any) -> dict[str, Any]:
    value = _BASE_ACCESS_RECEIPT(*args, **kwargs)
    if _ACTIVE_V1_AUDIT is None or len(_ACTIVE_V1_RECORDS) != len(contract.V1_TERMINAL_EXACT_PATHS):
        raise RuntimeError("protected Camera V1 terminal was not rehashed")
    output_root = ROOT / contract.OUTPUT_ROOT_RELATIVE_PATH
    sidecar_records = []
    for item in _ACTIVE_SIDECARS:
        raw = _read_regular(output_root / item["path"], expected_sha256=item["file_sha256"])
        parsed = contract.parse_canonical_json(raw, name=f"access receipt {item['path']}")
        contract.validate_metric_sidecar(parsed, update=item["update"])
        if len(raw) != item["byte_count"] or parsed["content_sha256"] != item["content_sha256"]:
            raise PermissionError("checkpoint metric sidecar changed before access receipt")
        sidecar_records.append(dict(item))
    return {
        **value,
        "protected_camera_v1_predecessor": {
            "terminal_audit": dict(contract.V1_TERMINAL_AUDIT_BINDING),
            "verdict": _ACTIVE_V1_AUDIT["verdict"],
            "qualified_checkpoint_exists": False,
            "terminal_records": list(_ACTIVE_V1_RECORDS),
            "all_rehashed": True,
        },
        "checkpoint_metric_sidecars": {
            "records": sidecar_records,
            "count": len(sidecar_records),
            "all_rehashed": True,
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
    core = {
        "schema": contract.FAILURE_SCHEMA,
        "status": "failed_numeric_physical_gate" if numeric else "failed_protected_camera_adaptation",
        "stage": stage,
        "failure_path": "failed.json",
        "attempt_identity": reservation["attempt_identity"],
        "published_prefix": ["reservation.json", *(path for path in paths if path != "reservation.json")],
        "artifacts": artifacts,
        "caller_ledger_paths": list(published),
        "exact_terminal_files": sorted([*paths, "failed.json"]),
        "exact_terminal_directories_including_root": directories,
        "all_existing_regular_artifacts_bound": True,
        "error": {"type": type(error).__name__, "message": str(error)},
        "closest_or_soft_promotion": False,
        "extension_or_retry_authorized": False,
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
    """Install only V2 lineage/reporting hooks around the exact V1 runner."""
    _ACTIVE_SIDECARS.clear()
    _ACTIVE_V1_RECORDS.clear()
    global _ACTIVE_V1_AUDIT
    _ACTIVE_V1_AUDIT = None
    originals = (_base._train, _base._publish_training, _base._update0_terminal, _base._access_receipt, _base._terminal_failure)
    _base._train = _train
    _base._publish_training = _publish_training
    _base._update0_terminal = _update0_terminal_with_v1
    _base._access_receipt = _access_receipt
    _base._terminal_failure = _terminal_failure
    try:
        return _BASE_RUN_PARENT(review_file_sha256=review_file_sha256, authorization_file_sha256=authorization_file_sha256)
    finally:
        _base._train, _base._publish_training, _base._update0_terminal, _base._access_receipt, _base._terminal_failure = originals


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
