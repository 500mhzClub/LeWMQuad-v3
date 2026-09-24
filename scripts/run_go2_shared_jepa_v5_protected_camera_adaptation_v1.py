#!/usr/bin/env python3
"""Run the one authorized protected Camera-only Shared-V5 adaptation attempt."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Sequence
import warnings


ROOT = Path(__file__).resolve().parents[1]
_CONTRACT_PATH = ROOT / "lewm/benchmarks/go2_shared_jepa_v5_protected_camera_adaptation_v1.py"
_SPEC = importlib.util.spec_from_file_location("_lewm_protected_camera_adaptation_contract", _CONTRACT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("cannot load protected Camera adaptation contract")
contract = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(contract)


def _fingerprint(value: os.stat_result) -> tuple[int, ...]:
    return (int(value.st_dev), int(value.st_ino), int(value.st_mode), int(value.st_size), int(value.st_mtime_ns), int(value.st_ctime_ns))


def _read_regular(path: Path, *, expected_sha256: str | None = None) -> bytes:
    if path.is_symlink():
        raise PermissionError(f"symlink input forbidden: {path}")
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise PermissionError(f"input is not regular: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0))
    try:
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _fingerprint(before) != _fingerprint(after):
        raise RuntimeError(f"input changed while read: {path}")
    raw = b"".join(chunks)
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise PermissionError(f"input hash changed: {path}")
    return raw


def _write_exclusive(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0), 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _publish_json(path: Path, core: Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
    value = contract.with_content_sha256(dict(core))
    raw = contract.canonical_json_bytes(value) + b"\n"
    _write_exclusive(path, raw)
    return value, raw


def _binding(path: str, value: Mapping[str, Any], raw: bytes) -> dict[str, Any]:
    return contract.artifact_binding(path, raw, content_sha256=str(value["content_sha256"]))


def _environment() -> dict[str, Any]:
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        raise PermissionError("protected Camera adaptation requires python -I -B")
    if "torch" in sys.modules or any(name.startswith("torch.") for name in sys.modules):
        raise PermissionError("Torch was imported before reservation")
    if os.environ.get("HIP_VISIBLE_DEVICES") != "0":
        raise PermissionError("protected Camera adaptation requires HIP_VISIBLE_DEVICES=0")
    conflicting = [name for name in contract._v1.CONFLICTING_ACCELERATOR_ENVIRONMENT if name in os.environ]
    threads = {name: os.environ.get(name) for name in contract._v1.THREAD_ENVIRONMENT}
    if conflicting or any(value != "1" for value in threads.values()):
        raise PermissionError("accelerator or native-thread environment changed")
    return {"hip_visible_devices": "0", "conflicting_selectors_absent": True, "native_thread_environment": threads, "isolated_python": True, "bytecode_disabled": True, "torch_module_absent": True}


def _load_authority_pre_reservation(review_sha: str, authorization_sha: str) -> tuple[dict[str, Any], bytes, dict[str, Any], bytes]:
    review_raw = _read_regular(ROOT / contract.REVIEW_RELATIVE_PATH, expected_sha256=review_sha)
    parsed_review = contract.parse_canonical_json(review_raw, name="independent review")
    declared_sources = parsed_review.get("reviewed_sources")
    if type(declared_sources) is not dict:
        raise PermissionError("reviewed sources are absent")
    review = contract.validate_review(parsed_review, expected_sources=declared_sources)
    review_binding = contract.artifact_binding(contract.REVIEW_RELATIVE_PATH, review_raw, content_sha256=review["content_sha256"])
    authorization_raw = _read_regular(ROOT / contract.AUTHORIZATION_RELATIVE_PATH, expected_sha256=authorization_sha)
    authorization = contract.validate_authorization(
        contract.parse_canonical_json(authorization_raw, name="execution authorization"),
        review_binding=review_binding,
        reviewer=str(review["reviewer"]),
    )
    return review, review_raw, authorization, authorization_raw


def _reserve(output_root: Path, review: Mapping[str, Any], review_raw: bytes, authorization: Mapping[str, Any], authorization_raw: bytes, environment: Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
    if output_root.exists() or output_root.is_symlink():
        raise RuntimeError("the sole protected Camera attempt is already reserved or terminal")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(output_root, mode=0o700)
    review_binding = contract.artifact_binding(contract.REVIEW_RELATIVE_PATH, review_raw, content_sha256=review["content_sha256"])
    authorization_binding = contract.artifact_binding(contract.AUTHORIZATION_RELATIVE_PATH, authorization_raw, content_sha256=authorization["content_sha256"])
    attempt = contract.canonical_json_sha256({"schema": f"{contract.SCHEMA_PREFIX}_attempt_identity_v1", "review": review_binding, "authorization": authorization_binding, "science_contract_sha256": contract.canonical_json_sha256(contract.science_contract())})
    core = {
        "schema": contract.RESERVATION_SCHEMA,
        "status": "reserved_before_torch_v4_terminal_update0_terminal_camera_raw_or_rgb",
        "attempt_index": 1,
        "maximum_attempts": 1,
        "attempt_identity": attempt,
        "independent_review": review_binding,
        "execution_authorization": authorization_binding,
        "reviewed_sources": dict(review["reviewed_sources"]),
        "science_contract": contract.science_contract(),
        "environment": dict(environment),
        "output_root_absent_before_reservation": True,
        "output_root_observed_absent_at_authorization": authorization["authority"]["output_root_observed_absent_at_authorization"],
        "torch_imported_before_reservation": False,
        "v4_terminal_opened_before_reservation": False,
        "update0_terminal_opened_before_reservation": False,
        "camera_raw_or_rgb_opened_before_reservation": False,
        "retry_authorized": False,
        "authority": dict(contract.DOWNSTREAM_DENIALS),
    }
    try:
        return _publish_json(output_root / "reservation.json", core)
    except BaseException as error:
        failure = {
            "schema": contract.FAILURE_SCHEMA,
            "status": "failed_reservation_commit",
            "stage": "reservation_commit",
            "attempt_identity": attempt,
            "error": {"type": type(error).__name__, "message": str(error)},
            "torch_imported": False,
            "v4_terminal_opened": False,
            "update0_terminal_opened": False,
            "camera_raw_or_rgb_opened": False,
            "g2_attempted": False,
            "navigation_attempted": False,
            "heldout_open_count": 0,
            "retry_authorized": False,
            "authority": dict(contract.DOWNSTREAM_DENIALS),
        }
        try:
            _publish_json(output_root / "reservation_failed.json", failure)
        except BaseException as terminal_error:
            raise RuntimeError("reservation commit and terminalization both failed") from terminal_error
        raise


def _validate_post_reservation_authority(review: Mapping[str, Any], review_raw: bytes, authorization: Mapping[str, Any], authorization_raw: bytes) -> dict[str, str]:
    sources = contract.current_source_bindings(ROOT)
    observed_review = contract.validate_review(contract.parse_canonical_json(review_raw, name="independent review"), expected_sources=sources)
    review_binding = contract.artifact_binding(contract.REVIEW_RELATIVE_PATH, review_raw, content_sha256=observed_review["content_sha256"])
    observed_authorization = contract.validate_authorization(
        contract.parse_canonical_json(authorization_raw, name="execution authorization"),
        review_binding=review_binding,
        reviewer=str(observed_review["reviewer"]),
    )
    if observed_review != dict(review) or observed_authorization != dict(authorization):
        raise PermissionError("authority changed across reservation")
    return sources


def _update0_terminal() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    audit_raw = _read_regular(ROOT / contract.UPDATE0_AUDIT_RELATIVE_PATH, expected_sha256=contract.UPDATE0_AUDIT_BINDING["file_sha256"])
    if len(audit_raw) != contract.UPDATE0_AUDIT_BINDING["byte_count"]:
        raise PermissionError("update0 audit byte count changed")
    audit = contract.validate_update0_audit(audit_raw)
    root = ROOT / contract.UPDATE0_ROOT_RELATIVE_PATH
    entries = list(root.iterdir()) if root.is_dir() and not root.is_symlink() else []
    if sorted(item.name for item in entries) != sorted(item["path"] for item in contract.UPDATE0_TERMINAL_ARTIFACTS.values()) or any(item.is_symlink() or not item.is_file() for item in entries):
        raise PermissionError("update0 exact four-file terminal inventory changed")
    records = []
    for kind, binding in sorted(contract.UPDATE0_TERMINAL_ARTIFACTS.items()):
        raw = _read_regular(root / binding["path"], expected_sha256=binding["file_sha256"])
        value = contract.parse_canonical_json(raw, name=f"update0 {kind}")
        if len(raw) != binding["byte_count"] or value.get("content_sha256") != binding["content_sha256"] or value.get("schema") != binding["schema"]:
            raise PermissionError(f"update0 {kind} binding changed")
        records.append({"kind": kind, **binding})
    return audit, records


def _load_diagnostic_runner() -> Any:
    relative = contract.DIAGNOSTIC_RUNNER_RELATIVE_PATH
    path = ROOT / relative
    raw = _read_regular(path, expected_sha256=contract.DIAGNOSTIC_SOURCE_SHA256[relative])
    spec = importlib.util.spec_from_file_location("_lewm_protected_camera_exact_update0_runner", path)
    if spec is None or spec.loader is None or not raw:
        raise ImportError("cannot load exact update0 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _subset_sha(runtime: Any, model: Any, prefixes: Sequence[str]) -> str:
    state = {name: value for name, value in model.state_dict().items() if name.startswith(tuple(prefixes))}
    if not state:
        raise RuntimeError("protected state subset is empty")
    return runtime.model_module.tensor_state_dict_sha256(state)


def _prepare_model(diagnostic: Any, runtime: Any, initial_state: Mapping[str, Any], device: Any) -> tuple[Any, list[Any], list[Any], list[Any], dict[str, Any]]:
    model = diagnostic._new_model(runtime, initial_state, device, train=True)
    model.requires_grad_(False)
    groups: dict[str, list[tuple[str, Any]]] = {"encoder": [], "evidence_head": [], "frozen": []}
    for name in model.state_dict():
        contract.parameter_partition(name)
    for name, parameter in model.named_parameters():
        component = contract.parameter_partition(name)
        if component in {"encoder", "evidence_head"}:
            parameter.requires_grad_(True)
            groups[component].append((name, parameter))
        else:
            groups["frozen"].append((name, parameter))
    counts = {name: sum(int(parameter.numel()) for _, parameter in groups[name]) for name in ("encoder", "evidence_head")}
    tensor_counts = {name: len(groups[name]) for name in ("encoder", "evidence_head")}
    if counts != contract.EXPECTED_PARAMETER_COUNTS or tensor_counts != contract.EXPECTED_PARAMETER_TENSOR_COUNTS or not groups["frozen"]:
        raise PermissionError("protected parameter partition changed")
    names = {name: [item for item, _ in groups[name]] for name in groups}
    if set(names["encoder"]) & set(names["evidence_head"]) or any(parameter.requires_grad for _, parameter in groups["frozen"]):
        raise PermissionError("protected trainable groups overlap or leaked")
    return model, [parameter for _, parameter in groups["evidence_head"]], [parameter for _, parameter in groups["encoder"]], [parameter for _, parameter in groups["frozen"]], {"parameter_counts": counts, "parameter_tensor_counts": tensor_counts, "parameter_names_sha256": {name: contract.canonical_json_sha256(values) for name, values in names.items()}}


def _assert_frozen_grads_none(frozen: Sequence[Any]) -> None:
    if any(parameter.grad is not None for parameter in frozen):
        raise RuntimeError("a protected frozen parameter acquired a gradient")


def _gradient_group_norm(runtime: Any, parameters: Sequence[Any], group: str, *, maximum: float | None = None) -> float:
    expected = contract.EXPECTED_PARAMETER_TENSOR_COUNTS.get(group)
    if expected is None or len(parameters) != expected:
        raise RuntimeError(f"{group} gradient tensor count changed")
    gradients = [parameter.grad for parameter in parameters]
    if any(gradient is None for gradient in gradients):
        raise RuntimeError(f"{group} parameter has no gradient")
    if not bool(runtime.torch.stack([runtime.torch.isfinite(gradient).all() for gradient in gradients]).all().item()):
        raise FloatingPointError(f"{group} parameter gradient became nonfinite")
    squared = runtime.torch.stack([gradient.detach().float().square().sum() for gradient in gradients]).sum()
    norm = math.sqrt(float(squared.detach().cpu()))
    if not math.isfinite(norm):
        raise FloatingPointError(f"{group} aggregate gradient norm became nonfinite")
    if maximum is not None and norm > maximum + contract.POST_CLIP_NORM_ASSERTION_TOLERANCE:
        raise RuntimeError(f"{group} post-clip gradient norm exceeds {maximum}")
    return norm


def _camera_pair(runtime: Any, model: Any, batch: Mapping[str, Any]) -> Any:
    forward = batch["forward"]
    current = model.forward_frame(forward["current_image"], forward["current_camera_origin_body_m"], forward["current_camera_basis_body_fru"], forward["current_ground_plane_z_body_m"])
    next_frame = model.forward_frame(forward["next_image"], forward["next_camera_origin_body_m"], forward["next_camera_basis_body_fru"], forward["next_ground_plane_z_body_m"])
    overlap = runtime.torch.ones_like(current.bev[:, :1], dtype=runtime.torch.bool)
    return runtime.model_module.SharedTrainingPairV5(
        current=current,
        next=next_frame,
        predicted_next_bev=next_frame.bev,
        stop_gradient_target_next_bev=next_frame.bev.detach(),
        commanded_warped_current_bev=current.bev,
        commanded_overlap_mask=overlap,
        realized_warped_current_bev=current.bev,
        realized_overlap_mask=overlap,
        jepa=None,
    )


def _scalar(value: Any) -> float:
    result = float(value.detach().cpu())
    if not math.isfinite(result):
        raise FloatingPointError("Camera adaptation scalar became nonfinite")
    return result


def _camera_components(loss: Any) -> dict[str, float]:
    result = {"camera_total": _scalar(loss.total)}
    for side in ("current", "next"):
        frame = getattr(loss, side)
        result.update({
            f"{side}_hierarchical_first_hit_nll": _scalar(frame.hierarchical_first_hit_nll),
            f"{side}_target_bin_offset_smooth_l1": _scalar(frame.target_bin_offset_smooth_l1),
            f"{side}_ground_clear_distance_state_balanced_bce": _scalar(frame.ground_clear_distance_state_balanced_bce),
            f"{side}_derived_raster_hierarchical_bce": _scalar(frame.derived_raster_hierarchical_bce.total),
            f"{side}_derived_raster_cell_nll": _scalar(frame.derived_raster_cell_nll),
        })
    return result


def _snapshot(runtime: Any, model: Any, output_root: Path, *, update: int, frozen_sha: str) -> dict[str, Any]:
    state = {name: value.detach().cpu().contiguous().clone() for name, value in sorted(model.state_dict().items())}
    state_sha = runtime.model_module.tensor_state_dict_sha256(state)
    frozen_observed = runtime.model_module.tensor_state_dict_sha256({name: value for name, value in state.items() if name.startswith(contract.FROZEN_STATE_PREFIXES)})
    trainable_sha = runtime.model_module.tensor_state_dict_sha256({name: value for name, value in state.items() if name.startswith(contract.TRAINABLE_PARAMETER_PREFIXES)})
    if frozen_observed != frozen_sha:
        raise RuntimeError("frozen state changed before snapshot")
    semantic = {
        "schema": contract.SNAPSHOT_SCHEMA,
        "update": update,
        "model_config": model.model_config.to_dict(),
        "state_sha256": state_sha,
        "frozen_state_sha256": frozen_sha,
        "trainable_state_sha256": trainable_sha,
        "initialization_state_sha256": contract.UPDATE0_STATE_SHA256,
        "schedule_prefix_indices_sha256": contract.CHECKPOINT_SCHEDULE_PREFIX_SHA256[update],
        "optimizer_contract": contract.OPTIMIZER_CONTRACT,
        "development_only": True,
        "resume_authorized": False,
        "runtime_ready": False,
    }
    content_sha = contract.canonical_json_sha256(semantic)
    buffer = io.BytesIO()
    runtime.torch.save({**semantic, "content_sha256": content_sha, "model_state_dict": state}, buffer)
    raw = buffer.getvalue()
    relative = f"checkpoints/update_{update}.pt"
    _write_exclusive(output_root / relative, raw)
    return {"path": relative, "file_sha256": hashlib.sha256(raw).hexdigest(), "content_sha256": content_sha, "byte_count": len(raw), "state_sha256": state_sha, "frozen_state_sha256": frozen_sha, "trainable_state_sha256": trainable_sha}


def _evaluate(runtime: Any, trainer: Any, model: Any, selection_pairs: Sequence[Mapping[str, Any]], device: Any, *, update: int, frozen_sha: str) -> dict[str, Any]:
    before = diagnostic_state = runtime.model_module.tensor_state_dict_sha256(model.state_dict())
    if _subset_sha(runtime, model, contract.FROZEN_STATE_PREFIXES) != frozen_sha:
        raise RuntimeError("frozen state changed before physical evaluation")
    model.eval()
    physical, camera_loss = trainer.physical_metrics(model, selection_pairs, device, arm="protected_camera_adaptation", stage=f"checkpoint_selection_update_{update}")
    model.train()
    after = runtime.model_module.tensor_state_dict_sha256(model.state_dict())
    frozen_after = _subset_sha(runtime, model, contract.FROZEN_STATE_PREFIXES)
    if before != after or frozen_after != frozen_sha:
        raise RuntimeError("physical evaluation mutated protected model state")
    evaluated = contract.evaluate_physical_scopes(physical)
    return {"update": update, "role": "checkpoint_selection", "pair_count": 495, "unique_endpoint_count": 924, "scopes": physical, "aggregate_complete_v4_loss": float(camera_loss), "evaluation": evaluated, "state_sha256_before": diagnostic_state, "state_sha256_after": after, "frozen_state_sha256_before_and_after": frozen_sha, "state_mutation_count": 0}


def _validate_schedule(indices: Sequence[int]) -> None:
    if len(indices) < contract.MAXIMUM_UPDATE * 16:
        raise PermissionError("bound V4 schedule is shorter than the protected prefix")
    for update, expected in contract.CHECKPOINT_SCHEDULE_PREFIX_SHA256.items():
        observed = contract.canonical_json_sha256(list(indices[: update * 16]))
        if observed != expected:
            raise PermissionError(f"bound V4 schedule prefix changed at update {update}")


def _train(runtime: Any, trainer: Any, model: Any, head: Sequence[Any], encoder: Sequence[Any], frozen: Sequence[Any], train_pairs: Sequence[Mapping[str, Any]], selection_pairs: Sequence[Mapping[str, Any]], indices: Sequence[int], vocabulary: Sequence[str], commanded: Any, device: Any, output_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int | None, dict[str, Any], dict[str, Any]]:
    frozen_sha = _subset_sha(runtime, model, contract.FROZEN_STATE_PREFIXES)
    optimizer = runtime.torch.optim.AdamW(
        [{"params": list(head), "lr": contract.learning_rates(1)[0], "group_name": "evidence_head"}, {"params": list(encoder), "lr": contract.learning_rates(1)[1], "group_name": "encoder"}],
        betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4, amsgrad=False,
    )
    trace: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
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
                _assert_frozen_grads_none(frozen)
                optimizer.zero_grad(set_to_none=True)
                sums: dict[str, float] = {}
                start = (update - 1) * 16
                update_indices = indices[start : start + 16]
                for micro in range(4):
                    batch = trainer.batch(train_pairs, update_indices[micro * 4 : (micro + 1) * 4], vocabulary, commanded, device, role="train", arm="protected_camera_adaptation", stage="camera_gradient")
                    pair = _camera_pair(runtime, model, batch)
                    camera = runtime.loss_adapter.observable_camera_ray_v4_loss_v4(model, pair, batch["current_supervision"], batch["next_supervision"])
                    if not bool(runtime.torch.isfinite(camera.total).item()):
                        raise FloatingPointError("Camera-only backward scalar became nonfinite")
                    (camera.total / 4.0).backward()
                    for name, value in _camera_components(camera).items():
                        sums[name] = sums.get(name, 0.0) + value / 4.0
                _assert_frozen_grads_none(frozen)
                head_pre_clip = _gradient_group_norm(runtime, head, "evidence_head")
                encoder_pre_clip = _gradient_group_norm(runtime, encoder, "encoder")
                head_norm = runtime.torch.nn.utils.clip_grad_norm_(head, max_norm=1.0)
                encoder_norm = runtime.torch.nn.utils.clip_grad_norm_(encoder, max_norm=1.0)
                if not bool(runtime.torch.isfinite(head_norm).item()) or not bool(runtime.torch.isfinite(encoder_norm).item()):
                    raise FloatingPointError("protected group gradient norm became nonfinite")
                head_post_clip = _gradient_group_norm(runtime, head, "evidence_head", maximum=1.0)
                encoder_post_clip = _gradient_group_norm(runtime, encoder, "encoder", maximum=1.0)
                optimizer.step()
                _assert_frozen_grads_none(frozen)
                trace.append({"schema": f"{contract.SCHEMA_PREFIX}_trace_row_v1", "update": update, "presentation_indices_sha256": contract.canonical_json_sha256(list(update_indices)), "head_learning_rate": head_lr, "encoder_learning_rate": encoder_lr, "microbatch_count": 4, "camera_backward_count": 4, "jepa_objective_count": 0, "jepa_backward_count": 0, "optimizer_step_count": update, "head_clip_invocation_count": update, "encoder_clip_invocation_count": update, "head_gradient_tensor_count": len(head), "encoder_gradient_tensor_count": len(encoder), "head_gradient_norm_before_clip": head_pre_clip, "encoder_gradient_norm_before_clip": encoder_pre_clip, "head_clip_return_norm": _scalar(head_norm), "encoder_clip_return_norm": _scalar(encoder_norm), "head_gradient_norm_after_clip": head_post_clip, "encoder_gradient_norm_after_clip": encoder_post_clip, "post_clip_group_norm_maximum": 1.0, "post_clip_norm_assertion_tolerance": contract.POST_CLIP_NORM_ASSERTION_TOLERANCE, "losses": sums, "ema_update_count": 0})
                if update in contract.CHECKPOINT_UPDATES:
                    if _subset_sha(runtime, model, contract.FROZEN_STATE_PREFIXES) != frozen_sha:
                        raise RuntimeError("frozen state changed during protected training")
                    snapshots.append(_snapshot(runtime, model, output_root, update=update, frozen_sha=frozen_sha))
                    metric = _evaluate(runtime, trainer, model, selection_pairs, device, update=update, frozen_sha=frozen_sha)
                    metrics.append(metric)
                    if metric["evaluation"]["all_nine_physical_pass"]:
                        selected = update
                        break
        finally:
            warnings.showwarning = original
    warning_receipt = collector.receipt()
    if warning_receipt["warning_count"] <= 0 or _subset_sha(runtime, model, contract.FROZEN_STATE_PREFIXES) != frozen_sha:
        raise RuntimeError("warning or frozen-state guard failed")
    operation_counts = {"complete_update_count": len(trace), "camera_objective_count": len(trace) * 4, "camera_backward_count": len(trace) * 4, "optimizer_construction_count": 1, "optimizer_step_count": len(trace), "head_clip_invocation_count": len(trace), "encoder_clip_invocation_count": len(trace), "total_clip_invocation_count": len(trace) * 2, "global_clip_invocation_count": 0, "trainable_gradient_tensor_checks_per_update": sum(contract.EXPECTED_PARAMETER_TENSOR_COUNTS.values()) * 2, "all_trainable_gradients_present_and_finite_before_clip": True, "all_trainable_gradients_finite_after_clip": True, "all_post_clip_group_norms_at_most_one_with_declared_fp32_tolerance": True, "post_clip_norm_assertion_tolerance": contract.POST_CLIP_NORM_ASSERTION_TOLERANCE, "jepa_objective_count": 0, "jepa_backward_count": 0, "ema_update_count": 0, "physical_selection_count": len(metrics), "frozen_state_mutation_count": 0}
    return trace, metrics, snapshots, selected, warning_receipt, {"frozen_state_sha256": frozen_sha, "final_state_sha256": runtime.model_module.tensor_state_dict_sha256(model.state_dict()), "operation_counts": operation_counts}


def _publish_training(output_root: Path, trace: Sequence[Mapping[str, Any]], metrics: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    trace_raw = b"".join(contract.canonical_json_bytes(row) + b"\n" for row in trace)
    _write_exclusive(output_root / "training_trace.jsonl", trace_raw)
    trace_binding = {"path": "training_trace.jsonl", "file_sha256": hashlib.sha256(trace_raw).hexdigest(), "content_sha256": contract.canonical_json_sha256(list(trace)), "byte_count": len(trace_raw), "row_count": len(trace)}
    updates = [int(row["update"]) for row in metrics]
    contract.validate_checkpoint_prefix(updates)
    metric_value, metric_raw = _publish_json(output_root / "checkpoint_metrics.json", {"schema": contract.METRICS_SCHEMA, "status": "fixed_prefix_evaluated", "checkpoint_updates": updates, "rows": list(metrics), "selection_rule": "earliest_all_nine_physical_pass", "soft_or_closest_promotion_authorized": False})
    return trace_binding, _binding("checkpoint_metrics.json", metric_value, metric_raw)


def _access_receipt(inputs: Any, authorization: Mapping[str, Any], reservation: Mapping[str, Any], sources: Mapping[str, str], update0_records: Sequence[Mapping[str, Any]], v4_audit: Mapping[str, Any], v4_rehash: Mapping[str, Any]) -> dict[str, Any]:
    consumed = inputs.rehash_consumed()
    allowed = {"authority", "index", "train", "checkpoint_selection"}
    observed_roles = {role for item in consumed["records"] for role in item["roles"]}
    if any(not set(item["roles"]).issubset(allowed) for item in consumed["records"]) or not {"train", "checkpoint_selection"}.issubset(observed_roles):
        raise PermissionError("protected adaptation consumed an unauthorized dataset role")
    if contract.current_source_bindings(ROOT) != dict(sources):
        raise PermissionError("a reviewed source changed during protected adaptation")
    authority_records = []
    for kind in ("independent_review", "execution_authorization"):
        binding = reservation[kind]
        raw = _read_regular(ROOT / binding["path"], expected_sha256=binding["file_sha256"])
        if len(raw) != binding["byte_count"]:
            raise PermissionError("review or authorization byte count changed")
        authority_records.append({"kind": kind, **binding})
    camera_records = []
    for kind in ("gate", "checkpoint"):
        binding = authorization["camera"][kind]
        raw = _read_regular(ROOT / binding["path"], expected_sha256=binding["file_sha256"])
        if len(raw) != binding["byte_count"]:
            raise PermissionError("Camera artifact byte count changed")
        camera_records.append({"kind": kind, **binding})
    return {
        "schema": contract.ACCESS_SCHEMA,
        "status": "all_consumed_development_inputs_rehashed",
        "roles_opened": ["train", "checkpoint_selection"],
        "probability_calibration_open_count": 0,
        "consumed": consumed,
        "camera": {"records": camera_records, "all_rehashed": True},
        "reviewed_sources": {"count": len(sources), "bindings": dict(sources), "all_rehashed": True},
        "review_and_authorization": {"records": authority_records, "all_rehashed": True},
        "update0_predecessor": {"terminal_audit": dict(contract.UPDATE0_AUDIT_BINDING), "terminal_records": list(update0_records), "all_rehashed": True},
        "v4_predecessor": {"audit_content_sha256": v4_audit["content_sha256"], "terminal_inventory_rehash": dict(v4_rehash), "initialization": dict(contract._diagnostic.V4_INITIALIZATION_BINDING), "schedule": dict(contract._diagnostic.V4_SCHEDULE_BINDING)},
        "g2_open_count": 0,
        "navigation_open_count": 0,
        "heldout_open_count": 0,
        "all_consumed_files_rehashed": True,
        "authority": dict(contract.DOWNSTREAM_DENIALS),
    }


def _existing_artifact_bindings(output_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    entries = list(output_root.rglob("*"))
    if any(item.is_symlink() or not (item.is_file() or item.is_dir()) for item in entries):
        raise PermissionError("failure inventory contains a symlink or special file")
    records = []
    for path in sorted((item for item in entries if item.is_file() and item.name != "failed.json"), key=lambda item: item.relative_to(output_root).as_posix()):
        relative = path.relative_to(output_root).as_posix()
        raw = _read_regular(path)
        record = {"path": relative, "file_sha256": hashlib.sha256(raw).hexdigest(), "byte_count": len(raw)}
        if path.suffix == ".json":
            try:
                value = contract.parse_canonical_json(raw, name=f"published {relative}")
            except (TypeError, ValueError, PermissionError):
                value = {}
            if contract._v1.is_sha256(value.get("content_sha256")):
                record["content_sha256"] = value["content_sha256"]
        records.append(record)
    directories = [".", *sorted(item.relative_to(output_root).as_posix() for item in entries if item.is_dir())]
    return records, directories


def _terminal_failure(output_root: Path, reservation: Mapping[str, Any], stage: str, error: BaseException, published: Mapping[str, Any], *, numeric: bool = False) -> None:
    records, directories = _existing_artifact_bindings(output_root)
    artifacts = {record["path"]: record for record in records}
    reservation_binding = artifacts.get("reservation.json")
    if type(reservation_binding) is not dict or reservation_binding.get("content_sha256") != reservation["content_sha256"]:
        raise RuntimeError("failure inventory cannot bind the committed reservation")
    paths = [record["path"] for record in records]
    prefix = ["reservation.json", *(path for path in paths if path != "reservation.json")]
    core = {"schema": contract.FAILURE_SCHEMA, "status": "failed_numeric_physical_gate" if numeric else "failed_protected_camera_adaptation", "stage": stage, "attempt_identity": reservation["attempt_identity"], "published_prefix": prefix, "artifacts": artifacts, "caller_ledger_paths": list(published), "exact_pre_failure_directories_including_root": directories, "all_existing_regular_artifacts_bound": True, "error": {"type": type(error).__name__, "message": str(error)}, "closest_or_soft_promotion": False, "extension_or_retry_authorized": False, "g2_attempted": False, "navigation_attempted": False, "heldout_open_count": 0, "authority": dict(contract.DOWNSTREAM_DENIALS)}
    try:
        _publish_json(output_root / "failed.json", core)
    except FileExistsError:
        pass


def _terminal_paths(output_root: Path) -> tuple[list[str], list[str]]:
    entries = list(output_root.rglob("*"))
    if any(item.is_symlink() for item in entries):
        raise PermissionError("output inventory contains a symlink")
    files = sorted(item.relative_to(output_root).as_posix() for item in entries if item.is_file())
    directories = [".", *sorted(item.relative_to(output_root).as_posix() for item in entries if item.is_dir())]
    return files, directories


def run_parent(*, review_file_sha256: str, authorization_file_sha256: str) -> int:
    environment = _environment()
    review, review_raw, authorization, authorization_raw = _load_authority_pre_reservation(review_file_sha256, authorization_file_sha256)
    output_root = ROOT / contract.OUTPUT_ROOT_RELATIVE_PATH
    reservation, reservation_raw = _reserve(output_root, review, review_raw, authorization, authorization_raw, environment)
    stage, published = "post_reservation_source_and_authority_validation", {}
    try:
        sources = _validate_post_reservation_authority(review, review_raw, authorization, authorization_raw)
        stage = "update0_exact_terminal_rehash"
        _, update0_records = _update0_terminal()
        stage = "exact_stack_and_v4_terminal_load"
        diagnostic = _load_diagnostic_runner()
        stack = diagnostic._load_v4_stack()
        v4_audit, initialization, schedule = diagnostic._predecessor_inputs()
        stage = "post_reservation_torch_import"
        runtime = stack._load_runtime()
        stage = "update0_reconstruction"
        inputs, trainer, device, initial_state, train_pairs, selection_pairs, indices, vocabulary, commanded, provenance = diagnostic._reconstruct(stack, runtime, authorization, output_root, reservation, initialization, schedule)
        _validate_schedule(indices)
        model, head, encoder, frozen, partition = _prepare_model(diagnostic, runtime, initial_state, device)
        stage = "protected_camera_training_and_fixed_selection"
        trace, metrics, snapshots, selected, warning_receipt, state = _train(runtime, trainer, model, head, encoder, frozen, train_pairs, selection_pairs, indices, vocabulary, commanded, device, output_root)
        for snapshot in snapshots:
            published[snapshot["path"]] = snapshot
        trace_binding, metrics_binding = _publish_training(output_root, trace, metrics)
        published["training_trace.jsonl"] = trace_binding
        published["checkpoint_metrics.json"] = metrics_binding
        stage = "all_input_rehash"
        access_core = _access_receipt(inputs, authorization, reservation, sources, update0_records, v4_audit, diagnostic._rehash_v4_terminal(v4_audit))
        access, access_raw = _publish_json(output_root / "access.json", {**access_core, "reservation": _binding("reservation.json", reservation, reservation_raw)})
        published["access.json"] = _binding("access.json", access, access_raw)
        if selected is None:
            error = RuntimeError("no fixed checkpoint passed all nine physical scopes by update 4000")
            _terminal_failure(output_root, reservation, "scientific_numeric_physical_gate", error, published, numeric=True)
            return 2
        if selected != next(row["update"] for row in metrics if row["evaluation"]["all_nine_physical_pass"]):
            raise RuntimeError("selected checkpoint is not the earliest physical pass")
        stage = "result_publication"
        selected_snapshot = next(item for item in snapshots if item["path"] == f"checkpoints/update_{selected}.pt")
        result, result_raw = _publish_json(output_root / "result.json", {
            "schema": contract.RESULT_SCHEMA,
            "status": "completed_protected_camera_adaptation_earliest_physical_pass",
            "reservation": _binding("reservation.json", reservation, reservation_raw),
            "access": _binding("access.json", access, access_raw),
            "selected_update": selected,
            "selected_checkpoint": selected_snapshot,
            "checkpoint_metrics": metrics_binding,
            "training_trace": trace_binding,
            "evaluated_checkpoint_updates": [row["update"] for row in metrics],
            "selection_rule": "earliest_all_nine_physical_pass",
            "partition": partition,
            "state": state,
            "warnings": warning_receipt,
            "provenance": provenance,
            "jepa_objective_count": 0,
            "jepa_backward_count": 0,
            "ema_update_count": 0,
            "runtime_ready": False,
            "authority": dict(contract.DOWNSTREAM_DENIALS),
        })
        published["result.json"] = _binding("result.json", result, result_raw)
        stage = "completion_publication"
        files, directories = _terminal_paths(output_root)
        expected_files = sorted(["reservation.json", *published])
        if files != expected_files or directories != [".", "checkpoints"]:
            raise PermissionError("pre-completion protected adaptation inventory changed")
        _publish_json(output_root / "completed.json", {
            "schema": contract.COMPLETION_SCHEMA,
            "status": "complete_camera_checkpoint_only_no_downstream_authority",
            "attempt_identity": reservation["attempt_identity"],
            "selected_update": selected,
            "selected_checkpoint": selected_snapshot,
            "exact_terminal_files": sorted([*files, "completed.json"]),
            "exact_terminal_directories_including_root": directories,
            "all_inputs_rehashed": True,
            "operation_counts": state["operation_counts"],
            "heldout_open_count": 0,
            "retry_authorized": False,
            "runtime_ready": False,
            "authority": dict(contract.DOWNSTREAM_DENIALS),
        })
        return 0
    except BaseException as error:
        _terminal_failure(output_root, reservation, stage, error, published)
        raise


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
