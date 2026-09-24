#!/usr/bin/env python3
"""Run one frozen-data update-zero transfer and gradient diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Callable, Mapping, Sequence
import warnings


ROOT = Path(__file__).resolve().parents[1]
_CONTRACT_PATH = ROOT / "lewm/benchmarks/go2_shared_jepa_v5_update0_transfer_gradient_diagnostic_v1.py"
_SPEC = importlib.util.spec_from_file_location("_lewm_update0_transfer_gradient_diagnostic_contract", _CONTRACT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("cannot load update-zero diagnostic contract")
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


def _binding(relative: str, value: Mapping[str, Any], raw: bytes) -> dict[str, Any]:
    return contract.artifact_binding(relative, raw, content_sha256=str(value["content_sha256"]))


def _read_bound_json(root: Path, binding: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    bound = contract.validate_binding(binding)
    raw = _read_regular(root / bound["path"], expected_sha256=bound["file_sha256"])
    if len(raw) != bound["byte_count"]:
        raise PermissionError(f"bound byte count changed: {name}")
    value = contract.parse_canonical_json(raw, name=name)
    if value["content_sha256"] != bound["content_sha256"]:
        raise PermissionError(f"bound content hash changed: {name}")
    return value


def _load_review_authorization(review_sha: str, authorization_sha: str) -> tuple[dict[str, Any], bytes, dict[str, Any], bytes, dict[str, str]]:
    sources = contract.current_source_bindings(ROOT)
    review_raw = _read_regular(ROOT / contract.REVIEW_RELATIVE_PATH, expected_sha256=review_sha)
    review = contract.validate_review(contract.parse_canonical_json(review_raw, name="independent review"), expected_sources=sources)
    review_binding = contract.artifact_binding(contract.REVIEW_RELATIVE_PATH, review_raw, content_sha256=review["content_sha256"])
    authorization_raw = _read_regular(ROOT / contract.AUTHORIZATION_RELATIVE_PATH, expected_sha256=authorization_sha)
    authorization = contract.validate_authorization(contract.parse_canonical_json(authorization_raw, name="execution authorization"), review_binding=review_binding)
    return review, review_raw, authorization, authorization_raw, sources


def _environment() -> dict[str, Any]:
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        raise PermissionError("exact diagnostic requires python -I -B")
    if "torch" in sys.modules or any(name.startswith("torch.") for name in sys.modules):
        raise PermissionError("Torch was imported before reservation")
    if os.environ.get("HIP_VISIBLE_DEVICES") != "0":
        raise PermissionError("exact diagnostic requires HIP_VISIBLE_DEVICES=0")
    conflicting = [name for name in contract._v1.CONFLICTING_ACCELERATOR_ENVIRONMENT if name in os.environ]
    threads = {name: os.environ.get(name) for name in contract._v1.THREAD_ENVIRONMENT}
    if conflicting or any(value != "1" for value in threads.values()):
        raise PermissionError("accelerator or native-thread environment changed")
    return {"hip_visible_devices": "0", "conflicting_selectors_absent": True, "native_thread_environment": threads, "isolated_python": True, "bytecode_disabled": True, "torch_module_absent": True}


def _reserve(output_root: Path, review: Mapping[str, Any], review_raw: bytes, authorization: Mapping[str, Any], authorization_raw: bytes, sources: Mapping[str, str], environment: Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
    if output_root.exists() or output_root.is_symlink():
        raise RuntimeError("the one diagnostic attempt is already reserved or terminal")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(output_root, mode=0o700)
    review_binding = contract.artifact_binding(contract.REVIEW_RELATIVE_PATH, review_raw, content_sha256=review["content_sha256"])
    authorization_binding = contract.artifact_binding(contract.AUTHORIZATION_RELATIVE_PATH, authorization_raw, content_sha256=authorization["content_sha256"])
    attempt = contract.canonical_json_sha256({"schema": f"{contract.SCHEMA_PREFIX}_attempt_identity_v1", "review": review_binding, "authorization": authorization_binding, "science_contract_sha256": contract.canonical_json_sha256(contract.science_contract())})
    core = {
        "schema": contract.RESERVATION_SCHEMA,
        "status": "reserved_before_torch_v4_terminal_camera_raw_or_rgb",
        "attempt_index": 1,
        "maximum_attempts": 1,
        "attempt_identity": attempt,
        "independent_review": review_binding,
        "execution_authorization": authorization_binding,
        "reviewed_sources": dict(sources),
        "science_contract": contract.science_contract(),
        "predecessor_audit": dict(contract.V4_TERMINAL_AUDIT_BINDING),
        "raw": authorization["raw"],
        "camera": authorization["camera"],
        "environment": dict(environment),
        "torch_imported_before_reservation": False,
        "v4_terminal_opened_before_reservation": False,
        "camera_raw_or_rgb_opened_before_reservation": False,
        "retry_authorized": False,
        "authority": dict(contract.DOWNSTREAM_DENIALS),
    }
    try:
        return _publish_json(output_root / "reservation.json", core)
    except BaseException as error:
        failure = {"schema": contract.FAILURE_SCHEMA, "status": "failed_reservation_commit", "stage": "reservation_commit", "attempt_identity": attempt, "error": {"type": type(error).__name__, "message": str(error)}, "torch_imported": False, "v4_terminal_opened": False, "camera_raw_or_rgb_opened": False, "g2_attempted": False, "navigation_attempted": False, "heldout_open_count": 0, "retry_authorized": False, "authority": dict(contract.DOWNSTREAM_DENIALS)}
        try:
            _publish_json(output_root / "reservation_failed.json", failure)
        except BaseException as terminal_error:
            raise RuntimeError("reservation commit and terminalization both failed") from terminal_error
        raise


def _load_v4_stack() -> Any:
    path = ROOT / contract.V4_RUNNER_RELATIVE_PATH
    if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != contract.V4_SOURCE_SHA256[contract.V4_RUNNER_RELATIVE_PATH]:
        raise PermissionError("the frozen V4 runner changed")
    spec = importlib.util.spec_from_file_location("_lewm_update0_diagnostic_installed_v4_runner", path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load the frozen V4 runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    stack = module.install()
    if stack.RawInputs.__name__ != "RawInputsV3" or stack.Trainer.__name__ != "TrainerV4":
        raise PermissionError("the exact V2/V3/V4 overlays were not installed")
    return stack


def _predecessor_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    audit_raw = _read_regular(ROOT / contract.V4_TERMINAL_AUDIT_RELATIVE_PATH, expected_sha256=contract.V4_TERMINAL_AUDIT_BINDING["file_sha256"])
    audit = contract.validate_v4_audit(audit_raw)
    root = ROOT / contract.V4_ROOT_RELATIVE_PATH
    initialization = _read_bound_json(root, contract.V4_INITIALIZATION_BINDING, name="V4 initialization")
    schedule = _read_bound_json(root, contract.V4_SCHEDULE_BINDING, name="V4 schedule")
    if initialization.get("schema") != contract._v4.INITIALIZATION_SCHEMA or initialization.get("complete_state_sha256") != contract.UPDATE0_STATE_SHA256 or schedule.get("schema") != contract._v4.SCHEDULE_SCHEMA or tuple(schedule.get("presentation_indices", ())[:16]) != contract.FIRST_PRESENTATION_INDICES:
        raise PermissionError("the bound V4 update-zero identity or first presentations changed")
    return audit, initialization, schedule


def _rehash_v4_terminal(audit: Mapping[str, Any]) -> dict[str, Any]:
    root = ROOT / contract.V4_ROOT_RELATIVE_PATH
    declared = {item["path"]: item for item in audit["terminal_inventory"]["artifacts"]}
    entries = list(root.rglob("*")) if root.is_dir() and not root.is_symlink() else []
    files = {item.relative_to(root).as_posix(): item for item in entries if item.is_file() and not item.is_symlink()}
    directories = {".", *(item.relative_to(root).as_posix() for item in entries if item.is_dir() and not item.is_symlink())}
    expected_directories = {"."} | {parent.as_posix() for name in declared for parent in Path(name).parents if parent.as_posix() != "."}
    if set(files) != set(declared) or directories != expected_directories or len(files) != 14 or any(item.is_symlink() for item in entries):
        raise PermissionError("the exact V4 terminal inventory changed")
    observed = []
    for name in sorted(files):
        raw = _read_regular(files[name], expected_sha256=declared[name]["file_sha256"])
        if len(raw) != declared[name]["byte_count"]:
            raise PermissionError("a V4 terminal artifact byte count changed")
        observed.append({"path": name, "file_sha256": hashlib.sha256(raw).hexdigest(), "byte_count": len(raw)})
    return {"exact_file_count": 14, "exact_directory_count_including_root": len(directories), "all_files_rehashed": True, "records_sha256": contract.canonical_json_sha256(observed), "records": observed}


def _state_sha(runtime: Any, model: Any) -> str:
    return runtime.model_module.tensor_state_dict_sha256(model.state_dict())


def _new_model(runtime: Any, initial_state: Mapping[str, Any], device: Any, *, train: bool) -> Any:
    runtime.torch.manual_seed(contract._v1.INITIALIZATION_SEED)
    runtime.torch.cuda.manual_seed_all(contract._v1.INITIALIZATION_SEED)
    model = runtime.model_module.SharedObservableCameraRayJepaV5().to(device)
    model.load_state_dict(initial_state, strict=True)
    model.train(train)
    if _state_sha(runtime, model) != contract.UPDATE0_STATE_SHA256:
        raise PermissionError("fresh model is not exact update zero")
    return model


def _reconstruct(stack: Any, runtime: Any, authorization: Mapping[str, Any], output_root: Path, reservation: Mapping[str, Any], initialization: Mapping[str, Any], schedule: Mapping[str, Any]) -> tuple[Any, Any, Any, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[int], list[str], Any, dict[str, Any]]:
    fit, gate, camera_binding = stack._camera_model_after_reservation(runtime, authorization)
    inputs = stack.RawInputs(runtime, authorization)
    trainer = stack.Trainer(runtime, inputs, output_root, reservation)
    device, resource = trainer.device()
    initial_state, receipt = trainer.initialize(fit)
    del fit
    train_pairs, selection_pairs = inputs.role_pairs("train"), inputs.role_pairs("checkpoint_selection")
    vocabulary, commanded_cpu = trainer.commanded_table(train_pairs)
    observed_initialization = contract.with_content_sha256({**receipt, "primitive_vocabulary": vocabulary, "commanded_delta_table": commanded_cpu.tolist(), "commanded_delta_table_sha256": contract.canonical_json_sha256(commanded_cpu.tolist())})
    if observed_initialization != initialization or runtime.model_module.tensor_state_dict_sha256(initial_state) != contract.UPDATE0_STATE_SHA256:
        raise PermissionError("reconstructed update zero differs from the full V4 initialization receipt")
    indices = list(schedule["presentation_indices"])
    pair_ids = [str(item["content_sha256"]) for item in train_pairs]
    recomputed_schedule = stack.contract.with_content_sha256({**stack.contract.schedule_core(indices, pair_ids), "presentation_indices": indices})
    if recomputed_schedule != schedule or len(indices) != contract._v1.PRESENTATION_COUNT or tuple(indices[:16]) != contract.FIRST_PRESENTATION_INDICES:
        raise PermissionError("the bound schedule does not recompute from ordered train pair identities")
    if [indices[start : start + 4] for start in range(0, 16, 4)] != [list(contract.FIRST_PRESENTATION_INDICES[start : start + 4]) for start in range(0, 16, 4)]:
        raise PermissionError("the four exact B4 chunks changed")
    return inputs, trainer, device, initial_state, train_pairs, selection_pairs, indices, vocabulary, commanded_cpu.to(device), {"camera_gate_content_sha256": gate["content_sha256"], "camera_checkpoint": camera_binding, "resource": resource}


def _metric_evaluation(runtime: Any, trainer: Any, initial_state: Mapping[str, Any], selection_pairs: Sequence[Mapping[str, Any]], vocabulary: Sequence[str], commanded: Any, device: Any) -> dict[str, Any]:
    model = _new_model(runtime, initial_state, device, train=False)
    before = _state_sha(runtime, model)
    physical, camera_loss = trainer.physical_metrics(model, selection_pairs, device, arm="update0_diagnostic", stage="update0_checkpoint_selection")
    jepa = trainer.jepa_metrics(model, selection_pairs, vocabulary, commanded, device, arm="update0_diagnostic", stage="update0_checkpoint_selection")
    scopes = {scope: {"physical": physical[scope], "jepa": jepa[scope]} for scope in contract._v1.SCOPES}
    evaluated = contract.evaluate_scopes(scopes)
    after = _state_sha(runtime, model)
    model.to("cpu")
    del model
    runtime.torch.cuda.empty_cache()
    if before != after or after != contract.UPDATE0_STATE_SHA256:
        raise RuntimeError("metric evaluation mutated update-zero state")
    return {"update": 0, "role": "checkpoint_selection", "pair_count": 495, "unique_endpoint_count": 924, "scopes": scopes, "aggregate_complete_v4_loss": camera_loss, "aggregate_prediction_to_persistence_ratio": jepa["aggregate"]["prediction_to_warped_persistence_ratio"], "evaluation": evaluated, "state_sha256_before": before, "state_sha256_after": after, "state_mutation_count": 0}


def _scalar(value: Any) -> float:
    observed = float(value.detach().cpu())
    if not math.isfinite(observed):
        raise FloatingPointError("diagnostic component became nonfinite")
    return observed


def _loss_values(joint: Any, model: Any) -> dict[str, float]:
    jepa, camera = joint.established_jepa, joint.observable_camera_ray_v4
    values: dict[str, Any] = {
        "joint_total": joint.total,
        "camera_total": camera.total,
        "jepa_total": jepa.total,
        "jepa_prediction": jepa.prediction,
        "jepa_equivariance": jepa.equivariance,
        "jepa_action_contrast": jepa.action_contrast,
        "jepa_variance": jepa.variance,
        "jepa_warped_persistence": jepa.warped_persistence,
        "jepa_prediction_to_persistence_ratio": jepa.prediction_to_persistence_ratio,
        "jepa_target_cross_sample_std_mean": jepa.target_cross_sample_std_mean,
        "jepa_target_cross_sample_effective_rank": jepa.target_cross_sample_effective_rank,
        "weighted_jepa_prediction": model.jepa_weight * jepa.prediction,
        "weighted_jepa_equivariance": model.equivariance_weight * jepa.equivariance,
        "weighted_jepa_action_contrast": model.action_contrast_weight * jepa.action_contrast,
        "weighted_jepa_variance": model.variance_weight * jepa.variance,
        "weighted_jepa_component_sum": model.jepa_weight * jepa.prediction + model.equivariance_weight * jepa.equivariance + model.action_contrast_weight * jepa.action_contrast + model.variance_weight * jepa.variance,
    }
    for side in ("current", "next"):
        frame = getattr(camera, side)
        values.update({
            f"{side}_hierarchical_first_hit_nll": frame.hierarchical_first_hit_nll,
            f"{side}_target_bin_offset_smooth_l1": frame.target_bin_offset_smooth_l1,
            f"{side}_ground_clear_distance_state_balanced_bce": frame.ground_clear_distance_state_balanced_bce,
            f"{side}_derived_raster_hierarchical_bce": frame.derived_raster_hierarchical_bce.total,
            f"{side}_derived_raster_cell_nll": frame.derived_raster_cell_nll,
        })
    return {name: _scalar(value) for name, value in values.items()}


def _component(name: str) -> str:
    matches = [component for component, prefix in contract.GRADIENT_COMPONENT_PREFIXES.items() if name.startswith(prefix)]
    if len(matches) != 1:
        raise PermissionError(f"trainable parameter escaped exhaustive diagnostic groups: {name}")
    return matches[0]


def _gradient_summary(runtime: Any, model: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    groups = {name: {"parameter_count": 0, "parameter_tensor_count": 0, "gradient_tensor_count": 0, "nonzero_gradient_tensor_count": 0, "squared_norm": 0.0} for name in contract.GRADIENT_COMPONENT_PREFIXES}
    gradients: dict[str, Any] = {}
    frozen_targets = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            if name.startswith(("target_encoder.", "target_bev_decoder.")):
                if parameter.grad is not None:
                    raise RuntimeError("an EMA target parameter acquired a gradient")
                frozen_targets += int(parameter.numel())
            continue
        component = _component(name)
        row = groups[component]
        row["parameter_count"] += int(parameter.numel())
        row["parameter_tensor_count"] += 1
        if parameter.grad is None:
            gradients[name] = runtime.torch.zeros_like(parameter, device="cpu", dtype=runtime.torch.float32)
            continue
        gradient = parameter.grad.detach().float()
        if not bool(runtime.torch.isfinite(gradient).all().item()):
            raise FloatingPointError("diagnostic gradient became nonfinite")
        cpu = gradient.cpu().contiguous().clone()
        gradients[name] = cpu
        squared = float(cpu.square().sum())
        row["gradient_tensor_count"] += 1
        row["nonzero_gradient_tensor_count"] += int(squared > 0.0)
        row["squared_norm"] += squared
    total_squared = sum(row["squared_norm"] for row in groups.values())
    public = {}
    for name, row in groups.items():
        public[name] = {key: value for key, value in row.items() if key != "squared_norm"} | {"gradient_norm": math.sqrt(row["squared_norm"])}
    if public["occupancy_head_expected_zero"]["gradient_norm"] != 0.0:
        raise RuntimeError("zero-weight occupancy head acquired a nonzero gradient")
    norm = math.sqrt(total_squared)
    factor = min(1.0, 1.0 / (norm + 1e-6))
    if frozen_targets <= 0:
        raise RuntimeError("the expected frozen EMA targets are absent")
    summary = {"components": public, "global_gradient_norm": norm, "counterfactual_clip_max_norm": 1.0, "counterfactual_clip_factor": factor, "counterfactual_norm_after_clip": norm * factor, "clip_grad_norm_invocation_count": 0, "all_gradients_finite": True, "frozen_target_parameter_count": frozen_targets, "all_target_parameters_frozen_and_gradient_free": True}
    summary["statistics_sha256"] = contract.canonical_json_sha256(summary)
    return summary, gradients


def _probe_branch(runtime: Any, trainer: Any, initial_state: Mapping[str, Any], train_pairs: Sequence[Mapping[str, Any]], indices: Sequence[int], vocabulary: Sequence[str], commanded: Any, device: Any, branch: str) -> tuple[dict[str, Any], dict[str, Any]]:
    model = _new_model(runtime, initial_state, device, train=True)
    before = _state_sha(runtime, model)
    model.zero_grad(set_to_none=True)
    sums: dict[str, float] = {}
    wrong_active = zero_active = variance_active = variance_cells = 0
    for start in range(0, 16, 4):
        batch = trainer.batch(train_pairs, indices[start : start + 4], vocabulary, commanded, device, role="train", arm=f"update0_{branch}_diagnostic", stage="update0_gradient")
        pair = model.forward_training_pair(**batch["forward"])
        joint = runtime.loss_adapter.combine_joint_losses_v4(model, pair, batch["current_supervision"], batch["next_supervision"])
        backward = joint.observable_camera_ray_v4.total if branch == "camera" else joint.established_jepa.total
        if not bool(runtime.torch.isfinite(backward).item()):
            raise FloatingPointError("diagnostic backward scalar became nonfinite")
        (backward / 4.0).backward()
        for name, value in _loss_values(joint, model).items():
            sums[name] = sums.get(name, 0.0) + value / 4.0
        counterfactuals = joint.established_jepa.counterfactuals
        wrong_active += int(_scalar(counterfactuals.wrong_action_contrast_loss) > 0.0)
        zero_active += int(_scalar(counterfactuals.zero_action_contrast_loss) > 0.0)
        features = runtime.torch.cat((pair.current.bev, pair.next.bev), dim=0).float()
        std = runtime.torch.sqrt(features.var(dim=0, unbiased=False) + 1e-4)
        variance_active += int((std < float(model.variance_target_std)).sum().detach().cpu())
        variance_cells += int(std.numel())
    gradients, tensors = _gradient_summary(runtime, model)
    after = _state_sha(runtime, model)
    model.zero_grad(set_to_none=True)
    model.to("cpu")
    del model
    runtime.torch.cuda.empty_cache()
    if before != after or after != contract.UPDATE0_STATE_SHA256:
        raise RuntimeError(f"{branch} gradient probe mutated update-zero state")
    return {
        "branch": branch,
        "backward_scalar": "observable_camera_ray_v4.total" if branch == "camera" else "established_jepa.total",
        "microbatch_count": 4,
        "presentation_count": 16,
        "component_loss_means": sums,
        "hinge_rates": {"wrong_action_microbatch_scalar_positive_fraction": wrong_active / 4.0, "zero_action_microbatch_scalar_positive_fraction": zero_active / 4.0, "variance_channel_cell_active_fraction": variance_active / variance_cells},
        "gradients": gradients,
        "state_sha256_before": before,
        "state_sha256_after": after,
        "state_mutation_count": 0,
        "optimizer_construction_count": 0,
        "optimizer_step_count": 0,
        "ema_update_count": 0,
    }, tensors


def _capture_branch(operation: Callable[[], tuple[dict[str, Any], dict[str, Any]]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    collector = contract.CompactDeterminismWarnings()
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        original = warnings.showwarning
        warnings.showwarning = collector
        try:
            public, tensors = operation()
        finally:
            warnings.showwarning = original
    receipt = collector.receipt()
    if receipt["warning_count"] <= 0:
        raise RuntimeError("gradient branch emitted no exact expected grid-sampler warning")
    return public, tensors, receipt


def _interaction(camera: Mapping[str, Any], jepa: Mapping[str, Any]) -> dict[str, Any]:
    if set(camera) != set(jepa):
        raise RuntimeError("camera and JEPA trainable parameter sets differ")
    rows: dict[str, dict[str, float | None]] = {}
    for component in (*contract.GRADIENT_COMPONENT_PREFIXES, "global"):
        names = list(camera) if component == "global" else [name for name in camera if _component(name) == component]
        c2 = sum(float(camera[name].square().sum()) for name in names)
        j2 = sum(float(jepa[name].square().sum()) for name in names)
        dot = sum(float((camera[name] * jepa[name]).sum()) for name in names)
        c_norm, j_norm = math.sqrt(c2), math.sqrt(j2)
        sum_norm = math.sqrt(sum(float((camera[name] + jepa[name]).square().sum()) for name in names))
        cosine = dot / (c_norm * j_norm) if c_norm > 0.0 and j_norm > 0.0 else None
        factor = min(1.0, 1.0 / (sum_norm + 1e-6))
        rows[component] = {"camera_norm": c_norm, "jepa_norm": j_norm, "dot": dot, "cosine": cosine, "camera_plus_jepa_norm": sum_norm, "norm_identity_residual": sum_norm * sum_norm - (c2 + j2 + 2.0 * dot), "camera_plus_jepa_counterfactual_clip_factor": factor, "camera_plus_jepa_counterfactual_norm_after_clip": sum_norm * factor}
    result: dict[str, Any] = {"components_and_global": rows, "clip_grad_norm_invocation_count": 0}
    result["statistics_sha256"] = contract.canonical_json_sha256(result)
    return result


def _gradient_probe(runtime: Any, trainer: Any, initial_state: Mapping[str, Any], train_pairs: Sequence[Mapping[str, Any]], indices: Sequence[int], vocabulary: Sequence[str], commanded: Any, device: Any) -> dict[str, Any]:
    camera, camera_tensors, camera_warning = _capture_branch(lambda: _probe_branch(runtime, trainer, initial_state, train_pairs, indices, vocabulary, commanded, device, "camera"))
    jepa, jepa_tensors, jepa_warning = _capture_branch(lambda: _probe_branch(runtime, trainer, initial_state, train_pairs, indices, vocabulary, commanded, device, "jepa"))
    interaction = _interaction(camera_tensors, jepa_tensors)
    del camera_tensors, jepa_tensors
    return {"presentation_indices": list(indices[:16]), "microbatch_indices": [list(indices[start : start + 4]) for start in range(0, 16, 4)], "camera": camera, "jepa": jepa, "camera_jepa_interaction": interaction, "warning_receipts": {"camera": camera_warning, "jepa": jepa_warning}, "determinism": {**contract._v4.RUNTIME_DETERMINISM, "warn_only_gradients_bitwise_repeatable": False}, "zero_optimizer_ema_or_state_mutation": True}


def _access_receipt(inputs: Any, authorization: Mapping[str, Any], v4_rehash: Mapping[str, Any], sources: Mapping[str, str], reservation: Mapping[str, Any]) -> dict[str, Any]:
    consumed = inputs.rehash_consumed()
    allowed = {"authority", "index", "train", "checkpoint_selection"}
    observed_roles = {role for item in consumed["records"] for role in item["roles"]}
    if any(not set(item["roles"]).issubset(allowed) for item in consumed["records"]) or not {"train", "checkpoint_selection"}.issubset(observed_roles):
        raise PermissionError("diagnostic consumed an unauthorized dataset role")
    camera_records = []
    for kind in ("gate", "checkpoint"):
        binding = authorization["camera"][kind]
        raw = _read_regular(ROOT / binding["path"], expected_sha256=binding["file_sha256"])
        if len(raw) != binding["byte_count"]:
            raise PermissionError("a consumed Camera artifact byte count changed")
        camera_records.append({"kind": kind, **binding})
    if contract.current_source_bindings(ROOT) != dict(sources):
        raise PermissionError("a reviewed source changed during the diagnostic")
    authority_records = []
    for kind in ("independent_review", "execution_authorization"):
        binding = reservation[kind]
        raw = _read_regular(ROOT / binding["path"], expected_sha256=binding["file_sha256"])
        if len(raw) != binding["byte_count"]:
            raise PermissionError("a review or authorization byte count changed")
        authority_records.append({"kind": kind, **binding})
    return {
        "schema": contract.ACCESS_SCHEMA,
        "status": "all_consumed_development_inputs_rehashed",
        "roles_opened": ["train", "checkpoint_selection"],
        "probability_calibration_open_count": 0,
        "camera": {"all_consumed_files_rehashed": True, "records": camera_records},
        "reviewed_sources": {"count": len(sources), "bindings": dict(sources), "all_rehashed": True},
        "review_and_authorization": {"records": authority_records, "all_rehashed": True},
        "predecessor": {"audit": dict(contract.V4_TERMINAL_AUDIT_BINDING), "initialization": dict(contract.V4_INITIALIZATION_BINDING), "schedule": dict(contract.V4_SCHEDULE_BINDING), "terminal_inventory_rehash": dict(v4_rehash)},
        "consumed": consumed,
        "g2_open_count": 0,
        "navigation_open_count": 0,
        "heldout_open_count": 0,
        "all_consumed_files_rehashed": True,
        "authority": dict(contract.DOWNSTREAM_DENIALS),
    }


def _terminal_failure(output_root: Path, reservation: Mapping[str, Any], reservation_raw: bytes, stage: str, error: BaseException, published: Mapping[str, tuple[Mapping[str, Any], bytes]]) -> None:
    artifacts = {"reservation": _binding("reservation.json", reservation, reservation_raw)}
    artifacts.update({name.removesuffix(".json"): _binding(name, value, raw) for name, (value, raw) in published.items()})
    core = {"schema": contract.FAILURE_SCHEMA, "status": "failed_diagnostic", "stage": stage, "attempt_identity": reservation["attempt_identity"], "artifacts": artifacts, "published_prefix": ["reservation.json", *published], "error": {"type": type(error).__name__, "message": str(error)}, "g2_attempted": False, "navigation_attempted": False, "heldout_open_count": 0, "retry_authorized": False, "authority": dict(contract.DOWNSTREAM_DENIALS)}
    try:
        _publish_json(output_root / "failed.json", core)
    except FileExistsError:
        pass


def run_parent(*, review_file_sha256: str, authorization_file_sha256: str) -> int:
    environment = _environment()
    review, review_raw, authorization, authorization_raw, sources = _load_review_authorization(review_file_sha256, authorization_file_sha256)
    output_root = ROOT / contract.OUTPUT_ROOT_RELATIVE_PATH
    reservation, reservation_raw = _reserve(output_root, review, review_raw, authorization, authorization_raw, sources, environment)
    stage, published = "post_reservation_exact_stack_load", {}
    try:
        stack = _load_v4_stack()
        stage = "v4_terminal_inputs"
        audit, initialization, schedule = _predecessor_inputs()
        stage = "post_reservation_torch_import"
        runtime = stack._load_runtime()
        stage = "update0_reconstruction"
        inputs, trainer, device, initial_state, train_pairs, selection_pairs, indices, vocabulary, commanded, provenance = _reconstruct(stack, runtime, authorization, output_root, reservation, initialization, schedule)
        stage = "checkpoint_selection_metrics"
        metrics = _metric_evaluation(runtime, trainer, initial_state, selection_pairs, vocabulary, commanded, device)
        stage = "first16_transfer_gradients"
        gradients = _gradient_probe(runtime, trainer, initial_state, train_pairs, indices, vocabulary, commanded, device)
        stage = "access_rehash"
        v4_rehash = _rehash_v4_terminal(audit)
        access, access_raw = _publish_json(output_root / "access.json", {**_access_receipt(inputs, authorization, v4_rehash, sources, reservation), "reservation": _binding("reservation.json", reservation, reservation_raw)})
        published["access.json"] = (access, access_raw)
        stage = "result_publication"
        result, result_raw = _publish_json(output_root / "result.json", {
            "schema": contract.RESULT_SCHEMA,
            "status": "completed_diagnostic_evidence_only",
            "reservation": _binding("reservation.json", reservation, reservation_raw),
            "access": _binding("access.json", access, access_raw),
            "predecessor_audit_content_sha256": audit["content_sha256"],
            "update0_state_sha256": contract.UPDATE0_STATE_SHA256,
            "checkpoint_selection": metrics,
            "gradient_probe": gradients,
            "provenance": provenance,
            "operation_counts": {"optimizer_construction_count": 0, "optimizer_step_count": 0, "ema_update_count": 0, "state_mutation_count": 0, "metric_scope_count": 9, "gradient_microbatch_count_per_branch": 4},
            "interpretation_authority": "diagnostic_measurement_only_no_selection_training_retry_or_downstream_gate_authority",
            "authority": dict(contract.DOWNSTREAM_DENIALS),
        })
        published["result.json"] = (result, result_raw)
        stage = "completion_publication"
        observed = sorted(item.name for item in output_root.iterdir())
        if observed != ["access.json", "reservation.json", "result.json"] or any(item.is_symlink() or not item.is_file() for item in output_root.iterdir()):
            raise PermissionError("pre-completion output inventory changed")
        _publish_json(output_root / "completed.json", {
            "schema": contract.COMPLETION_SCHEMA,
            "status": "complete_diagnostic_only_no_downstream_authority",
            "attempt_identity": reservation["attempt_identity"],
            "artifacts": {"reservation": _binding("reservation.json", reservation, reservation_raw), "access": _binding("access.json", access, access_raw), "result": _binding("result.json", result, result_raw)},
            "exact_terminal_inventory": list(contract.science_contract()["success_inventory"]),
            "all_inputs_rehashed": True,
            "optimizer_step_count": 0,
            "ema_update_count": 0,
            "state_mutation_count": 0,
            "heldout_open_count": 0,
            "retry_authorized": False,
            "authority": dict(contract.DOWNSTREAM_DENIALS),
        })
        return 0
    except BaseException as error:
        _terminal_failure(output_root, reservation, reservation_raw, stage, error, published)
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
