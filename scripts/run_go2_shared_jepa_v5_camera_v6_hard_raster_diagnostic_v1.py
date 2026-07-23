#!/usr/bin/env python3
"""Run the one authorized Camera V6 hard-raster diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.benchmarks import (
    go2_shared_jepa_v5_camera_v6_hard_raster_diagnostic_v1 as contract,
)


IMPLEMENTATION_AUTHOR = "/root"
EXPECTED_THREAD_ENVIRONMENT = {
    "BLIS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
EXPECTED_SELECTOR_ENVIRONMENT = {
    "HIP_VISIBLE_DEVICES": "0",
    "ROCR_VISIBLE_DEVICES": None,
    "CUDA_VISIBLE_DEVICES": None,
    "HSA_OVERRIDE_GFX_VERSION": None,
}
DIRECT_SOURCE_RELATIVE_PATHS = (
    "lewm/benchmarks/go2_shared_jepa_v5_camera_v6_hard_raster_diagnostic_v1.py",
    "scripts/run_go2_shared_jepa_v5_camera_v6_hard_raster_diagnostic_v1.py",
    "lewm/tests/test_go2_shared_jepa_v5_camera_v6_hard_raster_diagnostic_v1.py",
    "lewm/benchmarks/go2_shared_jepa_v5_protected_camera_adaptation_v6.py",
    "scripts/run_go2_shared_jepa_v5_matched_training_v4.py",
    "lewm/benchmarks/go2_shared_jepa_v5_matched_training_v4.py",
    "lewm/benchmarks/go2_observable_camera_ray_evidence_v4.py",
    "lewm/benchmarks/go2_observable_camera_ray_fit_v4_metrics.py",
    "lewm/models/observable_camera_ray_evidence_v4.py",
    "lewm/models/observable_camera_ray_evidence_v4_training.py",
    "lewm/models/shared_observable_camera_ray_jepa_v5.py",
    "lewm/models/shared_observable_camera_ray_jepa_v5_full_training_v4_loss.py",
    "lewm/models/encoders.py",
    "lewm/models/egomotion_bev_jepa.py",
    "lewm/models/observable_camera_ray_evidence_v4_gate_aligned_raster_nll_v12.py",
    "lewm/models/observable_camera_ray_evidence_v4_hierarchical_first_hit_v9.py",
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_regular(path: Path, *, expected_sha256: str | None = None) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise PermissionError(f"required regular file changed: {path}")
    raw = path.read_bytes()
    if expected_sha256 is not None and _sha256(raw) != expected_sha256:
        raise PermissionError(f"required file hash changed: {path}")
    return raw


def _read_bound_experiment(
    binding: Mapping[str, Any],
    *,
    kind: str,
    ledger: list[dict[str, Any]],
) -> bytes:
    event = {
        "sequence": len(ledger) + 1,
        "kind": kind,
        "path": str(binding["path"]),
        "expected_byte_count": int(binding["byte_count"]),
        "expected_file_sha256": str(binding["file_sha256"]),
        "status": "read_started",
    }
    ledger.append(event)
    try:
        raw = _read_regular(ROOT / str(binding["path"]))
    except BaseException as error:
        event["status"] = "read_failed"
        event["error_type"] = type(error).__name__
        raise
    event.update(
        {
            "observed_byte_count": len(raw),
            "observed_file_sha256": _sha256(raw),
            "status": "read_completed",
        }
    )
    if (
        len(raw) != binding["byte_count"]
        or event["observed_file_sha256"] != binding["file_sha256"]
    ):
        event["status"] = "binding_mismatch"
        raise PermissionError(f"{kind} binding changed")
    return raw


def _parse_json(raw: bytes, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PermissionError(f"{name} is not valid JSON") from error
    if type(value) is not dict:
        raise PermissionError(f"{name} must be a JSON object")
    return value


def _binding(path: str, raw: bytes, *, content_sha256: str | None = None) -> dict[str, Any]:
    result = {
        "path": path,
        "byte_count": len(raw),
        "file_sha256": _sha256(raw),
    }
    if content_sha256 is not None:
        result["content_sha256"] = content_sha256
    return result


def _publish_json(path: Path, core: Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
    value = contract.with_content_sha256(core)
    raw = contract.canonical_json_bytes(value)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
        os.unlink(temporary)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return value, raw


def _load_exact_module(path: Path, name: str, expected_sha256: str) -> Any:
    raw = _read_regular(path, expected_sha256=expected_sha256)
    if not raw:
        raise ImportError(f"module is empty: {path}")
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _validate_environment() -> dict[str, Any]:
    observed_threads = {name: os.environ.get(name) for name in EXPECTED_THREAD_ENVIRONMENT}
    observed_selectors = {name: os.environ.get(name) for name in EXPECTED_SELECTOR_ENVIRONMENT}
    if observed_threads != EXPECTED_THREAD_ENVIRONMENT:
        raise PermissionError("native thread environment changed")
    if observed_selectors != EXPECTED_SELECTOR_ENVIRONMENT:
        raise PermissionError("GPU selector environment changed")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise PermissionError("diagnostic requires Python -I -B")
    return {
        "native_thread_environment": observed_threads,
        "selector_environment": {
            name: "absent" if value is None else value
            for name, value in observed_selectors.items()
        },
        "precision": "float32",
        "autocast": False,
        "cpu_fallback": False,
        "python_flags": ["-I", "-B"],
    }


def _preregistration_binding() -> dict[str, Any]:
    path = ROOT / contract.PREREGISTRATION_RELATIVE_PATH
    raw = _read_regular(path)
    return _binding(contract.PREREGISTRATION_RELATIVE_PATH, raw)


def _validate_review(
    value: object,
    *,
    preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    review = contract.validate_content_sha256(value, schema=contract.REVIEW_SCHEMA)
    fields = {
        "schema",
        "status",
        "implementation_author",
        "reviewer",
        "preregistration",
        "reviewed_sources",
        "source_only",
        "experiment_open_counts",
        "findings",
        "authority",
        "content_sha256",
    }
    reviewer = review.get("reviewer")
    sources = review.get("reviewed_sources")
    if (
        set(review) != fields
        or review["status"] != "PASS"
        or review["implementation_author"] != IMPLEMENTATION_AUTHOR
        or type(reviewer) is not str
        or not reviewer.startswith("/root/")
        or reviewer == IMPLEMENTATION_AUTHOR
        or review["preregistration"] != dict(preregistration)
        or type(sources) is not dict
        or set(sources) != set(DIRECT_SOURCE_RELATIVE_PATHS)
        or review["source_only"] is not True
        or review["experiment_open_counts"]
        != {
            "checkpoint": 0,
            "dataset": 0,
            "rgb": 0,
            "gpu": 0,
            "navigation": 0,
            "heldout": 0,
        }
        or review["findings"] != []
        or review["authority"]
        != {
            "implementation_mutation": False,
            "experiment_execution": False,
            "checkpoint_or_data_access": False,
            "training": False,
            "promotion": False,
            "g2_navigation_or_heldout": False,
        }
    ):
        raise PermissionError("independent review contract changed")
    for relative, binding in sources.items():
        raw = _read_regular(ROOT / relative, expected_sha256=binding["file_sha256"])
        if binding != _binding(relative, raw):
            raise PermissionError(f"reviewed source binding changed: {relative}")
    return review


def _expected_raw_authority() -> dict[str, Any]:
    return {
        "root": (
            ".generated/go2_shared_observable_camera_ray_jepa_v5/"
            "development_raw_supervision_v1"
        ),
        "manifest": dict(contract.RAW_MANIFEST_BINDING),
        "audit": dict(contract.RAW_AUDIT_BINDING),
        "role_counts": {
            "checkpoint_selection": {
                "pairs": contract.SELECTION_PAIR_COUNT,
                "unique_endpoints": contract.SELECTION_ENDPOINT_COUNT,
                "scenes": 8,
            }
        },
        "grant": {
            "allowed_roles": ["checkpoint_selection"],
            "allowed_operations": [
                "development_rgb_decode",
                "fixed_calibration_read",
                "diagnostic_forward_inference",
                "soft_metric_reproduction",
                "hard_raster_diagnostic",
            ],
            "train_or_probability_calibration_role": False,
            "selection_decision": False,
            "g2_navigation_runtime_production_or_heldout": False,
        },
    }


def _validate_authorization(
    value: object,
    *,
    preregistration: Mapping[str, Any],
    review: Mapping[str, Any],
    review_binding: Mapping[str, Any],
) -> dict[str, Any]:
    authorization = contract.validate_content_sha256(
        value, schema=contract.AUTHORIZATION_SCHEMA
    )
    fields = {
        "schema",
        "status",
        "authorizer",
        "independent_review",
        "preregistration",
        "reviewed_sources",
        "inputs",
        "raw",
        "environment",
        "output_root",
        "command",
        "authority",
        "content_sha256",
    }
    authorizer = authorization.get("authorizer")
    expected_inputs = {
        "v6_terminal_audit": dict(contract.V6_TERMINAL_AUDIT_BINDING),
        "v6_update8000_sidecar": dict(contract.V6_SIDECAR_BINDING),
        "v6_update8000_checkpoint": dict(contract.V6_CHECKPOINT_BINDING),
    }
    if (
        set(authorization) != fields
        or authorization["status"] != "authorized_one_exact_forward_only_attempt"
        or type(authorizer) is not str
        or not authorizer.startswith("/root/")
        or authorizer in {IMPLEMENTATION_AUTHOR, review["reviewer"]}
        or authorization["independent_review"] != dict(review_binding)
        or authorization["preregistration"] != dict(preregistration)
        or authorization["reviewed_sources"] != review["reviewed_sources"]
        or authorization["inputs"] != expected_inputs
        or authorization["raw"] != _expected_raw_authority()
        or authorization["environment"]
        != {
            "status": "PASS_exactly_one_visible_discrete_R9700",
            "device": "cuda:0",
            "normalized_name_contains": "r9700",
            "visible_device_count": 1,
            "precision": "float32",
            "autocast": False,
            "selector_environment": {
                "HIP_VISIBLE_DEVICES": "0",
                "ROCR_VISIBLE_DEVICES": "absent",
                "CUDA_VISIBLE_DEVICES": "absent",
                "HSA_OVERRIDE_GFX_VERSION": "absent",
            },
            "native_thread_environment": EXPECTED_THREAD_ENVIRONMENT,
            "other_generated_mutator_active": False,
            "other_kfd_training_process_active": False,
            "competing_gpu_work_active": False,
            "output_root_absent": True,
            "checkpoint_dataset_rgb_or_heldout_open_count": 0,
            "tensor_allocation_count": 0,
        }
        or authorization["output_root"] != contract.OUTPUT_ROOT_RELATIVE_PATH
        or authorization["command"]
        != {
            "python_flags": ["-I", "-B"],
            "script": (
                "scripts/run_go2_shared_jepa_v5_camera_v6_"
                "hard_raster_diagnostic_v1.py"
            ),
            "attempt_index": 1,
            "maximum_attempt_count": 1,
        }
        or authorization["authority"] != contract.EXECUTION_AUTHORITY
    ):
        raise PermissionError("execution authorization contract changed")
    return authorization


def _load_authority(
    *,
    review_file_sha256: str,
    authorization_file_sha256: str,
) -> tuple[dict[str, Any], bytes, dict[str, Any], bytes, dict[str, Any]]:
    preregistration = _preregistration_binding()
    review_raw = _read_regular(
        ROOT / contract.REVIEW_RELATIVE_PATH,
        expected_sha256=review_file_sha256,
    )
    review = _validate_review(
        _parse_json(review_raw, name="independent review"),
        preregistration=preregistration,
    )
    review_binding = _binding(
        contract.REVIEW_RELATIVE_PATH,
        review_raw,
        content_sha256=review["content_sha256"],
    )
    authorization_raw = _read_regular(
        ROOT / contract.AUTHORIZATION_RELATIVE_PATH,
        expected_sha256=authorization_file_sha256,
    )
    authorization = _validate_authorization(
        _parse_json(authorization_raw, name="execution authorization"),
        preregistration=preregistration,
        review=review,
        review_binding=review_binding,
    )
    return review, review_raw, authorization, authorization_raw, preregistration


def _reserve(
    *,
    environment: Mapping[str, Any],
    review: Mapping[str, Any],
    review_raw: bytes,
    authorization: Mapping[str, Any],
    authorization_raw: bytes,
    preregistration: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], bytes]:
    output_root = ROOT / contract.OUTPUT_ROOT_RELATIVE_PATH
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("one-attempt diagnostic output root already exists")
    output_root.mkdir(mode=0o700, parents=False)
    review_binding = _binding(
        contract.REVIEW_RELATIVE_PATH,
        review_raw,
        content_sha256=review["content_sha256"],
    )
    authorization_binding = _binding(
        contract.AUTHORIZATION_RELATIVE_PATH,
        authorization_raw,
        content_sha256=authorization["content_sha256"],
    )
    try:
        reservation, raw = _publish_json(
            output_root / "reservation.json",
            {
                "schema": contract.RESERVATION_SCHEMA,
                "status": "reserved_attempt_consumed",
                "attempt_index": 1,
                "maximum_attempt_count": 1,
                "output_root": contract.OUTPUT_ROOT_RELATIVE_PATH,
                "preregistration": dict(preregistration),
                "independent_review": review_binding,
                "execution_authorization": authorization_binding,
                "reviewed_sources": review["reviewed_sources"],
                "environment": dict(environment),
                "pre_reservation_experiment_open_counts": {
                    "checkpoint": 0,
                    "sidecar": 0,
                    "dataset": 0,
                    "rgb": 0,
                    "tensor_runtime_import": 0,
                    "heldout": 0,
                },
                "retry_resume_repair_or_root_reuse_authorized": False,
            },
        )
    except BaseException as error:
        try:
            _publish_json(
                output_root / "reservation_failed.json",
                {
                    "schema": contract.FAILURE_SCHEMA,
                    "status": "failed_reservation_commit",
                    "stage": "reservation_commit",
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                    "attempt_consumed": True,
                    "experiment_input_open_count": 0,
                    "retry_resume_repair_or_root_reuse_authorized": False,
                },
            )
        except BaseException:
            pass
        raise
    return output_root, reservation, raw


def _validate_v6_terminal_audit(
    v6_contract: Any,
    *,
    ledger: list[dict[str, Any]],
) -> tuple[dict[str, Any], bytes]:
    binding = contract.V6_TERMINAL_AUDIT_BINDING
    raw = _read_bound_experiment(
        binding,
        kind="v6_terminal_audit",
        ledger=ledger,
    )
    value = _parse_json(raw, name="V6 terminal audit")
    if (
        value.get("content_sha256") != binding["content_sha256"]
        or value.get("outcome", {}).get("camera_checkpoint_qualified") is not False
        or value.get("outcome", {}).get("final_update") != 8000
        or value.get("outcome", {}).get("selected_checkpoint") is not None
    ):
        raise PermissionError("V6 terminal audit outcome changed")
    return value, raw


def _load_baseline_sidecar(
    v6_contract: Any,
    *,
    ledger: list[dict[str, Any]],
) -> tuple[dict[str, Any], bytes]:
    binding = contract.V6_SIDECAR_BINDING
    raw = _read_bound_experiment(
        binding,
        kind="v6_update8000_sidecar",
        ledger=ledger,
    )
    value = v6_contract.validate_metric_sidecar(
        _parse_json(raw, name="V6 update-8000 sidecar"),
        update=8000,
    )
    if value["content_sha256"] != binding["content_sha256"]:
        raise PermissionError("V6 sidecar content hash changed")
    contract.validate_v6_sidecar_checkpoint_binding(value["checkpoint"])
    return value, raw


def _subset_state_sha256(
    runtime: Any,
    state: Mapping[str, Any],
    prefixes: Sequence[str],
) -> str:
    subset = {
        name: value
        for name, value in state.items()
        if name.startswith(tuple(prefixes))
    }
    if not subset:
        raise PermissionError("checkpoint state subset is empty")
    return runtime.model_module.tensor_state_dict_sha256(subset)


def _load_checkpoint_once(
    runtime: Any,
    v6_contract: Any,
    *,
    device: Any,
    ledger: list[dict[str, Any]],
    operations: dict[str, Any],
) -> tuple[Any, bytes]:
    binding = contract.V6_CHECKPOINT_BINDING
    operations["checkpoint_filesystem_read_attempt_count"] += 1
    raw = _read_bound_experiment(
        binding,
        kind="v6_update8000_checkpoint",
        ledger=ledger,
    )
    operations["checkpoint_filesystem_read_count"] += 1
    operations["checkpoint_deserialization_attempt_count"] += 1
    value = runtime.torch.load(
        io.BytesIO(raw),
        map_location="cpu",
        weights_only=True,
    )
    operations["checkpoint_deserialization_count"] += 1
    expected_fields = {
        "schema",
        "update",
        "model_config",
        "state_sha256",
        "frozen_state_sha256",
        "trainable_state_sha256",
        "initialization_state_sha256",
        "schedule_prefix_indices_sha256",
        "optimizer_contract",
        "development_only",
        "resume_authorized",
        "runtime_ready",
        "content_sha256",
        "model_state_dict",
    }
    if type(value) is not dict or set(value) != expected_fields:
        raise PermissionError("V6 checkpoint payload fields changed")
    state = value["model_state_dict"]
    model = runtime.model_module.SharedObservableCameraRayJepaV5()
    operations["model_construction_count"] += 1
    semantic = {key: value[key] for key in expected_fields - {"content_sha256", "model_state_dict"}}
    state_sha = runtime.model_module.tensor_state_dict_sha256(state)
    frozen_sha = _subset_state_sha256(runtime, state, v6_contract.FROZEN_STATE_PREFIXES)
    trainable_sha = _subset_state_sha256(
        runtime, state, v6_contract.TRAINABLE_PARAMETER_PREFIXES
    )
    if (
        value["schema"] != v6_contract.SNAPSHOT_SCHEMA
        or value["update"] != 8000
        or value["model_config"] != model.model_config.to_dict()
        or value["state_sha256"] != binding["state_sha256"]
        or state_sha != binding["state_sha256"]
        or value["frozen_state_sha256"] != binding["frozen_state_sha256"]
        or frozen_sha != binding["frozen_state_sha256"]
        or value["trainable_state_sha256"] != binding["trainable_state_sha256"]
        or trainable_sha != binding["trainable_state_sha256"]
        or value["initialization_state_sha256"] != v6_contract.UPDATE0_STATE_SHA256
        or value["schedule_prefix_indices_sha256"]
        != v6_contract.CHECKPOINT_SCHEDULE_PREFIX_SHA256[8000]
        or value["optimizer_contract"] != v6_contract.OPTIMIZER_CONTRACT
        or value["development_only"] is not True
        or value["resume_authorized"] is not False
        or value["runtime_ready"] is not False
        or value["content_sha256"] != binding["content_sha256"]
        or v6_contract.canonical_json_sha256(semantic) != binding["content_sha256"]
    ):
        raise PermissionError("V6 checkpoint semantic or state binding changed")
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    model.eval()
    model.to(device)
    return model, raw


def _run_evaluation(
    *,
    runtime: Any,
    stack: Any,
    trainer: Any,
    inputs: Any,
    model: Any,
    device: Any,
    selection_pairs: Sequence[Mapping[str, Any]],
    operations: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = runtime.torch
    families = tuple(stack.contract.FAMILIES)
    if set(stack.contract.SCOPES) != set(contract.ALL_SCOPES):
        raise PermissionError("registered physical scopes changed")
    ids_by_family = {
        family: sorted(
            {
                str(pair[f"{side}_endpoint_sha256"])
                for pair in selection_pairs
                if pair["family"] == family
                for side in ("current", "next")
            }
        )
        for family in families
    }
    if (
        len(selection_pairs) != contract.SELECTION_PAIR_COUNT
        or sum(len(ids) for ids in ids_by_family.values())
        != contract.SELECTION_ENDPOINT_COUNT
        or any(len(ids) < 2 for ids in ids_by_family.values())
    ):
        raise PermissionError("checkpoint-selection endpoint population changed")

    soft_correct = {scope: runtime.MetricAccumulator() for scope in contract.ALL_SCOPES}
    soft_wrong = {scope: runtime.MetricAccumulator() for scope in contract.ALL_SCOPES}
    hard_correct = {scope: contract.HardRasterConfusion() for scope in contract.ALL_SCOPES}
    hard_wrong = {scope: contract.HardRasterConfusion() for scope in contract.ALL_SCOPES}
    model_state_before = runtime.model_module.tensor_state_dict_sha256(model.state_dict())
    with torch.inference_mode():
        for family in families:
            ids = ids_by_family[family]
            wrong_ids = ids[1:] + ids[:1]
            for start in range(0, len(ids), 4):
                target_ids = ids[start : start + 4]
                mapped_ids = wrong_ids[start : start + 4]
                target = [
                    inputs.frame(
                        item,
                        role="checkpoint_selection",
                        arm="hard_raster_diagnostic",
                        stage="matched_and_wrong_forward",
                    )
                    for item in target_ids
                ]
                mapped = [
                    inputs.frame(
                        item,
                        role="checkpoint_selection",
                        arm="hard_raster_diagnostic",
                        stage="matched_and_wrong_forward",
                    )
                    for item in mapped_ids
                ]
                origin = torch.stack([item["camera_origin"] for item in target]).to(
                    device
                ).float()
                basis = torch.stack([item["camera_basis"] for item in target]).to(
                    device
                ).float()
                ground = torch.stack([item["ground"] for item in target]).to(
                    device
                ).float()
                supervision = trainer.supervision(target, device)
                targets = runtime.derive_targets(
                    pixel_hit_mask=supervision.pixel_hit_mask,
                    pixel_first_hit_distance_m=supervision.pixel_first_hit_distance_m,
                    ground_support_in_frustum=supervision.ground_support_in_frustum,
                    ground_support_clear_to_target=(
                        supervision.ground_support_clear_to_target
                    ),
                )

                for arm_index, frames in enumerate((target, mapped)):
                    online = model.forward_frame(
                        torch.stack([item["image"] for item in frames]).to(device),
                        origin,
                        basis,
                        ground,
                    )
                    soft = runtime.soft_rasterize(
                        online.evidence,
                        camera_origin_body_m=origin,
                        camera_basis_body_fru=basis,
                        pixel_ray_chunk_size=model.model_config.v4_pixel_ray_chunk_size,
                    )
                    hard = contract.hard_raster_labels_from_raw_output(
                        online.evidence,
                        camera_origin_body_m=origin,
                        camera_basis_body_fru=basis,
                        ground_plane_z_body_m=ground,
                    )
                    soft_set = soft_correct if arm_index == 0 else soft_wrong
                    hard_set = hard_correct if arm_index == 0 else hard_wrong
                    for scope in ("aggregate", family):
                        soft_set[scope].update(
                            raw_output=online.evidence,
                            targets=targets,
                            soft_raster=soft,
                            target_raster_labels=supervision.target_raster_labels,
                            families=[family] * len(target),
                        )
                        hard_set[scope].update(
                            [item.output_labels for item in hard],
                            supervision.target_raster_labels,
                        )
                    if arm_index == 0:
                        operations["matched_forward_frame_presentation_count"] += len(
                            target
                        )
                    else:
                        operations[
                            "wrong_rgb_forward_frame_presentation_count"
                        ] += len(target)

    model_state_after = runtime.model_module.tensor_state_dict_sha256(model.state_dict())
    if model_state_before != model_state_after:
        raise RuntimeError("forward-only diagnostic mutated model state")
    soft_scopes = {
        scope: trainer._flatten_physical(
            soft_correct[scope].finalize(),
            soft_wrong[scope].finalize(),
        )
        for scope in contract.ALL_SCOPES
    }
    hard_scopes = {}
    for scope in contract.ALL_SCOPES:
        matched = hard_correct[scope].finalize()
        wrong = hard_wrong[scope].finalize()
        hard_scopes[scope] = {
            "matched": matched,
            "wrong": wrong,
            "matched_minus_wrong_balanced_accuracy": (
                matched["balanced_accuracy"] - wrong["balanced_accuracy"]
            ),
            "hard_nll_excluded": True,
        }
    operations.update(
        {
            "selection_pair_count": len(selection_pairs),
            "unique_endpoint_count": sum(
                len(ids) for ids in ids_by_family.values()
            ),
            "model_state_mutation_count": 0,
            "state_sha256_before": model_state_before,
            "state_sha256_after": model_state_after,
        }
    )
    if (
        operations["matched_forward_frame_presentation_count"]
        != contract.SELECTION_ENDPOINT_COUNT
        or operations["wrong_rgb_forward_frame_presentation_count"]
        != contract.SELECTION_ENDPOINT_COUNT
    ):
        raise PermissionError("forward presentation count changed")
    return soft_scopes, hard_scopes


def _initial_operation_counts() -> dict[str, Any]:
    return {
        "model_construction_count": 0,
        "checkpoint_filesystem_read_attempt_count": 0,
        "checkpoint_filesystem_read_count": 0,
        "checkpoint_deserialization_attempt_count": 0,
        "checkpoint_deserialization_count": 0,
        "sidecar_read_attempt_count": 0,
        "sidecar_read_count": 0,
        "selection_pair_count": 0,
        "unique_endpoint_count": 0,
        "matched_forward_frame_presentation_count": 0,
        "wrong_rgb_forward_frame_presentation_count": 0,
        "optimizer_construction_count": 0,
        "optimizer_step_count": 0,
        "backward_count": 0,
        "gradient_count": 0,
        "clip_count": 0,
        "ema_update_count": 0,
        "autocast_count": 0,
        "checkpoint_write_or_mutation_count": 0,
        "model_state_mutation_count": 0,
        "train_role_payload_open_count": 0,
        "probability_calibration_role_payload_open_count": 0,
        "g2_navigation_runtime_or_heldout_count": 0,
        "state_sha256_before": None,
        "state_sha256_after": None,
    }


def _current_access_core(
    *,
    status: str,
    top_level_reads: Sequence[Mapping[str, Any]],
    inputs: Any | None,
    operations: Mapping[str, Any],
) -> dict[str, Any]:
    consumed = getattr(inputs, "consumed", {})
    records = (
        [consumed[name] for name in sorted(consumed)]
        if type(consumed) is dict
        else []
    )
    return {
        "schema": contract.ACCESS_SCHEMA,
        "status": status,
        "top_level_reads": [dict(item) for item in top_level_reads],
        "records": records,
        "unique_input_file_count": len(records),
        "operation_counts": dict(operations),
        "forbidden_role_open_counts": {
            "train": sum(
                "train" in record.get("roles", ()) for record in records
            ),
            "probability_calibration": sum(
                "probability_calibration" in record.get("roles", ())
                for record in records
            ),
            "g2": 0,
            "navigation": 0,
            "runtime_or_production": 0,
            "heldout": 0,
        },
    }


def _publish_failure(
    output_root: Path,
    *,
    stage: str,
    error: BaseException,
    access_core: Mapping[str, Any] | None,
) -> None:
    if access_core is not None and not (output_root / "access.json").exists():
        try:
            _publish_json(output_root / "access.json", access_core)
        except BaseException:
            pass
    try:
        _publish_json(
            output_root / "failed.json",
            {
                "schema": contract.FAILURE_SCHEMA,
                "status": "FAIL_INTEGRITY",
                "stage": stage,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
                "attempt_consumed": True,
                "retry_resume_repair_or_root_reuse_authorized": False,
                "training_checkpoint_promotion_g2_navigation_or_heldout_authorized": False,
            },
        )
    except FileExistsError:
        pass


def _validate_terminal_inventory(
    output_root: Path,
    *,
    expected_names: set[str],
) -> None:
    entries = list(output_root.iterdir())
    if (
        {entry.name for entry in entries} != expected_names
        or any(entry.is_symlink() or not entry.is_file() for entry in entries)
        or any((entry.stat().st_mode & 0o777) != 0o444 for entry in entries)
    ):
        raise PermissionError("diagnostic terminal inventory changed")


def run(
    *,
    review_file_sha256: str,
    authorization_file_sha256: str,
) -> int:
    environment = _validate_environment()
    review, review_raw, authorization, authorization_raw, preregistration = (
        _load_authority(
            review_file_sha256=review_file_sha256,
            authorization_file_sha256=authorization_file_sha256,
        )
    )
    output_root, reservation, reservation_raw = _reserve(
        environment=environment,
        review=review,
        review_raw=review_raw,
        authorization=authorization,
        authorization_raw=authorization_raw,
        preregistration=preregistration,
    )
    stage = "post_reservation_runtime_import"
    top_level_reads: list[dict[str, Any]] = []
    operations = _initial_operation_counts()
    inputs: Any | None = None
    try:
        v6_contract_path = (
            ROOT
            / "lewm/benchmarks/go2_shared_jepa_v5_protected_camera_adaptation_v6.py"
        )
        v6_contract = _load_exact_module(
            v6_contract_path,
            "_lewm_camera_v6_hard_raster_diagnostic_v1_v6_contract",
            review["reviewed_sources"][
                "lewm/benchmarks/"
                "go2_shared_jepa_v5_protected_camera_adaptation_v6.py"
            ]["file_sha256"],
        )
        stack_loader_path = (
            ROOT / "scripts/run_go2_shared_jepa_v5_matched_training_v4.py"
        )
        stack_loader = _load_exact_module(
            stack_loader_path,
            "_lewm_camera_v6_hard_raster_diagnostic_v1_stack",
            review["reviewed_sources"][
                "scripts/run_go2_shared_jepa_v5_matched_training_v4.py"
            ]["file_sha256"],
        )
        stack = stack_loader.install()
        runtime = stack._load_runtime()
        stage = "device_validation"
        device_trainer = stack.Trainer(
            runtime,
            None,
            output_root,
            reservation,
        )
        device, device_record = device_trainer.device()
        del device_trainer
        stage = "v6_terminal_and_baseline_validation"
        terminal_audit, terminal_raw = _validate_v6_terminal_audit(
            v6_contract,
            ledger=top_level_reads,
        )
        operations["sidecar_read_attempt_count"] += 1
        sidecar, sidecar_raw = _load_baseline_sidecar(
            v6_contract,
            ledger=top_level_reads,
        )
        operations["sidecar_read_count"] += 1
        stage = "raw_checkpoint_selection_input_validation"
        inputs = stack.RawInputs.__new__(stack.RawInputs)
        stack.RawInputs.__init__(inputs, runtime, authorization)
        trainer = stack.Trainer(runtime, inputs, output_root, reservation)
        stage = "checkpoint_single_read_and_deserialization"
        model, checkpoint_raw = _load_checkpoint_once(
            runtime,
            v6_contract,
            device=device,
            ledger=top_level_reads,
            operations=operations,
        )
        stage = "checkpoint_selection_forward_only_evaluation"
        selection_pairs = inputs.role_pairs("checkpoint_selection")
        soft_scopes, hard_scopes = _run_evaluation(
            runtime=runtime,
            stack=stack,
            trainer=trainer,
            inputs=inputs,
            model=model,
            device=device,
            selection_pairs=selection_pairs,
            operations=operations,
        )
        stage = "immutable_soft_reproduction"
        baseline_metric = sidecar["metric"]
        baseline_scopes = baseline_metric["scopes"]
        soft_reproduction_exact = soft_scopes == baseline_scopes
        direct_reproduction_exact = (
            contract.direct_metric_projection(soft_scopes)
            == contract.direct_metric_projection(baseline_scopes)
        )
        if not soft_reproduction_exact or not direct_reproduction_exact:
            raise RuntimeError("immutable V6 soft physical metrics did not reproduce exactly")
        stage = "fixed_materiality_decision"
        materiality = contract.evaluate_materiality(hard_scopes)
        access_core = _current_access_core(
            status="PASS_exact_permitted_reads_only",
            top_level_reads=top_level_reads,
            inputs=inputs,
            operations=operations,
        )
        access_records = access_core["records"]
        if any(
            role in {"train", "probability_calibration"}
            for record in access_records
            for role in record["roles"]
        ):
            raise PermissionError("forbidden development role was consumed")
        if any(access_core["forbidden_role_open_counts"].values()):
            raise PermissionError("forbidden input role was consumed")
        access_core["bindings"] = {
            "v6_terminal_audit": _binding(
                contract.V6_TERMINAL_AUDIT_BINDING["path"],
                terminal_raw,
                content_sha256=terminal_audit["content_sha256"],
            ),
            "v6_update8000_sidecar": _binding(
                contract.V6_SIDECAR_BINDING["path"],
                sidecar_raw,
                content_sha256=sidecar["content_sha256"],
            ),
            "v6_update8000_checkpoint": {
                **_binding(
                    contract.V6_CHECKPOINT_BINDING["path"],
                    checkpoint_raw,
                    content_sha256=contract.V6_CHECKPOINT_BINDING[
                        "content_sha256"
                    ],
                ),
                "state_sha256": contract.V6_CHECKPOINT_BINDING["state_sha256"],
            },
        }
        access, access_raw = _publish_json(output_root / "access.json", access_core)
        stage = "result_publication"
        result, result_raw = _publish_json(
            output_root / "result.json",
            {
                "schema": contract.RESULT_SCHEMA,
                "status": materiality["scientific_verdict"],
                "integrity_verdict": "PASS",
                "scientific_verdict": materiality["scientific_verdict"],
                "reservation": _binding(
                    "reservation.json",
                    reservation_raw,
                    content_sha256=reservation["content_sha256"],
                ),
                "access": _binding(
                    "access.json",
                    access_raw,
                    content_sha256=access["content_sha256"],
                ),
                "device": device_record,
                "population": {
                    "role": "checkpoint_selection",
                    "pair_count": contract.SELECTION_PAIR_COUNT,
                    "unique_endpoint_count": contract.SELECTION_ENDPOINT_COUNT,
                    "scope_count": len(contract.ALL_SCOPES),
                    "cyclic_plus_one_within_family_wrong_rgb": True,
                },
                "soft_reproduction": {
                    "exact_full_physical_scope_projection": soft_reproduction_exact,
                    "exact_direct_metric_projection": direct_reproduction_exact,
                    "zero_tolerance": True,
                    "scopes": soft_scopes,
                },
                "hard_raster": {
                    "scopes": hard_scopes,
                    "nll": None,
                    "nll_excluded_as_non_comparable": True,
                },
                "materiality": materiality,
                "operation_counts": operations,
                "training_or_state_mutation_count": 0,
                "heldout_open_count": 0,
                "camera_checkpoint_qualified": False,
                "checkpoint_promoted": False,
                "downstream_authority": {
                    "successor_implementation_or_training": False,
                    "g2": False,
                    "navigation": False,
                    "runtime_or_production": False,
                    "heldout": False,
                },
            },
        )
        stage = "completion_publication"
        _validate_terminal_inventory(
            output_root,
            expected_names={"reservation.json", "access.json", "result.json"},
        )
        _publish_json(
            output_root / "completed.json",
            {
                "schema": contract.COMPLETION_SCHEMA,
                "status": "completed_one_attempt_consumed",
                "reservation": _binding(
                    "reservation.json",
                    reservation_raw,
                    content_sha256=reservation["content_sha256"],
                ),
                "access": _binding(
                    "access.json",
                    access_raw,
                    content_sha256=access["content_sha256"],
                ),
                "result": _binding(
                    "result.json",
                    result_raw,
                    content_sha256=result["content_sha256"],
                ),
                "exact_terminal_inventory": [
                    "access.json",
                    "completed.json",
                    "reservation.json",
                    "result.json",
                ],
                "terminal_file_mode": "0444",
                "attempt_consumed": True,
                "retry_resume_repair_or_root_reuse_authorized": False,
                "training_checkpoint_promotion_g2_navigation_or_heldout_authorized": False,
            },
        )
        _validate_terminal_inventory(
            output_root,
            expected_names={
                "reservation.json",
                "access.json",
                "result.json",
                "completed.json",
            },
        )
        return 0
    except BaseException as error:
        failure_access_core = _current_access_core(
            status=f"FAIL_prefix_at_{stage}",
            top_level_reads=top_level_reads,
            inputs=inputs,
            operations=operations,
        )
        _publish_failure(
            output_root,
            stage=stage,
            error=error,
            access_core=failure_access_core,
        )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-file-sha256", required=True)
    parser.add_argument("--authorization-file-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    return run(
        review_file_sha256=arguments.review_file_sha256,
        authorization_file_sha256=arguments.authorization_file_sha256,
    )


if __name__ == "__main__":
    raise SystemExit(main())
