#!/usr/bin/env python3
"""Admit the single frozen V4 checkpoint as a load-valid pre-G2 candidate."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULT_RELATIVE_PATH = (
    ".generated/go2_rgb_swept_progress_survival_joint_jepa_v4_"
    "residual_local_semantic_decoder/attempt_v1/result.json"
)
OUTPUT_RELATIVE_PATH = (
    ".generated/go2_rgb_swept_progress_survival_joint_jepa_v4_"
    "candidate_admission/attempt_v1"
)
EXPECTED_RESULT_FILE_SHA256 = (
    "bf93c96cf020553be74d51847c6876e345cd6cc391b05cec186e36b20ca15aa4"
)
EXPECTED_RESULT_CONTENT_SHA256 = (
    "27ecf4895dfea01a1e5bb4f6f13f3add6a182a8dfa4b9f8651204bd1e6222ad8"
)
RESULT_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v4_residual_local_"
    "semantic_decoder_result_v1"
)
CHECKPOINT_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v4_residual_local_"
    "semantic_decoder_checkpoint_v1"
)
RECEIPT_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v4_candidate_"
    "admission_receipt_v1"
)
FAILURE_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v4_candidate_"
    "admission_failure_v1"
)
SOURCE_BINDINGS = (
    {
        "role": "v4_model",
        "path": "lewm/models/geometry_anchored_swept_progress_survival_joint_jepa_v4_residual_local_semantic_decoder.py",
        "file_sha256": "1c5a26f02a856d9a84903063c53bf23095142d86885787556b09388c508711ef",
    },
    {
        "role": "v4_executor",
        "path": "scripts/execute_go2_rgb_swept_progress_survival_joint_jepa_v4_residual_local_semantic_decoder.py",
        "file_sha256": "243ef91ccec4e1fcdfa5a0c3f112bf4c645f46ba7de8692c1dddcb47f87c9f40",
    },
)
COMMITS = {
    "preregistration": "9f9ab784b4bfa827585ec095f2a7f7a30333480a",
    "source": "aaa47a138d0eeb78aa20d9524e67f813f7a74a41",
    "execution_binding": "5a48b878c97717e27bf7e4bdb1c6a13c1687117e",
    "terminal_result": "8b3a8063b087c81030189deadc6c5f6e1c7d44c3",
    "candidate_admission_preregistration": "b5b4ca50b50257872c9ee12a96b901710e35bac9",
}
ACCOUNTING = {
    "updates": 1_000,
    "presentations": 16_000,
    "microbatch_graphs": 4_000,
    "backward_calls": 4_000,
    "optimizer_steps": 1_000,
    "ema_steps": 1_000,
    "predictor_forwards": 4_000,
    "predictor_objectives": 4_000,
}
AUXILIARY_OBJECTIVE = {
    "name": "occupied_vs_rest_safety",
    "coefficient": 0.5,
    "logit_definition": (
        "occupied_semantic_logit_minus_logsumexp_free_and_unknown_semantic_logits"
    ),
    "row_balancing": (
        "per_raster_row_equal_average_of_present_occupied_and_rest_target_classes"
    ),
    "current_next_aggregation": "equal_average",
    "normalization": "binary_cross_entropy_with_logits_divided_by_log_2",
    "new_trainable_parameters": False,
}
DECODER_ARCHITECTURE = {
    "schema": "lewm_residual_local_semantic_decoder_v4_architecture_v1",
    "merge": "base_logits_plus_residual_logits",
    "base": {
        "type": "Conv2d", "in_channels": 64, "out_channels": 3,
        "kernel_size": [1, 1], "bias": True,
        "identity": "exact_existing_v3_semantic_head",
    },
    "residual": {
        "local": {
            "type": "Conv2d", "in_channels": 64, "out_channels": 64,
            "kernel_size": [3, 3], "stride": [1, 1], "padding": [1, 1],
            "bias": True,
        },
        "activation": {"type": "GELU", "approximate": "none"},
        "output": {
            "type": "Conv2d", "in_channels": 64, "out_channels": 3,
            "kernel_size": [1, 1], "bias": True,
            "weight_initialization": "exact_zeros",
            "bias_initialization": "exact_zeros",
        },
    },
    "added_trainable_parameter_count": 37_123,
    "initialization_seed": 20_260_713,
    "visibility_mask": "inherited_bev_lift_anchor_in_frustum_post_logits",
    "normalization_layers": 0,
}
INITIAL_DECODER_RECEIPT = {
    "architecture": DECODER_ARCHITECTURE,
    "initial_residual_output_exactly_zero": True,
    "semantic_parameter_count": 37_318,
    "added_parameter_count": 37_123,
    "all_semantic_parameters_in_lift_semantic_exactly_once": True,
    "visibility_mask": {
        "shape": [64, 64], "dtype": "bool", "true_cell_count": 1_964,
        "sha256": "cbcdb7d6fda08626522732ff092d90a87f5b5f2cd2534baf2bb4aa556d832753",
        "application": "inherited_post_logits",
    },
}
GATE_THRESHOLDS = {
    "semantic_balanced_accuracy_min": 0.80,
    "semantic_free_recall_min": 0.85,
    "semantic_occupied_recall_min": 0.70,
    "semantic_unknown_recall_min": 0.90,
    "semantic_rough_occupied_recall_min": 0.65,
    "informative_utility_min": 0.85,
    "family_informative_utility_min": 0.70,
    "selected_zero_prefix_rate_max": 0.05,
    "family_selected_zero_prefix_rate_max": 0.20,
    "pair_concordance_min": 0.75,
    "family_pair_concordance_min": 0.60,
    "positive_control_family_count_min": 6,
}
BASE_GATE_CHECKS = {
    "semantic_balanced_accuracy", "semantic_free_recall",
    "semantic_occupied_recall", "semantic_unknown_recall",
    "semantic_rough_occupied_recall", "selection_registered_families",
    "selection_informative_utility", "selection_zero_prefix_rate",
    "selection_pair_concordance", "all_family_utility",
    "all_family_zero_prefix_rate", "all_family_pair_concordance",
}
CONTROL_NAMES = (
    "coordinate_matched_persistence", "shuffled_action", "wrong_rgb",
    "train_action_mean_prior",
)
GATE_CHECKS = BASE_GATE_CHECKS | {
    f"{name}:{suffix}"
    for name in CONTROL_NAMES
    for suffix in (
        "positive_equal_scene_delta", "positive_bootstrap_lower_95",
        "positive_family_count",
    )
}
ACTION_VOCABULARY = (
    "arc_left", "arc_right", "backward", "forward_fast", "forward_medium",
    "forward_slow", "hold", "yaw_left", "yaw_right",
)
PAYLOAD_KEYS = {
    "schema", "development_only", "resume_authorized", "qualified",
    "constructor_initialization_seed", "semantic_decoder_initialization_seed",
    "experiment_seed", "initialization_source",
    "predecessor_experiment_checkpoint_read", "auxiliary_objective",
    "initial_semantic_decoder", "accounting", "model_state_dict",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _with_content_hash(core: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(core)
    value["content_sha256"] = _canonical_sha256(value)
    return value


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _parse_canonical(raw: bytes, *, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_no_duplicate_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON constant: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise PermissionError(f"{name} is not valid canonical JSON") from error
    if type(value) is not dict or raw != _canonical_bytes(value) + b"\n":
        raise PermissionError(f"{name} is not one canonical JSON object")
    core = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != _canonical_sha256(core):
        raise PermissionError(f"{name} content hash changed")
    return value


def _read_regular(path: Path, *, name: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise PermissionError(f"{name} is not one regular file")
        parts: list[bytes] = []
        while part := os.read(descriptor, 1024 * 1024):
            parts.append(part)
        raw = b"".join(parts)
        if len(raw) != metadata.st_size:
            raise OSError(f"{name} changed during its single read")
        return raw
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, raw: bytes) -> Mapping[str, Any]:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"write-once output exists: {path.name}")
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return {"path": path.name, "byte_count": len(raw),
            "file_sha256": hashlib.sha256(raw).hexdigest()}


def _write_json(path: Path, core: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    value = _with_content_hash(core)
    binding = dict(_atomic_write(path, _canonical_bytes(value) + b"\n"))
    binding["content_sha256"] = value["content_sha256"]
    return value, binding


def _fresh_output(repository_root: Path) -> Path:
    output = repository_root / OUTPUT_RELATIVE_PATH
    if output.exists() or output.is_symlink():
        raise FileExistsError("fresh V4 candidate admission attempt_v1 already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    return output


def _new_access() -> dict[str, int]:
    return {
        "source_file_reads": 0, "result_file_reads": 0,
        "checkpoint_file_reads": 0, "checkpoint_deserializations": 0,
        "synthetic_batches": 0, "candidate_copy_writes": 0,
        "dataset_reads": 0, "trace_reads": 0, "training_operations": 0,
        "backward_calls": 0, "optimizer_steps": 0, "ema_steps": 0,
        "accelerator_operations": 0, "calibration_operations": 0,
        "g2_operations": 0, "navigation_operations": 0,
        "heldout_reads": 0, "sealed_reads": 0,
    }


def _authority(pre_g2_candidate: bool) -> Mapping[str, bool]:
    return {
        "development_only": True, "pre_g2_candidate": pre_g2_candidate,
        "g2_qualified": False, "navigation_qualified": False,
        "promotion_performed": False, "deployment_authorized": False,
        "heldout_or_sealed_opened": False,
        "resume_or_training_authorized": False,
    }


def _validate_sources(repository_root: Path, access: dict[str, int]) -> list[Mapping[str, Any]]:
    receipts = []
    for binding in SOURCE_BINDINGS:
        raw = _read_regular(repository_root / binding["path"], name=binding["role"])
        access["source_file_reads"] += 1
        digest = hashlib.sha256(raw).hexdigest()
        if digest != binding["file_sha256"]:
            raise PermissionError(f"{binding['role']} source hash changed")
        receipts.append({**binding, "byte_count": len(raw)})
    return receipts


def _validate_result(raw: bytes) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if hashlib.sha256(raw).hexdigest() != EXPECTED_RESULT_FILE_SHA256:
        raise PermissionError("V4 result file hash changed")
    result = _parse_canonical(raw, name="V4 result")
    if result.get("content_sha256") != EXPECTED_RESULT_CONTENT_SHA256:
        raise PermissionError("V4 result content identity changed")
    gate = result.get("gate")
    if (
        result.get("schema") != RESULT_SCHEMA
        or result.get("status") != "PASS_FULL_ARM"
        or type(gate) is not dict
        or gate.get("passed") is not True
        or not all(value is True for value in gate.get("checks", {}).values())
        or gate != {
            "status": "PASS_FULL_ARM", "passed": True,
            "checks": {name: True for name in gate.get("checks", {})},
            "failed_checks": [], "thresholds": GATE_THRESHOLDS,
        }
        or set(gate["checks"]) != GATE_CHECKS
        or result.get("caps") != {"updates": 1_000, "presentations": 16_000}
        or result.get("seeds") != {
            "inherited_fresh_component_constructor": 20_260_712,
            "semantic_decoder": 20_260_713,
            "experiment_and_stochastic_execution": 20_260_728,
            "bootstrap": 20_260_728,
        }
    ):
        raise PermissionError("V4 result status, gate, cap, or seed receipt changed")
    training = result.get("training")
    scientific = result.get("scientific_change_from_v3")
    if (
        type(training) is not dict or training.get("accounting") != ACCOUNTING
        or training.get("joint_from_update_one") is not True
        or training.get("separate_head_or_predictor_training") is not False
        or type(scientific) is not dict
        or scientific.get("initial_semantic_decoder") != INITIAL_DECODER_RECEIPT
        or scientific.get("auxiliary_objective_unchanged") != AUXILIARY_OBJECTIVE
        or result.get("authority") != {
            "development_only": True,
            "g2_navigation_final_evaluation_opened": False,
            "heldout_or_sealed_opened": False, "checkpoint_qualified": False,
            "promotion_performed": False, "retry_or_resume_authorized": False,
        }
        or result.get("access", {}).get("forbidden_input_count") != 0
        or result.get("access", {}).get("g2_navigation_final_evaluation_open_count") != 0
    ):
        raise PermissionError("V4 result accounting, receipt, access, or authority changed")
    if not all(
        type(value) is bool
        for value in (
            *gate["checks"].values(),
            *result["authority"].values(),
            training["joint_from_update_one"],
            training["separate_head_or_predictor_training"],
        )
    ):
        raise PermissionError("V4 result boolean receipts are not exact booleans")
    checkpoint = training.get("checkpoint")
    if (
        type(checkpoint) is not dict
        or set(checkpoint) != {"path", "byte_count", "file_sha256"}
        or checkpoint.get("path") != "checkpoint_update_1000.pt"
        or type(checkpoint.get("byte_count")) is not int
        or checkpoint["byte_count"] <= 0
        or type(checkpoint.get("file_sha256")) is not str
        or len(checkpoint["file_sha256"]) != 64
        or checkpoint["file_sha256"].lower() != checkpoint["file_sha256"]
    ):
        raise PermissionError("V4 checkpoint binding is invalid")
    return result, checkpoint


def _tensor_inventory(state: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], str]:
    if not isinstance(state, Mapping) or not state:
        raise PermissionError("model_state_dict is empty or invalid")
    inventory = []
    for name in sorted(state):
        tensor = state[name]
        if type(name) is not str or type(tensor) is not torch.Tensor:
            raise PermissionError("model_state_dict must be tensor-only")
        if tensor.device.type != "cpu" or tensor.layout != torch.strided:
            raise PermissionError(f"state tensor {name} is not a dense CPU tensor")
        if not bool(torch.isfinite(tensor).all()):
            raise PermissionError(f"state tensor {name} is nonfinite")
        contiguous = tensor.detach().contiguous().reshape(-1)
        raw = contiguous.view(torch.uint8).numpy().tobytes(order="C")
        inventory.append({
            "name": name, "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "byte_count": len(raw),
            "tensor_byte_sha256": hashlib.sha256(raw).hexdigest(),
        })
    return inventory, _canonical_sha256(inventory)


def _model_class() -> Any:
    from lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v4_residual_local_semantic_decoder import (
        GeometryAnchoredSweptProgressSurvivalJointJepaV4,
    )
    return GeometryAnchoredSweptProgressSurvivalJointJepaV4


def _validate_payload(payload: Any) -> Mapping[str, Any]:
    if type(payload) is not dict or set(payload) != PAYLOAD_KEYS:
        raise PermissionError("checkpoint payload key set changed")
    expected = {
        "schema": CHECKPOINT_SCHEMA, "development_only": True,
        "resume_authorized": False, "qualified": False,
        "constructor_initialization_seed": 20_260_712,
        "semantic_decoder_initialization_seed": 20_260_713,
        "experiment_seed": 20_260_728,
        "initialization_source": "exact_n320_encoder_only",
        "predecessor_experiment_checkpoint_read": False,
        "auxiliary_objective": AUXILIARY_OBJECTIVE,
        "initial_semantic_decoder": INITIAL_DECODER_RECEIPT,
        "accounting": ACCOUNTING,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise PermissionError("checkpoint payload receipt changed")
    if (
        payload["development_only"] is not True
        or payload["resume_authorized"] is not False
        or payload["qualified"] is not False
        or payload["predecessor_experiment_checkpoint_read"] is not False
        or any(
            type(payload[name]) is not int
            for name in (
                "constructor_initialization_seed",
                "semantic_decoder_initialization_seed",
                "experiment_seed",
            )
        )
    ):
        raise PermissionError("checkpoint flag or seed types changed")
    return payload["model_state_dict"]


def _load_and_smoke(checkpoint_raw: bytes, access: dict[str, int]) -> Mapping[str, Any]:
    access["checkpoint_deserializations"] += 1
    payload = torch.load(io.BytesIO(checkpoint_raw), map_location="cpu", weights_only=True)
    state = _validate_payload(payload)
    inventory, inventory_digest = _tensor_inventory(state)
    encoder = {
        name.removeprefix("encoder."): tensor
        for name, tensor in state.items() if name.startswith("encoder.")
    }
    mask_name = "predictor.swept_progress_head.sweep_masks"
    if not encoder or mask_name not in state:
        raise PermissionError("checkpoint lacks constructor encoder or sweep masks")
    model = _model_class()(encoder, state[mask_name]).cpu()
    loaded = model.load_state_dict(state, strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise PermissionError("strict checkpoint load was not exact")
    model.eval().requires_grad_(False)
    before_inventory, before_digest = _tensor_inventory(model.state_dict())
    if before_digest != inventory_digest or before_inventory != inventory:
        raise PermissionError("strict-loaded model state differs from checkpoint")
    rgb = torch.linspace(0.0, 1.0, 3 * 112 * 112, dtype=torch.float32).reshape(1, 3, 112, 112)
    access["synthetic_batches"] += 1
    with torch.inference_mode():
        latent = model.encode_online(rgb)
        semantic = model.semantic_logits_from_latent(latent)
        prediction = model.predict_all_actions_with_survival(latent)
    tensors = {
        "latent": (latent, [1, 64, 64, 64]),
        "semantic_logits": (semantic, [1, 3, 64, 64]),
        "predicted_latents": (prediction.predicted_latents, [1, 9, 64, 64, 64]),
        "survival_logits": (prediction.survival_logits, [1, 9, 16]),
    }
    for name, (tensor, shape) in tensors.items():
        if list(tensor.shape) != shape or tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()):
            raise PermissionError(f"synthetic {name} receipt changed")
    if tuple(model.action_vocabulary) != ACTION_VOCABULARY:
        raise PermissionError("action vocabulary changed")
    _, after_digest = _tensor_inventory(model.state_dict())
    if after_digest != before_digest:
        raise PermissionError("synthetic inference mutated model state")
    return {
        "payload": {key: payload[key] for key in sorted(PAYLOAD_KEYS - {"model_state_dict"})},
        "state_inventory": inventory,
        "state_inventory_sha256": inventory_digest,
        "strict_load": {"passed": True, "missing_keys": [], "unexpected_keys": []},
        "synthetic_inference": {
            "device": "cpu", "input_shape": [1, 3, 112, 112],
            "input_dtype": "float32", "input_construction": "torch.linspace(0,1,3*112*112)",
            "output_shapes": {name: shape for name, (_, shape) in tensors.items()},
            "output_dtype": "float32", "all_finite": True,
            "action_vocabulary": list(ACTION_VOCABULARY), "state_mutated": False,
        },
    }


def admit_candidate(*, repository_root: Path = ROOT) -> Mapping[str, Any]:
    repository_root = Path(repository_root).absolute()
    output = _fresh_output(repository_root)
    access = _new_access()
    try:
        sources = _validate_sources(repository_root, access)
        result_path = repository_root / RESULT_RELATIVE_PATH
        result_raw = _read_regular(result_path, name="V4 result")
        access["result_file_reads"] += 1
        result, checkpoint_binding = _validate_result(result_raw)
        checkpoint_path = result_path.parent / checkpoint_binding["path"]
        checkpoint_raw = _read_regular(checkpoint_path, name="V4 checkpoint")
        access["checkpoint_file_reads"] += 1
        if (
            len(checkpoint_raw) != checkpoint_binding["byte_count"]
            or hashlib.sha256(checkpoint_raw).hexdigest() != checkpoint_binding["file_sha256"]
        ):
            raise PermissionError("checkpoint bytes do not match verified result binding")
        admission = _load_and_smoke(checkpoint_raw, access)
        copied = dict(_atomic_write(output / "candidate_checkpoint.pt", checkpoint_raw))
        access["candidate_copy_writes"] += 1
        if copied["byte_count"] != checkpoint_binding["byte_count"] or copied["file_sha256"] != checkpoint_binding["file_sha256"]:
            raise RuntimeError("candidate copy identity changed")
        receipt, _ = _write_json(output / "candidate_receipt.json", {
            "schema": RECEIPT_SCHEMA, "status": "ADMITTED_PRE_G2_CANDIDATE",
            "commits": COMMITS, "sources": sources,
            "result": {
                "path": RESULT_RELATIVE_PATH,
                "file_sha256": EXPECTED_RESULT_FILE_SHA256,
                "content_sha256": EXPECTED_RESULT_CONTENT_SHA256,
                "schema": result["schema"], "status": result["status"],
                "gate_check_count": len(result["gate"]["checks"]),
                "accounting": result["training"]["accounting"],
            },
            "checkpoint": {
                "original_binding": checkpoint_binding,
                "candidate_binding": copied,
                **admission,
            },
            "access": access, "authority": _authority(True),
        })
        return receipt
    except Exception as error:
        failure, _ = _write_json(output / "failure.json", {
            "schema": FAILURE_SCHEMA, "status": "FAILED_CLOSED",
            "error": {"type": type(error).__name__, "message": str(error)},
            "access": access, "authority": _authority(False),
        })
        return failure


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    result = admit_candidate()
    print(json.dumps({"status": result["status"], "output": OUTPUT_RELATIVE_PATH}, sort_keys=True))
    return 0 if result["status"] == "ADMITTED_PRE_G2_CANDIDATE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
