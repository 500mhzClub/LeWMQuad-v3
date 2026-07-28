#!/usr/bin/env python3
"""Calibrate the admitted V4 physical-evidence head on development roles."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import traceback
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PREREGISTRATION_COMMIT = "e983e0abd9349426f69262563e12d90a4488180e"
OUTPUT_RELATIVE_PATH = (
    ".generated/go2_rgb_swept_progress_survival_joint_jepa_v4_"
    "physical_evidence_calibration/attempt_v1"
)
CANDIDATE_ROOT_RELATIVE_PATH = (
    ".generated/go2_rgb_swept_progress_survival_joint_jepa_v4_"
    "candidate_admission/attempt_v1"
)
CANDIDATE_RECEIPT_FILE_SHA256 = (
    "7b21e9a908c05f56c344a74682ee0a3d912c449920d57ee9298619f53c9f66f1"
)
CANDIDATE_RECEIPT_CONTENT_SHA256 = (
    "247e9f1d81cb143631c4be4b85173f707516ff5cf32a0e9e08ca6d8100420f8f"
)
CANDIDATE_CHECKPOINT_BYTE_COUNT = 25_673_535
CANDIDATE_CHECKPOINT_SHA256 = (
    "f8a330d1a4834e4cc61f7acae00069f866a37a5693464e6fbb93b998a971d37a"
)
CANDIDATE_RECEIPT_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v4_"
    "candidate_admission_receipt_v1"
)
RESULT_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v4_"
    "physical_evidence_calibration_result_v1"
)
FAILURE_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v4_"
    "physical_evidence_calibration_failure_v1"
)
ROLE_COUNTS = {"probability_calibration": 415, "checkpoint_selection": 495}
ROLE_CELL_COUNTS = {
    role: count * 64 * 64 for role, count in ROLE_COUNTS.items()
}
BATCH_SIZE = 16
FREE_CANDIDATES = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99)
OCCUPIED_CANDIDATES = (0.01, 0.02, 0.05, 0.10, 0.20, 0.35)
UNKNOWN_CANDIDATES = OCCUPIED_CANDIDATES
OCCUPIED_DETECTION_CANDIDATES = (0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50)
SOURCE_SHA256 = {
    "lewm/benchmarks/go2_rgb_swept_progress_survival_joint_jepa_v4_g2_adapter.py": (
        "1ddbfd743d89614932823ae2247534ac6a76e2eaaf031911617a9311562b4b58"
    ),
    "lewm/hierarchical_probability_calibration.py": (
        "2a41a69d4bf981415f3c3ae6c437e78b3c07e781a603602f7ca58e4e6f785f2b"
    ),
    "lewm/benchmarks/traversability_metrics.py": (
        "97be0acb1a9cf6e170db90945c908a1a30b2ce0a230a5664024b8c06edd03396"
    ),
    "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v1.py": (
        "79e66a4ca5bd814030f374413e4ac0a2edda2552d0614ec23b54b6b0e52ff1b6"
    ),
    "scripts/run_go2_direct_egocentric_bev_state_jepa_v1.py": (
        "33617086a5481f2fa0bf8ae6993110c40bf8db85f066d1d6e874dde12fb07000"
    ),
    "scripts/run_go2_rgb_jepa_encoder_pretraining_v1.py": (
        "ce256dcb1ef67dff313855680365ce07d867aca986dfcad7b8e9493373fe099c"
    ),
    "lewm/benchmarks/go2_rgb_jepa_encoder_pretraining_v1.py": (
        "8c35f0cbafe78185ac74d4412914c177de20f899b0f009a9b9dc7aafdf7695a5"
    ),
    "scripts/run_go2_shared_jepa_v5_matched_training_v1.py": (
        "e98bd8cceed26288ebcbf8a02eac03c72be6d06a539953927754353e049a5578"
    ),
    "lewm/benchmarks/go2_shared_jepa_v5_matched_training_v1.py": (
        "53a7fac793a1b46764d49e7259fd637ec02b20111927effd01cdcd09682c206a"
    ),
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _content_sha256(value: Mapping[str, Any]) -> str:
    core = dict(value)
    core.pop("content_sha256", None)
    return hashlib.sha256(_canonical_bytes(core)).hexdigest()


def _hashed(core: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(core)
    value["content_sha256"] = hashlib.sha256(_canonical_bytes(value)).hexdigest()
    return value


def _no_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _parse_canonical(raw: bytes, *, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise PermissionError(f"{name} is not strict JSON") from error
    if type(value) is not dict or raw != _canonical_bytes(value) + b"\n":
        raise PermissionError(f"{name} is not canonical JSON")
    if value.get("content_sha256") != _content_sha256(value):
        raise PermissionError(f"{name} content hash changed")
    return value


def _read_regular(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise PermissionError(f"input is not a regular file: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        raw = b"".join(chunks)
        if len(raw) != metadata.st_size:
            raise OSError(f"input changed during read: {path}")
        return raw
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, raw: bytes) -> Mapping[str, Any]:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"write-once output exists: {path}")
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return {
        "path": path.name,
        "byte_count": len(raw),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _write_json(path: Path, core: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    value = _hashed(core)
    raw = _canonical_bytes(value) + b"\n"
    binding = dict(_atomic_write(path, raw))
    binding["content_sha256"] = value["content_sha256"]
    return value, binding


def _fresh_output(repository_root: Path) -> Path:
    output = repository_root / OUTPUT_RELATIVE_PATH
    if output.exists() or output.is_symlink():
        raise FileExistsError("fresh physical-evidence calibration attempt_v1 exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    return output


def _validate_sources(repository_root: Path) -> Mapping[str, str]:
    observed = {
        relative: hashlib.sha256(_read_regular(repository_root / relative)).hexdigest()
        for relative in SOURCE_SHA256
    }
    if observed != SOURCE_SHA256:
        raise PermissionError("frozen calibration dependency source changed")
    return observed


def _load_candidate(repository_root: Path, access: dict[str, int]) -> Any:
    from lewm.benchmarks import (
        go2_rgb_swept_progress_survival_joint_jepa_v4_g2_adapter as adapter,
    )

    root = repository_root / CANDIDATE_ROOT_RELATIVE_PATH
    receipt_raw = _read_regular(root / "candidate_receipt.json")
    access["candidate_receipt_reads"] += 1
    if hashlib.sha256(receipt_raw).hexdigest() != CANDIDATE_RECEIPT_FILE_SHA256:
        raise PermissionError("candidate receipt file hash changed")
    receipt = _parse_canonical(receipt_raw, name="candidate receipt")
    binding = receipt.get("checkpoint", {}).get("candidate_binding")
    if (
        receipt.get("schema") != CANDIDATE_RECEIPT_SCHEMA
        or receipt.get("status") != "ADMITTED_PRE_G2_CANDIDATE"
        or receipt.get("content_sha256") != CANDIDATE_RECEIPT_CONTENT_SHA256
        or receipt.get("authority", {}).get("pre_g2_candidate") is not True
        or receipt.get("authority", {}).get("g2_qualified") is not False
        or binding
        != {
            "path": "candidate_checkpoint.pt",
            "byte_count": CANDIDATE_CHECKPOINT_BYTE_COUNT,
            "file_sha256": CANDIDATE_CHECKPOINT_SHA256,
        }
    ):
        raise PermissionError("candidate receipt contract changed")
    checkpoint_raw = _read_regular(root / "candidate_checkpoint.pt")
    access["candidate_checkpoint_reads"] += 1
    if (
        len(checkpoint_raw) != CANDIDATE_CHECKPOINT_BYTE_COUNT
        or hashlib.sha256(checkpoint_raw).hexdigest() != CANDIDATE_CHECKPOINT_SHA256
    ):
        raise PermissionError("candidate checkpoint identity changed")
    access["candidate_checkpoint_loads"] += 1
    return adapter.load_checkpoint(checkpoint_raw)


def _build_data_boundary(repository_root: Path) -> tuple[Any, Any, Any, dict[str, Any]]:
    if repository_root.resolve() != ROOT.resolve():
        raise PermissionError("raw development inputs require the canonical repository root")
    import numpy as np
    from PIL import Image
    import torch
    from scripts import run_go2_direct_egocentric_bev_state_jepa_v1 as direct
    from scripts import run_go2_shared_jepa_v5_matched_training_v1 as matched

    runtime = SimpleNamespace(np=np, Image=Image, torch=torch)
    contract = direct.contract
    authorization = {
        "raw": {
            "manifest": dict(
                contract.RUNTIME_BINDINGS[contract.RAW_MANIFEST_RELATIVE_PATH]
            ),
            "audit": dict(
                contract.RUNTIME_BINDINGS[contract.RAW_AUDIT_RELATIVE_PATH]
            ),
        }
    }
    progress: dict[str, Any] = {}
    inputs = direct._construct_raw_inputs_with_progress(
        matched, runtime, authorization, progress
    )
    direct._normalize_endpoint_paths(inputs)
    loader = direct.DirectBevNarrowLoader(runtime, inputs, progress=progress)
    return runtime, inputs, loader, progress


def _collect_role(
    model: Any,
    loader: Any,
    pairs: Sequence[Mapping[str, Any]],
    *,
    role: str,
    torch: Any,
    batch_size: int = BATCH_SIZE,
) -> tuple[Any, Any, Mapping[str, Any]]:
    expected = ROLE_COUNTS[role]
    if len(pairs) != expected:
        raise PermissionError(f"{role} pair count changed")
    if len({str(pair.get("scene_id")) for pair in pairs}) != 8:
        raise PermissionError(f"{role} scene count changed")
    endpoint_ids = [str(pair.get("next_endpoint_sha256")) for pair in pairs]
    if any(not identity or identity == "None" for identity in endpoint_ids):
        raise PermissionError(f"{role} next-endpoint identity changed")
    logits_parts, label_parts = [], []
    state_before = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    for start in range(0, len(endpoint_ids), int(batch_size)):
        selected = endpoint_ids[start : start + int(batch_size)]
        images, labels = loader.endpoint_batch(
            selected,
            torch.device("cpu"),
            role=role,
            stage=f"v4_physical_evidence_{role}",
        )
        if tuple(images.shape) != (len(selected), 3, 112, 112):
            raise PermissionError(f"{role} RGB batch shape changed")
        if tuple(labels.shape) != (len(selected), 64, 64):
            raise PermissionError(f"{role} label batch shape changed")
        with torch.inference_mode():
            latent = model.encode_online(images)
            logits = model.semantic_logits_from_latent(latent)
        if (
            tuple(logits.shape) != (len(selected), 3, 64, 64)
            or logits.dtype != torch.float32
            or not bool(torch.isfinite(logits).all())
            or labels.dtype != torch.long
            or not bool(((labels >= 0) & (labels <= 2)).all())
        ):
            raise PermissionError(f"{role} semantic output or target changed")
        logits_parts.append(logits.detach().cpu().contiguous())
        label_parts.append(labels.detach().cpu().contiguous())
    logits = torch.cat(logits_parts)
    labels = torch.cat(label_parts)
    if logits.shape[0] * 64 * 64 != ROLE_CELL_COUNTS[role]:
        raise PermissionError(f"{role} cell count changed")
    if any(
        not torch.equal(value.detach().cpu(), state_before[name])
        for name, value in model.state_dict().items()
    ):
        raise RuntimeError("calibration inference mutated candidate state")
    return logits, labels, {
        "role": role,
        "pair_count": len(pairs),
        "cell_count": int(labels.numel()),
        "next_endpoint_order_sha256": hashlib.sha256(
            _canonical_bytes(endpoint_ids)
        ).hexdigest(),
        "batch_count": len(logits_parts),
        "model_state_mutated": False,
    }


def _physical_metrics(metrics: Any) -> Mapping[str, Any]:
    return {
        "admitted_observable_physical_free_precision": float(
            metrics.planner_admitted_free_precision
        ),
        "directly_observable_physical_obstacle_recall_within_2m": float(
            metrics.obstacle_detection_recall_within_range
        ),
        "useful_observable_physical_free_recall": float(
            metrics.useful_traversable_recall
        ),
        "observable_physical_obstacle_exclusion_recall_within_2m": float(
            metrics.obstacle_exclusion_recall_within_range
        ),
        "unknown_evidence_admission_rate": float(metrics.unknown_admission_rate),
        "free_probability_brier": float(metrics.free_probability_brier),
        "free_probability_ece": float(metrics.free_probability_ece),
    }


def _fit_select_score(
    calibration_logits: Any,
    calibration_labels: Any,
    selection_logits: Any,
    selection_labels: Any,
    *,
    provenance: Mapping[str, Any],
) -> Mapping[str, Any]:
    import numpy as np
    from lewm.benchmarks.traversability_metrics import (
        evaluate_traversability,
        select_conservative_thresholds,
    )
    from lewm.hierarchical_probability_calibration import (
        apply_hierarchical_probability_calibration,
        evaluate_hierarchical_probability_calibration,
        fit_hierarchical_probability_calibration,
        validate_hierarchical_probability_calibration,
    )

    calibration = fit_hierarchical_probability_calibration(
        calibration_logits,
        calibration_labels,
        provenance=provenance,
        mask=None,
        class_dim=1,
        maximum_iterations=80,
        ece_bins=15,
    )
    validate_hierarchical_probability_calibration(calibration)
    calibration_probabilities = apply_hierarchical_probability_calibration(
        calibration_logits, calibration, class_dim=1
    ).numpy()
    selection_probabilities = apply_hierarchical_probability_calibration(
        selection_logits, calibration, class_dim=1
    ).numpy()
    forward = np.linspace(-0.95, 5.35, calibration_labels.shape[-2], dtype=np.float64)
    left = np.linspace(-3.15, 3.15, calibration_labels.shape[-1], dtype=np.float64)
    distance_grid = np.sqrt(forward[:, None] ** 2 + left[None, :] ** 2)
    calibration_distances = np.broadcast_to(
        distance_grid, tuple(calibration_labels.shape)
    )
    selection_distances = np.broadcast_to(distance_grid, tuple(selection_labels.shape))
    threshold_selection = select_conservative_thresholds(
        calibration_probabilities,
        calibration_labels.numpy(),
        calibration_distances,
        free_probability_candidates=FREE_CANDIDATES,
        occupied_probability_candidates=OCCUPIED_CANDIDATES,
        unknown_probability_candidates=UNKNOWN_CANDIDATES,
        occupied_detection_probability_candidates=OCCUPIED_DETECTION_CANDIDATES,
        evaluation_mask=None,
        minimum_free_precision=0.99,
        minimum_obstacle_exclusion_recall=0.95,
        minimum_obstacle_detection_recall=0.95,
        obstacle_range_m=2.0,
    )
    thresholds = threshold_selection.thresholds
    calibration_traversability = evaluate_traversability(
        calibration_probabilities,
        calibration_labels.numpy(),
        calibration_distances,
        thresholds=thresholds,
        evaluation_mask=None,
        obstacle_range_m=2.0,
        calibration_bins=15,
    )
    selection_traversability = evaluate_traversability(
        selection_probabilities,
        selection_labels.numpy(),
        selection_distances,
        thresholds=thresholds,
        evaluation_mask=None,
        obstacle_range_m=2.0,
        calibration_bins=15,
    )
    selection_calibration = {
        "before": evaluate_hierarchical_probability_calibration(
            selection_logits, selection_labels, None, mask=None, class_dim=1, ece_bins=15
        ),
        "after": evaluate_hierarchical_probability_calibration(
            selection_logits,
            selection_labels,
            calibration,
            mask=None,
            class_dim=1,
            ece_bins=15,
        ),
    }
    physical = _physical_metrics(selection_traversability)
    checks = {
        "calibration_grid_has_passing_tuple": threshold_selection.passing_candidate_count
        > 0,
        "selection_physical_free_precision_ge_0_99": physical[
            "admitted_observable_physical_free_precision"
        ]
        >= 0.99,
        "selection_obstacle_detection_within_2m_ge_0_95": physical[
            "directly_observable_physical_obstacle_recall_within_2m"
        ]
        >= 0.95,
        "selection_useful_physical_free_recall_ge_0_90": physical[
            "useful_observable_physical_free_recall"
        ]
        >= 0.90,
        "selection_obstacle_exclusion_within_2m_ge_0_95": physical[
            "observable_physical_obstacle_exclusion_recall_within_2m"
        ]
        >= 0.95,
    }
    return {
        "calibration": calibration,
        "threshold_selection": {
            "candidate_count": threshold_selection.candidate_count,
            "passing_candidate_count": threshold_selection.passing_candidate_count,
            "thresholds": asdict(thresholds),
            "calibration_role_metrics": _physical_metrics(
                calibration_traversability
            ),
        },
        "selection": {
            "calibration_metrics": selection_calibration,
            "traversability": asdict(selection_traversability),
            "physical_evidence": physical,
        },
        "gate": {
            "status": (
                "PASS_DEVELOPMENT_PHYSICAL_EVIDENCE"
                if all(checks.values())
                else "FAIL_DEVELOPMENT_PHYSICAL_EVIDENCE"
            ),
            "passed": all(checks.values()),
            "checks": checks,
            "failed_checks": [name for name, passed in checks.items() if not passed],
        },
    }


def _new_access() -> dict[str, int]:
    return {
        "candidate_receipt_reads": 0,
        "candidate_checkpoint_reads": 0,
        "candidate_checkpoint_loads": 0,
        "calibration_fit_calls": 0,
        "threshold_selection_calls": 0,
        "model_backward_calls": 0,
        "model_optimizer_steps": 0,
        "model_ema_steps": 0,
        "predictor_calls": 0,
        "train_role_payload_requests": 0,
        "g2_operations": 0,
        "navigation_operations": 0,
        "heldout_reads": 0,
        "sealed_reads": 0,
    }


def execute(*, repository_root: Path = ROOT) -> Mapping[str, Any]:
    repository_root = Path(repository_root).absolute()
    output = _fresh_output(repository_root)
    access = _new_access()
    stage = "reserved_output"
    try:
        stage = "validated_sources"
        source_hashes = _validate_sources(repository_root)
        stage = "loaded_candidate"
        model = _load_candidate(repository_root, access)
        stage = "constructed_development_boundary"
        runtime, inputs, loader, progress = _build_data_boundary(repository_root)
        role_arrays: dict[str, tuple[Any, Any]] = {}
        role_receipts: dict[str, Mapping[str, Any]] = {}
        for role in ("probability_calibration", "checkpoint_selection"):
            stage = f"collected_{role}"
            pairs = inputs.role_pairs(role)
            logits, labels, receipt = _collect_role(
                model,
                loader,
                pairs,
                role=role,
                torch=runtime.torch,
            )
            role_arrays[role] = (logits, labels)
            role_receipts[role] = receipt
        stage = "fit_select_score"
        access["calibration_fit_calls"] += 1
        access["threshold_selection_calls"] += 1
        science = _fit_select_score(
            *role_arrays["probability_calibration"],
            *role_arrays["checkpoint_selection"],
            provenance={
                "role": "probability_calibration",
                "candidate_checkpoint_sha256": CANDIDATE_CHECKPOINT_SHA256,
                "candidate_receipt_content_sha256": CANDIDATE_RECEIPT_CONTENT_SHA256,
                "pair_count": ROLE_COUNTS["probability_calibration"],
                "cell_count": ROLE_CELL_COUNTS["probability_calibration"],
                "next_endpoint_order_sha256": role_receipts[
                    "probability_calibration"
                ]["next_endpoint_order_sha256"],
                "all_cells_used": True,
            },
        )
        stage = "validated_access"
        loader_counts = loader.model_facing_access_counts()
        if (
            loader_counts["endpoint_rgb_row_request_count"] != sum(ROLE_COUNTS.values())
            or loader_counts["raster_label_row_request_count"]
            != sum(ROLE_COUNTS.values())
            or any(
                loader_counts[name] != 0
                for name in (
                    "current_rgb_row_request_count",
                    "next_rgb_row_request_count",
                    "fixed_negative_rgb_row_request_count",
                )
            )
        ):
            raise PermissionError("model-facing development access changed")
        payload_records = [
            record
            for record in inputs.consumed.values()
            if record.get("kind") in {"development_rgb", "raw_supervision"}
        ]
        if any("train" in record.get("roles", []) for record in payload_records):
            access["train_role_payload_requests"] += 1
            raise PermissionError("train-role payload entered calibration")
        stage = "write_outputs"
        calibration_raw = _canonical_bytes(science["calibration"]) + b"\n"
        calibration_binding = _atomic_write(output / "calibration.json", calibration_raw)
        result, _ = _write_json(
            output / "result.json",
            {
                "schema": RESULT_SCHEMA,
                "status": science["gate"]["status"],
                "preregistration_commit": PREREGISTRATION_COMMIT,
                "candidate": {
                    "receipt_file_sha256": CANDIDATE_RECEIPT_FILE_SHA256,
                    "receipt_content_sha256": CANDIDATE_RECEIPT_CONTENT_SHA256,
                    "checkpoint_byte_count": CANDIDATE_CHECKPOINT_BYTE_COUNT,
                    "checkpoint_file_sha256": CANDIDATE_CHECKPOINT_SHA256,
                },
                "source_sha256": source_hashes,
                "roles": role_receipts,
                "calibration_artifact": {
                    **calibration_binding,
                    "content_sha256": science["calibration"]["content_sha256"],
                    "id": science["calibration"]["id"],
                },
                "threshold_selection": science["threshold_selection"],
                "selection": science["selection"],
                "gate": science["gate"],
                "routing": {
                    "status": "NOT_APPLICABLE",
                    "included_in_gate": False,
                    "reason": "physical_evidence_is_not_configuration_space",
                    "deferred_to": "G3_post_memory_multi_view_fusion",
                },
                "raw_access": {
                    "constructor_reads": progress.get("_raw_constructor_reads", {}),
                    "model_facing": loader_counts,
                    "consumed_unique_file_count": len(inputs.consumed),
                    "consumed_records_sha256": hashlib.sha256(
                        _canonical_bytes(
                            [inputs.consumed[name] for name in sorted(inputs.consumed)]
                        )
                    ).hexdigest(),
                },
                "access": access,
                "authority": {
                    "development_only": True,
                    "development_physical_evidence_passed": science["gate"]["passed"],
                    "g2_opened": False,
                    "g2_qualified": False,
                    "navigation_qualified": False,
                    "promotion_performed": False,
                    "deployment_authorized": False,
                    "training_or_resume_authorized": False,
                    "heldout_or_sealed_opened": False,
                },
            },
        )
        return result
    except Exception as error:
        failure, _ = _write_json(
            output / "failure.json",
            {
                "schema": FAILURE_SCHEMA,
                "status": "FAILED_OPERATIONALLY",
                "stage": stage,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
                "access": access,
                "authority": {
                    "development_only": True,
                    "development_physical_evidence_passed": False,
                    "g2_opened": False,
                    "g2_qualified": False,
                    "navigation_qualified": False,
                    "promotion_performed": False,
                    "deployment_authorized": False,
                    "training_or_resume_authorized": False,
                    "heldout_or_sealed_opened": False,
                },
            },
        )
        return failure


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    result = execute()
    print(
        json.dumps(
            {"status": result["status"], "output": OUTPUT_RELATIVE_PATH},
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    if result["status"] == "FAILED_OPERATIONALLY":
        return 1
    return 0 if result.get("gate", {}).get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
