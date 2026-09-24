#!/usr/bin/env python3
"""Run the one-shot DINOv2 physical-readout development calibration."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import __version__ as PILLOW_VERSION
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks import (  # noqa: E402
    go2_dinov2_physical_readout_calibration_v1 as calibration,
)
from lewm.benchmarks import go2_matched_branch_successor_screen_v1 as screen_data  # noqa: E402
from lewm.datasets import go2_world_model_counterfactual_pilot_v1 as pilot_consumer  # noqa: E402
from lewm.datasets.go2_world_model_counterfactual_pilot_v1 import (  # noqa: E402
    read_bound_rgb_bytes_v1,
)
from scripts import run_go2_matched_branch_successor_screen_v1 as predecessor_runner  # noqa: E402


SCHEMA = "lewm_go2_dinov2_physical_readout_calibration_result_v1"
TERMINAL_SCHEMA = "lewm_go2_dinov2_physical_readout_calibration_terminal_v1"
RESERVATION_SCHEMA = "lewm_go2_dinov2_physical_readout_calibration_reservation_v1"
EVAL_CACHE_SCHEMA = "lewm_go2_dinov2_physical_readout_eval_feature_cache_v1"
EVAL_CACHE_RECEIPT_SCHEMA = (
    "lewm_go2_dinov2_physical_readout_eval_feature_cache_receipt_v1"
)
AUTHORITY_SCHEMA = "lewm_go2_dinov2_physical_readout_calibration_execution_authority_v1"
AUTHORITY_STATUS = "AUTHORIZED_ONE_EXACT_EVAL_RGB_CALIBRATION_ATTEMPT"
SOURCE_REVIEW_SCHEMA = "lewm_go2_dinov2_physical_readout_calibration_source_review_v1"
SOURCE_REVIEW_STATUS = "PASS_INDEPENDENT_SOURCE_REVIEW"
SOURCE_REVIEW = REPO_ROOT / (
    "docs/lewm_go2_dinov2_physical_readout_calibration_v1_"
    "source_review_2026-08-03.json"
)
SOURCE_REVIEW_FIELDS = frozenset(
    {
        "schema",
        "status",
        "review_date",
        "reviewer",
        "protected_material_opened",
        "preregistration_binding",
        "source_bindings",
        "checks",
        "audit_history",
        "findings",
    }
)
SOURCE_REVIEW_CHECKS = frozenset(
    {
        "frozen_preregistration_binding_exact",
        "source_closure_complete_and_exact",
        "strict_loader_recomputes_physical_ranks_from_bound_labels_and_tolerance",
        "train_eval_roles_scenes_and_artifacts_disjoint_and_exact",
        "train_cache_reused_metadata_only_and_exact",
        "eval_rgb_exactly_once_through_bound_decoded_pixel_reader",
        "dinov2_source_checkpoint_preprocessing_and_output_exact",
        "descriptor_readouts_and_controls_match_preregistration",
        "oracle_random_hold_and_safety_semantics_match_preregistration",
        "scene_cluster_bootstrap_and_gate_directions_match_preregistration",
        "cache_only_replay_and_full_execution_rehash_match_preregistration",
        "authority_output_and_failure_handling_fail_closed",
        "no_training_collection_or_protected_access_authorized",
        "focused_tests_92_of_92_passed",
        "rocm_focused_tests_22_of_22_passed",
        "compile_and_whitespace_checks_passed",
    }
)

PASS_STATUS = "PASS_DINO_TRUE_FUTURE_PHYSICAL_READOUT_HEADROOM_ESTABLISHED"
STOP_STATUS = "STOP_DINO_TRUE_FUTURE_PHYSICAL_READOUT_HEADROOM_NOT_ESTABLISHED"
FAIL_STATUS = "FAIL_INFRASTRUCTURE_NO_SCIENTIFIC_DECISION"
TERMINAL_STATUSES = frozenset({PASS_STATUS, STOP_STATUS, FAIL_STATUS})

PREREGISTRATION = REPO_ROOT / (
    "docs/lewm_go2_dinov2_physical_readout_calibration_v1_"
    "preregistration_2026-08-03.md"
)
PREREGISTRATION_SHA256 = (
    "ff6e42042792ffc66c51ac9e6fd31d9da194cb22c5526edfd1ce3cfe22db55ee"
)
PREREGISTRATION_BYTE_COUNT = 10_285
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / ".generated/dev/go2_dinov2_physical_readout_calibration_v1/attempt_v1"
)
TRAIN_CACHE = REPO_ROOT / (
    ".generated/dev/go2_matched_branch_successor_screen_v1/attempt_v1/"
    "features/dinov2.pt"
)
TRAIN_CACHE_SHA256 = "164f1fef8c859976c93f7fc978e938c6c8f7f9963cf92bb154f51b23d158b34b"
TRAIN_CACHE_BYTE_COUNT = 302_107_682
TRAIN_CACHE_RECEIPT = TRAIN_CACHE.with_suffix(".json")
TRAIN_CACHE_RECEIPT_SHA256 = (
    "e94ec5d188811c44d4cc870e76d1888aa6f30ee6d423557ee9f3e2918a700994"
)
TRAIN_CACHE_RECEIPT_BYTE_COUNT = 1_770
DINO_REPOSITORY_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINO_CHECKPOINT_SHA256 = (
    "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
)
DINO_CHECKPOINT_BYTE_COUNT = 88_283_115
ROLE_STATE_COUNT = 128
ROLE_ARTIFACT_COUNT = 1_536
TOKEN_SHAPE = (256, 384)
EVAL_BATCH_SIZE = 32

SOURCE_PATHS = {
    "action_regret_evaluator": REPO_ROOT
    / "scripts/evaluate_go2_world_model_counterfactual_action_regret_v1.py",
    "counterfactual_benchmark_contract": REPO_ROOT
    / "lewm/benchmarks/go2_world_model_counterfactual_pilot_v1.py",
    "calibration_module": Path(calibration.__file__).resolve(),
    "calibration_test": REPO_ROOT
    / "lewm/tests/test_go2_dinov2_physical_readout_calibration_v1.py",
    "predecessor_model_module": REPO_ROOT
    / "lewm/models/go2_matched_branch_successor_screen_v1.py",
    "pilot_consumer": Path(pilot_consumer.__file__).resolve(),
    "posthoc_loader": Path(screen_data.posthoc.__file__).resolve(),
    "predecessor_runner": Path(predecessor_runner.__file__).resolve(),
    "runner": Path(__file__).resolve(),
    "runner_test": REPO_ROOT
    / "lewm/tests/test_run_go2_dinov2_physical_readout_calibration_v1.py",
    "screen_data": Path(screen_data.__file__).resolve(),
}


class CalibrationRunnerError(RuntimeError):
    """Raised when the frozen calibration or custody contract changes."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _reject_protected(path: Path, *, label: str) -> None:
    for part in path.parts:
        lowered = part.lower()
        if (
            lowered == "sealed_test.json"
            or lowered == "sealed"
            or lowered.startswith("sealed_")
            or lowered in {"heldout", "held_out", "held-out"}
            or lowered.startswith(("heldout_", "held_out_", "held-out-"))
        ):
            raise CalibrationRunnerError(f"{label} names protected material")


def _safe_path(path: Path, *, label: str, must_exist: bool = True) -> Path:
    selected = Path(os.path.abspath(os.fspath(path)))
    _reject_protected(selected, label=label)
    cursor = Path(selected.anchor)
    for part in selected.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise CalibrationRunnerError(f"{label} traverses a symlink")
        if not cursor.exists():
            if must_exist:
                raise CalibrationRunnerError(f"{label} is absent")
            break
    if must_exist and not selected.exists():
        raise CalibrationRunnerError(f"{label} is absent")
    return selected


def file_binding_v1(path: Path) -> dict[str, Any]:
    selected = _safe_path(path, label="bound file")
    if not selected.is_file():
        raise CalibrationRunnerError("bound path is not a file")
    digest = hashlib.sha256()
    size = 0
    with selected.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {"path": str(selected), "sha256": digest.hexdigest(), "byte_count": size}


def _require_binding(value: object, *, label: str) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"path", "sha256", "byte_count"}
        or not isinstance(value.get("path"), str)
        or not isinstance(value.get("sha256"), str)
        or len(str(value["sha256"])) != 64
        or type(value.get("byte_count")) is not int
        or int(value["byte_count"]) <= 0
    ):
        raise CalibrationRunnerError(f"{label} binding is malformed")
    actual = file_binding_v1(Path(str(value["path"])))
    if actual != dict(value):
        raise CalibrationRunnerError(f"{label} binding changed")
    return actual


def _read_bound_json(
    path: Path, *, expected_sha256: str, expected_byte_count: int, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = file_binding_v1(path)
    if (
        binding["sha256"] != expected_sha256
        or binding["byte_count"] != expected_byte_count
    ):
        raise CalibrationRunnerError(f"{label} caller binding changed")
    try:
        document = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CalibrationRunnerError(f"{label} is not valid JSON") from error
    if not isinstance(document, Mapping):
        raise CalibrationRunnerError(f"{label} is not a JSON object")
    return dict(document), binding


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    raw = _canonical_bytes(value) + b"\n"
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o644,
    )
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _save_torch_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o644,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def calibration_config_v1() -> dict[str, Any]:
    return {
        "eval_artifact_count": ROLE_ARTIFACT_COUNT,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "eval_state_count": ROLE_STATE_COUNT,
        "feature_shape": [ROLE_ARTIFACT_COUNT, *TOKEN_SHAPE],
        "replay_count": 1,
        "train_artifact_count": ROLE_ARTIFACT_COUNT,
        "train_state_count": ROLE_STATE_COUNT,
    }


def _fixed_input_bindings_v1() -> dict[str, dict[str, Any]]:
    return {
        "posthoc_manifest": {
            "path": str(Path(screen_data.POSTHOC_MANIFEST_BINDING["path"]).resolve()),
            "sha256": str(screen_data.POSTHOC_MANIFEST_BINDING["file_sha256"]),
            "byte_count": int(screen_data.POSTHOC_MANIFEST_BINDING["byte_count"]),
        },
        "posthoc_terminal": {
            "path": str(Path(screen_data.POSTHOC_TERMINAL_BINDING["path"]).resolve()),
            "sha256": str(screen_data.POSTHOC_TERMINAL_BINDING["file_sha256"]),
            "byte_count": int(screen_data.POSTHOC_TERMINAL_BINDING["byte_count"]),
        },
        "posthoc_terminal_review": {
            "path": str(
                Path(screen_data.POSTHOC_TERMINAL_REVIEW_BINDING["path"]).resolve()
            ),
            "sha256": str(
                screen_data.POSTHOC_TERMINAL_REVIEW_BINDING["file_sha256"]
            ),
            "byte_count": int(
                screen_data.POSTHOC_TERMINAL_REVIEW_BINDING["byte_count"]
            ),
        },
        "train_cache": {
            "path": str(TRAIN_CACHE.resolve()),
            "sha256": TRAIN_CACHE_SHA256,
            "byte_count": TRAIN_CACHE_BYTE_COUNT,
        },
        "train_cache_receipt": {
            "path": str(TRAIN_CACHE_RECEIPT.resolve()),
            "sha256": TRAIN_CACHE_RECEIPT_SHA256,
            "byte_count": TRAIN_CACHE_RECEIPT_BYTE_COUNT,
        },
    }


def _validate_source_review_v1(
    review: Mapping[str, Any],
    *,
    preregistration_binding: Mapping[str, Any],
    source_bindings: Mapping[str, Any],
) -> None:
    if (
        set(review) != SOURCE_REVIEW_FIELDS
        or review.get("schema") != SOURCE_REVIEW_SCHEMA
        or review.get("status") != SOURCE_REVIEW_STATUS
        or review.get("preregistration_binding") != preregistration_binding
        or review.get("source_bindings") != source_bindings
        or review.get("findings") != []
        or review.get("protected_material_opened") is not False
        or not isinstance(review.get("checks"), Mapping)
        or set(review["checks"]) != SOURCE_REVIEW_CHECKS
        or any(value is not True for value in review["checks"].values())
    ):
        raise CalibrationRunnerError("independent source review did not pass exactly")


def _read_authority(
    path: Path, *, expected_sha256: str, expected_byte_count: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    authority, authority_binding = _read_bound_json(
        path,
        expected_sha256=expected_sha256,
        expected_byte_count=expected_byte_count,
        label="execution authority",
    )
    required = {
        "schema",
        "status",
        "citable_as_scientific_evidence",
        "authorizes_collection",
        "authorizes_eval_rgb_access",
        "authorizes_model_training",
        "authorizes_retry_or_resume",
        "authorizes_train_rgb_access",
        "preregistration_binding",
        "source_review_binding",
        "source_bindings",
        "input_bindings",
        "encoder_source",
        "output_root",
        "environment",
        "config",
        "git_commit",
    }
    if (
        set(authority) != required
        or authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("status") != AUTHORITY_STATUS
        or authority.get("citable_as_scientific_evidence") is not False
        or authority.get("authorizes_collection") is not False
        or authority.get("authorizes_eval_rgb_access") is not True
        or authority.get("authorizes_model_training") is not False
        or authority.get("authorizes_retry_or_resume") is not False
        or authority.get("authorizes_train_rgb_access") is not False
        or authority.get("output_root") != str(DEFAULT_OUTPUT_ROOT.resolve())
        or authority.get("config") != calibration_config_v1()
    ):
        raise CalibrationRunnerError("execution authority contract changed")
    prereg = _require_binding(authority["preregistration_binding"], label="preregistration")
    if prereg != {
        "path": str(PREREGISTRATION.resolve()),
        "sha256": PREREGISTRATION_SHA256,
        "byte_count": PREREGISTRATION_BYTE_COUNT,
    }:
        raise CalibrationRunnerError("authority does not bind the frozen preregistration")
    inputs = authority.get("input_bindings")
    if not isinstance(inputs, Mapping) or dict(inputs) != _fixed_input_bindings_v1():
        raise CalibrationRunnerError("authority input closure changed")
    for label, binding in inputs.items():
        _require_binding(binding, label=f"input {label}")
    sources = authority.get("source_bindings")
    if not isinstance(sources, Mapping) or set(sources) != set(SOURCE_PATHS):
        raise CalibrationRunnerError("authority source closure changed")
    for label, expected_path in SOURCE_PATHS.items():
        actual = _require_binding(sources[label], label=f"source {label}")
        if actual["path"] != str(expected_path.resolve()):
            raise CalibrationRunnerError(f"source {label} path changed")
    review_binding = _require_binding(authority["source_review_binding"], label="source review")
    if review_binding["path"] != str(SOURCE_REVIEW.resolve()):
        raise CalibrationRunnerError("source review path changed")
    review, _ = _read_bound_json(
        Path(review_binding["path"]),
        expected_sha256=review_binding["sha256"],
        expected_byte_count=review_binding["byte_count"],
        label="source review",
    )
    _validate_source_review_v1(
        review,
        preregistration_binding=prereg,
        source_bindings=sources,
    )
    encoder = authority.get("encoder_source")
    if (
        not isinstance(encoder, Mapping)
        or set(encoder) != {"repo_path", "repo_commit", "checkpoint_binding"}
        or encoder.get("repo_commit") != DINO_REPOSITORY_COMMIT
    ):
        raise CalibrationRunnerError("DINO source contract changed")
    checkpoint = _require_binding(encoder["checkpoint_binding"], label="DINO checkpoint")
    if (
        checkpoint["sha256"] != DINO_CHECKPOINT_SHA256
        or checkpoint["byte_count"] != DINO_CHECKPOINT_BYTE_COUNT
    ):
        raise CalibrationRunnerError("DINO checkpoint identity changed")
    repo = _safe_path(Path(str(encoder["repo_path"])), label="DINO repository")
    observed_commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed_commit != DINO_REPOSITORY_COMMIT:
        raise CalibrationRunnerError("DINO repository commit changed")
    repository_status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if repository_status:
        raise CalibrationRunnerError("DINO repository is not clean at the bound commit")
    environment = authority.get("environment")
    if (
        not isinstance(environment, Mapping)
        or set(environment) != {"python", "torch", "hip", "numpy", "pillow"}
        or environment.get("python") != str(Path(sys.executable).resolve())
        or environment.get("torch") != torch.__version__
        or environment.get("hip") != torch.version.hip
        or environment.get("numpy") != np.__version__
        or environment.get("pillow") != PILLOW_VERSION
    ):
        raise CalibrationRunnerError("execution environment changed")
    commit = authority.get("git_commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or subprocess.run(
            ["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", commit, "HEAD"],
            check=False,
        ).returncode
        != 0
    ):
        raise CalibrationRunnerError("reviewed source commit is not an execution ancestor")
    return authority, authority_binding


def _artifact_ids(plan: object, *, role: str) -> tuple[str, ...]:
    if getattr(plan, "role", role) != role:
        raise CalibrationRunnerError(f"{role} feature plan role changed")
    values = tuple(getattr(plan, "artifact_ids", ()))
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise CalibrationRunnerError(f"{role} artifact order is malformed")
    if len(values) != len(set(values)):
        raise CalibrationRunnerError(f"{role} artifact order contains a duplicate")
    return values


def _split_plans(plans: object) -> tuple[object, object]:
    if isinstance(plans, Mapping):
        return plans["train"], plans["eval"]
    if hasattr(plans, "train") and hasattr(plans, "eval"):
        return plans.train, plans.eval
    values = tuple(plans)  # type: ignore[arg-type]
    if len(values) != 2:
        raise CalibrationRunnerError("calibration feature plans changed")
    return values[0], values[1]


def _load_train_cache_v1(bundle: object, train_plan: object) -> tuple[torch.Tensor, dict[str, Any]]:
    receipt, _ = _read_bound_json(
        TRAIN_CACHE_RECEIPT,
        expected_sha256=TRAIN_CACHE_RECEIPT_SHA256,
        expected_byte_count=TRAIN_CACHE_RECEIPT_BYTE_COUNT,
        label="frozen DINO train-cache receipt",
    )
    index = predecessor_runner.build_screen_index_v1(bundle)
    if tuple(index.artifact_ids) != _artifact_ids(train_plan, role="train"):
        raise CalibrationRunnerError("frozen train-cache artifact order changed")
    features = predecessor_runner._load_feature_cache(  # noqa: SLF001
        receipt, expected_encoder="dinov2", index=index
    )
    if (
        features.shape != (ROLE_ARTIFACT_COUNT, *TOKEN_SHAPE)
        or features.dtype != torch.float16
        or not bool(torch.isfinite(features).all())
    ):
        raise CalibrationRunnerError("frozen DINO train cache tensor changed")
    return features, receipt


def _load_dino_encoder_v1(
    authority: Mapping[str, Any], device: torch.device
) -> torch.nn.Module:
    adapted = {"encoder_sources": {"dinov2": authority["encoder_source"]}}
    return predecessor_runner._load_dino_encoder(adapted, device)  # noqa: SLF001


@torch.no_grad()
def extract_eval_feature_cache_v1(
    bundle: object,
    eval_plan: object,
    *,
    authority: Mapping[str, Any],
    device: torch.device,
    output_path: Path,
    expected_artifact_count: int = ROLE_ARTIFACT_COUNT,
    batch_size: int = EVAL_BATCH_SIZE,
) -> dict[str, Any]:
    artifact_ids = _artifact_ids(eval_plan, role="eval")
    if len(artifact_ids) != expected_artifact_count:
        raise CalibrationRunnerError("evaluation artifact count changed")
    if batch_size < 1:
        raise CalibrationRunnerError("evaluation feature batch size is invalid")
    encoder = _load_dino_encoder_v1(authority, device).eval().requires_grad_(False)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    batches: list[torch.Tensor] = []
    opened: list[str] = []
    started = time.perf_counter()
    for start in range(0, len(artifact_ids), batch_size):
        selected = artifact_ids[start : start + batch_size]
        prepared = []
        for artifact_id in selected:
            raw = read_bound_rgb_bytes_v1(bundle, artifact_id)
            opened.append(artifact_id)
            prepared.append(screen_data.preprocess_dinov2_png_bytes_v1(raw))
        inputs = torch.stack(prepared).to(device)
        raw_tokens = encoder.forward_features(inputs)["x_norm_patchtokens"]
        normalized = screen_data.normalize_dense_token_grid_v1(raw_tokens)
        batches.append(normalized.to(dtype=torch.float16, device="cpu"))
    elapsed = time.perf_counter() - started
    if tuple(opened) != artifact_ids or len(opened) != len(set(opened)):
        raise CalibrationRunnerError("evaluation RGB access was not exactly once in order")
    features = torch.cat(batches, dim=0)
    if (
        features.shape != (expected_artifact_count, *TOKEN_SHAPE)
        or features.dtype != torch.float16
        or not bool(torch.isfinite(features).all())
    ):
        raise CalibrationRunnerError("evaluation DINO cache tensor changed")
    order_sha256 = hashlib.sha256(_canonical_bytes(list(artifact_ids))).hexdigest()
    payload = {
        "schema": EVAL_CACHE_SCHEMA,
        "encoder": "dinov2",
        "role": "eval",
        "artifact_ids": artifact_ids,
        "artifact_order_sha256": order_sha256,
        "features": features,
    }
    _save_torch_exclusive(output_path, payload)
    receipt = {
        "schema": EVAL_CACHE_RECEIPT_SCHEMA,
        "encoder": "dinov2",
        "role": "eval",
        "binding": file_binding_v1(output_path),
        "source_bundle_manifest": dict(getattr(bundle, "manifest_binding")),
        "encoder_source": dict(authority["encoder_source"]),
        "preprocessing": predecessor_runner.feature_preprocessing_contract_v1("dinov2"),
        "artifact_order_sha256": order_sha256,
        "artifact_count": len(artifact_ids),
        "eval_artifact_open_count": len(opened),
        "train_artifact_open_count": 0,
        "decoded_pixel_verification_count": len(opened),
        "shape": list(features.shape),
        "storage_dtype": "float16",
        "elapsed_seconds": elapsed,
        "frames_per_second": len(artifact_ids) / elapsed,
        "peak_gpu_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    }
    _write_json_exclusive(output_path.with_suffix(".json"), receipt)
    del encoder, batches, features
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return receipt


def _load_eval_feature_cache_v1(
    receipt: Mapping[str, Any], eval_plan: object, *, expected_artifact_count: int = ROLE_ARTIFACT_COUNT
) -> torch.Tensor:
    artifact_ids = _artifact_ids(eval_plan, role="eval")
    order_sha256 = hashlib.sha256(_canonical_bytes(list(artifact_ids))).hexdigest()
    if (
        receipt.get("schema") != EVAL_CACHE_RECEIPT_SCHEMA
        or receipt.get("encoder") != "dinov2"
        or receipt.get("role") != "eval"
        or receipt.get("artifact_order_sha256") != order_sha256
        or receipt.get("artifact_count") != expected_artifact_count
        or receipt.get("eval_artifact_open_count") != expected_artifact_count
        or receipt.get("train_artifact_open_count") != 0
        or receipt.get("decoded_pixel_verification_count") != expected_artifact_count
        or receipt.get("shape") != [expected_artifact_count, *TOKEN_SHAPE]
        or receipt.get("storage_dtype") != "float16"
    ):
        raise CalibrationRunnerError("evaluation cache receipt changed")
    binding = _require_binding(receipt.get("binding"), label="evaluation feature cache")
    payload = torch.load(binding["path"], map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != EVAL_CACHE_SCHEMA
        or payload.get("encoder") != "dinov2"
        or payload.get("role") != "eval"
        or tuple(payload.get("artifact_ids", ())) != artifact_ids
        or payload.get("artifact_order_sha256") != order_sha256
        or not isinstance(payload.get("features"), torch.Tensor)
    ):
        raise CalibrationRunnerError("evaluation cache payload changed")
    features = payload["features"]
    if (
        features.shape != (expected_artifact_count, *TOKEN_SHAPE)
        or features.dtype != torch.float16
        or not bool(torch.isfinite(features).all())
    ):
        raise CalibrationRunnerError("evaluation cache tensor changed")
    return features


def _evaluate_v1(
    train_groups: Sequence[object],
    eval_groups: Sequence[object],
    train_features: torch.Tensor,
    eval_features: torch.Tensor,
) -> dict[str, Any]:
    result = calibration.evaluate_calibration_v1(
        train_groups,
        eval_groups,
        train_features.numpy(),
        eval_features.numpy(),
    )
    if not isinstance(result, Mapping):
        raise CalibrationRunnerError("pure calibration evaluator returned a non-object")
    _canonical_bytes(result)
    return dict(result)


def _verdict_status(
    verdict: Mapping[str, Any],
    *,
    evaluation: Mapping[str, Any],
    deterministic_replay_passed: bool,
) -> str:
    gates = verdict.get("gates")
    expected_gates = {
        "1_infrastructure_and_custody": {"passed": True},
        **dict(evaluation["gates"]),
        "7_deterministic_replay": {"passed": deterministic_replay_passed},
    }
    expected_passed = all(bool(gate["passed"]) for gate in expected_gates.values())
    expected_status = (
        FAIL_STATUS
        if not deterministic_replay_passed
        else PASS_STATUS if expected_passed else STOP_STATUS
    )
    if (
        set(verdict) != {"gates", "passed", "terminal_status"}
        or not isinstance(gates, Mapping)
        or dict(gates) != expected_gates
        or type(verdict.get("passed")) is not bool
        or verdict.get("passed") is not expected_passed
        or verdict.get("terminal_status") != expected_status
        or expected_status not in TERMINAL_STATUSES
    ):
        raise CalibrationRunnerError("pure calibration verdict contract changed")
    return expected_status


def _execution_bindings_unchanged(
    authority: Mapping[str, Any], *, authority_binding: Mapping[str, Any]
) -> None:
    closure: list[tuple[str, Mapping[str, Any]]] = [
        ("execution authority", authority_binding),
        ("preregistration", authority["preregistration_binding"]),
        ("source review", authority["source_review_binding"]),
        ("DINO checkpoint", authority["encoder_source"]["checkpoint_binding"]),
    ]
    closure.extend(
        (f"source {label}", expected)
        for label, expected in authority["source_bindings"].items()
    )
    closure.extend(
        (f"input {label}", expected)
        for label, expected in authority["input_bindings"].items()
    )
    for label, expected in closure:
        if file_binding_v1(Path(str(expected["path"]))) != expected:
            raise CalibrationRunnerError(f"{label} changed during execution")

    repo = _safe_path(
        Path(str(authority["encoder_source"]["repo_path"])),
        label="DINO repository",
    )
    observed_commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    repository_status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if observed_commit != DINO_REPOSITORY_COMMIT or repository_status:
        raise CalibrationRunnerError("DINO repository changed during execution")


def execute_v1(
    authority: Mapping[str, Any], *, authority_binding: Mapping[str, Any]
) -> dict[str, Any]:
    output_root = _safe_path(
        Path(str(authority["output_root"])), label="calibration output", must_exist=False
    )
    _safe_path(
        output_root.parent, label="calibration output parent", must_exist=False
    )
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json_exclusive(
        output_root / "reservation.json",
        {
            "schema": RESERVATION_SCHEMA,
            "authority_binding": dict(authority_binding),
            "attempt_root": str(output_root),
            "owner_pid": os.getpid(),
            "consumes_attempt": True,
        },
    )
    bundle = screen_data.load_bound_posthoc_bundle_v1()
    if getattr(bundle, "access_audit", {}).get("rgb_leaf_open_count") != 0:
        raise CalibrationRunnerError("strict bundle loader opened an RGB leaf")
    groups = getattr(bundle, "groups_by_role", {})
    if set(groups) != {"train", "eval"}:
        raise CalibrationRunnerError("bounded bundle roles changed")
    train_groups = tuple(groups["train"])
    eval_groups = tuple(groups["eval"])
    if len(train_groups) != ROLE_STATE_COUNT or len(eval_groups) != ROLE_STATE_COUNT:
        raise CalibrationRunnerError("bounded bundle state counts changed")
    train_plan, eval_plan = _split_plans(
        calibration.build_calibration_feature_plans_v1(train_groups, eval_groups)
    )
    train_ids = _artifact_ids(train_plan, role="train")
    eval_ids = _artifact_ids(eval_plan, role="eval")
    if (
        len(train_ids) != ROLE_ARTIFACT_COUNT
        or len(eval_ids) != ROLE_ARTIFACT_COUNT
        or set(train_ids) & set(eval_ids)
    ):
        raise CalibrationRunnerError("role feature plans changed or overlap")
    train_features, train_receipt = _load_train_cache_v1(bundle, train_plan)
    if not torch.cuda.is_available():
        raise CalibrationRunnerError("the authorized DINO extraction requires CUDA/ROCm")
    device = torch.device("cuda")
    eval_cache_path = output_root / "dinov2_eval.pt"
    eval_receipt = extract_eval_feature_cache_v1(
        bundle,
        eval_plan,
        authority=authority,
        device=device,
        output_path=eval_cache_path,
    )
    eval_features = _load_eval_feature_cache_v1(eval_receipt, eval_plan)
    evaluation = _evaluate_v1(
        train_groups, eval_groups, train_features, eval_features
    )
    first_identity = calibration.calibration_replay_identity_v1(evaluation)
    del train_features, eval_features

    # The replay is deliberately cache-only: there is no RGB-reader call on this path.
    replay_train_features, replay_train_receipt = _load_train_cache_v1(bundle, train_plan)
    replay_eval_features = _load_eval_feature_cache_v1(eval_receipt, eval_plan)
    replay_evaluation = _evaluate_v1(
        train_groups,
        eval_groups,
        replay_train_features,
        replay_eval_features,
    )
    replay_identity = calibration.calibration_replay_identity_v1(replay_evaluation)
    deterministic_replay_passed = (
        _canonical_bytes(evaluation) == _canonical_bytes(replay_evaluation)
        and first_identity == replay_identity
        and replay_train_receipt == train_receipt
        and _require_binding(eval_receipt["binding"], label="replay evaluation cache")
        == eval_receipt["binding"]
    )
    _execution_bindings_unchanged(authority, authority_binding=authority_binding)
    verdict = calibration.calibration_verdict_v1(
        evaluation,
        infrastructure_checks_passed=True,
        deterministic_replay_passed=deterministic_replay_passed,
    )
    if not isinstance(verdict, Mapping):
        raise CalibrationRunnerError("pure calibration verdict returned a non-object")
    status = _verdict_status(
        verdict,
        evaluation=evaluation,
        deterministic_replay_passed=deterministic_replay_passed,
    )
    report = {
        "schema": SCHEMA,
        "status": status,
        "citable_as_scientific_evidence": False,
        "development_calibration_only": True,
        "authorizes_model_training": False,
        "authority_binding": dict(authority_binding),
        "preregistration_binding": authority["preregistration_binding"],
        "source_bundle_manifest": dict(bundle.manifest_binding),
        "train_cache_receipt": train_receipt,
        "eval_cache_receipt": eval_receipt,
        "rgb_access": {
            "train_artifact_open_count": 0,
            "eval_artifact_open_count": ROLE_ARTIFACT_COUNT,
            "replay_artifact_open_count": 0,
        },
        "evaluation": evaluation,
        "replay": {
            "first_identity": first_identity,
            "replay_identity": replay_identity,
            "exactly_reproduced": deterministic_replay_passed,
            "cache_only": True,
        },
        "verdict": dict(verdict),
    }
    _write_json_exclusive(output_root / "result.json", report)
    terminal = {
        "schema": TERMINAL_SCHEMA,
        "status": status,
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "authorizes_model_training": False,
        "result_binding": file_binding_v1(output_root / "result.json"),
        "deterministic_replay_passed": deterministic_replay_passed,
        "failure": None,
    }
    _write_json_exclusive(output_root / "terminal.json", terminal)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--expected-authority-sha256", required=True)
    parser.add_argument("--expected-authority-byte-count", type=int, required=True)
    args = parser.parse_args(argv)
    authority, binding = _read_authority(
        args.authority,
        expected_sha256=args.expected_authority_sha256,
        expected_byte_count=args.expected_authority_byte_count,
    )
    output_root = Path(str(authority["output_root"]))
    existed = output_root.exists()
    try:
        report = execute_v1(authority, authority_binding=binding)
    except Exception as error:
        if not existed and output_root.is_dir() and not (output_root / "terminal.json").exists():
            _write_json_exclusive(
                output_root / "terminal.json",
                {
                    "schema": TERMINAL_SCHEMA,
                    "status": FAIL_STATUS,
                    "citable_as_scientific_evidence": False,
                    "authorizes_retry_or_resume": False,
                    "authorizes_model_training": False,
                    "result_binding": None,
                    "deterministic_replay_passed": False,
                    "failure": {
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    },
                },
            )
        raise
    print(json.dumps({"status": report["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORITY_SCHEMA",
    "AUTHORITY_STATUS",
    "CalibrationRunnerError",
    "DEFAULT_OUTPUT_ROOT",
    "EVAL_CACHE_RECEIPT_SCHEMA",
    "EVAL_CACHE_SCHEMA",
    "FAIL_STATUS",
    "PASS_STATUS",
    "SOURCE_REVIEW_SCHEMA",
    "SOURCE_REVIEW_STATUS",
    "STOP_STATUS",
    "calibration_config_v1",
    "execute_v1",
    "extract_eval_feature_cache_v1",
    "file_binding_v1",
]
