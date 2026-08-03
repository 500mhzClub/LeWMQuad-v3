#!/usr/bin/env python3
"""Run the preregistered train-only matched-branch engineering screen."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks import go2_matched_branch_successor_screen_v1 as screen_data  # noqa: E402
from lewm.datasets.go2_world_model_counterfactual_pilot_v1 import (  # noqa: E402
    read_bound_rgb_bytes_v1,
)
from lewm.datasets import go2_world_model_counterfactual_pilot_v1 as pilot_consumer  # noqa: E402
from lewm.models.go2_matched_branch_successor_screen_v1 import (  # noqa: E402
    CompactRSSMPredictorV1,
    DenseActionConditionedPredictorV1,
    DeterministicStateSpacePredictorV1,
)


SCHEMA = "lewm_go2_matched_branch_successor_screen_result_v1"
TERMINAL_SCHEMA = "lewm_go2_matched_branch_successor_screen_terminal_v1"
AUTHORITY_SCHEMA = "lewm_go2_matched_branch_successor_screen_execution_authority_v1"
AUTHORITY_STATUS = "AUTHORIZED_ONE_EXACT_TRAIN_ONLY_ENGINEERING_SCREEN"
PREREGISTRATION = (
    REPO_ROOT
    / "docs/lewm_go2_matched_branch_successor_screen_v1_preregistration_2026-08-03.md"
)
PREREGISTRATION_SHA256 = "7a31e3ceb7b4ffafccb4b3a763b09553d658125dba75e72a4b405da602b79e8a"
PREREGISTRATION_BYTE_COUNT = 9_274
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / ".generated/dev/go2_matched_branch_successor_screen_v1/attempt_v1"
)
ACTION_COUNT = 9
STATE_COUNT = 128
ARTIFACT_COUNT = 1_536
TRAINING_SEED = 2_026_080_301
TRACE_UPDATES = (0, 100, 200, 400, 800)
ARM_NAMES = (
    "dense_vjepa2_1",
    "dense_dinov2",
    "state_space_vjepa2_1",
    "rssm_vjepa2_1",
)
SOURCE_LABELS = {
    "data_module",
    "data_test",
    "model_module",
    "model_test",
    "pilot_consumer",
    "posthoc_loader",
    "runner",
    "runner_test",
}
SOURCE_PATHS = {
    "data_module": REPO_ROOT / "lewm/benchmarks/go2_matched_branch_successor_screen_v1.py",
    "data_test": REPO_ROOT / "lewm/tests/test_go2_matched_branch_successor_screen_data_v1.py",
    "model_module": REPO_ROOT / "lewm/models/go2_matched_branch_successor_screen_v1.py",
    "model_test": REPO_ROOT / "lewm/tests/test_go2_matched_branch_successor_screen_v1.py",
    "pilot_consumer": Path(pilot_consumer.__file__).resolve(),
    "posthoc_loader": Path(screen_data.posthoc.__file__).resolve(),
    "runner": Path(__file__).resolve(),
    "runner_test": REPO_ROOT / "lewm/tests/test_run_go2_matched_branch_successor_screen_v1.py",
}


class ScreenError(RuntimeError):
    """Raised when the frozen engineering-screen contract changes."""


@dataclass(frozen=True)
class ScreenIndexV1:
    state_ids: tuple[str, ...]
    family_ids: tuple[str, ...]
    scene_ids: tuple[str, ...]
    artifact_ids: tuple[str, ...]
    context_indices: torch.Tensor
    target_indices: torch.Tensor
    history_actions: torch.Tensor
    index_sha256: str


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


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


def _reject_protected(path: Path, *, label: str) -> None:
    for part in path.parts:
        lowered = part.lower()
        if (
            lowered == "sealed_test.json"
            or lowered == "sealed"
            or lowered.startswith("sealed_")
            or lowered in {"heldout", "held_out", "held-out"}
            or lowered.startswith("heldout_")
            or lowered.startswith("held_out_")
            or lowered.startswith("held-out-")
        ):
            raise ScreenError(f"{label} names protected material")


def _safe_path(path: Path, *, label: str, must_exist: bool = True) -> Path:
    selected = Path(os.path.abspath(os.fspath(path)))
    _reject_protected(selected, label=label)
    cursor = Path(selected.anchor)
    for part in selected.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ScreenError(f"{label} traverses a symlink")
        if not cursor.exists():
            if must_exist:
                raise ScreenError(f"{label} is absent")
            break
    if must_exist and not selected.is_file() and not selected.is_dir():
        raise ScreenError(f"{label} is absent")
    return selected


def file_binding_v1(path: Path) -> dict[str, Any]:
    selected = _safe_path(path, label="bound file")
    if not selected.is_file():
        raise ScreenError(f"bound path is not a file: {selected}")
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
        raise ScreenError(f"{label} binding is malformed")
    actual = file_binding_v1(Path(str(value["path"])))
    if actual != dict(value):
        raise ScreenError(f"{label} binding changed")
    return actual


def screen_config_v1() -> dict[str, Any]:
    return {
        "action_count": ACTION_COUNT,
        "arms": list(ARM_NAMES),
        "batch_states": 8,
        "cross_entropy_coefficient": 0.25,
        "cross_entropy_temperature": 0.1,
        "feature_batches": {"dinov2": 32, "vjepa2_1": 4},
        "gradient_clip_norm": 1.0,
        "hidden_dim": 128,
        "learning_rate": 3.0e-4,
        "maximum_projected_gpu_hours": 24.0,
        "retrieval_threshold": 0.50,
        "rssm_kl_coefficient": 0.01,
        "rssm_kl_reduction": "batchmean_after_latent_sum",
        "rssm_posterior_coefficient": 0.5,
        "rssm_stochastic_dim": 32,
        "seed": TRAINING_SEED,
        "trace_updates": list(TRACE_UPDATES),
        "updates": 800,
        "weight_decay": 1.0e-4,
        "maximum_error_to_persistence_ratio": 0.80,
    }


def feature_preprocessing_contract_v1(encoder_name: str) -> dict[str, Any]:
    """Return the exact image/token transform bound into a cache receipt."""

    common = {
        "decoded_input": {
            "format": "PNG",
            "mode": "RGB",
            "size": [224, 224],
        },
        "normalization": {
            "mean": list(screen_data.IMAGENET_MEAN),
            "std": list(screen_data.IMAGENET_STD),
        },
        "token_conversion": {
            "output_grid": [16, 16],
            "compute_dtype": "float32",
            "per_token_l2_normalization": True,
        },
    }
    if encoder_name == "dinov2":
        return {
            **common,
            "image_geometry": {"operation": "identity", "output_size": [224, 224]},
            "encoder_output_grid": [16, 16],
            "spatial_conversion": "identity",
        }
    if encoder_name == "vjepa2_1":
        return {
            **common,
            "image_geometry": {
                "resize": [438, 438],
                "resize_kernel": "PIL_BILINEAR",
                "center_crop": [384, 384],
                "image_mode_frames": 1,
            },
            "encoder_output_grid": [24, 24],
            "spatial_conversion": "torch_area_24x24_to_16x16",
        }
    raise ScreenError("unknown frozen feature encoder")


def _read_authority(
    path: Path, *, expected_sha256: str, expected_byte_count: int
) -> dict[str, Any]:
    actual = file_binding_v1(path)
    if (
        actual["sha256"] != expected_sha256
        or actual["byte_count"] != expected_byte_count
    ):
        raise ScreenError("execution authority caller binding changed")
    try:
        document = json.loads(Path(path).read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ScreenError("execution authority is not valid JSON") from error
    required = {
        "schema",
        "status",
        "citable_as_scientific_evidence",
        "authorizes_collection",
        "authorizes_eval_rgb_access",
        "preregistration_binding",
        "source_review_binding",
        "source_bindings",
        "encoder_sources",
        "pilot_root",
        "output_root",
        "environment",
        "config",
        "git_commit",
    }
    if (
        not isinstance(document, Mapping)
        or set(document) != required
        or document.get("schema") != AUTHORITY_SCHEMA
        or document.get("status") != AUTHORITY_STATUS
        or document.get("citable_as_scientific_evidence") is not False
        or document.get("authorizes_collection") is not False
        or document.get("authorizes_eval_rgb_access") is not False
        or document.get("config") != screen_config_v1()
        or document.get("pilot_root") != str(screen_data.POSTHOC_ROOT.resolve())
        or document.get("output_root") != str(DEFAULT_OUTPUT_ROOT.resolve())
    ):
        raise ScreenError("execution authority contract changed")
    frozen_source_commit = document.get("git_commit")
    if (
        not isinstance(frozen_source_commit, str)
        or len(frozen_source_commit) != 40
        or any(character not in "0123456789abcdef" for character in frozen_source_commit)
        or subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "merge-base",
                "--is-ancestor",
                frozen_source_commit,
                "HEAD",
            ],
            check=False,
        ).returncode
        != 0
    ):
        raise ScreenError("frozen source commit is not an ancestor of execution HEAD")
    prereg = _require_binding(
        document["preregistration_binding"], label="preregistration"
    )
    if prereg != {
        "path": str(PREREGISTRATION.resolve()),
        "sha256": PREREGISTRATION_SHA256,
        "byte_count": PREREGISTRATION_BYTE_COUNT,
    }:
        raise ScreenError("authority does not bind the frozen preregistration")
    source_review_binding = _require_binding(
        document["source_review_binding"], label="source review"
    )
    sources = document.get("source_bindings")
    if not isinstance(sources, Mapping) or set(sources) != SOURCE_LABELS:
        raise ScreenError("source closure labels changed")
    for label, binding in sources.items():
        actual_source = _require_binding(binding, label=f"source {label}")
        if actual_source["path"] != str(SOURCE_PATHS[label].resolve()):
            raise ScreenError(f"source {label} path changed")
    try:
        source_review = json.loads(Path(source_review_binding["path"]).read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ScreenError("source review is not valid JSON") from error
    if (
        not isinstance(source_review, Mapping)
        or source_review.get("schema")
        != "lewm_go2_matched_branch_successor_screen_source_review_v1"
        or source_review.get("status") != "PASS_INDEPENDENT_SOURCE_REVIEW"
        or source_review.get("preregistration_binding") != prereg
        or source_review.get("source_bindings") != sources
        or source_review.get("findings") != []
        or source_review.get("protected_material_opened") is not False
        or not isinstance(source_review.get("checks"), Mapping)
        or not source_review["checks"]
        or any(value is not True for value in source_review["checks"].values())
    ):
        raise ScreenError("independent source review did not pass exactly")
    encoders = document.get("encoder_sources")
    if not isinstance(encoders, Mapping) or set(encoders) != {"dinov2", "vjepa2_1"}:
        raise ScreenError("encoder source bindings changed")
    expected_commits = {
        "dinov2": "7764ea0f912e53c92e82eb78a2a1631e92725fc8",
        "vjepa2_1": "204698b45b3712590f06245fbfba32d3be539812",
    }
    for name, expected_commit in expected_commits.items():
        item = encoders[name]
        if (
            not isinstance(item, Mapping)
            or set(item) != {"repo_path", "repo_commit", "checkpoint_binding"}
            or item.get("repo_commit") != expected_commit
        ):
            raise ScreenError(f"{name} source contract changed")
        repo = _safe_path(Path(str(item["repo_path"])), label=f"{name} repository")
        if not repo.is_dir():
            raise ScreenError(f"{name} repository is not a directory")
        observed_commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if observed_commit != expected_commit:
            raise ScreenError(f"{name} repository commit changed")
        repository_status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if repository_status:
            raise ScreenError(f"{name} repository working tree is not clean")
        _require_binding(item["checkpoint_binding"], label=f"{name} checkpoint")
    environment = document.get("environment")
    if (
        not isinstance(environment, Mapping)
        or set(environment) != {"python", "torch", "hip"}
        or environment.get("python") != str(Path(sys.executable).resolve())
        or environment.get("torch") != torch.__version__
        or environment.get("hip") != torch.version.hip
    ):
        raise ScreenError("execution environment changed")
    return dict(document)


def build_screen_index_v1(bundle: Any) -> ScreenIndexV1:
    plan = screen_data.collect_train_feature_plan_v1(bundle)
    if len(plan.states) != STATE_COUNT or len(plan.artifact_ids) != ARTIFACT_COUNT:
        raise ScreenError("train artifact count changed")
    contexts = [list(state.context_artifact_indices) for state in plan.states]
    targets = [list(state.target_artifact_indices) for state in plan.states]
    histories = [
        list(state.candidate_inputs[0].history_action_ids) for state in plan.states
    ]
    index_document = {
        "state_ids": [state.state_id for state in plan.states],
        "families": [state.family for state in plan.states],
        "scenes": [state.scene_id for state in plan.states],
        "artifact_ids": list(plan.artifact_ids),
        "contexts": contexts,
        "targets": targets,
        "history_actions": histories,
    }
    return ScreenIndexV1(
        state_ids=tuple(index_document["state_ids"]),
        family_ids=tuple(index_document["families"]),
        scene_ids=tuple(index_document["scenes"]),
        artifact_ids=plan.artifact_ids,
        context_indices=torch.tensor(contexts, dtype=torch.long),
        target_indices=torch.tensor(targets, dtype=torch.long),
        history_actions=torch.tensor(histories, dtype=torch.long),
        index_sha256=hashlib.sha256(_canonical_bytes(index_document)).hexdigest(),
    )


def cosine_distance_matrix_v1(
    predictions: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    if (
        predictions.ndim != 4
        or targets.shape != predictions.shape
        or predictions.shape[0] < 1
        or predictions.shape[1] != ACTION_COUNT
        or predictions.shape[2] != 256
        or predictions.shape[3] < 1
        or predictions.dtype != torch.float32
        or targets.dtype != torch.float32
        or predictions.device != targets.device
    ):
        raise ScreenError("prediction and target panels must share (B,9,256,D)")
    if not bool(torch.isfinite(predictions).all()) or not bool(torch.isfinite(targets).all()):
        raise ScreenError("prediction or target panel is nonfinite")
    predicted = F.normalize(predictions, dim=-1)
    target = F.normalize(targets.detach(), dim=-1)
    similarity = torch.einsum("band,btnd->bat", predicted, target) / predictions.shape[2]
    distance = 1.0 - similarity
    if not bool(torch.isfinite(distance).all()):
        raise ScreenError("cosine distance matrix became nonfinite")
    return distance


def common_objective_v1(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    temperature: float = 0.1,
    cross_entropy_coefficient: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    matrix = cosine_distance_matrix_v1(predictions, targets)
    action_ids = torch.arange(ACTION_COUNT, device=matrix.device)
    matched = matrix[:, action_ids, action_ids].mean()
    contrastive = F.cross_entropy(
        (-matrix / temperature).reshape(-1, ACTION_COUNT),
        action_ids.unsqueeze(0).expand(matrix.shape[0], -1).reshape(-1),
    )
    total = matched + cross_entropy_coefficient * contrastive
    return total, {"matched": matched, "contrastive": contrastive}


def screen_metrics_from_panels_v1(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    last_context: torch.Tensor,
) -> dict[str, float]:
    matrix = cosine_distance_matrix_v1(predictions, targets)
    if last_context.shape != (
        predictions.shape[0],
        predictions.shape[2],
        predictions.shape[3],
    ):
        raise ScreenError("last context shape changed")
    action_ids = torch.arange(ACTION_COUNT, device=matrix.device)
    matched_rows = matrix[:, action_ids, action_ids]
    deranged_rows = matrix[:, (action_ids + 1) % ACTION_COUNT, action_ids]
    persistence = 1.0 - torch.einsum(
        "bnd,band->ba",
        F.normalize(last_context, dim=-1),
        F.normalize(targets, dim=-1),
    ) / targets.shape[2]
    matched = matched_rows.mean()
    persistence_mean = persistence.mean()
    retrieval = (matrix.argmin(dim=-1) == action_ids).float().mean()
    deranged = deranged_rows.mean()
    ratio = matched / persistence_mean.clamp_min(1.0e-12)
    values = {
        "matched_cosine_error": float(matched),
        "persistence_cosine_error": float(persistence_mean),
        "error_to_persistence_ratio": float(ratio),
        "branch_retrieval_accuracy": float(retrieval),
        "cyclic_deranged_cosine_error": float(deranged),
        "action_intervention_margin": float(deranged - matched),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ScreenError("screen metric became nonfinite")
    return values


def _load_vjepa_encoder(authority: Mapping[str, Any], device: torch.device) -> torch.nn.Module:
    item = authority["encoder_sources"]["vjepa2_1"]
    with screen_data.scoped_timm_drop_path_shim_v1():
        encoder, predictor = torch.hub.load(
            str(item["repo_path"]),
            "vjepa2_1_vit_base_384",
            source="local",
            pretrained=False,
        )
    del predictor
    payload = torch.load(
        item["checkpoint_binding"]["path"], map_location="cpu", weights_only=True
    )
    state = {
        key.replace("module.", "").replace("backbone.", ""): value
        for key, value in payload["ema_encoder"].items()
    }
    encoder.load_state_dict(state, strict=True)
    del payload, state
    return encoder.to(device).eval().requires_grad_(False)


def _load_dino_encoder(authority: Mapping[str, Any], device: torch.device) -> torch.nn.Module:
    item = authority["encoder_sources"]["dinov2"]
    encoder = torch.hub.load(
        str(item["repo_path"]), "dinov2_vits14", source="local", pretrained=False
    )
    state = torch.load(
        item["checkpoint_binding"]["path"], map_location="cpu", weights_only=True
    )
    encoder.load_state_dict(state, strict=True)
    del state
    return encoder.to(device).eval().requires_grad_(False)


@torch.no_grad()
def extract_feature_cache_v1(
    bundle: Any,
    index: ScreenIndexV1,
    *,
    encoder_name: str,
    authority: Mapping[str, Any],
    device: torch.device,
    output_path: Path,
) -> dict[str, Any]:
    if encoder_name == "vjepa2_1":
        encoder = _load_vjepa_encoder(authority, device)
        batch_size = int(authority["config"]["feature_batches"]["vjepa2_1"])
        preprocess = screen_data.preprocess_vjepa2_1_png_bytes_v1
        feature_dim = 768
    elif encoder_name == "dinov2":
        encoder = _load_dino_encoder(authority, device)
        batch_size = int(authority["config"]["feature_batches"]["dinov2"])
        preprocess = screen_data.preprocess_dinov2_png_bytes_v1
        feature_dim = 384
    else:
        raise ScreenError("unknown frozen feature encoder")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    batches: list[torch.Tensor] = []
    open_count = 0
    for start in range(0, len(index.artifact_ids), batch_size):
        ids = index.artifact_ids[start : start + batch_size]
        prepared = []
        for artifact_id in ids:
            raw = read_bound_rgb_bytes_v1(bundle, artifact_id)
            open_count += 1
            prepared.append(preprocess(raw))
        inputs = torch.stack(prepared).to(device)
        if encoder_name == "vjepa2_1":
            raw_tokens = encoder(inputs)
        else:
            raw_tokens = encoder.forward_features(inputs)["x_norm_patchtokens"]
        tokens = screen_data.normalize_dense_token_grid_v1(raw_tokens)
        batches.append(tokens.to(dtype=torch.float16, device="cpu"))
    elapsed = time.perf_counter() - started
    features = torch.cat(batches, dim=0)
    if features.shape != (ARTIFACT_COUNT, 256, feature_dim):
        raise ScreenError("frozen feature cache shape changed")
    payload = {
        "schema": "lewm_go2_matched_branch_successor_feature_cache_v1",
        "encoder": encoder_name,
        "index_sha256": index.index_sha256,
        "artifact_ids": index.artifact_ids,
        "features": features,
    }
    torch.save(payload, output_path)
    binding = file_binding_v1(output_path)
    report = {
        "schema": "lewm_go2_matched_branch_successor_feature_cache_receipt_v1",
        "encoder": encoder_name,
        "binding": binding,
        "source_bundle_manifest": dict(bundle.manifest_binding),
        "encoder_source": authority["encoder_sources"][encoder_name],
        "preprocessing": feature_preprocessing_contract_v1(encoder_name),
        "index_sha256": index.index_sha256,
        "artifact_order_sha256": hashlib.sha256(
            _canonical_bytes(list(index.artifact_ids))
        ).hexdigest(),
        "artifact_count": len(index.artifact_ids),
        "eval_artifact_open_count": 0,
        "train_artifact_open_count": open_count,
        "shape": list(features.shape),
        "storage_dtype": "float16",
        "elapsed_seconds": elapsed,
        "frames_per_second": len(index.artifact_ids) / elapsed,
        "peak_gpu_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    }
    _write_json_exclusive(output_path.with_suffix(".json"), report)
    del encoder, features, batches
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def _load_feature_cache(
    receipt: Mapping[str, Any], *, expected_encoder: str, index: ScreenIndexV1
) -> torch.Tensor:
    if (
        receipt.get("encoder") != expected_encoder
        or receipt.get("index_sha256") != index.index_sha256
        or receipt.get("artifact_order_sha256")
        != hashlib.sha256(_canonical_bytes(list(index.artifact_ids))).hexdigest()
        or receipt.get("preprocessing")
        != feature_preprocessing_contract_v1(expected_encoder)
        or receipt.get("artifact_count") != ARTIFACT_COUNT
        or receipt.get("eval_artifact_open_count") != 0
    ):
        raise ScreenError("feature cache receipt changed")
    binding = _require_binding(receipt["binding"], label=f"{expected_encoder} cache")
    payload = torch.load(binding["path"], map_location="cpu", weights_only=True)
    if (
        payload.get("encoder") != expected_encoder
        or payload.get("index_sha256") != index.index_sha256
        or tuple(payload.get("artifact_ids", ())) != index.artifact_ids
        or not isinstance(payload.get("features"), torch.Tensor)
    ):
        raise ScreenError("feature cache payload changed")
    return payload["features"]


def _build_model(arm: str, feature_dim: int, config: Mapping[str, Any]) -> torch.nn.Module:
    common = {
        "feature_dim": feature_dim,
        "hidden_dim": int(config["hidden_dim"]),
        "action_count": ACTION_COUNT,
    }
    if arm.startswith("dense_"):
        return DenseActionConditionedPredictorV1(**common)
    if arm == "state_space_vjepa2_1":
        return DeterministicStateSpacePredictorV1(**common)
    if arm == "rssm_vjepa2_1":
        return CompactRSSMPredictorV1(
            **common, stochastic_dim=int(config["rssm_stochastic_dim"])
        )
    raise ScreenError("unknown model arm")


def _batch_panels(
    features: torch.Tensor,
    index: ScreenIndexV1,
    state_ids: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    contexts = features[index.context_indices[state_ids]].to(device=device, dtype=torch.float32)
    targets = features[index.target_indices[state_ids]].to(device=device, dtype=torch.float32)
    histories = index.history_actions[state_ids].to(device)
    count = contexts.shape[0]
    branch_context = contexts[:, None].expand(-1, ACTION_COUNT, -1, -1, -1)
    branch_context = branch_context.reshape(
        count * ACTION_COUNT, 3, 256, contexts.shape[-1]
    ).contiguous()
    branch_history = histories[:, None].expand(-1, ACTION_COUNT, -1)
    branch_history = branch_history.reshape(count * ACTION_COUNT, 2).contiguous()
    candidates = torch.arange(ACTION_COUNT, device=device).repeat(count)
    return branch_context, branch_history, candidates, targets


@torch.no_grad()
def evaluate_arm_v1(
    model: torch.nn.Module,
    features: torch.Tensor,
    index: ScreenIndexV1,
    *,
    device: torch.device,
    batch_states: int = 4,
) -> dict[str, float]:
    model.eval()
    totals = {
        "matched": 0.0,
        "persistence": 0.0,
        "deranged": 0.0,
        "retrieval": 0.0,
        "rows": 0.0,
    }
    action_ids = torch.arange(ACTION_COUNT, device=device)
    for start in range(0, STATE_COUNT, batch_states):
        selected = torch.arange(start, min(STATE_COUNT, start + batch_states))
        context, history, candidates, targets = _batch_panels(
            features, index, selected, device
        )
        predictions = model(context, history, candidates).reshape(
            selected.numel(), ACTION_COUNT, 256, features.shape[-1]
        )
        matrix = cosine_distance_matrix_v1(predictions, targets)
        matched = matrix[:, action_ids, action_ids]
        deranged = matrix[:, (action_ids + 1) % ACTION_COUNT, action_ids]
        last_context = context.reshape(
            selected.numel(), ACTION_COUNT, 3, 256, features.shape[-1]
        )[:, 0, -1]
        persistence = 1.0 - torch.einsum(
            "bnd,band->ba",
            F.normalize(last_context, dim=-1),
            F.normalize(targets, dim=-1),
        ) / 256
        totals["matched"] += float(matched.sum())
        totals["persistence"] += float(persistence.sum())
        totals["deranged"] += float(deranged.sum())
        totals["retrieval"] += float(
            (matrix.argmin(dim=-1) == action_ids).sum()
        )
        totals["rows"] += float(selected.numel() * ACTION_COUNT)
    matched_mean = totals["matched"] / totals["rows"]
    persistence_mean = totals["persistence"] / totals["rows"]
    result = {
        "matched_cosine_error": matched_mean,
        "persistence_cosine_error": persistence_mean,
        "error_to_persistence_ratio": matched_mean / max(persistence_mean, 1.0e-12),
        "branch_retrieval_accuracy": totals["retrieval"] / totals["rows"],
        "cyclic_deranged_cosine_error": totals["deranged"] / totals["rows"],
        "action_intervention_margin": (
            totals["deranged"] / totals["rows"] - matched_mean
        ),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ScreenError("arm evaluation became nonfinite")
    return result


def train_arm_v1(
    arm: str,
    features: torch.Tensor,
    index: ScreenIndexV1,
    *,
    config: Mapping[str, Any],
    device: torch.device,
    output_path: Path,
    updates: int | None = None,
    trace_updates: Sequence[int] | None = None,
) -> dict[str, Any]:
    update_count = int(config["updates"] if updates is None else updates)
    traces_at = tuple(config["trace_updates"] if trace_updates is None else trace_updates)
    if 0 not in traces_at or update_count not in traces_at:
        raise ScreenError("training traces must include update zero and terminal")
    torch.manual_seed(int(config["seed"]))
    random.seed(int(config["seed"]))
    np.random.seed(int(config["seed"]) % (2**32))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(config["seed"]))
        torch.cuda.reset_peak_memory_stats(device)
    model = _build_model(arm, int(features.shape[-1]), config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    generator = torch.Generator(device="cpu").manual_seed(int(config["seed"]))
    ordering = torch.randperm(STATE_COUNT, generator=generator)
    cursor = 0
    traces: list[dict[str, Any]] = []
    started = time.perf_counter()
    if 0 in traces_at:
        traces.append({"update": 0, **evaluate_arm_v1(model, features, index, device=device)})
    nonfinite_count = 0
    last_components: dict[str, float] = {}
    for update in range(1, update_count + 1):
        batch_size = int(config["batch_states"])
        if cursor + batch_size > STATE_COUNT:
            ordering = torch.randperm(STATE_COUNT, generator=generator)
            cursor = 0
        selected = ordering[cursor : cursor + batch_size]
        cursor += batch_size
        context, history, candidates, targets = _batch_panels(
            features, index, selected, device
        )
        flat_targets = targets.reshape(
            batch_size * ACTION_COUNT, 256, features.shape[-1]
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prior_predictions = model(context, history, candidates).reshape(
            batch_size, ACTION_COUNT, 256, features.shape[-1]
        )
        loss, components = common_objective_v1(
            prior_predictions,
            targets,
            temperature=float(config["cross_entropy_temperature"]),
            cross_entropy_coefficient=float(config["cross_entropy_coefficient"]),
        )
        if isinstance(model, CompactRSSMPredictorV1):
            posterior = model.training_output(
                context,
                history,
                candidates,
                flat_targets,
                sample_posterior=True,
            )
            posterior_prediction = F.normalize(posterior.prediction, dim=-1)
            posterior_target = F.normalize(flat_targets.detach(), dim=-1)
            posterior_reconstruction = (
                1.0 - (posterior_prediction * posterior_target).sum(dim=-1)
            ).mean()
            kl = model.kl_divergence(posterior, reduction="batchmean")
            loss = (
                loss
                + float(config["rssm_posterior_coefficient"])
                * posterior_reconstruction
                + float(config["rssm_kl_coefficient"]) * kl
            )
            components = {
                **components,
                "posterior_reconstruction": posterior_reconstruction,
                "kl": kl,
            }
        if not bool(torch.isfinite(loss)):
            nonfinite_count += 1
            raise ScreenError(f"{arm} loss became nonfinite")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["gradient_clip_norm"])
        )
        if not bool(torch.isfinite(grad_norm)):
            nonfinite_count += 1
            raise ScreenError(f"{arm} gradient norm became nonfinite")
        optimizer.step()
        last_components = {
            "total": float(loss.detach()),
            "gradient_norm_before_clip": float(grad_norm.detach()),
            **{name: float(value.detach()) for name, value in components.items()},
        }
        if update in traces_at:
            traces.append(
                {
                    "update": update,
                    "objective": last_components,
                    **evaluate_arm_v1(model, features, index, device=device),
                }
            )
    elapsed = time.perf_counter() - started
    final = traces[-1]
    deterministic_repeat = evaluate_arm_v1(model, features, index, device=device)
    deterministic_repeat_passed = all(
        deterministic_repeat[key] == final[key] for key in deterministic_repeat
    )
    if not deterministic_repeat_passed:
        raise ScreenError(f"{arm} repeated evaluation changed")
    eligible = (
        final["error_to_persistence_ratio"]
        <= float(config["maximum_error_to_persistence_ratio"])
        and final["branch_retrieval_accuracy"]
        >= float(config["retrieval_threshold"])
        and final["action_intervention_margin"] > 0.0
        and nonfinite_count == 0
    )
    checkpoint = {
        "schema": "lewm_go2_matched_branch_successor_screen_checkpoint_v1",
        "arm": arm,
        "seed": int(config["seed"]),
        "update": update_count,
        "feature_dim": int(features.shape[-1]),
        "config": dict(config),
        "model_state_dict": {
            name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
        },
    }
    torch.save(checkpoint, output_path)
    checkpoint_binding = file_binding_v1(output_path)
    result = {
        "arm": arm,
        "seed": int(config["seed"]),
        "updates": update_count,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_seconds": elapsed,
        "updates_per_second": update_count / elapsed if update_count else 0.0,
        "peak_gpu_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "nonfinite_count": nonfinite_count,
        "deterministic_repeat_passed": deterministic_repeat_passed,
        "traces": traces,
        "final_metrics": {key: value for key, value in final.items() if key != "objective"},
        "engineering_eligible": eligible,
        "checkpoint_binding": checkpoint_binding,
    }
    del model, optimizer, checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _source_bindings_unchanged(authority: Mapping[str, Any]) -> None:
    for label, expected in authority["source_bindings"].items():
        if file_binding_v1(Path(expected["path"])) != expected:
            raise ScreenError(f"source {label} changed during execution")


def execute_v1(authority: Mapping[str, Any]) -> dict[str, Any]:
    output_root = Path(str(authority["output_root"]))
    _safe_path(output_root.parent, label="screen output parent", must_exist=False)
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "features").mkdir()
    (output_root / "checkpoints").mkdir()
    bundle = screen_data.load_bound_posthoc_bundle_v1()
    index = build_screen_index_v1(bundle)
    if bundle.access_audit.get("rgb_leaf_open_count") != 0:
        raise ScreenError("bundle loader opened RGB before feature extraction")
    if not torch.cuda.is_available():
        raise ScreenError("the preregistered GPU runtime screen requires a CUDA/ROCm device")
    device = torch.device("cuda")
    feature_receipts = {}
    for encoder_name in ("vjepa2_1", "dinov2"):
        feature_receipts[encoder_name] = extract_feature_cache_v1(
            bundle,
            index,
            encoder_name=encoder_name,
            authority=authority,
            device=device,
            output_path=output_root / "features" / f"{encoder_name}.pt",
        )
    arm_results = {}
    total_training_seconds = 0.0
    for arm in ARM_NAMES:
        encoder_name = "dinov2" if arm == "dense_dinov2" else "vjepa2_1"
        features = _load_feature_cache(
            feature_receipts[encoder_name],
            expected_encoder=encoder_name,
            index=index,
        )
        arm_results[arm] = train_arm_v1(
            arm,
            features,
            index,
            config=authority["config"],
            device=device,
            output_path=output_root / "checkpoints" / f"{arm}.pt",
        )
        total_training_seconds += arm_results[arm]["training_seconds"]
        del features
    fresh_frame_count = 12_288
    projected_feature_seconds = sum(
        fresh_frame_count / feature_receipts[name]["frames_per_second"]
        for name in ("vjepa2_1", "dinov2")
    )
    projected_training_seconds = total_training_seconds * 6.0 * 3.0
    projected_gpu_hours = (
        projected_feature_seconds + projected_training_seconds
    ) / 3600.0
    eligible = {name: bool(result["engineering_eligible"]) for name, result in arm_results.items()}
    collection_justified = (
        (eligible["dense_vjepa2_1"] or eligible["dense_dinov2"])
        and eligible["state_space_vjepa2_1"]
        and eligible["rssm_vjepa2_1"]
        and all(receipt["eval_artifact_open_count"] == 0 for receipt in feature_receipts.values())
        and projected_gpu_hours
        <= float(authority["config"]["maximum_projected_gpu_hours"])
    )
    _source_bindings_unchanged(authority)
    report = {
        "schema": SCHEMA,
        "status": "COMPLETE_ENGINEERING_SCREEN",
        "citable_as_scientific_evidence": False,
        "fresh_scene_generalization_measured": False,
        "navigation_usefulness_established": False,
        "authorizes_collection": False,
        "authority": dict(authority),
        "source_bundle_manifest": dict(bundle.manifest_binding),
        "screen_index": {
            "states": len(index.state_ids),
            "scenes": len(set(index.scene_ids)),
            "families": len(set(index.family_ids)),
            "artifacts": len(index.artifact_ids),
            "index_sha256": index.index_sha256,
            "eval_rgb_leaf_open_count": 0,
        },
        "device": {
            "type": str(device),
            "name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "hip": torch.version.hip,
        },
        "feature_caches": feature_receipts,
        "arms": arm_results,
        "eligibility": eligible,
        "runtime_projection": {
            "fresh_frames": fresh_frame_count,
            "feature_seconds": projected_feature_seconds,
            "training_seconds": projected_training_seconds,
            "gpu_hours": projected_gpu_hours,
            "maximum_gpu_hours": authority["config"]["maximum_projected_gpu_hours"],
        },
        "collection_justified": collection_justified,
        "next_route": (
            "FREEZE_AND_COLLECT_FOUR_256_STATE_MATCHED_BRANCH_SHARDS"
            if collection_justified
            else "STOP_BEFORE_FRESH_MATCHED_BRANCH_COLLECTION"
        ),
    }
    _write_json_exclusive(output_root / "result.json", report)
    result_binding = file_binding_v1(output_root / "result.json")
    terminal = {
        "schema": TERMINAL_SCHEMA,
        "status": (
            "COMPLETE_COLLECTION_JUSTIFIED"
            if collection_justified
            else "COMPLETE_COLLECTION_NOT_JUSTIFIED"
        ),
        "citable_as_scientific_evidence": False,
        "authorizes_collection": False,
        "result_binding": result_binding,
        "collection_justified": collection_justified,
        "next_route": report["next_route"],
    }
    _write_json_exclusive(output_root / "terminal.json", terminal)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--expected-authority-sha256", required=True)
    parser.add_argument("--expected-authority-byte-count", type=int, required=True)
    args = parser.parse_args(argv)
    authority = _read_authority(
        args.authority,
        expected_sha256=args.expected_authority_sha256,
        expected_byte_count=args.expected_authority_byte_count,
    )
    output_existed_before_execution = Path(str(authority["output_root"])).exists()
    try:
        report = execute_v1(authority)
    except Exception as error:
        output_root = Path(str(authority["output_root"]))
        if (
            not output_existed_before_execution
            and output_root.is_dir()
            and not (output_root / "terminal.json").exists()
        ):
            _write_json_exclusive(
                output_root / "terminal.json",
                {
                    "schema": TERMINAL_SCHEMA,
                    "status": "CONSUMED_TERMINAL_INFRASTRUCTURE_FAILURE",
                    "citable_as_scientific_evidence": False,
                    "authorizes_collection": False,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
            )
        raise
    print(json.dumps({
        "status": report["status"],
        "collection_justified": report["collection_justified"],
        "next_route": report["next_route"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
