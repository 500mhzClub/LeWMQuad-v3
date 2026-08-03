#!/usr/bin/env python3
"""Run the preregistered no-RGB dual residual token-adapter JEPA screen."""
from __future__ import annotations

import argparse
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

from lewm.models.go2_dual_residual_token_adapter_jepa_v1 import (  # noqa: E402
    JointResidualTokenAdapterJEPAV1,
)
from scripts import run_go2_matched_branch_successor_screen_v1 as predecessor  # noqa: E402


RESULT_SCHEMA = "lewm_go2_dual_residual_token_adapter_jepa_result_v1"
TERMINAL_SCHEMA = "lewm_go2_dual_residual_token_adapter_jepa_terminal_v1"
CHECKPOINT_SCHEMA = "lewm_go2_dual_residual_token_adapter_jepa_checkpoint_v1"
AUTHORITY_SCHEMA = "lewm_go2_dual_residual_token_adapter_jepa_execution_authority_v1"
AUTHORITY_STATUS = "AUTHORIZED_ONE_EXACT_NO_RGB_DUAL_ADAPTER_SCREEN"
PREREGISTRATION = (
    REPO_ROOT
    / "docs/lewm_go2_dual_residual_token_adapter_jepa_v1_preregistration_2026-08-03.md"
)
PREREGISTRATION_SHA256 = (
    "c3c9e76c450f842a64f56577b038117e579b0d3c0d0e2e8892bd4590886f6c09"
)
PREREGISTRATION_BYTE_COUNT = 16_756
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / ".generated/dev/go2_dual_residual_token_adapter_jepa_v1/attempt_v1"
)
PREDECESSOR_ROOT = (
    REPO_ROOT / ".generated/dev/go2_matched_branch_successor_screen_v1/attempt_v1"
)
HORIZON_ROOT = (
    REPO_ROOT / ".generated/dev/go2_dense_vjepa2_1_horizon_diagnostic_v1/attempt_v1"
)
INDEX_SHA256 = "b740e3efead2f79fd17337a9fa10784c91989e52e837d023b2cc02a2c19d018d"
ARTIFACT_ORDER_SHA256 = (
    "68f19fc1121d4e5d6cd85c8ac50dab8538c8507ebb9a0e70258228147be2ec73"
)
ARMS = ("residual_joint_vjepa2_1", "residual_joint_dinov2")
ENCODER_BY_ARM = {
    "residual_joint_vjepa2_1": "vjepa2_1",
    "residual_joint_dinov2": "dinov2",
}
FEATURE_DIM_BY_ENCODER = {"vjepa2_1": 768, "dinov2": 384}
FROZEN_CONTROLS = {
    "residual_joint_vjepa2_1": {
        "error_to_persistence_ratio": 0.9164363539053353,
        "branch_retrieval_accuracy": 0.2803819444444444,
        "action_intervention_margin": 0.028666746492187187,
    },
    "residual_joint_dinov2": {
        "error_to_persistence_ratio": 0.9666892593579309,
        "branch_retrieval_accuracy": 0.2048611111111111,
        "action_intervention_margin": 0.04603223171499038,
    },
}
MIDPOINT_GATES = {
    "residual_joint_vjepa2_1": {
        "maximum_error_to_persistence_ratio": 0.8582181769526677,
        "minimum_branch_retrieval_accuracy": 0.3901909722222222,
    },
    "residual_joint_dinov2": {
        "maximum_error_to_persistence_ratio": 0.8833446296789655,
        "minimum_branch_retrieval_accuracy": 0.3524305555555556,
    },
}

PREDECESSOR_BINDINGS = {
    "screen_result": {
        "path": str((PREDECESSOR_ROOT / "result.json").resolve()),
        "sha256": "a6caf2ed1950781815925ccc76b4dbbf40b0f331f4b14a5e60befc88f3aae605",
        "byte_count": 21_377,
    },
    "screen_terminal": {
        "path": str((PREDECESSOR_ROOT / "terminal.json").resolve()),
        "sha256": "bf3bf322c2f3db877be405ebf5ca1daf9dd1a5ffd667b769d44cccab22ede758",
        "byte_count": 510,
    },
    "screen_terminal_review": {
        "path": str(
            (
                REPO_ROOT
                / "docs/lewm_go2_matched_branch_successor_screen_v1_terminal_review_2026-08-03.json"
            ).resolve()
        ),
        "sha256": "c450baab14b50caed3469fa88f5812c92c02b04676059568e8dae3dc2e5bad83",
        "byte_count": 4_991,
    },
    "horizon_result": {
        "path": str((HORIZON_ROOT / "result.json").resolve()),
        "sha256": "ade09fc81d950bb4bf4d26f9620da9c46bacea945e39cef261020e6eb2121cad",
        "byte_count": 8_598,
    },
    "horizon_terminal": {
        "path": str((HORIZON_ROOT / "terminal.json").resolve()),
        "sha256": "39a5b3498be7b4fa84abd6ec566b01969b348c44f4403d834c585a0ef4e7c68a",
        "byte_count": 631,
    },
    "horizon_terminal_review": {
        "path": str(
            (
                REPO_ROOT
                / "docs/lewm_go2_dense_vjepa2_1_horizon_diagnostic_v1_terminal_review_2026-08-03.json"
            ).resolve()
        ),
        "sha256": "0751a9c2d6d2d7d7131ca32f3d3fdc5b4aa9740632fd9a84a51f5e87b82ee1cd",
        "byte_count": 4_913,
    },
    "vjepa2_1_feature_receipt": {
        "path": str((PREDECESSOR_ROOT / "features/vjepa2_1.json").resolve()),
        "sha256": "5d4f8a82d10a33c21b41f1543d6f56b3a230a38f67b02d3f8e7330a8d30180f5",
        "byte_count": 1_822,
    },
    "vjepa2_1_feature_cache": {
        "path": str((PREDECESSOR_ROOT / "features/vjepa2_1.pt").resolve()),
        "sha256": "3549855ea857906dfe3a4b55fc817633b5114b2457f8facaa4fa87f9eddd798b",
        "byte_count": 604_097_648,
    },
    "dinov2_feature_receipt": {
        "path": str((PREDECESSOR_ROOT / "features/dinov2.json").resolve()),
        "sha256": "e94ec5d188811c44d4cc870e76d1888aa6f30ee6d423557ee9f3e2918a700994",
        "byte_count": 1_770,
    },
    "dinov2_feature_cache": {
        "path": str((PREDECESSOR_ROOT / "features/dinov2.pt").resolve()),
        "sha256": "164f1fef8c859976c93f7fc978e938c6c8f7f9963cf92bb154f51b23d158b34b",
        "byte_count": 302_107_682,
    },
}

SOURCE_PATHS = {
    **{
        f"predecessor_{label}": path
        for label, path in predecessor.SOURCE_PATHS.items()
    },
    "horizon_runner": (
        REPO_ROOT / "scripts/run_go2_dense_vjepa2_1_horizon_diagnostic_v1.py"
    ),
    "horizon_runner_test": (
        REPO_ROOT / "lewm/tests/test_run_go2_dense_vjepa2_1_horizon_diagnostic_v1.py"
    ),
    "adapter_model": (
        REPO_ROOT / "lewm/models/go2_dual_residual_token_adapter_jepa_v1.py"
    ),
    "adapter_model_test": (
        REPO_ROOT / "lewm/tests/test_go2_dual_residual_token_adapter_jepa_v1.py"
    ),
    "dual_runner": Path(__file__).resolve(),
    "dual_runner_test": (
        REPO_ROOT / "lewm/tests/test_run_go2_dual_residual_token_adapter_jepa_v1.py"
    ),
}
SOURCE_LABELS = set(SOURCE_PATHS)


class DualScreenError(RuntimeError):
    """Raised when the frozen dual-screen contract changes."""


class ArmNonfiniteError(DualScreenError):
    """Raised for a registered arm-local numerical failure."""


class EffectiveRankDegenerateError(DualScreenError):
    """Raised when a finite pooled representation has no positive spectrum."""


def dual_config_v1() -> dict[str, Any]:
    """Return the exact preregistered treatment and decision configuration."""

    return {
        "action_count": 9,
        "arms": list(ARMS),
        "adapter_blocks": 2,
        "adapter_bottleneck": 64,
        "adapter_residual_scale": 0.125,
        "batch_states": 8,
        "cross_entropy_coefficient": 0.25,
        "cross_entropy_temperature": 0.1,
        "ema_momentum": 0.996,
        "ema_target_coefficient": 0.5,
        "frozen_target_coefficient": 0.5,
        "identity_coefficient": 0.10,
        "relative_variance_coefficient": 0.10,
        "relative_variance_floor": 0.90,
        "hidden_dim": 128,
        "learning_rate": 3.0e-4,
        "weight_decay": 1.0e-4,
        "adamw_betas": [0.9, 0.999],
        "adamw_eps": 1.0e-8,
        "adamw_amsgrad": False,
        "adamw_maximize": False,
        "adamw_foreach": False,
        "adamw_capturable": False,
        "adamw_differentiable": False,
        "adamw_fused": False,
        "gradient_clip_norm": 1.0,
        "seed": 2_026_080_301,
        "trace_updates": [0, 400, 800, 1_600],
        "minimum_updates": 800,
        "maximum_updates": 1_600,
        "maximum_error_to_persistence_ratio": 0.80,
        "minimum_branch_retrieval_accuracy": 0.50,
        "minimum_intervention_margin": 0.0,
        "minimum_retention_cosine": 0.965,
        "minimum_effective_rank_ratio": 0.90,
        "unit_norm_absolute_tolerance": 1.0e-5,
        "midpoint_gates": MIDPOINT_GATES,
        "evaluation_batch_states": 4,
        "retention_batch_artifacts": {"vjepa2_1": 16, "dinov2": 32},
        "cache_compute_dtype": "float32",
        "autocast_enabled": False,
    }


def _json_from_bound_file(binding: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    predecessor._require_binding(binding, label=label)  # noqa: SLF001
    try:
        value = json.loads(Path(str(binding["path"])).read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DualScreenError(f"{label} is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise DualScreenError(f"{label} is not a JSON object")
    return dict(value)


def _validate_source_review(
    binding: Mapping[str, Any],
    *,
    preregistration_binding: Mapping[str, Any],
    source_bindings: Mapping[str, Any],
) -> None:
    review = _json_from_bound_file(binding, label="source review")
    if (
        review.get("schema")
        != "lewm_go2_dual_residual_token_adapter_jepa_source_review_v1"
        or review.get("status") != "PASS_INDEPENDENT_SOURCE_REVIEW"
        or review.get("preregistration_binding") != preregistration_binding
        or review.get("source_bindings") != source_bindings
        or review.get("protected_material_opened") is not False
        or review.get("findings") != []
        or not isinstance(review.get("checks"), Mapping)
        or not review["checks"]
        or any(value is not True for value in review["checks"].values())
    ):
        raise DualScreenError("independent source review did not pass exactly")


def _read_authority(
    path: Path, *, expected_sha256: str, expected_byte_count: int
) -> dict[str, Any]:
    actual = predecessor.file_binding_v1(path)
    if actual["sha256"] != expected_sha256 or actual["byte_count"] != expected_byte_count:
        raise DualScreenError("execution authority caller binding changed")
    document = _json_from_bound_file(actual, label="execution authority")
    required = {
        "schema",
        "status",
        "citable_as_scientific_evidence",
        "authorizes_collection",
        "authorizes_rgb_access",
        "authorizes_feature_extraction",
        "authorizes_evaluation",
        "authorizes_retry",
        "preregistration_binding",
        "source_review_binding",
        "source_bindings",
        "predecessor_bindings",
        "output_root",
        "environment",
        "config",
        "git_commit",
    }
    if (
        set(document) != required
        or document.get("schema") != AUTHORITY_SCHEMA
        or document.get("status") != AUTHORITY_STATUS
        or document.get("citable_as_scientific_evidence") is not False
        or document.get("authorizes_collection") is not False
        or document.get("authorizes_rgb_access") is not False
        or document.get("authorizes_feature_extraction") is not False
        or document.get("authorizes_evaluation") is not False
        or document.get("authorizes_retry") is not False
        or document.get("output_root") != str(DEFAULT_OUTPUT_ROOT.resolve())
        or document.get("config") != dual_config_v1()
        or document.get("predecessor_bindings") != PREDECESSOR_BINDINGS
    ):
        raise DualScreenError("execution authority contract changed")
    commit = document.get("git_commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
        or subprocess.run(
            ["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", commit, "HEAD"],
            check=False,
        ).returncode
        != 0
    ):
        raise DualScreenError("frozen source commit is not an ancestor of execution HEAD")
    preregistration = predecessor._require_binding(  # noqa: SLF001
        document["preregistration_binding"], label="preregistration"
    )
    if preregistration != {
        "path": str(PREREGISTRATION.resolve()),
        "sha256": PREREGISTRATION_SHA256,
        "byte_count": PREREGISTRATION_BYTE_COUNT,
    }:
        raise DualScreenError("authority does not bind the frozen preregistration")
    sources = document.get("source_bindings")
    if not isinstance(sources, Mapping) or set(sources) != SOURCE_LABELS:
        raise DualScreenError("source closure labels changed")
    for label, expected in sources.items():
        observed = predecessor._require_binding(expected, label=f"source {label}")  # noqa: SLF001
        if observed["path"] != str(SOURCE_PATHS[label].resolve()):
            raise DualScreenError(f"source {label} path changed")
    _validate_source_review(
        document["source_review_binding"],
        preregistration_binding=preregistration,
        source_bindings=sources,
    )
    for label, binding in document["predecessor_bindings"].items():
        predecessor._require_binding(binding, label=f"predecessor {label}")  # noqa: SLF001
    environment = document.get("environment")
    if (
        not isinstance(environment, Mapping)
        or set(environment) != {"python", "torch", "hip"}
        or environment.get("python") != str(Path(sys.executable).resolve())
        or environment.get("torch") != torch.__version__
        or environment.get("hip") != torch.version.hip
    ):
        raise DualScreenError("execution environment changed")
    return document


def _validate_embedded_source_closures(
    authority: Mapping[str, Any],
    screen_result: Mapping[str, Any],
    horizon_result: Mapping[str, Any],
) -> None:
    screen_sources = screen_result.get("authority", {}).get("source_bindings")
    if not isinstance(screen_sources, Mapping) or any(
        authority["source_bindings"].get(f"predecessor_{label}")
        != screen_sources.get(label)
        for label in predecessor.SOURCE_LABELS
    ):
        raise DualScreenError("exact four-arm predecessor source closure changed")
    horizon_sources = horizon_result.get("authority", {}).get("source_bindings")
    if not isinstance(horizon_sources, Mapping):
        raise DualScreenError("horizon source closure is absent")
    for label in predecessor.SOURCE_LABELS:
        if (
            horizon_sources.get(f"predecessor_{label}")
            != authority["source_bindings"].get(f"predecessor_{label}")
        ):
            raise DualScreenError("horizon predecessor source closure changed")
    for label in ("horizon_runner", "horizon_runner_test"):
        if horizon_sources.get(label) != authority["source_bindings"].get(label):
            raise DualScreenError(f"{label} embedded source binding changed")


def _validate_feature_tensor(features: torch.Tensor, *, encoder: str) -> None:
    expected = (predecessor.ARTIFACT_COUNT, 256, FEATURE_DIM_BY_ENCODER[encoder])
    if features.shape != expected or features.dtype != torch.float16:
        raise DualScreenError(f"{encoder} feature cache tensor changed")
    for start in range(0, features.shape[0], 32):
        batch = features[start : start + 32].to(torch.float32)
        if not bool(torch.isfinite(batch).all()):
            raise DualScreenError(f"{encoder} feature cache contains a nonfinite token")
        norms = torch.linalg.vector_norm(batch, dim=-1)
        if not bool(
            torch.allclose(norms, torch.ones_like(norms), atol=1.0e-3, rtol=1.0e-3)
        ):
            raise DualScreenError(f"{encoder} feature cache is not token-normalized")


def load_bound_inputs_v1(
    authority: Mapping[str, Any],
) -> tuple[
    dict[str, torch.Tensor], predecessor.ScreenIndexV1, dict[str, Any]
]:
    """Load both exact train caches through the reviewed metadata-only path."""

    bindings = authority["predecessor_bindings"]
    screen_result = _json_from_bound_file(bindings["screen_result"], label="screen result")
    screen_terminal = _json_from_bound_file(
        bindings["screen_terminal"], label="screen terminal"
    )
    screen_review = _json_from_bound_file(
        bindings["screen_terminal_review"], label="screen terminal review"
    )
    horizon_result = _json_from_bound_file(
        bindings["horizon_result"], label="horizon result"
    )
    horizon_terminal = _json_from_bound_file(
        bindings["horizon_terminal"], label="horizon terminal"
    )
    horizon_review = _json_from_bound_file(
        bindings["horizon_terminal_review"], label="horizon terminal review"
    )
    if (
        screen_result.get("schema") != predecessor.SCHEMA
        or screen_result.get("status") != "COMPLETE_ENGINEERING_SCREEN"
        or screen_result.get("collection_justified") is not False
        or screen_result.get("navigation_usefulness_established") is not False
        or screen_terminal.get("schema") != predecessor.TERMINAL_SCHEMA
        or screen_terminal.get("status") != "COMPLETE_COLLECTION_NOT_JUSTIFIED"
        or screen_terminal.get("result_binding") != bindings["screen_result"]
        or screen_review.get("schema")
        != "lewm_go2_matched_branch_successor_screen_terminal_review_v1"
        or screen_review.get("result_binding") != bindings["screen_result"]
        or screen_review.get("terminal_binding") != bindings["screen_terminal"]
        or screen_review.get("protected_material_opened") is not False
        or screen_review.get("evaluation_rgb_opened") is not False
        or screen_review.get("findings") != []
        or horizon_result.get("schema")
        != "lewm_go2_dense_vjepa2_1_horizon_diagnostic_result_v1"
        or horizon_result.get("status") != "COMPLETE_FUTILITY_STOP"
        or horizon_result.get("training_set_capacity_established") is not False
        or horizon_result.get("collection_justified") is not False
        or horizon_terminal.get("schema")
        != "lewm_go2_dense_vjepa2_1_horizon_diagnostic_terminal_v1"
        or horizon_terminal.get("status") != "COMPLETE_FUTILITY_STOP"
        or horizon_terminal.get("result_binding") != bindings["horizon_result"]
        or horizon_review.get("schema")
        != "lewm_go2_dense_vjepa2_1_horizon_diagnostic_terminal_review_v1"
        or horizon_review.get("status")
        != "PASS_COMPLETE_FUTILITY_STOP_TERMINAL_REVIEW"
        or horizon_review.get("result_binding") != bindings["horizon_result"]
        or horizon_review.get("terminal_binding") != bindings["horizon_terminal"]
        or horizon_review.get("protected_material_opened") is not False
        or horizon_review.get("findings") != []
    ):
        raise DualScreenError("predecessor evidence contract changed")
    _validate_embedded_source_closures(authority, screen_result, horizon_result)

    bundle = predecessor.screen_data.load_bound_posthoc_bundle_v1()
    index = predecessor.build_screen_index_v1(bundle)
    if (
        index.index_sha256 != INDEX_SHA256
        or bundle.access_audit.get("rgb_leaf_open_count") != 0
        or screen_result.get("screen_index", {}).get("index_sha256") != INDEX_SHA256
        or screen_result.get("screen_index", {}).get("eval_rgb_leaf_open_count") != 0
    ):
        raise DualScreenError("metadata-only predecessor index changed")

    loaded: dict[str, torch.Tensor] = {}
    receipts = screen_result.get("feature_caches")
    if not isinstance(receipts, Mapping) or set(receipts) != {"vjepa2_1", "dinov2"}:
        raise DualScreenError("screen feature receipts changed")
    for encoder in ("vjepa2_1", "dinov2"):
        receipt = _json_from_bound_file(
            bindings[f"{encoder}_feature_receipt"], label=f"{encoder} receipt"
        )
        if (
            receipts.get(encoder) != receipt
            or receipt.get("binding") != bindings[f"{encoder}_feature_cache"]
            or receipt.get("index_sha256") != INDEX_SHA256
            or receipt.get("artifact_order_sha256") != ARTIFACT_ORDER_SHA256
            or receipt.get("train_artifact_open_count") != predecessor.ARTIFACT_COUNT
            or receipt.get("eval_artifact_open_count") != 0
        ):
            raise DualScreenError(f"{encoder} cache receipt changed")
        features = predecessor._load_feature_cache(  # noqa: SLF001
            receipt, expected_encoder=encoder, index=index
        )
        _validate_feature_tensor(features, encoder=encoder)
        loaded[encoder] = features
    return loaded, index, screen_result


def _unique_batch_v1(
    features: torch.Tensor,
    index: predecessor.ScreenIndexV1,
    selected: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    contexts = features[index.context_indices[selected]].to(device=device, dtype=torch.float32)
    targets = features[index.target_indices[selected]].to(device=device, dtype=torch.float32)
    histories = index.history_actions[selected].to(device=device)
    return contexts, targets, histories


@torch.no_grad()
def evaluate_arm_v1(
    model: JointResidualTokenAdapterJEPAV1,
    features: torch.Tensor,
    index: predecessor.ScreenIndexV1,
    *,
    device: torch.device,
    batch_states: int = 4,
) -> dict[str, float]:
    """Evaluate only in the unchanged frozen encoder coordinates."""

    model.eval()
    totals = {
        "matched": 0.0,
        "persistence": 0.0,
        "deranged": 0.0,
        "retrieval": 0.0,
        "rows": 0.0,
    }
    action_ids = torch.arange(predecessor.ACTION_COUNT, device=device)
    for start in range(0, predecessor.STATE_COUNT, batch_states):
        selected = torch.arange(start, min(predecessor.STATE_COUNT, start + batch_states))
        frozen_context, frozen_targets, history = _unique_batch_v1(
            features, index, selected, device
        )
        count = selected.numel()
        adapted_context = model.adapt_online(
            frozen_context.reshape(count * 3, 256, features.shape[-1])
        ).reshape(count, 3, 256, features.shape[-1])
        branch_context = adapted_context[:, None].expand(
            -1, predecessor.ACTION_COUNT, -1, -1, -1
        ).reshape(count * predecessor.ACTION_COUNT, 3, 256, features.shape[-1]).contiguous()
        branch_history = history[:, None].expand(-1, predecessor.ACTION_COUNT, -1)
        branch_history = branch_history.reshape(count * predecessor.ACTION_COUNT, 2).contiguous()
        candidates = torch.arange(predecessor.ACTION_COUNT, device=device).repeat(count)
        predictions = model.predict_from_adapted_context(
            branch_context, branch_history, candidates
        ).reshape(count, predecessor.ACTION_COUNT, 256, features.shape[-1])
        matrix = predecessor.cosine_distance_matrix_v1(predictions, frozen_targets)
        matched = matrix[:, action_ids, action_ids]
        deranged = matrix[:, (action_ids + 1) % predecessor.ACTION_COUNT, action_ids]
        persistence = 1.0 - torch.einsum(
            "bnd,band->ba",
            F.normalize(frozen_context[:, -1], dim=-1),
            F.normalize(frozen_targets, dim=-1),
        ) / 256
        totals["matched"] += float(matched.sum())
        totals["persistence"] += float(persistence.sum())
        totals["deranged"] += float(deranged.sum())
        totals["retrieval"] += float(
            (matrix.argmin(dim=-1) == action_ids).sum()
        )
        totals["rows"] += float(count * predecessor.ACTION_COUNT)
    matched_mean = totals["matched"] / totals["rows"]
    persistence_mean = totals["persistence"] / totals["rows"]
    result = {
        "matched_cosine_error": matched_mean,
        "persistence_cosine_error": persistence_mean,
        "error_to_persistence_ratio": matched_mean / max(persistence_mean, 1.0e-12),
        "branch_retrieval_accuracy": totals["retrieval"] / totals["rows"],
        "cyclic_deranged_cosine_error": totals["deranged"] / totals["rows"],
        "action_intervention_margin": totals["deranged"] / totals["rows"] - matched_mean,
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ArmNonfiniteError("frozen-space evaluation became nonfinite")
    return result


def effective_rank_v1(pooled: torch.Tensor) -> float:
    """Compute the exact preregistered float64 entropy effective rank."""

    if pooled.ndim != 2 or pooled.shape[0] < 1 or not bool(torch.isfinite(pooled).all()):
        raise DualScreenError("effective-rank input is malformed or nonfinite")
    values = pooled.detach().to(device="cpu", dtype=torch.float64)
    centered = values - values.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / values.shape[0]
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    total = eigenvalues.sum()
    if not bool(torch.isfinite(eigenvalues).all()) or not bool(torch.isfinite(total)):
        raise ArmNonfiniteError("effective-rank spectrum became nonfinite")
    if not bool(total > 0.0):
        raise EffectiveRankDegenerateError("effective-rank spectrum is nonpositive")
    probabilities = eigenvalues[eigenvalues > 0.0] / total
    rank = torch.exp(-(probabilities * torch.log(probabilities)).sum())
    value = float(rank)
    if not math.isfinite(value) or value <= 0.0:
        raise DualScreenError("effective rank became invalid")
    return value


@torch.no_grad()
def retention_metrics_v1(
    model: JointResidualTokenAdapterJEPAV1,
    features: torch.Tensor,
    *,
    device: torch.device,
    batch_artifacts: int,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure all-cache online retention and online/EMA normalization."""

    selected_config = dual_config_v1() if config is None else config
    model.eval()
    cosine_sum = 0.0
    token_count = 0
    maximum_online_norm_error = 0.0
    maximum_target_norm_error = 0.0
    frozen_pooled: list[torch.Tensor] = []
    adapted_pooled: list[torch.Tensor] = []
    for start in range(0, features.shape[0], batch_artifacts):
        frozen = features[start : start + batch_artifacts].to(
            device=device, dtype=torch.float32
        )
        online = model.adapt_online(frozen)
        target = model.adapt_target(frozen)
        if not (
            bool(torch.isfinite(frozen).all())
            and bool(torch.isfinite(online).all())
            and bool(torch.isfinite(target).all())
        ):
            raise ArmNonfiniteError("retention tokens became nonfinite")
        online_norm = torch.linalg.vector_norm(online, dim=-1)
        target_norm = torch.linalg.vector_norm(target, dim=-1)
        maximum_online_norm_error = max(
            maximum_online_norm_error, float((online_norm - 1.0).abs().max())
        )
        maximum_target_norm_error = max(
            maximum_target_norm_error, float((target_norm - 1.0).abs().max())
        )
        cosine = (online * F.normalize(frozen, dim=-1, eps=1.0e-12)).sum(dim=-1)
        cosine_sum += float(cosine.to(torch.float64).sum())
        token_count += cosine.numel()
        frozen_pooled.append(frozen.mean(dim=1).to(device="cpu", dtype=torch.float64))
        adapted_pooled.append(online.mean(dim=1).to(device="cpu", dtype=torch.float64))
    try:
        frozen_rank = effective_rank_v1(torch.cat(frozen_pooled, dim=0))
    except EffectiveRankDegenerateError as error:
        raise DualScreenError("frozen cache effective rank is degenerate") from error
    try:
        adapted_rank = effective_rank_v1(torch.cat(adapted_pooled, dim=0))
    except EffectiveRankDegenerateError:
        adapted_rank = 0.0
    mean_cosine = cosine_sum / token_count
    rank_ratio = adapted_rank / frozen_rank
    tolerance = float(selected_config["unit_norm_absolute_tolerance"])
    result = {
        "mean_online_to_frozen_token_cosine": mean_cosine,
        "frozen_effective_rank": frozen_rank,
        "adapted_effective_rank": adapted_rank,
        "effective_rank_ratio": rank_ratio,
        "maximum_online_unit_norm_error": maximum_online_norm_error,
        "maximum_ema_unit_norm_error": maximum_target_norm_error,
        "all_tokens_finite": True,
        "online_unit_norm_passed": maximum_online_norm_error <= tolerance,
        "ema_unit_norm_passed": maximum_target_norm_error <= tolerance,
    }
    result["retention_passed"] = (
        mean_cosine >= float(selected_config["minimum_retention_cosine"])
        and rank_ratio >= float(selected_config["minimum_effective_rank_ratio"])
        and result["online_unit_norm_passed"]
        and result["ema_unit_norm_passed"]
    )
    if not all(
        math.isfinite(value)
        for value in result.values()
        if type(value) is float
    ):
        raise ArmNonfiniteError("retention metric became nonfinite")
    return result


def _adapter_movement_v1(
    model: JointResidualTokenAdapterJEPAV1,
    initial_state: Mapping[str, torch.Tensor],
) -> float:
    squared = 0.0
    for name, value in model.online_adapter.state_dict().items():
        if name not in initial_state or value.shape != initial_state[name].shape:
            raise DualScreenError("online adapter state changed structurally")
        difference = value.detach().to(device="cpu", dtype=torch.float64) - initial_state[
            name
        ].to(dtype=torch.float64)
        squared += float((difference * difference).sum())
    movement = math.sqrt(squared)
    if not math.isfinite(movement):
        raise ArmNonfiniteError("adapter movement became nonfinite")
    return movement


def _capacity_passed_v1(
    metrics: Mapping[str, Any],
    retention: Mapping[str, Any],
    movement: float,
    *,
    config: Mapping[str, Any] | None = None,
) -> bool:
    selected = dual_config_v1() if config is None else config
    return (
        metrics["error_to_persistence_ratio"]
        <= float(selected["maximum_error_to_persistence_ratio"])
        and metrics["branch_retrieval_accuracy"]
        >= float(selected["minimum_branch_retrieval_accuracy"])
        and metrics["action_intervention_margin"]
        > float(selected["minimum_intervention_margin"])
        and retention.get("retention_passed") is True
        and movement > 0.0
    )


def _may_continue_v1(
    arm: str,
    trace_400: Mapping[str, Any],
    trace_800: Mapping[str, Any],
    retention: Mapping[str, Any],
    movement: float,
    *,
    config: Mapping[str, Any] | None = None,
) -> bool:
    selected = dual_config_v1() if config is None else config
    gates = selected["midpoint_gates"][arm]
    return (
        retention.get("retention_passed") is True
        and movement > 0.0
        and trace_800["action_intervention_margin"]
        > float(selected["minimum_intervention_margin"])
        and trace_800["error_to_persistence_ratio"]
        < trace_400["error_to_persistence_ratio"]
        and trace_800["branch_retrieval_accuracy"]
        > trace_400["branch_retrieval_accuracy"]
        and trace_800["error_to_persistence_ratio"]
        <= float(gates["maximum_error_to_persistence_ratio"])
        and trace_800["branch_retrieval_accuracy"]
        >= float(gates["minimum_branch_retrieval_accuracy"])
    )


def _finite_nested(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(_finite_nested(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_nested(item) for item in value)
    return True


def _cpu_clone_nested(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_clone_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone_nested(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone_nested(item) for item in value)
    return value


def _nested_equal_v1(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and left.keys() == right.keys()
            and all(_nested_equal_v1(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)  # type: ignore[arg-type]
            and all(
                _nested_equal_v1(left_item, right_item)
                for left_item, right_item in zip(left, right, strict=True)  # type: ignore[arg-type]
            )
        )
    return type(left) is type(right) and left == right


def _checkpoint_v1(
    path: Path,
    *,
    arm: str,
    model: JointResidualTokenAdapterJEPAV1,
    optimizer: torch.optim.Optimizer,
    initial_online_adapter_state: Mapping[str, torch.Tensor],
    movement: float,
    update: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if path.exists():
        raise DualScreenError("checkpoint path already exists")
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "arm": arm,
        "seed": int(config["seed"]),
        "update": update,
        "feature_dim": model.feature_dim,
        "config": dict(config),
        "adapter_movement_l2": movement,
        "model_state_dict": {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        },
        "optimizer_state_dict": _cpu_clone_nested(optimizer.state_dict()),
        "initial_online_adapter_state_dict": {
            name: tensor.detach().cpu().clone()
            for name, tensor in initial_online_adapter_state.items()
        },
    }
    if not _finite_nested(payload):
        raise ArmNonfiniteError("checkpoint state became nonfinite")
    torch.save(payload, path)
    observed = torch.load(path, map_location="cpu", weights_only=True)
    if not _finite_nested(observed) or not _nested_equal_v1(observed, payload):
        raise DualScreenError("checkpoint round-trip validation failed")
    return predecessor.file_binding_v1(path)


def _raise_if_nonfinite_training_state(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> None:
    if not all(
        not parameter.is_floating_point() or bool(torch.isfinite(parameter).all())
        for parameter in model.parameters()
    ) or not _finite_nested(optimizer.state_dict()):
        raise ArmNonfiniteError("model or optimizer state became nonfinite")


def train_arm_v1(
    arm: str,
    features: torch.Tensor,
    index: predecessor.ScreenIndexV1,
    *,
    config: Mapping[str, Any],
    device: torch.device,
    output_root: Path,
) -> dict[str, Any]:
    """Train one arm to its fixed update-800 or update-1600 terminal."""

    if arm not in ARMS:
        raise DualScreenError("unknown dual-screen arm")
    seed = int(config["seed"])
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    model = JointResidualTokenAdapterJEPAV1(
        feature_dim=int(features.shape[-1]),
    ).to(device)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(config["learning_rate"]),
        betas=tuple(config["adamw_betas"]),
        eps=float(config["adamw_eps"]),
        weight_decay=float(config["weight_decay"]),
        amsgrad=bool(config["adamw_amsgrad"]),
        maximize=bool(config["adamw_maximize"]),
        foreach=bool(config["adamw_foreach"]),
        capturable=bool(config["adamw_capturable"]),
        differentiable=bool(config["adamw_differentiable"]),
        fused=bool(config["adamw_fused"]),
    )
    initial_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.online_adapter.state_dict().items()
    }
    generator = torch.Generator(device="cpu").manual_seed(seed)
    ordering = torch.randperm(predecessor.STATE_COUNT, generator=generator)
    cursor = 0
    traces: list[dict[str, Any]] = []
    movement_trace: dict[str, float] = {}
    checkpoints: dict[str, Any] = {}
    nonfinite_count = 0
    completed_updates = 0
    status = ""
    deterministic_repeat_passed: bool | None = None
    retention_repeat_passed: bool | None = None
    started = time.perf_counter()
    last_objective: dict[str, float] = {}
    trace_400: dict[str, Any] | None = None
    numerical_failure_message: str | None = None
    execution_witness_passed: bool | None = None
    try:
        movement_trace["update_0"] = _adapter_movement_v1(model, initial_state)
        traces.append(
            {
                "update": 0,
                "adapter_movement_l2": movement_trace["update_0"],
                **evaluate_arm_v1(
                    model,
                    features,
                    index,
                    device=device,
                    batch_states=int(config["evaluation_batch_states"]),
                ),
            }
        )
        for update in range(1, int(config["maximum_updates"]) + 1):
            batch_size = int(config["batch_states"])
            if cursor + batch_size > predecessor.STATE_COUNT:
                ordering = torch.randperm(predecessor.STATE_COUNT, generator=generator)
                cursor = 0
            selected = ordering[cursor : cursor + batch_size]
            cursor += batch_size
            frozen_context, frozen_targets, history = _unique_batch_v1(
                features, index, selected, device
            )
            unique = torch.cat((frozen_context, frozen_targets), dim=1)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=False):
                adapted_unique = model.adapt_online(
                    unique.reshape(batch_size * 12, 256, features.shape[-1])
                ).reshape(batch_size, 12, 256, features.shape[-1])
                adapted_context = adapted_unique[:, :3]
                branch_context = adapted_context[:, None].expand(
                    -1, predecessor.ACTION_COUNT, -1, -1, -1
                ).reshape(
                    batch_size * predecessor.ACTION_COUNT,
                    3,
                    256,
                    features.shape[-1],
                ).contiguous()
                branch_history = history[:, None].expand(
                    -1, predecessor.ACTION_COUNT, -1
                ).reshape(batch_size * predecessor.ACTION_COUNT, 2).contiguous()
                candidates = torch.arange(
                    predecessor.ACTION_COUNT, device=device
                ).repeat(batch_size)
                predictions = model.predict_from_adapted_context(
                    branch_context, branch_history, candidates
                ).reshape(
                    batch_size, predecessor.ACTION_COUNT, 256, features.shape[-1]
                )
                ema_targets = model.adapt_target(
                    frozen_targets.reshape(
                        batch_size * predecessor.ACTION_COUNT, 256, features.shape[-1]
                    )
                ).reshape_as(frozen_targets)
                ema_common, ema_components = predecessor.common_objective_v1(
                    predictions,
                    ema_targets,
                    temperature=float(config["cross_entropy_temperature"]),
                    cross_entropy_coefficient=float(
                        config["cross_entropy_coefficient"]
                    ),
                )
                frozen_common, frozen_components = predecessor.common_objective_v1(
                    predictions,
                    frozen_targets,
                    temperature=float(config["cross_entropy_temperature"]),
                    cross_entropy_coefficient=float(
                        config["cross_entropy_coefficient"]
                    ),
                )
                identity = (
                    1.0
                    - (
                        adapted_unique
                        * F.normalize(unique, dim=-1, eps=1.0e-12)
                    ).sum(dim=-1)
                ).mean()
                frozen_std = torch.std(unique, dim=(0, 1, 2), correction=0).detach()
                adapted_std = torch.std(
                    adapted_unique, dim=(0, 1, 2), correction=0
                )
                relative_variance = F.relu(
                    float(config["relative_variance_floor"]) * frozen_std
                    - adapted_std
                ).mean()
                loss = (
                    float(config["ema_target_coefficient"]) * ema_common
                    + float(config["frozen_target_coefficient"]) * frozen_common
                    + float(config["identity_coefficient"]) * identity
                    + float(config["relative_variance_coefficient"])
                    * relative_variance
                )
            if not bool(torch.isfinite(loss)):
                raise ArmNonfiniteError("training loss became nonfinite")
            loss.backward()
            gradients = [
                parameter.grad
                for parameter in trainable_parameters
                if parameter.grad is not None
            ]
            if not gradients or not all(bool(torch.isfinite(item).all()) for item in gradients):
                raise ArmNonfiniteError("training gradient became absent or nonfinite")
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters, float(config["gradient_clip_norm"])
            )
            if not bool(torch.isfinite(grad_norm)):
                raise ArmNonfiniteError("gradient norm became nonfinite")
            optimizer.step()
            model.update_target_ema_(float(config["ema_momentum"]))
            _raise_if_nonfinite_training_state(model, optimizer)
            completed_updates = update
            last_objective = {
                "total": float(loss.detach()),
                "gradient_norm_before_clip": float(grad_norm.detach()),
                "ema_common": float(ema_common.detach()),
                "ema_matched": float(ema_components["matched"].detach()),
                "ema_contrastive": float(ema_components["contrastive"].detach()),
                "frozen_common": float(frozen_common.detach()),
                "frozen_matched": float(frozen_components["matched"].detach()),
                "frozen_contrastive": float(
                    frozen_components["contrastive"].detach()
                ),
                "identity": float(identity.detach()),
                "relative_variance": float(relative_variance.detach()),
            }
            if not all(math.isfinite(value) for value in last_objective.values()):
                raise ArmNonfiniteError("objective component became nonfinite")
            if update not in {400, 800, 1_600}:
                continue
            metrics = evaluate_arm_v1(
                model,
                features,
                index,
                device=device,
                batch_states=int(config["evaluation_batch_states"]),
            )
            movement = _adapter_movement_v1(model, initial_state)
            movement_trace[f"update_{update}"] = movement
            trace: dict[str, Any] = {
                "update": update,
                "objective": last_objective,
                "adapter_movement_l2": movement,
                **metrics,
            }
            if update == 400:
                trace_400 = trace
                traces.append(trace)
                execution_witness_passed = movement > 0.0
                continue

            repeat = evaluate_arm_v1(
                model,
                features,
                index,
                device=device,
                batch_states=int(config["evaluation_batch_states"]),
            )
            primary_repeat_passed = repeat == metrics
            deterministic_repeat_passed = (
                primary_repeat_passed
                if deterministic_repeat_passed is None
                else deterministic_repeat_passed and primary_repeat_passed
            )
            retention = retention_metrics_v1(
                model,
                features,
                device=device,
                batch_artifacts=int(
                    config["retention_batch_artifacts"][ENCODER_BY_ARM[arm]]
                ),
                config=config,
            )
            retention_repeat = retention_metrics_v1(
                model,
                features,
                device=device,
                batch_artifacts=int(
                    config["retention_batch_artifacts"][ENCODER_BY_ARM[arm]]
                ),
                config=config,
            )
            this_retention_repeat = retention_repeat == retention
            retention_repeat_passed = (
                this_retention_repeat
                if retention_repeat_passed is None
                else retention_repeat_passed and this_retention_repeat
            )
            retention["deterministic_repeat_passed"] = this_retention_repeat
            trace["primary_deterministic_repeat_passed"] = primary_repeat_passed
            trace["retention"] = retention
            traces.append(trace)
            checkpoints[f"update_{update}"] = _checkpoint_v1(
                output_root / f"{arm}_checkpoint_update_{update}.pt",
                arm=arm,
                model=model,
                optimizer=optimizer,
                initial_online_adapter_state=initial_state,
                movement=movement,
                update=update,
                config=config,
            )
            if (
                not primary_repeat_passed
                or not this_retention_repeat
                or execution_witness_passed is not True
                or movement <= 0.0
            ):
                status = "COMPLETE_QUALIFICATION_FAILURE_CAPACITY_NOT_ESTABLISHED"
                break
            if _capacity_passed_v1(metrics, retention, movement, config=config):
                status = "COMPLETE_TRAIN_SET_CAPACITY_ESTABLISHED"
                break
            if update == 800:
                if trace_400 is None:
                    raise DualScreenError("update-400 trace is absent")
                if _may_continue_v1(
                    arm,
                    trace_400,
                    trace,
                    retention,
                    movement,
                    config=config,
                ):
                    continue
                status = "COMPLETE_UPDATE_800_FUTILITY_STOP"
                break
            status = "COMPLETE_TRAIN_SET_CAPACITY_NOT_ESTABLISHED"
            break
    except (ArmNonfiniteError, FloatingPointError) as error:
        nonfinite_count += 1
        numerical_failure_message = str(error)
        status = "COMPLETE_NONFINITE_CAPACITY_NOT_ESTABLISHED"
    except predecessor.ScreenError as error:
        if "nonfinite" not in str(error).lower():
            raise
        nonfinite_count += 1
        numerical_failure_message = str(error)
        status = "COMPLETE_NONFINITE_CAPACITY_NOT_ESTABLISHED"

    if not status:
        raise DualScreenError("arm did not reach a registered terminal")
    elapsed = time.perf_counter() - started
    final_trace = traces[-1] if traces else {}
    capacity = status == "COMPLETE_TRAIN_SET_CAPACITY_ESTABLISHED"
    result = {
        "arm": arm,
        "encoder": ENCODER_BY_ARM[arm],
        "status": status,
        "seed": seed,
        "completed_updates": completed_updates,
        "maximum_updates": int(config["maximum_updates"]),
        "parameter_count_trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "parameter_count_ema": sum(
            parameter.numel() for parameter in model.target_adapter.parameters()
        ),
        "training_seconds": elapsed,
        "updates_per_second": completed_updates / elapsed if elapsed else 0.0,
        "peak_gpu_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "nonfinite_count": nonfinite_count,
        "numerical_failure_message": numerical_failure_message,
        "deterministic_repeat_passed": deterministic_repeat_passed,
        "retention_repeat_passed": retention_repeat_passed,
        "execution_witness_passed": execution_witness_passed,
        "adapter_movement_l2": movement_trace,
        "traces": traces,
        "final_metrics": {
            key: value
            for key, value in final_trace.items()
            if key
            in {
                "matched_cosine_error",
                "persistence_cosine_error",
                "error_to_persistence_ratio",
                "branch_retrieval_accuracy",
                "cyclic_deranged_cosine_error",
                "action_intervention_margin",
            }
        },
        "capacity_established": capacity,
        "checkpoint_bindings": checkpoints,
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _source_bindings_unchanged(authority: Mapping[str, Any]) -> None:
    for label, expected in authority["source_bindings"].items():
        if predecessor.file_binding_v1(Path(str(expected["path"]))) != expected:
            raise DualScreenError(f"source {label} changed during execution")
    for label, expected in authority["predecessor_bindings"].items():
        if predecessor.file_binding_v1(Path(str(expected["path"]))) != expected:
            raise DualScreenError(f"predecessor {label} changed during execution")


def execute_v1(authority: Mapping[str, Any]) -> dict[str, Any]:
    output_root = Path(str(authority["output_root"]))
    predecessor._safe_path(  # noqa: SLF001
        output_root.parent, label="dual-screen output parent", must_exist=False
    )
    output_root.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise DualScreenError("the preregistered dual screen requires a CUDA/ROCm GPU")
    features_by_encoder, index, screen_result = load_bound_inputs_v1(authority)
    device = torch.device("cuda")
    arms: dict[str, Any] = {}
    for arm in ARMS:
        encoder = ENCODER_BY_ARM[arm]
        arms[arm] = train_arm_v1(
            arm,
            features_by_encoder[encoder],
            index,
            config=authority["config"],
            device=device,
            output_root=output_root,
        )
    _source_bindings_unchanged(authority)
    eligible = [arm for arm in ARMS if arms[arm]["capacity_established"]]
    if eligible:
        status = "COMPLETE_BOTH_ATTEMPTED_AT_LEAST_ONE_CAPACITY_ESTABLISHED"
        next_route = "ELIGIBLE_ARMS_REQUIRE_SEPARATE_FRESH_SCENE_PREREGISTRATION"
    else:
        status = "COMPLETE_BOTH_ATTEMPTED_NO_CAPACITY_ESTABLISHED"
        next_route = "STOP_CACHED_TOKEN_ADAPTER_FAMILY_NO_FRESH_DATA_GENERATION"
    report = {
        "schema": RESULT_SCHEMA,
        "status": status,
        "citable_as_scientific_evidence": False,
        "authorizes_collection": False,
        "authorizes_rgb_access": False,
        "authorizes_feature_extraction": False,
        "authorizes_evaluation": False,
        "authorizes_retry": False,
        "fresh_scene_generalization_measured": False,
        "physical_action_ranking_measured": False,
        "planning_utility_measured": False,
        "navigation_usefulness_established": False,
        "authority": dict(authority),
        "predecessor_result_binding": authority["predecessor_bindings"][
            "screen_result"
        ],
        "predecessor_collection_justified": screen_result["collection_justified"],
        "screen_index": {
            "states": len(index.state_ids),
            "scenes": len(set(index.scene_ids)),
            "families": len(set(index.family_ids)),
            "artifacts": len(index.artifact_ids),
            "index_sha256": index.index_sha256,
        },
        "access_audit": {
            "rgb_leaf_open_count": 0,
            "evaluation_feature_open_count": 0,
            "evaluation_target_open_count": 0,
            "metadata_role_disjointness_inspected": True,
            "train_feature_cache_artifacts_per_encoder": predecessor.ARTIFACT_COUNT,
        },
        "device": {
            "type": str(device),
            "name": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "hip": torch.version.hip,
        },
        "frozen_controls": FROZEN_CONTROLS,
        "arms": arms,
        "both_arms_launched_and_resolved": True,
        "eligible_arms": eligible,
        "training_set_capacity_established": bool(eligible),
        "collection_justified": False,
        "next_route": next_route,
    }
    predecessor._write_json_exclusive(output_root / "result.json", report)  # noqa: SLF001
    result_binding = predecessor.file_binding_v1(output_root / "result.json")
    terminal = {
        "schema": TERMINAL_SCHEMA,
        "status": status,
        "citable_as_scientific_evidence": False,
        "authorizes_collection": False,
        "authorizes_rgb_access": False,
        "authorizes_feature_extraction": False,
        "authorizes_evaluation": False,
        "authorizes_retry": False,
        "result_binding": result_binding,
        "arm_statuses": {arm: arms[arm]["status"] for arm in ARMS},
        "completed_updates": {
            arm: arms[arm]["completed_updates"] for arm in ARMS
        },
        "both_arms_launched_and_resolved": True,
        "eligible_arms": eligible,
        "training_set_capacity_established": bool(eligible),
        "collection_justified": False,
        "next_route": next_route,
    }
    predecessor._write_json_exclusive(output_root / "terminal.json", terminal)  # noqa: SLF001
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
    output_root = Path(str(authority["output_root"]))
    output_existed = output_root.exists()
    try:
        report = execute_v1(authority)
    except Exception as error:
        if (
            not output_existed
            and output_root.is_dir()
            and not (output_root / "terminal.json").exists()
        ):
            predecessor._write_json_exclusive(  # noqa: SLF001
                output_root / "terminal.json",
                {
                    "schema": TERMINAL_SCHEMA,
                    "status": "CONSUMED_TERMINAL_INFRASTRUCTURE_FAILURE",
                    "citable_as_scientific_evidence": False,
                    "authorizes_collection": False,
                    "authorizes_rgb_access": False,
                    "authorizes_feature_extraction": False,
                    "authorizes_evaluation": False,
                    "authorizes_retry": False,
                    "both_arms_launched_and_resolved": False,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
            )
        raise
    print(
        json.dumps(
            {
                "status": report["status"],
                "arm_statuses": {
                    arm: report["arms"][arm]["status"] for arm in ARMS
                },
                "completed_updates": {
                    arm: report["arms"][arm]["completed_updates"] for arm in ARMS
                },
                "eligible_arms": report["eligible_arms"],
                "collection_justified": report["collection_justified"],
                "next_route": report["next_route"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
