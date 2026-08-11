#!/usr/bin/env python3
"""Frozen direct counterfactual-fidelity assay for the 32 epoch-21 predictors.

Stages B/C consume exactly the qualified 20-state, twelve-candidate oracle-v1.2
pilot reconstructed by Stage A.  No utility scorer, oracle label, branch outcome,
future observation, future latent or future proprioception crosses the predictor
inference boundary.  The assay compares each autoregressive H=1..4 prediction
directly with its own registered branch target and performs twelve-way branch
retrieval at every horizon.

The order of operations is deliberately fail closed:

1. validate the complete Stage-A identity/corpus receipts and latent index;
2. freeze a self-digested prospective assay specification (without opening a
   target-latent shard);
3. hash-validate every Stage-A shard;
4. verify the confirmatory commit, report, run package, launch receipts, run
   records and all 32 checkpoint bytes;
5. and only then call ``torch.load`` on the first predictor checkpoint.

Each checkpoint is evaluated one state (all twelve candidates) at a time into an
append-only, fsync'd JSONL ledger.  A partial exact prefix resumes; a corrupted or
differently-bound attempt is preserved and never mixed with the registered run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import dev_checkpoint_v1 as CK  # noqa: E402
from scripts import dev_proprio_predictor_v1 as P  # noqa: E402
from scripts import freeze_dev_proprio_run_package_v1 as FR  # noqa: E402
from scripts import run_dev_proprio_factorial_driver_v1 as D  # noqa: E402
from scripts import run_dev_v03_temporal_action_jepa_v1 as T  # noqa: E402


STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
OUT_ROOT = ROOT / ".generated/go2_counterfactual_fidelity_v1_2"
STAGE_A_DIR = OUT_ROOT
RESULT_DIR = OUT_ROOT / "predictor_assay"
PREDICTION_DIR = RESULT_DIR / "prediction_ledgers"
ASSAY_SPEC_PATH = RESULT_DIR / "assay_spec.json"
RESULT_PATH = RESULT_DIR / "result.json"

FROZEN_CONFIRMATORY_COMMIT = "443e5914694a533534486b629e95ec15f8df9b7a"
FROZEN_CONFIRMATORY_REPORT_DIGEST = (
    "60b0bb2d0b13ba47eac5e306c33d97dcfdce31102870edfc50b01f7f9b247161"
)
FROZEN_RUN_PACKAGE_DIGEST = (
    "cf0456bef0cbe7cd8f2cd666b600f91ebf845f6156d180569edf36be53552991"
)
FROZEN_PILOT_IDENTITY_MANIFEST_DIGEST = (
    "5f380bf7f49ef10437c7d9644f04dbef065f0550dfd30d0ec36208cda25d08cf"
)
FROZEN_CANDIDATE_BANK_DIGEST = (
    "85471e44a0fe8f3c59fff258e9b23933e306f69b6d590c832e2b8da1f34a8cd9"
)
FROZEN_ORACLE_V1_2_DIGEST = (
    "3ffbe1a87f7975c97e7ff42e50a6a00ca0f47d8840a434d0ff215c303bf6f0e4"
)
FROZEN_STAGE_A_ASSAY_SPEC_DIGEST = (
    "39545af7599da2f2a1bf171c050489eea9f8637137bc1a9c0af3a193d1aaaf3a"
)
FROZEN_STAGE_A_IDENTITY_MANIFEST_DIGEST = (
    "ce2cbbe8dab9a89ad6f85d16c56a9d712d791c8bbfd8925a8f01efc0c039705a"
)
FROZEN_RENDER_CONTRACT_DIGEST = (
    "2faa22e3b10a2c4199bdabdbc0ed0e1ff9c7c4ac48bb489daeb0fd70d5b65c17"
)
FROZEN_TEXTURED_RENDERER_DIGEST = (
    "df70a0c16ad421ae93a93c4d9dda0fd4d6f154f42d9710c7fc2f0242c3e8cb1b"
)
FROZEN_PREPROCESS_CONTRACT_DIGEST = (
    "2688ca405ed7e8bb86e82f1d111b7b865466f4d497b973a04a52af846b5da6a9"
)
FROZEN_PREPROCESSING_DIGEST = (
    "8e6aa177b094ea91d27b3c91bcd8f01835b8be5fc51796d145314982ea930fe5"
)
FROZEN_TARGET_ENCODER_DIGEST = (
    "15ff78a0205ba138a740f12f6eb9bb3f78bce9c5ba8c2849f7e83489a6b2b6a5"
)
FROZEN_TARGET_ENCODER_CHECKPOINT = Path(
    "/home/andrewknowles/.cache/vjepa2_1_vitl_dist_vitG_384.pt"
)
FROZEN_TARGET_ENCODER_CHECKPOINT_SHA256 = (
    "7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6"
)
FROZEN_TARGET_ENCODER_CHECKPOINT_BYTES = 5_151_198_524
FROZEN_LATENT_NORMALISATION_CONTRACT = (
    "raw final-block tokens rounded to float16; consumers reload float16 as "
    "float32 and apply F.layer_norm over the 1024-D token dimension"
)
FROZEN_SOURCE_PILOT_BRANCH_LEDGER_SHA256 = (
    "761c0de85296db70e044a177a75cbd1f12181c506a375a8827946468c8a6ce4c"
)
FROZEN_SOURCE_PILOT_GATE_SHA256 = (
    "e77bf3b27551aeeca2d5a2bfe92d04b949e08b3bf5e4e1f2168387a50c832834"
)

CONFIRMATORY_REPORT_PATH = D.OUT / "final_analysis.json"
RUN_PACKAGE_PATH = D.PROPRIO / "scientific_run_package.json"
INITIAL_RECEIPT_PATH = D.OUT / "launch_authorisation.json"
CONTINUATION_RECEIPT_PATH = D.OUT / "continuation_authorisation.json"
FROZEN_MASK_SOURCE_PATH = (
    Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03")
    / "two_step/evaluation/MATCHED_24_EPOCH_result_epochs_0_23.json"
)

FROZEN_N = 8
EXPECTED_STATES = 20
EXPECTED_BRANCHES = 240
EXPECTED_CANDIDATES = 12
EXPECTED_FAMILIES = 8
MAX_H = 4
TOKENS = 768
TOKEN_DIM = 1_024
CONTEXT_SLOTS = 3
SAMPLES_PER_SLOT = 5
FROZEN_INFERENCE_BATCH = 12
MASK_THRESHOLD_H1 = 0.7618998289108276
MASK_THRESHOLD_H2_PLUS = 0.8970220685005188
TIE_ATOL = 1e-12
FAMILIES = (
    "large_enclosed_maze",
    "local_composite_motifs",
    "loop_alias_stress",
    "medium_enclosed_maze",
    "open_obstacle_field",
    "rough_local_dynamics",
    "small_enclosed_maze",
    "visual_sensor_stress",
)
DIAGNOSTIC_FAMILY = "local_composite_motifs"

FROZEN_PREDICTOR_SOURCE_PATHS = (
    "scripts/dev_checkpoint_v1.py",
    "scripts/dev_proprio_predictor_v1.py",
    "scripts/run_dev_proprio_factorial_driver_v1.py",
    "scripts/run_dev_v03_temporal_action_jepa_v1.py",
    "scripts/dev_action_slew_reconstruction_v1.py",
    "scripts/build_dev_v03_proprio_action_manifest_v1.py",
    "scripts/freeze_dev_proprio_run_package_v1.py",
)
ASSAY_SOURCE_PATHS = (
    *FROZEN_PREDICTOR_SOURCE_PATHS,
    "scripts/eval_dev_v03_horizon_rollout_v1.py",
    "scripts/eval_dev_proprio_factorial_v1.py",
    "scripts/build_go2_counterfactual_fidelity_stage_a_v1_2.py",
    "scripts/encode_go2_counterfactual_fidelity_stage_a_v1_2.py",
    "scripts/analyze_go2_counterfactual_predictor_qualification_v1_2.py",
)


class AssayRefused(RuntimeError):
    """A frozen binding, completeness condition or leakage guard failed."""


@dataclass(frozen=True)
class PlanningState:
    """Allow-listed observed input for a state and its twelve hypotheses.

    Outcome rows and target-latent handles are intentionally not represented.
    """

    state_index: int
    state_id: str
    family: str
    scene_id: str
    episode_cluster_id: str
    context_key: str
    candidate_names: tuple[str, ...]
    candidate_indices: tuple[int, ...]
    action_blocks: tuple[tuple[tuple[float, ...], ...], ...]
    proprio_history: tuple[tuple[float, ...], ...]
    control_history: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class StageABundle:
    identity: dict[str, Any]
    receipt: dict[str, Any]
    latent_index: dict[str, Any]
    rows: list[dict[str, Any]]
    states: tuple[PlanningState, ...]
    row_by_pair: dict[tuple[str, int], dict[str, Any]]
    context_records: dict[str, dict[str, Any]]
    horizon_records: dict[str, dict[str, Any]]
    identity_digest: str
    corpus_digest: str
    branch_rows_sha256: str
    latent_index_digest: str
    target_encoder_checkpoint_sha256: str


@dataclass(frozen=True)
class FrozenCheckpoint:
    seed_index: int
    seed: int
    cell: str
    path: Path
    sha256: str
    bytes: int
    receipt_path: Path
    authorisation_receipt_digest: str


def sha256_file(path: Path, block_size: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def json_digest(payload: Mapping[str, Any], omit: Iterable[str] = ()) -> str:
    omitted = set(omit)
    material = {key: value for key, value in payload.items() if key not in omitted}
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def sequence_digest(values: Sequence[Any]) -> str:
    return hashlib.sha256(
        json.dumps(list(values), sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def legacy_json_digest(payload: Mapping[str, Any], omit: Iterable[str] = ()) -> str:
    """Digest an older artefact under its published default-JSON rule."""

    omitted = set(omit)
    material = {key: value for key, value in payload.items() if key not in omitted}
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssayRefused(message)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AssayRefused(f"unreadable {label} {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} is not a JSON object: {path}")
    return value


def read_jsonl_strict(path: Path, label: str) -> list[dict[str, Any]]:
    _require(path.is_file(), f"missing {label}: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AssayRefused(
                    f"malformed {label} at {path}:{line_number}: {exc}"
                ) from exc
            _require(isinstance(row, dict),
                     f"non-object {label} at {path}:{line_number}")
            rows.append(row)
    return rows


def verify_embedded_digest(payload: Mapping[str, Any], field: str,
                           label: str) -> str:
    stored = payload.get(field)
    _require(isinstance(stored, str) and len(stored) == 64,
             f"{label} has no valid {field}")
    actual = json_digest(payload, (field,))
    _require(actual == stored, f"{label} {field} mismatch: {actual} != {stored}")
    return stored


def _first(payload: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in payload:
            return payload[name]
    return None


def _safe_relative_path(value: Any, label: str) -> Path:
    _require(isinstance(value, str) and value, f"{label} has no path")
    candidate = Path(value)
    resolved = (STAGE_A_DIR / candidate).resolve() if not candidate.is_absolute() \
        else candidate.resolve()
    root = STAGE_A_DIR.resolve()
    _require(resolved == root or root in resolved.parents,
             f"{label} escapes the Stage-A root")
    return resolved


def _matrix(value: Any, rows: int, columns: int, label: str) -> tuple[tuple[float, ...], ...]:
    _require(isinstance(value, list) and len(value) == rows,
             f"{label} must have {rows} rows")
    output = []
    for row in value:
        _require(isinstance(row, list) and len(row) == columns
                 and all(_finite(item) for item in row),
                 f"{label} must be finite {rows}x{columns}")
        output.append(tuple(float(item) for item in row))
    return tuple(output)


def _normalise(tokens: torch.Tensor) -> torch.Tensor:
    """Exact frozen target/predictor layer-normalisation."""

    return F.layer_norm(tokens, (tokens.shape[-1],))


def changed_mask(current: torch.Tensor, target: torch.Tensor,
                 horizon: int) -> torch.Tensor:
    """Frozen changed-token definition; H>=2 reuses the H=2 threshold."""

    _require(horizon in (1, 2, 3, 4), f"invalid horizon {horizon}")
    threshold = MASK_THRESHOLD_H1 if horizon == 1 else MASK_THRESHOLD_H2_PLUS
    return (target - current).pow(2).mean(dim=-1) >= threshold


def direct_metrics(prediction: torch.Tensor, target: torch.Tensor,
                   current: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    """Exact frozen cosine/error/persistence definitions for one branch.

    A row is first reduced over its changed-token set.  Normalised error is the
    ratio of mean prediction token-MSE to mean persistence token-MSE, not a mean
    of tokenwise ratios.  Advantage is prediction cosine minus persistence cosine.
    """

    _require(prediction.shape == target.shape == current.shape,
             "direct metric tensor shapes differ")
    _require(mask.shape == target.shape[:-1], "direct metric mask shape differs")
    all_cosine = F.cosine_similarity(prediction.float(), target.float(), dim=-1)
    all_persistence_cosine = F.cosine_similarity(
        current.float(), target.float(), dim=-1)
    all_error = (prediction.float() - target.float()).pow(2).mean(-1)
    all_persistence_error = (current.float() - target.float()).pow(2).mean(-1)
    changed = int(mask.sum())
    if changed == 0:
        return {
            "changed_cosine": None,
            "normalised_error_vs_persistence": None,
            "persistence_changed_cosine": None,
            "advantage_over_persistence": None,
            "prediction_mse": None,
            "persistence_mse": None,
            "changed_tokens": 0,
            "total_tokens": int(mask.numel()),
            "full_token_cosine": float(all_cosine.mean()),
            "full_token_persistence_cosine": float(all_persistence_cosine.mean()),
            "full_token_normalised_error_vs_persistence": float(
                all_error.mean() / all_persistence_error.mean().clamp_min(1e-12)),
            "changed_metric_available": False,
        }
    cosine = all_cosine[mask]
    persistence_cosine = all_persistence_cosine[mask]
    error = all_error[mask]
    persistence_error = all_persistence_error[mask]
    changed_cosine = float(cosine.mean())
    persistence = float(persistence_cosine.mean())
    denominator = float(persistence_error.mean().clamp_min(1e-12))
    normalised_error = float(error.mean()) / denominator
    values: dict[str, Any] = {
        "changed_cosine": changed_cosine,
        "normalised_error_vs_persistence": normalised_error,
        "persistence_changed_cosine": persistence,
        "advantage_over_persistence": changed_cosine - persistence,
        "prediction_mse": float(error.mean()),
        "persistence_mse": float(persistence_error.mean()),
        "changed_tokens": changed,
        "total_tokens": int(mask.numel()),
        "full_token_cosine": float(all_cosine.mean()),
        "full_token_persistence_cosine": float(all_persistence_cosine.mean()),
        "full_token_normalised_error_vs_persistence": float(
            all_error.mean() / all_persistence_error.mean().clamp_min(1e-12)),
        "changed_metric_available": True,
    }
    _require(all(_finite(value) for key, value in values.items()
                 if key not in ("changed_tokens", "total_tokens",
                                "changed_metric_available")),
             "direct metric is non-finite")
    return values


def retrieval_metrics(similarity: np.ndarray,
                      candidate_names: Sequence[str]) -> dict[str, Any]:
    """Twelve-way own-branch retrieval under a fixed deterministic tie rule.

    Rows are predicted candidates and columns are registered target branches.
    Higher cosine is better.  Exact ties are ordered by frozen candidate index;
    fractional/optimistic ranks are never substituted after observing outcomes.
    """

    matrix = np.asarray(similarity, dtype=np.float64)
    n = len(candidate_names)
    _require(n == EXPECTED_CANDIDATES,
             f"retrieval needs exactly {EXPECTED_CANDIDATES} candidates")
    _require(matrix.shape == (n, n) and bool(np.isfinite(matrix).all()),
             "retrieval similarity matrix is malformed")
    ranks: list[int] = []
    winners: list[int] = []
    best_wrong_margins: list[float] = []
    mean_wrong_margins: list[float] = []
    pairwise_hits = 0
    exact_ties = 0
    confusion = np.zeros((n, n), dtype=np.int64)
    for query in range(n):
        order = sorted(range(n), key=lambda index: (-matrix[query, index], index))
        rank = order.index(query) + 1
        winner = order[0]
        ranks.append(rank)
        winners.append(winner)
        confusion[query, winner] += 1
        wrong = [float(matrix[query, index]) for index in range(n) if index != query]
        own = float(matrix[query, query])
        best_wrong_margins.append(own - max(wrong))
        mean_wrong_margins.append(own - float(np.mean(wrong)))
        pairwise_hits += sum(own > value for value in wrong)
        exact_ties += sum(abs(own - value) <= TIE_ATOL for value in wrong)
    harmonic = sum(1.0 / rank for rank in range(1, n + 1)) / n
    return {
        "queries": n,
        "top1": float(np.mean(np.asarray(ranks) <= 1)),
        "top3": float(np.mean(np.asarray(ranks) <= 3)),
        "mean_reciprocal_rank": float(np.mean([1.0 / rank for rank in ranks])),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "mean_margin_over_best_wrong": float(np.mean(best_wrong_margins)),
        "mean_margin_over_mean_wrong": float(np.mean(mean_wrong_margins)),
        "pairwise_accuracy": pairwise_hits / (n * (n - 1)),
        "own_wrong_exact_tie_rate": exact_ties / (n * (n - 1)),
        "ranks": ranks,
        "winner_indices": winners,
        "winner_candidates": [candidate_names[index] for index in winners],
        "confusion": confusion.tolist(),
        "candidate_order": list(candidate_names),
        "chance_references": {
            "top1": 1.0 / n,
            "top3": 3.0 / n,
            "mean_reciprocal_rank": harmonic,
            "mean_rank": (n + 1.0) / 2.0,
            "median_rank": (n + 1.0) / 2.0,
            "pairwise_accuracy": 0.5,
            "uniform_confusion_probability_per_cell": 1.0 / n,
        },
        "tie_rule": "descending cosine, then frozen target candidate index",
        "tie_atol": TIE_ATOL,
    }


def t_interval(values: Sequence[float]) -> dict[str, Any]:
    """Two-sided 95% t interval over exactly eight training-seed quadruplets."""

    _require(len(values) == FROZEN_N,
             f"paired inference requires exactly {FROZEN_N} seed values")
    array = np.asarray(values, dtype=np.float64)
    _require(bool(np.isfinite(array).all()), "seed vector contains non-finite values")
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    critical = 2.3646242510102993  # t_(0.975, 7)
    half = critical * sd / math.sqrt(FROZEN_N)
    return {
        "values": [float(value) for value in array],
        "n": FROZEN_N,
        "mean": mean,
        "sample_standard_deviation": sd,
        "t_critical_df7": critical,
        "two_sided_95_t_interval": [mean - half, mean + half],
    }


def source_bindings() -> dict[str, str]:
    """Bind the exact implementation used by this prospective assay."""

    return {relative: sha256_file(ROOT / relative) for relative in ASSAY_SOURCE_PATHS}


def verify_mask_source() -> dict[str, Any]:
    masks = read_json(FROZEN_MASK_SOURCE_PATH, "frozen changed-token mask source").get(
        "masks"
    )
    _require(isinstance(masks, dict), "frozen changed-token mask source has no masks")
    _require(float(masks.get("step1_threshold", float("nan"))) == MASK_THRESHOLD_H1,
             "frozen H1 changed-token threshold differs")
    _require(float(masks.get("step2_threshold", float("nan"))) == MASK_THRESHOLD_H2_PLUS,
             "frozen H2 changed-token threshold differs")
    return {
        "path": str(FROZEN_MASK_SOURCE_PATH),
        "sha256": sha256_file(FROZEN_MASK_SOURCE_PATH),
        "step1_threshold": MASK_THRESHOLD_H1,
        "step2_threshold_reused_h2_h4": MASK_THRESHOLD_H2_PLUS,
        "policy": "H1 uses step1; H2-H4 reuse step2; no assay-data threshold fitting",
    }


def prospective_spec_payload(bundle: StageABundle,
                             mask_source: Mapping[str, Any]) -> dict[str, Any]:
    """Construct the write-once spec without reading target shard contents."""

    return {
        "schema": "go2_counterfactual_predictor_assay_spec_v1_2",
        "status": STATUS,
        "prospective": True,
        "created_before_target_latent_scoring": True,
        "utility_scorer_used": False,
        "stage_a_identity_manifest_digest": bundle.identity_digest,
        "stage_a_corpus_digest": bundle.corpus_digest,
        "stage_a_branch_rows_sha256": bundle.branch_rows_sha256,
        "stage_a_latents_index_digest": bundle.latent_index_digest,
        "target_encoder_checkpoint_sha256": bundle.target_encoder_checkpoint_sha256,
        "pilot_identity_manifest_digest": FROZEN_PILOT_IDENTITY_MANIFEST_DIGEST,
        "candidate_bank_digest": FROZEN_CANDIDATE_BANK_DIGEST,
        "oracle_v1_2_digest_identity_only": FROZEN_ORACLE_V1_2_DIGEST,
        "confirmatory_commit": FROZEN_CONFIRMATORY_COMMIT,
        "confirmatory_report_digest": FROZEN_CONFIRMATORY_REPORT_DIGEST,
        "scientific_run_package_digest": FROZEN_RUN_PACKAGE_DIGEST,
        "seeds": list(D.SEED_REGISTRY[:FROZEN_N]),
        "cells": list(D.CELLS),
        "checkpoint_epoch": D.CHECKPOINT_EPOCH,
        "horizons_reported_individually": [1, 2, 3, 4],
        "primary_horizon_selected": False,
        "normalisation": {
            "context_and_targets": FROZEN_LATENT_NORMALISATION_CONTRACT,
            "predictions": (
                "scripts.dev_proprio_predictor_v1.unroll applies the frozen "
                "run_dev_v03_temporal_action_jepa_v1.normalise after every step"
            ),
            "source": "scripts/run_dev_v03_temporal_action_jepa_v1.py::normalise",
        },
        "changed_token_mask": dict(mask_source),
        "direct_metrics": {
            "changed_cosine":
                "mean token cosine(prediction,target) over target-specific changed mask",
            "persistence_changed_cosine":
                "mean token cosine(last observed context,target) over same mask",
            "advantage_over_persistence":
                "changed_cosine - persistence_changed_cosine",
            "normalised_error_vs_persistence": (
                "mean_changed_tokens(mean_dim((prediction-target)^2)) / "
                "max(mean_changed_tokens(mean_dim((current-target)^2)),1e-12)"
            ),
            "aggregation": (
                "candidate row mean -> state/episode-cluster mean -> family mean -> "
                "unweighted mean of all eight frozen families"
            ),
            "corpus_weighted":
                "changed-token pooled cosine/error numerators and denominators, separate",
            "numerical_device": "CPU float32, matching the frozen evaluator",
        },
        "branch_retrieval": {
            "queries": "each predicted candidate trajectory",
            "gallery": "the twelve registered true branch targets from the same state",
            "similarity": (
                "mean token cosine(pred_i,target_j) over the same complete aligned "
                "768-token grid for every gallery column"
            ),
            "changed_masks_used_for_retrieval": False,
            "correct": "candidate identity i == target identity j",
            "metrics": [
                "top1", "top3", "mean_reciprocal_rank", "mean_rank",
                "median_rank", "mean_margin_over_best_wrong",
                "mean_margin_over_mean_wrong", "pairwise_accuracy", "confusion",
            ],
            "tie_rule": "descending cosine, then frozen target candidate index",
            "tie_atol": TIE_ATOL,
            "numerical_device": (
                "the one frozen predictor device for float32 full-grid cosine; "
                "similarity matrices are durably ledgered"
            ),
            "chance": {
                "top1": 1 / 12,
                "top3": 3 / 12,
                "mean_reciprocal_rank": sum(1 / rank for rank in range(1, 13)) / 12,
                "mean_rank": 6.5,
                "pairwise_accuracy": 0.5,
            },
        },
        "paired_seed_analysis": {
            "replication_unit": "eight fixed training-seed quadruplets",
            "higher_is_better": [
                "changed_cosine", "advantage_over_persistence", "top1", "top3",
                "mean_reciprocal_rank", "pairwise_accuracy",
                "mean_margin_over_best_wrong", "mean_margin_over_mean_wrong",
            ],
            "lower_is_better": ["normalised_error_vs_persistence", "mean_rank"],
            "benefit": (
                "higher metrics: rollout-one_step; lower metrics: one_step-rollout"
            ),
            "B_RGB": "benefit(rgb)",
            "B_prop": "benefit(proprio)",
            "M": "(B_RGB+B_prop)/2",
            "J": "B_prop-B_RGB",
            "interval": "two-sided 95% Student-t, df=7",
        },
        "weighting": {
            "primary": "equal family at every H1-H4; no horizons combined",
            "secondary": "corpus weighted, reported separately",
            "all_families_primary": list(FAMILIES),
            "local_composite_motifs": "also called out as a family diagnostic",
        },
        "leakage_boundary": {
            "predictor_inputs": [
                "three observed context latent frames", "candidate post-slew action blocks",
                "observed control history", "observed proprioceptive history in prop cells",
            ],
            "forbidden": [
                "future RGB", "true target latents", "oracle labels or utility",
                "branch outcomes", "future proprioception", "privileged simulator state",
            ],
            "enforcement": "PlanningState allow-list and predict_state signature",
        },
        "source_bindings": source_bindings(),
        "stage_a_source_bindings": bundle.identity.get("source_bindings"),
    }


def freeze_assay_spec(bundle: StageABundle,
                      mask_source: Mapping[str, Any]) -> dict[str, Any]:
    payload = prospective_spec_payload(bundle, mask_source)
    payload["assay_spec_digest"] = json_digest(payload)
    if ASSAY_SPEC_PATH.is_file():
        existing = read_json(ASSAY_SPEC_PATH, "prospective assay specification")
        verify_embedded_digest(existing, "assay_spec_digest",
                               "prospective assay specification")
        _require(existing == payload,
                 "existing prospective assay specification differs; refusing to mix runs")
        return existing
    atomic_write_json(ASSAY_SPEC_PATH, payload)
    return payload


# ----------------------------------------------------- predictor provenance --
def _git(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *arguments], cwd=ROOT,
                          capture_output=True, text=True)


def _verify_authorisation_receipt(path: Path, label: str) -> dict[str, Any]:
    receipt = read_json(path, label)
    verify_embedded_digest(receipt, "receipt_digest", label)
    _require(receipt.get("run_package_digest") == FROZEN_RUN_PACKAGE_DIGEST,
             f"{label} binds a different run package")
    return receipt


def verify_frozen_predictor_lineage() -> tuple[list[FrozenCheckpoint], dict[str, Any]]:
    """Hash all 32 checkpoint bytes before any checkpoint payload is opened."""

    resolved = _git("rev-parse", "443e591^{commit}")
    _require(resolved.returncode == 0
             and resolved.stdout.strip() == FROZEN_CONFIRMATORY_COMMIT,
             "frozen confirmatory commit does not resolve to the registered object")
    ancestry = _git("merge-base", "--is-ancestor", FROZEN_CONFIRMATORY_COMMIT, "HEAD")
    _require(ancestry.returncode == 0,
             "frozen confirmatory commit is not an ancestor of current HEAD")
    witness = _git(
        "show",
        f"{FROZEN_CONFIRMATORY_COMMIT}:"
        "docs/lewm_go2_proprio_factorial_final_report_2026-08-10.md",
    )
    _require(witness.returncode == 0
             and FROZEN_CONFIRMATORY_REPORT_DIGEST in witness.stdout,
             "confirmatory commit does not witness the registered report digest")

    predictor_source_blobs: dict[str, dict[str, str]] = {}
    for relative in FROZEN_PREDICTOR_SOURCE_PATHS:
        frozen_blob = _git("rev-parse", f"{FROZEN_CONFIRMATORY_COMMIT}:{relative}")
        current_blob = _git("hash-object", "--", relative)
        _require(frozen_blob.returncode == 0 and current_blob.returncode == 0
                 and frozen_blob.stdout.strip() == current_blob.stdout.strip(),
                 f"current frozen inference source differs from confirmatory commit: "
                 f"{relative}")
        predictor_source_blobs[relative] = {
            "git_blob": frozen_blob.stdout.strip(),
            "sha256": sha256_file(ROOT / relative),
        }

    try:
        package = FR.verify(RUN_PACKAGE_PATH)
    except Exception as exc:
        raise AssayRefused(f"independent run-package verification failed: {exc}") from exc
    _require(package.get("package_digest") == FROZEN_RUN_PACKAGE_DIGEST,
             "scientific run-package digest differs")
    _require(package.get("budget", {}).get("checkpoint_epoch") == D.CHECKPOINT_EPOCH,
             "run package does not freeze epoch 21")
    _require(package.get("budget", {}).get("selection_permitted") is False,
             "run package permits checkpoint selection")
    _require(package.get("horizon_masks", {}).get("thresholds", {}).get("step1")
             == MASK_THRESHOLD_H1,
             "run package H1 mask threshold differs")
    _require(package.get("horizon_masks", {}).get("thresholds", {}).get("step2")
             == MASK_THRESHOLD_H2_PLUS,
             "run package H2-H4 mask threshold differs")

    report = read_json(CONFIRMATORY_REPORT_PATH, "frozen confirmatory report")
    verify_embedded_digest(report, "report_digest", "frozen confirmatory report")
    _require(report.get("report_digest") == FROZEN_CONFIRMATORY_REPORT_DIGEST,
             "confirmatory report digest differs")
    frozen_seeds = list(D.SEED_REGISTRY[:FROZEN_N])
    _require(report.get("quadruplets") == FROZEN_N
             and report.get("seeds") == frozen_seeds,
             "confirmatory report does not bind exactly the first eight seeds")
    _require(package.get("seed_identifiers", [])[:FROZEN_N] == frozen_seeds,
             "run package seed prefix differs from the frozen eight")
    _require(report.get("factorial_manifest_digest")
             == package.get("factorial_manifest_digest"),
             "confirmatory report and run package bind different factorial manifests")

    initial = _verify_authorisation_receipt(
        INITIAL_RECEIPT_PATH, "initial launch receipt")
    continuation = _verify_authorisation_receipt(
        CONTINUATION_RECEIPT_PATH, "continuation launch receipt")
    for receipt, expected_indices in (
        (initial, list(range(5))),
        (continuation, list(range(5, FROZEN_N))),
    ):
        _require(receipt.get("authorised_seed_indices") == expected_indices,
                 "launch receipt authorises a different seed prefix")
        _require(receipt.get("factorial_manifest_digest")
                 == package.get("factorial_manifest_digest"),
                 "launch receipt factorial digest differs from run package")
        _require(receipt.get("canonical_map_digest")
                 == package.get("canonical_cache_map_digest"),
                 "launch receipt canonical-map digest differs from run package")
        _require(receipt.get("seed_registry_digest")
                 == package.get("seed_registry_sha256"),
                 "launch receipt seed-registry digest differs from run package")
        launch_commit = str(receipt.get("launch_commit", ""))
        _require(bool(launch_commit)
                 and _git("merge-base", "--is-ancestor", launch_commit, "HEAD").returncode == 0,
                 f"launch receipt commit {launch_commit!r} is not an ancestor of HEAD")

    lineage_rows = report.get("attempt_lineage")
    _require(isinstance(lineage_rows, list) and len(lineage_rows) == FROZEN_N,
             "confirmatory report does not carry eight lineage records")
    lineage = {int(row["seed"]): row for row in lineage_rows}
    checkpoints: list[FrozenCheckpoint] = []
    hashing_started = time.time()
    for seed_index, seed in enumerate(frozen_seeds):
        _require(seed in lineage, f"confirmatory report omits seed {seed}")
        frozen = lineage[seed]
        _require(int(frozen.get("seed_index", -1)) == seed_index,
                 f"confirmatory lineage seed index differs for {seed}")
        _require(all(frozen.get(key) is True for key in (
            "completed", "all_cells_valid", "all_cells_24_epochs",
            "all_checkpoints_epoch_21", "shared_parameters_bit_identical",
            "batch_plan_identical_across_cells",
        )), f"confirmatory lineage marks seed {seed} technically invalid")

        seed_dir = D.OUT / f"seed_{seed}"
        run = read_json(seed_dir / "run_record.json", f"seed {seed} run record")
        expected_auth = initial if seed_index < 5 else continuation
        _require(run.get("completed") is True
                 and int(run.get("seed", -1)) == seed
                 and int(run.get("seed_index", -1)) == seed_index,
                 f"seed {seed} run record is incomplete or misidentified")
        _require(run.get("authorisation_receipt_digest")
                 == expected_auth["receipt_digest"]
                 == frozen.get("authorisation_receipt_digest"),
                 f"seed {seed} does not bind its frozen authorisation receipt")
        for run_field, package_field in (
            ("config_sha256", "model_configuration_sha256"),
            ("manifest_sha256", "base_manifest_rows_sha256"),
            ("normalisation_sha256", "normalisation_sha256"),
            ("seed_registry_sha256", "seed_registry_sha256"),
            ("factorial_manifest_digest", "factorial_manifest_digest"),
            ("canonical_map_digest", "canonical_cache_map_digest"),
        ):
            _require(run.get(run_field) == package.get(package_field),
                     f"seed {seed} {run_field} differs from the run package")
        _require(run.get("shared_parameters_bit_identical") is True
                 and run.get("batch_plan_identical_across_cells") is True,
                 f"seed {seed} pairing checks did not pass")
        _require(run.get("budget", {}).get("checkpoint_epoch") == D.CHECKPOINT_EPOCH
                 and run.get("budget", {}).get("selection_permitted") is False,
                 f"seed {seed} violates fixed checkpoint selection")
        _require(run.get("execution_order") == list(D.cell_order(seed_index)),
                 f"seed {seed} execution order differs from frozen schedule")

        cells = run.get("cells_run")
        _require(isinstance(cells, list) and len(cells) == len(D.CELLS),
                 f"seed {seed} does not contain four cells")
        by_cell = {str(cell.get("cell")): cell for cell in cells}
        _require(set(by_cell) == set(D.CELLS), f"seed {seed} cell set differs")
        receipt_path = seed_dir / "checkpoint_receipts.jsonl"
        receipts = read_jsonl_strict(receipt_path, f"seed {seed} checkpoint receipts")
        for cell in D.CELLS:
            cell_record = by_cell[cell]
            checkpoint_path = seed_dir / (
                f"seed_{seed}_{cell}_epoch{D.CHECKPOINT_EPOCH}.pt")
            _require(Path(str(cell_record.get("checkpoint"))).resolve()
                     == checkpoint_path.resolve(),
                     f"seed {seed} {cell} checkpoint path is not canonical")
            _require(cell_record.get("validity") == "valid"
                     and int(cell_record.get("epochs_trained", -1)) == D.EPOCHS
                     and int(cell_record.get("checkpoint_epoch", -1))
                     == D.CHECKPOINT_EPOCH,
                     f"seed {seed} {cell} is not a fixed-budget epoch-21 cell")
            _require(checkpoint_path.is_file(), f"missing checkpoint {checkpoint_path}")
            actual_sha = sha256_file(checkpoint_path)
            run_sha = str(cell_record.get("checkpoint_sha256"))
            report_sha = str(frozen.get("checkpoint_sha256", {}).get(cell))
            _require(actual_sha == run_sha == report_sha,
                     f"seed {seed} {cell} checkpoint hash disagrees across disk/run/report")
            matching = [entry for entry in receipts
                        if Path(str(entry.get("path", ""))).resolve()
                        == checkpoint_path.resolve()
                        and entry.get("sha256") == actual_sha]
            _require(len(matching) == 1,
                     f"seed {seed} {cell} has no unique checkpoint receipt")
            receipt = matching[0]
            _require(int(receipt.get("epoch", -1)) == D.CHECKPOINT_EPOCH
                     and int(receipt.get("bytes", -1)) == checkpoint_path.stat().st_size
                     and receipt.get("verified_reloadable") is True,
                     f"seed {seed} {cell} checkpoint receipt is incomplete")
            checkpoints.append(FrozenCheckpoint(
                seed_index=seed_index,
                seed=seed,
                cell=cell,
                path=checkpoint_path,
                sha256=actual_sha,
                bytes=checkpoint_path.stat().st_size,
                receipt_path=receipt_path,
                authorisation_receipt_digest=expected_auth["receipt_digest"],
            ))

    _require(len(checkpoints) == FROZEN_N * len(D.CELLS),
             "checkpoint inventory is not exactly 32 entries")
    return checkpoints, {
        "confirmatory_commit": FROZEN_CONFIRMATORY_COMMIT,
        "confirmatory_commit_ancestor_of_head": True,
        "confirmatory_report_digest": FROZEN_CONFIRMATORY_REPORT_DIGEST,
        "run_package_digest": FROZEN_RUN_PACKAGE_DIGEST,
        "run_package_sha256": sha256_file(RUN_PACKAGE_PATH),
        "run_package_independent_verifier":
            "scripts.freeze_dev_proprio_run_package_v1.verify",
        "initial_launch_receipt_digest": initial["receipt_digest"],
        "continuation_launch_receipt_digest": continuation["receipt_digest"],
        "frozen_seed_prefix": frozen_seeds,
        "checkpoint_count": len(checkpoints),
        "checkpoint_hash_verification_wall_time_s": round(time.time() - hashing_started, 3),
        "predictor_source_bindings_at_confirmatory_commit": predictor_source_blobs,
        "normalisation_sha256": package["normalisation_sha256"],
    }


# ---------------------------------------------------------- Stage-A corpus --
def _contains_value(value: Any, expected: Any) -> bool:
    if value == expected:
        return True
    if isinstance(value, Mapping):
        return any(_contains_value(item, expected) for item in value.values())
    if isinstance(value, list):
        return any(_contains_value(item, expected) for item in value)
    return False


def _history(value: Any, slots: int, samples: int, dimensions: int,
             label: str) -> tuple[tuple[float, ...], ...]:
    """Canonicalise flat, slot-flat or [slot,sample,dimension] observed histories."""

    array = np.asarray(value, dtype=np.float64)
    _require(array.size == slots * samples * dimensions
             and bool(np.isfinite(array).all()),
             f"{label} must contain {slots*samples*dimensions} finite values")
    array = array.reshape(slots, samples, dimensions)
    return tuple(tuple(float(v) for v in slot.reshape(-1)) for slot in array)


def _candidate_plan(value: Any, label: str) -> tuple[tuple[float, ...], ...]:
    array = np.asarray(value, dtype=np.float64)
    _require(array.shape == (MAX_H, P.ACTION_DIM)
             and bool(np.isfinite(array).all()),
             f"{label} must be finite [4,{P.ACTION_DIM}]")
    return tuple(tuple(float(v) for v in block) for block in array)


def _record_map(records: Any, kind: str, expected_shape: Sequence[int]) \
        -> dict[str, dict[str, Any]]:
    _require(isinstance(records, list), f"latent index {kind}_records is not a list")
    mapped: dict[str, dict[str, Any]] = {}
    for record in records:
        _require(isinstance(record, dict), f"latent {kind} record is not an object")
        _require(record.get("schema")
                 == "go2_counterfactual_fidelity_stage_a_latent_shard_receipt_v1_2"
                 and record.get("record_complete") is True
                 and record.get("kind") == kind,
                 f"latent {kind} record is not a complete shard receipt")
        verify_embedded_digest(record, "latent_shard_receipt_digest",
                               f"latent {kind} shard receipt")
        key = _first(
            record,
            "state_id" if kind == "context" else "branch_identity_digest",
            "branch_key", "identity", "key",
        )
        _require(isinstance(key, str) and key and key not in mapped,
                 f"latent {kind} record has a duplicate or missing identity")
        _require(record.get("shape") == list(expected_shape),
                 f"latent {kind} shard {key} has shape {record.get('shape')}, "
                 f"expected {list(expected_shape)}")
        _require(record.get("dtype") == "float16",
                 f"latent {kind} shard {key} dtype differs")
        digest = record.get("sha256")
        byte_count = _first(record, "byte_count", "bytes")
        _require(isinstance(digest, str) and len(digest) == 64
                 and int(byte_count if byte_count is not None else -1)
                 == int(np.prod(expected_shape)) * 2,
                 f"latent {kind} shard {key} has invalid hash/bytes metadata")
        path = _safe_relative_path(_first(record, "relative_path", "path"),
                                   f"latent {kind} shard {key}")
        sidecar_path = path.with_name(f"{path.name}.receipt.json")
        sidecar = read_json(sidecar_path, f"latent {kind} shard sidecar {key}")
        verify_embedded_digest(sidecar, "latent_shard_receipt_digest",
                               f"latent {kind} shard sidecar {key}")
        _require(sidecar == record,
                 f"latent {kind} index/sidecar receipt differs for {key}")
        stored = dict(record)
        stored["_resolved_path"] = str(path)
        mapped[key] = stored
    return mapped


def _candidate_index(row: Mapping[str, Any]) -> int:
    value = _first(row, "candidate_index", "candidate_id")
    _require(value is not None, "branch row has no candidate_index")
    return int(value)


def _candidate_name(row: Mapping[str, Any]) -> str:
    value = _first(row, "candidate", "candidate_name")
    _require(isinstance(value, str) and value, "branch row has no candidate name")
    return value


def _count_from(receipt: Mapping[str, Any], *names: str) -> int | None:
    for name in names:
        value = receipt.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    for section_name in ("expected", "actual", "counts"):
        section = receipt.get(section_name)
        if isinstance(section, Mapping):
            found = _count_from(section, *names)
            if found is not None:
                return found
    return None


def validate_stage_a_metadata() -> StageABundle:
    """Validate JSON identity/completion metadata without opening latent shards."""

    identity_path = STAGE_A_DIR / "stage_a_identity_manifest.json"
    rows_path = STAGE_A_DIR / "branch_rows.jsonl"
    receipt_path = STAGE_A_DIR / "corpus_receipt.json"
    index_path = STAGE_A_DIR / "latents_index.json"
    original_manifest_path = ROOT / ".generated/go2_oracle_branch_pilot_v1_2/state_manifest.json"
    original_rows_path = ROOT / ".generated/go2_oracle_branch_pilot_v1_2/pilot_branches.jsonl"
    original_gate_path = ROOT / ".generated/go2_oracle_branch_pilot_v1_2/gate_report.json"

    original_manifest = read_json(original_manifest_path, "frozen pilot identity manifest")
    _require(original_manifest.get("state_manifest_digest")
             == legacy_json_digest(original_manifest, ("state_manifest_digest",)),
             "frozen pilot identity manifest state_manifest_digest mismatch")
    _require(original_manifest["state_manifest_digest"]
             == FROZEN_PILOT_IDENTITY_MANIFEST_DIGEST,
             "source pilot identity manifest digest differs")
    original_gate = read_json(original_gate_path, "frozen pilot gate")
    _require(original_gate.get("gate", {}).get("pass") is True
             and original_gate.get("statistics", {}).get("attempted") == EXPECTED_BRANCHES
             and original_gate.get("statistics", {}).get("valid") == EXPECTED_BRANCHES,
             "source pilot gate is not the accepted complete 240-branch gate")
    source_hashes = {
        "state_manifest": sha256_file(original_manifest_path),
        "branch_rows": sha256_file(original_rows_path),
        "gate_report": sha256_file(original_gate_path),
    }

    identity = read_json(identity_path, "Stage-A identity manifest")
    identity_digest = verify_embedded_digest(
        identity, "stage_a_identity_manifest_digest", "Stage-A identity manifest")
    _require(identity_digest == FROZEN_STAGE_A_IDENTITY_MANIFEST_DIGEST,
             "Stage-A identity manifest differs from the prepare-frozen identity")
    _require(identity.get("schema")
             == "go2_counterfactual_fidelity_stage_a_identity_manifest_v1_2",
             "wrong Stage-A identity manifest schema")
    _require(identity.get("complete") is True,
             "Stage-A identity manifest is not complete")
    _require(_contains_value(identity, FROZEN_PILOT_IDENTITY_MANIFEST_DIGEST),
             "Stage-A identity does not bind the accepted pilot identity digest")
    _require(_contains_value(identity, FROZEN_CANDIDATE_BANK_DIGEST),
             "Stage-A identity does not bind the frozen candidate bank")
    _require(_contains_value(identity, FROZEN_ORACLE_V1_2_DIGEST),
             "Stage-A identity does not bind oracle v1.2")
    for label, digest in source_hashes.items():
        _require(_contains_value(identity, digest),
                 f"Stage-A identity does not bind source pilot {label} bytes")
    for field, expected in (
        ("source_state_manifest_digest", FROZEN_PILOT_IDENTITY_MANIFEST_DIGEST),
        ("source_pilot_branch_ledger_sha256",
         FROZEN_SOURCE_PILOT_BRANCH_LEDGER_SHA256),
        ("source_gate_report_sha256", FROZEN_SOURCE_PILOT_GATE_SHA256),
        ("candidate_bank_digest", FROZEN_CANDIDATE_BANK_DIGEST),
        ("oracle_v1_2_digest", FROZEN_ORACLE_V1_2_DIGEST),
        ("genesis_backend", "cpu"),
        ("assay_spec_digest", FROZEN_STAGE_A_ASSAY_SPEC_DIGEST),
        ("render_contract_digest", FROZEN_RENDER_CONTRACT_DIGEST),
        ("textured_v03_renderer_contract_digest", FROZEN_TEXTURED_RENDERER_DIGEST),
        ("preprocess_contract_digest", FROZEN_PREPROCESS_CONTRACT_DIGEST),
        ("preprocessing_digest", FROZEN_PREPROCESSING_DIGEST),
        ("target_encoder_digest", FROZEN_TARGET_ENCODER_DIGEST),
        ("target_encoder_checkpoint_sha256",
         FROZEN_TARGET_ENCODER_CHECKPOINT_SHA256),
        ("target_encoder_checkpoint_byte_count",
         FROZEN_TARGET_ENCODER_CHECKPOINT_BYTES),
        ("state_count_registered", EXPECTED_STATES),
        ("candidate_count_per_state_registered", EXPECTED_CANDIDATES),
        ("attempted_branch_count_registered", EXPECTED_BRANCHES),
    ):
        _require(identity.get(field) == expected,
                 f"Stage-A identity {field} differs from the frozen contract")
    identity_states = identity.get("states")
    _require(isinstance(identity_states, list) and len(identity_states) == EXPECTED_STATES,
             "Stage-A identity must contain exactly 20 registered states")
    state_identity_by_id: dict[str, dict[str, Any]] = {}
    registered_branches: dict[tuple[str, int], dict[str, Any]] = {}
    registered_digests: list[str] = []
    for expected_state_index, registered_state in enumerate(identity_states):
        _require(isinstance(registered_state, dict)
                 and int(registered_state.get("state_index", -1)) == expected_state_index,
                 "Stage-A registered state order/index differs")
        state_id = str(registered_state.get("state_id", ""))
        _require(state_id and state_id not in state_identity_by_id,
                 "Stage-A registered state identity is absent or duplicated")
        state_payload = {
            key: value for key, value in registered_state.items()
            if key not in ("state_identity_digest", "branch_identities")
        }
        _require(registered_state.get("state_identity_digest")
                 == json_digest(state_payload),
                 f"Stage-A state identity digest differs: {state_id}")
        branch_identities = registered_state.get("branch_identities")
        _require(isinstance(branch_identities, list)
                 and len(branch_identities) == EXPECTED_CANDIDATES,
                 f"{state_id} does not register twelve branch identities")
        for candidate_index, branch_identity in enumerate(branch_identities):
            _require(isinstance(branch_identity, dict)
                     and int(branch_identity.get("candidate_index", -1))
                     == candidate_index,
                     f"{state_id} branch identity order differs")
            stored = branch_identity.get("branch_identity_digest")
            _require(isinstance(stored, str)
                     and stored == json_digest(
                         branch_identity, ("branch_identity_digest",)),
                     f"{state_id} branch identity digest differs")
            _require(branch_identity.get("state_id") == state_id
                     and branch_identity.get("state_identity_digest")
                     == registered_state["state_identity_digest"],
                     f"{state_id} branch/state binding differs")
            registered_branches[(state_id, candidate_index)] = branch_identity
            registered_digests.append(stored)
        state_identity_by_id[state_id] = registered_state
    _require(identity.get("branch_identity_set_digest")
             == sequence_digest(sorted(registered_digests)),
             "Stage-A registered branch identity set digest differs")

    receipt = read_json(receipt_path, "Stage-A corpus receipt")
    _require(receipt.get("schema")
             == "go2_counterfactual_fidelity_stage_a_completion_receipt_v1_2",
             "wrong Stage-A receipt schema")
    receipt_digest_field = (
        "completion_receipt_digest" if "completion_receipt_digest" in receipt
        else ("receipt_digest" if "receipt_digest" in receipt
              else "corpus_receipt_digest")
    )
    verify_embedded_digest(receipt, receipt_digest_field, "Stage-A corpus receipt")
    _require(receipt.get("complete") is True, "Stage-A corpus receipt is incomplete")
    _require(_count_from(receipt, "state_count", "states") == EXPECTED_STATES,
             "Stage-A receipt does not certify exactly 20 states")
    _require(_count_from(receipt, "attempted_branch_count", "attempted_branches",
                         "branch_count", "rows") == EXPECTED_BRANCHES,
             "Stage-A receipt does not certify exactly 240 attempted branches")
    _require(_count_from(receipt, "valid_branch_count", "valid_branches", "valid")
             == EXPECTED_BRANCHES,
             "Stage-A receipt does not certify 240 valid branches")
    _require(receipt.get("stage_a_identity_manifest_digest") == identity_digest,
             "Stage-A receipt binds a different identity manifest")
    branch_rows_sha = sha256_file(rows_path)
    _require(receipt.get("branch_rows_sha256") == branch_rows_sha,
             "Stage-A branch ledger bytes differ from receipt")
    corpus_payload = receipt.get("corpus_digest_payload")
    corpus_digest = receipt.get("corpus_digest")
    _require(isinstance(corpus_payload, dict) and isinstance(corpus_digest, str)
             and json_digest(corpus_payload) == corpus_digest,
             "Stage-A corpus digest is not reproducible")

    rows = read_jsonl_strict(rows_path, "Stage-A branch ledger")
    _require(len(rows) == EXPECTED_BRANCHES,
             f"Stage-A branch ledger has {len(rows)} rows, expected 240")
    by_pair: dict[tuple[str, int], dict[str, Any]] = {}
    by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ledger_row_digests: list[str] = []
    for position, row in enumerate(rows):
        _require(row.get("schema")
                 == "go2_counterfactual_fidelity_stage_a_branch_row_v1_2",
                 "wrong Stage-A branch-row schema")
        verify_embedded_digest(row, "branch_row_digest", "Stage-A branch row")
        _require(row.get("record_complete") is True and row.get("valid") is True,
                 "Stage-A branch row is incomplete or invalid")
        _require(row.get("oracle_outcome_equal") is True,
                 "Stage-A branch reexecution differs from accepted outcome")
        _require(row.get("stage_a_identity_manifest_digest") == identity_digest,
                 "Stage-A row does not bind its identity manifest")
        state_id = str(row.get("state_id", ""))
        candidate_index = _candidate_index(row)
        _require(state_id and 0 <= candidate_index < EXPECTED_CANDIDATES,
                 "Stage-A row has invalid state/candidate identity")
        key = (state_id, candidate_index)
        _require(key not in by_pair, f"duplicate Stage-A pair {key}")
        expected_state = identity_states[position // EXPECTED_CANDIDATES]
        expected_candidate = position % EXPECTED_CANDIDATES
        _require(state_id == expected_state["state_id"]
                 and candidate_index == expected_candidate,
                 "Stage-A branch ledger order differs from frozen identity order")
        branch_identity = registered_branches.get(key)
        _require(branch_identity is not None
                 and row.get("branch_identity_digest")
                 == branch_identity["branch_identity_digest"]
                 and row.get("state_identity_digest")
                 == expected_state["state_identity_digest"]
                 and row.get("candidate") == branch_identity["candidate"]
                 and row.get("primitives") == branch_identity["primitives"],
                 f"Stage-A row differs from registered branch identity {key}")
        ledger_row_digests.append(str(row["branch_row_digest"]))
        by_pair[key] = row
        by_state[state_id].append(row)
    _require(len(by_state) == EXPECTED_STATES
             and all(len(group) == EXPECTED_CANDIDATES for group in by_state.values()),
             "Stage-A ledger is not 20 complete twelve-candidate states")
    _require(receipt.get("branch_row_digests") == ledger_row_digests,
             "Stage-A receipt branch-row digest order differs")
    _require(_count_from(receipt, "oracle_equal_branch_count") == EXPECTED_BRANCHES,
             "Stage-A receipt does not certify exact oracle replay for all branches")

    states: list[PlanningState] = []
    candidate_name_by_index: dict[int, str] = {}
    family_counts: Counter[str] = Counter()
    for state_index, (state_id, group) in enumerate(sorted(by_state.items())):
        group = sorted(group, key=_candidate_index)
        _require([_candidate_index(row) for row in group]
                 == list(range(EXPECTED_CANDIDATES)),
                 f"{state_id} candidate indices differ from 0..11")
        first = group[0]
        recorded_state_index = _first(first, "state_index", "position")
        if recorded_state_index is not None:
            state_index = int(recorded_state_index)
        family = str(first.get("family", ""))
        scene = str(_first(first, "scene_id", "scene") or "")
        cluster = str(_first(first, "episode_cluster_id", "episode_cluster") or scene)
        _require(family in FAMILIES and scene and cluster,
                 f"{state_id} family/scene/cluster binding is malformed")
        registered_state = state_identity_by_id[state_id]
        _require(family == registered_state.get("family")
                 and scene == registered_state.get("scene_id")
                 and cluster == registered_state.get("episode_cluster_id")
                 and int(first.get("state_index", -1))
                 == int(registered_state.get("state_index", -2))
                 and first.get("context_key") == state_id,
                 f"{state_id} row/state identity binding differs")
        family_counts[family] += 1
        context = first.get("context_frames")
        action_context = first.get("action_context_blocks")
        proprio_value = _first(first, "proprio", "proprio_history")
        control_value = _first(first, "control", "control_history")
        _require(isinstance(context, list) and len(context) == CONTEXT_SLOTS,
                 f"{state_id} does not carry three observed context frames")
        _require(action_context is not None,
                 f"{state_id} does not carry observed action context")
        _require(first.get("masks") == {
            "context_rgb_valid": [True] * CONTEXT_SLOTS,
            "observed_proprio_valid": [True] * (CONTEXT_SLOTS * SAMPLES_PER_SLOT),
            "observed_control_valid": [True] * (CONTEXT_SLOTS * SAMPLES_PER_SLOT),
            "future_proprio_available": [False] * MAX_H,
            "target_rgb_valid": [True] * MAX_H,
        }, f"{state_id} planning-time validity masks differ")
        _require(first.get("timing") == {
            "command_hz": 10,
            "ticks_per_block": 5,
            "seconds_per_block": 0.5,
            "context_boundary_offsets_blocks": [-2, -1, 0],
            "target_horizons_blocks": [1, 2, 3, 4],
        }, f"{state_id} planning-time timing contract differs")
        proprio = _history(proprio_value, CONTEXT_SLOTS, SAMPLES_PER_SLOT,
                           P.PROPRIO_DIM, f"{state_id} proprio")
        control = _history(control_value, CONTEXT_SLOTS, SAMPLES_PER_SLOT,
                          P.CONTROL_DIM, f"{state_id} control")
        names: list[str] = []
        plans: list[tuple[tuple[float, ...], ...]] = []
        for row in group:
            _require(row.get("family") == family
                     and str(_first(row, "scene_id", "scene")) == scene
                     and str(_first(row, "episode_cluster_id", "episode_cluster") or scene)
                     == cluster,
                     f"{state_id} candidate rows disagree on state identity")
            _require(row.get("context_frames") == context
                     and row.get("action_context_blocks") == action_context
                     and _first(row, "proprio", "proprio_history") == proprio_value
                     and _first(row, "control", "control_history") == control_value
                     and row.get("masks") == first.get("masks")
                     and row.get("timing") == first.get("timing")
                     and row.get("context_key") == state_id
                     and row.get("state_record_digest")
                     == first.get("state_record_digest"),
                     f"{state_id} candidate-specific observed context/history detected")
            candidate_index = _candidate_index(row)
            name = _candidate_name(row)
            if candidate_index in candidate_name_by_index:
                _require(candidate_name_by_index[candidate_index] == name,
                         f"candidate index {candidate_index} has inconsistent names")
            candidate_name_by_index[candidate_index] = name
            names.append(name)
            post_slew = np.asarray(row.get("candidate_post_slew_plan"), dtype=np.float64)
            _require(post_slew.shape == (MAX_H, SAMPLES_PER_SLOT, 3)
                     and bool(np.isfinite(post_slew).all()),
                     f"{state_id}|{name} post-slew plan must be finite [4,5,3]")
            plans.append(_candidate_plan(row.get("action_blocks"),
                                         f"{state_id}|{name} predictor action blocks"))
        states.append(PlanningState(
            state_index=state_index,
            state_id=state_id,
            family=family,
            scene_id=scene,
            episode_cluster_id=cluster,
            context_key=state_id,
            candidate_names=tuple(names),
            candidate_indices=tuple(range(EXPECTED_CANDIDATES)),
            action_blocks=tuple(plans),
            proprio_history=proprio,
            control_history=control,
        ))
    states.sort(key=lambda state: state.state_index)
    _require([state.state_index for state in states] == list(range(EXPECTED_STATES)),
             "Stage-A state indices/order differ from 0..19")
    _require(set(family_counts) == set(FAMILIES)
             and all(count >= 2 for count in family_counts.values()),
             f"Stage-A family coverage differs: {dict(family_counts)}")

    index = read_json(index_path, "Stage-A latent index")
    latent_index_digest = verify_embedded_digest(
        index, "latents_index_digest", "Stage-A latent index")
    _require(index.get("schema")
             == "go2_counterfactual_fidelity_stage_a_latents_index_v1_2",
             "wrong Stage-A latent-index schema")
    _require(index.get("complete") is True,
             "Stage-A latent index is not complete")
    _require(index.get("stage_a_identity_manifest_digest") == identity_digest,
             "Stage-A latent index binds a different identity manifest")
    _require(index.get("branch_rows_sha256") == branch_rows_sha,
             "Stage-A latent index binds a different branch ledger")
    _require(index.get("corpus_digest") == corpus_digest,
             "Stage-A latent index binds a different corpus")
    _require(index.get("context_shape") == [EXPECTED_STATES, CONTEXT_SLOTS, TOKENS, TOKEN_DIM],
             "Stage-A aggregate context shape differs")
    _require(index.get("horizon_shape")
             == [EXPECTED_BRANCHES, MAX_H, TOKENS, TOKEN_DIM],
             "Stage-A aggregate horizon shape differs")
    context_records = _record_map(index.get("context_records"), "context",
                                  (CONTEXT_SLOTS, TOKENS, TOKEN_DIM))
    horizon_records = _record_map(index.get("horizon_records"), "horizon",
                                  (MAX_H, TOKENS, TOKEN_DIM))
    _require(set(context_records) == {state.state_id for state in states},
             "Stage-A context shard identities differ from the 20 states")
    for state in states:
        context_record = context_records[state.state_id]
        context_frames = by_pair[(state.state_id, 0)].get("context_frames")
        _require(isinstance(context_frames, list)
                 and context_record.get("source_frame_set_digest")
                 == sequence_digest(context_frames)
                 and context_record.get("source_frame_sha256")
                 == [frame.get("sha256") for frame in context_frames],
                 f"Stage-A context shard source-frame binding differs: "
                 f"{state.state_id}")
        _require(context_record.get("state_identity_digest")
                 == state_identity_by_id[state.state_id]["state_identity_digest"],
                 f"Stage-A context shard state binding differs: {state.state_id}")
        _require(context_record.get("state_record_digest")
                 == by_pair[(state.state_id, 0)].get("state_record_digest"),
                 f"Stage-A context shard state-record binding differs: {state.state_id}")
    expected_horizon_keys = {
        str(by_pair[(state.state_id, candidate_index)]["branch_identity_digest"])
        for state in states for candidate_index in state.candidate_indices
    }
    _require(set(horizon_records) == expected_horizon_keys,
             "Stage-A horizon shard identities differ from registered branches")
    for (state_id, candidate_index), row in by_pair.items():
        branch_digest = str(row["branch_identity_digest"])
        record = horizon_records[branch_digest]
        horizon_frames = row.get("horizon_frames")
        _require(isinstance(horizon_frames, list)
                 and record.get("source_frame_set_digest")
                 == sequence_digest(horizon_frames)
                 and record.get("source_frame_sha256")
                 == [frame.get("sha256") for frame in horizon_frames],
                 f"Stage-A horizon shard source-frame binding differs: "
                 f"{branch_digest}")
        _require(record.get("branch_identity_digest") == branch_digest
                 and record.get("branch_row_digest") == row["branch_row_digest"]
                 and record.get("branch_key")
                 == f"{state_id}|{row['candidate']}"
                 and record.get("state_id") == state_id
                 and record.get("candidate") == row["candidate"]
                 and int(record.get("candidate_index", -1)) == candidate_index,
                 f"Stage-A horizon shard branch binding differs: {branch_digest}")
    encoder_sha = _first(index, "target_encoder_checkpoint_sha256",
                         "encoder_checkpoint_sha256", "checkpoint_sha256")
    if not isinstance(encoder_sha, str):
        encoder = index.get("encoder", {})
        encoder_sha = _first(encoder, "checkpoint_sha256",
                             "target_encoder_checkpoint_sha256") \
            if isinstance(encoder, Mapping) else None
    _require(isinstance(encoder_sha, str) and len(encoder_sha) == 64,
             "Stage-A latent index does not bind target-encoder weights")
    encoder_identity = index.get("encoder")
    _require(isinstance(encoder_identity, Mapping)
             and encoder_identity.get("checkpoint_sha256")
             == FROZEN_TARGET_ENCODER_CHECKPOINT_SHA256
             and Path(str(encoder_identity.get("checkpoint_path", ""))).resolve()
             == FROZEN_TARGET_ENCODER_CHECKPOINT.resolve(),
             "Stage-A latent-index encoder identity differs")
    _require(index.get("encoder_compute_dtype") == "float32",
             "Stage-A latent encoder compute dtype differs")
    _require(index.get("target_normalisation")
             == FROZEN_LATENT_NORMALISATION_CONTRACT,
             "Stage-A latent index target normalisation differs")
    binding_fields = (
        "render_contract_digest", "textured_v03_renderer_contract_digest",
        "preprocess_contract_digest", "preprocessing_digest",
        "target_encoder_digest", "target_encoder_checkpoint_sha256",
        "candidate_bank_digest", "oracle_v1_2_digest", "assay_spec_digest",
    )
    for binding in binding_fields:
        expected = identity.get(binding)
        _require(expected is not None and receipt.get(binding) == expected
                 and index.get(binding) == expected,
                 f"Stage-A cross-artifact {binding} binding differs")
        for kind, records in (("context", context_records),
                              ("horizon", horizon_records)):
            _require(all(record.get(binding) == expected
                         for record in records.values()),
                     f"Stage-A {kind} shard receipt {binding} binding differs")
    for kind, records in (("context", context_records),
                          ("horizon", horizon_records)):
        _require(all(record.get("encoder_compute_dtype") == "float32"
                     and record.get("target_normalisation")
                     == FROZEN_LATENT_NORMALISATION_CONTRACT
                     for record in records.values()),
                 f"Stage-A {kind} shard numerical contract differs")
    _require(FROZEN_TARGET_ENCODER_CHECKPOINT.is_file()
             and FROZEN_TARGET_ENCODER_CHECKPOINT.stat().st_size
             == FROZEN_TARGET_ENCODER_CHECKPOINT_BYTES,
             "frozen target-encoder checkpoint is missing or has wrong bytes")

    return StageABundle(
        identity=identity,
        receipt=receipt,
        latent_index=index,
        rows=rows,
        states=tuple(states),
        row_by_pair=by_pair,
        context_records=context_records,
        horizon_records=horizon_records,
        identity_digest=identity_digest,
        corpus_digest=corpus_digest,
        branch_rows_sha256=branch_rows_sha,
        latent_index_digest=latent_index_digest,
        target_encoder_checkpoint_sha256=encoder_sha,
    )


def validate_stage_a_latent_shards(bundle: StageABundle) -> dict[str, Any]:
    """Hash all shards after the prospective spec is durable, before lineage/load."""

    started = time.time()
    encoder_sha = sha256_file(FROZEN_TARGET_ENCODER_CHECKPOINT)
    _require(encoder_sha == FROZEN_TARGET_ENCODER_CHECKPOINT_SHA256,
             "frozen target-encoder checkpoint digest differs")
    total_bytes = 0
    aggregate_rows: list[dict[str, Any]] = []
    for kind, records in (("context", bundle.context_records),
                          ("horizon", bundle.horizon_records)):
        for key in sorted(records):
            record = records[key]
            path = Path(record["_resolved_path"])
            _require(path.is_file(), f"missing Stage-A {kind} shard {path}")
            byte_count = int(_first(record, "byte_count", "bytes"))
            _require(path.stat().st_size == byte_count,
                     f"Stage-A {kind} shard {key} byte count differs")
            actual = sha256_file(path)
            _require(actual == record["sha256"],
                     f"Stage-A {kind} shard {key} digest differs")
            total_bytes += byte_count
            aggregate_rows.append({"kind": kind, "key": key,
                                   "sha256": actual, "bytes": byte_count,
                                   "shape": record["shape"]})
    return {
        "complete": True,
        "context_shards": len(bundle.context_records),
        "horizon_shards": len(bundle.horizon_records),
        "bytes": total_bytes,
        "target_encoder_checkpoint_sha256": encoder_sha,
        "target_encoder_checkpoint_bytes": FROZEN_TARGET_ENCODER_CHECKPOINT_BYTES,
        "verified_shard_set_digest": sequence_digest(aggregate_rows),
        "wall_time_s": round(time.time() - started, 3),
    }


def _read_f16_shard(record: Mapping[str, Any]) -> np.ndarray:
    shape = tuple(int(value) for value in record["shape"])
    return np.asarray(np.memmap(Path(str(record["_resolved_path"])), mode="r",
                                dtype=np.float16, shape=shape), dtype=np.float32)


# --------------------------------------------------------- direct scoring --
def load_frozen_normalisation(expected_digest: str) -> dict[str, Any]:
    path = D.PROPRIO / "proprio_norm_stats.json"
    stats = read_json(path, "frozen proprio/control normalisation")
    verify_embedded_digest(stats, "sha256", "frozen proprio/control normalisation")
    _require(stats.get("sha256") == expected_digest,
             "proprio/control normalisation differs from run package")
    for key, length in (("mean", P.PROPRIO_DIM), ("std", P.PROPRIO_DIM),
                        ("control_mean", P.CONTROL_DIM),
                        ("control_std", P.CONTROL_DIM)):
        _require(isinstance(stats.get(key), list) and len(stats[key]) == length
                 and all(_finite(value) for value in stats[key]),
                 f"normalisation field {key} is malformed")
    return stats


@torch.no_grad()
def predict_state(model: P.ProprioActionPredictor, state: PlanningState,
                  context_record: Mapping[str, Any], stats: Mapping[str, Any],
                  use_proprio: bool, device: torch.device) -> np.ndarray:
    """Leakage boundary: observed state + hypotheses only, never a target handle."""

    count = EXPECTED_CANDIDATES
    context_raw = torch.from_numpy(_read_f16_shard(context_record)).float()
    # Exact factorial cache path: raw encoder f16 -> reload float32 -> layer norm.
    context = _normalise(context_raw).half().unsqueeze(0).expand(
        count, -1, -1, -1).to(device=device, dtype=torch.float32)
    proprio_array = np.asarray(state.proprio_history, dtype=np.float32).reshape(
        CONTEXT_SLOTS, SAMPLES_PER_SLOT, P.PROPRIO_DIM)
    control_array = np.asarray(state.control_history, dtype=np.float32).reshape(
        CONTEXT_SLOTS, SAMPLES_PER_SLOT, P.CONTROL_DIM)
    proprio = torch.from_numpy(proprio_array).unsqueeze(0).expand(
        count, -1, -1, -1).to(device)
    control = torch.from_numpy(control_array).unsqueeze(0).expand(
        count, -1, -1, -1).to(device)
    proprio, control = D.normalise_batch(proprio, control, dict(stats), device)
    plans = np.asarray(state.action_blocks, dtype=np.float32)
    _require(plans.shape == (EXPECTED_CANDIDATES, MAX_H, P.ACTION_DIM),
             f"{state.state_id} predictor action tensor shape differs")
    actions = [torch.from_numpy(plans[:, horizon]).to(device)
               for horizon in range(MAX_H)]
    outputs = P.unroll(model, context, actions,
                       proprio if use_proprio else None,
                       control, max_h=MAX_H)
    # Frozen evaluators round rollout outputs to f16 before metric computation.
    predicted = torch.stack([output.half().cpu() for output in outputs], dim=1)
    _require(tuple(predicted.shape)
             == (EXPECTED_CANDIDATES, MAX_H, TOKENS, TOKEN_DIM),
             f"{state.state_id} predicted trajectory shape differs")
    array = predicted.numpy()
    _require(bool(np.isfinite(array).all()),
             f"{state.state_id} predictor emitted non-finite values")
    return array


def score_state_predictions(bundle: StageABundle, state: PlanningState,
                            prediction_f16: np.ndarray,
                            device: torch.device) -> dict[str, Any]:
    """Outcome-side comparison, deliberately separate from ``predict_state``."""

    targets_raw = np.stack([
        _read_f16_shard(bundle.horizon_records[
            str(bundle.row_by_pair[(state.state_id, candidate_index)]
                ["branch_identity_digest"])
        ])
        for candidate_index in state.candidate_indices
    ], axis=0)
    context_raw = _read_f16_shard(bundle.context_records[state.context_key])
    # Exact factorial cache path: raw encoder f16 -> reload float32 -> layer norm.
    target = _normalise(torch.from_numpy(targets_raw).float())
    current = _normalise(torch.from_numpy(context_raw[-1]).float())
    prediction = torch.from_numpy(np.asarray(prediction_f16)).float()
    result: dict[str, Any] = {}
    for horizon_index in range(MAX_H):
        horizon = horizon_index + 1
        pred_h_cpu = prediction[:, horizon_index]
        target_h_cpu = target[:, horizon_index]
        masks = changed_mask(current.unsqueeze(0).expand_as(target_h_cpu),
                             target_h_cpu, horizon)
        # Own-future endpoints remain on CPU exactly as in the frozen evaluator.
        direct = [
            direct_metrics(pred_h_cpu[index], target_h_cpu[index], current, masks[index])
            for index in range(EXPECTED_CANDIDATES)
        ]
        # Full aligned token grid for every (query, gallery) pair.  Target-specific
        # masks are intentionally not used for retrieval comparability.
        pred_h = pred_h_cpu.to(device)
        target_h = target_h_cpu.to(device)
        pred_unit = F.normalize(pred_h, dim=-1)
        target_unit = F.normalize(target_h, dim=-1)
        token_similarity = torch.einsum("itd,jtd->ijt", pred_unit, target_unit)
        similarity = token_similarity.mean(dim=-1).double().cpu().numpy()
        retrieval = retrieval_metrics(similarity, state.candidate_names)
        result[str(horizon)] = {
            "direct": direct,
            "retrieval_similarity_matrix": similarity.tolist(),
            "retrieval": retrieval,
            "changed_rows_available": sum(
                bool(item["changed_metric_available"]) for item in direct),
            "changed_tokens": int(masks.sum()),
        }
        del pred_h_cpu, target_h_cpu, pred_h, target_h, masks
        del pred_unit, target_unit, token_similarity
    return result


def _checkpoint_stem(checkpoint: FrozenCheckpoint) -> str:
    return f"seed_{checkpoint.seed}_{checkpoint.cell}"


def _ledger_paths(checkpoint: FrozenCheckpoint) -> tuple[Path, Path, Path]:
    stem = _checkpoint_stem(checkpoint)
    return (PREDICTION_DIR / f"{stem}.jsonl",
            PREDICTION_DIR / f"{stem}.receipt.json",
            PREDICTION_DIR / stem)


def _ledger_spec(checkpoint: FrozenCheckpoint, bundle: StageABundle,
                 assay_spec_digest: str, normalisation_sha: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "go2_counterfactual_predictor_checkpoint_spec_v1_2",
        "assay_spec_digest": assay_spec_digest,
        "stage_a_identity_manifest_digest": bundle.identity_digest,
        "stage_a_corpus_digest": bundle.corpus_digest,
        "stage_a_latents_index_digest": bundle.latent_index_digest,
        "checkpoint_sha256": checkpoint.sha256,
        "checkpoint_bytes": checkpoint.bytes,
        "checkpoint_epoch": D.CHECKPOINT_EPOCH,
        "seed_index": checkpoint.seed_index,
        "seed": checkpoint.seed,
        "cell": checkpoint.cell,
        "normalisation_sha256": normalisation_sha,
        "states": [
            {"state_index": state.state_index, "state_id": state.state_id,
             "family": state.family, "candidate_names": list(state.candidate_names)}
            for state in bundle.states
        ],
        "prediction_shape_per_state":
            [EXPECTED_CANDIDATES, MAX_H, TOKENS, TOKEN_DIM],
        "prediction_dtype": "float16",
        "inference_batch": EXPECTED_CANDIDATES,
    }
    payload["ledger_spec_digest"] = json_digest(payload)
    return payload


def _preserve_attempt(path: Path, reason: str,
                      recovery: list[dict[str, Any]]) -> None:
    if not path.exists():
        return
    directory = RESULT_DIR / "invalid_attempts"
    directory.mkdir(parents=True, exist_ok=True)
    digest = sha256_file(path) if path.is_file() else "directory"
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    target = directory / f"{path.name}.{stamp}.{digest[:12]}.{reason}"
    counter = 0
    while target.exists():
        counter += 1
        target = directory / f"{path.name}.{stamp}.{digest[:12]}.{reason}.{counter}"
    if path.is_dir():
        shutil.copytree(path, target)
    else:
        shutil.copy2(path, target)
    recovery.append({"source": str(path), "preserved_copy": str(target),
                     "sha256": digest, "reason": reason})


def _prediction_shard_paths(shard_dir: Path, state: PlanningState) \
        -> tuple[Path, Path]:
    stem = f"state_{state.state_index:03d}_{hashlib.sha256(state.state_id.encode()).hexdigest()[:12]}"
    return shard_dir / f"{stem}.f16", shard_dir / f"{stem}.receipt.json"


def _write_prediction_shard(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    shape = (EXPECTED_CANDIDATES, MAX_H, TOKENS, TOKEN_DIM)
    _require(values.shape == shape, "prediction shard has wrong shape")
    memory = np.memmap(temporary, dtype=np.float16, mode="w+", shape=shape)
    memory[:] = values
    memory.flush()
    del memory
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _prediction_shard_receipt(checkpoint: FrozenCheckpoint, state: PlanningState,
                              spec_digest: str, path: Path) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema": "go2_counterfactual_prediction_state_shard_receipt_v1_2",
        "complete": True,
        "ledger_spec_digest": spec_digest,
        "checkpoint_sha256": checkpoint.sha256,
        "seed": checkpoint.seed,
        "cell": checkpoint.cell,
        "state_index": state.state_index,
        "state_id": state.state_id,
        "candidate_names": list(state.candidate_names),
        "relative_path": str(path.relative_to(RESULT_DIR)),
        "sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
        "shape": [EXPECTED_CANDIDATES, MAX_H, TOKENS, TOKEN_DIM],
        "dtype": "float16",
    }
    receipt["receipt_digest"] = json_digest(receipt)
    return receipt


def _validate_prediction_shard(checkpoint: FrozenCheckpoint, state: PlanningState,
                               spec_digest: str, shard_path: Path,
                               receipt_path: Path) -> dict[str, Any] | None:
    if not shard_path.is_file() or not receipt_path.is_file():
        return None
    try:
        receipt = read_json(receipt_path, "prediction state-shard receipt")
        verify_embedded_digest(receipt, "receipt_digest",
                               "prediction state-shard receipt")
        expected_bytes = EXPECTED_CANDIDATES * MAX_H * TOKENS * TOKEN_DIM * 2
        _require(receipt.get("complete") is True
                 and receipt.get("ledger_spec_digest") == spec_digest
                 and receipt.get("checkpoint_sha256") == checkpoint.sha256
                 and int(receipt.get("state_index", -1)) == state.state_index
                 and receipt.get("state_id") == state.state_id
                 and receipt.get("candidate_names") == list(state.candidate_names)
                 and receipt.get("shape")
                 == [EXPECTED_CANDIDATES, MAX_H, TOKENS, TOKEN_DIM]
                 and receipt.get("dtype") == "float16"
                 and int(receipt.get("byte_count", -1)) == expected_bytes
                 and shard_path.stat().st_size == expected_bytes
                 and receipt.get("sha256") == sha256_file(shard_path),
                 "prediction state shard/receipt binding differs")
        return receipt
    except (AssayRefused, OSError, ValueError):
        return None


def _load_prediction_shard(path: Path) -> np.ndarray:
    shape = (EXPECTED_CANDIDATES, MAX_H, TOKENS, TOKEN_DIM)
    return np.asarray(np.memmap(path, mode="r", dtype=np.float16, shape=shape))


def _read_ledger_prefix(path: Path, states: Sequence[PlanningState],
                        spec_digest: str) -> tuple[list[dict[str, Any]], str | None]:
    if not path.is_file():
        return [], None
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                return records, f"malformed_line_{line_number}"
            position = len(records)
            if position >= len(states):
                return records, "extra_record"
            state = states[position]
            if (record.get("ledger_spec_digest") != spec_digest
                    or int(record.get("state_index", -1)) != state.state_index
                    or record.get("state_id") != state.state_id):
                return records, f"binding_mismatch_line_{line_number}"
            stored = record.get("state_result_digest")
            if not isinstance(stored, str) or json_digest(
                    record, ("state_result_digest",)) != stored:
                return records, f"record_digest_mismatch_line_{line_number}"
            records.append(record)
    return records, None


def _checkpoint_receipt(checkpoint: FrozenCheckpoint, spec: Mapping[str, Any],
                        ledger_path: Path, records: Sequence[Mapping[str, Any]],
                        prediction_index: Mapping[str, Any],
                        started: float, resumed_states: int) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema": "go2_counterfactual_predictor_checkpoint_receipt_v1_2",
        "complete": True,
        "ledger_spec_digest": spec["ledger_spec_digest"],
        "assay_spec_digest": spec["assay_spec_digest"],
        "checkpoint_sha256": checkpoint.sha256,
        "checkpoint_bytes": checkpoint.bytes,
        "checkpoint_epoch": D.CHECKPOINT_EPOCH,
        "seed_index": checkpoint.seed_index,
        "seed": checkpoint.seed,
        "cell": checkpoint.cell,
        "states_expected": EXPECTED_STATES,
        "states_completed": len(records),
        "ledger_sha256": sha256_file(ledger_path),
        "state_result_digest_set": sequence_digest(
            [record["state_result_digest"] for record in records]),
        "prediction_shard_receipt_digest_set": sequence_digest(
            [record["prediction_shard"]["receipt_digest"] for record in records]),
        "predictions_index_digest": prediction_index["predictions_index_digest"],
        "wall_time_s_this_invocation": round(time.time() - started, 3),
        "resumed_completed_states": resumed_states,
    }
    receipt["receipt_digest"] = json_digest(receipt)
    return receipt


def _write_prediction_index(checkpoint: FrozenCheckpoint, bundle: StageABundle,
                            spec: Mapping[str, Any], shard_dir: Path,
                            records: Sequence[Mapping[str, Any]],
                            recovery: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    by_state = {str(record["state_id"]): record for record in records}
    branch_records: list[dict[str, Any]] = []
    branch_identity_digests: list[str] = []
    for state in bundle.states:
        state_result = by_state[state.state_id]
        shard = state_result["prediction_shard"]
        for candidate_index, candidate in enumerate(state.candidate_names):
            source_row = bundle.row_by_pair[(state.state_id, candidate_index)]
            branch_digest = source_row.get("branch_identity_digest")
            _require(isinstance(branch_digest, str) and len(branch_digest) == 64,
                     f"{state.state_id}|{candidate} has no branch identity digest")
            branch_identity_digests.append(branch_digest)
            branch_records.append({
                "position": len(branch_records),
                "seed_index": checkpoint.seed_index,
                "seed": checkpoint.seed,
                "cell": checkpoint.cell,
                "checkpoint_sha256": checkpoint.sha256,
                "branch_identity_digest": branch_digest,
                "state_index": state.state_index,
                "state_id": state.state_id,
                "family": state.family,
                "candidate": candidate,
                "candidate_index": candidate_index,
                "relative_path": shard["relative_path"],
                "sha256": shard["sha256"],
                "byte_count": shard["byte_count"],
                "state_shard_shape": shard["shape"],
                "branch_slice": [candidate_index, 0, 0, 0],
                "branch_shape": [MAX_H, TOKENS, TOKEN_DIM],
                "dtype": "float16",
            })
    index: dict[str, Any] = {
        "schema": "go2_counterfactual_predictor_predictions_index_v1_2",
        "complete": True,
        "utility_scorer_used": False,
        "assay_spec_digest": spec["assay_spec_digest"],
        "ledger_spec_digest": spec["ledger_spec_digest"],
        "stage_a_identity_manifest_digest": bundle.identity_digest,
        "stage_a_corpus_digest": bundle.corpus_digest,
        "stage_a_latents_index_digest": bundle.latent_index_digest,
        "scientific_run_package_digest": FROZEN_RUN_PACKAGE_DIGEST,
        "confirmatory_commit": FROZEN_CONFIRMATORY_COMMIT,
        "checkpoint_sha256": checkpoint.sha256,
        "checkpoint_epoch": D.CHECKPOINT_EPOCH,
        "seed_index": checkpoint.seed_index,
        "seed": checkpoint.seed,
        "cell": checkpoint.cell,
        "states": EXPECTED_STATES,
        "branches": EXPECTED_BRANCHES,
        "state_shards": [record["prediction_shard"] for record in records],
        "branch_records": branch_records,
        "ordered_branch_identity_set_digest": sequence_digest(branch_identity_digests),
        "prediction_representation": (
            "autoregressive H1-H4 normalized predictor tokens, rounded float16 "
            "exactly as the frozen direct evaluator"
        ),
        "source_bindings": {
            relative: sha256_file(ROOT / relative)
            for relative in FROZEN_PREDICTOR_SOURCE_PATHS
        },
    }
    index["predictions_index_digest"] = json_digest(index)
    path = shard_dir / "predictions_index.json"
    if path.is_file():
        try:
            existing = read_json(path, "checkpoint predictions index")
            verify_embedded_digest(existing, "predictions_index_digest",
                                   "checkpoint predictions index")
            _require(existing == index,
                     "existing checkpoint predictions index differs")
            return existing
        except (AssayRefused, OSError, ValueError) as exc:
            if recovery is None:
                raise
            _preserve_attempt(path, f"invalid_predictions_index:{exc}", recovery)
    atomic_write_json(path, index)
    return index


def _load_complete_checkpoint_ledger(checkpoint: FrozenCheckpoint,
                                     bundle: StageABundle,
                                     spec: Mapping[str, Any],
                                     recovery: list[dict[str, Any]]) \
        -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    ledger_path, receipt_path, shard_dir = _ledger_paths(checkpoint)
    records, error = _read_ledger_prefix(
        ledger_path, bundle.states, str(spec["ledger_spec_digest"]))
    if error is not None:
        _preserve_attempt(ledger_path, error, recovery)
        if receipt_path.exists():
            _preserve_attempt(receipt_path, error, recovery)
        return None
    if len(records) != EXPECTED_STATES or not receipt_path.is_file():
        return None
    try:
        receipt = read_json(receipt_path, "checkpoint prediction receipt")
        verify_embedded_digest(receipt, "receipt_digest",
                               "checkpoint prediction receipt")
        _require(receipt.get("complete") is True
                 and receipt.get("ledger_spec_digest") == spec["ledger_spec_digest"]
                 and receipt.get("checkpoint_sha256") == checkpoint.sha256
                 and int(receipt.get("states_completed", -1)) == EXPECTED_STATES
                 and receipt.get("ledger_sha256") == sha256_file(ledger_path)
                 and receipt.get("state_result_digest_set") == sequence_digest(
                     [record["state_result_digest"] for record in records]),
                 "checkpoint prediction receipt differs")
        for state, record in zip(bundle.states, records):
            shard_path, shard_receipt_path = _prediction_shard_paths(shard_dir, state)
            shard_receipt = _validate_prediction_shard(
                checkpoint, state, str(spec["ledger_spec_digest"]),
                shard_path, shard_receipt_path)
            _require(shard_receipt is not None
                     and record.get("prediction_shard", {}).get("receipt_digest")
                     == shard_receipt["receipt_digest"],
                     f"completed checkpoint has invalid prediction shard {state.state_id}")
        prediction_index = _write_prediction_index(
            checkpoint, bundle, spec, shard_dir, records, recovery)
        _require(receipt.get("predictions_index_digest")
                 == prediction_index["predictions_index_digest"],
                 "checkpoint receipt binds a different predictions index")
        return records, receipt
    except AssayRefused:
        _preserve_attempt(receipt_path, "invalid_checkpoint_receipt", recovery)
        return None


def score_checkpoint(checkpoint: FrozenCheckpoint, bundle: StageABundle,
                     assay_spec: Mapping[str, Any], normalisation: Mapping[str, Any],
                     device: torch.device, recovery: list[dict[str, Any]]) \
        -> tuple[list[dict[str, Any]], dict[str, Any]]:
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    ledger_path, receipt_path, shard_dir = _ledger_paths(checkpoint)
    spec = _ledger_spec(checkpoint, bundle, str(assay_spec["assay_spec_digest"]),
                        str(normalisation["sha256"]))
    complete = _load_complete_checkpoint_ledger(
        checkpoint, bundle, spec, recovery)
    if complete is not None:
        return complete
    records, error = _read_ledger_prefix(
        ledger_path, bundle.states, str(spec["ledger_spec_digest"]))
    if error is not None:
        _preserve_attempt(ledger_path, error, recovery)
        if records:
            atomic_write_jsonl(ledger_path, records)
        else:
            ledger_path.unlink(missing_ok=True)
    # A JSONL prefix is resumable only through the last digest-verified state
    # shard.  Preserve the original ledger before truncating an invalid tail.
    verified_prefix = 0
    for state, record in zip(bundle.states, records):
        shard_path, shard_receipt_path = _prediction_shard_paths(shard_dir, state)
        shard_receipt = _validate_prediction_shard(
            checkpoint, state, str(spec["ledger_spec_digest"]),
            shard_path, shard_receipt_path)
        if (shard_receipt is None
                or record.get("prediction_shard", {}).get("receipt_digest")
                != shard_receipt.get("receipt_digest")):
            break
        verified_prefix += 1
    if verified_prefix != len(records):
        _preserve_attempt(ledger_path, "invalid_prediction_shard_in_prefix", recovery)
        records = records[:verified_prefix]
        if records:
            atomic_write_jsonl(ledger_path, records)
        else:
            ledger_path.unlink(missing_ok=True)
    resumed = len(records)
    started = time.time()

    # A complete verified ledger whose final receipt was interrupted needs no
    # model load.  Reconstruct its index and receipt from durable state shards.
    if resumed == EXPECTED_STATES:
        prediction_index = _write_prediction_index(
            checkpoint, bundle, spec, shard_dir, records, recovery)
        receipt = _checkpoint_receipt(
            checkpoint, spec, ledger_path, records, prediction_index,
            started, resumed)
        receipt["recovery"] = (
            "complete verified ledger/shards recovered without a valid final receipt"
        )
        receipt.pop("receipt_digest")
        receipt["receipt_digest"] = json_digest(receipt)
        atomic_write_json(receipt_path, receipt)
        return records, receipt

    # All 32 checkpoint bytes have already been verified before this torch.load.
    payload = torch.load(checkpoint.path, map_location="cpu", weights_only=False)
    _require(payload.get("schema") == CK.SCHEMA
             and int(payload.get("epoch", -1)) == D.CHECKPOINT_EPOCH
             and int(payload.get("seed", -1)) == checkpoint.seed,
             f"checkpoint payload metadata differs: {checkpoint.path}")
    expected_cell = D.CELL_SPEC[checkpoint.cell]
    model_config = payload.get("model_config", {})
    _require(model_config.get("cell") == checkpoint.cell
             and model_config.get("use_proprio") == expected_cell["use_proprio"]
             and model_config.get("rollout") == expected_cell["rollout"]
             and int(model_config.get("width", 384)) == 384,
             f"checkpoint payload cell configuration differs: {checkpoint.path}")
    model = P.build_paired(checkpoint.seed,
                           use_proprio=bool(expected_cell["use_proprio"]),
                           width=384, depth=6, heads=6).to(device)
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    except (KeyError, RuntimeError) as exc:
        raise AssayRefused(f"cannot load checkpoint {checkpoint.path}: {exc}") from exc
    del payload
    model.eval()

    shard_dir.mkdir(parents=True, exist_ok=True)
    mode = "a" if records else "w"
    with ledger_path.open(mode, encoding="utf-8") as sink:
        for state in bundle.states[len(records):]:
            shard_path, shard_receipt_path = _prediction_shard_paths(shard_dir, state)
            shard_receipt = _validate_prediction_shard(
                checkpoint, state, str(spec["ledger_spec_digest"]),
                shard_path, shard_receipt_path)
            if shard_receipt is None:
                if shard_path.exists():
                    _preserve_attempt(shard_path, "invalid_prediction_shard", recovery)
                if shard_receipt_path.exists():
                    _preserve_attempt(shard_receipt_path,
                                      "invalid_prediction_shard_receipt", recovery)
                predictions = predict_state(
                    model, state, bundle.context_records[state.context_key],
                    normalisation, bool(expected_cell["use_proprio"]), device)
                _write_prediction_shard(shard_path, predictions)
                shard_receipt = _prediction_shard_receipt(
                    checkpoint, state, str(spec["ledger_spec_digest"]), shard_path)
                atomic_write_json(shard_receipt_path, shard_receipt)
            else:
                predictions = _load_prediction_shard(shard_path)
            per_horizon = score_state_predictions(bundle, state, predictions, device)
            record: dict[str, Any] = {
                "schema": "go2_counterfactual_predictor_state_result_v1_2",
                "ledger_spec_digest": spec["ledger_spec_digest"],
                "state_index": state.state_index,
                "state_id": state.state_id,
                "family": state.family,
                "scene_id": state.scene_id,
                "episode_cluster_id": state.episode_cluster_id,
                "candidate_names": list(state.candidate_names),
                "prediction_shard": shard_receipt,
                "per_horizon": per_horizon,
            }
            record["state_result_digest"] = json_digest(record)
            sink.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            sink.flush()
            os.fsync(sink.fileno())
            records.append(record)
            print(f"[assay] seed {checkpoint.seed} {checkpoint.cell}: "
                  f"{len(records)}/{EXPECTED_STATES}", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    prediction_index = _write_prediction_index(
        checkpoint, bundle, spec, shard_dir, records, recovery)
    receipt = _checkpoint_receipt(checkpoint, spec, ledger_path, records,
                                  prediction_index,
                                  started, resumed)
    atomic_write_json(receipt_path, receipt)
    return records, receipt


# --------------------------------------------------------- aggregation --
DIRECT_METRICS = (
    "changed_cosine",
    "normalised_error_vs_persistence",
    "persistence_changed_cosine",
    "advantage_over_persistence",
    "prediction_mse",
    "persistence_mse",
    "full_token_cosine",
    "full_token_persistence_cosine",
    "full_token_normalised_error_vs_persistence",
)
RETRIEVAL_METRICS = (
    "top1",
    "top3",
    "mean_reciprocal_rank",
    "mean_rank",
    "median_rank",
    "mean_margin_over_best_wrong",
    "mean_margin_over_mean_wrong",
    "pairwise_accuracy",
    "own_wrong_exact_tie_rate",
)
METRIC_DIRECTIONS = {
    "changed_cosine": "higher",
    "normalised_error_vs_persistence": "lower",
    "persistence_changed_cosine": "higher",
    "advantage_over_persistence": "higher",
    "prediction_mse": "lower",
    "persistence_mse": "lower",
    "full_token_cosine": "higher",
    "full_token_persistence_cosine": "higher",
    "full_token_normalised_error_vs_persistence": "lower",
    "retrieval_top1": "higher",
    "retrieval_top3": "higher",
    "retrieval_mean_reciprocal_rank": "higher",
    "retrieval_mean_rank": "lower",
    "retrieval_median_rank": "lower",
    "retrieval_mean_margin_over_best_wrong": "higher",
    "retrieval_mean_margin_over_mean_wrong": "higher",
    "retrieval_pairwise_accuracy": "higher",
    "retrieval_own_wrong_exact_tie_rate": "lower",
}


def _mean_optional(values: Iterable[Any]) -> float | None:
    finite = [float(value) for value in values if _finite(value)]
    return float(np.mean(finite)) if finite else None


def _aggregate_subset(records: Sequence[Mapping[str, Any]], horizon: int) -> dict[str, Any]:
    direct_rows: list[Mapping[str, Any]] = []
    direct_by_state: list[list[Mapping[str, Any]]] = []
    retrieval_rows: list[Mapping[str, Any]] = []
    matrices: list[np.ndarray] = []
    candidate_order: list[str] | None = None
    for record in records:
        block = record["per_horizon"][str(horizon)]
        direct_rows.extend(block["direct"])
        direct_by_state.append(list(block["direct"]))
        retrieval_rows.append(block["retrieval"])
        matrices.append(np.asarray(block["retrieval_similarity_matrix"], dtype=np.float64))
        names = list(record["candidate_names"])
        if candidate_order is None:
            candidate_order = names
        _require(candidate_order == names, "candidate order differs across states")
    _require(candidate_order is not None and records, "cannot aggregate an empty state set")

    # Frozen equal-family path: candidate-row means -> state/episode mean.  The
    # caller then forms family means and the equal-eight-family mean.  This
    # remains state-balanced when a zero-change row makes one candidate's
    # changed-token value unavailable.
    state_means = {
        metric: [_mean_optional(row.get(metric) for row in state_rows)
                 for state_rows in direct_by_state]
        for metric in DIRECT_METRICS
    }
    direct = {metric: _mean_optional(values)
              for metric, values in state_means.items()}
    changed_rows = [row for row in direct_rows if int(row["changed_tokens"]) > 0]
    changed_tokens = sum(int(row["changed_tokens"]) for row in changed_rows)
    direct["rows"] = len(direct_rows)
    direct["changed_rows_available"] = len(changed_rows)
    direct["changed_rows_unavailable"] = len(direct_rows) - len(changed_rows)
    direct["changed_tokens"] = changed_tokens
    direct["states_available_by_metric"] = {
        metric: sum(_finite(value) for value in values)
        for metric, values in state_means.items()
    }
    if changed_tokens:
        direct["token_pooled_changed_cosine"] = sum(
            float(row["changed_cosine"]) * int(row["changed_tokens"])
            for row in changed_rows) / changed_tokens
        direct["token_pooled_persistence_changed_cosine"] = sum(
            float(row["persistence_changed_cosine"]) * int(row["changed_tokens"])
            for row in changed_rows) / changed_tokens
        direct["token_pooled_advantage_over_persistence"] = (
            direct["token_pooled_changed_cosine"]
            - direct["token_pooled_persistence_changed_cosine"]
        )
        prediction_sse = sum(
            float(row["prediction_mse"]) * int(row["changed_tokens"])
            for row in changed_rows)
        persistence_sse = sum(
            float(row["persistence_mse"]) * int(row["changed_tokens"])
            for row in changed_rows)
        direct["token_pooled_prediction_mse"] = prediction_sse / changed_tokens
        direct["token_pooled_persistence_mse"] = persistence_sse / changed_tokens
        direct["token_pooled_normalised_error_vs_persistence"] = (
            prediction_sse / max(persistence_sse, 1e-12)
        )
    else:
        for key in (
            "token_pooled_changed_cosine",
            "token_pooled_persistence_changed_cosine",
            "token_pooled_advantage_over_persistence",
            "token_pooled_prediction_mse",
            "token_pooled_persistence_mse",
            "token_pooled_normalised_error_vs_persistence",
        ):
            direct[key] = None

    all_ranks = [rank for row in retrieval_rows for rank in row["ranks"]]
    confusion = np.sum([np.asarray(row["confusion"], dtype=np.int64)
                        for row in retrieval_rows], axis=0)
    retrieval = {
        metric: _mean_optional(row.get(metric) for row in retrieval_rows)
        for metric in RETRIEVAL_METRICS
    }
    # Corpus median is over every candidate query, not a mean of state medians.
    retrieval["median_rank"] = float(np.median(all_ranks))
    retrieval.update({
        "states": len(records),
        "queries": len(all_ranks),
        "confusion": confusion.tolist(),
        "candidate_order": candidate_order,
        "chance_references": retrieval_rows[0]["chance_references"],
        "similarity_distribution": {
            "mean": float(np.mean(matrices)),
            "standard_deviation": float(np.std(matrices, ddof=0)),
            "minimum": float(np.min(matrices)),
            "maximum": float(np.max(matrices)),
        },
    })
    return {"direct": direct, "retrieval": retrieval}


def aggregate_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(len(records) == EXPECTED_STATES,
             "checkpoint aggregation requires exactly 20 state results")
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for position, record in enumerate(records):
        _require(int(record.get("state_index", -1)) == position,
                 "checkpoint result state order differs")
        family = str(record.get("family", ""))
        _require(family in FAMILIES, f"unknown family {family}")
        by_family[family].append(record)
    _require(set(by_family) == set(FAMILIES)
             and all(len(by_family[family]) >= 2 for family in FAMILIES),
             "checkpoint result lacks frozen family coverage")

    per_horizon: dict[str, Any] = {}
    for horizon in range(1, MAX_H + 1):
        per_family = {
            family: _aggregate_subset(by_family[family], horizon)
            for family in FAMILIES
        }
        equal_direct: dict[str, Any] = {}
        direct_keys = list(DIRECT_METRICS) + [
            "token_pooled_changed_cosine",
            "token_pooled_persistence_changed_cosine",
            "token_pooled_advantage_over_persistence",
            "token_pooled_prediction_mse",
            "token_pooled_persistence_mse",
            "token_pooled_normalised_error_vs_persistence",
        ]
        for metric in direct_keys:
            equal_direct[metric] = _mean_optional(
                per_family[family]["direct"].get(metric) for family in FAMILIES)
        equal_direct["families_available_by_metric"] = {
            metric: sum(_finite(per_family[family]["direct"].get(metric))
                        for family in FAMILIES)
            for metric in direct_keys
        }
        equal_retrieval = {
            metric: _mean_optional(
                per_family[family]["retrieval"].get(metric) for family in FAMILIES)
            for metric in RETRIEVAL_METRICS
        }
        equal_retrieval["chance_references"] = next(iter(per_family.values()))[
            "retrieval"]["chance_references"]
        corpus = _aggregate_subset(records, horizon)
        per_horizon[str(horizon)] = {
            "equal_family": {"direct": equal_direct,
                             "retrieval": equal_retrieval},
            "corpus_weighted": corpus,
            "per_family": per_family,
            "local_composite_motifs_diagnostic":
                per_family[DIAGNOSTIC_FAMILY],
        }
    return {"per_horizon": per_horizon,
            "states": len(records),
            "families": {family: len(by_family[family]) for family in FAMILIES}}


def optional_t_interval(values: Sequence[float | None]) -> dict[str, Any]:
    if len(values) != FROZEN_N:
        raise AssayRefused("optional interval requires eight registered seeds")
    if not all(_finite(value) for value in values):
        return {
            "values": [float(value) if _finite(value) else None for value in values],
            "n_registered": FROZEN_N,
            "n_available": sum(_finite(value) for value in values),
            "estimable": False,
            "reason": "at least one seed value is unavailable; no imputation",
        }
    result = t_interval([float(value) for value in values])
    result["estimable"] = True
    return result


def _metric_value(result: Mapping[str, Any], horizon: int, weighting: str,
                  metric: str, family: str | None = None) -> float | None:
    block = result["per_horizon"][str(horizon)]
    if family is not None:
        source = block["per_family"][family]
    else:
        source = block[weighting]
    if metric.startswith("retrieval_"):
        value = source["retrieval"].get(metric.removeprefix("retrieval_"))
    else:
        corpus_alias = {
            "changed_cosine": "token_pooled_changed_cosine",
            "normalised_error_vs_persistence":
                "token_pooled_normalised_error_vs_persistence",
            "persistence_changed_cosine":
                "token_pooled_persistence_changed_cosine",
            "advantage_over_persistence":
                "token_pooled_advantage_over_persistence",
            "prediction_mse": "token_pooled_prediction_mse",
            "persistence_mse": "token_pooled_persistence_mse",
        }
        source_key = corpus_alias.get(metric, metric) \
            if weighting == "corpus_weighted" and family is None else metric
        value = source["direct"].get(source_key)
    return float(value) if _finite(value) else None


def paired_effects(cells: Mapping[int, Mapping[str, Mapping[str, Any]]],
                   horizon: int, weighting: str, metric: str,
                   direction: str, family: str | None = None) -> dict[str, Any]:
    seeds = list(D.SEED_REGISTRY[:FROZEN_N])
    _require(set(cells) == set(seeds), "paired analysis seed set differs")
    levels: dict[str, list[float | None]] = {cell: [] for cell in D.CELLS}
    b_rgb: list[float | None] = []
    b_prop: list[float | None] = []
    main: list[float | None] = []
    interaction: list[float | None] = []
    sign = 1.0 if direction == "higher" else -1.0
    for seed in seeds:
        values = {
            cell: _metric_value(cells[seed][cell], horizon, weighting, metric, family)
            for cell in D.CELLS
        }
        for cell in D.CELLS:
            levels[cell].append(values[cell])
        if all(_finite(value) for value in values.values()):
            rgb = sign * (float(values["rgb_rollout"])
                          - float(values["rgb_one_step"]))
            prop = sign * (float(values["proprio_rollout"])
                           - float(values["proprio_one_step"]))
            b_rgb.append(rgb)
            b_prop.append(prop)
            main.append((rgb + prop) / 2)
            interaction.append(prop - rgb)
        else:
            b_rgb.append(None); b_prop.append(None); main.append(None); interaction.append(None)
    return {
        "metric": metric,
        "direction": direction,
        "benefit_sign": ("rollout-one_step" if direction == "higher"
                         else "one_step-rollout"),
        "weighting": weighting,
        "family": family,
        "replication_unit": "training-seed quadruplet",
        "cell_levels": {cell: optional_t_interval(values)
                        for cell, values in levels.items()},
        "B_RGB": optional_t_interval(b_rgb),
        "B_prop": optional_t_interval(b_prop),
        "M": optional_t_interval(main),
        "J": optional_t_interval(interaction),
    }


def analyse(cells: Mapping[int, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    analysis: dict[str, Any] = {
        "no_primary_horizon_selected": True,
        "horizons_reported_separately": [1, 2, 3, 4],
        "equal_family": {},
        "corpus_weighted": {},
        "per_family": {},
        "retrieval_confusion_across_seeds": {},
    }
    for horizon in range(1, MAX_H + 1):
        key = f"H{horizon}"
        analysis["equal_family"][key] = {
            metric: paired_effects(cells, horizon, "equal_family", metric, direction)
            for metric, direction in METRIC_DIRECTIONS.items()
        }
        analysis["corpus_weighted"][key] = {
            metric: paired_effects(cells, horizon, "corpus_weighted", metric, direction)
            for metric, direction in METRIC_DIRECTIONS.items()
        }
        analysis["per_family"][key] = {
            family: {
                metric: paired_effects(cells, horizon, "per_family", metric,
                                       direction, family)
                for metric, direction in METRIC_DIRECTIONS.items()
            }
            for family in FAMILIES
        }
        analysis["retrieval_confusion_across_seeds"][key] = {}
        for cell in D.CELLS:
            overall = np.sum([
                np.asarray(cells[seed][cell]["per_horizon"][str(horizon)]
                           ["corpus_weighted"]["retrieval"]["confusion"],
                           dtype=np.int64)
                for seed in D.SEED_REGISTRY[:FROZEN_N]
            ], axis=0)
            per_family_confusion = {
                family: np.sum([
                    np.asarray(cells[seed][cell]["per_horizon"][str(horizon)]
                               ["per_family"][family]["retrieval"]["confusion"],
                               dtype=np.int64)
                    for seed in D.SEED_REGISTRY[:FROZEN_N]
                ], axis=0).tolist()
                for family in FAMILIES
            }
            candidate_order = cells[D.SEED_REGISTRY[0]][cell]["per_horizon"][
                str(horizon)]["corpus_weighted"]["retrieval"]["candidate_order"]
            analysis["retrieval_confusion_across_seeds"][key][cell] = {
                "overall": overall.tolist(),
                "per_family": per_family_confusion,
                "candidate_order": candidate_order,
                "seeds": list(D.SEED_REGISTRY[:FROZEN_N]),
            }
    return analysis


def _checkpoint_inventory(checkpoints: Sequence[FrozenCheckpoint]) -> list[dict[str, Any]]:
    return [{
        "seed_index": checkpoint.seed_index,
        "seed": checkpoint.seed,
        "cell": checkpoint.cell,
        "epoch": D.CHECKPOINT_EPOCH,
        "path": str(checkpoint.path),
        "sha256": checkpoint.sha256,
        "bytes": checkpoint.bytes,
        "checkpoint_receipt": str(checkpoint.receipt_path),
        "authorisation_receipt_digest": checkpoint.authorisation_receipt_digest,
    } for checkpoint in checkpoints]


def _prediction_storage(records_by_checkpoint: Mapping[tuple[int, str],
                                                       Sequence[Mapping[str, Any]]],
                        receipts: Sequence[Mapping[str, Any]]) -> int:
    total = 0
    seen: set[str] = set()
    for (seed, cell), records in records_by_checkpoint.items():
        checkpoint = FrozenCheckpoint(0, seed, cell, Path(), "", 0, Path(), "")
        ledger_path, receipt_path, shard_dir = _ledger_paths(checkpoint)
        for path in (ledger_path, receipt_path, shard_dir / "predictions_index.json"):
            if path.is_file() and str(path) not in seen:
                seen.add(str(path)); total += path.stat().st_size
        for record in records:
            shard = RESULT_DIR / record["prediction_shard"]["relative_path"]
            sidecar = shard.with_suffix(".receipt.json")
            for path in (shard, sidecar):
                if path.is_file() and str(path) not in seen:
                    seen.add(str(path)); total += path.stat().st_size
    return total


def _safe_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def synthetic_self_test() -> dict[str, Any]:
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    prediction = target.clone()
    current = -target
    metric = direct_metrics(prediction, target, current,
                            torch.tensor([True, True]))
    _require(abs(float(metric["changed_cosine"]) - 1.0) < 1e-7,
             "synthetic cosine test failed")
    _require(abs(float(metric["normalised_error_vs_persistence"])) < 1e-7,
             "synthetic normalised-error test failed")
    empty = direct_metrics(prediction, target, current,
                           torch.tensor([False, False]))
    _require(empty["changed_metric_available"] is False
             and empty["changed_cosine"] is None,
             "zero-changed-token handling test failed")
    similarity = np.eye(EXPECTED_CANDIDATES, dtype=np.float64)
    retrieval = retrieval_metrics(
        similarity, [f"candidate_{index}" for index in range(EXPECTED_CANDIDATES)])
    _require(retrieval["top1"] == retrieval["top3"]
             == retrieval["mean_reciprocal_rank"] == 1.0
             and retrieval["mean_rank"] == 1.0
             and retrieval["pairwise_accuracy"] == 1.0,
             "synthetic retrieval test failed")
    interval = t_interval([0.0] * FROZEN_N)
    _require(interval["two_sided_95_t_interval"] == [0.0, 0.0],
             "synthetic t-interval test failed")
    return {"passed": True,
            "direct_metric": metric,
            "zero_change": empty,
            "retrieval": {key: retrieval[key] for key in (
                "top1", "top3", "mean_reciprocal_rank", "mean_rank",
                "pairwise_accuracy", "chance_references")},
            "t_interval": interval}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true",
                        help="run pure synthetic tests; open no scientific artefact")
    parser.add_argument("--validate-only", action="store_true",
                        help="validate Stage A/spec/shards/32 hashes but load no checkpoint")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(synthetic_self_test(), indent=2))
        return 0

    total_started = time.time()
    metadata_started = time.time()
    bundle = validate_stage_a_metadata()
    mask_source = verify_mask_source()
    assay_spec = freeze_assay_spec(bundle, mask_source)
    metadata_wall = time.time() - metadata_started
    shard_verification = validate_stage_a_latent_shards(bundle)
    lineage_started = time.time()
    checkpoints, lineage = verify_frozen_predictor_lineage()
    lineage_wall = time.time() - lineage_started
    if args.validate_only:
        print(json.dumps({
            "validated_only": True,
            "assay_spec_digest": assay_spec["assay_spec_digest"],
            "stage_a_identity_manifest_digest": bundle.identity_digest,
            "stage_a_corpus_digest": bundle.corpus_digest,
            "stage_a_latents_index_digest": bundle.latent_index_digest,
            "latent_shards": shard_verification,
            "checkpoint_count": len(checkpoints),
            "predictor_checkpoints_loaded": 0,
        }, indent=2))
        return 0

    normalisation = load_frozen_normalisation(str(lineage["normalisation_sha256"]))
    device = D.resolve_device()
    recovery: list[dict[str, Any]] = []
    cells: dict[int, dict[str, dict[str, Any]]] = {
        seed: {} for seed in D.SEED_REGISTRY[:FROZEN_N]
    }
    records_by_checkpoint: dict[tuple[int, str], list[dict[str, Any]]] = {}
    checkpoint_receipts: list[dict[str, Any]] = []
    checkpoint_map = {(checkpoint.seed, checkpoint.cell): checkpoint
                      for checkpoint in checkpoints}
    scoring_started = time.time()
    for seed in D.SEED_REGISTRY[:FROZEN_N]:
        for cell in D.CELLS:
            checkpoint = checkpoint_map[(seed, cell)]
            records, receipt = score_checkpoint(
                checkpoint, bundle, assay_spec, normalisation, device, recovery)
            records_by_checkpoint[(seed, cell)] = records
            cells[seed][cell] = aggregate_records(records)
            checkpoint_receipts.append(receipt)
    scoring_wall = time.time() - scoring_started
    analysis_started = time.time()
    final_analysis = analyse(cells)
    analysis_wall = time.time() - analysis_started

    checkpoint_list = _checkpoint_inventory(checkpoints)
    report: dict[str, Any] = {
        "schema": "go2_counterfactual_predictor_qualification_result_v1_2",
        "status": STATUS,
        "complete": True,
        "utility_scorer_used": False,
        "assay_spec_digest": assay_spec["assay_spec_digest"],
        "prospective_assay_spec": assay_spec,
        "sequential_gate": {
            "stage_a_complete_before_lineage_verification": True,
            "stage_a_latent_shards_hash_verified_before_lineage_verification": True,
            "all_32_checkpoint_hashes_verified_before_first_torch_load": True,
            "no_scientific_checkpoint_loaded_for_stage_a_validation": True,
            "verdict": "PASS",
        },
        "stage_a": {
            "identity_manifest_digest": bundle.identity_digest,
            "corpus_digest": bundle.corpus_digest,
            "branch_rows_sha256": bundle.branch_rows_sha256,
            "latents_index_digest": bundle.latent_index_digest,
            "states": EXPECTED_STATES,
            "branches": EXPECTED_BRANCHES,
            "candidate_count": EXPECTED_CANDIDATES,
            "families": Counter(state.family for state in bundle.states),
            "target_encoder_checkpoint_sha256":
                bundle.target_encoder_checkpoint_sha256,
            "latent_shard_verification": shard_verification,
            "generation_runtime_reported": {
                "completed_branch_wall_time_s":
                    bundle.receipt.get("runtime_s_completed_rows"),
                "completion_invocation_wall_time_s":
                    bundle.receipt.get("runtime_s_this_invocation"),
            },
            "generation_storage_reported_bytes":
                bundle.receipt.get("storage_bytes"),
        },
        "predictor_provenance": lineage,
        "verified_checkpoints": checkpoint_list,
        "checkpoint_prediction_receipts": checkpoint_receipts,
        "predictor_input_separation": {
            "implementation": "PlanningState allow-list + predict_state signature",
            "observed_context_only": True,
            "candidate_post_slew_actions_only": True,
            "shared_observed_control_history": True,
            "observed_proprioception_only_for_proprio_cells": True,
            "future_proprioception_supplied": False,
            "target_latents_supplied_to_predictor": False,
            "future_rgb_supplied": False,
            "oracle_labels_or_outcomes_supplied": False,
            "target_access_separated_into_score_state_predictions": True,
        },
        "weighting": {
            "equal_family_primary_at_each_horizon": True,
            "corpus_weighted_separate": True,
            "horizons_combined": False,
            "primary_horizon_selected": False,
            "families": list(FAMILIES),
        },
        "cells_by_seed": cells,
        "paired_seed_analysis": final_analysis,
        "chance_references": {
            "candidates": EXPECTED_CANDIDATES,
            "top1": 1 / 12,
            "top3": 3 / 12,
            "mean_reciprocal_rank": sum(1 / rank for rank in range(1, 13)) / 12,
            "mean_rank": 6.5,
            "pairwise_accuracy": 0.5,
        },
        "recovery": {
            "events": recovery,
            "invalid_or_interrupted_attempts_preserved": bool(recovery),
            "exact_resume_unit": "checkpoint x state (all twelve candidates)",
        },
        "runtime": {
            "stage_a_metadata_and_spec_wall_time_s": round(metadata_wall, 3),
            "stage_a_shard_hash_verification_wall_time_s":
                shard_verification["wall_time_s"],
            "checkpoint_lineage_hash_verification_wall_time_s": round(lineage_wall, 3),
            "predictor_scoring_wall_time_s": round(scoring_wall, 3),
            "analysis_wall_time_s": round(analysis_wall, 3),
            "total_wall_time_s": round(time.time() - total_started, 3),
        },
        "storage": {
            "stage_a_bound_json_and_ledger_bytes": sum(
                path.stat().st_size for path in (
                    STAGE_A_DIR / "stage_a_identity_manifest.json",
                    STAGE_A_DIR / "branch_rows.jsonl",
                    STAGE_A_DIR / "corpus_receipt.json",
                    STAGE_A_DIR / "latents_index.json",
                ) if path.is_file()),
            "stage_a_rendered_branch_storage_reported_bytes":
                bundle.receipt.get("storage_bytes"),
            "stage_a_latent_bytes": shard_verification["bytes"],
            "target_encoder_checkpoint_bytes_read_only":
                FROZEN_TARGET_ENCODER_CHECKPOINT_BYTES,
            "frozen_checkpoint_bytes_read_only":
                sum(checkpoint.bytes for checkpoint in checkpoints),
            "prediction_ledgers_shards_indices_receipts_bytes":
                _prediction_storage(records_by_checkpoint, checkpoint_receipts),
            "scope_note": "explicit bound files only; no recursive custody-root traversal",
        },
    }
    report = _safe_json(report)
    report["report_digest"] = json_digest(report)
    if RESULT_PATH.is_file():
        try:
            existing = read_json(RESULT_PATH, "existing direct assay result")
            verify_embedded_digest(existing, "report_digest",
                                   "existing direct assay result")
        except (AssayRefused, OSError, ValueError) as exc:
            _preserve_attempt(RESULT_PATH, f"invalid_final_result:{exc}", recovery)
            report["recovery"] = _safe_json({
                "events": recovery,
                "invalid_or_interrupted_attempts_preserved": bool(recovery),
                "exact_resume_unit": "checkpoint x state (all twelve candidates)",
            })
            report["runtime"]["total_wall_time_s"] = round(
                time.time() - total_started, 3)
            report.pop("report_digest")
            report["report_digest"] = json_digest(report)
        else:
            immutable_keys = (
                "assay_spec_digest", "stage_a", "verified_checkpoints",
                "cells_by_seed", "paired_seed_analysis", "chance_references",
            )
            _require(all(existing.get(key) == report.get(key) for key in immutable_keys),
                     "existing complete direct-assay result differs from recomputation")
            print(json.dumps({"reused_complete_result": True,
                              "report_digest": existing["report_digest"],
                              "result": str(RESULT_PATH)}, indent=2))
            return 0
    atomic_write_json(RESULT_PATH, report)
    print(json.dumps({
        "complete": True,
        "report_digest": report["report_digest"],
        "assay_spec_digest": report["assay_spec_digest"],
        "checkpoints": len(checkpoints),
        "result": str(RESULT_PATH),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
