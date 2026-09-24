#!/usr/bin/env python3
"""Run the frozen oracle-v1.3 scorer latent-dependence diagnostic.

This is a descriptive, post-failure diagnostic.  It validates and reuses the
frozen failed ViT-L scorer, applies the preregistered A--G transformations to
the full stored ``[4, 768, 1024]`` trajectories *before* the existing spatial
mean, evaluates the already examined development calibration split once, and
stops.  It cannot train, publish a scorer, open predictor material, or reach a
final benchmark.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lewm.oracle import go2_scorer_failure_attribution_v1_contract as CONTRACT  # noqa: E402


STATUS = CONTRACT.STATUS

SCHEMA = "go2_scorer_v1_3_latent_dependence_diagnostic_result_v1"
AUTHORISATION_SCHEMA = (
    "go2_scorer_v1_3_latent_dependence_evaluation_authorisation_v1")
TECHNICAL_FAILURE_SCHEMA = (
    "go2_scorer_v1_3_latent_dependence_technical_failure_v1")
RESULT_SELF_KEY = "latent_dependence_result_digest"
AUTHORISATION_SELF_KEY = "evaluation_authorisation_digest"
TECHNICAL_FAILURE_SELF_KEY = "technical_failure_digest"

HORIZONS = CONTRACT.HORIZONS
TOKENS = CONTRACT.TOKENS
TOKEN_DIM = CONTRACT.TOKEN_DIM
RAW_SHAPE = (HORIZONS, TOKENS, TOKEN_DIM)
FIT_ROWS = CONTRACT.FIT_ROWS
CALIBRATION_ROWS = CONTRACT.FRESH_CALIBRATION_ROWS
CANDIDATES_PER_STATE = 12
INVARIANCE_ATOL = float(getattr(CONTRACT, "INVARIANCE_ATOL", 2e-6))
MATCHED_TERMINAL_REPLAY_ATOL = 1e-6

A_MATCHED = "A_matched"
B_WITHIN_STATE_DERANGEMENT = "B_within_state_candidate_derangement"
C_HORIZON_REVERSED = "C_horizon_reversed"
D_TOKEN_PERMUTED = "D_fixed_token_permutation"
E_SPATIAL_MEAN_REPEATED = "E_spatial_mean_repeated"
F_FIT_MEAN_TRAJECTORY = "F_fit_mean_trajectory"
G_SINGLE_HORIZON = tuple(f"G_H{horizon}_only" for horizon in range(1, 5))
VARIANT_IDS = (
    A_MATCHED,
    B_WITHIN_STATE_DERANGEMENT,
    C_HORIZON_REVERSED,
    D_TOKEN_PERMUTED,
    E_SPATIAL_MEAN_REPEATED,
    F_FIT_MEAN_TRAJECTORY,
    *G_SINGLE_HORIZON,
)
RECOMPUTED_RESULT_FIELDS = (
    "results",
    "architecture_invariance_checks",
    "matched_condition_terminal_replay",
    "frozen_no_latent_baseline",
)
TRANSFORMATIONS = CONTRACT.TRANSFORMATION_SUITE
SAMPLE_TICKS = (1, 288, 576, 864, FIT_ROWS)

# Literal predecessor identities intentionally avoid replaying the historical
# exact-source-diff validator after this new diagnostic source is committed.
# The protected V1.3 validator itself is never weakened or modified.
FROZEN = {
    "qualification_report_digest":
        CONTRACT.FROZEN_VITL_QUALIFICATION_TERMINAL_DIGEST,
    "training_view_digest":
        CONTRACT.FROZEN_TRAINING_VIEW_DIGEST,
    "latent_index_digest":
        CONTRACT.FROZEN_LATENT_INDEX_DIGEST,
    "encoding_receipt_digest":
        "91bb919714c4cd0dc19da988b80dc10612715a0a64d6bd1b553c84d41cdefb5a",
    "target_encoder_digest":
        CONTRACT.FROZEN_TARGET_ENCODER_DIGEST,
    "target_encoder_checkpoint_sha256":
        CONTRACT.FROZEN_TARGET_ENCODER_CHECKPOINT_SHA256,
    "preprocessing_digest":
        "8e6aa177b094ea91d27b3c91bcd8f01835b8be5fc51796d145314982ea930fe5",
    "training_run_digest":
        "f9d9f2d78360f5155596e6eebfacadad4aa47afb21f2b5bfcf0a5637708622b7",
    "binding_digest":
        "c783a16d28d3770f0dc253633aabd4af45d543122b9dbb20190334bd0ce2e7e5",
    "latent_checkpoint_sha256":
        CONTRACT.FROZEN_VITL_FINAL_CHECKPOINT_SHA256,
    "latent_initial_state_digest":
        "f74eba729b0f9fbeb9cdb502a3c5f6bf239bc8ba500a5398f59a91bc2c4dead5",
    "latent_final_state_digest":
        CONTRACT.FROZEN_VITL_FINAL_STATE_DIGEST,
    "failed_scorer_sha256":
        CONTRACT.FROZEN_VITL_FAILURE_ARTIFACT_SHA256,
    "baseline_checkpoint_sha256":
        CONTRACT.FROZEN_BASELINE_CHECKPOINT_SHA256,
    "baseline_state_digest":
        CONTRACT.FROZEN_BASELINE_STATE_DIGEST,
    "baseline_receipt_digest":
        CONTRACT.FROZEN_BASELINE_RECEIPT_DIGEST,
}


class LatentDependenceError(RuntimeError):
    """The frozen predecessor or diagnostic contract changed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LatentDependenceError(message)


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def file_sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _signed(value: Mapping[str, Any], self_key: str) -> dict[str, Any]:
    payload = dict(value)
    _require(self_key not in payload, f"{self_key} was already present")
    payload[self_key] = canonical_digest(payload)
    return payload


def _validate_signed(value: Mapping[str, Any], self_key: str,
                     label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} is not an object")
    payload = dict(value)
    recorded = payload.pop(self_key, None)
    _require(isinstance(recorded, str) and len(recorded) == 64,
             f"{label} self digest is malformed")
    _require(recorded == canonical_digest(payload),
             f"{label} self digest does not verify")
    payload[self_key] = recorded
    return payload


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True,
                       allow_nan=False) + "\n").encode("utf-8")


def publish_json_once(path: Path, value: Mapping[str, Any], *, label: str) -> None:
    raw = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        _require(path.is_file() and not path.is_symlink(),
                 f"{label} path is not a regular file")
        _require(path.read_bytes() == raw, f"{label} is already different")
        return
    try:
        position = 0
        while position < len(raw):
            position += os.write(descriptor, raw[position:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def managed_generated_root(root: Path = ROOT) -> Path:
    alias = (root / CONTRACT.GENERATED_ROOT).absolute()
    if root.resolve() == ROOT.resolve():
        _require(alias.is_symlink(),
                 "registered failure-attribution output alias is absent")
        try:
            target = alias.resolve(strict=True)
        except OSError as exc:
            raise LatentDependenceError(
                "registered failure-attribution output alias is broken") from exc
        registered = CONTRACT.REGISTERED_GENERATED_TARGET_ROOT.absolute()
        _require(target == registered and registered.is_dir()
                 and not registered.is_symlink(),
                 "registered failure-attribution output target changed")
    else:
        _require(alias.is_dir() and not alias.is_symlink(),
                 "synthetic failure-attribution output root is invalid")
    return alias


def diagnostic_contract_path(root: Path = ROOT) -> Path:
    return managed_generated_root(root) / "diagnostic_contract.json"


def diagnostic_root(root: Path = ROOT) -> Path:
    return managed_generated_root(root) / "latent_dependence"


def evaluation_authorisation_path(root: Path = ROOT) -> Path:
    return diagnostic_root(root) / "evaluation_authorisation.json"


def result_path(root: Path = ROOT) -> Path:
    return diagnostic_root(root) / "result.json"


def technical_failure_path(root: Path = ROOT) -> Path:
    return diagnostic_root(root) / "technical_failure.json"


def within_state_derangement(
        rows: Sequence[Mapping[str, Any]], *,
        candidates_per_state: int = CANDIDATES_PER_STATE,
        ) -> tuple[np.ndarray, dict[str, Any]]:
    """Return destination->source positions for fixed rotate-one hash order."""

    groups: dict[str, list[int]] = {}
    for position, row in enumerate(rows):
        groups.setdefault(str(row["state_id"]), []).append(position)
    mapping = np.full(len(rows), -1, dtype=np.int64)
    receipt_groups = []
    for state_id in sorted(groups):
        positions = groups[state_id]
        _require(len(positions) == candidates_per_state,
                 f"state {state_id} does not have {candidates_per_state} rows")
        candidates = [int(rows[position]["candidate_index"])
                      for position in positions]
        _require(sorted(candidates) == list(range(candidates_per_state)),
                 f"state {state_id} candidate bank changed")
        state_digests = {str(rows[position]["state_identity_digest"])
                         for position in positions}
        _require(len(state_digests) == 1,
                 f"state {state_id} has multiple state identities")
        by_branch = {
            str(rows[position]["branch_identity_digest"]): position
            for position in positions
        }
        _require(len(by_branch) == candidates_per_state,
                 f"state {state_id} branch identities are duplicated")
        contracted = CONTRACT.within_state_candidate_derangement(
            next(iter(state_digests)), list(by_branch))
        pairs = []
        for destination_digest in sorted(contracted):
            source_digest = contracted[destination_digest]
            destination = by_branch[destination_digest]
            source = by_branch[source_digest]
            _require(destination != source, "derangement retained one row")
            mapping[destination] = source
            pairs.append({
                "destination_branch_identity_digest": destination_digest,
                "destination_training_view_row_digest":
                    rows[destination]["training_view_row_digest"],
                "destination_candidate_index":
                    int(rows[destination]["candidate_index"]),
                "source_branch_identity_digest": source_digest,
                "source_training_view_row_digest":
                    rows[source]["training_view_row_digest"],
                "source_candidate_index": int(rows[source]["candidate_index"]),
            })
        receipt_groups.append({"state_id": state_id, "pairs": pairs})
    _require(bool(np.all(mapping >= 0))
             and sorted(mapping.tolist()) == list(range(len(rows))),
             "within-state mapping is not a complete permutation")
    receipt = {
        "algorithm": TRANSFORMATIONS[
            "B_WITHIN_STATE_CANDIDATE_DERANGEMENT"]["algorithm"],
        "namespace": CONTRACT.DERANGEMENT_NAMESPACE,
        "rows": len(rows),
        "states": len(groups),
        "candidates_per_state": candidates_per_state,
        "groups": receipt_groups,
    }
    receipt["mapping_digest"] = canonical_digest(receipt)
    return mapping, receipt


def fixed_token_permutation() -> tuple[np.ndarray, dict[str, Any]]:
    """One source-bound permutation, shared by all rows and horizons."""

    permutation = np.asarray(CONTRACT.SPATIAL_TOKEN_PERMUTATION,
                             dtype=np.int64)
    _require(sorted(permutation.tolist()) == list(range(TOKENS)),
             "token mapping is not a permutation")
    _require(not np.array_equal(permutation, np.arange(TOKENS)),
             "hash-derived token permutation is unexpectedly identity")
    receipt = {
        "algorithm": "contract_sha256_sort_token_indices",
        "namespace": CONTRACT.SPATIAL_PERMUTATION_NAMESPACE,
        "tokens": TOKENS,
        "permutation": permutation.tolist(),
        "permutation_tensor_digest": array_digest(permutation),
        "contract_permutation_digest":
            CONTRACT.SPATIAL_TOKEN_PERMUTATION_DIGEST,
    }
    receipt["permutation_digest"] = canonical_digest(receipt)
    return permutation, receipt


def spatial_mean(raw: np.ndarray) -> np.ndarray:
    value = np.asarray(raw, dtype=np.float32)
    _require(value.ndim == 3, "raw latent trajectory is not rank three")
    return value.mean(axis=1, dtype=np.float32)


def apply_raw_transform(
        variant: str, raw: np.ndarray, *, token_permutation: np.ndarray,
        fit_mean_trajectory: np.ndarray) -> np.ndarray:
    """Apply an A--G transform before the scorer's existing spatial mean."""

    value = np.asarray(raw, dtype=np.float32)
    fit_mean = np.asarray(fit_mean_trajectory, dtype=np.float32)
    _require(value.ndim == 3 and fit_mean.shape == value.shape,
             "raw and fit-mean trajectory shapes differ")
    _require(value.shape[0] == HORIZONS,
             "latent trajectory no longer has four horizons")
    if variant in {A_MATCHED, B_WITHIN_STATE_DERANGEMENT}:
        return value
    if variant == C_HORIZON_REVERSED:
        return value[::-1, :, :]
    if variant == D_TOKEN_PERMUTED:
        _require(len(token_permutation) == value.shape[1],
                 "token permutation width changed")
        return value[:, token_permutation, :]
    if variant == E_SPATIAL_MEAN_REPEATED:
        mean = value.mean(axis=1, dtype=np.float32)
        return np.repeat(mean[:, None, :], value.shape[1], axis=1)
    if variant == F_FIT_MEAN_TRAJECTORY:
        return fit_mean
    if variant in G_SINGLE_HORIZON:
        keep = G_SINGLE_HORIZON.index(variant)
        result = fit_mean.copy()
        result[keep] = value[keep]
        return result
    raise LatentDependenceError(f"unknown latent transformation {variant}")


def compute_fit_mean_trajectory(
        rows: Sequence[Mapping[str, Any]], horizon: Any, *,
        raw_shape: tuple[int, int, int] = RAW_SHAPE,
        expected_rows: int = FIT_ROWS,
        sample_ticks: Sequence[int] = SAMPLE_TICKS,
        ) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute the fit-only full-trajectory mean in one frozen row order."""

    _require(len(rows) == expected_rows,
             "fit mean received the wrong number of rows")
    ticks = tuple(sorted(set(int(tick) for tick in sample_ticks)))
    _require(all(1 <= tick <= expected_rows for tick in ticks),
             "fit-mean sample tick is outside the fit row count")
    accumulator = np.zeros(raw_shape, dtype=np.float64)
    samples = []
    row_digests = []
    ordered_rows = sorted(rows, key=lambda row: (
        str(row["state_id"]), int(row["candidate_index"]),
        str(row["branch_identity_digest"])))
    for count, row in enumerate(ordered_rows, 1):
        raw = np.asarray(horizon[int(row["_latent_index"])], dtype=np.float64)
        _require(raw.shape == raw_shape,
                 "fit latent trajectory shape changed")
        accumulator += raw
        row_digests.append(str(row["training_view_row_digest"]))
        if count in ticks:
            samples.append({
                "rows_accumulated": count,
                "float64_accumulator_digest": array_digest(accumulator),
            })
    mean = (accumulator / float(expected_rows)).astype(np.float32)
    receipt = {
        "algorithm": "contract_fit_row_order_float64_streaming_mean",
        "contract": dict(CONTRACT.FIT_MEAN_TRAJECTORY_CONTRACT),
        "fit_rows": expected_rows,
        "raw_shape": list(raw_shape),
        "fit_row_digest_sequence_digest": canonical_digest(row_digests),
        "sample_ticks": list(ticks),
        "samples": samples,
        "fit_mean_trajectory_digest": array_digest(mean),
    }
    receipt["fit_mean_receipt_digest"] = canonical_digest(receipt)
    return mean, receipt


def materialise_variant_spatial_means(
        rows: Sequence[Mapping[str, Any]], horizon: Any, *,
        derangement: np.ndarray, token_permutation: np.ndarray,
        fit_mean_trajectory: np.ndarray,
        ) -> dict[str, np.ndarray]:
    """Open calibration shards once per destination and emit compact means."""

    _require(len(rows) == len(derangement),
             "derangement and calibration rows differ")
    result = {
        variant: np.empty((len(rows), HORIZONS, TOKEN_DIM), dtype=np.float32)
        for variant in VARIANT_IDS
    }
    for destination, row in enumerate(rows):
        raw = np.asarray(horizon[int(row["_latent_index"])], dtype=np.float32)
        _require(raw.shape == RAW_SHAPE,
                 "calibration latent trajectory shape changed")
        source_row = rows[int(derangement[destination])]
        deranged_raw = np.asarray(
            horizon[int(source_row["_latent_index"])], dtype=np.float32)
        _require(deranged_raw.shape == RAW_SHAPE,
                 "deranged calibration trajectory shape changed")
        for variant in VARIANT_IDS:
            selected = deranged_raw if variant == B_WITHIN_STATE_DERANGEMENT else raw
            transformed = apply_raw_transform(
                variant, selected, token_permutation=token_permutation,
                fit_mean_trajectory=fit_mean_trajectory)
            result[variant][destination] = spatial_mean(transformed)
    return result


def metric_delta(value: Any, reference: Any) -> Any:
    """Recursively subtract complete metric trees, aligning per-state rows."""

    if isinstance(value, Mapping) and isinstance(reference, Mapping):
        result = {}
        for key in value:
            if key == "state_id":
                result[key] = value[key]
            elif key in reference:
                delta = metric_delta(value[key], reference[key])
                if delta is not None:
                    result[key] = delta
        return result
    if isinstance(value, list) and isinstance(reference, list):
        if (all(isinstance(item, Mapping) and "state_id" in item for item in value)
                and all(isinstance(item, Mapping) and "state_id" in item
                        for item in reference)):
            by_state = {str(item["state_id"]): item for item in reference}
            return [metric_delta(item, by_state[str(item["state_id"])])
                    for item in value if str(item["state_id"]) in by_state]
        return [metric_delta(left, right)
                for left, right in zip(value, reference)]
    if (not isinstance(value, bool) and isinstance(value, (int, float))
            and not isinstance(reference, bool)
            and isinstance(reference, (int, float))):
        if math.isfinite(float(value)) and math.isfinite(float(reference)):
            return float(value) - float(reference)
        return None
    return None


def prediction_invariance_error(
        matched: Mapping[str, np.ndarray], transformed: Mapping[str, np.ndarray],
        *, atol: float = INVARIANCE_ATOL) -> dict[str, Any]:
    values = {}
    for key in ("progress", "safety", "completion", "utility"):
        difference = np.abs(np.asarray(transformed[key], dtype=np.float64)
                            - np.asarray(matched[key], dtype=np.float64))
        maximum = float(difference.max(initial=0.0))
        values[key] = {
            "max_abs_error": maximum,
            "mean_abs_error": float(difference.mean()) if difference.size else 0.0,
            "within_absolute_tolerance": maximum <= atol,
        }
    return {
        "absolute_tolerance": atol,
        "heads": values,
        "all_within_absolute_tolerance": all(
            value["within_absolute_tolerance"] for value in values.values()),
    }


def _resolve_recorded_path(raw: Any, *, root: Path) -> Path:
    _require(isinstance(raw, str) and raw, "recorded path is absent")
    path = Path(raw)
    return path if path.is_absolute() else root / path


def load_frozen_predecessor(*, root: Path = ROOT) -> dict[str, Any]:
    """Narrow literal validator for the failed terminal and final checkpoint."""

    import torch
    from scripts import train_go2_utility_scorer_v1_2 as BASE
    from scripts import train_go2_utility_scorer_v1_3 as V13

    encoded = V13.load_preserved_encoded_training_view_for_replacement(
        root=root, verify_encoder_checkpoint=False)
    corpus = V13.corpus_from_encoded_bundle({**encoded, "root": root})
    terminal = V13._validate_signed(
        V13._read_json(V13.qualification_path(root),
                       label="frozen V1.3 qualification terminal"),
        V13.QUALIFICATION_SELF_KEY, "frozen V1.3 qualification terminal")
    _require(terminal[V13.QUALIFICATION_SELF_KEY]
             == FROZEN["qualification_report_digest"],
             "frozen V1.3 qualification digest changed")
    _require(terminal.get("terminal_kind") == "QUALIFICATION_FAILURE"
             and terminal.get("complete") is True
             and terminal.get("qualified") is False
             and terminal.get("scorer_package_sha256") is None,
             "failed V1.3 scorer was altered or reinterpreted")
    exact = {
        "training_view_digest": FROZEN["training_view_digest"],
        "latent_index_digest": FROZEN["latent_index_digest"],
        "target_encoder_digest": FROZEN["target_encoder_digest"],
        "target_encoder_checkpoint_sha256":
            FROZEN["target_encoder_checkpoint_sha256"],
        "preprocessing_digest": FROZEN["preprocessing_digest"],
        "training_run_digest": FROZEN["training_run_digest"],
        "binding_digest": FROZEN["binding_digest"],
        "failed_scorer_sha256": FROZEN["failed_scorer_sha256"],
    }
    _require(all(terminal.get(key) == value for key, value in exact.items()),
             "frozen V1.3 terminal binding changed")
    _require(encoded["receipt"]["encoding_receipt_digest"]
             == FROZEN["encoding_receipt_digest"],
             "frozen encoding receipt changed")

    receipt = terminal["training_receipts"]["latent"]
    checkpoint_path = _resolve_recorded_path(
        receipt["final_checkpoint"], root=root)
    _require(checkpoint_path.is_file() and not checkpoint_path.is_symlink()
             and file_sha256(checkpoint_path)
             == receipt.get("final_checkpoint_sha256")
             == FROZEN["latent_checkpoint_sha256"],
             "frozen latent final checkpoint bytes changed")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict")
    _require(isinstance(state, Mapping)
             and BASE.state_dict_digest(state)
             == checkpoint.get("model_state_digest")
             == receipt.get("final_state_digest")
             == FROZEN["latent_final_state_digest"],
             "frozen latent checkpoint model state changed")
    _require(checkpoint.get("completed_epoch") == 60
             and checkpoint.get("fixed_final_epoch") == 60
             and checkpoint.get("epoch_selection")
             == "final_epoch_only_no_selection"
             and checkpoint.get("training_run_digest")
             == FROZEN["training_run_digest"]
             and checkpoint.get("initial_state_digest")
             == FROZEN["latent_initial_state_digest"]
             and BASE.structured_digest(checkpoint["optimizer_state_dict"])
             == checkpoint.get("optimizer_state_digest")
             and BASE.structured_digest(checkpoint["rng_state"])
             == checkpoint.get("rng_state_digest")
             and BASE.tensor_digest(checkpoint["order_generator_state"])
             == checkpoint.get("order_generator_state_sha256")
             and BASE.tensor_digest(
                 checkpoint["last_epoch_order"].detach().cpu().to(torch.int64))
             == checkpoint.get("last_epoch_order_sha256"),
             "frozen latent checkpoint execution state changed")

    failed_path = V13.failed_scorer_path(root)
    _require(failed_path.is_file() and not failed_path.is_symlink()
             and file_sha256(failed_path)
             == FROZEN["failed_scorer_sha256"],
             "frozen failed-scorer artifact changed")
    artifact = torch.load(failed_path, map_location="cpu", weights_only=False)
    failed_state = artifact.get("latent")
    _require(artifact.get("qualified") is False
             and artifact.get("training_run_digest")
             == FROZEN["training_run_digest"]
             and isinstance(failed_state, Mapping)
             and BASE.state_dict_digest(failed_state)
             == FROZEN["latent_final_state_digest"],
             "frozen failed-scorer identity changed")
    _require(all(torch.equal(state[name], failed_state[name]) for name in state),
             "failed scorer and final checkpoint states differ")

    baseline = terminal["no_latent_baseline_receipt"]
    _require(baseline.get("sha256") == FROZEN["baseline_checkpoint_sha256"]
             and baseline.get("final_state_digest")
             == FROZEN["baseline_state_digest"]
             and baseline.get("baseline_receipt_digest")
             == FROZEN["baseline_receipt_digest"],
             "frozen baseline receipt changed")
    model = BASE.UtilityScorer(use_latent=True)
    model.load_state_dict(failed_state, strict=True)
    return {
        "corpus": corpus,
        "terminal": terminal,
        "model": model,
        "checkpoint_path": checkpoint_path,
        "failed_scorer_path": failed_path,
    }


def _device(name: str):
    import torch
    if name == "auto":
        value = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        value = torch.device(name)
    _require(value.type != "cuda" or torch.cuda.is_available(),
             "CUDA was requested but is unavailable")
    return value


def _static_features(rows: Sequence[Mapping[str, Any]], device: Any):
    import torch
    action = np.asarray([
        [value for block in row["action_blocks"] for value in block]
        for row in rows], dtype=np.float32)
    goal = np.asarray([row["goal_binding_input"] for row in rows],
                      dtype=np.float32)
    _require(action.shape == (len(rows), 40)
             and goal.shape == (len(rows), 3),
             "frozen action/goal input shape changed")
    targets = {
        key: torch.tensor([row[key] for row in rows], dtype=torch.float32,
                          device=device)
        for key in ("progress", "safety", "completion")
    }
    return torch.from_numpy(np.concatenate([action, goal], axis=1)).to(device), targets


def _safe_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe_json(member) for key, member in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(member) for member in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def matched_terminal_replay(
        observed: Mapping[str, Any], frozen: Mapping[str, Any], *,
        atol: float = MATCHED_TERMINAL_REPLAY_ATOL,
        ) -> dict[str, Any]:
    """Compare a matched metric tree to the exact frozen terminal tree."""

    _require(math.isfinite(atol) and atol >= 0.0,
             "matched-terminal replay tolerance is invalid")
    left = _safe_json(observed)
    right = _safe_json(frozen)
    mismatch_count = 0
    first_mismatch: dict[str, str] | None = None
    numeric_leaf_count = 0
    maximum = 0.0
    maximum_path: str | None = None

    def mismatch(path: str, reason: str) -> None:
        nonlocal mismatch_count, first_mismatch
        mismatch_count += 1
        if first_mismatch is None:
            first_mismatch = {"path": path, "reason": reason}

    def compare(value: Any, reference: Any, path: str) -> None:
        nonlocal numeric_leaf_count, maximum, maximum_path
        if isinstance(value, Mapping) or isinstance(reference, Mapping):
            if not isinstance(value, Mapping) or not isinstance(reference, Mapping):
                mismatch(path, "mapping type differs")
                return
            value_keys = set(value)
            reference_keys = set(reference)
            if value_keys != reference_keys:
                mismatch(path, "mapping keys differ")
            for key in sorted(value_keys & reference_keys, key=str):
                compare(value[key], reference[key], f"{path}.{key}")
            return
        if isinstance(value, list) or isinstance(reference, list):
            if not isinstance(value, list) or not isinstance(reference, list):
                mismatch(path, "sequence type differs")
                return
            if len(value) != len(reference):
                mismatch(path, "sequence length differs")
            for index, (member, expected) in enumerate(zip(value, reference)):
                compare(member, expected, f"{path}[{index}]")
            return
        numeric_value = (isinstance(value, (int, float))
                         and not isinstance(value, bool))
        numeric_reference = (isinstance(reference, (int, float))
                             and not isinstance(reference, bool))
        if numeric_value or numeric_reference:
            if not numeric_value or not numeric_reference:
                mismatch(path, "numeric type differs")
                return
            numeric_leaf_count += 1
            if not (math.isfinite(float(value))
                    and math.isfinite(float(reference))):
                mismatch(path, "non-finite numeric value is not canonical")
                return
            error = abs(float(value) - float(reference))
            if error > maximum:
                maximum = error
                maximum_path = path
            if error > atol:
                mismatch(path, "absolute numeric error exceeds tolerance")
            return
        if type(value) is not type(reference) or value != reference:
            mismatch(path, "canonical value differs")

    compare(left, right, "$")
    observed_digest = canonical_digest(left)
    frozen_digest = canonical_digest(right)
    matches = mismatch_count == 0
    return {
        "comparison": (
            "closed canonical metric tree; exact keys/order/non-numeric "
            "values; finite numeric leaves use absolute tolerance only"
        ),
        "absolute_tolerance": atol,
        "relative_tolerance": 0.0,
        "compared_scopes": list(right),
        "numeric_leaf_count": numeric_leaf_count,
        "max_abs_error": maximum,
        "max_abs_error_path": maximum_path,
        "observed_metrics_digest": observed_digest,
        "frozen_terminal_metrics_digest": frozen_digest,
        "canonical_equal": observed_digest == frozen_digest,
        "mismatch_count": mismatch_count,
        "first_mismatch": first_mismatch,
        "matches_frozen_terminal": matches,
        "verdict": "MATCH" if matches else "MISMATCH",
    }


def evaluate_variants(
        *, model: Any, rows: list[dict[str, Any]],
        variant_means: Mapping[str, np.ndarray], device: Any,
        baseline: Mapping[str, Any], frozen_matched: Mapping[str, Any],
        ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    import torch
    from scripts import train_go2_utility_scorer_v1_2 as BASE

    action_goal, targets = _static_features(rows, device)
    model.to(device)
    results = {}
    matched_predictions = None
    matched_replay = None
    invariance = {}
    for variant in VARIANT_IDS:
        latent = torch.from_numpy(variant_means[variant]).to(device)
        overall, predictions = BASE.evaluate_model(
            model, latent, action_goal, rows, targets)
        metrics = {
            "overall": overall,
            "per_family": BASE._grouped_calibration(
                rows, targets, predictions, "family"),
            "per_stratum": BASE._grouped_calibration(
                rows, targets, predictions, "stratum"),
        }
        if variant == A_MATCHED:
            matched_predictions = predictions
            _require(set(frozen_matched).issubset(metrics),
                     "frozen matched metric scopes are unavailable")
            observed_matched = {
                scope: metrics[scope] for scope in frozen_matched
            }
            matched_replay = matched_terminal_replay(
                observed_matched, frozen_matched)
            _require(matched_replay["matches_frozen_terminal"] is True,
                     "A-matched metrics do not replay frozen ViT-L terminal")
        elif variant in {
                C_HORIZON_REVERSED, D_TOKEN_PERMUTED,
                E_SPATIAL_MEAN_REPEATED}:
            _require(matched_predictions is not None,
                     "A-matched predictions were not evaluated first")
            invariance[variant] = prediction_invariance_error(
                matched_predictions, predictions)
        results[variant] = {"metrics": metrics}
    matched = results[A_MATCHED]["metrics"]
    for variant in VARIANT_IDS:
        results[variant]["delta_vs_matched"] = metric_delta(
            results[variant]["metrics"], matched)
        results[variant]["delta_vs_frozen_no_latent_baseline"] = metric_delta(
            results[variant]["metrics"], baseline)
    _require(set(invariance) == {
                 C_HORIZON_REVERSED, D_TOKEN_PERMUTED,
                 E_SPATIAL_MEAN_REPEATED}
             and all(value["all_within_absolute_tolerance"]
                     for value in invariance.values()),
             "architecture-mandated invariance check failed")
    _require(matched_replay is not None,
             "A-matched terminal replay was not performed")
    return (_safe_json(results), _safe_json(invariance),
            _safe_json(matched_replay))


def _git_output(root: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=root, text=True,
            stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LatentDependenceError(f"cannot bind diagnostic source: {exc}") from exc


def build_source_closure(*, root: Path = ROOT) -> dict[str, Any]:
    """Bind only the contract's explicit eight source paths."""

    commit = _git_output(root, "rev-parse", "HEAD")
    status = _git_output(root, "status", "--porcelain")
    _require(not status, "diagnostic execution requires clean committed source")
    files = {}
    for relative in CONTRACT.SOURCE_CLOSURE_PATHS:
        path = root / relative
        _require(path.is_file() and not path.is_symlink(),
                 f"source-closure path changed: {relative}")
        files[relative] = {
            "path": relative,
            "sha256": file_sha256(path),
            "byte_count": path.stat().st_size,
        }
    unsigned = {
        "schema": CONTRACT.SOURCE_CLOSURE_SCHEMA,
        "source_repository_commit": commit,
        "source_repository_clean": True,
        "git_status_porcelain_v1": "",
        "files": files,
    }
    return CONTRACT.validate_source_closure({
        **unsigned,
        CONTRACT.SOURCE_CLOSURE_SELF_KEY: CONTRACT.canonical_digest(unsigned),
    })


def bound_contract(source_closure: Mapping[str, Any]) -> dict[str, Any]:
    value = CONTRACT.contract(source_closure)
    _require(CONTRACT.validate_contract(value) == value,
             "failure-attribution contract does not validate")
    return value


def load_bound_contract(
        source_closure: Mapping[str, Any], *, root: Path = ROOT,
) -> dict[str, Any]:
    """Require the installed contract to bind the live clean source closure."""

    path = diagnostic_contract_path(root)
    _require(path.is_file() and not path.is_symlink(),
             "installed diagnostic contract is absent")
    try:
        installed_raw = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LatentDependenceError(
            "installed diagnostic contract is unreadable") from exc
    try:
        installed = CONTRACT.validate_contract(installed_raw)
    except CONTRACT.ScorerFailureAttributionContractError as exc:
        raise LatentDependenceError(
            "installed diagnostic contract is invalid") from exc
    expected = bound_contract(source_closure)
    _require(installed == expected,
             "installed diagnostic contract does not bind live source closure")
    return installed


def independently_recompute_terminal_evidence(
        *, root: Path, contract_digest: str, device_name: str = "auto",
        ) -> dict[str, Any]:
    """Rebuild A--G and their complete metrics without writing an artefact.

    This is a read-only validation replay from the frozen corpus, latent index,
    scorer state, action/goal inputs, and labels.  It is not a second
    authorised diagnostic session and does not publish predictions or a new
    result.
    """

    predecessor = load_frozen_predecessor(root=root)
    corpus = predecessor["corpus"]
    fit_rows = corpus["fit_rows"]
    calibration_rows = corpus["calibration_rows"]
    _require(len(fit_rows) == FIT_ROWS
             and len(calibration_rows) == CALIBRATION_ROWS,
             "frozen split cardinality changed during validation replay")

    derangement, derangement_receipt = within_state_derangement(
        calibration_rows)
    token_permutation, token_receipt = fixed_token_permutation()
    fit_mean, fit_mean_receipt = compute_fit_mean_trajectory(
        fit_rows, corpus["horizon"])
    transformation_freeze = {
        "failure_attribution_contract_digest": contract_digest,
        "transformations_contract_digest": canonical_digest(
            TRANSFORMATIONS),
        "variant_ids": list(VARIANT_IDS),
        "derangement": derangement_receipt,
        "token_permutation": token_receipt,
        "fit_mean": fit_mean_receipt,
        "freeze_digest": canonical_digest({
            "derangement_mapping_digest":
                derangement_receipt["mapping_digest"],
            "token_permutation_digest":
                token_receipt["permutation_digest"],
            "fit_mean_receipt_digest":
                fit_mean_receipt["fit_mean_receipt_digest"],
        }),
    }
    variant_means = materialise_variant_spatial_means(
        calibration_rows, corpus["horizon"],
        derangement=derangement, token_permutation=token_permutation,
        fit_mean_trajectory=fit_mean)
    baseline = {
        "overall": predecessor["terminal"]["results"]["no_latent"][
            "calibration"],
        "per_family": predecessor["terminal"]["results"]["no_latent"][
            "per_family_calibration"],
        "per_stratum": predecessor["terminal"]["results"]["no_latent"][
            "per_stratum_calibration"],
    }
    frozen_latent = predecessor["terminal"]["results"]["latent"]
    frozen_matched = {
        "overall": frozen_latent["calibration"],
        "per_family": frozen_latent["per_family_calibration"],
    }
    if "per_stratum_calibration" in frozen_latent:
        frozen_matched["per_stratum"] = frozen_latent[
            "per_stratum_calibration"]
    results, invariance, matched_replay = evaluate_variants(
        model=predecessor["model"], rows=calibration_rows,
        variant_means=variant_means, device=_device(device_name),
        baseline=baseline, frozen_matched=frozen_matched)
    return _safe_json({
        "transformation_freeze": transformation_freeze,
        "results": results,
        "architecture_invariance_checks": invariance,
        "matched_condition_terminal_replay": matched_replay,
        "frozen_no_latent_baseline": baseline,
    })


def validate_recomputed_terminal_evidence(
        *, result: Mapping[str, Any], authorisation: Mapping[str, Any],
        recomputed: Mapping[str, Any],
        ) -> dict[str, Any]:
    """Require exact canonical equality with a read-only validation replay."""

    expected_fields = {"transformation_freeze", *RECOMPUTED_RESULT_FIELDS}
    _require(set(recomputed) == expected_fields,
             "independent replay evidence schema is not closed")
    expected_freeze = recomputed.get("transformation_freeze")
    recorded_freeze = authorisation.get("transformation_freeze")
    _require(isinstance(expected_freeze, Mapping)
             and isinstance(recorded_freeze, Mapping)
             and recorded_freeze == expected_freeze,
             "recorded A--G transformation freeze does not replay")
    evidence_digests = {}
    for field in RECOMPUTED_RESULT_FIELDS:
        recorded = result.get(field)
        expected = recomputed.get(field)
        _require(isinstance(recorded, Mapping)
                 and isinstance(expected, Mapping)
                 and recorded == expected,
                 f"recorded {field} does not exactly replay frozen inputs")
        evidence_digests[field] = canonical_digest(expected)
    return {
        "validation_mode": "READ_ONLY_INDEPENDENT_DETERMINISTIC_REPLAY",
        "transformation_freeze_digest": canonical_digest(expected_freeze),
        "result_field_digests": evidence_digests,
        "writes": 0,
    }


def validate_result_for_consumption(
        *, root: Path = ROOT, device_name: str = "auto",
        ) -> dict[str, Any]:
    """Validate and independently replay the complete terminal evidence."""

    source_closure = build_source_closure(root=root)
    study_contract = load_bound_contract(source_closure, root=root)
    result = _validate_signed(
        json.loads(result_path(root).read_text()), RESULT_SELF_KEY,
        "latent-dependence result")
    authorisation = _validate_signed(
        json.loads(evaluation_authorisation_path(root).read_text()),
        AUTHORISATION_SELF_KEY,
        "latent-dependence evaluation authorisation")
    contract_digest = study_contract[CONTRACT.CONTRACT_SELF_KEY]
    _require(result.get("schema") == SCHEMA
             and result.get("status") == STATUS
             and result.get("complete") is True
             and result.get("source_closure_digest")
             == source_closure[CONTRACT.SOURCE_CLOSURE_SELF_KEY]
             and result.get("failure_attribution_contract_digest")
             == contract_digest
             and result.get("predecessor_terminal_digest")
             == FROZEN["qualification_report_digest"]
             and result.get("predecessor_failed_scorer_sha256")
             == FROZEN["failed_scorer_sha256"]
             and result.get("predecessor_latent_final_state_digest")
             == FROZEN["latent_final_state_digest"]
             and result.get("training_view_digest")
             == FROZEN["training_view_digest"]
             and result.get("latent_index_digest")
             == FROZEN["latent_index_digest"]
             and result.get("evaluation_authorisation_digest")
             == authorisation[AUTHORISATION_SELF_KEY]
             and result.get("calibration_diagnostic_session_count") == 1
             and set(result.get("results", {})) == set(VARIANT_IDS)
             and result.get("training_executions") == 0
             and result.get("scorer_package_published") is False
             and result.get("predictor_retrained") is False
             and result.get("predictor_checkpoints_opened") == 0
             and result.get("predictor_utility_shards_opened") == 0
             and result.get("final_200_state_corpus_generated") is False,
             "latent-dependence result binding changed")
    replay = result.get("matched_condition_terminal_replay", {})
    _require(replay.get("matches_frozen_terminal") is True
             and replay.get("verdict") == "MATCH"
             and replay.get("mismatch_count") == 0
             and replay.get("absolute_tolerance")
             == MATCHED_TERMINAL_REPLAY_ATOL,
             "matched latent condition no longer replays the frozen terminal")
    freeze = authorisation.get("transformation_freeze", {})
    _require(authorisation.get("schema") == AUTHORISATION_SCHEMA
             and authorisation.get("status") == STATUS
             and authorisation.get("complete") is True
             and authorisation.get("source_commit")
             == source_closure["source_repository_commit"]
             and authorisation.get("source_closure_digest")
             == source_closure[CONTRACT.SOURCE_CLOSURE_SELF_KEY]
             and authorisation.get("failure_attribution_contract_digest")
             == contract_digest
             and authorisation.get("predecessor_terminal_digest")
             == FROZEN["qualification_report_digest"]
             and freeze.get("failure_attribution_contract_digest")
             == contract_digest
             and freeze.get("variant_ids") == list(VARIANT_IDS)
             and result.get("transformation_freeze_digest")
             == freeze.get("freeze_digest")
             and authorisation.get(
                 "calibration_diagnostic_sessions_authorised") == 1
             and authorisation.get(
                 "calibration_diagnostic_sessions_completed_before_issue") == 0
             and authorisation.get("training_authorised") is False
             and authorisation.get("retry_or_resume_authorised") is False
             and authorisation.get("raw_prediction_persistence_authorised")
             is False,
             "latent-dependence evaluation authorisation changed")
    recomputed = independently_recompute_terminal_evidence(
        root=root, contract_digest=contract_digest, device_name=device_name)
    validate_recomputed_terminal_evidence(
        result=result, authorisation=authorisation, recomputed=recomputed)
    return result


def run_once(*, root: Path = ROOT, device_name: str = "auto") -> dict[str, Any]:
    """Run exactly one frozen-scorer diagnostic session."""

    if result_path(root).exists() or result_path(root).is_symlink():
        return validate_result_for_consumption(
            root=root, device_name=device_name)
    _require(not technical_failure_path(root).exists()
             and not technical_failure_path(root).is_symlink()
             and not evaluation_authorisation_path(root).exists()
             and not evaluation_authorisation_path(root).is_symlink(),
             "the sole latent-dependence session was already consumed")
    stage = "predecessor_validation"
    try:
        source_closure = build_source_closure(root=root)
        study_contract = load_bound_contract(source_closure, root=root)
        study_contract_digest = study_contract[CONTRACT.CONTRACT_SELF_KEY]
        predecessor = load_frozen_predecessor(root=root)
        corpus = predecessor["corpus"]
        fit_rows = corpus["fit_rows"]
        calibration_rows = corpus["calibration_rows"]
        _require(len(fit_rows) == FIT_ROWS
                 and len(calibration_rows) == CALIBRATION_ROWS,
                 "frozen split cardinality changed")

        stage = "transformation_freeze"
        derangement, derangement_receipt = within_state_derangement(
            calibration_rows)
        token_permutation, token_receipt = fixed_token_permutation()
        fit_mean, fit_mean_receipt = compute_fit_mean_trajectory(
            fit_rows, corpus["horizon"])
        transformation_freeze = {
            "failure_attribution_contract_digest": study_contract_digest,
            "transformations_contract_digest": canonical_digest(
                TRANSFORMATIONS),
            "variant_ids": list(VARIANT_IDS),
            "derangement": derangement_receipt,
            "token_permutation": token_receipt,
            "fit_mean": fit_mean_receipt,
            "freeze_digest": canonical_digest({
                "derangement_mapping_digest":
                    derangement_receipt["mapping_digest"],
                "token_permutation_digest":
                    token_receipt["permutation_digest"],
                "fit_mean_receipt_digest":
                    fit_mean_receipt["fit_mean_receipt_digest"],
            }),
        }
        authorisation = _signed({
            "schema": AUTHORISATION_SCHEMA,
            "status": STATUS,
            "complete": True,
            "source_commit": source_closure["source_repository_commit"],
            "source_closure_digest":
                source_closure[CONTRACT.SOURCE_CLOSURE_SELF_KEY],
            "failure_attribution_contract_digest": study_contract_digest,
            "predecessor_terminal_digest":
                FROZEN["qualification_report_digest"],
            "predecessor_latent_final_state_digest":
                FROZEN["latent_final_state_digest"],
            "transformation_freeze": transformation_freeze,
            "calibration_diagnostic_sessions_authorised": 1,
            "calibration_diagnostic_sessions_completed_before_issue": 0,
            "training_authorised": False,
            "retry_or_resume_authorised": False,
            "raw_prediction_persistence_authorised": False,
            "predictor_checkpoints_opened": 0,
            "predictor_utility_shards_opened": 0,
            "qualified_scorer_package_publication_authorised": False,
            "final_200_state_corpus_generated": False,
        }, AUTHORISATION_SELF_KEY)
        publish_json_once(
            evaluation_authorisation_path(root), authorisation,
            label="latent-dependence evaluation authorisation")

        # The A--G calibration trajectories are not materialised or forwarded
        # until their mapping, token permutation, and fit-only mean digests are
        # frozen in the immutable authorisation above.
        stage = "calibration_diagnostic_session"
        variant_means = materialise_variant_spatial_means(
            calibration_rows, corpus["horizon"],
            derangement=derangement, token_permutation=token_permutation,
            fit_mean_trajectory=fit_mean)
        baseline = {
            "overall": predecessor["terminal"]["results"]["no_latent"][
                "calibration"],
            "per_family": predecessor["terminal"]["results"]["no_latent"][
                "per_family_calibration"],
            "per_stratum": predecessor["terminal"]["results"]["no_latent"][
                "per_stratum_calibration"],
        }
        frozen_latent = predecessor["terminal"]["results"]["latent"]
        frozen_matched = {
            "overall": frozen_latent["calibration"],
            "per_family": frozen_latent["per_family_calibration"],
        }
        if "per_stratum_calibration" in frozen_latent:
            frozen_matched["per_stratum"] = frozen_latent[
                "per_stratum_calibration"]
        results, invariance, matched_replay = evaluate_variants(
            model=predecessor["model"], rows=calibration_rows,
            variant_means=variant_means, device=_device(device_name),
            baseline=baseline, frozen_matched=frozen_matched)
        payload = _signed(_safe_json({
            "schema": SCHEMA,
            "status": STATUS,
            "complete": True,
            "scientific_role": "DESCRIPTIVE_FAILURE_ATTRIBUTION_ONLY",
            "source_closure_digest":
                source_closure[CONTRACT.SOURCE_CLOSURE_SELF_KEY],
            "failure_attribution_contract_digest": study_contract_digest,
            "predecessor_terminal_digest":
                FROZEN["qualification_report_digest"],
            "predecessor_failed_scorer_sha256":
                FROZEN["failed_scorer_sha256"],
            "predecessor_latent_final_checkpoint_sha256":
                FROZEN["latent_checkpoint_sha256"],
            "predecessor_latent_final_state_digest":
                FROZEN["latent_final_state_digest"],
            "training_view_digest": FROZEN["training_view_digest"],
            "latent_index_digest": FROZEN["latent_index_digest"],
            "encoding_receipt_digest": FROZEN["encoding_receipt_digest"],
            "evaluation_authorisation_digest":
                authorisation[AUTHORISATION_SELF_KEY],
            "transformation_freeze_digest":
                transformation_freeze["freeze_digest"],
            "calibration_diagnostic_session_count": 1,
            "results": results,
            "matched_condition_terminal_replay": matched_replay,
            "architecture_invariance_checks": invariance,
            "frozen_no_latent_baseline": baseline,
            "raw_predictions_persisted": False,
            "training_executions": 0,
            "scorer_package_published": False,
            "predictor_retrained": False,
            "predictor_checkpoints_opened": 0,
            "predictor_utility_shards_opened": 0,
            "final_200_state_corpus_generated": False,
            "retry_or_resume_authorised": False,
        }), RESULT_SELF_KEY)
        publish_json_once(
            result_path(root), payload, label="latent-dependence result")
        return validate_result_for_consumption(
            root=root, device_name=device_name)
    except BaseException as exc:
        path = technical_failure_path(root)
        if not path.exists() and not path.is_symlink():
            failure = _signed({
                "schema": TECHNICAL_FAILURE_SCHEMA,
                "status": "INVALID_TECHNICAL_LATENT_DEPENDENCE_SESSION",
                "complete": True,
                "stage": stage,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
                "retry_or_resume_authorised": False,
                "training_executions": 0,
                "raw_predictions_persisted": False,
                "predictor_checkpoints_opened": 0,
                "predictor_utility_shards_opened": 0,
                "final_200_state_corpus_generated": False,
            }, TECHNICAL_FAILURE_SELF_KEY)
            publish_json_once(path, failure,
                              label="latent-dependence technical failure")
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto",
                        choices=("auto", "cpu", "cuda"))
    args = parser.parse_args(argv)
    result = run_once(device_name=args.device)
    print(json.dumps({
        "status": result["status"],
        "result_digest": result[RESULT_SELF_KEY],
        "calibration_diagnostic_session_count":
            result["calibration_diagnostic_session_count"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
