#!/usr/bin/env python3
"""One bounded utility-scorer transfer to the frozen 20-state development set.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.

This consumer is intentionally separate from the paused 200-state final-corpus
implementation.  It cannot generate a branch, render or encode a frame, load a
world-model checkpoint, or run predictor inference.  It first verifies that the
single shared scorer passed every frozen true-latent qualification gate.  Only
then may it open the already frozen Stage-A target shards and B/C prediction
shards from the completed counterfactual-predictor qualification.

Scoring is interruption safe at one immutable unit: true targets, the no-latent
baseline, or one of the 32 seed/cell prediction packages.  Invalid score shards
are preserved and only that exact registered unit is regenerated.  A complete
result bound to the same prospective specification is reused without repeating
the exploratory analysis.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_go2_counterfactual_predictor_qualification_v1_2 as A  # noqa: E402
from scripts import train_go2_utility_scorer_v1_2 as S  # noqa: E402
from lewm.oracle.go2_scorer_contract_v1_2 import contract_digest  # noqa: E402

STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
OUT_DIR = S.PACKAGE_DIR / "counterfactual_development_transfer_v1_2"
SCORE_DIR = OUT_DIR / "score_shards"
SPEC_PATH = OUT_DIR / "development_transfer_spec.json"
RESULT_PATH = OUT_DIR / "result.json"

FROZEN_PREDICTOR_QUALIFICATION_COMMIT = (
    "ee47b47e7964c16360f265c4cfbe7f8181d16402"
)
FROZEN_STAGE_A_IDENTITY_DIGEST = (
    "ce2cbbe8dab9a89ad6f85d16c56a9d712d791c8bbfd8925a8f01efc0c039705a"
)
FROZEN_STAGE_A_CORPUS_DIGEST = (
    "f84eb3271f1a3b7052bbf2e84240453e84772b0a530e60ec47f723a44e2e10e9"
)
FROZEN_STAGE_A_LATENT_INDEX_DIGEST = (
    "861285ec9c8fc6c92c6f3a31cade0f031172bf6818d76d1899634a60c7e5c291"
)
FROZEN_BC_RESULT_DIGEST = (
    "3b5c500b4b1326056ce18c6276d7842f4230faec36f8f29cc65945f54527bbcb"
)
FROZEN_OCCUPANCY_RESULT_DIGEST = (
    "09dc413d9ce30c2cb19c99e93eeaad410983a7f53575387bc6694f3844a070d6"
)
FROZEN_OCCUPANCY_GATE_DIGEST = (
    "4bf9a92144fa728d953c9dffebb235c9b476ded59d7462a107fe2e6ade0894e4"
)

EXPECTED_STATES = 20
EXPECTED_BRANCHES = 240
EXPECTED_CANDIDATES = 12
EXPECTED_CHECKPOINTS = 32
EXPECTED_FAMILIES = 8
TOKENS = 768
TOKEN_DIM = 1024
HORIZONS = 4
ACTION_GOAL_DIM = 43
SCORE_TIE_TOLERANCE = 0.02
T_CRITICAL_95_DF7 = 2.3646242510102993
CELLS = tuple(A.D.CELLS)
SEEDS = tuple(A.D.SEED_REGISTRY[:8])
FAMILIES = tuple(A.FAMILIES)

METRIC_DIRECTIONS = {
    "normalised_rank_regret": "lower",
    "absolute_rank_regret": "lower",
    "realised_selected_utility": "higher",
    "spearman_rank_correlation": "higher",
    "top1_recovery": "higher",
    "top3_recovery": "higher",
    "pairwise_ordering_accuracy": "higher",
    "candidate_score_spread": "descriptive",
    "scorer_tie_rate": "lower",
}
RECOVERY_EVENTS: list[dict[str, Any]] = []


class DevelopmentTransferRefused(RuntimeError):
    """A frozen qualification, provenance, or no-leakage gate failed."""


@dataclass(frozen=True)
class ScorerBundle:
    qualification: dict[str, Any]
    package: dict[str, Any]
    package_sha256: str
    latent: S.UtilityScorer
    no_latent: S.UtilityScorer


@dataclass(frozen=True)
class PredictionPackage:
    seed: int
    cell: str
    checkpoint_sha256: str
    index: dict[str, Any]
    state_shards: dict[str, dict[str, Any]]
    input_digest: str
    storage_bytes: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DevelopmentTransferRefused(message)


def canonical_digest(value: Any, omit: Iterable[str] = ()) -> str:
    omitted = set(omit)
    if isinstance(value, Mapping):
        value = {key: item for key, item in value.items() if key not in omitted}
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
        default=str).encode("utf-8")).hexdigest()


def legacy_digest(value: Any, omit: Iterable[str] = ()) -> str:
    """Digest convention used by the frozen trainer and B/C result."""

    omitted = set(omit)
    if isinstance(value, Mapping):
        value = {key: item for key, item in value.items() if key not in omitted}
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DevelopmentTransferRefused(f"unreadable {label}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} is not a JSON object")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    encoded = (json.dumps(value, indent=2, sort_keys=True,
                          allow_nan=False, default=str) + "\n").encode()
    with temporary.open("wb") as sink:
        sink.write(encoded); sink.flush(); os.fsync(sink.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_relative(root: Path, value: Any, label: str) -> Path:
    _require(isinstance(value, str) and value, f"{label} path is absent")
    candidate = Path(value)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    allowed = root.resolve()
    _require(resolved != allowed and allowed in resolved.parents,
             f"{label} escapes its frozen artifact root")
    return resolved


# ----------------------------------------------------------- scorer gate -----
def validate_qualified_scorer() -> ScorerBundle:
    """Refuse before torch.load unless every frozen qualification gate passed."""

    qualification = read_json(S.PACKAGE_DIR / "qualification.json",
                              "scorer qualification")
    _require(qualification.get("schema") == "go2_utility_scorer_v1_2_qualification",
             "wrong scorer qualification schema")
    _require(qualification.get("qualification_report_digest")
             == legacy_digest(qualification, ("qualification_report_digest",)),
             "scorer qualification self digest differs")
    criteria = qualification.get("criteria")
    _require(isinstance(criteria, Mapping) and criteria
             and all(value is True for value in criteria.values()),
             "shared scorer did not pass every frozen criterion")
    _require(qualification.get("qualified") is True
             and qualification.get("qualification_evaluations") == 1
             and qualification.get("epoch_selection_permitted") is False,
             "scorer was not frozen by the one-shot final-epoch qualification")
    _require(qualification.get("scorer_contract_v1_2_digest") == contract_digest(),
             "qualified scorer binds a different current frozen contract")
    for key in S.SELECTOR_BINDING_KEYS:
        _require(isinstance(qualification.get(key), str)
                 and S.HEX64.fullmatch(qualification[key]) is not None,
                 f"qualified scorer has no {key}")
    _require(qualification["state_selector_amendment_digest"]
             == S.STATE_SELECTOR.state_selector_amendment_digest(),
             "qualified scorer binds a different state-selector amendment")
    try:
        current_source = S.clean_source_binding()
    except RuntimeError as exc:
        raise DevelopmentTransferRefused(
            f"development transfer requires the scorer's clean committed source: {exc}"
        ) from exc
    expected_source_bindings = {
        "source_repository_commit": current_source["source_repository_commit"],
        "clean_source_binding_digest": S.canonical_digest(current_source),
        "bound_implementations_digest": current_source["bound_implementations_digest"],
    }
    for key, expected in expected_source_bindings.items():
        _require(qualification.get(key) == expected,
                 f"qualified scorer {key} differs from current clean source")
    for key in ("clean_source_launch_receipt_digest",
                "scorer_contract_artifact_digest"):
        _require(isinstance(qualification.get(key), str)
                 and len(qualification[key]) == 64,
                 f"qualified scorer has no {key}")
    package_sha = qualification.get("scorer_package_sha256")
    _require(isinstance(package_sha, str) and len(package_sha) == 64,
             "qualification has no frozen scorer-package digest")
    package_path = S.PACKAGE_DIR / "scorer_package.pt"
    _require(package_path.is_file() and sha256_file(package_path) == package_sha,
             "qualified scorer package bytes differ")
    receipt = read_json(S.PACKAGE_DIR / "scorer_package_receipt.json",
                        "scorer package receipt")
    _require(receipt.get("scorer_package_receipt_digest")
             == legacy_digest(receipt, ("scorer_package_receipt_digest",)),
             "scorer package receipt self digest differs")
    _require(receipt.get("complete") is True and receipt.get("qualified") is True
             and receipt.get("scorer_package_sha256") == package_sha
             and receipt.get("scorer_contract_v1_2_digest") == contract_digest()
             and all(receipt.get(key) == qualification.get(key)
                     for key in S.SCORER_PROVENANCE_BINDING_KEYS),
             "scorer package receipt is incomplete or differently bound")
    baseline_receipt = qualification.get("no_latent_baseline_package")
    _require(isinstance(baseline_receipt, Mapping)
             and baseline_receipt.get("receipt_digest")
             == legacy_digest(baseline_receipt, ("receipt_digest",))
             and baseline_receipt.get("complete") is True
             and baseline_receipt.get("training_run_digest")
             == qualification.get("training_run_digest")
             and all(baseline_receipt.get(key) == qualification.get(key)
                     for key in S.SCORER_PROVENANCE_BINDING_KEYS),
             "no-latent baseline package receipt is absent or invalid")
    baseline_path = _safe_relative(
        S.PACKAGE_DIR, baseline_receipt.get("path"), "no-latent baseline package")
    _require(baseline_path.is_file()
             and baseline_path.stat().st_size == baseline_receipt.get("byte_count")
             and sha256_file(baseline_path) == baseline_receipt.get("sha256"),
             "no-latent baseline package bytes differ")

    # torch.load is deliberately below every JSON/byte qualification gate.
    package = torch.load(package_path, map_location="cpu", weights_only=False)
    _require(package.get("qualified") is True
             and package.get("scorer_contract_v1_2_digest") == contract_digest()
             and package.get("training_run_digest")
             == qualification.get("training_run_digest")
             and all(package.get(key) == qualification.get(key)
                     for key in S.SCORER_PROVENANCE_BINDING_KEYS)
             and package.get("final_epoch") == 60
             and package.get("epoch_selection") == "final_epoch_only_no_selection",
             "scorer package metadata differs from the frozen qualified run")
    final_digests = package.get("final_state_digests")
    _require(isinstance(final_digests, Mapping), "scorer final-state digests absent")
    for name in ("latent", "no_latent"):
        _require(isinstance(package.get(name), Mapping)
                 and S.state_dict_digest(package[name]) == final_digests.get(name),
                 f"{name} final state digest differs")
    baseline_package = torch.load(
        baseline_path, map_location="cpu", weights_only=False)
    _require(baseline_package.get("schema")
             == "go2_utility_no_latent_baseline_package_v1_2"
             and baseline_package.get("training_run_digest")
             == qualification.get("training_run_digest")
             and baseline_package.get("scorer_contract_v1_2_digest")
             == contract_digest()
             and all(baseline_package.get(key) == qualification.get(key)
                     for key in S.SCORER_PROVENANCE_BINDING_KEYS)
             and isinstance(baseline_package.get("model_state_dict"), Mapping)
             and S.state_dict_digest(baseline_package["model_state_dict"])
             == final_digests.get("no_latent")
             == baseline_receipt.get("final_state_digest"),
             "separate no-latent baseline package differs from the qualified scorer")
    latent = S.UtilityScorer(use_latent=True)
    no_latent = S.UtilityScorer(use_latent=False)
    latent.load_state_dict(package["latent"], strict=True)
    no_latent.load_state_dict(baseline_package["model_state_dict"], strict=True)
    latent.eval(); no_latent.eval()
    return ScorerBundle(qualification, package, str(package_sha), latent, no_latent)


def prospective_spec(scorer: ScorerBundle) -> dict[str, Any]:
    sources = (
        "scripts/apply_go2_utility_scorer_to_counterfactual_development_v1_2.py",
        "scripts/train_go2_utility_scorer_v1_2.py",
        "scripts/analyze_go2_counterfactual_predictor_qualification_v1_2.py",
        "lewm/oracle/go2_scorer_contract_v1_2.py",
    )
    value: dict[str, Any] = {
        "schema": "go2_utility_scorer_counterfactual_development_transfer_spec_v1_2",
        "status": STATUS, "frozen_before_prediction_shard_access": True,
        "predictor_qualification_commit": FROZEN_PREDICTOR_QUALIFICATION_COMMIT,
        "scorer_contract_v1_2_digest": contract_digest(),
        "qualification_report_digest":
            scorer.qualification["qualification_report_digest"],
        "scorer_package_sha256": scorer.package_sha256,
        "scorer_source_bindings": {
            key: scorer.qualification[key] for key in S.LAUNCH_BINDING_KEYS
        },
        "scorer_selector_successor_bindings": {
            key: scorer.qualification[key] for key in S.SELECTOR_BINDING_KEYS
        },
        "frozen_inputs": {
            "stage_a_identity_manifest_digest": FROZEN_STAGE_A_IDENTITY_DIGEST,
            "stage_a_corpus_digest": FROZEN_STAGE_A_CORPUS_DIGEST,
            "stage_a_latents_index_digest": FROZEN_STAGE_A_LATENT_INDEX_DIGEST,
            "direct_fidelity_and_retrieval_result_digest": FROZEN_BC_RESULT_DIGEST,
            "occupancy_result_digest_not_consumed": FROZEN_OCCUPANCY_RESULT_DIGEST,
            "occupancy_gate_digest_not_consumed": FROZEN_OCCUPANCY_GATE_DIGEST,
        },
        "scope": {"states": 20, "branches": 240, "candidates_per_state": 12,
                  "predictor_checkpoints": 32, "new_branches": 0,
                  "new_frames": 0, "new_latents": 0,
                  "predictor_checkpoint_loads": 0, "predictor_inference": False},
        "true_target_handling": (
            "reload raw float16 target tokens as float32, apply token-wise "
            "F.layer_norm over 1024 dimensions, then mean over 768 tokens"),
        "predicted_handling": (
            "reload already-normalised frozen float16 predictor tokens as float32, "
            "then mean over 768 tokens; no second calibration or normalisation"),
        "scorer_features": {
            "latent": "full H1-H4 spatial-mean trajectory [4,1024]",
            "action": "frozen 4x10 post-slew action_blocks flattened to 40D",
            "goal": "frozen goal_binding_input [sin(bearing),cos(bearing),range]",
        },
        "metric_contract": {
            "true_utility_tie_tolerance": SCORE_TIE_TOLERANCE,
            "score_tie_tolerance": SCORE_TIE_TOLERANCE,
            "state_first": True,
            "equal_family_primary": "state mean -> family mean -> unweighted 8-family mean",
            "corpus_weighted_secondary": "unweighted mean of the 20 state metrics",
            "replication": "eight predictor-training seed quadruplets, df=7",
            "directions": METRIC_DIRECTIONS,
        },
        "source_sha256": {relative: sha256_file(ROOT / relative)
                          for relative in sources},
    }
    value["development_transfer_spec_digest"] = legacy_digest(value)
    return value


def freeze_spec(value: Mapping[str, Any]) -> None:
    if SPEC_PATH.exists():
        existing = read_json(SPEC_PATH, "development transfer spec")
        _require(existing == value, "existing development-transfer spec differs")
        return
    atomic_json(SPEC_PATH, value)


# ---------------------------------------------------------- frozen inputs ----
def validate_stage_a() -> tuple[A.StageABundle, dict[str, Any]]:
    bundle = A.validate_stage_a_metadata()
    _require(bundle.identity_digest == FROZEN_STAGE_A_IDENTITY_DIGEST,
             "Stage-A identity differs")
    _require(bundle.corpus_digest == FROZEN_STAGE_A_CORPUS_DIGEST,
             "Stage-A corpus differs")
    _require(bundle.latent_index_digest == FROZEN_STAGE_A_LATENT_INDEX_DIGEST,
             "Stage-A latent index differs")
    # This transfer consumes horizon targets only.  The completed B/C assay
    # already bound the encoder checkpoint and context shards; reopening the
    # 5-GB encoder or unused observed-context storage here would add no evidence.
    # Verify the exact 240 target shards that will actually be scored.
    started = time.time()
    rows: list[dict[str, Any]] = []
    total_bytes = 0
    for key in sorted(bundle.horizon_records):
        record = bundle.horizon_records[key]
        path = Path(str(record["_resolved_path"]))
        byte_count = int(record["byte_count"])
        _require(path.is_file() and path.stat().st_size == byte_count
                 and sha256_file(path) == record["sha256"],
                 f"Stage-A horizon target shard differs: {key}")
        total_bytes += byte_count
        rows.append({"key": key, "sha256": record["sha256"],
                     "byte_count": byte_count, "shape": record["shape"]})
    _require(len(rows) == EXPECTED_BRANCHES,
             "Stage-A target-shard set does not contain 240 branches")
    verification = {
        "complete": True, "horizon_shards": len(rows), "bytes": total_bytes,
        "target_encoder_checkpoint_opened": False,
        "context_shards_opened": False,
        "verified_shard_set_digest": A.sequence_digest(rows),
        "wall_time_s": round(time.time() - started, 3),
    }
    return bundle, verification


def ordered_rows(bundle: A.StageABundle) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    states = sorted(bundle.states, key=lambda value: value.state_index)
    _require([state.state_index for state in states] == list(range(EXPECTED_STATES)),
             "development state indices are not contiguous")
    for state in states:
        for candidate_index in range(EXPECTED_CANDIDATES):
            row = dict(bundle.row_by_pair[(state.state_id, candidate_index)])
            _require(row.get("valid") is True
                     and row.get("oracle_outcome_equal") is True,
                     "development branch is not a valid frozen oracle-equal row")
            action = np.asarray(row.get("action_blocks"), dtype=np.float64)
            goal = np.asarray(row.get("goal_binding_input"), dtype=np.float64)
            _require(action.shape == (HORIZONS, 10)
                     and goal.shape == (3,)
                     and np.isfinite(action).all() and np.isfinite(goal).all(),
                     "development scorer action/goal input differs")
            for key in ("utility", "progress", "safety", "completion"):
                _require(isinstance(row.get(key), (int, float))
                         and math.isfinite(float(row[key])),
                         f"development oracle label {key} is invalid")
            rows.append(row)
    _require(len(rows) == EXPECTED_BRANCHES, "development row order is incomplete")
    return rows


def action_goal(rows: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    result = np.empty((len(rows), ACTION_GOAL_DIM), dtype=np.float32)
    for position, row in enumerate(rows):
        action = np.asarray(row["action_blocks"], dtype=np.float32).reshape(-1)
        goal = np.asarray(row["goal_binding_input"], dtype=np.float32)
        result[position] = np.concatenate([action, goal])
    return torch.from_numpy(result)


def validate_bc_result(bundle: A.StageABundle) -> tuple[
        dict[str, Any], dict[tuple[int, str], str],
        dict[tuple[int, str], dict[str, Any]]]:
    result = read_json(A.RESULT_PATH, "frozen B/C result")
    _require(result.get("report_digest")
             == A.json_digest(result, ("report_digest",))
             == FROZEN_BC_RESULT_DIGEST,
             "frozen B/C result digest differs")
    _require(result.get("complete") is True and result.get("utility_scorer_used") is False,
             "B/C result is incomplete or already utility-scored")
    stage = result.get("stage_a", {})
    _require(stage.get("identity_manifest_digest") == bundle.identity_digest
             and stage.get("corpus_digest") == bundle.corpus_digest
             and stage.get("latents_index_digest") == bundle.latent_index_digest,
             "B/C result binds a different Stage-A corpus")
    inventory: dict[tuple[int, str], str] = {}
    for record in result.get("verified_checkpoints", []):
        key = (int(record["seed"]), str(record["cell"]))
        _require(key not in inventory, "duplicate B/C checkpoint inventory row")
        inventory[key] = str(record["sha256"])
    expected = {(seed, cell) for seed in SEEDS for cell in CELLS}
    _require(set(inventory) == expected and len(inventory) == EXPECTED_CHECKPOINTS,
             "B/C checkpoint inventory is not the frozen eight quadruplets")
    receipts: dict[tuple[int, str], dict[str, Any]] = {}
    for receipt in result.get("checkpoint_prediction_receipts", []):
        _require(isinstance(receipt, dict), "B/C checkpoint receipt is not an object")
        key = (int(receipt["seed"]), str(receipt["cell"]))
        _require(key not in receipts
                 and receipt.get("receipt_digest")
                 == A.json_digest(receipt, ("receipt_digest",)),
                 "B/C checkpoint receipt is duplicated or has a bad digest")
        receipts[key] = receipt
    _require(set(receipts) == expected,
             "B/C result does not bind all 32 checkpoint prediction receipts")
    return result, inventory, receipts


def validate_prediction_package(bundle: A.StageABundle, seed: int, cell: str,
                                checkpoint_sha256: str,
                                frozen_checkpoint_receipt: Mapping[str, Any]
                                ) -> PredictionPackage:
    directory = A.PREDICTION_DIR / f"seed_{seed}_{cell}"
    index = read_json(directory / "predictions_index.json",
                      f"prediction index {seed}/{cell}")
    _require(index.get("predictions_index_digest")
             == A.json_digest(index, ("predictions_index_digest",)),
             f"prediction index self digest differs {seed}/{cell}")
    _require(index.get("schema")
             == "go2_counterfactual_predictor_predictions_index_v1_2"
             and index.get("complete") is True
             and index.get("utility_scorer_used") is False,
             f"prediction index incomplete {seed}/{cell}")
    for key, expected in (
        ("stage_a_identity_manifest_digest", bundle.identity_digest),
        ("stage_a_corpus_digest", bundle.corpus_digest),
        ("stage_a_latents_index_digest", bundle.latent_index_digest),
        ("checkpoint_sha256", checkpoint_sha256),
        ("checkpoint_epoch", A.D.CHECKPOINT_EPOCH),
        ("seed", seed), ("cell", cell), ("states", EXPECTED_STATES),
        ("branches", EXPECTED_BRANCHES),
    ):
        _require(index.get(key) == expected,
                 f"prediction index {key} differs {seed}/{cell}")
    state_records = index.get("state_shards")
    branch_records = index.get("branch_records")
    _require(isinstance(state_records, list) and len(state_records) == EXPECTED_STATES
             and isinstance(branch_records, list)
             and len(branch_records) == EXPECTED_BRANCHES,
             f"prediction index counts differ {seed}/{cell}")

    state_shards: dict[str, dict[str, Any]] = {}
    storage = 0
    for record in state_records:
        _require(isinstance(record, dict), "prediction state-shard record is not an object")
        state_id = str(record.get("state_id"))
        _require(state_id not in state_shards, "duplicate prediction state shard")
        path = _safe_relative(A.RESULT_DIR, record.get("relative_path"),
                              "prediction state shard")
        sidecar = path.with_suffix(".receipt.json")
        receipt = read_json(sidecar, "prediction state-shard receipt")
        _require(receipt == record
                 and receipt.get("receipt_digest")
                 == A.json_digest(receipt, ("receipt_digest",)),
                 f"prediction shard receipt/index differs {seed}/{cell}/{state_id}")
        expected_bytes = EXPECTED_CANDIDATES * HORIZONS * TOKENS * TOKEN_DIM * 2
        _require(receipt.get("shape") == [EXPECTED_CANDIDATES, HORIZONS, TOKENS, TOKEN_DIM]
                 and receipt.get("dtype") == "float16"
                 and int(receipt.get("byte_count", -1)) == expected_bytes
                 and path.is_file() and path.stat().st_size == expected_bytes
                 and sha256_file(path) == receipt.get("sha256"),
                 f"prediction shard bytes differ {seed}/{cell}/{state_id}")
        stored = dict(record); stored["_path"] = str(path)
        state_shards[state_id] = stored
        storage += expected_bytes + sidecar.stat().st_size
    _require(set(state_shards) == {state.state_id for state in bundle.states},
             f"prediction state identities differ {seed}/{cell}")

    expected_branch_digests: list[str] = []
    for position, record in enumerate(branch_records):
        state_index, candidate_index = divmod(position, EXPECTED_CANDIDATES)
        state = sorted(bundle.states, key=lambda value: value.state_index)[state_index]
        row = bundle.row_by_pair[(state.state_id, candidate_index)]
        expected_branch_digests.append(str(row["branch_identity_digest"]))
        _require(record.get("position") == position
                 and record.get("state_id") == state.state_id
                 and int(record.get("candidate_index", -1)) == candidate_index
                 and record.get("candidate") == row.get("candidate")
                 and record.get("branch_identity_digest") == row.get("branch_identity_digest")
                 and record.get("sha256") == state_shards[state.state_id]["sha256"],
                 f"prediction branch index differs {seed}/{cell}/{position}")
    _require(index.get("ordered_branch_identity_set_digest")
             == A.sequence_digest(expected_branch_digests),
             f"prediction branch order digest differs {seed}/{cell}")

    ledger_path = A.PREDICTION_DIR / f"seed_{seed}_{cell}.jsonl"
    checkpoint_receipt = read_json(
        A.PREDICTION_DIR / f"seed_{seed}_{cell}.receipt.json",
        f"checkpoint prediction receipt {seed}/{cell}")
    _require(checkpoint_receipt.get("receipt_digest")
             == A.json_digest(checkpoint_receipt, ("receipt_digest",))
             and checkpoint_receipt.get("complete") is True
             and checkpoint_receipt.get("predictions_index_digest")
             == index["predictions_index_digest"]
             and checkpoint_receipt.get("checkpoint_sha256") == checkpoint_sha256
             and ledger_path.is_file()
             and checkpoint_receipt.get("ledger_sha256") == sha256_file(ledger_path),
             f"checkpoint prediction receipt differs {seed}/{cell}")
    _require(checkpoint_receipt == frozen_checkpoint_receipt,
             f"checkpoint receipt bytes are not the receipt bound by B/C {seed}/{cell}")
    storage += (directory / "predictions_index.json").stat().st_size
    storage += ledger_path.stat().st_size
    input_digest = legacy_digest({
        "predictions_index_digest": index["predictions_index_digest"],
        "state_shard_receipt_digests": [
            state_shards[state.state_id]["receipt_digest"]
            for state in sorted(bundle.states, key=lambda value: value.state_index)],
    })
    return PredictionPackage(seed, cell, checkpoint_sha256, index,
                             state_shards, input_digest, storage)


# --------------------------------------------------------------- scoring -----
def _model_scores(model: S.UtilityScorer, latent: torch.Tensor | None,
                  context: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        progress, safety, completion = model(latent, context)
        utility = S.composite(progress, safety, completion)
    return utility.detach().cpu().numpy().astype(np.float32)


def _atomic_f32(path: Path, values: np.ndarray) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    array = np.asarray(values, dtype=np.float32)
    with temporary.open("wb") as sink:
        sink.write(array.tobytes(order="C")); sink.flush(); os.fsync(sink.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return sha256_file(path), path.stat().st_size


def _preserve_invalid(path: Path, reason: str) -> str | None:
    if not path.exists():
        return None
    target_dir = OUT_DIR / "invalid_attempts"
    target_dir.mkdir(parents=True, exist_ok=True)
    digest = sha256_file(path)
    target = target_dir / f"{path.name}.{digest[:12]}.{reason}.{time.time_ns()}"
    shutil.move(str(path), str(target))
    RECOVERY_EVENTS.append({"source": str(path), "preserved_as": str(target),
                            "sha256": digest, "reason": reason})
    return str(target)


def _score_receipt(unit: str, input_digest: str, scorer: ScorerBundle,
                   score_path: Path) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": "go2_utility_scorer_development_score_shard_receipt_v1_2",
        "status": STATUS, "complete": True, "unit": unit,
        "input_digest": input_digest,
        "scorer_package_sha256": scorer.package_sha256,
        "scorer_contract_v1_2_digest": contract_digest(),
        "rows": EXPECTED_BRANCHES, "shape": [EXPECTED_BRANCHES],
        "dtype": "float32", "path": str(score_path.relative_to(OUT_DIR)),
        "sha256": sha256_file(score_path), "byte_count": score_path.stat().st_size,
    }
    value["receipt_digest"] = legacy_digest(value)
    return value


def validate_existing_result(result: Mapping[str, Any], spec_digest: str,
                             scorer: ScorerBundle) -> None:
    """Require a terminal report and every score shard it claims to bind."""

    _require(result.get("result_digest")
             == legacy_digest(result, ("result_digest",))
             and result.get("complete") is True
             and result.get("development_transfer_spec_digest") == spec_digest
             and result.get("scorer_package_sha256") == scorer.package_sha256,
             "existing development-transfer result differs")
    receipts = result.get("score_shard_receipts")
    expected_units = {"no_latent", "true_latent"} | {
        f"seed_{seed}_{cell}" for seed in SEEDS for cell in CELLS
    }
    _require(isinstance(receipts, list) and len(receipts) == len(expected_units),
             "existing transfer result has an incomplete score-shard ledger")
    observed: set[str] = set()
    for receipt in receipts:
        _require(isinstance(receipt, Mapping),
                 "existing score-shard receipt is not an object")
        unit = str(receipt.get("unit"))
        _require(unit in expected_units and unit not in observed,
                 "existing score-shard units are duplicated or unregistered")
        observed.add(unit)
        _require(receipt.get("receipt_digest")
                 == legacy_digest(receipt, ("receipt_digest",))
                 and receipt.get("complete") is True
                 and receipt.get("scorer_package_sha256") == scorer.package_sha256
                 and receipt.get("scorer_contract_v1_2_digest") == contract_digest()
                 and receipt.get("shape") == [EXPECTED_BRANCHES]
                 and receipt.get("dtype") == "float32",
                 f"existing {unit} score-shard receipt differs")
        path = _safe_relative(OUT_DIR, receipt.get("path"),
                              f"existing {unit} score shard")
        sidecar = SCORE_DIR / f"{unit}.receipt.json"
        _require(path.is_file()
                 and path.stat().st_size == EXPECTED_BRANCHES * 4
                 and path.stat().st_size == receipt.get("byte_count")
                 and sha256_file(path) == receipt.get("sha256")
                 and sidecar.is_file()
                 and read_json(sidecar, f"existing {unit} score receipt")
                 == dict(receipt),
                 f"existing {unit} score-shard bytes differ")
    _require(observed == expected_units,
             "existing transfer result omits a registered score-shard unit")


def _existing_scores(unit: str, input_digest: str, scorer: ScorerBundle
                     ) -> tuple[np.ndarray, dict[str, Any]] | None:
    score_path = SCORE_DIR / f"{unit}.f32"
    receipt_path = SCORE_DIR / f"{unit}.receipt.json"
    if not score_path.exists() and not receipt_path.exists():
        return None
    try:
        receipt = read_json(receipt_path, f"{unit} score receipt")
        _require(receipt.get("receipt_digest")
                 == legacy_digest(receipt, ("receipt_digest",))
                 and receipt.get("complete") is True
                 and receipt.get("unit") == unit
                 and receipt.get("input_digest") == input_digest
                 and receipt.get("scorer_package_sha256") == scorer.package_sha256
                 and receipt.get("shape") == [EXPECTED_BRANCHES]
                 and receipt.get("dtype") == "float32"
                 and score_path.stat().st_size == EXPECTED_BRANCHES * 4
                 and receipt.get("sha256") == sha256_file(score_path),
                 f"invalid existing {unit} score shard")
        return np.fromfile(score_path, dtype=np.float32), receipt
    except (DevelopmentTransferRefused, OSError, ValueError):
        _preserve_invalid(score_path, "invalid_score")
        _preserve_invalid(receipt_path, "invalid_receipt")
        return None


def _finish_scores(unit: str, input_digest: str, scorer: ScorerBundle,
                   values: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    score_path = SCORE_DIR / f"{unit}.f32"
    receipt_path = SCORE_DIR / f"{unit}.receipt.json"
    _atomic_f32(score_path, values)
    receipt = _score_receipt(unit, input_digest, scorer, score_path)
    atomic_json(receipt_path, receipt)
    return np.asarray(values, dtype=np.float32), receipt


def score_no_latent(rows: Sequence[Mapping[str, Any]], scorer: ScorerBundle,
                    context: torch.Tensor) -> tuple[np.ndarray, dict[str, Any]]:
    input_digest = legacy_digest({
        "branch_identity_digests": [row["branch_identity_digest"] for row in rows],
        "action_goal_sha256": S.tensor_digest(context),
        "model_state_digest": scorer.package["final_state_digests"]["no_latent"],
    })
    prior = _existing_scores("no_latent", input_digest, scorer)
    if prior is not None:
        return prior
    return _finish_scores("no_latent", input_digest, scorer,
                          _model_scores(scorer.no_latent, None, context))


def score_true_latents(bundle: A.StageABundle, rows: Sequence[Mapping[str, Any]],
                       scorer: ScorerBundle, context: torch.Tensor
                       ) -> tuple[np.ndarray, dict[str, Any]]:
    records = [bundle.horizon_records[str(row["branch_identity_digest"])] for row in rows]
    input_digest = legacy_digest({
        "stage_a_latents_index_digest": bundle.latent_index_digest,
        "target_receipts": [record["latent_shard_receipt_digest"] for record in records],
        "normalisation": "f16->f32->F.layer_norm(1024)->mean_tokens",
        "model_state_digest": scorer.package["final_state_digests"]["latent"],
    })
    prior = _existing_scores("true_latent", input_digest, scorer)
    if prior is not None:
        return prior
    scores = np.empty(EXPECTED_BRANCHES, dtype=np.float32)
    for state_index in range(EXPECTED_STATES):
        start = state_index * EXPECTED_CANDIDATES
        batch = np.stack([
            A._read_f16_shard(records[position])
            for position in range(start, start + EXPECTED_CANDIDATES)], axis=0)
        tokens = F.layer_norm(torch.from_numpy(batch.astype(np.float32)), (TOKEN_DIM,))
        latent = tokens.mean(dim=2)
        scores[start:start + EXPECTED_CANDIDATES] = _model_scores(
            scorer.latent, latent, context[start:start + EXPECTED_CANDIDATES])
    return _finish_scores("true_latent", input_digest, scorer, scores)


def score_prediction_package(package: PredictionPackage, bundle: A.StageABundle,
                             scorer: ScorerBundle, context: torch.Tensor
                             ) -> tuple[np.ndarray, dict[str, Any]]:
    unit = f"seed_{package.seed}_{package.cell}"
    input_digest = legacy_digest({
        "prediction_package_input_digest": package.input_digest,
        "normalisation": "already-normalised-f16->f32->mean_tokens",
        "model_state_digest": scorer.package["final_state_digests"]["latent"],
    })
    prior = _existing_scores(unit, input_digest, scorer)
    if prior is not None:
        return prior
    scores = np.empty(EXPECTED_BRANCHES, dtype=np.float32)
    for state in sorted(bundle.states, key=lambda value: value.state_index):
        record = package.state_shards[state.state_id]
        shape = (EXPECTED_CANDIDATES, HORIZONS, TOKENS, TOKEN_DIM)
        tokens = np.asarray(np.memmap(record["_path"], mode="r", dtype=np.float16,
                                     shape=shape), dtype=np.float32)
        latent = torch.from_numpy(tokens.mean(axis=2, dtype=np.float32))
        start = state.state_index * EXPECTED_CANDIDATES
        scores[start:start + EXPECTED_CANDIDATES] = _model_scores(
            scorer.latent, latent, context[start:start + EXPECTED_CANDIDATES])
    return _finish_scores(unit, input_digest, scorer, scores)


# --------------------------------------------------------------- metrics -----
def _mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def state_metrics(rows: Sequence[Mapping[str, Any]], scores: np.ndarray
                 ) -> list[dict[str, Any]]:
    _require(len(rows) == len(scores) == EXPECTED_BRANCHES,
             "metric input has the wrong branch count")
    output: list[dict[str, Any]] = []
    for state_index in range(EXPECTED_STATES):
        start = state_index * EXPECTED_CANDIDATES
        selected = rows[start:start + EXPECTED_CANDIDATES]
        truth = np.asarray([row["utility"] for row in selected], dtype=np.float64)
        predicted = np.asarray(scores[start:start + EXPECTED_CANDIDATES],
                               dtype=np.float64)
        order = np.argsort(-predicted, kind="mergesort")
        chosen = int(order[0])
        best = truth == truth.max()
        spread = float(truth.max() - truth.min())
        pair_correct = pair_considered = 0
        for left in range(EXPECTED_CANDIDATES):
            for right in range(left + 1, EXPECTED_CANDIDATES):
                true_gap = float(truth[left] - truth[right])
                if abs(true_gap) <= SCORE_TIE_TOLERANCE:
                    continue
                pair_considered += 1
                pair_correct += int(
                    float(predicted[left] - predicted[right]) * true_gap > 0)
        output.append({
            "state_index": state_index,
            "state_id": str(selected[0]["state_id"]),
            "family": str(selected[0]["family"]),
            "normalised_rank_regret": (
                0.0 if spread <= 0 else float((truth.max() - truth[chosen]) / spread)),
            "absolute_rank_regret": float(truth.max() - truth[chosen]),
            "realised_selected_utility": float(truth[chosen]),
            "spearman_rank_correlation": S.spearman(truth, predicted),
            "top1_recovery": float(best[chosen]),
            "top3_recovery": float(np.any(best[order[:3]])),
            "pairwise_ordering_accuracy": (
                pair_correct / pair_considered if pair_considered else float("nan")),
            "pairs_considered": pair_considered,
            "candidate_score_spread": float(predicted.max() - predicted.min()),
            "scorer_tie_rate": float(
                np.sum(np.abs(predicted - predicted.max())
                       <= SCORE_TIE_TOLERANCE) > 1),
            "selected_candidate_index": chosen,
            "selected_candidate": str(selected[chosen]["candidate"]),
        })
    return output


def aggregate_metrics(states: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_family: dict[str, dict[str, float]] = {}
    for family in FAMILIES:
        selected = [row for row in states if row["family"] == family]
        _require(selected, f"development family {family} is absent")
        per_family[family] = {
            metric: _mean([float(row[metric]) for row in selected])
            for metric in METRIC_DIRECTIONS
        }
        per_family[family]["states"] = len(selected)
    equal_family = {
        metric: _mean([per_family[family][metric] for family in FAMILIES])
        for metric in METRIC_DIRECTIONS
    }
    corpus = {
        metric: _mean([float(row[metric]) for row in states])
        for metric in METRIC_DIRECTIONS
    }
    return {"equal_family": equal_family, "corpus_weighted": corpus,
            "per_family": per_family, "per_state": list(states)}


def t_interval(values: Sequence[float]) -> dict[str, Any]:
    _require(len(values) == 8,
             "paired treatment interval requires exactly eight registered seeds")
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    defined = array[finite]
    complete = bool(finite.all())
    mean = float(defined.mean()) if len(defined) else None
    sd = float(defined.std(ddof=1)) if len(defined) > 1 else None
    interval = None
    if complete:
        assert mean is not None and sd is not None
        half = T_CRITICAL_95_DF7 * sd / math.sqrt(8)
        interval = [mean - half, mean + half]
    return {
        "values_by_seed": [float(value) if math.isfinite(float(value)) else None
                           for value in array],
        "mean": mean,
        "sample_standard_deviation": sd,
        "two_sided_95_t_interval": interval,
        "registered_n": 8,
        "defined_n": int(finite.sum()),
        "degrees_of_freedom": 7 if complete else None,
        "interval_available": complete,
        "unavailable_reason": (None if complete else
                               "one or more registered seed values undefined"),
    }


def _benefit(one_step: float, rollout: float, direction: str) -> float:
    if direction == "lower":
        return float(one_step - rollout)
    return float(rollout - one_step)


def paired_factorial(cells: Mapping[int, Mapping[str, Mapping[str, Any]]],
                     weighting: str, metric: str,
                     family: str | None = None) -> dict[str, Any]:
    direction = METRIC_DIRECTIONS[metric]
    def value(seed: int, cell: str) -> float:
        result = cells[seed][cell]
        if family is None:
            return float(result[weighting][metric])
        return float(result["per_family"][family][metric])
    rgb = [_benefit(value(seed, "rgb_one_step"), value(seed, "rgb_rollout"),
                    direction) for seed in SEEDS]
    prop = [_benefit(value(seed, "proprio_one_step"),
                     value(seed, "proprio_rollout"), direction) for seed in SEEDS]
    main = [(left + right) / 2 for left, right in zip(rgb, prop)]
    interaction = [right - left for left, right in zip(rgb, prop)]
    return {"sign_convention": (
                "positive means rollout benefit" if direction != "descriptive"
                else "rollout minus one-step; descriptive, no benefit direction"),
            "B_RGB": t_interval(rgb), "B_prop": t_interval(prop),
            "M": t_interval(main), "J": t_interval(interaction)}


def analyse_cells(cells: Mapping[int, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    cell_means: dict[str, Any] = {}
    for cell in CELLS:
        cell_means[cell] = {}
        for weighting in ("equal_family", "corpus_weighted"):
            cell_means[cell][weighting] = {
                metric: t_interval([cells[seed][cell][weighting][metric]
                                    for seed in SEEDS])
                for metric in METRIC_DIRECTIONS
            }
        cell_means[cell]["per_family"] = {
            family: {metric: t_interval([
                cells[seed][cell]["per_family"][family][metric]
                for seed in SEEDS]) for metric in METRIC_DIRECTIONS}
            for family in FAMILIES
        }
    paired = {
        weighting: {
            metric: paired_factorial(cells, weighting, metric)
            for metric in METRIC_DIRECTIONS
        } for weighting in ("equal_family", "corpus_weighted")
    }
    per_family_principal = {
        family: {
            metric: paired_factorial(cells, "equal_family", metric, family)
            for metric in ("normalised_rank_regret", "realised_selected_utility")
        } for family in FAMILIES
    }
    return {"cell_means_across_seeds": cell_means,
            "paired_seed_factorial": paired,
            "per_family_principal_factorial": per_family_principal}


def safe_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def synthetic_self_test() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    scores: list[float] = []
    for state_index in range(EXPECTED_STATES):
        family = FAMILIES[state_index % len(FAMILIES)]
        for candidate_index in range(EXPECTED_CANDIDATES):
            rows.append({"state_id": f"state-{state_index}", "family": family,
                         "candidate": f"c{candidate_index}",
                         "utility": float(candidate_index)})
            scores.append(float(candidate_index))
    states = state_metrics(rows, np.asarray(scores, dtype=np.float32))
    aggregate = aggregate_metrics(states)
    _require(aggregate["equal_family"]["normalised_rank_regret"] == 0.0
             and aggregate["equal_family"]["top1_recovery"] == 1.0
             and aggregate["equal_family"]["pairwise_ordering_accuracy"] == 1.0,
             "synthetic perfect-ranking test failed")
    interval = t_interval([0.0] * 8)
    _require(interval["two_sided_95_t_interval"] == [0.0, 0.0],
             "synthetic t-interval test failed")
    return {"pass": True, "perfect_ranking": aggregate["equal_family"],
            "zero_interval": interval}


# ------------------------------------------------------------------- main ----
def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true",
                        help="pure synthetic metrics only; open no scientific artifact")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(safe_json(synthetic_self_test()), indent=2))
        return 0

    started = time.time()
    scorer = validate_qualified_scorer()
    spec = prospective_spec(scorer)
    freeze_spec(spec)
    if RESULT_PATH.is_file():
        try:
            existing = read_json(RESULT_PATH, "development transfer result")
            validate_existing_result(
                existing, spec["development_transfer_spec_digest"], scorer)
            print(json.dumps(existing, indent=2))
            return 0
        except (DevelopmentTransferRefused, OSError, ValueError) as exc:
            _preserve_invalid(RESULT_PATH, "invalid_terminal_result")
            RECOVERY_EVENTS.append({
                "source": str(RESULT_PATH),
                "reason": f"terminal_result_revalidation_failed:{exc}",
                "action": "preserved_result_then_resumed_exact_missing_score_units",
            })

    bundle, target_verification = validate_stage_a()
    rows = ordered_rows(bundle)
    context = action_goal(rows)
    bc_result, checkpoint_inventory, checkpoint_receipts = validate_bc_result(bundle)

    score_receipts: list[dict[str, Any]] = []
    no_latent_scores, receipt = score_no_latent(rows, scorer, context)
    score_receipts.append(receipt)
    true_scores, receipt = score_true_latents(bundle, rows, scorer, context)
    score_receipts.append(receipt)
    no_latent = aggregate_metrics(state_metrics(rows, no_latent_scores))
    true_latent = aggregate_metrics(state_metrics(rows, true_scores))

    packages: dict[tuple[int, str], PredictionPackage] = {}
    cells: dict[int, dict[str, dict[str, Any]]] = {seed: {} for seed in SEEDS}
    for seed in SEEDS:
        for cell in CELLS:
            package = validate_prediction_package(
                bundle, seed, cell, checkpoint_inventory[(seed, cell)],
                checkpoint_receipts[(seed, cell)])
            packages[(seed, cell)] = package
            scores, receipt = score_prediction_package(package, bundle, scorer, context)
            score_receipts.append(receipt)
            cells[seed][cell] = aggregate_metrics(state_metrics(rows, scores))

    analysis = analyse_cells(cells)
    output: dict[str, Any] = {
        "schema": "go2_utility_scorer_counterfactual_development_transfer_result_v1_2",
        "status": STATUS, "complete": True,
        "exploratory_fixed_development_states": True,
        "development_transfer_spec_digest": spec["development_transfer_spec_digest"],
        "scorer_contract_v1_2_digest": contract_digest(),
        "qualification_report_digest": scorer.qualification["qualification_report_digest"],
        "scorer_package_sha256": scorer.package_sha256,
        "scorer_selector_successor_bindings":
            spec["scorer_selector_successor_bindings"],
        "frozen_inputs": spec["frozen_inputs"],
        "scope": {"states": EXPECTED_STATES, "branches": EXPECTED_BRANCHES,
                  "candidates_per_state": EXPECTED_CANDIDATES,
                  "checkpoint_prediction_packages": EXPECTED_CHECKPOINTS},
        "true_latent_scorer_diagnostic": true_latent,
        "no_latent_baseline": no_latent,
        "true_latent_vs_no_latent_equal_family": {
            "sign_convention": "positive means true-latent scorer benefit; spread is descriptive",
            "benefit": {
                metric: _benefit(no_latent["equal_family"][metric],
                                 true_latent["equal_family"][metric], direction)
                for metric, direction in METRIC_DIRECTIONS.items()},
        },
        "cells_by_seed": cells,
        "exploratory_paired_seed_analysis": analysis,
        "interpretation_boundary": {
            "scorer_qualified_on_independent_true_latent_calibration": True,
            "development_states_fixed": 20,
            "training_seed_replication_units": 8,
            "state_or_branch_count_used_as_model_replication": False,
            "planning_claim_authorised": False,
            "final_benchmark_claim_authorised": False,
        },
        "no_leakage": {
            "predictor_checkpoints_loaded": 0, "predictors_rerun": False,
            "new_branches": 0, "new_frames": 0, "new_true_latents": 0,
            "new_predictor_latents": 0, "model_specific_calibration": False,
            "oracle_labels_used_as_scorer_input": False,
            "oracle_labels_used_for_post_score_evaluation_only": True,
        },
        "score_shard_receipts": score_receipts,
        "recovery": {"events": RECOVERY_EVENTS,
                     "invalid_or_interrupted_attempts_preserved": bool(RECOVERY_EVENTS),
                     "resume_unit": "one true/no-latent/checkpoint score vector"},
        "runtime": {"total_wall_time_s": round(time.time() - started, 3)},
        "storage": {
            "source_target_latent_bytes_read_only": target_verification["bytes"],
            "source_prediction_package_bytes_read_only": sum(
                package.storage_bytes for package in packages.values()),
            "score_shards_and_receipts_bytes": sum(
                (OUT_DIR / receipt["path"]).stat().st_size
                + (SCORE_DIR / f"{receipt['unit']}.receipt.json").stat().st_size
                for receipt in score_receipts),
            "scope_note": "explicit bound files only; no recursive custody-root traversal",
        },
        "source_bc_result_digest": bc_result["report_digest"],
    }
    output = safe_json(output)
    output["result_digest"] = legacy_digest(output)
    atomic_json(RESULT_PATH, output)
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
