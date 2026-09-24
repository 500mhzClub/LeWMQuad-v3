#!/usr/bin/env python3
"""Frozen planning analysis for the 32 epoch-21 factorial predictors.

This is the only stage that may open the scientific predictor checkpoints.  It
therefore fails closed until both preceding gates are durably satisfied:

* the single shared utility-scorer package has a digest-matched qualification
  report whose frozen criteria all pass; and
* the complete 200-state/2,400-branch final corpus and its H=1..4 latent blobs
  pass their identity, completeness and oracle-identifiability checks.

Predictor inference receives an allow-listed :class:`PlanningRow`, never a
branch row.  That type contains observed context identity, observed
proprioception/control history, the snapshot-time goal binding and hypothetical
post-slew actions only.  Realised RGB/latents, oracle components, utility,
future proprioception and simulator state cannot cross the inference boundary.

Each checkpoint writes an append-only prediction ledger.  A complete ledger is
reused only when its receipt binds the exact checkpoint, scorer package, final
corpus, latent blobs and ordered state/candidate keys.  A verified partial
prefix resumes at the first missing record.  A corrupt or differently-bound
attempt is preserved before a clean ledger is reconstructed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import dev_checkpoint_v1 as CK  # noqa: E402
from scripts import dev_action_slew_reconstruction_v1 as SLEW  # noqa: E402
from scripts import dev_proprio_predictor_v1 as P  # noqa: E402
from scripts import run_dev_proprio_factorial_driver_v1 as D  # noqa: E402
from scripts import train_go2_utility_scorer_v1_2 as S  # noqa: E402
from lewm.oracle.go2_scorer_contract_v1_2 import (  # noqa: E402
    CANDIDATE_BANK_DIGEST,
    SCORER,
    contract,
    contract_digest,
    preprocess_contract_digest,
    render_contract_digest,
    target_encoder_digest,
)
from lewm.oracle.go2_textured_v03_renderer import (  # noqa: E402
    renderer_contract_digest as textured_v03_renderer_contract_digest,
)

STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
OUT_ROOT = ROOT / ".generated/go2_branch_corpus_v1_2"
FINAL_DIR = OUT_ROOT / "final_eval"
PACKAGE_DIR = ROOT / ".generated/go2_utility_scorer_v1_2"
RESULT_DIR = ROOT / ".generated/go2_planning_result_v1_2"
PREDICTION_DIR = RESULT_DIR / "predictions"

FROZEN_CONFIRMATORY_COMMIT = "443e5914694a533534486b629e95ec15f8df9b7a"
FROZEN_CONFIRMATORY_REPORT_DIGEST = (
    "60b0bb2d0b13ba47eac5e306c33d97dcfdce31102870edfc50b01f7f9b247161"
)
FROZEN_RUN_PACKAGE_DIGEST = (
    "cf0456bef0cbe7cd8f2cd666b600f91ebf845f6156d180569edf36be53552991"
)
CONFIRMATORY_REPORT_PATH = D.OUT / "final_analysis.json"
RUN_PACKAGE_PATH = D.PROPRIO / "scientific_run_package.json"
INITIAL_RECEIPT_PATH = D.OUT / "launch_authorisation.json"
CONTINUATION_RECEIPT_PATH = D.OUT / "continuation_authorisation.json"

FROZEN_N = 8
EXPECTED_STATES = 200
EXPECTED_BRANCHES = 2_400
EXPECTED_FAMILIES = 8
STATES_PER_FAMILY = 25
EXPECTED_CANDIDATES = 12
FROZEN_INFERENCE_BATCH = 12
MAX_H = 4
TOKENS = 768
TOKEN_DIM = 1_024
CONTEXT_SLOTS = 3
ACTION_DIM_PER_BLOCK = 10
ACTION_GOAL_DIM = 43
TIE_TOLERANCE = 0.02
# The scorer contract uses the same frozen tolerance both to omit
# oracle-indistinguishable pairwise comparisons and to identify score ties.
SCORE_TIE_TOLERANCE = TIE_TOLERANCE
DIAGNOSTIC_FAMILY = "local_composite_motifs"
FROZEN_PREDICTOR_SOURCE_PATHS = (
    "scripts/dev_checkpoint_v1.py",
    "scripts/dev_proprio_predictor_v1.py",
    "scripts/run_dev_proprio_factorial_driver_v1.py",
    "scripts/run_dev_v03_temporal_action_jepa_v1.py",
    "scripts/dev_action_slew_reconstruction_v1.py",
    "scripts/build_dev_v03_proprio_action_manifest_v1.py",
)

METRIC_DIRECTIONS = {
    "normalised_rank_regret": "lower",
    "absolute_rank_regret": "lower",
    "realised_selected_utility": "higher",
    "spearman_rank_correlation": "higher",
    "top1_recovery": "higher",
    "top3_recovery": "higher",
    "pairwise_ordering_accuracy": "higher",
    "candidate_score_spread": "descriptive_higher",
    "scorer_tie_rate": "lower",
}


class PlanningRefused(RuntimeError):
    """A frozen binding, completeness condition or sequential gate failed."""


class FinalCorpusGateFailed(PlanningRefused):
    """The frozen final-corpus gate failed; predictor loading is prohibited."""

    def __init__(self, report: dict[str, Any]) -> None:
        self.report = report
        failed = [name for name, passed in report["gate"]["components"].items()
                  if not passed]
        super().__init__("final corpus identifiability gate failed: " + ", ".join(failed))


@dataclass(frozen=True)
class PlanningRow:
    """The complete and exclusive model-side input for one candidate.

    Deliberately absent: horizon paths/latents, progress, safety, completion,
    utility, branch validity/outcome, pose, velocity and future proprioception.
    """

    position: int
    state_id: str
    state_index: int
    family: str
    candidate: str
    candidate_index: int
    context_index: int
    action_blocks: tuple[tuple[float, ...], ...]
    proprio_history: tuple[tuple[float, ...], ...]
    control_history: tuple[tuple[float, ...], ...]
    goal_binding: tuple[float, float, float]


@dataclass(frozen=True)
class CorpusBundle:
    manifest: dict[str, Any]
    receipt: dict[str, Any]
    index: dict[str, Any]
    rows: list[dict[str, Any]]
    planning_rows: list[PlanningRow]
    context_path: Path | None
    horizon_path: Path | None
    context_records: dict[str, dict[str, Any]]
    horizon_records: dict[str, dict[str, Any]]
    context_positions: dict[str, int]
    horizon_positions: dict[str, int]
    context_shape: tuple[int, ...]
    horizon_shape: tuple[int, ...]
    context_digest: str
    horizon_digest: str
    corpus_digest: str
    branch_rows_sha256: str


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


@dataclass(frozen=True)
class ObservedContextStore:
    """Observed context latents only; contains no outcome-side storage handle."""

    path: Path | None
    records: dict[str, dict[str, Any]]
    shape: tuple[int, ...]


# ------------------------------------------------------------------ digests --
def sha256_file(path: Path, block_size: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def json_digest(payload: Mapping[str, Any], omit: Iterable[str] = ()) -> str:
    omitted = set(omit)
    material = {key: value for key, value in payload.items() if key not in omitted}
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()


def sequence_digest(values: Sequence[Any]) -> str:
    return hashlib.sha256(json.dumps(list(values), sort_keys=True).encode()).hexdigest()


def scoring_implementation_bindings() -> dict[str, str]:
    """SHA-256 bind every source file executed in predictor/scorer inference."""

    relative_paths = (*FROZEN_PREDICTOR_SOURCE_PATHS,
                      "scripts/train_go2_utility_scorer_v1_2.py",
                      "lewm/oracle/go2_scorer_contract_v1_2.py",
                      "scripts/score_go2_predictors_v1_2.py")
    return {relative: sha256_file(ROOT / relative) for relative in relative_paths}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PlanningRefused(message)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _same_float(a: Any, b: Any, tolerance: float = 1e-7) -> bool:
    return _finite(a) and _finite(b) and abs(float(a) - float(b)) <= tolerance


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def read_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"missing {label}: {path}")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanningRefused(f"unreadable {label} {path}: {exc}") from exc
    _require(isinstance(payload, dict), f"{label} is not a JSON object: {path}")
    return payload


def read_jsonl_strict(path: Path, label: str) -> list[dict[str, Any]]:
    _require(path.is_file(), f"missing {label}: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PlanningRefused(
                    f"malformed {label} at {path}:{line_number}: {exc}") from exc
            _require(isinstance(row, dict),
                     f"non-object {label} record at {path}:{line_number}")
            records.append(row)
    return records


def verify_embedded_digest(payload: dict[str, Any], field: str, label: str,
                           *, required: bool = True) -> str | None:
    stored = payload.get(field)
    if stored is None:
        _require(not required, f"{label} has no {field}")
        return None
    actual = json_digest(payload, (field,))
    _require(actual == stored, f"{label} {field} mismatch: {actual} != {stored}")
    return str(stored)


# ------------------------------------------------------- scorer gate/package --
def validate_scorer_artifacts() -> dict[str, Any]:
    qualification_path = PACKAGE_DIR / "qualification.json"
    package_path = PACKAGE_DIR / "scorer_package.pt"
    qualification = read_json(qualification_path, "scorer qualification report")

    if "qualification_report_digest" in qualification:
        verify_embedded_digest(qualification, "qualification_report_digest",
                               "scorer qualification report")
    elif "qualification_digest" in qualification:
        verify_embedded_digest(qualification, "qualification_digest",
                               "scorer qualification report")
    else:
        raise PlanningRefused(
            "scorer qualification report has no verified self digest")
    _require(qualification.get("qualified") is True,
             "scorer qualification did not pass; predictor loading is prohibited")
    criteria = qualification.get("criteria")
    _require(isinstance(criteria, dict) and criteria,
             "scorer qualification has no frozen criteria")
    _require(all(value is True for value in criteria.values()),
             "one or more frozen scorer qualification criteria failed")
    _require(qualification.get("scorer_contract_v1_2_digest") == contract_digest(),
             "scorer qualification contract digest differs from v1.2")
    _require(int(qualification.get("fit_states", -1)) == 96,
             "scorer qualification does not bind exactly 96 fit states")
    _require(int(qualification.get("calibration_states", -1)) == 24,
             "scorer qualification does not bind exactly 24 calibration states")
    _require(qualification.get("scene_disjoint") is True,
             "scorer fit/calibration qualification is not scene-disjoint")
    _require(int(qualification.get("qualification_evaluations", -1)) == 1,
             "scorer qualification was not the single frozen evaluation")
    _require(qualification.get("epoch_selection_permitted") is False,
             "scorer qualification permits epoch selection")
    _require(int(qualification.get("predictor_checkpoints_loaded", -1)) == 0,
             "scientific predictor checkpoints were opened during scorer fitting")

    dominance = qualification.get("baseline_dominance_pairwise")
    if dominance is None:
        latent = qualification.get("latent_scorer", {}).get("calibration", {})
        no_latent = qualification.get("no_latent_baseline", {}).get("calibration", {})
        try:
            dominance = (latent["composite"]["pairwise_ordering_accuracy"]
                         - no_latent["composite"]["pairwise_ordering_accuracy"])
        except (KeyError, TypeError):
            dominance = None
    _require(_finite(dominance) and float(dominance) >= 0.05,
             "latent scorer does not beat the no-latent baseline by 0.05")

    _require(package_path.is_file(), f"missing frozen scorer package: {package_path}")
    package_sha = sha256_file(package_path)
    _require(qualification.get("scorer_package_sha256") == package_sha,
             "scorer package bytes differ from the qualified package")

    package_receipt_path = PACKAGE_DIR / "scorer_package_receipt.json"
    package_receipt = None
    _require(package_receipt_path.is_file(),
             "qualified scorer package has no frozen package receipt")
    if package_receipt_path.is_file():
        package_receipt = read_json(package_receipt_path, "scorer package receipt")
        if "receipt_digest" in package_receipt:
            verify_embedded_digest(package_receipt, "receipt_digest",
                                   "scorer package receipt")
        elif "scorer_package_receipt_digest" in package_receipt:
            verify_embedded_digest(package_receipt, "scorer_package_receipt_digest",
                                   "scorer package receipt")
        else:
            raise PlanningRefused("scorer package receipt has no verified self digest")
        _require(package_receipt.get("complete") is True,
                 "scorer package receipt is not complete")
        _require(package_receipt.get("qualified") is True,
                 "scorer package receipt is not qualified")
        _require(package_receipt.get("scorer_package_sha256") == package_sha,
                 "scorer package receipt binds different bytes")
        _require(package_receipt.get("scorer_contract_v1_2_digest") == contract_digest(),
                 "scorer package receipt binds a different scorer contract")
        for field, expected in (
            ("target_encoder_digest", target_encoder_digest()),
            ("target_encoder_checkpoint_sha256",
             contract()["target_encoder"]["checkpoint_sha256"]),
            ("render_contract_digest", render_contract_digest()),
            ("preprocess_contract_digest", preprocess_contract_digest()),
            ("preprocessing_digest", S.FROZEN_PREPROCESSING_DIGEST),
        ):
            _require(package_receipt.get(field) == expected,
                     f"scorer package receipt {field} differs")

    return {
        "qualification": qualification,
        "qualification_path": str(qualification_path),
        "qualification_sha256": sha256_file(qualification_path),
        "package_path": str(package_path),
        "package_sha256": package_sha,
        "package_bytes": package_path.stat().st_size,
        "package_receipt": package_receipt,
    }


def _encoder_sha(payload: Any) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    for key in ("checkpoint_sha256", "encoder_checkpoint_sha256",
                "target_encoder_checkpoint_sha256", "target_encoder_sha256",
                "weights_sha256"):
        value = payload.get(key)
        if isinstance(value, str) and len(value) == 64:
            return value
    for key in ("target_encoder", "encoder", "corpus_bindings", "bindings"):
        value = _encoder_sha(payload.get(key))
        if value is not None:
            return value
    return None


def validate_cross_stage_encoder(scorer_provenance: Mapping[str, Any],
                                 bundle: CorpusBundle) -> dict[str, Any]:
    final_encoder = bundle.index.get("encoder")
    final_sha = _encoder_sha(final_encoder)
    _require(final_sha is not None,
             "final latent index does not bind target-encoder weights by SHA-256")
    sources: dict[str, str] = {}
    qualification_sha = _encoder_sha(scorer_provenance.get("qualification"))
    if qualification_sha is not None:
        sources["qualification"] = qualification_sha
    receipt_sha = _encoder_sha(scorer_provenance.get("package_receipt"))
    if receipt_sha is not None:
        sources["scorer_package_receipt"] = receipt_sha
    _require(bool(sources),
             "scorer qualification/package receipt does not bind target-encoder weights")
    _require(all(value == final_sha for value in sources.values()),
             "scorer package and final corpus use different target-encoder weights")

    expected_encoder = contract()["target_encoder"]
    _require(final_sha == expected_encoder["checkpoint_sha256"],
             "final corpus target-encoder checkpoint differs from the v1.2 contract")
    checkpoint_path = Path(str(final_encoder.get("checkpoint_path", ""))).expanduser()
    expected_path = Path(str(expected_encoder["checkpoint"])).expanduser()
    _require(checkpoint_path.resolve() == expected_path.resolve(),
             "final latent index references a different target-encoder checkpoint path")
    _require(checkpoint_path.is_file()
             and checkpoint_path.stat().st_size
             == int(expected_encoder["checkpoint_byte_count"]),
             "bound target-encoder checkpoint is missing or has the wrong byte count")
    _require(sha256_file(checkpoint_path) == final_sha,
             "bound target-encoder checkpoint bytes differ")

    preprocess = bundle.index.get("preprocess")
    _require(isinstance(preprocess, str)
             and "dev_frozen_dense_representation_encoders_v1" in preprocess,
             "final latent index does not bind the frozen predictor preprocessing")
    _require(bundle.index.get("preprocessing_digest")
             == S.FROZEN_PREPROCESSING_DIGEST,
             "final latent index preprocessing digest differs")
    for label, payload in (("qualification", scorer_provenance.get("qualification")),
                           ("scorer package receipt",
                            scorer_provenance.get("package_receipt"))):
        if not isinstance(payload, Mapping):
            continue
        preprocessing_digest = payload.get("preprocessing_digest")
        if preprocessing_digest is not None:
            _require(preprocessing_digest == S.FROZEN_PREPROCESSING_DIGEST,
                     f"{label} preprocessing digest differs")
        target_digest = payload.get("target_encoder_digest")
        if target_digest is not None:
            _require(target_digest == target_encoder_digest(),
                     f"{label} target-encoder contract digest differs")
        render_digest = payload.get("render_contract_digest")
        if render_digest is not None:
            _require(render_digest == render_contract_digest(),
                     f"{label} rendering contract digest differs")
        preprocess_digest = payload.get("preprocess_contract_digest")
        if preprocess_digest is not None:
            _require(preprocess_digest == preprocess_contract_digest(),
                     f"{label} preprocess-contract digest differs")
    return {
        "target_encoder_checkpoint_sha256": final_sha,
        "matched_sources": sources,
        "preprocess": preprocess,
        "target_normalisation": bundle.index["target_normalisation"],
        "latent_token_layout": [MAX_H, TOKENS, TOKEN_DIM],
    }


def tensor_state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_scorers(provenance: dict[str, Any], device: torch.device
                 ) -> tuple[S.UtilityScorer, S.UtilityScorer, str]:
    package_path = Path(provenance["package_path"])
    # The package bytes were hashed and matched to a passing qualification above.
    payload = torch.load(package_path, map_location="cpu", weights_only=False)
    _require(payload.get("qualified") is True,
             "scorer package payload itself is not qualified")
    _require(payload.get("contract_digest") == contract_digest(),
             "scorer package payload binds a different contract")
    _require(payload.get("weights") == SCORER["weights"],
             "scorer package composite weights differ from the frozen contract")
    _require(payload.get("normalisation") == S.NORMALISATION,
             "scorer package normalisation differs from the frozen identity transforms")
    _require(payload.get("goal_binding")
             == "[sin(bearing_body_rad), cos(bearing_body_rad), range_m]",
             "scorer package goal-binding implementation differs")
    _require(payload.get("preprocess") == S.EXPECTED_PREPROCESS,
             "scorer package preprocessing implementation differs")
    _require(payload.get("target_normalisation") == S.EXPECTED_TARGET_NORMALISATION,
             "scorer package target normalisation differs")
    _require(int(payload.get("final_epoch", -1)) == int(SCORER["training"]["epochs"])
             and payload.get("epoch_selection") == "final_epoch_only_no_selection",
             "scorer package is not the fixed-budget final epoch")
    _require(payload.get("target_encoder_digest")
             == target_encoder_digest()
             and payload.get("target_encoder_checkpoint_sha256")
             == contract()["target_encoder"]["checkpoint_sha256"],
             "scorer package target-encoder binding differs")
    _require(payload.get("render_contract_digest")
             == render_contract_digest()
             and payload.get("preprocess_contract_digest")
             == preprocess_contract_digest()
             and payload.get("preprocessing_digest") == S.FROZEN_PREPROCESSING_DIGEST,
             "scorer package rendering/preprocessing binding differs")
    architecture = payload.get("architecture", {})
    expected_architecture = {
        "tokens": TOKENS,
        "token_dim": TOKEN_DIM,
        "horizons": MAX_H,
        "hidden": 512,
        "action_dim": 40,
        "goal_dim": 3,
    }
    _require(all(architecture.get(key) == value
                 for key, value in expected_architecture.items()),
             f"scorer package architecture differs: {architecture}")
    _require(payload.get("model_specific_calibration") in (None, False),
             "model-specific scorer calibration is prohibited")

    recorded_state_digests = payload.get("final_state_digests")
    _require(isinstance(recorded_state_digests, dict),
             "scorer package has no final component-state digests")
    for name in ("latent", "no_latent"):
        _require(name in payload
                 and S.state_dict_digest(payload[name]) == recorded_state_digests.get(name),
                 f"scorer package {name} state digest does not verify")
    package_receipt = provenance.get("package_receipt")
    if isinstance(package_receipt, Mapping):
        _require(package_receipt.get("final_state_digests") == recorded_state_digests,
                 "scorer package receipt binds different final states")

    latent = S.UtilityScorer(use_latent=True,
                             hidden=int(architecture["hidden"])).to(device)
    baseline = S.UtilityScorer(use_latent=False,
                               hidden=int(architecture["hidden"])).to(device)
    try:
        latent.load_state_dict(payload["latent"], strict=True)
        baseline.load_state_dict(payload["no_latent"], strict=True)
    except (KeyError, RuntimeError) as exc:
        raise PlanningRefused(f"scorer package state is incompatible: {exc}") from exc
    latent.eval()
    baseline.eval()
    state_digest = tensor_state_digest(latent)
    return latent, baseline, state_digest


# ------------------------------------------------------------ final corpus --
def _receipt_count(receipt: Mapping[str, Any], *names: str) -> int | None:
    for name in names:
        if name in receipt:
            try:
                return int(receipt[name])
            except (TypeError, ValueError):
                return None
    for container_name in ("expected", "actual", "counts"):
        container = receipt.get(container_name)
        if isinstance(container, Mapping):
            for name in names:
                if name in container:
                    try:
                        return int(container[name])
                    except (TypeError, ValueError):
                        return None
    return None


def _nested_receipt_count(receipt: Mapping[str, Any], container_name: str,
                          *names: str) -> int | None:
    container = receipt.get(container_name)
    if not isinstance(container, Mapping):
        return None
    for name in names:
        if name in container:
            try:
                return int(container[name])
            except (TypeError, ValueError):
                return None
    return None


def _matrix(value: Any, rows: int, columns: int, label: str) -> tuple[tuple[float, ...], ...]:
    _require(isinstance(value, list) and len(value) == rows,
             f"{label} must have shape ({rows},{columns})")
    converted: list[tuple[float, ...]] = []
    for row in value:
        _require(isinstance(row, list) and len(row) == columns,
                 f"{label} must have shape ({rows},{columns})")
        _require(all(_finite(item) for item in row), f"{label} contains non-finite values")
        converted.append(tuple(float(item) for item in row))
    return tuple(converted)


def make_planning_rows(rows: list[dict[str, Any]],
                       context_positions: Mapping[str, int]) -> list[PlanningRow]:
    planning: list[PlanningRow] = []
    for position, row in enumerate(rows):
        blocks = _matrix(row.get("action_blocks"), MAX_H, ACTION_DIM_PER_BLOCK,
                         f"{row.get('state_id')}|{row.get('candidate')} action_blocks")
        proprio = _matrix(row.get("proprio"),
                          CONTEXT_SLOTS * P.SAMPLES_PER_SLOT, P.PROPRIO_DIM,
                          f"{row.get('state_id')} proprio history")
        control = _matrix(row.get("control"),
                          CONTEXT_SLOTS * P.SAMPLES_PER_SLOT, P.CONTROL_DIM,
                          f"{row.get('state_id')} control history")
        goal = row.get("goal")
        _require(isinstance(goal, dict), f"{row.get('state_id')} has no goal binding")
        bearing, distance = goal.get("bearing_body_rad"), goal.get("range_m")
        _require(_finite(bearing) and _finite(distance),
                 f"{row.get('state_id')} has a non-finite planning goal binding")
        state_id = str(row["state_id"])
        _require(state_id in context_positions,
                 f"{state_id} is absent from the context-latent index")
        planning.append(PlanningRow(
            position=position,
            state_id=state_id,
            state_index=int(row["state_index"]),
            family=str(row["family"]),
            candidate=str(row["candidate"]),
            candidate_index=int(row["candidate_index"]),
            context_index=int(context_positions[state_id]),
            action_blocks=blocks,
            proprio_history=proprio,
            control_history=control,
            goal_binding=(math.sin(float(bearing)), math.cos(float(bearing)),
                          float(distance)),
        ))
    return planning


def _validate_blob(path: Path, shape: Sequence[int], expected_sha: Any,
                   label: str) -> str:
    _require(path.is_file(), f"missing {label} blob: {path}")
    elements = math.prod(int(value) for value in shape)
    expected_bytes = elements * np.dtype(np.float16).itemsize
    _require(path.stat().st_size == expected_bytes,
             f"{label} blob size {path.stat().st_size} != {expected_bytes}")
    actual_sha = sha256_file(path)
    _require(expected_sha == actual_sha,
             f"{label} blob digest mismatch: {actual_sha} != {expected_sha}")
    return actual_sha


def _safe_relative_artifact(relative: Any, label: str) -> Path:
    _require(isinstance(relative, str) and relative,
             f"{label} has no relative path")
    candidate = Path(relative)
    _require(not candidate.is_absolute() and ".." not in candidate.parts,
             f"{label} path escapes the final-corpus directory: {relative}")
    for part in candidate.parts:
        _require(part != "sealed" and not part.startswith("sealed_"),
                 f"{label} path enters prohibited sealed material")
    resolved = (FINAL_DIR / candidate).resolve()
    _require(resolved.is_relative_to(FINAL_DIR.resolve()),
             f"{label} path escapes the final-corpus directory")
    return resolved


def _shard_key(record: Mapping[str, Any], kind: str) -> str:
    for name in ("key", f"{kind}_key", "state_candidate_key"):
        value = record.get(name)
        if isinstance(value, str) and value:
            return value
    if kind == "context":
        value = record.get("state_id")
        if isinstance(value, str) and value:
            return value
    state_id, candidate = record.get("state_id"), record.get("candidate")
    if isinstance(state_id, str) and isinstance(candidate, str):
        return f"{state_id}|{candidate}"
    raise PlanningRefused(f"{kind} latent shard has no identity key")


def _validate_shards(records: Any, *, kind: str,
                     expected_keys: set[str], expected_shape: Sequence[int]
                     ) -> tuple[dict[str, dict[str, Any]], str]:
    _require(isinstance(records, list) and len(records) == len(expected_keys),
             f"{kind}_records must contain exactly {len(expected_keys)} shards")
    mapped: dict[str, dict[str, Any]] = {}
    digest_rows: list[dict[str, Any]] = []
    for position, entry in enumerate(records):
        _require(isinstance(entry, dict), f"{kind} shard record is not an object")
        key = _shard_key(entry, kind)
        _require(key in expected_keys, f"unknown {kind} shard key {key}")
        _require(key not in mapped, f"duplicate {kind} shard key {key}")
        shape = entry.get("shape")
        _require(shape == list(expected_shape),
                 f"{kind} shard {key} has shape {shape}, expected {list(expected_shape)}")
        relative = entry.get("relative_path", entry.get("path"))
        path = _safe_relative_artifact(relative, f"{kind} shard {key}")
        _validate_blob(path, shape, entry.get("sha256"), f"{kind} shard {key}")
        recorded_bytes = entry.get("byte_count", entry.get("bytes"))
        _require(int(recorded_bytes if recorded_bytes is not None else -1)
                 == path.stat().st_size,
                 f"{kind} shard {key} byte count differs")
        stored = dict(entry)
        stored["_resolved_path"] = str(path)
        mapped[key] = stored
        digest_rows.append({"position": position, "identity": key,
                            "sha256": entry["sha256"],
                            "byte_count": int(recorded_bytes), "shape": shape})
    _require(set(mapped) == expected_keys, f"{kind} shard set is incomplete")
    return mapped, sequence_digest(digest_rows)


def validate_final_corpus() -> CorpusBundle:
    manifest_path = FINAL_DIR / "state_manifest.json"
    rows_path = FINAL_DIR / "branch_rows.jsonl"
    receipt_path = FINAL_DIR / "corpus_receipt.json"
    index_path = FINAL_DIR / "latents_index.json"
    context_path = FINAL_DIR / "context.f16"
    horizon_path = FINAL_DIR / "horizon.f16"

    manifest = read_json(manifest_path, "final state identity manifest")
    state_manifest_digest = verify_embedded_digest(
        manifest, "state_manifest_digest", "final state identity manifest")
    _require(manifest.get("schema") == "go2_branch_corpus_v1_2_state_manifest"
             and manifest.get("complete") is True,
             "final state identity manifest is not a complete v1.2 manifest")
    _require(manifest.get("pool") == "final_eval", "state manifest is not final_eval")
    _require(manifest.get("genesis_backend") == "cpu",
             "final state manifest does not bind the qualified CPU backend")
    _require(manifest.get("candidate_bank_digest") == CANDIDATE_BANK_DIGEST,
             "final state manifest candidate-bank digest differs")
    frozen_contract = contract()
    for field, expected in (
        ("selection_digest", frozen_contract["corpus_selection_digest"]),
        ("oracle_v1_2_digest", frozen_contract["oracle_v1_2_digest"]),
        ("progress_contract_digest", frozen_contract["progress_target_digest"]),
        ("safety_contract_digest", frozen_contract["safety_target_digest"]),
        ("scorer_contract_v1_2_digest", contract_digest()),
        ("render_contract_digest", render_contract_digest()),
        ("textured_v03_renderer_contract_digest",
         textured_v03_renderer_contract_digest()),
        ("preprocess_contract_digest", preprocess_contract_digest()),
        ("preprocessing_digest", S.FROZEN_PREPROCESSING_DIGEST),
        ("target_encoder_digest", target_encoder_digest()),
        ("target_encoder_checkpoint_sha256",
         frozen_contract["target_encoder"]["checkpoint_sha256"]),
    ):
        _require(manifest.get(field) == expected,
                 f"final state manifest {field} differs")
    boundary = manifest.get("boundary_digest", manifest.get("boundary"))
    _require(boundary == S.FROZEN_BRANCH_BOUNDARY_DIGEST,
             "final state manifest branch boundary differs")

    states = manifest.get("states")
    _require(isinstance(states, list) and len(states) == EXPECTED_STATES,
             f"final state manifest must contain exactly {EXPECTED_STATES} states")
    state_ids = [str(state.get("state_id")) for state in states]
    _require(len(set(state_ids)) == EXPECTED_STATES,
             "final state manifest contains duplicate state identities")
    _require(len({str(state.get("scene_id")) for state in states}) == EXPECTED_STATES,
             "final state manifest is not one-state-per-scene")
    _require(len({str(state.get("episode_cluster_id")) for state in states})
             == EXPECTED_STATES,
             "final state manifest is not episode-cluster-disjoint")
    family_counts = Counter(str(state.get("family")) for state in states)
    _require(len(family_counts) == EXPECTED_FAMILIES,
             f"final corpus has {len(family_counts)} families, expected 8")
    _require(all(count == STATES_PER_FAMILY for count in family_counts.values()),
             f"final state family counts differ from 25: {dict(family_counts)}")
    state_by_id = {str(state["state_id"]): state for state in states}
    registered_branches: dict[tuple[str, int], dict[str, Any]] = {}
    registered_branch_digests: list[str] = []
    for expected_state_index, state in enumerate(states):
        _require(int(state.get("state_index", -1)) == expected_state_index,
                 "final state manifest state order/index differs")
        _require(not any(field in state for field in (
            "progress", "safety", "completion", "utility", "branch_outcome")),
            f"{state.get('state_id')} identity contains post-outcome fields")
        state_identity_payload = {
            "schema": "go2_branch_state_identity_v1_2",
            "selection_digest": manifest["selection_digest"],
            "scorer_contract_v1_2_digest": contract_digest(),
            "state": {key: value for key, value in state.items() if key not in {
                "state_identity_digest", "state_index", "candidate_indices",
                "branch_identities",
            }},
        }
        _require(state.get("state_identity_digest")
                 == json_digest(state_identity_payload),
                 f"{state.get('state_id')} state identity digest does not verify")
        candidates = state.get("candidate_indices")
        _require(candidates == list(range(EXPECTED_CANDIDATES)),
                 f"{state.get('state_id')} does not bind all twelve candidates in order")
        _require(isinstance(state.get("goal"), dict),
                 f"{state.get('state_id')} has no snapshot-time goal binding")
        identities = state.get("branch_identities")
        _require(isinstance(identities, list)
                 and len(identities) == EXPECTED_CANDIDATES,
                 f"{state.get('state_id')} has no complete pre-outcome branch identities")
        for candidate_index, identity in enumerate(identities):
            _require(isinstance(identity, dict),
                     f"{state.get('state_id')} branch identity is malformed")
            stored_digest = identity.get("branch_identity_digest")
            _require(isinstance(stored_digest, str)
                     and stored_digest == json_digest(
                         identity, ("branch_identity_digest",)),
                     f"{state.get('state_id')} branch identity digest does not verify")
            _require(identity.get("state_id") == state.get("state_id")
                     and identity.get("state_identity_digest")
                     == state.get("state_identity_digest")
                     and int(identity.get("candidate_index", -1)) == candidate_index
                     and identity.get("goal") == state.get("goal"),
                     f"{state.get('state_id')} branch identity binding differs")
            registered_branches[(str(state["state_id"]), candidate_index)] = identity
            registered_branch_digests.append(stored_digest)
    _require(len(registered_branches) == EXPECTED_BRANCHES,
             "final manifest does not register exactly 2,400 unique branches")
    _require(manifest.get("attempted_branch_count_registered") == EXPECTED_BRANCHES,
             "final manifest registered branch count differs")
    _require(manifest.get("branch_identity_set_digest")
             == sequence_digest(sorted(registered_branch_digests)),
             "final manifest branch identity set digest does not verify")
    candidate_appearances = manifest.get("candidate_appearances")
    _require(isinstance(candidate_appearances, dict)
             and len(candidate_appearances) == EXPECTED_CANDIDATES
             and all(int(count) == EXPECTED_STATES
                     for count in candidate_appearances.values()),
             "final manifest candidate appearances are not exactly 200 each")

    allocation_path = FINAL_DIR / "candidate_allocation_manifest.json"
    allocation = read_json(allocation_path, "final candidate allocation manifest")
    allocation_digest = verify_embedded_digest(
        allocation, "allocation_manifest_digest",
        "final candidate allocation manifest")
    _require(manifest.get("candidate_allocation_manifest_digest")
             == allocation_digest,
             "state manifest binds a different candidate allocation")
    _require(allocation.get("source_identity_manifest_digest")
             == manifest.get("pre_allocation_identity_manifest_digest")
             and allocation.get("candidate_bank_digest") == CANDIDATE_BANK_DIGEST,
             "final candidate allocation source/bank binding differs")
    assignments = allocation.get("assignments")
    _require(isinstance(assignments, list) and len(assignments) == EXPECTED_STATES,
             "final candidate allocation does not contain 200 state assignments")
    assigned = {str(item.get("state_id")): item for item in assignments}
    _require(len(assigned) == EXPECTED_STATES and set(assigned) == set(state_ids),
             "final candidate allocation state set differs")
    for state_id, state in state_by_id.items():
        assignment = assigned[state_id]
        _require(assignment.get("state_identity_digest")
                 == state.get("state_identity_digest")
                 and assignment.get("candidate_indices")
                 == list(range(EXPECTED_CANDIDATES)),
                 f"{state_id} final candidate allocation differs")

    receipt = read_json(receipt_path, "final corpus completion receipt")
    if "receipt_digest" in receipt:
        verify_embedded_digest(receipt, "receipt_digest", "final corpus receipt")
    _require(receipt.get("complete") is True, "final corpus receipt is not complete")
    _require(receipt.get("pool", "final_eval") == "final_eval",
             "final corpus receipt names the wrong pool")
    expected_states = _nested_receipt_count(receipt, "expected", "states", "state_count")
    actual_states = _nested_receipt_count(receipt, "actual", "states", "state_count")
    expected_rows = _nested_receipt_count(
        receipt, "expected", "rows", "branches", "row_count",
        "attempted_branches", "attempt_count", "attempted")
    actual_rows = _nested_receipt_count(
        receipt, "actual", "rows", "branches", "row_count",
        "attempted_branches", "attempt_count", "attempted")
    if expected_states is None and actual_states is None:
        expected_states = actual_states = _receipt_count(receipt, "states", "state_count")
    if expected_rows is None and actual_rows is None:
        expected_rows = actual_rows = _receipt_count(
            receipt, "rows", "branches", "row_count", "attempted_branches",
            "attempt_count", "attempted")
    _require(expected_states == actual_states == EXPECTED_STATES,
             "final corpus receipt does not certify expected=actual=200 states")
    _require(expected_rows == actual_rows == EXPECTED_BRANCHES,
             "final corpus receipt does not certify expected=actual=2,400 branches")
    _require(receipt.get("state_manifest_digest") == state_manifest_digest,
             "final corpus receipt binds a different state identity manifest")
    _require(receipt.get("candidate_allocation_manifest_digest")
             == allocation_digest,
             "final corpus receipt binds a different candidate allocation")
    for field, expected in (
        ("candidate_bank_digest", CANDIDATE_BANK_DIGEST),
        ("selection_digest", frozen_contract["corpus_selection_digest"]),
        ("oracle_v1_2_digest", frozen_contract["oracle_v1_2_digest"]),
        ("progress_contract_digest", frozen_contract["progress_target_digest"]),
        ("safety_contract_digest", frozen_contract["safety_target_digest"]),
        ("scorer_contract_v1_2_digest", contract_digest()),
        ("render_contract_digest", render_contract_digest()),
        ("textured_v03_renderer_contract_digest",
         textured_v03_renderer_contract_digest()),
        ("preprocess_contract_digest", preprocess_contract_digest()),
        ("target_encoder_digest", target_encoder_digest()),
        ("target_encoder_checkpoint_sha256",
         frozen_contract["target_encoder"]["checkpoint_sha256"]),
    ):
        _require(receipt.get(field) == expected, f"final corpus receipt {field} differs")
    receipt_preprocess = receipt.get("preprocessing_digest",
                                     receipt.get("preprocess_contract_digest"))
    _require(receipt_preprocess == S.FROZEN_PREPROCESSING_DIGEST,
             "final corpus receipt preprocessing binding differs")
    receipt_boundary = receipt.get("boundary_digest", receipt.get("boundary"))
    _require(receipt_boundary == S.FROZEN_BRANCH_BOUNDARY_DIGEST,
             "final corpus receipt branch boundary differs")

    branch_rows_sha = sha256_file(rows_path)
    _require(receipt.get("branch_rows_sha256") == branch_rows_sha,
             "final branch ledger bytes differ from the completion receipt")
    corpus_digest = receipt.get("corpus_digest")
    _require(isinstance(corpus_digest, str) and len(corpus_digest) == 64,
             "final corpus receipt has no frozen corpus_digest")
    corpus_digest_payload = receipt.get("corpus_digest_payload")
    _require(isinstance(corpus_digest_payload, dict)
             and json_digest(corpus_digest_payload) == corpus_digest,
             "final corpus receipt digest is not independently reproducible")

    rows = read_jsonl_strict(rows_path, "final branch ledger")
    _require(len(rows) == EXPECTED_BRANCHES,
             f"final branch ledger has {len(rows)} rows, expected 2,400")
    pair_keys: list[tuple[str, int]] = []
    candidate_names: dict[int, str] = {}
    valid_count = sum(bool(row.get("valid")) for row in rows)
    receipt_valid_count = _nested_receipt_count(
        receipt, "actual", "valid_branches", "valid_count", "valid")
    if receipt_valid_count is None:
        receipt_valid_count = _receipt_count(
            receipt, "valid_branches", "valid_count", "valid")
    _require(receipt_valid_count == valid_count,
             "final corpus receipt valid-branch count differs from its ledger")
    for row in rows:
        state_id = str(row.get("state_id"))
        _require(state_id in state_by_id, f"branch row references unknown state {state_id}")
        state = state_by_id[state_id]
        candidate_index = int(row.get("candidate_index", -1))
        _require(0 <= candidate_index < EXPECTED_CANDIDATES,
                 f"{state_id} has invalid candidate index {candidate_index}")
        _require(row.get("schema") == "go2_branch_corpus_v1_2_branch_row"
                 and row.get("record_complete") is True,
                 f"{state_id}|{candidate_index} is not a complete v1.2 branch record")
        pair_keys.append((state_id, candidate_index))
        name = str(row.get("candidate"))
        if candidate_index in candidate_names:
            _require(candidate_names[candidate_index] == name,
                     f"candidate index {candidate_index} has inconsistent names")
        candidate_names[candidate_index] = name
        _require(str(row.get("family")) == str(state.get("family")),
                 f"{state_id} branch family differs from identity manifest")
        _require(str(row.get("scene_id")) == str(state.get("scene_id")),
                 f"{state_id} branch scene differs from identity manifest")
        _require(str(row.get("episode_cluster_id"))
                 == str(state.get("episode_cluster_id"))
                 and int(row.get("source_step", -1)) == int(state.get("source_step", -2))
                 and int(row.get("state_index", -1)) == int(state.get("state_index", -2)),
                 f"{state_id}|{name} branch state binding differs")
        _require(row.get("state_identity_digest")
                 == state.get("state_identity_digest"),
                 f"{state_id}|{name} binds a different state identity")
        registered_identity = registered_branches[(state_id, candidate_index)]
        _require(row.get("branch_identity_digest")
                 == registered_identity.get("branch_identity_digest")
                 and name == registered_identity.get("candidate")
                 and row.get("primitives") == registered_identity.get("primitives"),
                 f"{state_id}|{name} differs from its pre-outcome branch identity")
        _require(row.get("state_manifest_digest") == state_manifest_digest,
                 f"{state_id}|{name} binds a different identity manifest")
        _require(row.get("oracle_v1_2_digest") == frozen_contract["oracle_v1_2_digest"],
                 f"{state_id}|{name} binds a different oracle")
        for aliases, expected in (
            (("scorer_contract_v1_2_digest",), contract_digest()),
            (("candidate_allocation_manifest_digest",), allocation_digest),
            (("candidate_bank_digest",), CANDIDATE_BANK_DIGEST),
            (("progress_contract_digest", "progress_target_digest"),
             frozen_contract["progress_target_digest"]),
            (("safety_contract_digest", "safety_target_digest"),
             frozen_contract["safety_target_digest"]),
            (("selection_digest",), frozen_contract["corpus_selection_digest"]),
            (("render_contract_digest",), render_contract_digest()),
            (("textured_v03_renderer_contract_digest",),
             textured_v03_renderer_contract_digest()),
            (("preprocess_contract_digest",), preprocess_contract_digest()),
            (("preprocessing_digest",), S.FROZEN_PREPROCESSING_DIGEST),
            (("target_encoder_digest",), target_encoder_digest()),
            (("target_encoder_checkpoint_sha256",),
             frozen_contract["target_encoder"]["checkpoint_sha256"]),
        ):
            observed = next((row.get(alias) for alias in aliases if alias in row), None)
            _require(observed == expected,
                     f"{state_id}|{name} binding {aliases[0]} differs")
        row_boundary = row.get("boundary_digest", row.get("boundary"))
        _require(row_boundary == S.FROZEN_BRANCH_BOUNDARY_DIGEST,
                 f"{state_id}|{name} branch boundary differs")
        row_digest = row.get("branch_row_digest")
        _require(isinstance(row_digest, str)
                 and row_digest == json_digest(row, ("branch_row_digest",)),
                 f"{state_id}|{name} branch-row digest does not verify")
        _require(isinstance(row.get("branch_identity_digest"), str)
                 and len(row["branch_identity_digest"]) == 64,
                 f"{state_id}|{name} has no frozen branch identity")
        _require(json.dumps(row.get("goal"), sort_keys=True)
                 == json.dumps(state.get("goal"), sort_keys=True),
                 f"{state_id}|{name} changed the snapshot-time goal binding")
        goal = state["goal"]
        expected_goal_input = [
            math.sin(float(goal["bearing_body_rad"])),
            math.cos(float(goal["bearing_body_rad"])),
            float(goal["range_m"]),
        ]
        observed_goal_input = row.get("goal_binding_input")
        _require(isinstance(observed_goal_input, list)
                 and len(observed_goal_input) == 3
                 and all(_same_float(left, right, 1e-7)
                         for left, right in zip(observed_goal_input,
                                                expected_goal_input)),
                 f"{state_id}|{name} numeric goal binding differs")
        previous_applied = row.get("previous_applied_command")
        _require(isinstance(previous_applied, list) and len(previous_applied) == 3
                 and all(_finite(value) for value in previous_applied)
                 and float(previous_applied[1]) == 0.0,
                 f"{state_id}|{name} has no restored previous applied command")
        post_slew_plan = row.get("candidate_post_slew_plan")
        _require(isinstance(post_slew_plan, list) and len(post_slew_plan) == MAX_H,
                 f"{state_id}|{name} has no four-block post-slew plan")
        try:
            post_slew_array = np.asarray(post_slew_plan, dtype=np.float64)
            _require(post_slew_array.shape == (MAX_H, SLEW.TICKS, 3),
                     f"{state_id}|{name} post-slew plan has the wrong shape")
            reconstructed_blocks: list[Any] = []
            reconstructed_previous = tuple(float(value)
                                           for value in previous_applied)
            primitives = row.get("primitives")
            _require(isinstance(primitives, list) and len(primitives) == MAX_H,
                     f"{state_id}|{name} candidate primitive plan differs")
            for primitive in primitives:
                reconstructed, reconstructed_previous = SLEW.reconstruct_block(
                    str(primitive), reconstructed_previous)
                reconstructed_blocks.append(reconstructed)
            reconstructed_array = np.asarray(reconstructed_blocks, dtype=np.float64)
            flattened_plan = post_slew_array[:, :, list(SLEW.ACTIVE_CHANNELS)].reshape(
                MAX_H, ACTION_DIM_PER_BLOCK)
            action_blocks = np.asarray(row.get("action_blocks"), dtype=np.float64)
        except (TypeError, ValueError, SLEW.LateralMotionRejected) as exc:
            raise PlanningRefused(
                f"{state_id}|{name} post-slew action plan is malformed") from exc
        _require(action_blocks.shape == flattened_plan.shape
                 and bool(np.isfinite(flattened_plan).all())
                 and bool(np.isfinite(action_blocks).all())
                 and bool(np.allclose(post_slew_array, reconstructed_array,
                                      rtol=0.0, atol=1e-12))
                 and bool(np.allclose(action_blocks, flattened_plan,
                                      rtol=0.0, atol=1e-7)),
                 f"{state_id}|{name} scorer action blocks differ from post-slew plan")
        masks = row.get("masks")
        _require(isinstance(masks, dict)
                 and masks.get("context_rgb_valid") == [True] * CONTEXT_SLOTS
                 and masks.get("observed_proprio_valid")
                 == [True] * (CONTEXT_SLOTS * P.SAMPLES_PER_SLOT)
                 and masks.get("observed_control_valid")
                 == [True] * (CONTEXT_SLOTS * P.SAMPLES_PER_SLOT)
                 and masks.get("future_proprio_available") == [False] * MAX_H,
                 f"{state_id}|{name} planning-time masks differ")
        timing = row.get("timing")
        _require(isinstance(timing, dict)
                 and timing.get("command_hz") == 10
                 and timing.get("ticks_per_block") == P.SAMPLES_PER_SLOT
                 and _same_float(timing.get("seconds_per_block"), 0.5)
                 and timing.get("context_boundary_offsets_blocks") == [-2, -1, 0]
                 and timing.get("target_horizons_blocks") == [1, 2, 3, 4],
                 f"{state_id}|{name} planning timing differs")
        if row.get("valid"):
            for key in ("progress", "safety", "completion", "utility"):
                _require(_finite(row.get(key)), f"valid {state_id}|{name} has no {key}")
            expected_utility = (float(row["progress"]) - 2.0 * float(row["safety"])
                                + 0.5 * float(row["completion"]))
            _require(_same_float(row["utility"], expected_utility, 1e-6),
                     f"{state_id}|{name} utility does not match oracle v1.2")
        else:
            _require(bool(row.get("invalid_reason")),
                     f"invalid {state_id}|{name} has no reason code")

    _require(len(set(pair_keys)) == EXPECTED_BRANCHES,
             "final branch ledger contains duplicate state/candidate pairs")
    expected_pairs = {(state_id, candidate) for state_id in state_ids
                      for candidate in range(EXPECTED_CANDIDATES)}
    _require(set(pair_keys) == expected_pairs,
             "final branch ledger does not contain exactly every registered pair")
    _require(set(candidate_names) == set(range(EXPECTED_CANDIDATES)),
             "final branch ledger does not identify all twelve candidates")

    # Canonical inference/analysis order is frozen independently of file append order.
    rows.sort(key=lambda row: (int(row["state_index"]), int(row["candidate_index"])))
    _require([int(row["state_index"]) for row in rows]
             == [index for index in range(EXPECTED_STATES)
                 for _ in range(EXPECTED_CANDIDATES)],
             "final branch rows do not cover the registered state indices 0..199")

    # All planning-time histories are state properties and must be shared across
    # its twelve hypothetical candidates.
    by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_state[str(row["state_id"])].append(row)
    for state_id, group in by_state.items():
        reference = group[0]
        for field in (
            "context_frames", "context_paths", "proprio", "control", "goal",
            "goal_binding_input", "previous_applied_command",
            "action_context_blocks", "timing",
        ):
            material = json.dumps(reference.get(field), sort_keys=True)
            _require(all(json.dumps(row.get(field), sort_keys=True) == material
                         for row in group[1:]),
                     f"{state_id} has candidate-dependent planning-time {field}")
        planning_mask_names = (
            "context_rgb_valid", "observed_proprio_valid",
            "observed_control_valid", "future_proprio_available",
        )
        reference_masks = reference["masks"]
        reference_planning_masks = {
            name: reference_masks[name] for name in planning_mask_names
        }
        _require(all({name: row["masks"][name] for name in planning_mask_names}
                     == reference_planning_masks for row in group[1:]),
                 f"{state_id} has candidate-dependent planning-time masks")

    index = read_json(index_path, "final latent index")
    _require(index.get("schema") == "go2_branch_corpus_v1_2_latents_index_v2"
             and index.get("complete") is True,
             "final latent index is not a complete v2 shard index")
    if "latents_index_digest" in index:
        verify_embedded_digest(index, "latents_index_digest", "final latent index")
    elif "index_digest" in index:
        verify_embedded_digest(index, "index_digest", "final latent index")
    _require(index.get("pool") == "final_eval", "latent index is not final_eval")
    _require(int(index.get("tokens", -1)) == TOKENS
             and int(index.get("token_dim", -1)) == TOKEN_DIM
             and int(index.get("horizons", -1)) == MAX_H
             and int(index.get("context_slots", -1)) == CONTEXT_SLOTS
             and index.get("dtype") == "float16",
             "final latent token layout differs from 768x1024")
    _require(index.get("target_normalisation")
             == "F.layer_norm over the token dimension",
             "final latent target normalisation differs")
    context_record_list = index.get("context_records")
    horizon_record_list = index.get("horizon_records")
    sharded = context_record_list is not None or horizon_record_list is not None
    _require((context_record_list is None) == (horizon_record_list is None),
             "latent index mixes sharded and non-sharded record declarations")
    context_states = ([str(value) for value in index.get("context_states", [])]
                      if not sharded
                      else [_shard_key(record, "context")
                            for record in context_record_list])
    horizon_keys = ([str(value) for value in index.get("horizon_keys", [])]
                    if not sharded
                    else [_shard_key(record, "horizon")
                          for record in horizon_record_list])
    _require(len(context_states) == EXPECTED_STATES
             and len(set(context_states)) == EXPECTED_STATES
             and set(context_states) == set(state_ids),
             "context latent index does not cover exactly the 200 registered states")
    expected_horizon_keys = {
        f"{row['state_id']}|{row['candidate']}" for row in rows if row.get("valid")
    }
    valid_branch_count = len(expected_horizon_keys)
    _require(len(horizon_keys) == valid_branch_count
             and len(set(horizon_keys)) == valid_branch_count
             and set(horizon_keys) == expected_horizon_keys,
             "true-latent index does not cover exactly all oracle-valid branches")
    context_shape = index.get("context_shape", index.get("aggregate_context_shape"))
    horizon_shape = index.get("horizon_shape", index.get("aggregate_horizon_shape"))
    _require(context_shape == [EXPECTED_STATES, CONTEXT_SLOTS, TOKENS, TOKEN_DIM],
             f"unexpected context latent shape {context_shape}")
    _require(horizon_shape == [valid_branch_count, MAX_H, TOKENS, TOKEN_DIM],
             f"unexpected horizon latent shape {horizon_shape}")
    context_records: dict[str, dict[str, Any]] = {}
    horizon_records: dict[str, dict[str, Any]] = {}
    if sharded:
        context_records, context_digest = _validate_shards(
            context_record_list, kind="context", expected_keys=set(state_ids),
            expected_shape=[CONTEXT_SLOTS, TOKENS, TOKEN_DIM])
        horizon_records, horizon_digest = _validate_shards(
            horizon_record_list, kind="horizon", expected_keys=expected_horizon_keys,
            expected_shape=[MAX_H, TOKENS, TOKEN_DIM])
        for field, actual in (("context_records_digest", context_digest),
                              ("horizon_records_digest", horizon_digest)):
            if field in index:
                _require(index[field] == actual, f"final latent index {field} differs")
        for state_id, entry in context_records.items():
            _require(entry.get("state_identity_digest")
                     == state_by_id[state_id].get("state_identity_digest"),
                     f"context shard {state_id} binds a different state identity")
        row_by_horizon_key = {
            f"{row['state_id']}|{row['candidate']}": row
            for row in rows if row.get("valid")
        }
        for key, entry in horizon_records.items():
            row = row_by_horizon_key[key]
            _require(entry.get("branch_identity_digest")
                     == row.get("branch_identity_digest")
                     and entry.get("state_id") == row.get("state_id")
                     and entry.get("candidate") == row.get("candidate")
                     and int(entry.get("candidate_index", -1))
                     == int(row.get("candidate_index", -2)),
                     f"horizon shard {key} binds a different branch identity")
        context_path_or_none: Path | None = None
        horizon_path_or_none: Path | None = None
    else:
        context_digest = _validate_blob(
            context_path, context_shape, index.get("context_sha256"), "context")
        horizon_digest = _validate_blob(
            horizon_path, horizon_shape, index.get("horizon_sha256"), "horizon")
        context_path_or_none = context_path
        horizon_path_or_none = horizon_path
    for field, expected in (
        ("state_manifest_digest", state_manifest_digest),
        ("branch_rows_sha256", branch_rows_sha),
        ("corpus_digest", corpus_digest),
        ("scorer_contract_v1_2_digest", contract_digest()),
        ("target_encoder_digest", target_encoder_digest()),
        ("preprocessing_digest", S.FROZEN_PREPROCESSING_DIGEST),
    ):
        _require(index.get(field) == expected, f"final latent index {field} differs")

    context_positions = {state_id: position
                         for position, state_id in enumerate(context_states)}
    horizon_positions = {key: position for position, key in enumerate(horizon_keys)}
    planning_rows = make_planning_rows(rows, context_positions)

    return CorpusBundle(
        manifest=manifest,
        receipt=receipt,
        index=index,
        rows=rows,
        planning_rows=planning_rows,
        context_path=context_path_or_none,
        horizon_path=horizon_path_or_none,
        context_records=context_records,
        horizon_records=horizon_records,
        context_positions=context_positions,
        horizon_positions=horizon_positions,
        context_shape=tuple(int(value) for value in context_shape),
        horizon_shape=tuple(int(value) for value in horizon_shape),
        context_digest=context_digest,
        horizon_digest=horizon_digest,
        corpus_digest=str(corpus_digest),
        branch_rows_sha256=branch_rows_sha,
    )


def identifiability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The unchanged oracle-v1.2 final-corpus identifiability estimator."""

    by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_state[str(row["state_id"])].append(row)
    valid_states = separated = 0
    distinct_levels: list[int] = []
    spreads: list[float] = []
    families: dict[str, int] = defaultdict(int)
    invalid = sum(not bool(row.get("valid")) for row in rows)
    for group in by_state.values():
        valid = [row for row in group if row.get("valid")]
        if len(valid) < 2:
            continue
        valid_states += 1
        families[str(group[0]["family"])] += 1
        utilities = sorted((float(row["utility"]) for row in valid), reverse=True)
        separated += int(utilities[0] - utilities[1] > TIE_TOLERANCE)
        levels: list[float] = []
        for value in utilities:
            if not levels or abs(value - levels[-1]) > TIE_TOLERANCE:
                levels.append(value)
        distinct_levels.append(len(levels))
        spreads.append(utilities[0] - utilities[-1])
    attempted = len(rows)
    return {
        "attempted": attempted,
        "valid": attempted - invalid,
        "invalid": invalid,
        "invalid_rate": invalid / attempted if attempted else 1.0,
        "states_scored": valid_states,
        "uniquely_separated_fraction": separated / valid_states if valid_states else 0.0,
        "median_distinct_levels": (float(np.median(distinct_levels))
                                   if distinct_levels else 0.0),
        "median_spread": float(np.median(spreads)) if spreads else 0.0,
        "families_with_two_valid_states": sum(count >= 2 for count in families.values()),
        "families_present": len(families),
        "per_family_valid_states": dict(sorted(families.items())),
    }


def apply_final_gate(bundle: CorpusBundle) -> dict[str, Any]:
    statistics = identifiability(bundle.rows)
    components = {
        "uniquely_separated_ge_0.70":
            statistics["uniquely_separated_fraction"] >= 0.70,
        "median_distinct_levels_ge_5": statistics["median_distinct_levels"] >= 5,
        "median_spread_ge_0.10": statistics["median_spread"] >= 0.10,
        "invalid_rate_le_0.20": statistics["invalid_rate"] <= 0.20,
        "all_eight_families_two_valid_states":
            statistics["families_with_two_valid_states"] >= EXPECTED_FAMILIES,
    }
    report: dict[str, Any] = {
        "schema": "go2_final_corpus_identifiability_gate_v1_2",
        "status": STATUS,
        "frozen_before_predictor_loading": True,
        "state_manifest_digest": bundle.manifest["state_manifest_digest"],
        "corpus_digest": bundle.corpus_digest,
        "branch_rows_sha256": bundle.branch_rows_sha256,
        "candidate_bank_digest": CANDIDATE_BANK_DIGEST,
        "oracle_v1_2_digest": contract()["oracle_v1_2_digest"],
        "tie_tolerance": TIE_TOLERANCE,
        "statistics": statistics,
        "gate": {"components": components, "pass": all(components.values())},
    }
    report["gate_report_digest"] = json_digest(report)
    path = FINAL_DIR / "final_gate.json"
    if path.is_file():
        existing = read_json(path, "existing final corpus gate")
        if "gate_report_digest" in existing:
            verify_embedded_digest(existing, "gate_report_digest",
                                   "existing final corpus gate")
        elif "final_gate_digest" in existing:
            verify_embedded_digest(existing, "final_gate_digest",
                                   "existing final corpus gate")
        else:
            raise PlanningRefused(
                "existing final corpus gate has no verified self digest")
        _require(existing.get("state_manifest_digest")
                 == bundle.manifest["state_manifest_digest"]
                 and existing.get("corpus_digest") == bundle.corpus_digest,
                 "existing final corpus gate binds a different corpus")
        _require(existing.get("statistics") == statistics
                 and existing.get("gate") == report["gate"],
                 "existing final corpus gate differs from exact recomputation")
    else:
        atomic_write_json(path, report)
    if not report["gate"]["pass"]:
        raise FinalCorpusGateFailed(report)
    return report


# ----------------------------------------------------- predictor provenance --
def _git(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True, text=True)


def _verify_receipt(path: Path, label: str) -> dict[str, Any]:
    receipt = read_json(path, label)
    verify_embedded_digest(receipt, "receipt_digest", label)
    _require(receipt.get("run_package_digest") == FROZEN_RUN_PACKAGE_DIGEST,
             f"{label} binds a different run package")
    return receipt


def verify_frozen_predictor_lineage() -> tuple[list[FrozenCheckpoint], dict[str, Any]]:
    resolved = _git("rev-parse", "443e591^{commit}")
    _require(resolved.returncode == 0
             and resolved.stdout.strip() == FROZEN_CONFIRMATORY_COMMIT,
             "frozen confirmatory commit does not resolve to the registered object")
    ancestry = _git("merge-base", "--is-ancestor", FROZEN_CONFIRMATORY_COMMIT, "HEAD")
    _require(ancestry.returncode == 0,
             "frozen confirmatory commit is not an ancestor of current HEAD")
    witness = _git("show", f"{FROZEN_CONFIRMATORY_COMMIT}:"
                   "docs/lewm_go2_proprio_factorial_final_report_2026-08-10.md")
    _require(witness.returncode == 0
             and FROZEN_CONFIRMATORY_REPORT_DIGEST in witness.stdout,
             "frozen confirmatory commit does not witness the registered report digest")

    predictor_source_blobs: dict[str, dict[str, str]] = {}
    for relative in FROZEN_PREDICTOR_SOURCE_PATHS:
        frozen_blob = _git("rev-parse", f"{FROZEN_CONFIRMATORY_COMMIT}:{relative}")
        current_blob = _git("hash-object", "--", relative)
        _require(frozen_blob.returncode == 0 and current_blob.returncode == 0
                 and frozen_blob.stdout.strip() == current_blob.stdout.strip(),
                 f"current predictor inference source differs from the frozen "
                 f"confirmatory commit: {relative}")
        predictor_source_blobs[relative] = {
            "git_blob": frozen_blob.stdout.strip(),
            "sha256": sha256_file(ROOT / relative),
        }

    package = read_json(RUN_PACKAGE_PATH, "frozen scientific run package")
    verify_embedded_digest(package, "package_digest", "scientific run package")
    _require(package["package_digest"] == FROZEN_RUN_PACKAGE_DIGEST,
             "scientific run-package digest differs")
    _require(package.get("budget", {}).get("checkpoint_epoch") == D.CHECKPOINT_EPOCH,
             "run package does not freeze epoch 21")
    _require(package.get("budget", {}).get("selection_permitted") is False,
             "run package permits checkpoint selection")

    report = read_json(CONFIRMATORY_REPORT_PATH, "frozen confirmatory report")
    verify_embedded_digest(report, "report_digest", "confirmatory report")
    _require(report["report_digest"] == FROZEN_CONFIRMATORY_REPORT_DIGEST,
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

    initial = _verify_receipt(INITIAL_RECEIPT_PATH, "initial launch receipt")
    continuation = _verify_receipt(CONTINUATION_RECEIPT_PATH,
                                   "continuation launch receipt")
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
        _require(bool(launch_commit), "launch receipt has no source commit")
        _require(_git("merge-base", "--is-ancestor", launch_commit, "HEAD").returncode == 0,
                 f"receipt launch commit {launch_commit} is not an ancestor of HEAD")

    lineage_rows = report.get("attempt_lineage")
    _require(isinstance(lineage_rows, list) and len(lineage_rows) == FROZEN_N,
             "confirmatory report does not carry eight attempt-lineage records")
    lineage = {int(row["seed"]): row for row in lineage_rows}
    checkpoints: list[FrozenCheckpoint] = []
    checkpoint_hash_started = time.time()
    for seed_index, seed in enumerate(frozen_seeds):
        _require(seed in lineage, f"confirmatory lineage omits seed {seed}")
        frozen = lineage[seed]
        _require(int(frozen.get("seed_index", -1)) == seed_index,
                 f"confirmatory lineage seed index differs for {seed}")
        _require(all(frozen.get(key) is True for key in (
            "completed", "all_cells_valid", "all_cells_24_epochs",
            "all_checkpoints_epoch_21", "shared_parameters_bit_identical",
            "batch_plan_identical_across_cells")),
            f"confirmatory lineage marks seed {seed} technically invalid")

        seed_dir = D.OUT / f"seed_{seed}"
        run = read_json(seed_dir / "run_record.json", f"seed {seed} run record")
        expected_receipt = initial if seed_index < 5 else continuation
        _require(run.get("completed") is True and int(run.get("seed", -1)) == seed,
                 f"seed {seed} run record is incomplete or misidentified")
        _require(int(run.get("seed_index", -1)) == seed_index,
                 f"seed {seed} run record has the wrong registry index")
        _require(run.get("authorisation_receipt_digest")
                 == expected_receipt["receipt_digest"]
                 == frozen.get("authorisation_receipt_digest"),
                 f"seed {seed} does not bind its frozen launch receipt")
        for run_field, package_field in (
            ("config_sha256", "model_configuration_sha256"),
            ("manifest_sha256", "base_manifest_rows_sha256"),
            ("normalisation_sha256", "normalisation_sha256"),
            ("seed_registry_sha256", "seed_registry_sha256"),
            ("factorial_manifest_digest", "factorial_manifest_digest"),
            ("canonical_map_digest", "canonical_cache_map_digest"),
        ):
            _require(run.get(run_field) == package.get(package_field),
                     f"seed {seed} {run_field} differs from frozen run package")
        _require(run.get("shared_parameters_bit_identical") is True
                 and run.get("batch_plan_identical_across_cells") is True,
                 f"seed {seed} pairing checks did not pass")
        _require(run.get("budget", {}).get("checkpoint_epoch") == D.CHECKPOINT_EPOCH
                 and run.get("budget", {}).get("selection_permitted") is False,
                 f"seed {seed} run record violates checkpoint selection")

        cells = run.get("cells_run")
        _require(isinstance(cells, list) and len(cells) == len(D.CELLS),
                 f"seed {seed} does not contain one four-cell run")
        by_cell = {str(cell.get("cell")): cell for cell in cells}
        _require(set(by_cell) == set(D.CELLS), f"seed {seed} cell set differs")
        _require(run.get("execution_order") == list(D.cell_order(seed_index)),
                 f"seed {seed} execution order differs from frozen schedule")

        receipt_path = seed_dir / "checkpoint_receipts.jsonl"
        checkpoint_receipts = read_jsonl_strict(
            receipt_path, f"seed {seed} checkpoint receipts")
        for cell in D.CELLS:
            cell_record = by_cell[cell]
            expected_path = seed_dir / (
                f"seed_{seed}_{cell}_epoch{D.CHECKPOINT_EPOCH}.pt")
            _require(Path(str(cell_record.get("checkpoint"))).resolve()
                     == expected_path.resolve(),
                     f"seed {seed} {cell} checkpoint path is not canonical")
            _require(cell_record.get("validity") == "valid"
                     and int(cell_record.get("epochs_trained", -1)) == D.EPOCHS
                     and int(cell_record.get("checkpoint_epoch", -1))
                     == D.CHECKPOINT_EPOCH,
                     f"seed {seed} {cell} is not a valid fixed-budget epoch-21 cell")
            _require(expected_path.is_file(), f"missing checkpoint {expected_path}")
            actual_sha = sha256_file(expected_path)
            expected_sha = str(cell_record.get("checkpoint_sha256"))
            report_sha = str(frozen.get("checkpoint_sha256", {}).get(cell))
            _require(actual_sha == expected_sha == report_sha,
                     f"seed {seed} {cell} checkpoint digest differs from frozen lineage")
            matching = [entry for entry in checkpoint_receipts
                        if Path(str(entry.get("path", ""))).resolve()
                        == expected_path.resolve()
                        and entry.get("sha256") == actual_sha]
            _require(len(matching) == 1,
                     f"seed {seed} {cell} has no unique digest-matched checkpoint receipt")
            disk_receipt = matching[0]
            _require(int(disk_receipt.get("epoch", -1)) == D.CHECKPOINT_EPOCH
                     and int(disk_receipt.get("bytes", -1)) == expected_path.stat().st_size
                     and disk_receipt.get("verified_reloadable") is True,
                     f"seed {seed} {cell} checkpoint receipt is incomplete")
            checkpoints.append(FrozenCheckpoint(
                seed_index=seed_index,
                seed=seed,
                cell=cell,
                path=expected_path,
                sha256=actual_sha,
                bytes=expected_path.stat().st_size,
                receipt_path=receipt_path,
                authorisation_receipt_digest=expected_receipt["receipt_digest"],
            ))

    _require(len(checkpoints) == FROZEN_N * len(D.CELLS),
             "frozen checkpoint list is not exactly 32 entries")
    verification = {
        "confirmatory_commit": FROZEN_CONFIRMATORY_COMMIT,
        "confirmatory_commit_ancestor_of_head": True,
        "confirmatory_report_digest": FROZEN_CONFIRMATORY_REPORT_DIGEST,
        "run_package_digest": FROZEN_RUN_PACKAGE_DIGEST,
        "run_package_sha256": sha256_file(RUN_PACKAGE_PATH),
        "initial_launch_receipt_digest": initial["receipt_digest"],
        "continuation_launch_receipt_digest": continuation["receipt_digest"],
        "frozen_seed_prefix": frozen_seeds,
        "checkpoint_count": len(checkpoints),
        "predictor_source_bindings_at_confirmatory_commit": predictor_source_blobs,
        "checkpoint_hash_verification_wall_time_s":
            round(time.time() - checkpoint_hash_started, 3),
    }
    return checkpoints, verification


# ---------------------------------------------------------- scorer features --
def action_goal_tensor(rows: Sequence[PlanningRow], device: torch.device) -> torch.Tensor:
    values = np.empty((len(rows), ACTION_GOAL_DIM), dtype=np.float32)
    for index, row in enumerate(rows):
        action = [value for block in row.action_blocks for value in block]
        _require(len(action) == 40, f"{row.state_id}|{row.candidate} action is not 40-D")
        values[index, :40] = action
        values[index, 40:] = row.goal_binding
    return torch.from_numpy(values).to(device)


@torch.no_grad()
def scorer_components(model: S.UtilityScorer, latent: torch.Tensor | None,
                      action_goal: torch.Tensor) -> dict[str, np.ndarray]:
    progress, safety_logit, completion_logit = model(latent, action_goal)
    safety = torch.sigmoid(safety_logit)
    completion = torch.sigmoid(completion_logit)
    utility = S.composite(progress, safety_logit, completion_logit)
    return {
        "predicted_progress": progress.detach().cpu().numpy().astype(np.float64),
        "predicted_safety": safety.detach().cpu().numpy().astype(np.float64),
        "predicted_completion": completion.detach().cpu().numpy().astype(np.float64),
        "predicted_utility": utility.detach().cpu().numpy().astype(np.float64),
    }


def _read_f16_shard(entry: Mapping[str, Any]) -> np.ndarray:
    path = Path(str(entry["_resolved_path"]))
    shape = tuple(int(value) for value in entry["shape"])
    return np.asarray(np.memmap(path, mode="r", dtype=np.float16, shape=shape),
                      dtype=np.float32)


def load_context_batch(store: ObservedContextStore,
                       rows: Sequence[PlanningRow],
                       contiguous: np.memmap | None = None) -> np.ndarray:
    if store.records:
        cache: dict[str, np.ndarray] = {}
        values = []
        for row in rows:
            if row.state_id not in cache:
                cache[row.state_id] = _read_f16_shard(
                    store.records[row.state_id])
            values.append(cache[row.state_id])
        return np.stack(values, axis=0)
    _require(contiguous is not None, "contiguous context store is not open")
    return np.asarray(contiguous[[row.context_index for row in rows]],
                      dtype=np.float32)


def load_horizon_batch(bundle: CorpusBundle,
                       rows: Sequence[PlanningRow],
                       contiguous: np.memmap | None = None) -> np.ndarray:
    keys = [f"{row.state_id}|{row.candidate}" for row in rows]
    if bundle.horizon_records:
        return np.stack([_read_f16_shard(bundle.horizon_records[key]) for key in keys],
                        axis=0)
    _require(contiguous is not None, "contiguous horizon store is not open")
    positions = [bundle.horizon_positions[key] for key in keys]
    return np.asarray(contiguous[positions], dtype=np.float32)


def score_diagnostics(bundle: CorpusBundle, latent_scorer: S.UtilityScorer,
                      baseline_scorer: S.UtilityScorer, device: torch.device,
                      batch_size: int) -> tuple[dict[str, Any], float]:
    started = time.time()
    horizon = (np.memmap(bundle.horizon_path, mode="r", dtype=np.float16,
                         shape=bundle.horizon_shape)
               if bundle.horizon_path is not None else None)
    count = len(bundle.planning_rows)
    upper = np.empty(count, dtype=np.float64)
    baseline = np.empty(count, dtype=np.float64)
    upper_components = {
        name: np.empty(count, dtype=np.float64) for name in (
            "predicted_progress", "predicted_safety", "predicted_completion")
    }
    baseline_components = {name: np.empty(count, dtype=np.float64)
                           for name in upper_components}
    # Invalid branch outcomes have no oracle target and therefore no true
    # future latent shard.  They are still predictor-scored below, but are
    # excluded from every outcome metric exactly as invalid branches are
    # excluded by the frozen oracle gate.
    upper.fill(0.0)
    for values in upper_components.values():
        values.fill(0.0)
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        planning = bundle.planning_rows[start:stop]
        action_goal = action_goal_tensor(planning, device)
        base = scorer_components(baseline_scorer, None, action_goal)
        baseline[start:stop] = base["predicted_utility"]
        valid_offsets = [offset for offset, corpus_row in enumerate(
            bundle.rows[start:stop]) if corpus_row.get("valid")]
        if valid_offsets:
            valid_planning = [planning[offset] for offset in valid_offsets]
            # The true future is read here only for the explicitly separated
            # upper-bound diagnostic. Predictor inference below has no reference
            # to this store or to CorpusBundle.rows.
            trajectory = torch.from_numpy(
                load_horizon_batch(bundle, valid_planning, horizon).mean(axis=2)).to(device)
            valid_action_goal = action_goal[valid_offsets]
            scored = scorer_components(latent_scorer, trajectory, valid_action_goal)
            destination = np.asarray([start + offset for offset in valid_offsets])
            upper[destination] = scored["predicted_utility"]
            for name in upper_components:
                upper_components[name][destination] = scored[name]
        for name in baseline_components:
            baseline_components[name][start:stop] = base[name]
    if horizon is not None:
        del horizon
    diagnostics = {
        "true_latent_upper_bound": evaluate_scores(bundle.rows, upper,
                                                    upper_components),
        "no_latent_baseline": evaluate_scores(bundle.rows, baseline,
                                               baseline_components),
        "separation": (
            "true-latent is an upper-bound scorer diagnostic only; no true future "
            "tensor is reachable from predictor inference"),
    }
    return diagnostics, time.time() - started


# ------------------------------------------------------------ rank metrics --
def _deterministic_order(indices: Sequence[int], rows: Sequence[dict[str, Any]],
                         scores: np.ndarray) -> list[int]:
    return sorted(indices,
                  key=lambda index: (-float(scores[index]),
                                     int(rows[index]["candidate_index"])))


def _state_pairwise(indices: Sequence[int], rows: Sequence[dict[str, Any]],
                    scores: np.ndarray) -> tuple[int, int]:
    correct = considered = 0
    for left_position in range(len(indices)):
        for right_position in range(left_position + 1, len(indices)):
            left, right = indices[left_position], indices[right_position]
            gap = float(rows[left]["utility"]) - float(rows[right]["utility"])
            if abs(gap) <= TIE_TOLERANCE:
                continue
            considered += 1
            correct += int((float(scores[left]) - float(scores[right])) * gap > 0)
    return correct, considered


def _aggregate_state_metrics(states: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not states:
        return {key: None for key in METRIC_DIRECTIONS} | {
            "states": 0, "valid_candidates": 0, "pairwise_pairs_considered": 0}
    result: dict[str, Any] = {
        "states": len(states),
        "valid_candidates": int(sum(int(state["valid_candidates"]) for state in states)),
        "pairwise_pairs_considered": int(sum(int(state["pairs_considered"])
                                                 for state in states)),
    }
    for key in METRIC_DIRECTIONS:
        if key == "pairwise_ordering_accuracy":
            correct = sum(int(state["pairs_correct"]) for state in states)
            considered = result["pairwise_pairs_considered"]
            result[key] = float(correct / considered) if considered else None
            continue
        values = [float(state[key]) for state in states if _finite(state.get(key))]
        result[key] = float(np.mean(values)) if values else None
    return result


def evaluate_scores(rows: list[dict[str, Any]], scores: np.ndarray,
                    components: Mapping[str, np.ndarray] | None = None
                    ) -> dict[str, Any]:
    scores = np.asarray(scores, dtype=np.float64)
    _require(scores.shape == (len(rows),),
             f"score vector has shape {scores.shape}, expected {(len(rows),)}")
    _require(bool(np.isfinite(scores).all()), "candidate scores contain NaN or infinity")
    if components is not None:
        for name, values in components.items():
            values = np.asarray(values)
            _require(values.shape == scores.shape and bool(np.isfinite(values).all()),
                     f"{name} vector is incomplete or non-finite")

    # Every candidate is predictor-scored and remains in its immutable ledger.
    # Oracle-invalid branches have neither a valid utility nor a true target
    # latent, so outcome metrics use the unchanged oracle-pilot population rule:
    # rank the oracle-valid subset and omit a state with fewer than two valid
    # candidates.  No penalty or utility is invented for a technical invalidity.
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row.get("valid") and _finite(row.get("utility")):
            grouped[str(row["state_id"])].append(index)
    all_state_ids = {str(row["state_id"]) for row in rows}
    eligible_grouped = {state_id: indices for state_id, indices in grouped.items()
                        if len(indices) >= 2}
    omitted_state_ids = sorted(
        all_state_ids - set(eligible_grouped),
        key=lambda state_id: int(next(row["state_index"] for row in rows
                                      if str(row["state_id"]) == state_id)),
    )

    state_metrics: list[dict[str, Any]] = []
    for state_id in sorted(
            eligible_grouped,
            key=lambda value: int(rows[eligible_grouped[value][0]]["state_index"])):
        indices = eligible_grouped[state_id]
        predicted_order = _deterministic_order(indices, rows, scores)
        oracle_order = sorted(indices,
                              key=lambda index: (-float(rows[index]["utility"]),
                                                 int(rows[index]["candidate_index"])))
        chosen = predicted_order[0]
        oracle_best = oracle_order[0]
        truth = np.asarray([float(rows[index]["utility"]) for index in indices],
                           dtype=np.float64)
        predicted = np.asarray([float(scores[index]) for index in indices],
                               dtype=np.float64)
        spread = float(truth.max() - truth.min())
        absolute = float(truth.max() - float(rows[chosen]["utility"]))
        normalised = 0.0 if spread <= 0 else absolute / spread
        pair_correct, pair_considered = _state_pairwise(indices, rows, scores)
        spearman = S.spearman(truth, predicted)
        score_spread = float(predicted.max() - predicted.min())
        state_entry: dict[str, Any] = {
            "state_id": state_id,
            "state_index": int(rows[indices[0]]["state_index"]),
            "family": str(rows[indices[0]]["family"]),
            "valid_candidates": len(indices),
            "selected_candidate": str(rows[chosen]["candidate"]),
            "selected_candidate_index": int(rows[chosen]["candidate_index"]),
            "oracle_best_candidate": str(rows[oracle_best]["candidate"]),
            "oracle_best_candidate_index": int(rows[oracle_best]["candidate_index"]),
            "realised_selected_utility": float(rows[chosen]["utility"]),
            "oracle_best_utility": float(rows[oracle_best]["utility"]),
            "absolute_rank_regret": absolute,
            "normalised_rank_regret": normalised,
            "spearman_rank_correlation": spearman,
            "top1_recovery": float(chosen == oracle_best),
            "top3_recovery": float(oracle_best in predicted_order[:3]),
            "pairwise_ordering_accuracy": (
                float(pair_correct / pair_considered) if pair_considered else None),
            "pairs_correct": pair_correct,
            "pairs_considered": pair_considered,
            "candidate_score_spread": score_spread,
            "scorer_tie_rate": float(
                sum(abs(float(scores[index]) - float(scores[chosen]))
                    <= SCORE_TIE_TOLERANCE for index in indices) > 1),
        }
        state_metrics.append(state_entry)

    per_family = {
        family: _aggregate_state_metrics(
            [state for state in state_metrics if state["family"] == family])
        for family in sorted({state["family"] for state in state_metrics})
    }
    primary_families = [family for family in sorted(per_family)
                        if family != DIAGNOSTIC_FAMILY]
    _require(len(primary_families) == EXPECTED_FAMILIES - 1,
             "primary estimator must contain seven non-diagnostic families")
    primary_states = [state for state in state_metrics
                      if state["family"] != DIAGNOSTIC_FAMILY]
    primary_equal: dict[str, Any] = {
        "families": primary_families,
        "family_count": len(primary_families),
        "states": len(primary_states),
        "diagnostic_family_excluded": DIAGNOSTIC_FAMILY,
    }
    for metric in METRIC_DIRECTIONS:
        values = [per_family[family][metric] for family in primary_families]
        finite = [float(value) for value in values if _finite(value)]
        primary_equal[metric] = (float(np.mean(finite))
                                 if len(finite) == len(values) else None)

    component_summary: dict[str, Any] = {}
    if components is not None:
        valid_indices = [index for index, row in enumerate(rows) if row.get("valid")]
        for name, values in components.items():
            array = np.asarray(values, dtype=np.float64)[valid_indices]
            component_summary[name] = {
                "mean": float(array.mean()),
                "sd": float(array.std(ddof=0)),
                "min": float(array.min()),
                "max": float(array.max()),
            }

    return {
        "metric_definitions": {
            "absolute_rank_regret": "best oracle utility - selected oracle utility",
            "normalised_rank_regret": "absolute regret / within-state oracle utility spread",
            "ties": ("oracle and predicted ties break by frozen candidate-bank index; "
                     "pairwise accuracy ignores true gaps <= 0.02"),
            "scorer_tie_rate": (
                "fraction of states with more than one candidate score within the "
                "frozen 0.02 tolerance of that state's maximum score"),
            "candidate_score_spread": "within-state maximum score - minimum score",
            "score_tie_tolerance": SCORE_TIE_TOLERANCE,
        },
        "evaluation_population": {
            "predictor_scores_in_ledger": len(rows),
            "all_twelve_candidates_predictor_scored_per_state": True,
            "oracle_valid_candidates_used_for_outcome_metrics": True,
            "invalid_branch_utility_imputed": False,
            "state_requires_at_least_two_oracle_valid_candidates": True,
            "states_total": len(all_state_ids),
            "states_evaluated": len(state_metrics),
            "states_omitted_fewer_than_two_valid_candidates": omitted_state_ids,
        },
        "primary_equal_family": primary_equal,
        "primary_corpus_weighted": _aggregate_state_metrics(primary_states),
        "all_corpus_weighted_diagnostic": _aggregate_state_metrics(state_metrics),
        "per_family": per_family,
        "local_composite_motifs_diagnostic": per_family.get(DIAGNOSTIC_FAMILY),
        "state_metrics": state_metrics,
        "component_score_distribution": component_summary,
    }


# ------------------------------------------------------ prediction ledgers --
def _ledger_paths(checkpoint: FrozenCheckpoint) -> tuple[Path, Path]:
    stem = f"seed_{checkpoint.seed}_{checkpoint.cell}"
    return PREDICTION_DIR / f"{stem}.jsonl", PREDICTION_DIR / f"{stem}.receipt.json"


def _ledger_spec(checkpoint: FrozenCheckpoint, bundle: CorpusBundle,
                 scorer_package_sha: str, normalisation_sha: str) -> dict[str, Any]:
    ordered_keys = [f"{row.state_id}|{row.candidate}" for row in bundle.planning_rows]
    spec: dict[str, Any] = {
        "schema": "go2_predictor_score_ledger_spec_v1_2",
        "seed_index": checkpoint.seed_index,
        "seed": checkpoint.seed,
        "cell": checkpoint.cell,
        "checkpoint_epoch": D.CHECKPOINT_EPOCH,
        "checkpoint_sha256": checkpoint.sha256,
        "scorer_contract_v1_2_digest": contract_digest(),
        "scorer_package_sha256": scorer_package_sha,
        "state_manifest_digest": bundle.manifest["state_manifest_digest"],
        "corpus_digest": bundle.corpus_digest,
        "branch_rows_sha256": bundle.branch_rows_sha256,
        "context_latent_store_digest": bundle.context_digest,
        "horizon_latent_store_digest_not_an_input": bundle.horizon_digest,
        "normalisation_sha256": normalisation_sha,
        "scoring_implementation_sha256": scoring_implementation_bindings(),
        "ordered_planning_keys_sha256": sequence_digest(ordered_keys),
        "rows": len(ordered_keys),
        "inference_batch": FROZEN_INFERENCE_BATCH,
        "shared_scorer": True,
        "model_specific_calibration": False,
        "predictor_inputs": [
            "observed visual context", "candidate post-slew action trajectory",
            "shared observed applied-command control history",
            "observed proprioceptive history for proprio cells only",
            "snapshot-time bearing/range goal binding",
        ],
        "forbidden_predictor_inputs": [
            "realised future RGB", "true future latent", "oracle utility",
            "branch outcome", "future proprioception", "privileged simulator state",
        ],
    }
    spec["ledger_spec_digest"] = json_digest(spec)
    return spec


def _preserve_attempt(path: Path, reason: str, recovery: list[dict[str, Any]]) -> None:
    if not path.is_file():
        return
    digest = sha256_file(path)
    attempts = PREDICTION_DIR / "invalid_or_interrupted_attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    destination = attempts / f"{path.name}.{reason}.{digest[:16]}"
    if destination.is_file():
        _require(sha256_file(destination) == digest,
                 f"preserved-attempt name collision at {destination}")
        path.unlink()
    else:
        os.replace(path, destination)
    recovery.append({"path": str(path), "preserved_as": str(destination),
                     "sha256": digest, "reason": reason})


def reuse_completed_result(scorer_provenance: Mapping[str, Any],
                           bundle: CorpusBundle,
                           checkpoints: Sequence[FrozenCheckpoint],
                           recovery: list[dict[str, Any]]) -> dict[str, Any] | None:
    result_path = RESULT_DIR / "planning_result.json"
    receipt_path = RESULT_DIR / "planning_result_receipt.json"
    if not result_path.exists() and not receipt_path.exists():
        return None
    try:
        report = read_json(result_path, "existing planning result")
        receipt = read_json(receipt_path, "existing planning result receipt")
        verify_embedded_digest(report, "report_digest", "existing planning result")
        verify_embedded_digest(receipt, "receipt_digest",
                               "existing planning result receipt")
        _require(report.get("complete") is True and receipt.get("complete") is True,
                 "existing planning result is incomplete")
        _require(receipt.get("planning_result_sha256") == sha256_file(result_path),
                 "existing planning result bytes differ from receipt")
        _require(receipt.get("report_digest") == report.get("report_digest"),
                 "existing planning result digest differs from receipt")
        _require(receipt.get("scorer_package_sha256")
                 == scorer_provenance["package_sha256"],
                 "existing planning result uses a different scorer package")
        _require(receipt.get("corpus_digest") == bundle.corpus_digest,
                 "existing planning result uses a different final corpus")
        _require(int(receipt.get("checkpoint_count", -1)) == len(checkpoints) == 32,
                 "existing planning result does not bind 32 checkpoints")
        expected_hashes = {(item.seed, item.cell): item.sha256 for item in checkpoints}
        observed = report.get("verified_checkpoints")
        _require(isinstance(observed, list) and len(observed) == 32,
                 "existing planning result checkpoint list is incomplete")
        observed_hashes = {
            (int(item["seed"]), str(item["cell"])): str(item["sha256"])
            for item in observed
        }
        _require(observed_hashes == expected_hashes,
                 "existing planning result checkpoint hashes differ")
        _require(receipt.get("prediction_receipts_complete") is True,
                 "existing planning result has incomplete prediction receipts")
        recorded_prediction_receipts = report.get("prediction_receipts")
        _require(isinstance(recorded_prediction_receipts, list)
                 and len(recorded_prediction_receipts) == 32,
                 "existing planning result does not contain 32 prediction receipts")
        recorded_by_key = {
            (int(item["seed"]), str(item["cell"])): item
            for item in recorded_prediction_receipts
        }
        normalisation_sha = read_json(
            RUN_PACKAGE_PATH, "scientific run package")["normalisation_sha256"]
        for checkpoint in checkpoints:
            spec = _ledger_spec(checkpoint, bundle,
                                str(scorer_provenance["package_sha256"]),
                                str(normalisation_sha))
            records = _load_completed_ledger(checkpoint, bundle, spec, recovery)
            _ledger_path, prediction_receipt_path = _ledger_paths(checkpoint)
            _require(records is not None and len(records) == EXPECTED_BRANCHES
                     and prediction_receipt_path.is_file(),
                     f"existing result ledger is incomplete for {checkpoint.seed}/"
                     f"{checkpoint.cell}")
            current_receipt = read_json(
                prediction_receipt_path, "current prediction receipt")
            _require(current_receipt == recorded_by_key.get(
                (checkpoint.seed, checkpoint.cell)),
                f"existing result prediction receipt changed for {checkpoint.seed}/"
                f"{checkpoint.cell}")
        return report
    except (PlanningRefused, KeyError, TypeError, ValueError) as exc:
        _preserve_attempt(result_path, "invalid_final_result", recovery)
        _preserve_attempt(receipt_path, "invalid_final_result_receipt", recovery)
        recovery.append({"reason": "existing final result was not reusable",
                         "detail": str(exc)})
        return None


def _read_ledger_prefix(path: Path, expected: Sequence[PlanningRow], spec_digest: str
                        ) -> tuple[list[dict[str, Any]], str | None]:
    if not path.is_file():
        return [], None
    prefix: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                return prefix, f"malformed_line_{line_number}"
            position = len(prefix)
            if position >= len(expected):
                return prefix, "surplus_records"
            row = expected[position]
            checks = (
                record.get("ledger_spec_digest") == spec_digest,
                record.get("position") == position,
                record.get("state_id") == row.state_id,
                record.get("candidate") == row.candidate,
                record.get("candidate_index") == row.candidate_index,
                _finite(record.get("predicted_utility")),
                _finite(record.get("predicted_progress")),
                _finite(record.get("predicted_safety")),
                _finite(record.get("predicted_completion")),
            )
            if not all(checks):
                return prefix, f"binding_mismatch_line_{line_number}"
            prefix.append(record)
    return prefix, None


def _prediction_receipt(checkpoint: FrozenCheckpoint, ledger_path: Path,
                        spec: dict[str, Any], records: Sequence[dict[str, Any]],
                        wall_time_s: float, resumed_records: int) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema": "go2_predictor_score_ledger_receipt_v1_2",
        "status": STATUS,
        "complete": True,
        "seed_index": checkpoint.seed_index,
        "seed": checkpoint.seed,
        "cell": checkpoint.cell,
        "checkpoint_sha256": checkpoint.sha256,
        "ledger_spec_digest": spec["ledger_spec_digest"],
        "records_expected": len(records),
        "records_completed": len(records),
        "resumed_verified_records": resumed_records,
        "ledger_sha256": sha256_file(ledger_path),
        "score_vector_sha256": sequence_digest(
            [record["predicted_utility"] for record in records]),
        "wall_time_s_this_invocation": round(wall_time_s, 3),
        "ledger_bytes": ledger_path.stat().st_size,
        "finished_unix_s": time.time(),
    }
    receipt["receipt_digest"] = json_digest(receipt)
    return receipt


def _load_completed_ledger(checkpoint: FrozenCheckpoint,
                           bundle: CorpusBundle,
                           spec: dict[str, Any],
                           recovery: list[dict[str, Any]]
                           ) -> list[dict[str, Any]] | None:
    ledger_path, receipt_path = _ledger_paths(checkpoint)
    prefix, error = _read_ledger_prefix(
        ledger_path, bundle.planning_rows, spec["ledger_spec_digest"])
    if error is not None:
        _preserve_attempt(ledger_path, error, recovery)
        _preserve_attempt(receipt_path, error, recovery)
        if prefix:
            atomic_write_jsonl(ledger_path, prefix)
            recovery.append({"path": str(ledger_path), "reason": error,
                             "recovered_verified_prefix_records": len(prefix)})
        return None
    if not receipt_path.is_file():
        return prefix if len(prefix) == len(bundle.planning_rows) else None
    try:
        receipt = read_json(receipt_path, "prediction ledger receipt")
        verify_embedded_digest(receipt, "receipt_digest", "prediction ledger receipt")
        _require(receipt.get("complete") is True,
                 "prediction ledger receipt is incomplete")
        _require(receipt.get("ledger_spec_digest") == spec["ledger_spec_digest"],
                 "prediction ledger receipt has a different spec")
        _require(receipt.get("checkpoint_sha256") == checkpoint.sha256,
                 "prediction ledger receipt has a different checkpoint")
        _require(int(receipt.get("records_completed", -1)) == len(bundle.planning_rows),
                 "prediction ledger receipt has the wrong record count")
        _require(int(receipt.get("records_expected", -1)) == len(bundle.planning_rows),
                 "prediction ledger receipt expected a different record count")
        _require(len(prefix) == len(bundle.planning_rows),
                 "complete prediction receipt has an incomplete ledger")
        _require(receipt.get("ledger_sha256") == sha256_file(ledger_path),
                 "prediction ledger bytes differ from receipt")
        _require(receipt.get("score_vector_sha256") == sequence_digest(
            [record["predicted_utility"] for record in prefix]),
            "prediction score vector differs from receipt")
    except PlanningRefused:
        _preserve_attempt(receipt_path, "invalid_receipt", recovery)
        return prefix if len(prefix) == len(bundle.planning_rows) else None
    return prefix


def _load_normalisation(expected_digest: str) -> tuple[dict[str, Any], str]:
    stats_path = D.PROPRIO / "proprio_norm_stats.json"
    stats = read_json(stats_path, "frozen proprio/control normalisation")
    verify_embedded_digest(stats, "sha256", "proprio/control normalisation")
    _require(stats["sha256"] == expected_digest,
             "proprio/control normalisation differs from frozen run package")
    for key, length in (("mean", P.PROPRIO_DIM), ("std", P.PROPRIO_DIM),
                        ("control_mean", P.CONTROL_DIM),
                        ("control_std", P.CONTROL_DIM)):
        _require(isinstance(stats.get(key), list) and len(stats[key]) == length
                 and all(_finite(value) for value in stats[key]),
                 f"normalisation {key} is malformed")
    return stats, stats["sha256"]


@torch.no_grad()
def _predict_batch(model: P.ProprioActionPredictor,
                   scorer: S.UtilityScorer,
                   planning: Sequence[PlanningRow],
                   observed_context: ObservedContextStore,
                   context: np.memmap | None,
                   stats: dict[str, Any],
                   use_proprio: bool,
                   device: torch.device) -> dict[str, np.ndarray]:
    # This function's signature is the leakage boundary: it cannot see branch
    # rows, oracle labels, horizon latents, future frames or a simulator handle.
    context_array = load_context_batch(observed_context, planning, context)
    visual = torch.from_numpy(context_array).to(device)
    proprio = torch.tensor([row.proprio_history for row in planning],
                            dtype=torch.float32, device=device).reshape(
                                len(planning), CONTEXT_SLOTS, P.SAMPLES_PER_SLOT,
                                P.PROPRIO_DIM)
    control = torch.tensor([row.control_history for row in planning],
                           dtype=torch.float32, device=device).reshape(
                               len(planning), CONTEXT_SLOTS, P.SAMPLES_PER_SLOT,
                               P.CONTROL_DIM)
    proprio, control = D.normalise_batch(proprio, control, stats, device)
    actions = [torch.tensor([row.action_blocks[horizon] for row in planning],
                            dtype=torch.float32, device=device)
               for horizon in range(MAX_H)]
    steps = P.unroll(model, visual, actions,
                     proprio if use_proprio else None,
                     control, max_h=MAX_H)
    trajectory = torch.stack([step.mean(dim=1) for step in steps], dim=1)
    return scorer_components(scorer, trajectory,
                             action_goal_tensor(planning, device))


def score_checkpoint(checkpoint: FrozenCheckpoint,
                     bundle: CorpusBundle,
                     scorer: S.UtilityScorer,
                     scorer_package_sha: str,
                     scorer_state_digest: str,
                     normalisation: dict[str, Any],
                     normalisation_sha: str,
                     device: torch.device,
                     batch_size: int,
                     recovery: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    ledger_path, receipt_path = _ledger_paths(checkpoint)
    spec = _ledger_spec(checkpoint, bundle, scorer_package_sha, normalisation_sha)
    records = _load_completed_ledger(checkpoint, bundle, spec, recovery)
    if records is not None and len(records) == len(bundle.planning_rows):
        scores = np.asarray([record["predicted_utility"] for record in records],
                            dtype=np.float64)
        return scores, read_json(receipt_path, "completed prediction receipt") \
            if receipt_path.is_file() else _finish_unreceipted_ledger(
                checkpoint, ledger_path, receipt_path, spec, records)

    prefix, error = _read_ledger_prefix(
        ledger_path, bundle.planning_rows, spec["ledger_spec_digest"])
    _require(error is None, f"failed to recover prediction ledger: {error}")
    records = list(prefix)
    resume_at = len(records)
    started = time.time()

    # All 32 bytes were already digest-verified before this first torch.load.
    payload = torch.load(checkpoint.path, map_location="cpu", weights_only=False)
    _require(payload.get("schema") == CK.SCHEMA,
             f"{checkpoint.path} is not a {CK.SCHEMA} checkpoint")
    _require(int(payload.get("epoch", -1)) == D.CHECKPOINT_EPOCH,
             f"{checkpoint.path} payload is not epoch 21")
    _require(int(payload.get("seed", -1)) == checkpoint.seed,
             f"{checkpoint.path} payload has the wrong seed")
    expected_cell_spec = D.CELL_SPEC[checkpoint.cell]
    model_config = payload.get("model_config", {})
    _require(model_config.get("cell") == checkpoint.cell
             and model_config.get("use_proprio") == expected_cell_spec["use_proprio"]
             and model_config.get("rollout") == expected_cell_spec["rollout"],
             f"{checkpoint.path} payload has the wrong cell configuration")
    width = int(model_config.get("width", 384))
    _require(width == 384, f"{checkpoint.path} predictor width differs from frozen 384")
    model = P.build_paired(checkpoint.seed,
                           use_proprio=expected_cell_spec["use_proprio"],
                           width=width, depth=6, heads=6).to(device)
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    except (KeyError, RuntimeError) as exc:
        raise PlanningRefused(f"cannot load frozen checkpoint {checkpoint.path}: {exc}") from exc
    model.eval()
    del payload

    observed_context = ObservedContextStore(
        path=bundle.context_path, records=bundle.context_records,
        shape=bundle.context_shape)
    context = (np.memmap(observed_context.path, mode="r", dtype=np.float16,
                         shape=observed_context.shape)
               if observed_context.path is not None else None)
    mode = "a" if ledger_path.is_file() and resume_at else "w"
    with ledger_path.open(mode, encoding="utf-8") as sink:
        if mode == "a" and ledger_path.stat().st_size and not ledger_path.read_bytes().endswith(b"\n"):
            sink.write("\n")
        for start in range(resume_at, len(bundle.planning_rows), batch_size):
            stop = min(start + batch_size, len(bundle.planning_rows))
            planning = bundle.planning_rows[start:stop]
            values = _predict_batch(
                model, scorer, planning, observed_context, context, normalisation,
                bool(expected_cell_spec["use_proprio"]), device)
            for offset, row in enumerate(planning):
                record = {
                    "ledger_spec_digest": spec["ledger_spec_digest"],
                    "position": row.position,
                    "state_id": row.state_id,
                    "state_index": row.state_index,
                    "family": row.family,
                    "candidate": row.candidate,
                    "candidate_index": row.candidate_index,
                    "predicted_progress": float(values["predicted_progress"][offset]),
                    "predicted_safety": float(values["predicted_safety"][offset]),
                    "predicted_completion": float(values["predicted_completion"][offset]),
                    "predicted_utility": float(values["predicted_utility"][offset]),
                }
                _require(all(_finite(record[key]) for key in (
                    "predicted_progress", "predicted_safety",
                    "predicted_completion", "predicted_utility")),
                    f"non-finite prediction at {row.state_id}|{row.candidate}")
                sink.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                records.append(record)
            sink.flush()
            os.fsync(sink.fileno())
            print(f"[score] seed {checkpoint.seed} {checkpoint.cell}: "
                  f"{stop}/{len(bundle.planning_rows)}", flush=True)
    if context is not None:
        del context
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _require(tensor_state_digest(scorer) == scorer_state_digest,
             "shared scorer parameters changed during predictor scoring")
    receipt = _prediction_receipt(
        checkpoint, ledger_path, spec, records,
        time.time() - started, resume_at)
    atomic_write_json(receipt_path, receipt)
    return (np.asarray([record["predicted_utility"] for record in records],
                       dtype=np.float64), receipt)


def _finish_unreceipted_ledger(checkpoint: FrozenCheckpoint, ledger_path: Path,
                               receipt_path: Path, spec: dict[str, Any],
                               records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    receipt = _prediction_receipt(checkpoint, ledger_path, spec, records, 0.0,
                                  len(records))
    receipt["recovery"] = "complete verified ledger found without its receipt"
    receipt.pop("receipt_digest")
    receipt["receipt_digest"] = json_digest(receipt)
    atomic_write_json(receipt_path, receipt)
    return receipt


# ---------------------------------------------------------- frozen analysis --
def t_interval(values: Sequence[float]) -> dict[str, Any]:
    """Two-sided 95% Student-t interval over exactly eight seed quadruplets."""

    _require(len(values) == FROZEN_N,
             f"paired analysis requires exactly {FROZEN_N} seed values")
    array = np.asarray(values, dtype=np.float64)
    _require(bool(np.isfinite(array).all()), "paired seed metric contains non-finite values")
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    critical_t_df7 = 2.3646242510102993
    half = critical_t_df7 * sd / math.sqrt(FROZEN_N)
    return {
        "values": [float(value) for value in array],
        "n": FROZEN_N,
        "mean": mean,
        "sample_standard_deviation": sd,
        "t_critical_df7": critical_t_df7,
        "two_sided_95_t_interval": [mean - half, mean + half],
    }


def optional_t_interval(values: Sequence[float | None]) -> dict[str, Any]:
    """Exact df=7 interval when all eight values exist; otherwise report why not.

    This is used only for secondary metrics that are mathematically undefined in
    a degenerate state/family (for example Spearman under constant scores).  It
    never imputes a value and never relaxes the primary regret analysis.
    """

    _require(len(values) == FROZEN_N,
             f"paired analysis requires exactly {FROZEN_N} seed values")
    if all(_finite(value) for value in values):
        return t_interval([float(value) for value in values])
    safe_values = [float(value) if _finite(value) else None for value in values]
    return {
        "values": safe_values,
        "n": FROZEN_N,
        "n_finite": sum(value is not None for value in safe_values),
        "mean": None,
        "sample_standard_deviation": None,
        "t_critical_df7": 2.3646242510102993,
        "two_sided_95_t_interval": None,
        "unavailable_reason": (
            "at least one seed value is mathematically undefined; no imputation "
            "or reduced-N interval is permitted"),
    }


def paired_factorial(cells: Mapping[str, Mapping[int, dict[str, Any]]],
                     seeds: Sequence[int], metric: str, aggregation: str,
                     *, direction: str) -> dict[str, Any]:
    _require(direction in {"lower", "higher", "descriptive_higher"},
             f"unknown metric direction {direction}")

    def value(cell: str, seed: int) -> float | None:
        metric_value = cells[cell][seed][aggregation][metric]
        return float(metric_value) if _finite(metric_value) else None

    # For loss-like metrics, positive B means rollout reduced the metric.  For
    # benefit-like/descriptive metrics, positive means rollout increased it.
    if direction == "lower":
        difference = lambda one, rollout: one - rollout
        positive = "rollout reduced the metric"
    else:
        difference = lambda one, rollout: rollout - one
        positive = "rollout increased the metric"
    def effect(one: float | None, rollout: float | None) -> float | None:
        return (difference(one, rollout)
                if one is not None and rollout is not None else None)

    b_rgb = [effect(value("rgb_one_step", seed), value("rgb_rollout", seed))
             for seed in seeds]
    b_prop = [effect(value("proprio_one_step", seed),
                     value("proprio_rollout", seed)) for seed in seeds]
    main = [((rgb + prop) / 2.0
             if rgb is not None and prop is not None else None)
            for rgb, prop in zip(b_rgb, b_prop)]
    interaction = [(prop - rgb
                    if rgb is not None and prop is not None else None)
                   for rgb, prop in zip(b_rgb, b_prop)]
    return {
        "metric": metric,
        "aggregation": aggregation,
        "direction": direction,
        "positive_effect_definition": positive,
        "per_seed": {
            str(seed): {"B_RGB": b_rgb[index], "B_prop": b_prop[index],
                        "M": main[index], "J": interaction[index]}
            for index, seed in enumerate(seeds)
        },
        "B_RGB": optional_t_interval(b_rgb),
        "B_prop": optional_t_interval(b_prop),
        "M": optional_t_interval(main),
        "J": optional_t_interval(interaction),
        "cell_values": {
            cell: optional_t_interval([value(cell, seed) for seed in seeds])
            for cell in D.CELLS
        },
    }


def analyse(cells: Mapping[str, Mapping[int, dict[str, Any]]],
            seeds: Sequence[int]) -> dict[str, Any]:
    _require(list(seeds) == list(D.SEED_REGISTRY[:FROZEN_N]),
             "analysis seed list is not the first eight frozen registry entries")
    primary = {
        metric: paired_factorial(cells, seeds, metric, "primary_equal_family",
                                 direction=direction)
        for metric, direction in METRIC_DIRECTIONS.items()
    }
    corpus_weighted = {
        metric: paired_factorial(cells, seeds, metric, "primary_corpus_weighted",
                                 direction=direction)
        for metric, direction in METRIC_DIRECTIONS.items()
    }
    families = sorted(next(iter(next(iter(cells.values())).values()))["per_family"])
    per_family: dict[str, Any] = {family: {} for family in families}
    # Family values are already an aggregation block, so apply the identical
    # paired-seed arithmetic directly rather than routing through a cell-level
    # aggregation key.
    for family in families:
        for metric, direction in METRIC_DIRECTIONS.items():
            def family_value(cell: str, seed: int) -> float | None:
                observed = cells[cell][seed]["per_family"][family][metric]
                return float(observed) if _finite(observed) else None
            difference = ((lambda one, rollout: one - rollout)
                          if direction == "lower"
                          else (lambda one, rollout: rollout - one))
            def family_effect(one: float | None,
                              rollout: float | None) -> float | None:
                return (difference(one, rollout)
                        if one is not None and rollout is not None else None)
            b_rgb = [family_effect(family_value("rgb_one_step", seed),
                                   family_value("rgb_rollout", seed))
                     for seed in seeds]
            b_prop = [family_effect(family_value("proprio_one_step", seed),
                                    family_value("proprio_rollout", seed))
                      for seed in seeds]
            main = [((left + right) / 2
                     if left is not None and right is not None else None)
                    for left, right in zip(b_rgb, b_prop)]
            interaction = [(right - left
                            if left is not None and right is not None else None)
                           for left, right in zip(b_rgb, b_prop)]
            per_family[family][metric] = {
                "direction": direction,
                "B_RGB": optional_t_interval(b_rgb),
                "B_prop": optional_t_interval(b_prop),
                "M": optional_t_interval(main),
                "J": optional_t_interval(interaction),
            }

    main_regret = primary["normalised_rank_regret"]["M"]
    interval = main_regret["two_sided_95_t_interval"]
    _require(isinstance(interval, list) and len(interval) == 2,
             "primary normalised-rank-regret interval is unavailable")
    conclusion = (
        "rollout training reduces primary equal-family normalised rank regret"
        if interval[0] > 0 else
        "the primary rollout effect on normalised rank regret is not above zero"
    )
    return {
        "schema": "go2_frozen_planning_analysis_v1_2",
        "primary_endpoint": "normalised rank regret; lower is better",
        "primary_estimator": (
            "within each state's oracle-valid candidate subset -> unweighted family "
            "means -> unweighted mean of the seven primary families; a state with "
            "fewer than two oracle-valid candidates is omitted exactly as in the "
            "oracle identifiability estimator; local_composite_motifs is excluded and "
            "diagnostic only"),
        "replication_unit": "the training-seed quadruplet; N=8",
        "states_and_branches_are_not_replications": True,
        "primary_equal_family": primary,
        "corpus_weighted_secondary": corpus_weighted,
        "per_family": {family: value for family, value in per_family.items()
                       if family != DIAGNOSTIC_FAMILY},
        "local_composite_motifs_diagnostic": per_family.get(DIAGNOSTIC_FAMILY),
        "primary_conclusion": conclusion,
        "interpretation_guard": (
            "Prediction fidelity, true-latent scorer qualification, predicted-trajectory "
            "ranking, realised selected utility and proprioception-by-rollout interaction "
            "are distinct. Rollout is not called planning-improving unless regret falls "
            "or realised selected utility rises."),
    }


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def known_storage(bundle: CorpusBundle, scorer_provenance: dict[str, Any],
                  checkpoints: Sequence[FrozenCheckpoint]) -> dict[str, Any]:
    corpus_files = [
        FINAL_DIR / "state_manifest.json", FINAL_DIR / "branch_rows.jsonl",
        FINAL_DIR / "candidate_allocation_manifest.json",
        FINAL_DIR / "corpus_receipt.json", FINAL_DIR / "latents_index.json",
        FINAL_DIR / "final_gate.json",
    ]
    if bundle.context_path is not None:
        corpus_files.append(bundle.context_path)
    else:
        corpus_files.extend(Path(entry["_resolved_path"])
                            for entry in bundle.context_records.values())
    if bundle.horizon_path is not None:
        corpus_files.append(bundle.horizon_path)
    else:
        corpus_files.extend(Path(entry["_resolved_path"])
                            for entry in bundle.horizon_records.values())
    predictor_files = []
    for checkpoint in checkpoints:
        predictor_files.extend(_ledger_paths(checkpoint))
    return {
        "final_corpus_known_files_bytes": sum(path.stat().st_size for path in corpus_files
                                               if path.is_file()),
        "final_corpus_rendered_frames_reported_bytes":
            bundle.receipt.get("storage_bytes"),
        "final_latent_shards_reported_bytes": bundle.index.get("storage_bytes"),
        "scorer_package_bytes": scorer_provenance["package_bytes"],
        "scorer_fit_and_training_reported":
            scorer_provenance["qualification"].get("storage"),
        "frozen_checkpoint_bytes_read_only": sum(item.bytes for item in checkpoints),
        "prediction_ledgers_and_receipts_bytes": sum(
            path.stat().st_size for path in predictor_files if path.is_file()),
        "scope_note": "known bound files only; no recursive custody-root traversal",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=FROZEN_INFERENCE_BATCH,
                        choices=[FROZEN_INFERENCE_BATCH],
                        help="frozen per-state inference batch (cannot be changed)")
    args = parser.parse_args()
    _require(args.batch > 0, "--batch must be positive")

    total_started = time.time()
    scorer_gate_started = time.time()
    scorer_provenance = validate_scorer_artifacts()
    scorer_gate_wall = time.time() - scorer_gate_started
    corpus_gate_started = time.time()
    bundle = validate_final_corpus()
    cross_stage_encoder = validate_cross_stage_encoder(scorer_provenance, bundle)
    gate_report = apply_final_gate(bundle)  # raises before any checkpoint is opened
    corpus_gate_wall = time.time() - corpus_gate_started

    checkpoints, predictor_provenance = verify_frozen_predictor_lineage()
    recovery: list[dict[str, Any]] = []
    prior_result = reuse_completed_result(
        scorer_provenance, bundle, checkpoints, recovery)
    if prior_result is not None:
        print(json.dumps(_safe_json({
            "reused_complete_result": True,
            "report_digest": prior_result["report_digest"],
            "primary_conclusion": prior_result["analysis"]["primary_conclusion"],
            "result": str(RESULT_DIR / "planning_result.json"),
        }), indent=2))
        return 0
    device = D.resolve_device()
    latent_scorer, baseline_scorer, scorer_state_digest = load_scorers(
        scorer_provenance, device)
    normalisation, normalisation_sha = _load_normalisation(
        read_json(RUN_PACKAGE_PATH, "scientific run package")["normalisation_sha256"])

    diagnostics, diagnostics_wall = score_diagnostics(
        bundle, latent_scorer, baseline_scorer, device, args.batch)
    _require(tensor_state_digest(latent_scorer) == scorer_state_digest,
             "shared scorer parameters changed during true-latent diagnostics")

    checkpoint_map = {(item.seed, item.cell): item for item in checkpoints}
    cells: dict[str, dict[int, dict[str, Any]]] = {cell: {} for cell in D.CELLS}
    prediction_receipts: list[dict[str, Any]] = []
    for seed in D.SEED_REGISTRY[:FROZEN_N]:
        for cell in D.CELLS:
            checkpoint = checkpoint_map[(seed, cell)]
            scores, receipt = score_checkpoint(
                checkpoint, bundle, latent_scorer,
                scorer_provenance["package_sha256"], scorer_state_digest,
                normalisation, normalisation_sha, device, args.batch, recovery)
            ledger_path, _receipt_path = _ledger_paths(checkpoint)
            records = read_jsonl_strict(ledger_path, "completed predictor score ledger")
            components = {
                name: np.asarray([row[name] for row in records], dtype=np.float64)
                for name in ("predicted_progress", "predicted_safety",
                             "predicted_completion")
            }
            cells[cell][seed] = evaluate_scores(bundle.rows, scores, components)
            prediction_receipts.append(receipt)
            print(f"[complete] seed {seed} {cell}: primary regret "
                  f"{cells[cell][seed]['primary_equal_family']['normalised_rank_regret']:.6f}",
                  flush=True)

    _require(tensor_state_digest(latent_scorer) == scorer_state_digest,
             "the single shared scorer was not byte-identical across all 32 checkpoints")
    analysis_started = time.time()
    seeds = list(D.SEED_REGISTRY[:FROZEN_N])
    frozen_analysis = analyse(cells, seeds)
    analysis_wall = time.time() - analysis_started

    checkpoint_list = [{
        "seed_index": item.seed_index, "seed": item.seed, "cell": item.cell,
        "epoch": D.CHECKPOINT_EPOCH, "path": str(item.path),
        "sha256": item.sha256, "bytes": item.bytes,
        "checkpoint_receipt": str(item.receipt_path),
        "authorisation_receipt_digest": item.authorisation_receipt_digest,
    } for item in checkpoints]
    report: dict[str, Any] = {
        "schema": "go2_planning_result_v1_2",
        "status": STATUS,
        "complete": True,
        "sequential_gates": {
            "scorer_qualified": True,
            "scorer_qualification_criteria":
                scorer_provenance["qualification"]["criteria"],
            "final_corpus_identifiability": gate_report,
            "predictors_loaded_only_after_both_gates": True,
        },
        "scorer": {
            "contract_digest": contract_digest(),
            "package_sha256": scorer_provenance["package_sha256"],
            "shared_latent_scorer_state_digest": scorer_state_digest,
            "same_scorer_for_all_checkpoints": True,
            "model_specific_calibration": False,
            "qualification": scorer_provenance["qualification"],
            "target_encoder_and_preprocessing_binding": cross_stage_encoder,
        },
        "final_corpus": {
            "state_manifest_digest": bundle.manifest["state_manifest_digest"],
            "corpus_digest": bundle.corpus_digest,
            "branch_rows_sha256": bundle.branch_rows_sha256,
            "context_latent_store_digest": bundle.context_digest,
            "horizon_latent_store_digest": bundle.horizon_digest,
            "states": EXPECTED_STATES, "attempted_branches": EXPECTED_BRANCHES,
        },
        "predictor_provenance": predictor_provenance,
        "scoring_implementation_bindings": scoring_implementation_bindings(),
        "verified_checkpoints": checkpoint_list,
        "prediction_receipts": prediction_receipts,
        "predictor_input_separation": {
            "implementation": "PlanningRow allow-list + _predict_batch signature",
            "observed_inputs_only": True,
            "true_future_used_only_for_separate_upper_bound": True,
            "future_proprioception_supplied": False,
            "oracle_labels_supplied": False,
        },
        "invalid_branch_analysis_policy": {
            "all_twelve_candidates_predictor_scored_and_ledgered": True,
            "outcome_metrics_rank_only_oracle_valid_candidates": True,
            "states_with_fewer_than_two_valid_candidates_omitted": True,
            "invalid_utility_imputation_or_penalty": False,
            "basis": "unchanged oracle-pilot valid-subset population rule",
        },
        "diagnostics": diagnostics,
        "cells": cells,
        "analysis": frozen_analysis,
        "recovery": {
            "events": recovery,
            "invalid_or_interrupted_attempts_preserved": len(recovery) > 0,
        },
        "runtime": {
            "scorer_artifact_gate_wall_time_s_this_invocation":
                round(scorer_gate_wall, 3),
            "scorer_fit_and_qualification":
                scorer_provenance["qualification"].get(
                    "runtime", scorer_provenance["qualification"].get("wall_time_s")),
            "final_corpus_generation": {
                "completed_rows_wall_time_s":
                    bundle.receipt.get("runtime_s_completed_rows"),
                "last_generation_invocation_wall_time_s":
                    bundle.receipt.get("runtime_s_this_invocation"),
            },
            "final_corpus_latent_encoding": {
                "last_encoding_invocation_wall_time_s":
                    bundle.index.get("wall_time_s_this_invocation"),
            },
            "final_corpus_encoder_and_identifiability_gate_wall_time_s_this_invocation":
                round(corpus_gate_wall, 3),
            "true_latent_and_no_latent_diagnostics_wall_time_s":
                round(diagnostics_wall, 3),
            "checkpoint_digest_verification_wall_time_s":
                predictor_provenance["checkpoint_hash_verification_wall_time_s"],
            "predictor_scoring_wall_time_s_this_invocation": round(sum(
                float(receipt.get("wall_time_s_this_invocation", 0.0))
                for receipt in prediction_receipts), 3),
            "final_analysis_wall_time_s": round(analysis_wall, 3),
            "total_wall_time_s_this_invocation": round(time.time() - total_started, 3),
        },
        "storage": known_storage(bundle, scorer_provenance, checkpoints),
        "nothing_left_running_by_this_process": True,
    }
    report = _safe_json(report)
    report["report_digest"] = json_digest(report)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULT_DIR / "planning_result.json"
    atomic_write_json(result_path, report)
    result_receipt: dict[str, Any] = {
        "schema": "go2_planning_result_receipt_v1_2", "complete": True,
        "planning_result_sha256": sha256_file(result_path),
        "report_digest": report["report_digest"],
        "scorer_package_sha256": scorer_provenance["package_sha256"],
        "corpus_digest": bundle.corpus_digest,
        "checkpoint_count": len(checkpoints),
        "prediction_receipts_complete": all(
            receipt.get("complete") is True for receipt in prediction_receipts),
    }
    result_receipt["receipt_digest"] = json_digest(result_receipt)
    atomic_write_json(RESULT_DIR / "planning_result_receipt.json", result_receipt)
    print(json.dumps(_safe_json({
        "report_digest": report["report_digest"],
        "primary_normalised_rank_regret":
            frozen_analysis["primary_equal_family"]["normalised_rank_regret"],
        "primary_realised_selected_utility":
            frozen_analysis["primary_equal_family"]["realised_selected_utility"],
        "primary_conclusion": frozen_analysis["primary_conclusion"],
        "result": str(result_path),
    }), indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FinalCorpusGateFailed as exc:
        print(json.dumps(_safe_json(exc.report), indent=2))
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
    except PlanningRefused as exc:
        print(f"planning scoring refused: {exc}", file=sys.stderr)
        raise SystemExit(3)
