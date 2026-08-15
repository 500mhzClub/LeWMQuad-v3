#!/usr/bin/env python3
"""Train and qualify the frozen shared scorer once on oracle-v1.3 labels.

Only the 96 preserved fit states and 24 prospectively selected fresh
calibration states are admitted.  The model, optimisation budget, paired
no-latent baseline, final-epoch rule, metrics, and thresholds are reused from
the frozen v1.2 scorer implementation.  No predictor artifact or final
evaluation corpus is opened by this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


# Required before torch initialises CUDA deterministic matrix multiplication.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lewm.oracle import go2_scorer_fit_oracle_v1_3_contract as CONTRACT  # noqa: E402
from scripts import encode_go2_scorer_fit_oracle_v1_3 as ENCODER  # noqa: E402
from scripts import train_go2_utility_scorer_v1_2 as FROZEN_TRAINER  # noqa: E402


STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
QUALIFICATION_SCHEMA = CONTRACT.QUALIFICATION_SCHEMA
PACKAGE_SCHEMA = CONTRACT.SCORER_PACKAGE_SCHEMA
PACKAGE_RECEIPT_SCHEMA = CONTRACT.SCORER_PACKAGE_RECEIPT_SCHEMA
BASELINE_SCHEMA = CONTRACT.NO_LATENT_BASELINE_SCHEMA
BASELINE_RECEIPT_SCHEMA = CONTRACT.NO_LATENT_BASELINE_RECEIPT_SCHEMA
TRAINING_AUTHORISATION_SCHEMA = (
    "go2_utility_scorer_oracle_v1_3_training_execution_authorisation_v1")
EVALUATION_AUTHORISATION_SCHEMA = (
    "go2_utility_scorer_oracle_v1_3_qualification_evaluation_authorisation_v1")
QUALIFICATION_SELF_KEY = "qualification_report_digest"
PACKAGE_RECEIPT_SELF_KEY = "scorer_package_receipt_digest"
BASELINE_RECEIPT_SELF_KEY = "baseline_receipt_digest"
TRAINING_AUTHORISATION_SELF_KEY = "training_execution_authorisation_digest"
EVALUATION_AUTHORISATION_SELF_KEY = "qualification_evaluation_authorisation_digest"

EXPECTED_FIT_STATES = 96
EXPECTED_FIT_ROWS = 1_152
EXPECTED_CALIBRATION_STATES = 24
EXPECTED_CALIBRATION_ROWS = 288
EXPECTED_ROWS = 1_440
EXPECTED_UPDATES_PER_EPOCH = 18
EXPECTED_UPDATES_PER_MODEL = 1_080
EXPECTED_PRESENTATIONS_PER_MODEL = 69_120
SOURCE_DIGEST_KEYS = ENCODER.SOURCE_DIGEST_KEYS


class V13TrainingError(RuntimeError):
    """The frozen run, its encoded corpus, or its one-shot terminal changed."""


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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V13TrainingError(message)


def _is_digest(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def scorer_root(root: Path = ROOT) -> Path:
    return ENCODER._generated_root(root) / "scorer"


def qualification_path(root: Path = ROOT) -> Path:
    return root / CONTRACT.QUALIFICATION_PATH


def scorer_package_path(root: Path = ROOT) -> Path:
    return root / CONTRACT.SCORER_PACKAGE_PATH


def scorer_package_receipt_path(root: Path = ROOT) -> Path:
    return root / CONTRACT.SCORER_PACKAGE_RECEIPT_PATH


def baseline_path(root: Path = ROOT) -> Path:
    return root / CONTRACT.NO_LATENT_BASELINE_PATH


def baseline_receipt_path(root: Path = ROOT) -> Path:
    return root / CONTRACT.NO_LATENT_BASELINE_RECEIPT_PATH


def failed_scorer_path(root: Path = ROOT) -> Path:
    return root / CONTRACT.FAILED_SCORER_PATH


def training_authorisation_path(root: Path = ROOT) -> Path:
    return root / CONTRACT.TRAINING_EXECUTION_AUTHORISATION_PATH


def evaluation_authorisation_path(root: Path = ROOT) -> Path:
    return root / CONTRACT.QUALIFICATION_EVALUATION_AUTHORISATION_PATH


def _signed(value: Mapping[str, Any], self_key: str) -> dict[str, Any]:
    payload = dict(value)
    _require(self_key not in payload, f"{self_key} already present")
    payload[self_key] = canonical_digest(payload)
    return payload


def _validate_signed(value: Mapping[str, Any], self_key: str,
                     label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} is not an object")
    payload = dict(value)
    recorded = payload.pop(self_key, None)
    _require(_is_digest(recorded), f"{label} self digest is malformed")
    _require(recorded == canonical_digest(payload),
             f"{label} self digest does not verify")
    payload[self_key] = recorded
    return payload


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True,
                       allow_nan=False) + "\n").encode("utf-8")


def publish_json_once(path: Path, value: Mapping[str, Any], *, label: str) -> None:
    """Install one immutable JSON artifact or require byte identity."""

    raw = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        _require(path.is_file() and not path.is_symlink(),
                 f"{label} is not a regular file")
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


def frozen_training_budget() -> dict[str, Any]:
    budget = dict(FROZEN_TRAINER.SCORER["training"])
    _require(budget == {
        "budget": "fixed epoch budget, FINAL-epoch weights, no best-epoch selection",
        "epochs": 60,
        "batch": 64,
        "lr": 3e-4,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "optimiser": "AdamW",
        "seed": 20260811,
        "fit_calibration_split": "BY SCENE, never by branch or row",
    }, "frozen scorer training budget changed")
    _require(math.ceil(EXPECTED_FIT_ROWS / budget["batch"])
             == EXPECTED_UPDATES_PER_EPOCH
             and budget["epochs"] * EXPECTED_UPDATES_PER_EPOCH
             == EXPECTED_UPDATES_PER_MODEL
             and budget["epochs"] * EXPECTED_FIT_ROWS
             == EXPECTED_PRESENTATIONS_PER_MODEL,
             "frozen training execution counts changed")
    contract_budget = CONTRACT.SCORER_TRAINING_CONTRACT
    _require(isinstance(contract_budget, Mapping),
             "v1.3 scorer training contract is absent")
    nested = contract_budget.get("training", contract_budget)
    for key in ("epochs", "batch", "lr", "weight_decay", "grad_clip",
                "optimiser", "seed"):
        _require(nested.get(key) == budget[key],
                 f"v1.3 contract changed frozen training field {key}")
    _require({
        key: contract_budget.get(key) for key in (
            "updates_per_epoch", "updates_per_model", "final_epoch_only",
            "shared_scorer_runs", "required_no_latent_baseline_runs",
            "paired_no_latent_baseline_required",
            "retry_or_hyperparameter_tuning_after_outcomes",
            "missing_label_policy", "degenerate_label_policy",
        )
    } == {
        "updates_per_epoch": EXPECTED_UPDATES_PER_EPOCH,
        "updates_per_model": EXPECTED_UPDATES_PER_MODEL,
        "final_epoch_only": True,
        "shared_scorer_runs": 1,
        "required_no_latent_baseline_runs": 1,
        "paired_no_latent_baseline_required": True,
        "retry_or_hyperparameter_tuning_after_outcomes": False,
        "missing_label_policy": "stop before encoding/training",
        "degenerate_label_policy":
            "stop before training or fail qualification",
    }, "v1.3 one-shot scorer contract changed")
    _require(dict(FROZEN_TRAINER.WEIGHTS) == {
                 "progress": 1.0, "safety": -2.0, "completion": 0.5
             }
             and (FROZEN_TRAINER.TOKENS, FROZEN_TRAINER.TOKEN_DIM,
                  FROZEN_TRAINER.HORIZONS, FROZEN_TRAINER.HIDDEN_DIM,
                  FROZEN_TRAINER.ACTION_DIM, FROZEN_TRAINER.GOAL_DIM)
             == (768, 1_024, 4, 512, 40, 3),
             "protected scorer architecture or utility weights changed")
    return budget


def validate_frozen_qualification_thresholds() -> dict[str, Any]:
    expected = {
        "progress_spearman_min": 0.50,
        "safety_roc_auc_min": 0.75,
        "safety_ece_max": 0.10,
        "completion_roc_auc_min": 0.75,
        "completion_ece_max": 0.10,
        "composite_within_state_pairwise_accuracy_min": 0.65,
        "no_latent_pairwise_margin_min": 0.05,
        "completion_nondegenerate_in_fit_and_calibration": True,
        "conjunction_required": True,
        "tie_tolerance": 0.02,
        "evaluation_count": 1,
        "failure_is_terminal": True,
    }
    observed = dict(CONTRACT.QUALIFICATION_THRESHOLDS)
    _require(observed == expected and FROZEN_TRAINER.TIE_TOLERANCE == 0.02,
             "v1.3 qualification thresholds changed")
    return observed


def training_execution_counts() -> dict[str, Any]:
    frozen_training_budget()
    return {
        "fit_examples": EXPECTED_FIT_ROWS,
        "calibration_examples": EXPECTED_CALIBRATION_ROWS,
        "epochs": 60,
        "batch_size": 64,
        "optimizer_updates_per_epoch": EXPECTED_UPDATES_PER_EPOCH,
        "optimizer_updates_per_model": EXPECTED_UPDATES_PER_MODEL,
        "example_presentations_per_model": EXPECTED_PRESENTATIONS_PER_MODEL,
        "models": ["shared_true_latent_scorer", "no_latent_baseline"],
        "registered_training_executions": 1,
        "qualification_evaluations": 1,
        "final_epoch_only": True,
        "best_epoch_selection_permitted": False,
        "retune_or_second_seed_permitted": False,
    }


def qualification_criteria(
        latent_calibration: Mapping[str, Any],
        baseline_calibration: Mapping[str, Any],
        fit_distribution: Mapping[str, Any],
        calibration_distribution: Mapping[str, Any],
        ) -> tuple[dict[str, bool], dict[str, Any], float]:
    """Delegate the exact frozen metrics and conjunctive thresholds."""

    validate_frozen_qualification_thresholds()
    return FROZEN_TRAINER.qualification_criteria(
        latent_calibration, baseline_calibration,
        fit_distribution, calibration_distribution)


def _normalised_row(row: Mapping[str, Any], latent_index: int) -> dict[str, Any]:
    result = dict(row)
    projection = row.get("label_projection")
    if isinstance(projection, Mapping):
        for key in ("progress", "safety", "completion", "utility"):
            result[key] = projection[key]
    result["split_role"] = ENCODER._row_role(row)
    result["_latent_index"] = latent_index
    return result


def corpus_from_encoded_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Join rows to exact latent records without opening predictor material."""

    view = ENCODER.validate_training_view_structure(bundle["view"])
    index = ENCODER.validate_latent_index(
        bundle["index"], view, root=Path(bundle.get("root", ROOT)),
        verify_encoder_checkpoint=False)
    records = index["horizon_records"]
    positions = {record["training_view_row_digest"]: position
                 for position, record in enumerate(records)}
    _require(len(positions) == EXPECTED_ROWS,
             "latent record identities are duplicated")
    rows = []
    for row in view["rows"]:
        digest = row["training_view_row_digest"]
        _require(digest in positions, "training row lacks a true latent")
        rows.append(_normalised_row(row, positions[digest]))
    rows.sort(key=lambda row: (
        row["split_role"] != "fit", str(row["state_id"]),
        int(row["candidate_index"])))
    fit = [row for row in rows if row["split_role"] == "fit"]
    calibration = [row for row in rows if row["split_role"] == "calibration"]
    _require(len(fit) == EXPECTED_FIT_ROWS
             and len({row["state_id"] for row in fit}) == EXPECTED_FIT_STATES,
             "fit corpus is not 96 states / 1,152 rows")
    _require(len(calibration) == EXPECTED_CALIBRATION_ROWS
             and len({row["state_id"] for row in calibration})
             == EXPECTED_CALIBRATION_STATES,
             "calibration corpus is not 24 states / 288 rows")
    _require(not ({row["scene_id"] for row in fit}
                  & {row["scene_id"] for row in calibration}),
             "fit/calibration scenes overlap")
    store = FROZEN_TRAINER.HorizonShardStore(
        records, Path(bundle.get("encoded_root", ENCODER.encoded_root())))
    return {"view": view, "index": index, "rows": rows,
            "fit_rows": fit, "calibration_rows": calibration,
            "horizon": store}


def _configure_frozen_trainer(root: Path) -> None:
    FROZEN_TRAINER.PACKAGE_DIR = scorer_root(root)
    FROZEN_TRAINER.INITIALISATIONS_DIR_NAME = "initialisations"
    FROZEN_TRAINER.TRAINING_DIR_NAME = "training"


def _validate_registered_initialisation(
        name: str, registration: Mapping[str, Any], *, root: Path) -> None:
    expected = scorer_root(root) / "initialisations" / f"{name}.pt"
    path = Path(str(registration.get("path", "")))
    _require(not registration.get("rejected_registrations")
             and registration.get("recovery_decision") in {
                 "registered_frozen_initialisation",
                 "reused_verified_registered_initialisation",
             }
             and path.resolve() == expected.resolve()
             and path.is_file() and not path.is_symlink(),
             f"{name} lacks its one canonical registered initialisation")


def _preflight_registered_training(
        name: str, *, use_latent: bool,
        registration: Mapping[str, Any], training_run_digest: str,
        device: torch.device, budget: Mapping[str, Any], training_rows: int,
        continuation_authorised: bool, root: Path) -> None:
    """Reject any state the inherited trainer would turn into a restart."""

    model_root = scorer_root(root) / "training" / name
    attempts = sorted(path for path in model_root.glob("attempt_*")
                      if path.is_dir() and not path.is_symlink()) \
        if model_root.is_dir() and not model_root.is_symlink() else []
    candidates = FROZEN_TRAINER._checkpoint_candidates(model_root) \
        if model_root.is_dir() and not model_root.is_symlink() else []
    _require(not attempts or continuation_authorised,
             f"{name} has training attempts without the immutable run authority")
    _require(not attempts or candidates,
             f"{name} has a non-resumable prior attempt; retry is forbidden")
    if not candidates:
        return
    checkpoint_path = candidates[0]
    _require(checkpoint_path.is_file() and not checkpoint_path.is_symlink(),
             f"{name} newest checkpoint is not a regular file")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu",
                             weights_only=False)
        execution = FROZEN_TRAINER._execution_fingerprint(device)
        FROZEN_TRAINER._validate_checkpoint(
            payload, name=name, use_latent=use_latent,
            training_run_digest=training_run_digest,
            initial_state_digest=registration["initial_state_digest"],
            execution=execution, training_rows=training_rows,
            epochs=int(budget["epochs"]), path=checkpoint_path)
        probe = FROZEN_TRAINER.UtilityScorer(use_latent=use_latent)
        probe.load_state_dict(payload["model_state_dict"], strict=True)
        probe.to(device)
        optimiser = FROZEN_TRAINER._new_optimiser(probe, budget)
        optimiser.load_state_dict(payload["optimizer_state_dict"])
        generator = torch.Generator(device="cpu")
        generator.set_state(payload["order_generator_state"])
    except Exception as exc:
        raise V13TrainingError(
            f"{name} newest checkpoint is not exactly resumable; retry is "
            f"forbidden: {exc}") from exc


def _binding_payload(corpus: Mapping[str, Any]) -> dict[str, Any]:
    view, index = corpus["view"], corpus["index"]
    return {
        "schema": "go2_utility_scorer_oracle_v1_3_training_binding_v1",
        "oracle_v1_3_digest": view["oracle_v1_3_digest"],
        "scorer_fit_oracle_v1_3_contract_digest":
            view["scorer_fit_oracle_v1_3_contract_digest"],
        "authority_digest": view["authority_digest"],
        **{key: view[key] for key in SOURCE_DIGEST_KEYS},
        "training_view_digest": index["training_view_digest"],
        "latent_index_digest": index[ENCODER.LATENT_INDEX_SELF_KEY],
        "encoding_receipt_digest": None,
        "normalisation": FROZEN_TRAINER.NORMALISATION,
        "architecture": {
            "tokens": FROZEN_TRAINER.TOKENS,
            "token_dim": FROZEN_TRAINER.TOKEN_DIM,
            "horizons": FROZEN_TRAINER.HORIZONS,
            "hidden": FROZEN_TRAINER.HIDDEN_DIM,
            "action_dim": FROZEN_TRAINER.ACTION_DIM,
            "goal_dim": FROZEN_TRAINER.GOAL_DIM,
            "separate_component_heads": True,
            "paired_no_latent_baseline": True,
        },
        "training": frozen_training_budget(),
        "training_execution_counts": training_execution_counts(),
        "utility_weights": dict(FROZEN_TRAINER.WEIGHTS),
        "qualification_thresholds": validate_frozen_qualification_thresholds(),
        "learning_rate_schedule": "constant",
        "final_epoch_only": True,
        "epoch_selection_permitted": False,
        "model_specific_calibration": None,
    }


def _authorisation_payload(
        *, binding_digest: str, training_run_digest: str,
        initialisations: Mapping[str, Mapping[str, Any]],
        ) -> dict[str, Any]:
    return _signed({
        "schema": TRAINING_AUTHORISATION_SCHEMA,
        "status": STATUS,
        "complete": True,
        "binding_digest": binding_digest,
        "training_run_digest": training_run_digest,
        "registered_training_executions": 1,
        "model_initial_state_digests": {
            name: value["initial_state_digest"]
            for name, value in initialisations.items()
        },
        "models": ["latent", "no_latent"],
        "retune_permitted": False,
        "second_seed_permitted": False,
        "predictor_checkpoints_opened": 0,
        "predictor_utility_shards_opened": 0,
        "final_200_state_corpus_generated": False,
    }, TRAINING_AUTHORISATION_SELF_KEY)


def _evaluation_authorisation_payload(
        *, training_run_digest: str,
        final_state_digests: Mapping[str, str],
        ) -> dict[str, Any]:
    return _signed({
        "schema": EVALUATION_AUTHORISATION_SCHEMA,
        "status": STATUS,
        "complete": True,
        "training_run_digest": training_run_digest,
        "final_state_digests": dict(final_state_digests),
        "qualification_evaluations_authorised": 1,
        "qualification_evaluations_completed_before_issue": 0,
        "repeat_after_interruption_permitted": False,
        "predictor_checkpoints_opened": 0,
        "predictor_utility_shards_opened": 0,
        "final_200_state_corpus_generated": False,
    }, EVALUATION_AUTHORISATION_SELF_KEY)


def _safe_json(value: Any) -> Any:
    return FROZEN_TRAINER._safe_json(value)


def _write_torch_once(payload: Mapping[str, Any], path: Path,
                      identity: Mapping[str, Any]) -> str:
    _require(not path.is_symlink(), "refusing a symlinked scorer artifact")
    digest = FROZEN_TRAINER._write_once_torch(
        dict(payload), path, identity)
    _require(path.is_file() and not path.is_symlink()
             and file_sha256(path) == digest,
             "immutable scorer artifact bytes changed")
    installed = torch.load(path, map_location="cpu", weights_only=False)
    _require(isinstance(installed, Mapping)
             and all(installed.get(key) == value
                     for key, value in identity.items()),
             "immutable scorer artifact identity changed")
    if "final_state_digest" in identity:
        _require(FROZEN_TRAINER.state_dict_digest(
                     installed["model_state_dict"])
                 == identity["final_state_digest"],
                 "baseline artifact state digest changed")
    if "final_state_digests" in identity:
        _require(all(FROZEN_TRAINER.state_dict_digest(installed[name])
                     == expected for name, expected
                     in identity["final_state_digests"].items()),
                 "shared scorer artifact state digest changed")
    return digest


def _terminal_common(corpus: Mapping[str, Any]) -> dict[str, Any]:
    view, index = corpus["view"], corpus["index"]
    return {
        "schema": QUALIFICATION_SCHEMA,
        "status": STATUS,
        "oracle_v1_3_digest": view["oracle_v1_3_digest"],
        "scorer_fit_oracle_v1_3_contract_digest":
            view["scorer_fit_oracle_v1_3_contract_digest"],
        "authority_digest": view["authority_digest"],
        **{key: view[key] for key in SOURCE_DIGEST_KEYS},
        "training_view_digest": index["training_view_digest"],
        "latent_index_digest": index[ENCODER.LATENT_INDEX_SELF_KEY],
        "fit_states": EXPECTED_FIT_STATES,
        "fit_rows": EXPECTED_FIT_ROWS,
        "calibration_states": EXPECTED_CALIBRATION_STATES,
        "calibration_rows": EXPECTED_CALIBRATION_ROWS,
        "scene_disjoint": True,
        "historical_calibration_qualification_rows": 0,
        "historical_calibration_disposition":
            view["historical_calibration_disposition"],
        "target_encoder_digest": index["target_encoder_digest"],
        "target_encoder_checkpoint_sha256":
            index["target_encoder_checkpoint_sha256"],
        "preprocess_contract_digest": index["preprocess_contract_digest"],
        "preprocessing_digest": index["preprocessing_digest"],
        "predictor_checkpoints_loaded": 0,
        "predictor_utility_shards_opened": 0,
        "final_200_state_corpus_generated": False,
    }


def _pretraining_failure(corpus: Mapping[str, Any], gate: Mapping[str, Any],
                         *, root: Path) -> dict[str, Any]:
    report = _signed({
        **_terminal_common(corpus),
        "terminal_kind": "PRETRAINING_COMPLETION_DEGENERACY_FAILURE",
        "complete": True,
        "qualified": False,
        "training_execution_count": 0,
        "qualification_evaluations": 0,
        "completion_degeneracy_gate": dict(gate),
        "criteria": {},
        "epoch_selection_permitted": False,
        "retry_or_retune_permitted": False,
    }, QUALIFICATION_SELF_KEY)
    publish_json_once(
        qualification_path(root), report,
        label="v1.3 pretraining failure terminal")
    return report


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(),
             f"{label} is missing or not regular")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise V13TrainingError(f"cannot read {label}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} is not an object")
    return value


def load_and_validate_training_terminal_for_consumption(
        *, root: Path = ROOT, require_qualified: bool | None = None,
        verify_encoder_checkpoint: bool = False,
        ) -> dict[str, Any]:
    encoded = ENCODER.load_and_validate_encoded_training_view_for_consumption(
        root=root, verify_encoder_checkpoint=verify_encoder_checkpoint)
    encoded = {**encoded, "root": root}
    corpus = corpus_from_encoded_bundle(encoded)
    report = _validate_signed(
        _read_json(qualification_path(root), label="v1.3 qualification terminal"),
        QUALIFICATION_SELF_KEY, "v1.3 qualification terminal")
    common = _terminal_common(corpus)
    _require(all(report.get(key) == value for key, value in common.items()),
             "v1.3 qualification terminal corpus lineage changed")
    kind = report.get("terminal_kind")
    _require(kind in {"PRETRAINING_COMPLETION_DEGENERACY_FAILURE",
                      "QUALIFICATION_PASS", "QUALIFICATION_FAILURE"},
             "v1.3 qualification terminal kind changed")
    qualified = report.get("qualified") is True
    _require(qualified == (kind == "QUALIFICATION_PASS"),
             "v1.3 terminal verdict and kind disagree")
    if require_qualified is not None:
        _require(qualified is require_qualified,
                 "v1.3 scorer has the wrong required verdict")
    if kind == "PRETRAINING_COMPLETION_DEGENERACY_FAILURE":
        gate = FROZEN_TRAINER.full_bank_v2_completion_degeneracy(
            corpus["fit_rows"], corpus["calibration_rows"])
        _require(report.get("complete") is True
                 and report.get("training_execution_count") == 0
                 and report.get("qualification_evaluations") == 0
                 and report.get("completion_degeneracy_gate") == gate
                 and gate["pass"] is False,
                 "pretraining degeneracy failure does not replay")
        return {"terminal": report, "terminal_kind": kind,
                "terminal_digest": report[QUALIFICATION_SELF_KEY],
                "qualified": False,
                "predictor_artifact_access_authorised": False,
                "corpus": corpus}

    _require(report.get("training_execution_count") == 1
             and report.get("qualification_evaluations") == 1
             and report.get("epoch_selection_permitted") is False
             and report.get("retry_or_retune_permitted") is False
             and report.get("training_execution_counts")
             == training_execution_counts(),
             "v1.3 one-shot execution accounting changed")
    criteria = report.get("criteria")
    results = report.get("results")
    distributions = report.get("label_distributions")
    _require(isinstance(criteria, Mapping)
             and isinstance(results, Mapping)
             and isinstance(distributions, Mapping),
             "v1.3 terminal metrics are absent")
    expected_criteria, expected_details, expected_dominance = qualification_criteria(
        results["latent"]["calibration"],
        results["no_latent"]["calibration"],
        distributions["fit"]["overall"],
        distributions["calibration"]["overall"])
    _require(dict(criteria) == expected_criteria
             and report.get("criterion_details") == expected_details
             and report.get("baseline_dominance_pairwise") == expected_dominance
             and qualified == all(criteria.values()),
             "v1.3 frozen qualification criteria do not replay")
    training_auth = _validate_signed(
        _read_json(training_authorisation_path(root),
                   label="training authorisation"),
        TRAINING_AUTHORISATION_SELF_KEY, "training authorisation")
    evaluation_auth = _validate_signed(
        _read_json(evaluation_authorisation_path(root),
                   label="qualification evaluation authorisation"),
        EVALUATION_AUTHORISATION_SELF_KEY,
        "qualification evaluation authorisation")
    expected_training_auth = _authorisation_payload(
        binding_digest=str(report.get("binding_digest")),
        training_run_digest=str(report.get("training_run_digest")),
        initialisations=report.get("initialisations", {}))
    expected_evaluation_auth = _evaluation_authorisation_payload(
        training_run_digest=str(report.get("training_run_digest")),
        final_state_digests=report.get("final_state_digests", {}))
    receipts = report.get("training_receipts")
    _require(training_auth == expected_training_auth
             and evaluation_auth == expected_evaluation_auth
             and report.get("training_execution_authorisation_digest")
             == training_auth[TRAINING_AUTHORISATION_SELF_KEY]
             and report.get("qualification_evaluation_authorisation_digest")
             == evaluation_auth[EVALUATION_AUTHORISATION_SELF_KEY]
             and isinstance(receipts, Mapping)
             and set(receipts) == {"latent", "no_latent"}
             and all(isinstance(receipt, Mapping)
                     and not receipt.get("rejected_checkpoints")
                     and receipt.get("final_epoch") == 60
                     and receipt.get("epoch_selection")
                     == "final_epoch_only_no_selection"
                     for receipt in receipts.values()),
             "one-shot authorisations differ from terminal")

    baseline_receipt = _validate_signed(
        _read_json(baseline_receipt_path(root), label="baseline receipt"),
        BASELINE_RECEIPT_SELF_KEY, "baseline receipt")
    _require(baseline_path(root).is_file()
             and not baseline_path(root).is_symlink()
             and baseline_path(root).stat().st_size
             == baseline_receipt.get("byte_count")
             and file_sha256(baseline_path(root))
             == baseline_receipt.get("sha256")
             and report.get("no_latent_baseline_receipt") == baseline_receipt,
             "v1.3 no-latent baseline package changed")
    result: dict[str, Any] = {
        "terminal": report, "terminal_kind": kind,
        "terminal_digest": report[QUALIFICATION_SELF_KEY],
        "qualified": qualified,
        "predictor_artifact_access_authorised": False,
        "corpus": corpus,
        "baseline_path": baseline_path(root),
        "baseline_receipt": baseline_receipt,
    }
    if qualified:
        package_receipt = _validate_signed(
            _read_json(scorer_package_receipt_path(root),
                       label="scorer package receipt"),
            PACKAGE_RECEIPT_SELF_KEY, "scorer package receipt")
        _require(scorer_package_path(root).is_file()
                 and not scorer_package_path(root).is_symlink()
                 and scorer_package_path(root).stat().st_size
                 == package_receipt.get("byte_count")
                 and file_sha256(scorer_package_path(root))
                 == package_receipt.get("sha256")
                 == report.get("scorer_package_sha256"),
                 "qualified v1.3 scorer package changed")
        result.update({"scorer_package_path": scorer_package_path(root),
                       "scorer_package_receipt": package_receipt})
    else:
        _require(failed_scorer_path(root).is_file()
                 and not failed_scorer_path(root).is_symlink()
                 and file_sha256(failed_scorer_path(root))
                 == report.get("failed_scorer_sha256"),
                 "failed v1.3 scorer package changed")
    return result


def _optional_terminal(*, root: Path) -> dict[str, Any] | None:
    path = qualification_path(root)
    if not path.exists() and not path.is_symlink():
        return None
    return load_and_validate_training_terminal_for_consumption(
        root=root, verify_encoder_checkpoint=False)


def train_and_qualify(*, root: Path = ROOT,
                      device_name: str = "auto") -> dict[str, Any]:
    ENCODER._require_registered_generated_root(root)
    retained = _optional_terminal(root=root)
    if retained is not None:
        return retained
    evaluation_path = evaluation_authorisation_path(root)
    _require(not evaluation_path.exists() and not evaluation_path.is_symlink(),
             "qualification evaluation was already authorised without a "
             "terminal; repeating it is forbidden")
    _configure_frozen_trainer(root)
    encoded = ENCODER.load_and_validate_encoded_training_view_for_consumption(
        root=root, verify_encoder_checkpoint=False)
    encoded = {**encoded, "root": root}
    corpus = corpus_from_encoded_bundle(encoded)
    fit_rows = corpus["fit_rows"]
    calibration_rows = corpus["calibration_rows"]
    completion_gate = FROZEN_TRAINER.full_bank_v2_completion_degeneracy(
        fit_rows, calibration_rows)
    if completion_gate["pass"] is not True:
        report = _pretraining_failure(
            corpus, completion_gate, root=root)
        return {"terminal": report,
                "terminal_kind": report["terminal_kind"],
                "terminal_digest": report[QUALIFICATION_SELF_KEY],
                "qualified": False,
                "predictor_artifact_access_authorised": False,
                "corpus": corpus}

    if device_name == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    _require(device.type != "cuda" or torch.cuda.is_available(),
             "CUDA was requested but is unavailable")
    budget = frozen_training_budget()
    binding = _binding_payload(corpus)
    binding["encoding_receipt_digest"] = encoded["receipt"][
        ENCODER.ENCODING_RECEIPT_SELF_KEY]
    binding_digest = canonical_digest(binding)

    fit_features = FROZEN_TRAINER.features(
        fit_rows, corpus["horizon"], device)
    calibration_features = FROZEN_TRAINER.features(
        calibration_rows, corpus["horizon"], device)
    models: dict[str, FROZEN_TRAINER.UtilityScorer] = {}
    initialisations: dict[str, dict[str, Any]] = {}
    for name, use_latent in (("latent", True), ("no_latent", False)):
        model, registration = FROZEN_TRAINER.register_initialisation(
            name, use_latent=use_latent, seed=int(budget["seed"]),
            binding_digest=binding_digest)
        models[name] = model
        initialisations[name] = registration
        _validate_registered_initialisation(
            name, registration, root=root)
    training_run_digest = canonical_digest({
        "binding_digest": binding_digest,
        "initial_state_digests": {
            name: value["initial_state_digest"]
            for name, value in initialisations.items()
        },
    })
    training_auth = _authorisation_payload(
        binding_digest=binding_digest,
        training_run_digest=training_run_digest,
        initialisations=initialisations)
    training_auth_path = training_authorisation_path(root)
    continuation_authorised = (
        training_auth_path.exists() or training_auth_path.is_symlink())
    publish_json_once(
        training_auth_path, training_auth,
        label="v1.3 one-shot training authorisation")

    for name, use_latent in (("latent", True), ("no_latent", False)):
        _preflight_registered_training(
            name, use_latent=use_latent,
            registration=initialisations[name],
            training_run_digest=training_run_digest, device=device,
            budget=budget, training_rows=len(fit_rows),
            continuation_authorised=continuation_authorised, root=root)

    packages: dict[str, dict[str, torch.Tensor]] = {}
    training_receipts: dict[str, dict[str, Any]] = {}
    for name, use_latent in (("latent", True), ("no_latent", False)):
        packages[name], training_receipts[name] = (
            FROZEN_TRAINER.train_registered_model(
                name, models[name], use_latent=use_latent,
                latent=fit_features[0], action_goal=fit_features[1],
                targets=fit_features[2], device=device, budget=budget,
                training_run_digest=training_run_digest,
                initialisation=initialisations[name]))
        _require(not training_receipts[name].get("rejected_checkpoints"),
                 "invalid checkpoint would turn the one run into a retry")
        _require(training_receipts[name].get("final_epoch") == budget["epochs"]
                 and training_receipts[name].get("epoch_selection")
                 == "final_epoch_only_no_selection",
                 f"{name} training did not end at the frozen final epoch")
        models[name].load_state_dict(packages[name], strict=True)
        models[name].to(device)
    final_state_digests = {
        name: FROZEN_TRAINER.state_dict_digest(state)
        for name, state in packages.items()
    }
    evaluation_auth = _evaluation_authorisation_payload(
        training_run_digest=training_run_digest,
        final_state_digests=final_state_digests)
    publish_json_once(
        evaluation_path, evaluation_auth,
        label="v1.3 one-shot qualification evaluation authorisation")

    results: dict[str, Any] = {}
    calibration_predictions: dict[str, dict[str, np.ndarray]] = {}
    for name in ("latent", "no_latent"):
        fit_result, _ = FROZEN_TRAINER.evaluate_model(
            models[name], fit_features[0], fit_features[1],
            fit_rows, fit_features[2])
        calibration_result, predictions = FROZEN_TRAINER.evaluate_model(
            models[name], calibration_features[0], calibration_features[1],
            calibration_rows, calibration_features[2])
        results[name] = {
            "fit": fit_result,
            "calibration": calibration_result,
            "per_family_calibration": FROZEN_TRAINER._grouped_calibration(
                calibration_rows, calibration_features[2], predictions,
                "family"),
            "per_stratum_calibration": FROZEN_TRAINER._grouped_calibration(
                calibration_rows, calibration_features[2], predictions,
                "stratum"),
        }
        calibration_predictions[name] = predictions
    fit_distribution = FROZEN_TRAINER.label_distribution(fit_rows)
    calibration_distribution = FROZEN_TRAINER.label_distribution(calibration_rows)
    grouped_distributions = {
        "fit": FROZEN_TRAINER.grouped_label_distributions(fit_rows),
        "calibration": FROZEN_TRAINER.grouped_label_distributions(
            calibration_rows),
    }
    criteria, criterion_details, dominance = qualification_criteria(
        results["latent"]["calibration"],
        results["no_latent"]["calibration"],
        fit_distribution, calibration_distribution)
    qualified = all(criteria.values())
    paired = FROZEN_TRAINER._paired_baseline_diagnostics(
        calibration_rows,
        calibration_predictions["latent"]["utility"],
        calibration_predictions["no_latent"]["utility"])

    artifact_common = {
        "status": STATUS,
        "training_run_digest": training_run_digest,
        "binding_digest": binding_digest,
        "bindings": binding,
        "oracle_v1_3_digest": corpus["view"]["oracle_v1_3_digest"],
        "scorer_fit_oracle_v1_3_contract_digest":
            corpus["view"]["scorer_fit_oracle_v1_3_contract_digest"],
        "authority_digest": corpus["view"]["authority_digest"],
        **{key: corpus["view"][key] for key in SOURCE_DIGEST_KEYS},
        "training_view_digest": corpus["index"]["training_view_digest"],
        "latent_index_digest": corpus["index"][ENCODER.LATENT_INDEX_SELF_KEY],
        "initial_state_digests": {
            name: value["initial_state_digest"]
            for name, value in initialisations.items()
        },
        "final_state_digests": final_state_digests,
        "final_epoch": int(budget["epochs"]),
        "epoch_selection": "final_epoch_only_no_selection",
        "normalisation": FROZEN_TRAINER.NORMALISATION,
        "utility_weights": dict(FROZEN_TRAINER.WEIGHTS),
        "qualified": qualified,
    }
    baseline_artifact = {
        "schema": BASELINE_SCHEMA,
        **artifact_common,
        "model_state_dict": packages["no_latent"],
        "final_state_digest": final_state_digests["no_latent"],
    }
    baseline_sha = _write_torch_once(
        baseline_artifact, baseline_path(root), {
            "schema": BASELINE_SCHEMA,
            "training_run_digest": training_run_digest,
            "final_state_digest": final_state_digests["no_latent"],
        })
    baseline_receipt = _signed({
        "schema": BASELINE_RECEIPT_SCHEMA,
        "status": STATUS,
        "complete": True,
        "training_run_digest": training_run_digest,
        "path": str(baseline_path(root).relative_to(root)),
        "sha256": baseline_sha,
        "byte_count": baseline_path(root).stat().st_size,
        "final_state_digest": final_state_digests["no_latent"],
        "final_epoch": int(budget["epochs"]),
    }, BASELINE_RECEIPT_SELF_KEY)
    publish_json_once(
        baseline_receipt_path(root), baseline_receipt,
        label="v1.3 no-latent baseline receipt")

    shared_artifact = {
        "schema": PACKAGE_SCHEMA,
        **artifact_common,
        "latent": packages["latent"],
        "no_latent": packages["no_latent"],
        "qualification_criteria": criteria,
    }
    scorer_sha: str | None = None
    failed_sha: str | None = None
    package_receipt: dict[str, Any] | None = None
    identity = {
        "schema": PACKAGE_SCHEMA,
        "training_run_digest": training_run_digest,
        "qualified": qualified,
        "final_state_digests": final_state_digests,
    }
    if qualified:
        scorer_sha = _write_torch_once(
            shared_artifact, scorer_package_path(root), identity)
        package_receipt = _signed({
            "schema": PACKAGE_RECEIPT_SCHEMA,
            "status": STATUS,
            "complete": True,
            "qualified": True,
            "training_run_digest": training_run_digest,
            "path": str(scorer_package_path(root).relative_to(root)),
            "sha256": scorer_sha,
            "byte_count": scorer_package_path(root).stat().st_size,
            "final_state_digests": final_state_digests,
            "binding_digest": binding_digest,
        }, PACKAGE_RECEIPT_SELF_KEY)
        publish_json_once(
            scorer_package_receipt_path(root), package_receipt,
            label="qualified v1.3 scorer package receipt")
    else:
        failed_sha = _write_torch_once(
            shared_artifact, failed_scorer_path(root), identity)

    report = _safe_json({
        **_terminal_common(corpus),
        "terminal_kind": "QUALIFICATION_PASS" if qualified
            else "QUALIFICATION_FAILURE",
        "complete": True,
        "qualified": qualified,
        "training_execution_count": 1,
        "qualification_evaluations": 1,
        "training_run_digest": training_run_digest,
        "binding_digest": binding_digest,
        "training_execution_authorisation_digest":
            training_auth[TRAINING_AUTHORISATION_SELF_KEY],
        "qualification_evaluation_authorisation_digest":
            evaluation_auth[EVALUATION_AUTHORISATION_SELF_KEY],
        "training_execution_counts": training_execution_counts(),
        "epoch_selection_permitted": False,
        "retry_or_retune_permitted": False,
        "label_distributions": grouped_distributions,
        "completion_prevalence_by_split_and_family":
            FROZEN_TRAINER.completion_by_split_family(
                fit_rows, calibration_rows),
        "results": results,
        "baseline_dominance_pairwise": dominance,
        "paired_latent_vs_no_latent_calibration": paired,
        "criterion_details": criterion_details,
        "criteria": criteria,
        "initialisations": initialisations,
        "training_receipts": training_receipts,
        "final_state_digests": final_state_digests,
        "scorer_package_sha256": scorer_sha,
        "failed_scorer_sha256": failed_sha,
        "scorer_package_receipt": package_receipt,
        "no_latent_baseline_receipt": baseline_receipt,
    })
    report = _signed(report, QUALIFICATION_SELF_KEY)
    publish_json_once(
        qualification_path(root), report,
        label="v1.3 immutable qualification terminal")
    return load_and_validate_training_terminal_for_consumption(
        root=root, require_qualified=qualified,
        verify_encoder_checkpoint=False)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args(argv)
    result = train_and_qualify(device_name=args.device)
    terminal = result["terminal"]
    print(json.dumps({
        "status": terminal["terminal_kind"],
        "qualified": result["qualified"],
        "qualification_report_digest": result["terminal_digest"],
        "training_execution_count": terminal["training_execution_count"],
        "qualification_evaluations": terminal["qualification_evaluations"],
        "predictor_utility_shards_opened": 0,
        "final_200_state_corpus_generated": False,
    }, indent=2, sort_keys=True))
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
