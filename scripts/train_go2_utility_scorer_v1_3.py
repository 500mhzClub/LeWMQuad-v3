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
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
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
INVALID_ATTEMPT_SELF_KEY = CONTRACT.INVALID_TRAINING_ATTEMPT_SELF_KEY
REPLACEMENT_AUTHORISATION_SELF_KEY = (
    CONTRACT.SCORER_TRAINING_REPLACEMENT_AUTHORISATION_SELF_KEY
)

EXPECTED_FIT_STATES = 96
EXPECTED_FIT_ROWS = 1_152
EXPECTED_CALIBRATION_STATES = 24
EXPECTED_CALIBRATION_ROWS = 288
EXPECTED_ROWS = 1_440
EXPECTED_UPDATES_PER_EPOCH = 18
EXPECTED_UPDATES_PER_MODEL = 1_080
EXPECTED_PRESENTATIONS_PER_MODEL = 69_120
SOURCE_DIGEST_KEYS = ENCODER.SOURCE_DIGEST_KEYS
REPAIR_CHANGED_PATHS = frozenset({
    "lewm/oracle/go2_scorer_fit_oracle_v1_3_contract.py",
    "lewm/tests/test_train_go2_utility_scorer_v1_2.py",
    "lewm/tests/test_train_go2_utility_scorer_v1_3.py",
    "lewm/tests/test_go2_scorer_fit_oracle_v1_3_contract.py",
    "scripts/train_go2_utility_scorer_v1_2.py",
    "scripts/train_go2_utility_scorer_v1_3.py",
})
SERIALIZER_SMOKE_NODE = (
    "lewm/tests/test_train_go2_utility_scorer_v1_2.py::"
    "UtilityScorerTrainerTests::"
    "test_tiny_production_training_checkpoint_receipt_smoke"
)
ORIGINAL_EXCEPTION_TYPE = "RuntimeError"
ORIGINAL_EXCEPTION_MESSAGE = (
    "self.dim() cannot be 0 to view Float as Byte (different element sizes)"
)
ORIGINAL_TRACEBACK_FRAMES = [
    {
        "path": "scripts/train_go2_utility_scorer_v1_3.py", "line": 1021,
        "source": "raise SystemExit(main())",
    },
    {
        "path": "scripts/train_go2_utility_scorer_v1_3.py", "line": 1006,
        "source": "result = train_and_qualify(device_name=args.device)",
    },
    {
        "path": "scripts/train_go2_utility_scorer_v1_3.py", "line": 809,
        "source": "FROZEN_TRAINER.train_registered_model(",
    },
    {
        "path": "scripts/train_go2_utility_scorer_v1_2.py", "line": 3109,
        "source": '"optimizer_state_digest": structured_digest(optimizer_state),',
    },
    {
        "path": "scripts/train_go2_utility_scorer_v1_2.py", "line": 312,
        "source": "update(value)",
    },
    {
        "path": "scripts/train_go2_utility_scorer_v1_2.py", "line": 295,
        "source": "update(item[key])",
        "note": "repeated recursive frame in nested AdamW state",
    },
    {
        "path": "scripts/train_go2_utility_scorer_v1_2.py", "line": 284,
        "source": "digest.update(tensor_digest(item).encode(\"ascii\"))",
    },
    {
        "path": "scripts/train_go2_utility_scorer_v1_2.py", "line": 272,
        "source": "digest.update(value.view(torch.uint8).numpy().tobytes())",
    },
]


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
    return root / CONTRACT.SCORER_TRAINING_REPLACEMENT_QUALIFICATION_PATH


def scorer_package_path(root: Path = ROOT) -> Path:
    return root / CONTRACT.SCORER_TRAINING_REPLACEMENT_PACKAGE_PATH


def scorer_package_receipt_path(root: Path = ROOT) -> Path:
    return root / CONTRACT.SCORER_TRAINING_REPLACEMENT_PACKAGE_RECEIPT_PATH


def baseline_path(root: Path = ROOT) -> Path:
    return root / CONTRACT.SCORER_TRAINING_REPLACEMENT_BASELINE_PATH


def baseline_receipt_path(root: Path = ROOT) -> Path:
    return root / CONTRACT.SCORER_TRAINING_REPLACEMENT_BASELINE_RECEIPT_PATH


def failed_scorer_path(root: Path = ROOT) -> Path:
    return root / CONTRACT.SCORER_TRAINING_REPLACEMENT_FAILED_SCORER_PATH


def original_training_authorisation_path(root: Path = ROOT) -> Path:
    return root / CONTRACT.TRAINING_EXECUTION_AUTHORISATION_PATH


def invalid_attempt_receipt_path(root: Path = ROOT) -> Path:
    return root / CONTRACT.INVALID_TRAINING_ATTEMPT_RECEIPT_PATH


def replacement_authorisation_path(root: Path = ROOT) -> Path:
    return root / CONTRACT.SCORER_TRAINING_REPLACEMENT_AUTHORISATION_PATH


def evaluation_authorisation_path(root: Path = ROOT) -> Path:
    return root / (
        CONTRACT.SCORER_TRAINING_REPLACEMENT_EVALUATION_AUTHORISATION_PATH)


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
    return {"view": view, "index": index, "receipt": bundle.get("receipt"),
            "rows": rows,
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
        root: Path) -> None:
    """Require a fresh replacement namespace; resume and retry are forbidden."""

    model_root = (root / CONTRACT.SCORER_TRAINING_REPLACEMENT_CHECKPOINTS_ROOT
                  / name)
    attempts = sorted(path for path in model_root.glob("attempt_*")
                      if path.is_dir() and not path.is_symlink()) \
        if model_root.is_dir() and not model_root.is_symlink() else []
    candidates = FROZEN_TRAINER._checkpoint_candidates(model_root) \
        if model_root.is_dir() and not model_root.is_symlink() else []
    del use_latent, registration, training_run_digest, device, budget, training_rows
    _require(not candidates,
             f"{name} has a checkpoint; resume or retry is not authorised")
    if name == "latent":
        expected = model_root / "attempt_000"
        _require(attempts == [expected]
                 and (expected / "attempt.json").is_file()
                 and not (expected / "attempt.json").is_symlink(),
                 "latent replacement requires exactly the preserved failed "
                 "attempt_000 and no later attempt")
    else:
        _require(not attempts,
                 "no-latent baseline was already started; replacement is "
                 "not authorised")


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


def _git_output(root: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=root, text=True,
            stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise V13TrainingError(f"cannot validate replacement source: {exc}") from exc


def replacement_source_binding(root: Path = ROOT) -> dict[str, Any]:
    """Bind a clean committed repair and reject any broader source change."""

    head = _git_output(root, "rev-parse", "HEAD")
    _require(head != CONTRACT.INVALID_TRAINING_ATTEMPT_SOURCE_COMMIT,
             "replacement source repair was not committed")
    _require(not _git_output(
        root, "status", "--porcelain"),
        "replacement training requires a clean tracked source tree")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor",
         CONTRACT.INVALID_TRAINING_ATTEMPT_SOURCE_COMMIT, head],
        cwd=root, check=False, capture_output=True, text=True)
    _require(ancestor.returncode == 0,
             "failed-attempt source is not an ancestor of repaired source")
    changed = frozenset(filter(None, _git_output(
        root, "diff", "--name-only",
        f"{CONTRACT.INVALID_TRAINING_ATTEMPT_SOURCE_COMMIT}..{head}").splitlines()))
    _require(changed == REPAIR_CHANGED_PATHS,
             f"replacement source changed outside the narrow allowlist: "
             f"{sorted(changed ^ REPAIR_CHANGED_PATHS)}")
    bindings = []
    for relative in sorted(changed):
        path = root / relative
        _require(path.is_file() and not path.is_symlink(),
                 f"replacement source path changed: {relative}")
        bindings.append({
            "path": relative,
            "sha256": file_sha256(path),
            "byte_count": path.stat().st_size,
        })
    return {
        "source_commit": head,
        "source_clean": True,
        "failed_attempt_source_commit":
            CONTRACT.INVALID_TRAINING_ATTEMPT_SOURCE_COMMIT,
        "changed_paths": sorted(changed),
        "changed_source_bindings": bindings,
        "changed_source_bindings_digest": canonical_digest(bindings),
    }


def run_serializer_smoke(root: Path = ROOT) -> dict[str, Any]:
    """Run only the tiny training fixture through the failed production path."""

    command = [sys.executable, "-m", "pytest", "-q", SERIALIZER_SMOKE_NODE]
    environment = dict(os.environ)
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    completed = subprocess.run(
        command, cwd=root, env=environment, check=False,
        capture_output=True, text=True)
    _require(completed.returncode == 0,
             "scalar-safe production-path smoke failed:\n"
             + completed.stdout + completed.stderr)
    test_path = root / SERIALIZER_SMOKE_NODE.split("::", 1)[0]
    serializer_path = root / "scripts/train_go2_utility_scorer_v1_2.py"
    return {
        "non_scientific_tiny_fixture": True,
        "fresh_calibration_opened": False,
        "command": command,
        "return_code": completed.returncode,
        "stdout_sha256": hashlib.sha256(
            completed.stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(
            completed.stderr.encode("utf-8")).hexdigest(),
        "serializer_source_sha256": file_sha256(serializer_path),
        "serializer_source_byte_count": serializer_path.stat().st_size,
        "focused_test_sha256": file_sha256(test_path),
        "focused_test_byte_count": test_path.stat().st_size,
        "passed": True,
    }


def registered_data_order_plan(fit_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recompute the exact frozen CPU shuffle sequence without model execution."""

    _require(len(fit_rows) == EXPECTED_FIT_ROWS,
             "data-order plan requires the exact fit rows")
    base = [str(row["training_view_row_digest"]) for row in fit_rows]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260811)
    permutations = []
    presentations = []
    for epoch in range(1, 61):
        order = torch.randperm(EXPECTED_FIT_ROWS, generator=generator)
        permutation = order.tolist()
        permutations.append({
            "epoch": epoch,
            "permutation_tensor_digest": FROZEN_TRAINER.tensor_digest(
                order.to(torch.int64)),
        })
        presentations.append({
            "epoch": epoch,
            "ordered_training_view_row_digests_digest": canonical_digest(
                [base[index] for index in permutation]),
        })
    result = {
        "base_order": "state_id_then_candidate_index",
        "base_training_view_row_digest_sequence_digest": canonical_digest(base),
        "generator": "torch.Generator(device='cpu')",
        "seed": 20260811,
        "algorithm": "torch.randperm(1152, generator=generator)",
        "epochs": 60,
        "batch_size": 64,
        "updates_per_epoch": 18,
        "permutations": permutations,
        "row_presentations": presentations,
        "final_generator_state_digest": FROZEN_TRAINER.tensor_digest(
            generator.get_state()),
    }
    result["permutation_plan_digest"] = canonical_digest(permutations)
    result["row_presentation_plan_digest"] = canonical_digest(presentations)
    _require(
        result["base_training_view_row_digest_sequence_digest"]
        == "c862d0814efb0cbac179eedf9835d869a4dd3588e66c2df668feb44e469e1296"
        and result["permutation_plan_digest"]
        == "8e0f2c195f57fa3b883bb8830a4067f95e7965716c851be31b369d5e997c255d"
        and result["row_presentation_plan_digest"]
        == "85b1b96ad3aab1442c71a90e6afdbb3e3dc87e8115cb0f9c127953531f7efefb"
        and permutations[0]["permutation_tensor_digest"]
        == "d41a76b417fb0c2b0a9959e447a8b0a004d9793b74f6386b4e3418789184a103"
        and permutations[-1]["permutation_tensor_digest"]
        == "e71b4cb6ea9bf0854e603894457e265204cec4978256bb5e6d08a00e6026735a"
        and result["final_generator_state_digest"]
        == "f1826a6a0c7f2cde2dcd028393e1229f2a6931099a22b8c31f97b968dbc77cb2",
        "registered data-order plan changed")
    return result


def _raw_artifact_binding(path: Path, *, root: Path) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(),
             f"preserved artifact is absent or non-regular: {path}")
    return {
        "path": str(path.relative_to(root)),
        "sha256": file_sha256(path),
        "byte_count": path.stat().st_size,
    }


def _require_preserved_raw_binding(relative: str, *, root: Path) -> dict[str, Any]:
    path = ENCODER._generated_root(root) / relative
    observed = _raw_artifact_binding(path, root=root)
    expected = CONTRACT.SCORER_TRAINING_INTEGRITY_REPLACEMENT[
        "preserved_raw_bindings"][relative]
    _require({key: observed[key] for key in ("sha256", "byte_count")}
             == expected,
             f"preserved pre-replacement bytes changed: {relative}")
    return observed


def _validate_original_failed_attempt(
        *, root: Path, initialisations: Mapping[str, Mapping[str, Any]],
        data_order: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    original_auth_path = original_training_authorisation_path(root)
    original_auth = _validate_signed(
        _read_json(original_auth_path, label="original training authorisation"),
        TRAINING_AUTHORISATION_SELF_KEY, "original training authorisation")
    expected_auth = _authorisation_payload(
        binding_digest="c783a16d28d3770f0dc253633aabd4af45d543122b9dbb20190334bd0ce2e7e5",
        training_run_digest=(
            "f9d9f2d78360f5155596e6eebfacadad4aa47afb21f2b5bfcf0a5637708622b7"),
        initialisations=initialisations)
    _require(original_auth == expected_auth,
             "original one-shot training authorisation changed")
    _require_preserved_raw_binding(
        "scorer/training_execution_authorisation.json", root=root)
    _require_preserved_raw_binding(
        "scorer/initialisations/latent.pt", root=root)
    _require_preserved_raw_binding(
        "scorer/initialisations/no_latent.pt", root=root)
    attempt_binding = _require_preserved_raw_binding(
        "scorer/training/latent/attempt_000/attempt.json", root=root)
    attempt_path = root / attempt_binding["path"]
    attempt = _read_json(attempt_path, label="original latent attempt marker")
    _require(
        attempt.get("schema") == "go2_utility_scorer_training_attempt_v1"
        and attempt.get("attempt") == 0
        and attempt.get("model_name") == "latent"
        and attempt.get("training_run_digest")
        == original_auth["training_run_digest"]
        and attempt.get("initial_state_digest")
        == initialisations["latent"]["initial_state_digest"]
        and attempt.get("start_after_completed_epoch") == 0
        and attempt.get("resume_source") is None
        and attempt.get("recovery_decision")
        == "started_from_registered_initialisation",
        "original failed-attempt marker changed")
    attempt_dir = attempt_path.parent
    _require({path.name for path in attempt_dir.iterdir()} == {"attempt.json"},
             "original failed attempt published an unexpected artifact")
    original_training_root = root / CONTRACT.TRAINING_CHECKPOINTS_ROOT
    no_latent_root = original_training_root / "no_latent"
    _require(not no_latent_root.exists() and not no_latent_root.is_symlink(),
             "original no-latent baseline unexpectedly started")
    for path in (
        root / CONTRACT.QUALIFICATION_EVALUATION_AUTHORISATION_PATH,
        root / CONTRACT.QUALIFICATION_PATH,
        root / CONTRACT.SCORER_PACKAGE_PATH,
        root / CONTRACT.SCORER_PACKAGE_RECEIPT_PATH,
        root / CONTRACT.NO_LATENT_BASELINE_PATH,
        root / CONTRACT.NO_LATENT_BASELINE_RECEIPT_PATH,
        root / CONTRACT.FAILED_SCORER_PATH,
    ):
        _require(not path.exists() and not path.is_symlink(),
                 f"original failed run unexpectedly published {path}")
    receipt = _signed({
        "schema": CONTRACT.INVALID_TRAINING_ATTEMPT_SCHEMA,
        "status": CONTRACT.INVALID_TRAINING_ATTEMPT_STATUS,
        "complete": True,
        "scientific_scorer_run": False,
        "original_source_commit":
            CONTRACT.INVALID_TRAINING_ATTEMPT_SOURCE_COMMIT,
        "run_identity": original_auth["training_run_digest"],
        "binding_digest": original_auth["binding_digest"],
        "training_execution_authorisation_digest":
            original_auth[TRAINING_AUTHORISATION_SELF_KEY],
        "scorer_seed": 20260811,
        "model_initialisations": {
            name: {
                "initial_state_digest": value["initial_state_digest"],
                "path": value["path"],
                "sha256": value["sha256"],
            } for name, value in initialisations.items()
        },
        "data_order_plan": dict(data_order),
        "completed_optimizer_updates": 18,
        "completed_epoch_update_loops": 1,
        "published_checkpoint_epochs": 0,
        "checkpoint_published": False,
        "resumable_checkpoint_exists": False,
        "no_latent_baseline_started": False,
        "calibration_or_qualification_started": False,
        "scorer_metric_inspected": False,
        "predictor_checkpoint_opened": False,
        "predictor_utility_shard_opened": False,
        "development_transfer_package_opened": False,
        "exception": {
            "type": ORIGINAL_EXCEPTION_TYPE,
            "message": ORIGINAL_EXCEPTION_MESSAGE,
            "traceback_frames": copy.deepcopy(ORIGINAL_TRACEBACK_FRAMES),
            "capture": "observed console PTY traceback",
            "durable_stderr_log_path": None,
            "capture_limitation": "PTY traceback was not persisted as a file",
        },
        "attempt_marker": attempt_binding,
        "attempt_marker_payload_digest": canonical_digest(attempt),
        "preserved_not_deleted_or_overwritten": True,
        "resume_permitted": False,
    }, INVALID_ATTEMPT_SELF_KEY)
    return receipt, original_auth


def _preserved_science_bindings(
        *, root: Path, corpus: Mapping[str, Any]) -> dict[str, Any]:
    view, index = corpus["view"], corpus["index"]
    raw = {
        relative: _require_preserved_raw_binding(relative, root=root)
        for relative in (
            "training_view.json", "equivalence_receipt.json",
            "replay_overlay_manifest.json",
            "fresh_calibration/state_manifest.json",
            "fresh_calibration/corpus_receipt.json",
            "encoded_training_view/latent_index.json",
            "encoded_training_view/encoding_receipt.json",
        )
    }
    attempt_files = sorted(
        (root / CONTRACT.REPLAY_ATTEMPTS_ROOT).glob("*.json"))
    overlay_files = sorted(
        (root / CONTRACT.REPLAY_OVERLAYS_ROOT).glob("*.json"))
    expected_names = {f"{value}.json" for value in CONTRACT.FAILED_BRANCH_ALLOWLIST}
    _require(len(attempt_files) == len(overlay_files) == 18
             and {path.name for path in attempt_files} == expected_names
             and {path.name for path in overlay_files} == expected_names,
             "exact eighteen replay artifacts changed")
    marker_bindings = [_raw_artifact_binding(path, root=root)
                       for path in attempt_files]
    overlay_bindings = [_raw_artifact_binding(path, root=root)
                        for path in overlay_files]
    fit_states = sorted({str(row["state_identity_digest"])
                         for row in corpus["fit_rows"]})
    calibration_states = sorted({str(row["state_identity_digest"])
                                 for row in corpus["calibration_rows"]})
    fit_scenes = sorted({str(row["scene_id"]) for row in corpus["fit_rows"]})
    calibration_scenes = sorted({str(row["scene_id"])
                                 for row in corpus["calibration_rows"]})
    historical = view["historical_calibration_disposition"]
    disposition_key = next(key for key in historical
                           if key.endswith("disposition_digest"))
    result = {
        "oracle_v1_3_digest": view["oracle_v1_3_digest"],
        "scorer_fit_oracle_v1_3_contract_digest":
            view["scorer_fit_oracle_v1_3_contract_digest"],
        **{key: view[key] for key in SOURCE_DIGEST_KEYS},
        "training_view_digest": view["training_view_digest"],
        "latent_index_digest": index[ENCODER.LATENT_INDEX_SELF_KEY],
        "encoding_receipt_digest": corpus["receipt"][
            ENCODER.ENCODING_RECEIPT_SELF_KEY],
        "target_encoder_digest": index["target_encoder_digest"],
        "target_encoder_checkpoint_sha256":
            index["target_encoder_checkpoint_sha256"],
        "preprocess_contract_digest": index["preprocess_contract_digest"],
        "preprocessing_digest": index["preprocessing_digest"],
        "fit_state_count": len(fit_states),
        "fit_state_identity_set_digest": canonical_digest(fit_states),
        "fit_scene_set_digest": canonical_digest(fit_scenes),
        "fresh_calibration_state_count": len(calibration_states),
        "fresh_calibration_state_identity_set_digest":
            canonical_digest(calibration_states),
        "fresh_calibration_scene_set_digest": canonical_digest(calibration_scenes),
        "historical_calibration_disposition_digest": historical[disposition_key],
        "training_row_count": len(corpus["rows"]),
        "missing_label_count": view["missing_label_count"],
        "registered_replay_count": len(attempt_files),
        "registered_replay_identity_set_digest": canonical_digest(
            sorted(CONTRACT.FAILED_BRANCH_ALLOWLIST)),
        "replay_attempt_marker_bindings_digest":
            canonical_digest(marker_bindings),
        "replay_overlay_file_bindings_digest":
            canonical_digest(overlay_bindings),
        "raw_artifact_bindings": raw,
    }
    expected = CONTRACT.SCORER_TRAINING_INTEGRITY_REPLACEMENT[
        "frozen_scientific_inputs"]
    _require(
        result["oracle_v1_3_digest"] == expected["oracle_v1_3_digest"]
        and result["scorer_fit_oracle_v1_3_contract_digest"]
        == expected["scorer_fit_contract_digest"]
        and result["v2_corpus_digest"] == CONTRACT.FROZEN_CORPUS_DIGEST
        and result["equivalence_receipt_digest"]
        == expected["legacy_equivalence_receipt_digest"]
        and result["replay_overlay_manifest_digest"]
        == expected["replay_overlay_manifest_digest"]
        and result["fresh_calibration_state_manifest_digest"]
        == expected["fresh_calibration_state_manifest_digest"]
        and result["fresh_calibration_corpus_digest"]
        == expected["fresh_calibration_corpus_digest"]
        and result["training_view_digest"] == expected["training_view_digest"]
        and result["latent_index_digest"] == expected["latent_index_digest"]
        and result["encoding_receipt_digest"]
        == expected["encoding_receipt_digest"]
        and result["fit_state_count"] == 96
        and result["fit_state_identity_set_digest"]
        == "858ad55b14d0079ea11c49a1c79b2245c7adb71846493c449e7eb3cf1d16900a"
        and result["fit_scene_set_digest"]
        == "a7ef974169522a270f407de1b1b6023583816f82f76a9b8b9cc0b896bfa67373"
        and result["fresh_calibration_state_count"] == 24
        and result["fresh_calibration_state_identity_set_digest"]
        == "730e4a4835f748ad28f1fae9422c8613d8fd56a2afe0135720842c7203c04b7c"
        and result["fresh_calibration_scene_set_digest"]
        == "91fcf0d00b7c6122a9af7e2f2db6e585070390caede5950259f13bdc90480e8e"
        and result["historical_calibration_disposition_digest"]
        == "8e8b7aba9f55c62ec1fbefffafc324794df564234d348ed6a8f35e6afb3d072a"
        and result["registered_replay_identity_set_digest"]
        == "d2386c2a6d99ea4695d6afc85708d3cf99a1657489e6e6c9cd52bb91d50b56dd"
        and result["replay_attempt_marker_bindings_digest"]
        == "baf2cf718367118f6b05d30365fa93405c9e1ef139e5a9d34cb2053e9f80e2cf"
        and result["replay_overlay_file_bindings_digest"]
        == "5233bca4c560c844324f3e937469faef850d4fba4e6588ae07945a6484872b44"
        and result["training_row_count"] == 1440
        and result["missing_label_count"] == 0,
        "frozen scientific input digest inventory changed")
    return result


def _replacement_authorisation_payload(
        *, invalid_attempt: Mapping[str, Any], source: Mapping[str, Any],
        science: Mapping[str, Any], binding: Mapping[str, Any],
        binding_digest: str, training_run_digest: str,
        initialisations: Mapping[str, Mapping[str, Any]],
        data_order: Mapping[str, Any], smoke: Mapping[str, Any],
        ) -> dict[str, Any]:
    scientific_contract = {
        "architecture": binding["architecture"],
        "normalisation": binding["normalisation"],
        "utility_weights": binding["utility_weights"],
        "training": binding["training"],
        "training_execution_counts": binding["training_execution_counts"],
        "qualification_thresholds": binding["qualification_thresholds"],
        "learning_rate_schedule": binding["learning_rate_schedule"],
        "final_epoch_only": binding["final_epoch_only"],
        "epoch_selection_permitted": binding["epoch_selection_permitted"],
        "model_specific_calibration": binding["model_specific_calibration"],
        "action_input": "exact 4x10 post-slew action blocks",
        "goal_binding": "exact frozen 3-vector goal_binding_input",
        "targets": ["progress", "safety", "completion"],
    }
    return _signed({
        "schema": CONTRACT.SCORER_TRAINING_REPLACEMENT_AUTHORISATION_SCHEMA,
        "status": STATUS,
        "complete": True,
        "replacement_contract_digest":
            CONTRACT.scorer_training_integrity_replacement_digest(),
        "replacement_attempt_number": 1,
        "maximum_authorised_replacement_attempts": 1,
        "invalid_original_attempt_receipt_digest":
            invalid_attempt[INVALID_ATTEMPT_SELF_KEY],
        "invalid_original_attempt_status": invalid_attempt["status"],
        "original_exception": invalid_attempt["exception"],
        "repaired_source": dict(source),
        "serializer_smoke": dict(smoke),
        "frozen_scientific_inputs": dict(science),
        "scientific_training_contract": scientific_contract,
        "scientific_training_contract_digest":
            canonical_digest(scientific_contract),
        "binding_digest": binding_digest,
        "training_run_digest": training_run_digest,
        "registered_seed": 20260811,
        "registered_initial_model_artifacts": {
            name: {
                "path": value["path"],
                "sha256": value["sha256"],
                "initial_state_digest": value["initial_state_digest"],
            } for name, value in initialisations.items()
        },
        "registered_data_and_batch_order": dict(data_order),
        "restart_from_identical_registered_initialisation": True,
        "reuse_original_eighteen_updates": False,
        "final_epoch_selection": "final_epoch_only_no_selection",
        "authorisation_reason": (
            "the original attempt could not publish a checkpoint or expose a "
            "scientific result"
        ),
        "authorised_because_of_model_performance": False,
        "further_replacement_automatically_permitted": False,
        "later_defect_requires_new_explicit_decision": True,
        "predictor_checkpoints_opened": 0,
        "predictor_utility_shards_opened": 0,
        "final_200_state_corpus_generated": False,
    }, REPLACEMENT_AUTHORISATION_SELF_KEY)


def issue_training_integrity_replacement_authorisation(
        *, root: Path, corpus: Mapping[str, Any], binding: Mapping[str, Any],
        binding_digest: str, training_run_digest: str,
        initialisations: Mapping[str, Mapping[str, Any]],
        ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Publish the invalid lineage and sole replacement authority once."""

    source = replacement_source_binding(root)
    data_order = registered_data_order_plan(corpus["fit_rows"])
    invalid, original_auth = _validate_original_failed_attempt(
        root=root, initialisations=initialisations, data_order=data_order)
    _require(training_run_digest == original_auth["training_run_digest"]
             and binding_digest == original_auth["binding_digest"],
             "replacement run differs from the invalid run identity")
    publish_json_once(
        invalid_attempt_receipt_path(root), invalid,
        label="invalid original scorer attempt receipt")
    science = _preserved_science_bindings(root=root, corpus=corpus)
    path = replacement_authorisation_path(root)
    if path.exists() or path.is_symlink():
        authority = _validate_signed(
            _read_json(path, label="replacement authorisation"),
            REPLACEMENT_AUTHORISATION_SELF_KEY, "replacement authorisation")
        _require(authority.get("repaired_source") == source
                 and authority.get("frozen_scientific_inputs") == science
                 and authority.get("invalid_original_attempt_receipt_digest")
                 == invalid[INVALID_ATTEMPT_SELF_KEY]
                 and authority.get("training_run_digest") == training_run_digest,
                 "existing replacement authorisation changed")
        return invalid, authority
    smoke = run_serializer_smoke(root)
    authority = _replacement_authorisation_payload(
        invalid_attempt=invalid, source=source, science=science,
        binding=binding, binding_digest=binding_digest,
        training_run_digest=training_run_digest,
        initialisations=initialisations, data_order=data_order, smoke=smoke)
    publish_json_once(path, authority, label="sole scorer integrity replacement")
    return invalid, authority


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
        replacement_authorisation_digest: str,
        ) -> dict[str, Any]:
    return _signed({
        "schema": EVALUATION_AUTHORISATION_SCHEMA,
        "status": STATUS,
        "complete": True,
        "training_run_digest": training_run_digest,
        "integrity_replacement_authorisation_digest":
            replacement_authorisation_digest,
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


def _validate_preserved_workflow_inputs_for_replacement(
        *, root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any],
                                dict[str, Any]]:
    """Replay frozen producer validation without requiring its old live source."""

    _require(root.resolve() == ROOT.resolve(),
             "replacement consumption is restricted to the registered repository")
    workflow = ENCODER.WORKFLOW
    out_root = ENCODER._generated_root(root)
    authority, correction = workflow.load_selector_integrity_replacement_authority(
        root=None, out_root=out_root)
    corpus = workflow.load_v2_corpus(v2_root=workflow.V2_ROOT)
    plan = workflow.load_replay_plan(
        out_root=out_root, v2_root=workflow.V2_ROOT, authority=authority)
    equivalence = workflow._read_regular_json(
        out_root / "equivalence_receipt.json")
    workflow.validate_equivalence_receipt(
        equivalence, authority, corpus["rows"])
    overlay_manifest = workflow._read_regular_json(
        out_root / "replay_overlay_manifest.json")
    workflow._validate_self_digest(
        overlay_manifest, workflow.REPLAY_OVERLAY_MANIFEST_SELF_KEY)
    pool, exclusion = workflow.B.scene_pool("scorer_fit")
    attempt = workflow._read_regular_json(
        out_root / "fresh_calibration/selection_attempt.json")
    workflow.validate_fresh_selection_attempt(
        attempt, authority=authority, correction=correction,
        v2_state_manifest=corpus["state_manifest"], plan=plan,
        equivalence=equivalence, overlay_manifest=overlay_manifest,
        pool=pool, exclusion=exclusion)
    terminal = workflow._read_regular_json(
        out_root / "fresh_calibration/selection_terminal.json")
    states = workflow.validate_fresh_selection_terminal(
        terminal, attempt=attempt, correction=correction, out_root=out_root)
    manifest = workflow._read_regular_json(
        out_root / "fresh_calibration/state_manifest.json")
    workflow.validate_fresh_calibration_manifest(
        manifest, authority=authority,
        v2_state_manifest=corpus["state_manifest"],
        selector_integrity_replacement_authority=correction)
    expected_manifest = workflow.build_fresh_calibration_manifest(
        authority=authority, v2_state_manifest=corpus["state_manifest"],
        states=states, exclusion_binding=attempt["manifest_exclusion_binding"],
        selector_integrity_replacement_authority=correction)
    _require(manifest == expected_manifest,
             "fresh calibration manifest changed after target encoding")
    return authority, correction, corpus, manifest


def _materialise_preserved_training_view_for_replacement(
        *, root: Path, authority: Mapping[str, Any],
        manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Materialise the immutable view while bypassing only live-source equality."""

    workflow = ENCODER.WORKFLOW
    path = root / CONTRACT.TRAINING_VIEW_PATH
    reference = workflow._read_regular_json(path)
    workflow._validate_training_view_shape(reference)
    _require(
        reference.get("authority_digest")
        == authority[workflow.AUTHORITY_SELF_KEY]
        and reference.get("oracle_v1_3_digest")
        == authority["oracle_v1_3_digest"]
        and reference.get("scorer_fit_oracle_v1_3_contract_digest")
        == authority["scorer_fit_oracle_v1_3_contract_digest"]
        and reference.get("fresh_calibration_state_manifest_digest")
        == manifest[workflow.FRESH_STATE_MANIFEST_SELF_KEY],
        "training view is bound to another preserved v1.3 authority")
    rows = []
    for reference_row in reference["rows"]:
        workflow._validate_self_digest(
            reference_row, "training_view_row_digest")
        input_row, _ = workflow._resolve_bound_input(reference_row["input"])
        label_source, _ = workflow._resolve_bound_input(
            reference_row["label_source"])
        labels = (
            workflow._label_projection(label_source["labels"])
            if reference_row["source_kind"] == workflow.SOURCE_KIND_REPLAY
            else workflow._label_projection(label_source))
        _require(labels == reference_row["label_projection"],
                 "materialised label differs from the frozen projection")
        frame_root = root / reference_row["frame_root"]
        rows.append({
            **{key: reference_row[key] for key in (
                "role", "source_kind", "state_id", "state_identity_digest",
                "scene_id", "family", "stratum", "candidate_index",
                "branch_identity_digest", "training_view_row_digest")},
            "frame_root": reference_row["frame_root"],
            "context_frames": workflow._normalise_frame_records(
                input_row["context_frames"], frame_root),
            "horizon_frames": workflow._normalise_frame_records(
                input_row["horizon_frames"], frame_root),
            "action_blocks": input_row["action_blocks"],
            "action_context_blocks": input_row["action_context_blocks"],
            "previous_applied_command": input_row["previous_applied_command"],
            "goal_binding_input": input_row["goal_binding_input"],
            "proprio": input_row["proprio"],
            "control": input_row["control"],
            "masks": input_row["masks"],
            "timing": input_row["timing"],
            **labels,
        })
    result = copy.deepcopy(reference)
    result["reference_rows"] = result.pop("rows")
    result["rows"] = rows
    return ENCODER.validate_training_view_structure(result)


def load_preserved_encoded_training_view_for_replacement(
        *, root: Path = ROOT,
        verify_encoder_checkpoint: bool = False) -> dict[str, Any]:
    """Validate exact frozen bytes without reopening corpus/encoding execution."""

    ENCODER._require_registered_generated_root(root)
    authority, _correction, _corpus, manifest = (
        _validate_preserved_workflow_inputs_for_replacement(root=root))
    view = _materialise_preserved_training_view_for_replacement(
        root=root, authority=authority, manifest=manifest)
    index_path = ENCODER.latent_index_path(root)
    receipt_path = ENCODER.encoding_receipt_path(root)
    _require(index_path.is_file() and not index_path.is_symlink(),
             "frozen latent index is absent")
    _require(receipt_path.is_file() and not receipt_path.is_symlink(),
             "frozen encoding receipt is absent")
    index = ENCODER.validate_latent_index(
        json.loads(index_path.read_text()), view, root=root,
        verify_encoder_checkpoint=verify_encoder_checkpoint)
    receipt = ENCODER._validate_signed(
        json.loads(receipt_path.read_text()), ENCODER.ENCODING_RECEIPT_SELF_KEY,
        "frozen v1.3 encoding receipt")
    _require(receipt == ENCODER._receipt_payload(index, root=root),
             "frozen encoding receipt differs from exact index bytes")
    return {"view": view, "index": index, "receipt": receipt,
            "encoded_root": ENCODER.encoded_root(root)}


def load_and_validate_training_terminal_for_consumption(
        *, root: Path = ROOT, require_qualified: bool | None = None,
        verify_encoder_checkpoint: bool = False,
        ) -> dict[str, Any]:
    encoded = load_preserved_encoded_training_view_for_replacement(
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
    replacement_auth = _validate_signed(
        _read_json(replacement_authorisation_path(root),
                   label="replacement authorisation"),
        REPLACEMENT_AUTHORISATION_SELF_KEY, "replacement authorisation")
    evaluation_auth = _validate_signed(
        _read_json(evaluation_authorisation_path(root),
                   label="qualification evaluation authorisation"),
        EVALUATION_AUTHORISATION_SELF_KEY,
        "qualification evaluation authorisation")
    binding = _binding_payload(corpus)
    binding["encoding_receipt_digest"] = corpus["receipt"][
        ENCODER.ENCODING_RECEIPT_SELF_KEY]
    data_order = registered_data_order_plan(corpus["fit_rows"])
    invalid, _original_auth = _validate_original_failed_attempt(
        root=root, initialisations=report.get("initialisations", {}),
        data_order=data_order)
    science = _preserved_science_bindings(root=root, corpus=corpus)
    expected_replacement_auth = _replacement_authorisation_payload(
        invalid_attempt=invalid, source=replacement_source_binding(root),
        science=science, binding=binding,
        binding_digest=str(report.get("binding_digest")),
        training_run_digest=str(report.get("training_run_digest")),
        initialisations=report.get("initialisations", {}),
        data_order=data_order, smoke=replacement_auth.get("serializer_smoke", {}))
    expected_evaluation_auth = _evaluation_authorisation_payload(
        training_run_digest=str(report.get("training_run_digest")),
        final_state_digests=report.get("final_state_digests", {}),
        replacement_authorisation_digest=replacement_auth[
            REPLACEMENT_AUTHORISATION_SELF_KEY])
    receipts = report.get("training_receipts")
    _require(replacement_auth == expected_replacement_auth
             and evaluation_auth == expected_evaluation_auth
             and report.get("integrity_replacement_authorisation_digest")
             == replacement_auth[REPLACEMENT_AUTHORISATION_SELF_KEY]
             and report.get("invalid_original_attempt_receipt_digest")
             == invalid[INVALID_ATTEMPT_SELF_KEY]
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
             "replacement authorisations differ from terminal")

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
    encoded = load_preserved_encoded_training_view_for_replacement(
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
    invalid_attempt, replacement_auth = (
        issue_training_integrity_replacement_authorisation(
            root=root, corpus=corpus, binding=binding,
            binding_digest=binding_digest,
            training_run_digest=training_run_digest,
            initialisations=initialisations))

    for name, use_latent in (("latent", True), ("no_latent", False)):
        _preflight_registered_training(
            name, use_latent=use_latent,
            registration=initialisations[name],
            training_run_digest=training_run_digest, device=device,
            budget=budget, training_rows=len(fit_rows), root=root)

    fit_features = FROZEN_TRAINER.features(
        fit_rows, corpus["horizon"], device)
    calibration_features = FROZEN_TRAINER.features(
        calibration_rows, corpus["horizon"], device)

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
        expected_attempt = "attempt_001" if name == "latent" else "attempt_000"
        _require(Path(str(training_receipts[name]["final_checkpoint"])).parent.name
                 == expected_attempt
                 and training_receipts[name].get("resume_source") is None
                 and training_receipts[name].get("recovery_decision")
                 == "started_from_registered_initialisation",
                 f"{name} did not start the sole replacement from frozen init")
        models[name].load_state_dict(packages[name], strict=True)
        models[name].to(device)
    final_state_digests = {
        name: FROZEN_TRAINER.state_dict_digest(state)
        for name, state in packages.items()
    }
    evaluation_auth = _evaluation_authorisation_payload(
        training_run_digest=training_run_digest,
        final_state_digests=final_state_digests,
        replacement_authorisation_digest=replacement_auth[
            REPLACEMENT_AUTHORISATION_SELF_KEY])
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
        "invalid_original_attempt_receipt_digest":
            invalid_attempt[INVALID_ATTEMPT_SELF_KEY],
        "integrity_replacement_authorisation_digest":
            replacement_auth[REPLACEMENT_AUTHORISATION_SELF_KEY],
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
        "invalid_original_attempt_receipt_digest":
            invalid_attempt[INVALID_ATTEMPT_SELF_KEY],
        "integrity_replacement_authorisation_digest":
            replacement_auth[REPLACEMENT_AUTHORISATION_SELF_KEY],
        "replacement_attempt_number": 1,
        "maximum_authorised_replacement_attempts": 1,
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
