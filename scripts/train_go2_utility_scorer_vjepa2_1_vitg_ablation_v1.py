#!/usr/bin/env python3
"""Train and evaluate the one exploratory V-JEPA 2.1 ViT-g scorer.

``EXPLORATORY_ENCODER_SCALE_ABLATION`` is deliberately narrower than the
qualified-scorer workflow.  This module consumes only the preserved
oracle-v1.3 scorer view and its separately encoded true ViT-g trajectories.
It trains one latent scorer, reuses the already frozen encoder-independent
no-latent result, evaluates the existing 24 development calibration states
once, and stops.  It has no predictor, planner, or final-corpus route.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import encode_go2_scorer_fit_vjepa2_1_vitg_ablation_v1 as ENCODER  # noqa: E402
from scripts import train_go2_utility_scorer_v1_2 as FROZEN  # noqa: E402
from scripts import train_go2_utility_scorer_v1_3 as V13  # noqa: E402


STATUS = "EXPLORATORY_ENCODER_SCALE_ABLATION"
COMPARISON_CLASSIFICATION = "SCALE_ONLY"
SCHEMA = "go2_vjepa2_1_vitg_frozen_encoder_scorer_ablation_v1"
CONTRACT_SCHEMA = "go2_vjepa2_1_vitg_exploratory_scorer_contract_v1"
INITIALISATION_SCHEMA = "go2_vjepa2_1_vitg_scorer_initialisation_v1"
ATTEMPT_SCHEMA = "go2_vjepa2_1_vitg_scorer_training_attempt_v1"
CHECKPOINT_SCHEMA = "go2_vjepa2_1_vitg_scorer_final_checkpoint_v1"
EVALUATION_AUTH_SCHEMA = "go2_vjepa2_1_vitg_scorer_evaluation_authorisation_v1"
TECHNICAL_FAILURE_SCHEMA = "go2_vjepa2_1_vitg_scorer_technical_failure_v1"

GENERATED_RELATIVE = Path(".generated/go2_scorer_fit_vjepa2_1_vitg_ablation_v1")
SCORER_SEED = 20260811
TOKENS = 768
TOKEN_DIM = 1_408
HORIZONS = 4
HIDDEN_DIM = 512
ACTION_DIM = 40
GOAL_DIM = 3
FIT_STATES = 96
FIT_ROWS = 1_152
CALIBRATION_STATES = 24
CALIBRATION_ROWS = 288
TOTAL_ROWS = 1_440
UPDATES_PER_EPOCH = 18
EPOCHS = 60
TOTAL_UPDATES = 1_080
PRESENTATIONS = 69_120

ORIGINAL_PRIMARY = {
    "safety_auc": 0.7043234199,
    "latent_over_baseline_pairwise_gain": 0.0317880795,
}
PRIMARY_THRESHOLDS = {
    "safety_auc_min": 0.75,
    "latent_over_baseline_pairwise_gain_min": 0.05,
}
FROZEN_V13_TERMINAL_DIGEST = (
    "441f52d4199ba152825f30a9f5422b80537f68b9f7a3633f4e01610f964de419"
)
FROZEN_BASELINE_RECEIPT_DIGEST = (
    "454bc81c3077d62cac661a4ccac7212b3eb3860eda3177f9b8879f27632abc25"
)
FROZEN_BASELINE_SHA256 = (
    "cfd07d2ad739ef884f3d8ebc3faa01a0b807ef6f19049874eb7fc6ecc9c418ca"
)
FROZEN_BASELINE_STATE_DIGEST = (
    "33e7bcffbfab16371fb8e7e233490c33c442336edac823c19733214fa87d91d1"
)


class ViTGScorerError(RuntimeError):
    """The frozen exploratory contract or its one execution changed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ViTGScorerError(message)


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


def _is_digest(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _signed(value: Mapping[str, Any], self_key: str) -> dict[str, Any]:
    result = dict(value)
    _require(self_key not in result, f"{self_key} was already present")
    result[self_key] = canonical_digest(result)
    return result


def _validate_signed(value: Mapping[str, Any], self_key: str,
                     label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} is not an object")
    result = dict(value)
    recorded = result.pop(self_key, None)
    _require(_is_digest(recorded), f"{label} self digest is malformed")
    _require(recorded == canonical_digest(result),
             f"{label} self digest does not verify")
    result[self_key] = recorded
    return result


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(),
             f"{label} is absent or not a regular file")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ViTGScorerError(f"cannot read {label}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} is not an object")
    return value


def _publish_json_once(path: Path, value: Mapping[str, Any], *, label: str) -> None:
    V13.publish_json_once(path, value, label=label)


def ablation_root(root: Path = ROOT) -> Path:
    return root / GENERATED_RELATIVE


def scorer_root(root: Path = ROOT) -> Path:
    return ablation_root(root) / "scorer"


def contract_path(root: Path = ROOT) -> Path:
    return scorer_root(root) / "exploratory_scorer_contract.json"


def initialisation_path(root: Path = ROOT) -> Path:
    return scorer_root(root) / "initialisation.pt"


def attempt_root(root: Path = ROOT) -> Path:
    return scorer_root(root) / "training/attempt_000"


def final_checkpoint_path(root: Path = ROOT) -> Path:
    return attempt_root(root) / "final_epoch_060.pt"


def evaluation_authorisation_path(root: Path = ROOT) -> Path:
    return scorer_root(root) / "evaluation_authorisation.json"


def terminal_path(root: Path = ROOT) -> Path:
    return scorer_root(root) / "exploratory_result.json"


def technical_failure_path(root: Path = ROOT) -> Path:
    return scorer_root(root) / "technical_failure.json"


class ViTGUtilityScorer(nn.Module):
    """The frozen scorer with only its latent input width changed to 1,408."""

    def __init__(self, *, token_dim: int = TOKEN_DIM,
                 hidden: int = HIDDEN_DIM) -> None:
        super().__init__()
        _require(token_dim == TOKEN_DIM, "ViT-g latent width is not 1,408")
        _require(hidden == HIDDEN_DIM, "frozen scorer hidden width changed")
        self.token_dim = token_dim
        self.per_horizon = nn.Sequential(
            nn.Linear(token_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.attention = nn.Linear(hidden, 1)
        self.context = nn.Sequential(
            nn.Linear(ACTION_DIM + GOAL_DIM, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden))
        self.fuse = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.SiLU())
        self.progress = nn.Linear(hidden, 1)
        self.safety = nn.Linear(hidden, 1)
        self.completion = nn.Linear(hidden, 1)

    def forward(self, latent: torch.Tensor,
                action_goal: torch.Tensor) -> tuple[torch.Tensor, ...]:
        per_horizon = self.per_horizon(latent)
        attention = torch.softmax(self.attention(per_horizon), dim=1)
        visual = (per_horizon * attention).sum(dim=1)
        context = self.context(action_goal)
        fused = self.fuse(torch.cat((visual, context), dim=-1))
        return (self.progress(fused).squeeze(-1),
                self.safety(fused).squeeze(-1),
                self.completion(fused).squeeze(-1))


def dimension_aware_projection_key(*, source_state_digest: str,
                                   token_dim: int = TOKEN_DIM) -> dict[str, Any]:
    payload = {
        "schema": "go2_vitg_dimension_aware_parameter_key_v1",
        "registered_seed": SCORER_SEED,
        "parameter": "per_horizon.0.weight",
        "shape": [HIDDEN_DIM, token_dim],
        "dtype": "torch.float32",
        "source_vitl_initial_state_digest": source_state_digest,
        "initialiser": "torch.nn.init.kaiming_uniform_a_sqrt5_cpu_generator",
    }
    return {**payload, "key_digest": canonical_digest(payload)}


def _projection_from_key(key: Mapping[str, Any]) -> torch.Tensor:
    digest = str(key["key_digest"])
    seed = int(digest[:16], 16) % (2 ** 63 - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    weight = torch.empty((HIDDEN_DIM, TOKEN_DIM), dtype=torch.float32)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5), generator=generator)
    return weight


def build_dimension_aware_initial_state(
        source_state: Mapping[str, torch.Tensor],
        ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Copy every compatible ViT-L tensor; derive only the wider projection."""

    source_digest = FROZEN.state_dict_digest(source_state)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SCORER_SEED)
        model = ViTGUtilityScorer()
    target = FROZEN._cpu_state(model)
    changed: list[str] = []
    copied: list[str] = []
    for name, tensor in target.items():
        source = source_state.get(name)
        if (isinstance(source, torch.Tensor)
                and source.shape == tensor.shape and source.dtype == tensor.dtype):
            target[name] = source.detach().cpu().clone()
            copied.append(name)
            continue
        _require(name == "per_horizon.0.weight",
                 f"unexpected shape-incompatible scorer parameter: {name}")
        key = dimension_aware_projection_key(
            source_state_digest=source_digest)
        target[name] = _projection_from_key(key)
        changed.append(name)
    _require(changed == ["per_horizon.0.weight"],
             "the ViT-g scorer changed more than its input projection")
    model.load_state_dict(target, strict=True)
    old_parameter_count = sum(value.numel() for value in source_state.values())
    new_parameter_count = sum(value.numel() for value in target.values())
    receipt = {
        "registered_seed": SCORER_SEED,
        "source_vitl_initial_state_digest": source_digest,
        "initial_state_digest": FROZEN.state_dict_digest(target),
        "copied_shape_compatible_parameters": copied,
        "copied_parameter_count": sum(target[name].numel() for name in copied),
        "dimension_changed_parameters": changed,
        "dimension_aware_projection_key": key,
        "vitl_parameter_count": old_parameter_count,
        "vitg_parameter_count": new_parameter_count,
        "parameter_count_increase": new_parameter_count - old_parameter_count,
    }
    _require(receipt["parameter_count_increase"]
             == HIDDEN_DIM * (TOKEN_DIM - FROZEN.TOKEN_DIM),
             "unexpected parameter-count change")
    return target, receipt


def _load_vitl_initialisation(path: Path, *, expected_sha256: str,
                              expected_state_digest: str
                              ) -> Mapping[str, torch.Tensor]:
    _require(path.is_file() and not path.is_symlink(),
             "frozen ViT-L initialisation is absent")
    _require(file_sha256(path) == expected_sha256,
             "frozen ViT-L initialisation bytes changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _require(isinstance(payload, Mapping)
             and payload.get("schema")
             == "go2_scorer_registered_initialisation_v1"
             and payload.get("model_name") == "latent"
             and payload.get("use_latent") is True
             and payload.get("registered_seed") == SCORER_SEED,
             "frozen ViT-L initialisation metadata changed")
    state = payload.get("model_state_dict")
    _require(isinstance(state, Mapping)
             and FROZEN.state_dict_digest(state) == expected_state_digest,
             "frozen ViT-L initial state changed")
    return state


def issue_initialisation(*, source_path: Path, source_sha256: str,
                         source_state_digest: str,
                         root: Path = ROOT) -> dict[str, Any]:
    state = _load_vitl_initialisation(
        source_path, expected_sha256=source_sha256,
        expected_state_digest=source_state_digest)
    initial_state, receipt = build_dimension_aware_initial_state(state)
    payload = {
        "schema": INITIALISATION_SCHEMA,
        "status": STATUS,
        "registered_seed": SCORER_SEED,
        "source_vitl_initialisation_path": str(source_path),
        "source_vitl_initialisation_sha256": source_sha256,
        **receipt,
        "model_state_dict": initial_state,
    }
    path = initialisation_path(root)
    if path.exists() or path.is_symlink():
        _require(path.is_file() and not path.is_symlink(),
                 "ViT-g initialisation path is not regular")
        installed = torch.load(path, map_location="cpu", weights_only=False)
        _require(isinstance(installed, Mapping)
                 and installed.get("schema") == INITIALISATION_SCHEMA
                 and installed.get("initial_state_digest")
                 == receipt["initial_state_digest"]
                 and FROZEN.state_dict_digest(installed["model_state_dict"])
                 == receipt["initial_state_digest"],
                 "installed ViT-g initialisation differs")
    else:
        FROZEN.atomic_torch_save(payload, path)
    return {
        **receipt,
        "path": str(path),
        "sha256": file_sha256(path),
        "byte_count": path.stat().st_size,
    }


class ViTGHorizonShardStore:
    """Validated, lazy FP16 row shards with shape [4, 768, 1,408]."""

    def __init__(self, records: Sequence[Mapping[str, Any]],
                 encoded_root: Path) -> None:
        self.records = list(records)
        self.encoded_root = encoded_root
        self.shape = (len(self.records), HORIZONS, TOKENS, TOKEN_DIM)

    def __getitem__(self, item: Any) -> np.ndarray:
        scalar = np.isscalar(item)
        if isinstance(item, slice):
            positions = list(range(*item.indices(len(self.records))))
        elif scalar:
            positions = [int(item)]
        else:
            positions = [int(value) for value in np.asarray(item).reshape(-1)]
        arrays = []
        for position in positions:
            record = self.records[position]
            relative = Path(str(record["path"]))
            _require(not relative.is_absolute()
                     and ".." not in relative.parts
                     and not any(part == "sealed" or part.startswith("sealed_")
                                 for part in relative.parts),
                     "latent record path escapes its validated namespace")
            path = self.encoded_root / relative
            arrays.append(np.memmap(
                path, mode="r", dtype=np.float16,
                shape=(HORIZONS, TOKENS, TOKEN_DIM)))
        result = np.stack(arrays, axis=0)
        return result[0] if scalar else result


def _row_role(row: Mapping[str, Any]) -> str:
    role = row.get("role", row.get("split_role"))
    if "role" in row and "split_role" in row:
        _require(row["role"] == row["split_role"],
                 "training row role aliases disagree")
    _require(role in {"fit", "calibration"}, "training row role is invalid")
    return str(role)


def corpus_from_encoded_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    view, index = bundle["view"], bundle["index"]
    records = index.get("horizon_records")
    _require(isinstance(records, list) and len(records) == TOTAL_ROWS,
             "ViT-g latent index does not contain 1,440 rows")
    positions = {record["training_view_row_digest"]: position
                 for position, record in enumerate(records)}
    _require(len(positions) == TOTAL_ROWS,
             "ViT-g latent record identities are duplicated")
    rows: list[dict[str, Any]] = []
    for source in view["rows"]:
        digest = source["training_view_row_digest"]
        _require(digest in positions, "training row has no ViT-g latent")
        row = dict(source)
        projection = source.get("label_projection")
        if isinstance(projection, Mapping):
            for key in ("progress", "safety", "completion", "utility"):
                row[key] = projection[key]
        row["split_role"] = _row_role(source)
        row["_latent_index"] = positions[digest]
        rows.append(row)
    rows.sort(key=lambda row: (
        row["split_role"] != "fit", str(row["state_id"]),
        int(row["candidate_index"])))
    fit = [row for row in rows if row["split_role"] == "fit"]
    calibration = [row for row in rows
                   if row["split_role"] == "calibration"]
    _require(len(fit) == FIT_ROWS
             and len({row["state_id"] for row in fit}) == FIT_STATES,
             "ViT-g fit corpus is not 96 states / 1,152 rows")
    _require(len(calibration) == CALIBRATION_ROWS
             and len({row["state_id"] for row in calibration})
             == CALIBRATION_STATES,
             "ViT-g calibration corpus is not 24 states / 288 rows")
    _require(not ({row["scene_id"] for row in fit}
                  & {row["scene_id"] for row in calibration}),
             "fit/calibration scenes overlap")
    return {
        "view": view,
        "index": index,
        "receipt": bundle["receipt"],
        "resource": bundle.get("resource"),
        "rows": rows,
        "fit_rows": fit,
        "calibration_rows": calibration,
        "horizon": ViTGHorizonShardStore(
            records, Path(bundle["encoded_root"])),
    }


def materialise_features(rows: list[dict[str, Any]],
                         horizon: ViTGHorizonShardStore,
                         device: torch.device, *, latent_chunk: int = 8
                         ) -> tuple[torch.Tensor, torch.Tensor,
                                    dict[str, torch.Tensor]]:
    """Apply the unchanged spatial mean and action/goal input contract."""

    positions = np.asarray([row["_latent_index"] for row in rows],
                           dtype=np.int64)
    latent_mean = np.empty((len(rows), HORIZONS, TOKEN_DIM), dtype=np.float32)
    for start in range(0, len(rows), latent_chunk):
        selected = np.asarray(
            horizon[positions[start:start + latent_chunk]], dtype=np.float32)
        _require(selected.shape[1:] == (HORIZONS, TOKENS, TOKEN_DIM),
                 "ViT-g latent shard shape changed")
        latent_mean[start:start + len(selected)] = selected.mean(
            axis=2, dtype=np.float32)
    action = np.empty((len(rows), ACTION_DIM), dtype=np.float32)
    for index, row in enumerate(rows):
        flattened = [value for block in row["action_blocks"] for value in block]
        _require(len(flattened) == ACTION_DIM,
                 "candidate action stopped matching the frozen 40-D input")
        action[index] = np.asarray(flattened, dtype=np.float32)
    goal = np.asarray([row["goal_binding_input"] for row in rows],
                      dtype=np.float32)
    _require(goal.shape == (len(rows), GOAL_DIM),
             "goal binding stopped matching the frozen 3-D input")
    action_goal = np.concatenate((action, goal), axis=-1)
    targets = {
        key: torch.tensor([row[key] for row in rows], dtype=torch.float32,
                          device=device)
        for key in ("progress", "safety", "completion")
    }
    return (torch.from_numpy(latent_mean).to(device),
            torch.from_numpy(action_goal).to(device), targets)


def _resolve_recorded_path(raw: Any, *, root: Path) -> Path:
    _require(isinstance(raw, str) and raw, "recorded path is absent")
    path = Path(raw)
    return path if path.is_absolute() else root / path


def validate_reused_no_latent_baseline(
        corpus: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    """Validate the old baseline and metrics without training or forwarding it."""

    terminal = _validate_signed(
        _read_json(V13.qualification_path(root), label="ViT-L scorer terminal"),
        V13.QUALIFICATION_SELF_KEY, "ViT-L scorer terminal")
    _require(terminal[V13.QUALIFICATION_SELF_KEY]
             == FROZEN_V13_TERMINAL_DIGEST,
             "completed ViT-L scorer terminal changed")
    _require(terminal.get("terminal_kind") == "QUALIFICATION_FAILURE"
             and terminal.get("qualified") is False
             and terminal.get("scorer_package_sha256") is None,
             "completed ViT-L failure was altered or reinterpreted")
    vitl_metrics = terminal["results"]["latent"]["calibration"]
    baseline_metrics = terminal["results"]["no_latent"]["calibration"]
    old_auc = float(vitl_metrics["safety"]["auc_any_hazard"])
    old_gain = float(terminal["baseline_dominance_pairwise"])
    _require(abs(old_auc - ORIGINAL_PRIMARY["safety_auc"]) <= 5e-11
             and abs(old_gain
                     - ORIGINAL_PRIMARY["latent_over_baseline_pairwise_gain"])
             <= 5e-11,
             "completed ViT-L primary result changed")

    receipt = _validate_signed(
        _read_json(V13.baseline_receipt_path(root),
                   label="frozen no-latent baseline receipt"),
        V13.BASELINE_RECEIPT_SELF_KEY, "frozen no-latent baseline receipt")
    _require(receipt[V13.BASELINE_RECEIPT_SELF_KEY]
             == FROZEN_BASELINE_RECEIPT_DIGEST
             and receipt.get("sha256") == FROZEN_BASELINE_SHA256
             and receipt.get("final_state_digest")
             == FROZEN_BASELINE_STATE_DIGEST,
             "frozen no-latent baseline receipt changed")
    checkpoint = _resolve_recorded_path(receipt["path"], root=root)
    _require(checkpoint.is_file() and not checkpoint.is_symlink()
             and file_sha256(checkpoint) == FROZEN_BASELINE_SHA256,
             "frozen no-latent baseline checkpoint bytes changed")
    artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = artifact.get("model_state_dict")
    _require(isinstance(state, Mapping)
             and FROZEN.state_dict_digest(state)
             == FROZEN_BASELINE_STATE_DIGEST,
             "frozen no-latent baseline state changed")
    binding = artifact.get("bindings")
    _require(isinstance(binding, Mapping)
             and binding.get("training_view_digest")
             == corpus["index"]["training_view_digest"]
             and binding.get("architecture", {}).get("action_dim") == ACTION_DIM
             and binding.get("architecture", {}).get("goal_dim") == GOAL_DIM,
             "no-latent baseline inputs differ from the ViT-g study")
    source_initialisation = terminal["initialisations"]["latent"]
    initial_path = _resolve_recorded_path(
        source_initialisation["path"], root=root)
    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": FROZEN_BASELINE_SHA256,
        "state_digest": FROZEN_BASELINE_STATE_DIGEST,
        "receipt_digest": FROZEN_BASELINE_RECEIPT_DIGEST,
        "metrics": baseline_metrics,
        "per_family_metrics": terminal["results"]["no_latent"][
            "per_family_calibration"],
        "vitl_metrics": vitl_metrics,
        "vitl_per_family_metrics": terminal["results"]["latent"][
            "per_family_calibration"],
        "vitl_terminal_digest": terminal[V13.QUALIFICATION_SELF_KEY],
        "vitl_pairwise_gain": old_gain,
        "source_vitl_initialisation_path": initial_path,
        "source_vitl_initialisation_sha256": source_initialisation["sha256"],
        "source_vitl_initial_state_digest": source_initialisation[
            "initial_state_digest"],
        "raw_predictions_persisted": False,
        "reuse_mode": (
            "frozen checkpoint identity plus immutable existing aggregate and "
            "per-family calibration results; baseline is not retrained or "
            "re-evaluated"
        ),
    }


def _git_output(root: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=root, text=True,
            stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ViTGScorerError(f"cannot bind scorer source: {exc}") from exc


def frozen_budget() -> dict[str, Any]:
    budget = V13.frozen_training_budget()
    _require(int(budget["seed"]) == SCORER_SEED
             and int(budget["epochs"]) == EPOCHS
             and int(budget["batch"]) == 64
             and math.ceil(FIT_ROWS / int(budget["batch"]))
             == UPDATES_PER_EPOCH,
             "frozen scorer budget changed")
    return budget


def build_exploratory_contract(
        corpus: Mapping[str, Any], baseline: Mapping[str, Any],
        initialisation: Mapping[str, Any], *, source_commit: str,
        ) -> dict[str, Any]:
    index, receipt, view = corpus["index"], corpus["receipt"], corpus["view"]
    resource = corpus.get("resource")
    _require(isinstance(resource, Mapping),
             "validated ViT-g resource receipt is absent")
    encoder_contract = resource.get("encoder_contract")
    _require(isinstance(encoder_contract, Mapping)
             and resource.get("encoder_contract_digest")
             == index.get("encoder_contract_digest"),
             "ViT-g encoder contract differs from the latent index")
    _require(index.get("tokens") == TOKENS
             and index.get("token_dim") == TOKEN_DIM
             and index.get("horizons") == HORIZONS
             and index.get("row_count") == TOTAL_ROWS,
             "ViT-g latent index shape/cardinality changed")
    index_self_key = ENCODER.LATENT_INDEX_SELF_KEY
    receipt_self_key = ENCODER.ENCODING_RECEIPT_SELF_KEY
    data_order = V13.registered_data_order_plan(corpus["fit_rows"])
    architecture = {
        "trajectory_horizons": HORIZONS,
        "dense_tokens_per_horizon": TOKENS,
        "spatial_aggregation": FROZEN.NORMALISATION["spatial_aggregation"],
        "vitl_latent_width": FROZEN.TOKEN_DIM,
        "vitg_latent_width": TOKEN_DIM,
        "hidden_dim": HIDDEN_DIM,
        "action_dim": ACTION_DIM,
        "goal_dim": GOAL_DIM,
        "temporal_aggregation": "learned softmax attention over four horizons",
        "heads": ["progress", "safety", "completion"],
        "only_shape_change": "per_horizon.0.weight",
        "vitl_parameter_count": initialisation["vitl_parameter_count"],
        "vitg_parameter_count": initialisation["vitg_parameter_count"],
        "parameter_count_increase": initialisation["parameter_count_increase"],
    }
    payload = {
        "schema": CONTRACT_SCHEMA,
        "status": STATUS,
        "complete": True,
        "comparison_classification": COMPARISON_CLASSIFICATION,
        "source_commit": source_commit,
        "training_view_digest": index["training_view_digest"],
        "oracle_v1_3_digest": index["oracle_v1_3_digest"],
        "scorer_fit_oracle_v1_3_contract_digest": index[
            "scorer_fit_oracle_v1_3_contract_digest"],
        "latent_index_digest": index[index_self_key],
        "encoding_receipt_digest": receipt[receipt_self_key],
        "target_encoder_digest": index["encoder_contract_digest"],
        "target_encoder_checkpoint_sha256": index[
            "target_encoder_checkpoint_sha256"],
        "encoder_source_commit": encoder_contract["source_repository_commit"],
        "selected_checkpoint_key": resource["checkpoint_state_key_opened"],
        "preprocess_contract_digest": canonical_digest(
            encoder_contract["preprocessing"]),
        "preprocessing_digest": canonical_digest({
            "preprocessing": encoder_contract["preprocessing"],
            "output_post_normalisation": encoder_contract["output"][
                "post_normalisation"],
        }),
        "encoder_compute_dtype": index["encoder_compute_dtype"],
        "latent_storage_dtype": index["latent_storage_dtype"],
        "architecture": architecture,
        "normalisation": dict(FROZEN.NORMALISATION),
        "utility_weights": dict(FROZEN.WEIGHTS),
        "training": frozen_budget(),
        "training_execution": {
            "fit_states": FIT_STATES,
            "fit_rows": FIT_ROWS,
            "epochs": EPOCHS,
            "updates_per_epoch": UPDATES_PER_EPOCH,
            "updates": TOTAL_UPDATES,
            "presentations": PRESENTATIONS,
            "models_trained": ["vitg_true_latent_scorer"],
            "no_latent_baseline_retrained": False,
            "final_epoch_only": True,
            "best_epoch_selection": False,
            "calibration_evaluations": 1,
        },
        "data_order_plan": data_order,
        "initialisation": dict(initialisation),
        "baseline": {
            key: baseline[key] for key in (
                "checkpoint_sha256", "state_digest", "receipt_digest",
                "vitl_terminal_digest", "reuse_mode")
        },
        "qualification_thresholds": V13.validate_frozen_qualification_thresholds(),
        "primary_exploratory_quantities": [
            "safety_auc", "latent_over_baseline_pairwise_gain"],
        "primary_thresholds": dict(PRIMARY_THRESHOLDS),
        "calibration_interpretation": STATUS,
        "calibration_previously_examined": True,
        "predictor_retraining_authorised": False,
        "predictor_checkpoint_access_route_present": False,
        "predictor_utility_scoring_route_present": False,
        "final_200_state_generation_route_present": False,
        "qualified_package_publication_route_present": False,
        "view_source_digests": {
            key: view[key] for key in getattr(ENCODER, "SOURCE_DIGEST_KEYS", ())
        },
    }
    return _signed(payload, "exploratory_scorer_contract_digest")


def issue_exploratory_contract(
        corpus: Mapping[str, Any], baseline: Mapping[str, Any],
        initialisation: Mapping[str, Any], *, root: Path = ROOT,
        ) -> dict[str, Any]:
    source_commit = _git_output(root, "rev-parse", "HEAD")
    _require(not _git_output(root, "status", "--porcelain"),
             "scorer contract requires a clean committed source tree")
    contract = build_exploratory_contract(
        corpus, baseline, initialisation, source_commit=source_commit)
    _publish_json_once(
        contract_path(root), contract, label="ViT-g exploratory scorer contract")
    installed = _validate_signed(
        _read_json(contract_path(root), label="ViT-g exploratory scorer contract"),
        "exploratory_scorer_contract_digest",
        "ViT-g exploratory scorer contract")
    _require(installed == contract, "installed exploratory contract changed")
    return installed


def _device(device_name: str) -> torch.device:
    if device_name == "auto":
        result = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        result = torch.device(device_name)
    _require(result.type != "cuda" or torch.cuda.is_available(),
             "accelerator requested but unavailable")
    return result


def _record_technical_failure(*, stage: str, error: BaseException,
                              completed_updates: int, completed_epochs: int,
                              root: Path) -> None:
    path = technical_failure_path(root)
    if path.exists() or path.is_symlink():
        return
    payload = _signed({
        "schema": TECHNICAL_FAILURE_SCHEMA,
        "status": "INVALID_TECHNICAL_VITG_SCORER_ATTEMPT",
        "complete": True,
        "stage": stage,
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "traceback": traceback.format_exc(),
        "completed_optimizer_updates": completed_updates,
        "completed_epochs": completed_epochs,
        "checkpoint_published": final_checkpoint_path(root).is_file(),
        "calibration_evaluation_authorised":
            evaluation_authorisation_path(root).is_file(),
        "retry_or_resume_authorised": False,
        "predictor_checkpoints_opened": 0,
        "predictor_utility_shards_opened": 0,
        "final_200_state_corpus_generated": False,
    }, "technical_failure_digest")
    _publish_json_once(path, payload, label="ViT-g scorer technical failure")


def train_latent_once(
        model: ViTGUtilityScorer, features: tuple[torch.Tensor, torch.Tensor,
                                                  Mapping[str, torch.Tensor]],
        *, initialisation: Mapping[str, Any], contract: Mapping[str, Any],
        device: torch.device, root: Path = ROOT,
        ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Run the exact fixed latent training once; no resume or retry exists."""

    directory = attempt_root(root)
    _require(not directory.exists() and not directory.is_symlink(),
             "the sole ViT-g scorer training attempt was already consumed")
    directory.mkdir(parents=True, exist_ok=False)
    attempt = _signed({
        "schema": ATTEMPT_SCHEMA,
        "status": STATUS,
        "attempt_number": 1,
        "maximum_attempts": 1,
        "contract_digest": contract["exploratory_scorer_contract_digest"],
        "initial_state_digest": initialisation["initial_state_digest"],
        "registered_seed": SCORER_SEED,
        "start_epoch": 1,
        "fixed_final_epoch": EPOCHS,
        "resume_source": None,
        "retry_or_resume_authorised": False,
        "calibration_opened": False,
    }, "attempt_digest")
    _publish_json_once(directory / "attempt.json", attempt,
                       label="ViT-g scorer attempt marker")

    completed_updates = completed_epochs = 0
    started = time.time()
    try:
        budget = frozen_budget()
        FROZEN.configure_determinism(SCORER_SEED)
        installed = torch.load(
            initialisation["path"], map_location="cpu", weights_only=False)
        model.load_state_dict(installed["model_state_dict"], strict=True)
        model.to(device)
        optimiser = torch.optim.AdamW(
            model.parameters(), lr=float(budget["lr"]),
            weight_decay=float(budget["weight_decay"]))
        latent, action_goal, targets = features
        _require(int(latent.shape[0]) == FIT_ROWS,
                 "training feature count changed")
        order_generator = torch.Generator(device="cpu")
        order_generator.manual_seed(SCORER_SEED)
        mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()
        epoch_trace = []
        for epoch in range(1, EPOCHS + 1):
            model.train()
            order = torch.randperm(FIT_ROWS, generator=order_generator)
            epoch_updates = 0
            for start in range(0, FIT_ROWS, int(budget["batch"])):
                index = order[start:start + int(budget["batch"])].to(device)
                progress, safety, completion = model(
                    latent[index], action_goal[index])
                loss = (mse(progress, targets["progress"][index])
                        + bce(safety, targets["safety"][index])
                        + bce(completion, targets["completion"][index]))
                _require(bool(torch.isfinite(loss).item()),
                         "non-finite training loss")
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                _require(all(parameter.grad is None
                             or bool(torch.isfinite(parameter.grad).all().item())
                             for parameter in model.parameters()),
                         "non-finite scorer gradient")
                nn.utils.clip_grad_norm_(
                    model.parameters(), float(budget["grad_clip"]))
                optimiser.step()
                _require(all(bool(torch.isfinite(parameter).all().item())
                             for parameter in model.parameters()),
                         "non-finite scorer parameter")
                completed_updates += 1
                epoch_updates += 1
            _require(epoch_updates == UPDATES_PER_EPOCH,
                     "optimizer updates per epoch changed")
            completed_epochs = epoch
            epoch_trace.append({
                "epoch": epoch,
                "completed_updates": completed_updates,
                "technical_finite": True,
                "performance_metric_inspected": False,
            })
            print(f"[vitg-scorer] completed technical epoch {epoch:02d}/60",
                  flush=True)
        _require(completed_updates == TOTAL_UPDATES,
                 "fixed optimizer-update budget changed")
        state = FROZEN._cpu_state(model)
        optimizer_state = optimiser.state_dict()
        checkpoint = {
            "schema": CHECKPOINT_SCHEMA,
            "status": STATUS,
            "attempt_number": 1,
            "contract_digest": contract["exploratory_scorer_contract_digest"],
            "initial_state_digest": initialisation["initial_state_digest"],
            "final_state_digest": FROZEN.state_dict_digest(state),
            "model_state_dict": state,
            "optimizer_state_dict": optimizer_state,
            "optimizer_state_digest": FROZEN.structured_digest(optimizer_state),
            "registered_seed": SCORER_SEED,
            "completed_epoch": EPOCHS,
            "completed_optimizer_updates": completed_updates,
            "example_presentations": PRESENTATIONS,
            "epoch_selection": "final_epoch_only_no_selection",
            "learning_rate_schedule": "constant",
            "last_epoch_order_digest": FROZEN.tensor_digest(
                order.to(torch.int64)),
            "final_order_generator_state_digest": FROZEN.tensor_digest(
                order_generator.get_state()),
            "technical_trace": epoch_trace,
            "training_wall_time_s": round(time.time() - started, 3),
        }
        FROZEN.atomic_torch_save(checkpoint, final_checkpoint_path(root))
        receipt = {
            "path": str(final_checkpoint_path(root)),
            "sha256": file_sha256(final_checkpoint_path(root)),
            "byte_count": final_checkpoint_path(root).stat().st_size,
            "final_state_digest": checkpoint["final_state_digest"],
            "optimizer_state_digest": checkpoint["optimizer_state_digest"],
            "completed_epoch": EPOCHS,
            "completed_optimizer_updates": completed_updates,
            "example_presentations": PRESENTATIONS,
            "training_wall_time_s": checkpoint["training_wall_time_s"],
            "technical_validity": True,
        }
        return state, receipt
    except BaseException as exc:
        _record_technical_failure(
            stage="latent_training", error=exc,
            completed_updates=completed_updates,
            completed_epochs=completed_epochs, root=root)
        raise


def _authorise_evaluation(*, contract: Mapping[str, Any],
                          training: Mapping[str, Any], root: Path) -> dict[str, Any]:
    path = evaluation_authorisation_path(root)
    _require(not path.exists() and not path.is_symlink(),
             "ViT-g calibration evaluation was already authorised")
    payload = _signed({
        "schema": EVALUATION_AUTH_SCHEMA,
        "status": STATUS,
        "complete": True,
        "contract_digest": contract["exploratory_scorer_contract_digest"],
        "final_state_digest": training["final_state_digest"],
        "calibration_evaluations_authorised": 1,
        "calibration_evaluations_completed_before_issue": 0,
        "repeat_after_interruption_authorised": False,
        "predictor_checkpoints_opened": 0,
        "predictor_utility_shards_opened": 0,
        "final_200_state_corpus_generated": False,
    }, "evaluation_authorisation_digest")
    _publish_json_once(path, payload,
                       label="ViT-g scorer evaluation authorisation")
    return payload


def exploratory_decision(*, safety_auc: float, pairwise_gain: float,
                         vitl_safety_auc: float,
                         vitl_pairwise_gain: float) -> dict[str, Any]:
    safety_gate = safety_auc >= PRIMARY_THRESHOLDS["safety_auc_min"]
    gain_gate = pairwise_gain >= PRIMARY_THRESHOLDS[
        "latent_over_baseline_pairwise_gain_min"]
    delta_auc = safety_auc - vitl_safety_auc
    delta_gain = pairwise_gain - vitl_pairwise_gain
    if safety_gate and gain_gate and delta_auc > 0.0 and delta_gain > 0.0:
        signal = "STRONG_SCALING_SIGNAL"
        conclusion = (
            "larger matched V-JEPA 2.1 features are promising; a fresh "
            "untouched qualification study would still be required")
    elif ((not safety_gate and not gain_gate)
          or delta_auc < 0.0 or delta_gain < 0.0):
        signal = "NO_SCALING_SIGNAL"
        conclusion = (
            "close this encoder-scale ablation; do not try ViT-G 2B "
            "automatically")
    else:
        signal = "MIXED_SIGNAL"
        conclusion = (
            "encoder scale alone is not established; do not proceed "
            "automatically to predictor retraining")
    return {
        "classification": signal,
        "safety_auc_gate_met": safety_gate,
        "latent_over_baseline_pairwise_gain_gate_met": gain_gate,
        "delta_vitg_minus_vitl_safety_auc": delta_auc,
        "delta_vitg_minus_vitl_latent_gain": delta_gain,
        "conclusion": conclusion,
        "next_decision_only": (
            "whether a fresh independent ViT-g scorer qualification study "
            "is justified"),
    }


def _output_storage_bytes(root: Path) -> int:
    directory = scorer_root(root)
    if not directory.exists():
        return 0
    return sum(path.stat().st_size for path in directory.rglob("*")
               if path.is_file() and not path.is_symlink())


def run_once(*, root: Path = ROOT, device_name: str = "auto") -> dict[str, Any]:
    """Execute the sole latent training and sole exploratory evaluation."""

    if terminal_path(root).exists() or terminal_path(root).is_symlink():
        return _validate_signed(
            _read_json(terminal_path(root), label="ViT-g exploratory result"),
            "exploratory_result_digest", "ViT-g exploratory result")
    _require(not technical_failure_path(root).exists()
             and not attempt_root(root).exists()
             and not evaluation_authorisation_path(root).exists(),
             "the sole ViT-g scorer attempt was consumed; retry is forbidden")
    _require(not _git_output(root, "status", "--porcelain"),
             "ViT-g scorer execution requires clean committed source")
    started = time.time()
    bundle = ENCODER.load_and_validate_encoded_training_view_for_consumption(
        root=root, verify_encoder_checkpoint=False)
    bundle = {**bundle, "resource": ENCODER.load_resource_smoke_receipt(root=root)}
    corpus = corpus_from_encoded_bundle(bundle)
    baseline = validate_reused_no_latent_baseline(corpus, root=root)
    initialisation = issue_initialisation(
        source_path=baseline["source_vitl_initialisation_path"],
        source_sha256=baseline["source_vitl_initialisation_sha256"],
        source_state_digest=baseline["source_vitl_initial_state_digest"],
        root=root)
    contract = issue_exploratory_contract(
        corpus, baseline, initialisation, root=root)
    device = _device(device_name)

    # Only fit shards are materialised before all 1,080 updates complete.
    fit_features = materialise_features(
        corpus["fit_rows"], corpus["horizon"], device)
    model = ViTGUtilityScorer()
    final_state, training = train_latent_once(
        model, fit_features, initialisation=initialisation,
        contract=contract, device=device, root=root)
    model.load_state_dict(final_state, strict=True)
    model.to(device)
    fit_metrics, _ = FROZEN.evaluate_model(
        model, fit_features[0], fit_features[1], corpus["fit_rows"],
        fit_features[2])

    evaluation_authorisation = _authorise_evaluation(
        contract=contract, training=training, root=root)
    try:
        # This is the only ViT-g access to fresh-calibration latent values.
        calibration_features = materialise_features(
            corpus["calibration_rows"], corpus["horizon"], device)
        calibration_metrics, predictions = FROZEN.evaluate_model(
            model, calibration_features[0], calibration_features[1],
            corpus["calibration_rows"], calibration_features[2])
        per_family = FROZEN._grouped_calibration(
            corpus["calibration_rows"], calibration_features[2], predictions,
            "family")
        per_stratum = FROZEN._grouped_calibration(
            corpus["calibration_rows"], calibration_features[2], predictions,
            "stratum")
    except BaseException as exc:
        _record_technical_failure(
            stage="calibration_evaluation", error=exc,
            completed_updates=TOTAL_UPDATES, completed_epochs=EPOCHS,
            root=root)
        raise

    fit_distribution = FROZEN.label_distribution(corpus["fit_rows"])
    calibration_distribution = FROZEN.label_distribution(
        corpus["calibration_rows"])
    criteria, criterion_details, pairwise_gain = V13.qualification_criteria(
        calibration_metrics, baseline["metrics"],
        fit_distribution, calibration_distribution)
    decision = exploratory_decision(
        safety_auc=float(calibration_metrics["safety"]["auc_any_hazard"]),
        pairwise_gain=pairwise_gain,
        vitl_safety_auc=float(baseline["vitl_metrics"]["safety"][
            "auc_any_hazard"]),
        vitl_pairwise_gain=float(baseline["vitl_pairwise_gain"]))
    result = {
        "schema": SCHEMA,
        "status": STATUS,
        "complete": True,
        "comparison_classification": COMPARISON_CLASSIFICATION,
        "scientific_result_valid": True,
        "exploratory_not_confirmatory": True,
        "training_execution_count": 1,
        "calibration_evaluation_count": 1,
        "contract_digest": contract["exploratory_scorer_contract_digest"],
        "latent_index_digest": contract["latent_index_digest"],
        "encoding_receipt_digest": contract["encoding_receipt_digest"],
        "target_encoder_checkpoint_sha256": contract[
            "target_encoder_checkpoint_sha256"],
        "initialisation": initialisation,
        "training": training,
        "evaluation_authorisation_digest": evaluation_authorisation[
            "evaluation_authorisation_digest"],
        "results": {
            "vitg": {
                "fit": fit_metrics,
                "calibration": calibration_metrics,
                "per_family_calibration": per_family,
                "per_stratum_calibration": per_stratum,
            },
            "vitl_frozen_failure": {
                "calibration": baseline["vitl_metrics"],
                "per_family_calibration": baseline[
                    "vitl_per_family_metrics"],
                "terminal_digest": baseline["vitl_terminal_digest"],
            },
            "no_latent_reused": {
                "calibration": baseline["metrics"],
                "per_family_calibration": baseline["per_family_metrics"],
                "checkpoint_sha256": baseline["checkpoint_sha256"],
                "state_digest": baseline["state_digest"],
                "receipt_digest": baseline["receipt_digest"],
                "retrained": False,
                "reevaluated": False,
                "reuse_mode": baseline["reuse_mode"],
            },
        },
        "latent_over_baseline_pairwise_gain": pairwise_gain,
        "frozen_qualification_criteria": criteria,
        "frozen_qualification_details": criterion_details,
        "would_meet_all_original_gates": all(criteria.values()),
        "exploratory_decision": decision,
        "qualified_scorer_package_published": False,
        "fresh_independent_qualification_required_for_publication": True,
        "technical_failures": [],
        "invalid_scientific_attempts": [],
        "predictor_retrained": False,
        "predictor_checkpoints_opened_for_utility": 0,
        "predictor_utility_shards_opened": 0,
        "vitg_applied_to_vitl_predictor_outputs": False,
        "final_200_state_corpus_generated": False,
        "wall_time_s": round(time.time() - started, 3),
        "scorer_storage_bytes_before_terminal": _output_storage_bytes(root),
        "nothing_left_running_by_this_process_after_exit": True,
    }
    signed = _signed(FROZEN._safe_json(result), "exploratory_result_digest")
    _publish_json_once(
        terminal_path(root), signed, label="ViT-g exploratory scorer result")
    return signed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto",
                        choices=("auto", "cpu", "cuda"))
    args = parser.parse_args(argv)
    result = run_once(device_name=args.device)
    print(json.dumps({
        "status": result["status"],
        "decision": result["exploratory_decision"]["classification"],
        "safety_auc": result["results"]["vitg"]["calibration"][
            "safety"]["auc_any_hazard"],
        "latent_over_baseline_pairwise_gain": result[
            "latent_over_baseline_pairwise_gain"],
        "result_digest": result["exploratory_result_digest"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
