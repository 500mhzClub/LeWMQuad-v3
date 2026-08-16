#!/usr/bin/env python3
"""Train the one exploratory final-layer token-attentive ViT-L readout.

This consumer is deliberately downstream of the frozen failure-attribution
contract.  It changes only the readout over the already encoded
``[H=4, 768, 1024]`` true target tokens, reuses the frozen no-latent baseline,
performs one final-epoch calibration evaluation, and has no predictor,
planner, package-promotion, or final-corpus route.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import types
from typing import Any, Mapping, Sequence


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lewm.oracle import go2_scorer_failure_attribution_v1_contract as CONTRACT  # noqa: E402
from scripts import train_go2_utility_scorer_v1_2 as FROZEN  # noqa: E402
from scripts import train_go2_utility_scorer_v1_3 as V13  # noqa: E402
from scripts import train_go2_utility_scorer_vjepa2_1_vitg_ablation_v1 as VITG  # noqa: E402


STATUS = "EXPLORATORY_FINAL_LAYER_ATTENTIVE_READOUT"
SCHEMA = "go2_v1_3_final_layer_attentive_readout_result_v1"
INITIALISATION_SCHEMA = "go2_v1_3_final_layer_attentive_readout_initialisation_v1"
ATTEMPT_SCHEMA = "go2_v1_3_final_layer_attentive_readout_attempt_v1"
CHECKPOINT_SCHEMA = "go2_v1_3_final_layer_attentive_readout_checkpoint_v1"
EVALUATION_AUTH_SCHEMA = "go2_v1_3_final_layer_attentive_readout_evaluation_authorisation_v1"
TECHNICAL_FAILURE_SCHEMA = "go2_v1_3_final_layer_attentive_readout_technical_failure_v1"

HORIZONS = 4
TOKENS = 768
TOKEN_DIM = 1024
HIDDEN_DIM = 512
ACTION_DIM = 40
GOAL_DIM = 3
QUERIES = 3
POOLER_DEPTH = 4
POOLER_HEADS = 16
POOLER_MLP_RATIO = 4.0
EPOCHS = 60
FIT_ROWS = 1152
CALIBRATION_ROWS = 288
EFFECTIVE_BATCH = 64
MICROBATCH = 4
UPDATES_PER_EPOCH = 18
TOTAL_UPDATES = 1080
PRESENTATIONS = 69120
DATA_ORDER_SEED = 20_260_811

FROZEN_VITL_TERMINAL_DIGEST = (
    "441f52d4199ba152825f30a9f5422b80537f68b9f7a3633f4e01610f964de419"
)
FROZEN_VITL_FAILED_SCORER_SHA256 = (
    "c9959582a0c8f266133623d143a6679a6502cad7a06fc9604d0a1a75ade174ef"
)
FROZEN_VITL_FINAL_CHECKPOINT_SHA256 = (
    "83a57e61808aa6a23b5a56ba428b7dc926932ea14067c27995ebfc365ed7ec8b"
)
FROZEN_VITL_FINAL_STATE_DIGEST = (
    "bb0f947cca8c724961f3bf98a2d717854e038e2625297019bc1b0957e4896874"
)
ORIGINAL_PRIMARY = {
    "safety_auc": 0.7043234199,
    "latent_over_baseline_pairwise_gain": 0.0317880795,
}
VITG_PRIMARY = {
    "safety_auc": 0.6332379770,
    "latent_over_baseline_pairwise_gain": 0.0019867550,
}
PRIMARY_THRESHOLDS = {
    "safety_auc_min": 0.75,
    "latent_over_baseline_pairwise_gain_min": 0.05,
}

OFFICIAL_REPOSITORY_COMMIT = "204698b45b3712590f06245fbfba32d3be539812"
OFFICIAL_REPOSITORY = Path(
    "/home/andrewknowles/.cache/vjepa2-204698b45b3712590f06245fbfba32d3be539812"
)
OFFICIAL_FILES = {
    "src/models/attentive_pooler.py": (
        "9be7047d6bfce50575956a57e36d87a37bf63ae84ec92a9ba8649bf1ab7d5feb", 4372),
    "src/models/utils/modules.py": (
        "b93f6c7e0747deb216419c000c2878f11a9189024a9adeacfd437e172396dff0", 23001),
    "src/utils/tensors.py": (
        "782b58bd2af456e184750e5318ab773105108383f61b280fe4c7a90f46add2c8", 1832),
    "configs/eval_2_1/vitl-384/in1k.yaml": (
        "c9e378792ae3437ca77d3c9d6f7ff3f448128312cca34c25b4718a1365937129", 3735),
    "evals/image_classification_frozen/eval.py": (
        "ff35b2729d45fc6b212275bec580704673b69058b00064f6e54b90e01e1a50e0", 15577),
}
OFFICIAL_POOLER_BINDING_DIGEST = (
    "f436439c72e725bfd7f3caab517f2b7c870cac1cf4060623fe0c1f6da63591e6"
)


class AttentiveReadoutError(RuntimeError):
    """The frozen exploratory readout contract or its one execution changed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AttentiveReadoutError(message)


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


def _signed(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = dict(value)
    _require(key not in result, f"{key} already present")
    result[key] = canonical_digest(result)
    return result


def _validate_signed(value: Mapping[str, Any], key: str,
                     label: str) -> dict[str, Any]:
    result = dict(value)
    recorded = result.pop(key, None)
    _require(isinstance(recorded, str) and len(recorded) == 64,
             f"{label} self digest is malformed")
    _require(recorded == canonical_digest(result),
             f"{label} self digest does not verify")
    result[key] = recorded
    return result


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(),
             f"{label} is absent or not a regular file")
    value = json.loads(path.read_text())
    _require(isinstance(value, dict), f"{label} is not an object")
    return value


def _publish_json_once(path: Path, value: Mapping[str, Any], label: str) -> None:
    V13.publish_json_once(path, value, label=label)


def generated_root(root: Path = ROOT) -> Path:
    return root / CONTRACT.GENERATED_ROOT


def require_generated_root(root: Path = ROOT) -> Path:
    logical = generated_root(root)
    if root.resolve() != ROOT.resolve():
        logical.mkdir(parents=True, exist_ok=True)
        return logical
    _require(logical.is_symlink(),
             "registered failure-attribution output alias is absent")
    target = logical.resolve()
    _require(target == CONTRACT.REGISTERED_GENERATED_TARGET_ROOT
             and target.is_dir() and not target.is_symlink(),
             "registered failure-attribution output alias changed")
    return logical


def contract_path(root: Path = ROOT) -> Path:
    return generated_root(root) / "diagnostic_contract.json"


def scorer_root(root: Path = ROOT) -> Path:
    return generated_root(root) / "attentive_readout"


def initialisation_path(root: Path = ROOT) -> Path:
    return scorer_root(root) / "initialisation.pt"


def attempt_root(root: Path = ROOT) -> Path:
    return scorer_root(root) / "training/attempt_000"


def final_checkpoint_path(root: Path = ROOT) -> Path:
    return attempt_root(root) / "final_epoch_060.pt"


def evaluation_authorisation_path(root: Path = ROOT) -> Path:
    return scorer_root(root) / "evaluation_authorisation.json"


def result_path(root: Path = ROOT) -> Path:
    return scorer_root(root) / "exploratory_result.json"


def technical_failure_path(root: Path = ROOT) -> Path:
    return scorer_root(root) / "technical_failure.json"


def safety_audit_path(root: Path = ROOT) -> Path:
    return generated_root(root) / "safety_observability/audit.json"


def latent_dependence_result_path(root: Path = ROOT) -> Path:
    return generated_root(root) / "latent_dependence/result.json"


def _git_output(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.strip()


def source_closure(root: Path = ROOT) -> dict[str, Any]:
    status = _git_output(root, "status", "--porcelain=v1")
    _require(status == "", "diagnostic source must be clean and committed")
    head = _git_output(root, "rev-parse", "HEAD")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", CONTRACT.SOURCE_BASE_COMMIT, head],
        cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _require(ancestor.returncode == 0,
             "diagnostic source does not descend from the frozen ViT-g result")
    files: dict[str, Any] = {}
    for relative in CONTRACT.SOURCE_CLOSURE_PATHS:
        path = root / relative
        _require(path.is_file() and not path.is_symlink(),
                 f"source closure path is absent: {relative}")
        files[relative] = {
            "path": relative, "sha256": file_sha256(path),
            "byte_count": path.stat().st_size,
        }
    unsigned = {
        "schema": CONTRACT.SOURCE_CLOSURE_SCHEMA,
        "source_repository_commit": head,
        "source_repository_clean": True,
        "git_status_porcelain_v1": status,
        "files": files,
    }
    return {
        **unsigned,
        CONTRACT.SOURCE_CLOSURE_SELF_KEY: CONTRACT.canonical_digest(unsigned),
    }


def issue_contract(root: Path = ROOT) -> dict[str, Any]:
    require_generated_root(root)
    closure = source_closure(root)
    value = CONTRACT.build_contract(closure)
    path = contract_path(root)
    if path.exists() or path.is_symlink():
        existing = CONTRACT.validate_contract(
            _read_json(path, "failure-attribution contract"))
        _require(existing == value,
                 "existing diagnostic contract belongs to other source bytes")
        return existing
    _publish_json_once(path, value, "failure-attribution contract")
    return value


def validate_official_pooler_source() -> dict[str, Any]:
    _require(OFFICIAL_REPOSITORY.is_dir()
             and not OFFICIAL_REPOSITORY.is_symlink(),
             "pinned official V-JEPA repository is absent")
    head = _git_output(OFFICIAL_REPOSITORY, "rev-parse", "HEAD")
    _require(head == OFFICIAL_REPOSITORY_COMMIT,
             "official V-JEPA repository commit changed")
    files = {}
    for relative, (expected_sha, expected_bytes) in OFFICIAL_FILES.items():
        path = OFFICIAL_REPOSITORY / relative
        _require(path.is_file() and not path.is_symlink()
                 and path.stat().st_size == expected_bytes
                 and file_sha256(path) == expected_sha,
                 f"official attentive-pooler source changed at {relative}")
        files[relative] = {"sha256": expected_sha, "byte_count": expected_bytes}
    payload = CONTRACT.official_attentive_pooler_binding_payload()
    _require(payload["commit"] == head and payload["files"] == files
             and canonical_digest(payload) == OFFICIAL_POOLER_BINDING_DIGEST
             and OFFICIAL_POOLER_BINDING_DIGEST
             == CONTRACT.OFFICIAL_ATTENTIVE_POOLER_BINDING["binding_digest"],
             "official attentive-pooler binding payload changed")
    config = payload["config"]
    _require(config == {
        "embed_dim": HIDDEN_DIM,
        "depth": POOLER_DEPTH,
        "num_heads": POOLER_HEADS,
        "mlp_ratio": POOLER_MLP_RATIO,
        "norm_layer": "torch.nn.LayerNorm",
        "norm_eps": 1e-5,
        "activation": "GELU",
        "qkv_bias": True,
        "complete_block": True,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "drop_path": 0.0,
        "init_std": 0.02,
        "use_activation_checkpointing": True,
    }, "runner attentive-pooler configuration changed")
    return {
        **payload,
        "binding_digest": OFFICIAL_POOLER_BINDING_DIGEST,
        "rectangular_sequence_compatible": True,
        "dependency_compatibility": dict(
            CONTRACT.OFFICIAL_ATTENTIVE_POOLER_BINDING[
                "dependency_compatibility"]),
    }


def _official_pooler_class():
    validate_official_pooler_source()
    repo = str(OFFICIAL_REPOSITORY)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        importlib.import_module("timm.models.layers")
    except ModuleNotFoundError:
        # The pinned source imports only timm's small stochastic-depth helper.
        # The frozen official probe sets drop_path=0.0, so this dependency is
        # an import-only boundary; still provide the standard implementation
        # and later assert every instantiated block uses nn.Identity.
        def drop_path(value, drop_prob=0.0, training=False, scale_by_keep=True):
            if drop_prob == 0.0 or not training:
                return value
            keep_prob = 1.0 - float(drop_prob)
            shape = (value.shape[0],) + (1,) * (value.ndim - 1)
            random = value.new_empty(shape).bernoulli_(keep_prob)
            if keep_prob > 0.0 and scale_by_keep:
                random.div_(keep_prob)
            return value * random

        timm = types.ModuleType("timm")
        models = types.ModuleType("timm.models")
        layers = types.ModuleType("timm.models.layers")
        layers.drop_path = drop_path
        timm.models = models
        models.layers = layers
        sys.modules.setdefault("timm", timm)
        sys.modules.setdefault("timm.models", models)
        sys.modules.setdefault("timm.models.layers", layers)
    module = importlib.import_module("src.models.attentive_pooler")
    _require(Path(module.__file__).resolve()
             == (OFFICIAL_REPOSITORY / "src/models/attentive_pooler.py").resolve(),
             "AttentivePooler was imported from an unpinned source")
    modules = importlib.import_module("src.models.utils.modules")
    tensors = importlib.import_module("src.utils.tensors")
    _require(Path(modules.__file__).resolve()
             == (OFFICIAL_REPOSITORY / "src/models/utils/modules.py").resolve()
             and Path(tensors.__file__).resolve()
             == (OFFICIAL_REPOSITORY / "src/utils/tensors.py").resolve(),
             "official attentive-pooler transitive source changed")
    return module.AttentivePooler


def fixed_horizon_embeddings() -> torch.Tensor:
    """Canonical non-trainable H1--H4 sinusoidal identity buffer."""

    raw = CONTRACT.horizon_embedding_float32_bytes()
    _require(hashlib.sha256(raw).hexdigest()
             == CONTRACT.HORIZON_EMBEDDING_SHA256,
             "frozen horizon embedding bytes changed")
    return torch.from_numpy(
        np.frombuffer(raw, dtype="<f4").copy().reshape(HORIZONS, HIDDEN_DIM))


def attentive_seed() -> tuple[int, str]:
    return CONTRACT.ATTENTIVE_SEED, CONTRACT.ATTENTIVE_SEED_KEY_DIGEST


class FinalLayerAttentiveUtilityScorer(nn.Module):
    """Official three-query pooler over all final-layer H1--H4 tokens."""

    def __init__(self, pooler_class=None) -> None:
        super().__init__()
        pooler_class = pooler_class or _official_pooler_class()
        self.token_projection = nn.Linear(TOKEN_DIM, HIDDEN_DIM)
        self.register_buffer("horizon_embeddings", fixed_horizon_embeddings(),
                             persistent=True)
        self.pooler = pooler_class(
            num_queries=QUERIES, embed_dim=HIDDEN_DIM,
            num_heads=POOLER_HEADS, mlp_ratio=POOLER_MLP_RATIO,
            depth=POOLER_DEPTH, norm_layer=nn.LayerNorm, init_std=0.02,
            qkv_bias=True, complete_block=True,
            use_activation_checkpointing=True)
        if hasattr(self.pooler, "blocks") and self.pooler.blocks is not None:
            _require(all(isinstance(block.drop_path, nn.Identity)
                         for block in self.pooler.blocks),
                     "official probe unexpectedly enabled stochastic depth")
        self.context = nn.Sequential(
            nn.Linear(ACTION_DIM + GOAL_DIM, HIDDEN_DIM), nn.SiLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM))
        self.fuse = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM), nn.SiLU())
        self.progress = nn.Linear(HIDDEN_DIM, 1)
        self.safety = nn.Linear(HIDDEN_DIM, 1)
        self.completion = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, latent: torch.Tensor,
                action_goal: torch.Tensor) -> tuple[torch.Tensor, ...]:
        _require(latent.ndim == 4
                 and tuple(latent.shape[1:]) == (HORIZONS, TOKENS, TOKEN_DIM),
                 "attentive latent input shape changed")
        projected = self.token_projection(latent)
        projected = projected + self.horizon_embeddings[None, :, None, :]
        sequence = projected.reshape(len(projected), HORIZONS * TOKENS,
                                     HIDDEN_DIM)
        component = self.pooler(sequence)
        _require(tuple(component.shape[1:]) == (QUERIES, HIDDEN_DIM),
                 "official pooler output shape changed")
        context = self.context(action_goal)[:, None, :].expand(-1, QUERIES, -1)
        fused = self.fuse(torch.cat((component, context), dim=-1))
        return (self.progress(fused[:, 0]).squeeze(-1),
                self.safety(fused[:, 1]).squeeze(-1),
                self.completion(fused[:, 2]).squeeze(-1))


def build_initialisation(*, source_path: Path,
                         source_sha256: str,
                         source_state_digest: str,
                         root: Path = ROOT) -> dict[str, Any]:
    path = initialisation_path(root)
    _require(source_path.is_file() and not source_path.is_symlink()
             and file_sha256(source_path) == source_sha256,
             "frozen ViT-L initialisation bytes changed")
    source_artifact = torch.load(source_path, map_location="cpu",
                                 weights_only=False)
    source_state = source_artifact.get("model_state_dict")
    _require(isinstance(source_state, Mapping)
             and FROZEN.state_dict_digest(source_state) == source_state_digest,
             "frozen ViT-L initial state changed")
    seed, seed_digest = attentive_seed()
    FROZEN.configure_determinism(seed)
    model = FinalLayerAttentiveUtilityScorer()
    state = FROZEN._cpu_state(model)
    _require(sum(parameter.numel() for parameter in model.parameters())
             == CONTRACT.ATTENTIVE_READOUT_ARCHITECTURE[
                 "trainable_parameter_count"],
             "attentive trainable parameter count changed")
    initialisation_receipt = {
        "algorithm": (
            "construct the complete frozen attentive architecture once after "
            "configure_determinism(architecture_seed)"
        ),
        "all_trainable_parameters_use_architecture_seed": True,
        "copied_predecessor_parameter_count": 0,
        "source_vitl_initialisation_is_lineage_only": True,
        "nontrainable_horizon_embedding_digest": FROZEN.tensor_digest(
            state["horizon_embeddings"]),
    }
    expected = {
        "schema": INITIALISATION_SCHEMA,
        "status": STATUS,
        "registered_seed": seed,
        "architecture_seed_digest": seed_digest,
        "source_vitl_initialisation_sha256": source_sha256,
        "source_vitl_initial_state_digest": source_state_digest,
        "model_state_dict": state,
        "initial_state_digest": FROZEN.state_dict_digest(state),
        "parameter_initialisation": initialisation_receipt,
        "trainable_parameter_count": sum(p.numel() for p in model.parameters()),
    }
    if path.exists() or path.is_symlink():
        _require(path.is_file() and not path.is_symlink(),
                 "attentive initialisation path changed")
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        _require(set(artifact) == set(expected)
                 and all(key == "model_state_dict"
                         or artifact.get(key) == expected[key]
                         for key in expected)
                 and isinstance(artifact.get("model_state_dict"), Mapping)
                 and FROZEN.state_dict_digest(artifact["model_state_dict"])
                 == expected["initial_state_digest"]
                 and all(torch.equal(artifact["model_state_dict"][key], value)
                         for key, value in state.items()),
                 "attentive initialisation changed")
        return artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    FROZEN.atomic_torch_save(expected, path)
    return expected


def _small_features(rows: Sequence[Mapping[str, Any]],
                    device: torch.device) -> tuple[torch.Tensor,
                                                   dict[str, torch.Tensor]]:
    action = np.empty((len(rows), ACTION_DIM), dtype=np.float32)
    for index, row in enumerate(rows):
        flat = [value for block in row["action_blocks"] for value in block]
        _require(len(flat) == ACTION_DIM,
                 "candidate action stopped matching the frozen 40-D input")
        action[index] = np.asarray(flat, dtype=np.float32)
    goal = np.asarray([row["goal_binding_input"] for row in rows],
                      dtype=np.float32)
    _require(goal.shape == (len(rows), GOAL_DIM),
             "goal input stopped matching the frozen 3-D binding")
    action_goal = torch.from_numpy(np.concatenate((action, goal), axis=-1)).to(device)
    targets = {
        key: torch.tensor([row[key] for row in rows], dtype=torch.float32,
                          device=device)
        for key in ("progress", "safety", "completion")
    }
    return action_goal, targets


def _token_batch(rows: Sequence[Mapping[str, Any]], store: Any,
                 indices: Sequence[int], device: torch.device) -> torch.Tensor:
    positions = np.asarray([rows[int(index)]["_latent_index"] for index in indices],
                           dtype=np.int64)
    array = np.asarray(store[positions], dtype=np.float32)
    _require(tuple(array.shape[1:]) == (HORIZONS, TOKENS, TOKEN_DIM)
             and np.all(np.isfinite(array)),
             "ViT-L final-layer token shard changed")
    return torch.from_numpy(array).to(device)


def frozen_budget() -> dict[str, Any]:
    budget = dict(FROZEN.SCORER["training"])
    FROZEN._validate_budget(budget)
    _require(int(budget["batch"]) == EFFECTIVE_BATCH
             and int(budget["epochs"]) == EPOCHS,
             "frozen scorer batch or epoch budget changed")
    return budget


def registered_fit_rows_and_data_order(
        rows: Sequence[Mapping[str, Any]],
        ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Recompute the exact predecessor row/permutation-order witnesses."""

    ordered = sorted((dict(row) for row in rows), key=lambda row: (
        str(row["state_id"]), int(row["candidate_index"])))
    _require(len(ordered) == FIT_ROWS
             and len({str(row["state_id"]) for row in ordered}) == 96,
             "registered fit row inventory changed")
    witness = V13.registered_data_order_plan(ordered)
    expected = CONTRACT.DATA_ORDER_CONTRACT
    _require(all(witness.get(key) == value for key, value in expected.items()
                 if key not in {"recomputation_boundary", "rows"}),
             "registered data-order witness changed")
    return ordered, witness


def _record_technical_failure(*, stage: str, error: BaseException,
                              epochs: int, updates: int,
                              root: Path) -> None:
    path = technical_failure_path(root)
    if path.exists() or path.is_symlink():
        return
    payload = _signed({
        "schema": TECHNICAL_FAILURE_SCHEMA,
        "status": "INVALID_TECHNICAL_EXPLORATORY_ATTEMPT",
        "stage": stage, "exception_type": type(error).__name__,
        "exception": str(error), "traceback": traceback.format_exc(),
        "completed_epochs": epochs, "completed_optimizer_updates": updates,
        "retry_or_resume_authorised": False,
        "calibration_evaluation_completed": False,
        "predictor_checkpoints_opened": 0,
        "predictor_utility_shards_opened": 0,
        "final_200_state_corpus_generated": False,
    }, "technical_failure_digest")
    _publish_json_once(path, payload, "attentive-readout technical failure")


def train_once(model: FinalLayerAttentiveUtilityScorer, *,
               rows: list[dict[str, Any]], store: Any,
               initialisation: Mapping[str, Any],
               diagnostic_contract: Mapping[str, Any],
               diagnostic_prerequisites: Mapping[str, Any],
               data_order_witness: Mapping[str, Any],
               device: torch.device, root: Path = ROOT) -> tuple[dict[str, torch.Tensor],
                                                                 dict[str, Any]]:
    directory = attempt_root(root)
    _require(not directory.exists() and not directory.is_symlink(),
             "the sole attentive-readout attempt was consumed")
    directory.mkdir(parents=True, exist_ok=False)
    contract_digest = diagnostic_contract.get("diagnostic_contract_digest")
    attempt = _signed({
        "schema": ATTEMPT_SCHEMA, "status": STATUS,
        "attempt_number": 1, "maximum_attempts": 1,
        "diagnostic_contract_digest": contract_digest,
        "diagnostic_prerequisites": dict(diagnostic_prerequisites),
        "initial_state_digest": initialisation["initial_state_digest"],
        "registered_seed": initialisation["registered_seed"],
        "data_order_seed": DATA_ORDER_SEED,
        "data_order_witness": {
            key: data_order_witness[key] for key in (
                "base_training_view_row_digest_sequence_digest",
                "permutation_plan_digest", "row_presentation_plan_digest",
                "final_generator_state_digest")
        },
        "effective_batch": EFFECTIVE_BATCH, "microbatch": MICROBATCH,
        "start_epoch": 1, "fixed_final_epoch": EPOCHS,
        "resume_source": None, "retry_or_resume_authorised": False,
        "calibration_opened": False,
    }, "attempt_digest")
    _publish_json_once(directory / "attempt.json", attempt,
                       "attentive-readout attempt")
    completed_epochs = completed_updates = 0
    started = time.time()
    try:
        budget = frozen_budget()
        seed = int(initialisation["registered_seed"])
        FROZEN.configure_determinism(seed)
        model.load_state_dict(initialisation["model_state_dict"], strict=True)
        model.to(device)
        optimiser = torch.optim.AdamW(
            model.parameters(), lr=float(budget["lr"]),
            weight_decay=float(budget["weight_decay"]))
        action_goal, targets = _small_features(rows, device)
        order_generator = torch.Generator(device="cpu")
        order_generator.manual_seed(DATA_ORDER_SEED)
        epoch_trace = []
        for epoch in range(1, EPOCHS + 1):
            model.train()
            order = torch.randperm(FIT_ROWS, generator=order_generator)
            epoch_updates = 0
            for start in range(0, FIT_ROWS, EFFECTIVE_BATCH):
                batch_cpu = order[start:start + EFFECTIVE_BATCH]
                _require(len(batch_cpu) == EFFECTIVE_BATCH,
                         "effective training batch changed")
                optimiser.zero_grad(set_to_none=True)
                loss_sum = torch.zeros((), dtype=torch.float32, device=device)
                for micro_start in range(0, EFFECTIVE_BATCH, MICROBATCH):
                    micro_cpu = batch_cpu[micro_start:micro_start + MICROBATCH]
                    micro = micro_cpu.to(device)
                    token = _token_batch(rows, store, micro_cpu.tolist(), device)
                    progress, safety, completion = model(
                        token, action_goal[micro])
                    loss = (
                        F.mse_loss(progress, targets["progress"][micro],
                                   reduction="sum")
                        + F.binary_cross_entropy_with_logits(
                            safety, targets["safety"][micro], reduction="sum")
                        + F.binary_cross_entropy_with_logits(
                            completion, targets["completion"][micro],
                            reduction="sum")) / EFFECTIVE_BATCH
                    _require(bool(torch.isfinite(loss).item()),
                             "non-finite attentive training loss")
                    loss.backward()
                    loss_sum += loss.detach()
                    del token
                _require(all(parameter.grad is None
                             or bool(torch.isfinite(parameter.grad).all().item())
                             for parameter in model.parameters()),
                         "non-finite attentive gradient")
                nn.utils.clip_grad_norm_(model.parameters(),
                                         float(budget["grad_clip"]))
                optimiser.step()
                _require(all(bool(torch.isfinite(parameter).all().item())
                             for parameter in model.parameters()),
                         "non-finite attentive parameter")
                completed_updates += 1
                epoch_updates += 1
            _require(epoch_updates == UPDATES_PER_EPOCH,
                     "optimizer updates per epoch changed")
            completed_epochs = epoch
            epoch_trace.append({
                "epoch": epoch, "completed_updates": completed_updates,
                "technical_finite": True,
                "performance_metric_inspected": False,
            })
            print(f"[attentive-readout] completed technical epoch {epoch:02d}/60",
                  flush=True)
        _require(completed_updates == TOTAL_UPDATES,
                 "fixed optimizer-update budget changed")
        _require(FROZEN.tensor_digest(order.to(torch.int64))
                 == data_order_witness["permutations"][-1][
                     "permutation_tensor_digest"]
                 and FROZEN.tensor_digest(order_generator.get_state())
                 == data_order_witness["final_generator_state_digest"],
                 "executed training order differs from registered plan")
        state = FROZEN._cpu_state(model)
        optimiser_state = optimiser.state_dict()
        checkpoint = {
            "schema": CHECKPOINT_SCHEMA, "status": STATUS,
            "attempt_number": 1,
            "attempt_digest": attempt["attempt_digest"],
            "diagnostic_contract_digest": contract_digest,
            "diagnostic_prerequisites": dict(diagnostic_prerequisites),
            "initial_state_digest": initialisation["initial_state_digest"],
            "final_state_digest": FROZEN.state_dict_digest(state),
            "model_state_dict": state,
            "optimizer_state_dict": optimiser_state,
            "optimizer_state_digest": FROZEN.structured_digest(optimiser_state),
            "registered_seed": seed, "completed_epoch": EPOCHS,
            "data_order_seed": DATA_ORDER_SEED,
            "data_order_witness": attempt["data_order_witness"],
            "completed_optimizer_updates": completed_updates,
            "example_presentations": PRESENTATIONS,
            "effective_batch": EFFECTIVE_BATCH, "microbatch": MICROBATCH,
            "epoch_selection": "final_epoch_only_no_selection",
            "learning_rate_schedule": "constant",
            "last_epoch_order_digest": FROZEN.tensor_digest(order.to(torch.int64)),
            "final_order_generator_state_digest": FROZEN.tensor_digest(
                order_generator.get_state()),
            "technical_trace": epoch_trace,
            "training_wall_time_s": round(time.time() - started, 3),
        }
        FROZEN.atomic_torch_save(checkpoint, final_checkpoint_path(root))
        return state, {
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
            "attempt_digest": checkpoint["attempt_digest"],
            "data_order_witness": checkpoint["data_order_witness"],
        }
    except BaseException as exc:
        _record_technical_failure(
            stage="attentive_training", error=exc, epochs=completed_epochs,
            updates=completed_updates, root=root)
        raise


def _evaluate_streaming(model: nn.Module, *, rows: list[dict[str, Any]],
                        store: Any, device: torch.device,
                        batch: int = MICROBATCH) -> tuple[dict[str, Any],
                                                         dict[str, np.ndarray],
                                                         dict[str, torch.Tensor]]:
    action_goal, targets_device = _small_features(rows, device)
    predicted = {key: [] for key in ("progress", "safety", "completion", "utility")}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch):
            indices = list(range(start, min(start + batch, len(rows))))
            tokens = _token_batch(rows, store, indices, device)
            progress, safety_logit, completion_logit = model(
                tokens, action_goal[start:start + len(indices)])
            safety = torch.sigmoid(safety_logit)
            completion = torch.sigmoid(completion_logit)
            utility = FROZEN.composite(progress, safety_logit, completion_logit)
            for key, value in (
                ("progress", progress), ("safety", safety),
                ("completion", completion), ("utility", utility),
            ):
                predicted[key].append(
                    value.detach().cpu().numpy().astype(np.float64))
            del tokens
    arrays = {key: np.concatenate(value) for key, value in predicted.items()}
    targets_cpu = {key: value.detach().cpu() for key, value in targets_device.items()}
    true = {key: value.numpy().astype(np.float64)
            for key, value in targets_cpu.items()}
    metrics = FROZEN._evaluate_arrays(rows, true, arrays)
    return metrics, arrays, targets_cpu


def _metric_delta(value: Any, reference: Any) -> Any:
    if isinstance(value, Mapping) and isinstance(reference, Mapping):
        return {key: _metric_delta(member, reference[key])
                for key, member in value.items() if key in reference}
    if (not isinstance(value, bool) and isinstance(value, (int, float))
            and not isinstance(reference, bool)
            and isinstance(reference, (int, float))
            and math.isfinite(float(value)) and math.isfinite(float(reference))):
        return float(value) - float(reference)
    return None


def _load_frozen_vitg_result(*, root: Path) -> dict[str, Any]:
    value = VITG._validate_signed(
        VITG._read_json(VITG.terminal_path(root), label="frozen ViT-g result"),
        "exploratory_result_digest", "frozen ViT-g result")
    _require(value["exploratory_result_digest"]
             == CONTRACT.FROZEN_VITG_RESULT_DIGEST
             and value.get("complete") is True
             and value.get("scientific_result_valid") is True
             and value.get("exploratory_not_confirmatory") is True
             and value.get("training_execution_count") == 1
             and value.get("calibration_evaluation_count") == 1
             and value.get("predictor_retrained") is False
             and value.get("predictor_checkpoints_opened_for_utility") == 0
             and value.get("predictor_utility_shards_opened") == 0
             and value.get("final_200_state_corpus_generated") is False
             and isinstance(value.get("results", {}).get("vitg"), Mapping),
             "frozen ViT-g exploratory result changed")
    return value


def per_family_primary_consistency(
        *, attentive: Mapping[str, Any], existing_vitl: Mapping[str, Any],
        baseline: Mapping[str, Any]) -> dict[str, Any]:
    families = sorted(set(attentive) | set(existing_vitl) | set(baseline))
    _require(set(attentive) == set(existing_vitl) == set(baseline),
             "per-family comparison sets changed")
    rows = []
    inconsistent = []
    for family in families:
        left, old, no_latent = (
            attentive[family], existing_vitl[family], baseline[family])
        left_auc = left["safety"].get("auc_any_hazard")
        old_auc = old["safety"].get("auc_any_hazard")
        left_pairwise = left["composite"].get("pairwise_ordering_accuracy")
        old_pairwise = old["composite"].get("pairwise_ordering_accuracy")
        baseline_pairwise = no_latent["composite"].get(
            "pairwise_ordering_accuracy")
        auc_values = (left_auc, old_auc)
        auc_delta = (None if any(
            value is None or not math.isfinite(float(value))
            for value in auc_values) else float(left_auc) - float(old_auc))
        left_values = (left_pairwise, baseline_pairwise)
        old_values = (old_pairwise, baseline_pairwise)
        left_gain = (None if any(
            value is None or not math.isfinite(float(value))
            for value in left_values) else
            float(left_pairwise) - float(baseline_pairwise))
        old_gain = (None if any(
            value is None or not math.isfinite(float(value))
            for value in old_values) else
            float(old_pairwise) - float(baseline_pairwise))
        gain_delta = (None if left_gain is None or old_gain is None else
                      left_gain - old_gain)
        is_inconsistent = ((auc_delta is not None and auc_delta < 0.0)
                           or (gain_delta is not None and gain_delta < 0.0))
        if is_inconsistent:
            inconsistent.append(family)
        rows.append({
            "family": family,
            "attentive_minus_existing_vitl_safety_auc": auc_delta,
            "attentive_latent_over_baseline_pairwise_gain": left_gain,
            "existing_vitl_latent_over_baseline_pairwise_gain": old_gain,
            "attentive_minus_existing_vitl_latent_gain": gain_delta,
            "inconsistent_primary_improvement": is_inconsistent,
        })
    return {
        "rule": dict(CONTRACT.INTERPRETATION_RULES["per_family_consistency"]),
        "families": rows,
        "inconsistent_families": inconsistent,
        "no_inconsistent_per_family_primary_improvement": not inconsistent,
    }


def exploratory_decision(*, safety_auc: float, pairwise_gain: float,
                         family_consistent: bool) -> dict[str, Any]:
    delta_auc = safety_auc - ORIGINAL_PRIMARY["safety_auc"]
    delta_gain = pairwise_gain - ORIGINAL_PRIMARY[
        "latent_over_baseline_pairwise_gain"]
    safety_gate = safety_auc >= PRIMARY_THRESHOLDS["safety_auc_min"]
    gain_gate = pairwise_gain >= PRIMARY_THRESHOLDS[
        "latent_over_baseline_pairwise_gain_min"]
    if (safety_gate and gain_gate and delta_auc > 0 and delta_gain > 0
            and family_consistent):
        classification = "STRONG_READOUT_SIGNAL"
        conclusion = (
            "final-layer ViT-L features contain more usable planning information "
            "than the original scorer extracted; fresh qualification is required")
    elif (not safety_gate and not gain_gate) or (delta_auc <= 0 and delta_gain <= 0):
        classification = "NO_READOUT_SIGNAL"
        conclusion = (
            "close learned utility scoring for the current final-layer H1-H4 "
            "latent contract; do not train another probe architecture")
    else:
        classification = "MIXED_READOUT_SIGNAL"
        conclusion = (
            "the readout hypothesis is not established; do not apply this "
            "scorer to predictor outputs")
    return {
        "classification": classification,
        "safety_auc_gate_met": safety_gate,
        "latent_over_baseline_pairwise_gain_gate_met": gain_gate,
        "delta_attentive_minus_existing_vitl_safety_auc": delta_auc,
        "delta_attentive_minus_existing_vitl_latent_gain": delta_gain,
        "no_inconsistent_per_family_primary_improvement": family_consistent,
        "conclusion": conclusion,
    }


def _load_contract(root: Path) -> dict[str, Any]:
    require_generated_root(root)
    value = CONTRACT.validate_contract(
        _read_json(contract_path(root), "failure-attribution contract"))
    _require(value == CONTRACT.build_contract(source_closure(root)),
             "diagnostic contract is not bound to the live clean source")
    return value


def _load_required_diagnostics(
        diagnostic_contract: Mapping[str, Any], *, root: Path,
        ) -> dict[str, Any]:
    """Recompute/validate both frozen diagnostics before model construction."""

    from scripts import diagnose_go2_scorer_v1_3_latent_dependence_v1 as LATENT
    from scripts import run_go2_safety_observability_diagnostic_v1 as SAFETY

    contract_digest = diagnostic_contract[CONTRACT.CONTRACT_SELF_KEY]
    _require(SAFETY.terminal_path(generated_root(root)).is_file()
             and not SAFETY.terminal_path(generated_root(root)).is_symlink()
             and SAFETY.audit_path(generated_root(root)).is_file()
             and not SAFETY.audit_path(generated_root(root)).is_symlink(),
             "safety-observability terminal and audit must preexist")
    safety = SAFETY.issue_audit(out_root=generated_root(root))
    _require(safety.get("schema") == SAFETY.AUDIT_SCHEMA
             and safety.get("complete") is True
             and safety.get("status") == CONTRACT.STATUS
             and safety.get("contract_digest") == contract_digest
             and safety.get("branch_count") == CONTRACT.EXPECTED_BRANCHES
             and isinstance(safety.get("branches"), list)
             and len(safety["branches"]) == CONTRACT.EXPECTED_BRANCHES,
             "completed safety-observability diagnostic changed")

    latent = LATENT.validate_result_for_consumption(root=root)
    _require(latent.get("schema") == LATENT.SCHEMA
             and latent.get("complete") is True
             and latent.get("status") == CONTRACT.STATUS
             and latent.get("failure_attribution_contract_digest")
             == contract_digest
             and latent.get("source_closure_digest")
             == diagnostic_contract["source_closure"][
                 CONTRACT.SOURCE_CLOSURE_SELF_KEY]
             and latent.get("calibration_diagnostic_session_count") == 1
             and isinstance(latent.get("results"), Mapping)
             and set(latent["results"])
             == set(CONTRACT.DIAGNOSTIC_PREREQUISITES["latent_dependence"][
                 "required_variants"])
             and latent.get("training_executions") == 0
             and latent.get("predictor_checkpoints_opened") == 0
             and latent.get("predictor_utility_shards_opened") == 0,
             "completed latent-dependence diagnostic changed")
    return {
        "safety_observability_audit_digest": safety[SAFETY.AUDIT_SELF_KEY],
        "safety_observability_terminal_digest": safety[
            "terminal_manifest_digest"],
        "latent_dependence_result_digest": latent[LATENT.RESULT_SELF_KEY],
        "diagnostic_execution_order": list(
            CONTRACT.DIAGNOSTIC_PREREQUISITES["execution_order"]),
    }


def validate_result_for_consumption(*, root: Path = ROOT) -> dict[str, Any]:
    """Validate the immutable terminal against live source and checkpoint."""

    diagnostic_contract = _load_contract(root)
    prerequisites = _load_required_diagnostics(
        diagnostic_contract, root=root)
    result = _validate_signed(
        _read_json(result_path(root), "attentive result"),
        "attentive_result_digest", "attentive result")
    _require(result.get("schema") == SCHEMA
             and result.get("status") == STATUS
             and result.get("complete") is True
             and result.get("scientific_result_valid") is True
             and result.get("exploratory_not_qualification") is True
             and result.get("diagnostic_contract_digest")
             == diagnostic_contract[CONTRACT.CONTRACT_SELF_KEY]
             and result.get("diagnostic_prerequisites") == prerequisites
             and result.get("official_pooler_binding_digest")
             == OFFICIAL_POOLER_BINDING_DIGEST
             and result.get("training_execution_count") == 1
             and result.get("calibration_evaluation_count") == 1
             and result.get("qualified_scorer_package_published") is False
             and result.get("predictor_retrained") is False
             and result.get("predictor_checkpoints_opened_for_utility") == 0
             and result.get("predictor_utility_shards_opened") == 0
             and result.get("final_200_state_corpus_generated") is False,
             "attentive result binding changed")
    _require(initialisation_path(root).is_file()
             and not initialisation_path(root).is_symlink(),
             "attentive initialisation artifact is absent")
    initialisation = torch.load(
        initialisation_path(root), map_location="cpu", weights_only=False)
    FROZEN.configure_determinism(CONTRACT.ATTENTIVE_SEED)
    reconstructed = FinalLayerAttentiveUtilityScorer()
    reconstructed_state = FROZEN._cpu_state(reconstructed)
    reconstructed_digest = FROZEN.state_dict_digest(reconstructed_state)
    _require(initialisation.get("schema") == INITIALISATION_SCHEMA
             and initialisation.get("status") == STATUS
             and initialisation.get("registered_seed")
             == CONTRACT.ATTENTIVE_SEED
             and initialisation.get("architecture_seed_digest")
             == CONTRACT.ATTENTIVE_SEED_KEY_DIGEST
             and initialisation.get("initial_state_digest")
             == reconstructed_digest
             and FROZEN.state_dict_digest(initialisation["model_state_dict"])
             == reconstructed_digest
             and all(torch.equal(initialisation["model_state_dict"][key], value)
                     for key, value in reconstructed_state.items())
             and initialisation.get("parameter_initialisation", {}).get(
                 "copied_predecessor_parameter_count") == 0
             and initialisation.get("parameter_initialisation", {}).get(
                 "all_trainable_parameters_use_architecture_seed") is True
             and result.get("initialisation", {}).get("initial_state_digest")
             == reconstructed_digest
             and result.get("initialisation")
             == {key: value for key, value in initialisation.items()
                 if key != "model_state_dict"},
             "attentive deterministic initialisation changed")
    training = result.get("training")
    _require(isinstance(training, Mapping)
             and Path(str(training.get("path"))).absolute()
             == final_checkpoint_path(root).absolute()
             and final_checkpoint_path(root).is_file()
             and not final_checkpoint_path(root).is_symlink()
             and file_sha256(final_checkpoint_path(root))
             == training.get("sha256"),
             "attentive final checkpoint file changed")
    checkpoint = torch.load(
        final_checkpoint_path(root), map_location="cpu", weights_only=False)
    _require(checkpoint.get("schema") == CHECKPOINT_SCHEMA
             and checkpoint.get("diagnostic_contract_digest")
             == diagnostic_contract[CONTRACT.CONTRACT_SELF_KEY]
             and checkpoint.get("diagnostic_prerequisites") == prerequisites
             and checkpoint.get("initial_state_digest")
             == reconstructed_digest
             and checkpoint.get("completed_epoch") == EPOCHS
             and checkpoint.get("completed_optimizer_updates") == TOTAL_UPDATES
             and checkpoint.get("data_order_seed") == DATA_ORDER_SEED
             and checkpoint.get("data_order_witness")
             == training.get("data_order_witness")
             and all(checkpoint["data_order_witness"].get(key)
                     == CONTRACT.DATA_ORDER_CONTRACT[key] for key in (
                         "base_training_view_row_digest_sequence_digest",
                         "permutation_plan_digest",
                         "row_presentation_plan_digest"))
             and checkpoint["data_order_witness"].get(
                 "final_generator_state_digest")
             == "f1826a6a0c7f2cde2dcd028393e1229f2a6931099a22b8c31f97b968dbc77cb2"
             and FROZEN.state_dict_digest(checkpoint["model_state_dict"])
             == checkpoint.get("final_state_digest")
             == training.get("final_state_digest")
             and FROZEN.structured_digest(checkpoint["optimizer_state_dict"])
             == checkpoint.get("optimizer_state_digest")
             == training.get("optimizer_state_digest"),
             "attentive final checkpoint content changed")
    _require(result["results"]["existing_vitl_frozen"]["terminal_digest"]
             == FROZEN_VITL_TERMINAL_DIGEST
             and result["results"]["vitg_frozen"]["result_digest"]
             == CONTRACT.FROZEN_VITG_RESULT_DIGEST
             and result["results"]["no_latent_reused"]["checkpoint_sha256"]
             == CONTRACT.FROZEN_BASELINE_CHECKPOINT_SHA256
             and result["results"]["no_latent_reused"]["state_digest"]
             == CONTRACT.FROZEN_BASELINE_STATE_DIGEST
             and result["results"]["no_latent_reused"]["receipt_digest"]
             == CONTRACT.FROZEN_BASELINE_RECEIPT_DIGEST,
             "attentive frozen comparison binding changed")
    evaluation = _validate_signed(
        _read_json(evaluation_authorisation_path(root),
                   "attentive evaluation authorisation"),
        "evaluation_authorisation_digest",
        "attentive evaluation authorisation")
    _require(evaluation["evaluation_authorisation_digest"]
             == result.get("evaluation_authorisation_digest")
             and evaluation.get("schema") == EVALUATION_AUTH_SCHEMA
             and evaluation.get("status") == STATUS
             and evaluation.get("diagnostic_contract_digest")
             == diagnostic_contract[CONTRACT.CONTRACT_SELF_KEY]
             and evaluation.get("diagnostic_prerequisites") == prerequisites
             and evaluation.get("final_checkpoint_sha256")
             == training.get("sha256")
             and evaluation.get("final_state_digest")
             == training.get("final_state_digest")
             and evaluation.get("calibration_states") == 24
             and evaluation.get("calibration_rows") == CALIBRATION_ROWS
             and evaluation.get("maximum_evaluations") == 1
             and evaluation.get("evaluation_number") == 1
             and evaluation.get("calibration_not_opened_before_authorisation")
             is True,
             "attentive evaluation authorisation changed")
    attempt = _validate_signed(
        _read_json(attempt_root(root) / "attempt.json",
                   "attentive training attempt"),
        "attempt_digest", "attentive training attempt")
    _require(attempt.get("schema") == ATTEMPT_SCHEMA
             and attempt.get("status") == STATUS
             and attempt.get("attempt_number") == 1
             and attempt.get("maximum_attempts") == 1
             and attempt.get("diagnostic_contract_digest")
             == diagnostic_contract[CONTRACT.CONTRACT_SELF_KEY]
             and attempt.get("diagnostic_prerequisites") == prerequisites
             and attempt.get("initial_state_digest") == reconstructed_digest
             and attempt.get("registered_seed") == CONTRACT.ATTENTIVE_SEED
             and attempt.get("data_order_seed") == DATA_ORDER_SEED
             and attempt.get("data_order_witness")
             == checkpoint.get("data_order_witness")
             and attempt.get("resume_source") is None
             and attempt.get("retry_or_resume_authorised") is False
             and attempt.get("calibration_opened") is False,
             "attentive training attempt lineage changed")
    _require(checkpoint.get("attempt_digest") == attempt["attempt_digest"]
             and training.get("attempt_digest") == attempt["attempt_digest"],
             "attentive checkpoint attempt binding changed")
    vitg = _load_frozen_vitg_result(root=root)
    _require(result["results"]["vitg_frozen"]["calibration"]
             == vitg["results"]["vitg"]["calibration"]
             and result["results"]["vitg_frozen"][
                 "per_family_calibration"]
             == vitg["results"]["vitg"]["per_family_calibration"]
             and set(result["results"]["attentive"][
                 "per_family_calibration"])
             == set(result["results"]["vitg_frozen"][
                 "per_family_calibration"]),
             "attentive frozen ViT-g comparison changed")
    return result


def run_once(*, root: Path = ROOT, device_name: str = "auto") -> dict[str, Any]:
    if result_path(root).exists() or result_path(root).is_symlink():
        return validate_result_for_consumption(root=root)
    _require(not attempt_root(root).exists()
             and not evaluation_authorisation_path(root).exists()
             and not technical_failure_path(root).exists(),
             "the sole attentive-readout attempt was consumed")
    _require(not _git_output(root, "status", "--porcelain"),
             "attentive execution requires clean committed source")
    diagnostic_contract = _load_contract(root)
    prerequisites = _load_required_diagnostics(
        diagnostic_contract, root=root)
    bundle = V13.load_preserved_encoded_training_view_for_replacement(
        root=root, verify_encoder_checkpoint=False)
    corpus = V13.corpus_from_encoded_bundle({**bundle, "root": root})
    fit_rows, data_order_witness = registered_fit_rows_and_data_order(
        corpus["fit_rows"])
    baseline = VITG.validate_reused_no_latent_baseline(corpus, root=root)
    vitg_result = _load_frozen_vitg_result(root=root)
    _require(baseline["vitl_terminal_digest"] == FROZEN_VITL_TERMINAL_DIGEST,
             "frozen ViT-L failure changed")
    initialisation = build_initialisation(
        source_path=baseline["source_vitl_initialisation_path"],
        source_sha256=baseline["source_vitl_initialisation_sha256"],
        source_state_digest=baseline["source_vitl_initial_state_digest"], root=root)
    device = VITG._device(device_name)
    model = FinalLayerAttentiveUtilityScorer()
    started = time.time()
    final_state, training = train_once(
        model, rows=fit_rows, store=corpus["horizon"],
        initialisation=initialisation, diagnostic_contract=diagnostic_contract,
        diagnostic_prerequisites=prerequisites,
        data_order_witness=data_order_witness,
        device=device, root=root)
    model.load_state_dict(final_state, strict=True)
    model.to(device)
    evaluation_auth = _signed({
        "schema": EVALUATION_AUTH_SCHEMA, "status": STATUS,
        "diagnostic_contract_digest": diagnostic_contract[
            "diagnostic_contract_digest"],
        "diagnostic_prerequisites": prerequisites,
        "final_checkpoint_sha256": training["sha256"],
        "final_state_digest": training["final_state_digest"],
        "calibration_states": 24, "calibration_rows": CALIBRATION_ROWS,
        "maximum_evaluations": 1, "evaluation_number": 1,
        "calibration_not_opened_before_authorisation": True,
    }, "evaluation_authorisation_digest")
    _publish_json_once(evaluation_authorisation_path(root), evaluation_auth,
                       "attentive evaluation authorisation")
    try:
        calibration_metrics, predictions, targets = _evaluate_streaming(
            model, rows=corpus["calibration_rows"], store=corpus["horizon"],
            device=device)
        per_family = FROZEN._grouped_calibration(
            corpus["calibration_rows"], targets, predictions, "family")
        per_stratum = FROZEN._grouped_calibration(
            corpus["calibration_rows"], targets, predictions, "stratum")
    except BaseException as exc:
        _record_technical_failure(
            stage="calibration_evaluation", error=exc, epochs=EPOCHS,
            updates=TOTAL_UPDATES, root=root)
        raise
    fit_distribution = FROZEN.label_distribution(corpus["fit_rows"])
    calibration_distribution = FROZEN.label_distribution(corpus["calibration_rows"])
    criteria, details, pairwise_gain = V13.qualification_criteria(
        calibration_metrics, baseline["metrics"], fit_distribution,
        calibration_distribution)
    safety_auc = float(calibration_metrics["safety"]["auc_any_hazard"])
    family_consistency = per_family_primary_consistency(
        attentive=per_family,
        existing_vitl=baseline["vitl_per_family_metrics"],
        baseline=baseline["per_family_metrics"])
    decision = exploratory_decision(
        safety_auc=safety_auc, pairwise_gain=pairwise_gain,
        family_consistent=family_consistency[
            "no_inconsistent_per_family_primary_improvement"])
    comparisons = {
        "attentive_minus_existing_vitl": {
            "overall": _metric_delta(
                calibration_metrics, baseline["vitl_metrics"]),
            "per_family": _metric_delta(
                per_family, baseline["vitl_per_family_metrics"]),
        },
        "attentive_minus_vitg": {
            "overall": _metric_delta(
                calibration_metrics,
                vitg_result["results"]["vitg"]["calibration"]),
            "per_family": _metric_delta(
                per_family,
                vitg_result["results"]["vitg"][
                    "per_family_calibration"]),
        },
        "attentive_minus_no_latent": {
            "overall": _metric_delta(calibration_metrics, baseline["metrics"]),
            "per_family": _metric_delta(
                per_family, baseline["per_family_metrics"]),
        },
    }
    result = _signed(FROZEN._safe_json({
        "schema": SCHEMA, "status": STATUS, "complete": True,
        "scientific_result_valid": True,
        "exploratory_not_qualification": True,
        "diagnostic_contract_digest": diagnostic_contract[
            "diagnostic_contract_digest"],
        "diagnostic_prerequisites": prerequisites,
        "official_pooler_binding_digest": OFFICIAL_POOLER_BINDING_DIGEST,
        "initialisation": {key: value for key, value in initialisation.items()
                           if key != "model_state_dict"},
        "training": training,
        "evaluation_authorisation_digest": evaluation_auth[
            "evaluation_authorisation_digest"],
        "training_execution_count": 1, "calibration_evaluation_count": 1,
        "results": {
            "attentive": {
                "calibration": calibration_metrics,
                "per_family_calibration": per_family,
                "per_stratum_calibration": per_stratum,
            },
            "existing_vitl_frozen": {
                "calibration": baseline["vitl_metrics"],
                "per_family_calibration": baseline["vitl_per_family_metrics"],
                "terminal_digest": baseline["vitl_terminal_digest"],
            },
            "vitg_frozen": {
                "result_digest": vitg_result["exploratory_result_digest"],
                "calibration": vitg_result["results"]["vitg"][
                    "calibration"],
                "per_family_calibration": vitg_result["results"]["vitg"][
                    "per_family_calibration"],
                "per_stratum_calibration": vitg_result["results"]["vitg"][
                    "per_stratum_calibration"],
                "latent_over_baseline_pairwise_gain": vitg_result[
                    "latent_over_baseline_pairwise_gain"],
                "conclusion": vitg_result["exploratory_decision"][
                    "classification"],
            },
            "no_latent_reused": {
                "calibration": baseline["metrics"],
                "per_family_calibration": baseline["per_family_metrics"],
                "checkpoint_sha256": baseline["checkpoint_sha256"],
                "state_digest": baseline["state_digest"],
                "receipt_digest": baseline["receipt_digest"],
                "retrained": False, "reevaluated": False,
            },
        },
        "latent_over_baseline_pairwise_gain": pairwise_gain,
        "metric_comparisons": comparisons,
        "per_family_primary_consistency": family_consistency,
        "frozen_original_gate_replay": {"criteria": criteria, "details": details},
        "would_meet_all_original_gates": all(criteria.values()),
        "exploratory_decision": decision,
        "technical_failures": [], "invalid_attempts": [],
        "qualified_scorer_package_published": False,
        "predictor_retrained": False,
        "predictor_checkpoints_opened_for_utility": 0,
        "predictor_utility_shards_opened": 0,
        "final_200_state_corpus_generated": False,
        "wall_time_s": round(time.time() - started, 3),
        "nothing_left_running_by_this_process_after_exit": True,
    }), "attentive_result_digest")
    _publish_json_once(result_path(root), result, "attentive result")
    return validate_result_for_consumption(root=root)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=("issue-contract", "run"))
    parser.add_argument("--device", default="auto",
                        choices=("auto", "cpu", "cuda"))
    args = parser.parse_args(argv)
    if args.stage == "issue-contract":
        result = issue_contract(root=ROOT)
        print(json.dumps({
            "status": result["status"],
            "diagnostic_contract_digest": result["diagnostic_contract_digest"],
        }, sort_keys=True))
        return 0
    result = run_once(device_name=args.device)
    print(json.dumps({
        "status": result["status"],
        "decision": result["exploratory_decision"]["classification"],
        "safety_auc": result["results"]["attentive"]["calibration"][
            "safety"]["auc_any_hazard"],
        "latent_over_baseline_pairwise_gain": result[
            "latent_over_baseline_pairwise_gain"],
        "result_digest": result["attentive_result_digest"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
