#!/usr/bin/env python3
"""Run the one-shot grounded dense-DINO joint-JEPA V1 experiment.

The runner deliberately owns all filesystem and lifecycle policy.  The model
module has no filesystem access and the benchmark module owns only pure data
projections, losses, metrics, and gates.  In particular, this file enforces
the asymmetric access order required by the preregistration::

    train receipts -> train contexts -> physical-only checkpoint
      -> train successors -> joint checkpoint -> evaluation receipts/contexts

Evaluation successors are never readable through this runner.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import copy
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
from io import BytesIO
import itertools
import json
import math
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time
from types import MappingProxyType
from typing import Any

import numpy as np
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SCHEMA = "lewm_go2_grounded_dense_dino_joint_jepa_v1_result_v1"
TERMINAL_SCHEMA = "lewm_go2_grounded_dense_dino_joint_jepa_v1_terminal_v1"
CHECKPOINT_SCHEMA = "lewm_go2_grounded_dense_dino_joint_jepa_v1_checkpoint_v1"
AUTHORITY_SCHEMA = "lewm_go2_grounded_dense_dino_joint_jepa_v1_execution_authority_v1"
AUTHORITY_STATUS = "AUTHORIZED_ONE_EXACT_GROUNDED_DENSE_DINO_JOINT_JEPA_V1"
SOURCE_REVIEW_SCHEMA = "lewm_go2_grounded_dense_dino_joint_jepa_v1_source_review_v1"
SOURCE_REVIEW_STATUS = "PASS_INDEPENDENT_SOURCE_REVIEW"

PREREGISTRATION = REPO_ROOT / (
    "docs/lewm_go2_grounded_dense_dino_joint_jepa_v1_"
    "preregistration_2026-08-04.md"
)
PREREGISTRATION_SHA256 = (
    "4bbc46aef723379392470d6e271354b4f4e50c46e57a52ad76bcdf57366daaf2"
)
PREREGISTRATION_BYTE_COUNT = 9_268
DEFAULT_OUTPUT_ROOT = REPO_ROOT / (
    ".generated/dev/go2_grounded_dense_dino_joint_jepa_v1/attempt_v1"
)
POSTHOC_ROOT = REPO_ROOT / (
    ".generated/dev/lewm-go2-wm-bounded-branch-posthoc-join-admission-v1"
)
PHYSICS_ROOT = REPO_ROOT / (
    ".generated/dev/lewm-go2-wm-bounded-branch-experiment-integrity-replacement-v1"
)

DINO_REPOSITORY_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINO_CHECKPOINT_SHA256 = (
    "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
)
DINO_CHECKPOINT_BYTE_COUNT = 88_283_115
POSTHOC_MANIFEST_SHA256 = (
    "87448995c905107453814a5e7e4cd9968d31cbc0e308513d17bc038c6585f15e"
)
POSTHOC_MANIFEST_BYTE_COUNT = 11_964
POSTHOC_MANIFEST_SCHEMA = (
    "lewm_go2_world_model_bounded_branch_posthoc_join_admission_manifest_v1"
)
POSTHOC_MANIFEST_STATUS = "COMPLETE_POSTHOC_METADATA_DERIVATION_PENDING_REVIEW"

STATE_COUNT = 128
SCENE_COUNT = 16
FAMILY_COUNT = 8
ACTION_COUNT = 9
CONTEXT_COUNT = 3
FULL_TOKEN_COUNT = 257
PATCH_TOKEN_COUNT = 256
FEATURE_DIM = 384
COMMAND_STEPS = 5
COMMAND_CHANNELS = 3
COMMAND_DIM = 15
PHYSICAL_INPUT_DIM = 12
PHYSICAL_OUTPUT_DIM = 4

MODEL_SEED = 2_026_080_405
SAMPLER_SEED = 2_026_080_406
BOOTSTRAP_SEED = 2_026_080_407
MICROBATCH_STATES = 2
ACCUMULATION_STEPS = 4
EFFECTIVE_BATCH_STATES = 8
MAX_UPDATES = 800
TRACE_UPDATES = (0, 400, 800)
EMA_MOMENTUM = 0.996

ARM_ORDER = ("physical_only_matched", "joint_jepa_grounded")
TASK_ARM = "task_action_only"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SOURCE_PATHS = {
    "model": REPO_ROOT / "lewm/models/go2_grounded_dense_dino_joint_jepa_v1.py",
    "benchmark": REPO_ROOT / "lewm/benchmarks/go2_grounded_dense_dino_joint_jepa_v1.py",
    "runner": Path(__file__).resolve(),
    "model_test": REPO_ROOT / "lewm/tests/test_go2_grounded_dense_dino_joint_jepa_v1.py",
    "benchmark_test": REPO_ROOT
    / "lewm/tests/test_go2_grounded_dense_dino_joint_jepa_v1_benchmark.py",
    "runner_test": REPO_ROOT / "lewm/tests/test_run_go2_grounded_dense_dino_joint_jepa_v1.py",
    "dense_predictor": REPO_ROOT / "lewm/models/dense_dinov2_temporal_predictor.py",
    "counterfactual_consumer": REPO_ROOT
    / "lewm/datasets/go2_world_model_counterfactual_pilot_v1.py",
    "physical_loader": REPO_ROOT
    / "lewm/benchmarks/go2_matched_branch_physical_outcome_screen_v1.py",
    "feature_plan_and_bootstrap": REPO_ROOT
    / "lewm/benchmarks/go2_dinov2_physical_readout_calibration_v1.py",
    "task_action_baseline": REPO_ROOT
    / "lewm/benchmarks/go2_dinov2_dense_shared_spatial_readout_calibration_v1.py",
    "counterfactual_contract": REPO_ROOT
    / "lewm/benchmarks/go2_world_model_counterfactual_pilot_v1.py",
    "action_regret_evaluator": REPO_ROOT
    / "scripts/evaluate_go2_world_model_counterfactual_action_regret_v1.py",
    "predictor_core": REPO_ROOT / "lewm/models/predictor.py",
    "physical_outcome_model": REPO_ROOT
    / "lewm/models/go2_matched_branch_physical_outcome_screen_v1.py",
    "dense_readout_model": REPO_ROOT
    / "lewm/models/go2_dinov2_dense_shared_spatial_readout_calibration_v1.py",
}


class GroundedRunnerError(RuntimeError):
    """The frozen runner, data, or lifecycle contract changed."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise GroundedRunnerError("value is not canonical finite JSON") from error


def _strict_json_loads(raw: bytes, *, label: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise GroundedRunnerError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise GroundedRunnerError(f"{label} contains nonfinite constant {value}")

    try:
        value = json.loads(
            raw,
            object_pairs_hook=unique,
            parse_constant=reject_constant,
        )
    except GroundedRunnerError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GroundedRunnerError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise GroundedRunnerError(f"{label} must be a JSON object")
    _canonical_bytes(value)
    return value


def _reject_protected(path: Path, *, label: str) -> None:
    """Reject sealed, held-out, and protected path components before I/O."""

    for part in PurePosixPath(path.as_posix()).parts:
        lowered = part.lower()
        if (
            lowered == "sealed_test.json"
            or lowered == "sealed"
            or lowered.startswith("sealed_")
            or lowered in {"heldout", "held_out", "held-out", "protected"}
            or lowered.startswith(("heldout_", "held_out_", "held-out-", "protected_"))
        ):
            raise GroundedRunnerError(f"{label} names protected material")


def safe_path_v1(path: Path, *, label: str, must_exist: bool = True) -> Path:
    selected = Path(os.path.abspath(os.fspath(path)))
    _reject_protected(selected, label=label)
    cursor = Path(selected.anchor)
    for part in selected.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise GroundedRunnerError(f"{label} traverses a symlink")
        if not cursor.exists():
            if must_exist:
                raise GroundedRunnerError(f"{label} is absent")
            break
    if must_exist and not selected.exists():
        raise GroundedRunnerError(f"{label} is absent")
    return selected


def file_binding_v1(path: Path) -> dict[str, Any]:
    selected = safe_path_v1(path, label="bound file")
    if not selected.is_file():
        raise GroundedRunnerError(f"bound path is not a file: {selected}")
    digest = hashlib.sha256()
    byte_count = 0
    with selected.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
    return {
        "path": str(selected),
        "sha256": digest.hexdigest(),
        "byte_count": byte_count,
    }


def _binding(path: Path, sha256: str, byte_count: int) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256,
        "byte_count": byte_count,
    }


def fixed_input_bindings_v1() -> dict[str, dict[str, Any]]:
    """Exact scientific files; state receipts are recursively bound by result."""

    values = {
        "posthoc_manifest": (
            POSTHOC_ROOT / "manifest.json",
            POSTHOC_MANIFEST_SHA256,
            POSTHOC_MANIFEST_BYTE_COUNT,
        ),
        "posthoc_terminal": (
            POSTHOC_ROOT / "terminal.json",
            "a1590fffc673f7676016bb70d4b4f5530f24b9a49bf05e84dcec6bc1756fbe56",
            1_250,
        ),
        "posthoc_terminal_review": (
            REPO_ROOT
            / "docs/lewm_go2_world_model_bounded_branch_posthoc_join_admission_v1_terminal_review_2026-08-02.json",
            "bfd0250357d0f681c674db6c54ea4a8c4d5e617230332383beda3db3e0f38669",
            2_844,
        ),
        "posthoc_rgb_manifest": (
            POSTHOC_ROOT / "rgb_manifest.json",
            "5e03afa7665ffef54a1cab5e37135a18d42761bc844ecefacaa433f75a1b1f7e",
            1_880_307,
        ),
        "posthoc_train_rows": (
            POSTHOC_ROOT / "train.jsonl",
            "edc6f88bb105c39575477fbfbb0224bf0312cf5ee3e90551f86a9c11c2ebb447",
            30_432_624,
        ),
        "posthoc_eval_rows": (
            POSTHOC_ROOT / "eval.jsonl",
            "531debbc431f2f8afc83a491b491b8822134c831b16ca4d283fe1e7f4ba07768",
            30_411_588,
        ),
        "physics_result": (
            PHYSICS_ROOT / "physics_result.json",
            "25caf0a5d4c69e99559a663aa4cae96fb23ef191ccf34486804c3f2243553314",
            183_320,
        ),
        "physics_receipt_check": (
            PHYSICS_ROOT / "physics_receipt_check.json",
            "faeb50293bc684e35b6d725b027983ad0110e739db2d7b1aca1926e89a547dc6",
            892,
        ),
        "collection_terminal": (
            PHYSICS_ROOT / "terminal_supervision.json",
            "f7d2796139645892d22ad6bb99d26caffc2b5c3dcac2a655b1883b299d22bff4",
            12_949,
        ),
        "collection_plan": (
            REPO_ROOT
            / "docs/lewm_go2_world_model_bounded_branch_integrity_replacement_v1_exact_plan_2026-08-02.json",
            "8fe34054bb9ae709b6a8ecfea5fdae55c742d1b2e22af3c289d27a77f11c66ef",
            343_973,
        ),
        "calibration_receipt": (
            REPO_ROOT
            / ".generated/dev/lewm-go2-wm-counterfactual-calibration-v3-textured-v03-posthoc-analysis-v1/calibration_receipt.json",
            "58d1291ede7ee03a93d68eb7cec80c9322c47cd0b1d5fd1c41bf8f4b49ad484e",
            72_475,
        ),
    }
    return {
        label: _binding(path, sha256, byte_count)
        for label, (path, sha256, byte_count) in values.items()
    }


def _require_binding(value: object, *, label: str, rehash: bool = True) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"path", "sha256", "byte_count"}
        or not isinstance(value.get("path"), str)
        or not isinstance(value.get("sha256"), str)
        or len(str(value["sha256"])) != 64
        or type(value.get("byte_count")) is not int
        or int(value["byte_count"]) <= 0
    ):
        raise GroundedRunnerError(f"{label} binding is malformed")
    selected = dict(value)
    _reject_protected(Path(str(selected["path"])), label=label)
    if rehash and file_binding_v1(Path(str(selected["path"]))) != selected:
        raise GroundedRunnerError(f"{label} binding changed")
    return selected


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


def _save_torch_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o644,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            torch.save(dict(value), handle)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def runner_config_v1() -> dict[str, Any]:
    return {
        "action_count": ACTION_COUNT,
        "arm_order": list(ARM_ORDER),
        "bootstrap_draws": 10_000,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "candidate_sampler_seed": SAMPLER_SEED,
        "checkpoint_updates": [400, 800],
        "ema_momentum": EMA_MOMENTUM,
        "gradient_clip_norm": 1.0,
        "infonce_temperature": 0.10,
        "infonce_weight": 0.10,
        "learning_rates": {"online_tail": 3.0e-5, "predictor_head": 3.0e-4},
        "maximum_updates": MAX_UPDATES,
        "microbatch_states": MICROBATCH_STATES,
        "accumulation_steps": ACCUMULATION_STEPS,
        "effective_batch_states": EFFECTIVE_BATCH_STATES,
        "model_seed": MODEL_SEED,
        "optimizer": {
            "name": "AdamW",
            "betas": [0.9, 0.999],
            "epsilon": 1.0e-8,
            "weight_decay": 1.0e-4,
        },
        "physical_mse_weight": 1.0,
        "physical_rank_margin_m": 0.05,
        "physical_rank_weight": 0.25,
        "trace_updates": list(TRACE_UPDATES),
        "train_state_count": STATE_COUNT,
        "eval_state_count": STATE_COUNT,
    }


def configure_determinism_v1() -> dict[str, Any]:
    """Enable the preregistered strict deterministic execution policy."""

    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        workspace = ":4096:8"
    elif workspace not in {":4096:8", ":16:8"}:
        raise GroundedRunnerError("CUBLAS_WORKSPACE_CONFIG is not deterministic")
    torch.use_deterministic_algorithms(True, warn_only=False)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = False
    enabled = bool(torch.are_deterministic_algorithms_enabled())
    warn_only = bool(torch.is_deterministic_algorithms_warn_only_enabled())
    cudnn_deterministic = bool(torch.backends.cudnn.deterministic)
    cudnn_benchmark = bool(torch.backends.cudnn.benchmark)
    if not enabled or warn_only or not cudnn_deterministic or cudnn_benchmark:
        raise GroundedRunnerError("strict deterministic algorithms were not enabled")
    return {
        "requested": "strict_deterministic_algorithms",
        "torch_deterministic_algorithms": enabled,
        "torch_deterministic_warn_only": warn_only,
        "cudnn_deterministic": cudnn_deterministic,
        "cudnn_benchmark": cudnn_benchmark,
        "cudnn_allow_tf32": bool(
            getattr(torch.backends.cudnn, "allow_tf32", False)
        ),
        "cuda_matmul_allow_tf32": bool(
            getattr(getattr(torch.backends.cuda, "matmul", object()), "allow_tf32", False)
        ),
        "cublas_workspace_config": workspace,
        "nondeterministic_operation_policy": "error",
    }


@dataclass
class AccessLedgerV1:
    """Fail-closed semantic access ledger, independent of loader internals."""

    stage: str = "created"
    physical_checkpoint_durable: bool = False
    joint_checkpoint_durable: bool = False
    receipt_loads: dict[str, int] = field(
        default_factory=lambda: {"train": 0, "eval": 0}
    )
    role_index_opens: dict[str, int] = field(
        default_factory=lambda: {"train": 0, "eval": 0}
    )
    state_receipt_opens: dict[str, int] = field(
        default_factory=lambda: {"train": 0, "eval": 0}
    )
    rgb_opens: dict[str, int] = field(
        default_factory=lambda: {
            "train_context": 0,
            "train_successor": 0,
            "eval_context": 0,
            "eval_successor": 0,
        }
    )
    opened_artifacts: set[tuple[str, str, str]] = field(default_factory=set)
    opened_receipts: set[tuple[str, str]] = field(default_factory=set)

    def load_receipts(self, role: str) -> None:
        if role == "train":
            if self.receipt_loads["train"] != 0 or self.receipt_loads["eval"] != 0:
                raise GroundedRunnerError("train receipts must be the first and sole train load")
            self.receipt_loads["train"] = 1
            self.stage = "train_receipts"
            return
        if role == "eval":
            if not (self.physical_checkpoint_durable and self.joint_checkpoint_durable):
                raise GroundedRunnerError("evaluation receipts opened before both checkpoints")
            if self.receipt_loads["eval"] != 0:
                raise GroundedRunnerError("evaluation receipts loaded more than once")
            self.receipt_loads["eval"] = 1
            self.stage = "evaluation"
            return
        raise GroundedRunnerError("unknown receipt role")

    def open_role_index(self, role: str, path: str) -> None:
        if role not in {"train", "eval"} or self.receipt_loads[role] != 1:
            raise GroundedRunnerError("role index opened outside its authorized stage")
        if self.role_index_opens[role] != 0 or not isinstance(path, str) or not path:
            raise GroundedRunnerError("role index opened more than once")
        self.role_index_opens[role] = 1

    def open_state_receipt(self, role: str, path: str) -> None:
        if role not in {"train", "eval"} or self.receipt_loads[role] != 1:
            raise GroundedRunnerError("state receipt opened outside its authorized stage")
        identity = (role, path)
        if not isinstance(path, str) or not path or identity in self.opened_receipts:
            raise GroundedRunnerError("state receipt opened more than once")
        self.opened_receipts.add(identity)
        self.state_receipt_opens[role] += 1

    def open_rgb(self, role: str, kind: str, artifact_id: str) -> None:
        if role not in {"train", "eval"} or kind not in {"context", "successor"}:
            raise GroundedRunnerError("unknown RGB role or kind")
        if not isinstance(artifact_id, str) or not artifact_id:
            raise GroundedRunnerError("RGB artifact identity is empty")
        key = f"{role}_{kind}"
        if role == "train" and kind == "context":
            if self.receipt_loads["train"] != 1 or self.physical_checkpoint_durable:
                raise GroundedRunnerError("train context RGB opened outside context stage")
        elif role == "train" and kind == "successor":
            if not self.physical_checkpoint_durable or self.joint_checkpoint_durable:
                raise GroundedRunnerError("train successor RGB opened outside joint stage")
        elif role == "eval" and kind == "context":
            if self.receipt_loads["eval"] != 1:
                raise GroundedRunnerError("evaluation context RGB opened before eval receipts")
        else:
            raise GroundedRunnerError("evaluation successor RGB is forbidden")
        identity = (role, kind, artifact_id)
        if identity in self.opened_artifacts:
            raise GroundedRunnerError("an RGB artifact was opened more than once")
        self.opened_artifacts.add(identity)
        self.rgb_opens[key] += 1

    def checkpoint(self, arm: str) -> None:
        if arm == "physical_only_matched":
            if self.physical_checkpoint_durable:
                raise GroundedRunnerError("physical checkpoint already durable")
            self.physical_checkpoint_durable = True
            self.stage = "physical_checkpoint"
        elif arm == "joint_jepa_grounded":
            if not self.physical_checkpoint_durable or self.joint_checkpoint_durable:
                raise GroundedRunnerError("joint checkpoint order changed")
            self.joint_checkpoint_durable = True
            self.stage = "joint_checkpoint"
        else:
            raise GroundedRunnerError("unknown checkpoint arm")

    def audit(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "physical_checkpoint_durable": self.physical_checkpoint_durable,
            "joint_checkpoint_durable": self.joint_checkpoint_durable,
            "receipt_loads": dict(self.receipt_loads),
            "role_index_opens": dict(self.role_index_opens),
            "state_receipt_opens": dict(self.state_receipt_opens),
            "rgb_opens": dict(self.rgb_opens),
            "unique_rgb_artifacts": len(self.opened_artifacts),
            "evaluation_successor_rgb_open_count": self.rgb_opens["eval_successor"],
        }


def finalized_access_audit_v1(
    ledger: AccessLedgerV1, *, evaluation_opened: bool
) -> dict[str, Any]:
    """Validate and return the exact successful-attempt access cardinalities."""

    expected_eval_roles = 1 if evaluation_opened else 0
    expected_rgb = {
        "train_context": STATE_COUNT * CONTEXT_COUNT,
        "train_successor": STATE_COUNT * ACTION_COUNT,
        "eval_context": STATE_COUNT * CONTEXT_COUNT if evaluation_opened else 0,
        "eval_successor": 0,
    }
    audit = ledger.audit()
    if (
        audit["physical_checkpoint_durable"] is not True
        or audit["joint_checkpoint_durable"] is not True
        or audit["receipt_loads"] != {"train": 1, "eval": expected_eval_roles}
        or audit["role_index_opens"] != {"train": 1, "eval": expected_eval_roles}
        or audit["state_receipt_opens"]
        != {"train": STATE_COUNT, "eval": STATE_COUNT * expected_eval_roles}
        or audit["rgb_opens"] != expected_rgb
        or audit["unique_rgb_artifacts"] != sum(expected_rgb.values())
        or audit["evaluation_successor_rgb_open_count"] != 0
        or audit["stage"] != ("evaluation" if evaluation_opened else "joint_checkpoint")
    ):
        raise GroundedRunnerError("final access cardinalities changed")
    return audit


@dataclass(frozen=True)
class SharedRoleMetadataV1:
    source_root: Path
    manifest: Mapping[str, Any]
    artifacts: Mapping[str, Any]
    tolerances: Mapping[str, Any]
    requested_blocks: tuple[Any, ...]
    receipt_bindings: Mapping[str, tuple[Mapping[str, Any], ...]]


@dataclass(frozen=True)
class RoleRuntimeDataV1:
    role: str
    bundle: Any
    plan: Any
    physical_inputs: torch.Tensor
    targets: torch.Tensor
    history_commands: torch.Tensor
    candidate_commands: torch.Tensor
    relative_goals: torch.Tensor
    dense_ranks: torch.Tensor
    context_artifact_ids: tuple[tuple[str, str, str], ...]
    successor_artifact_ids: tuple[tuple[str, ...], ...]
    identity_sha256: str


@dataclass(frozen=True)
class MicrobatchSelectionV1:
    state_indices: torch.Tensor
    candidate_action_ids: torch.Tensor


def _read_bound_json_file_v1(binding: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    return _strict_json_loads(_read_bound_bytes_once_v1(binding, label=label), label=label)


def _read_bound_bytes_once_v1(
    binding: Mapping[str, Any], *, label: str
) -> bytes:
    """Read one bound file once while validating its exact bytes.

    Role access receipts count semantic file opens.  Hashing and then reading a
    role file would silently perform two opens, so the production loader hashes
    the bytes obtained by its sole semantic read instead.
    """

    normalized = _require_binding(binding, label=label, rehash=False)
    selected = safe_path_v1(Path(str(normalized["path"])), label=label)
    if not selected.is_file():
        raise GroundedRunnerError(f"{label} is not a regular file")
    try:
        raw = selected.read_bytes()
    except OSError as error:
        raise GroundedRunnerError(f"cannot read {label}") from error
    if (
        len(raw) != int(normalized["byte_count"])
        or hashlib.sha256(raw).hexdigest() != normalized["sha256"]
    ):
        raise GroundedRunnerError(f"{label} binding changed")
    return raw


def _read_role_jsonl_v1(authority: Mapping[str, Any], *, role: str) -> list[dict[str, Any]]:
    if role not in {"train", "eval"}:
        raise GroundedRunnerError("JSONL role changed")
    lines = _read_bound_bytes_once_v1(
        authority["input_bindings"][f"posthoc_{role}_rows"],
        label=f"{role} role rows",
    ).splitlines()
    rows = [
        _strict_json_loads(line, label=f"{role} row {index}")
        for index, line in enumerate(lines, start=1)
    ]
    if len(rows) != STATE_COUNT:
        raise GroundedRunnerError(f"{role} role row count changed")
    return rows


def _consumer_compatible_sync_document_v1(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only redundant zero-valued synchronization statistics."""

    result = copy.deepcopy(dict(value))
    if "document" in result and isinstance(result["document"], Mapping):
        result = copy.deepcopy(dict(result["document"]))
    sync = result.get("synchronization_audit")
    components = sync.get("components") if isinstance(sync, Mapping) else None
    if not isinstance(sync, dict) or not isinstance(components, Mapping):
        raise GroundedRunnerError("synchronization components are absent")
    projected: dict[str, Any] = {}
    for name, component in components.items():
        if (
            not isinstance(component, Mapping)
            or set(component)
            != {
                "exact_equal",
                "max_abs_difference",
                "per_lane_max_abs_difference",
                "rms_difference",
                "shape_per_lane",
            }
            or component.get("exact_equal") is not True
            or float(component.get("max_abs_difference", -1.0)) != 0.0
            or float(component.get("rms_difference", -1.0)) != 0.0
            or not isinstance(component.get("per_lane_max_abs_difference"), list)
            or len(component["per_lane_max_abs_difference"]) != ACTION_COUNT
            or any(float(item) != 0.0 for item in component["per_lane_max_abs_difference"])
        ):
            raise GroundedRunnerError("synchronization diagnostic is not redundant")
        projected[str(name)] = {
            "exact_equal": True,
            "max_abs_difference": 0.0,
            "shape_per_lane": list(component["shape_per_lane"]),
        }
    sync["components"] = projected
    return result


def _load_shared_role_metadata_v1(authority: Mapping[str, Any]) -> SharedRoleMetadataV1:
    """Load common metadata, but no train/eval role row or state receipt."""

    from lewm.datasets import go2_world_model_counterfactual_pilot_v1 as consumer
    manifest = _read_bound_json_file_v1(
        authority["input_bindings"]["posthoc_manifest"], label="posthoc manifest"
    )
    if (
        manifest.get("schema") != POSTHOC_MANIFEST_SCHEMA
        or manifest.get("status") != POSTHOC_MANIFEST_STATUS
        or manifest.get("derived_output_root") != str(POSTHOC_ROOT.resolve())
        or manifest.get("rgb_artifacts") != 2 * STATE_COUNT * (CONTEXT_COUNT + ACTION_COUNT)
        or manifest.get("role_scene_counts") != {"train": SCENE_COUNT, "eval": SCENE_COUNT}
    ):
        raise GroundedRunnerError("posthoc manifest contract changed")
    leaves = manifest.get("derived_leaf_bindings")
    if not isinstance(leaves, Mapping):
        raise GroundedRunnerError("posthoc derived leaves are absent")
    for leaf, label in (
        ("rgb_manifest", "posthoc_rgb_manifest"),
        ("train", "posthoc_train_rows"),
        ("eval", "posthoc_eval_rows"),
    ):
        value = leaves.get(leaf)
        if not isinstance(value, Mapping):
            raise GroundedRunnerError(f"posthoc {leaf} binding is absent")
        normalized = {
            "path": str((POSTHOC_ROOT / str(value.get("path"))).resolve())
            if not Path(str(value.get("path"))).is_absolute()
            else str(Path(str(value.get("path"))).resolve()),
            "sha256": value.get("sha256", value.get("file_sha256")),
            "byte_count": value.get("byte_count"),
        }
        if normalized != authority["input_bindings"][label]:
            raise GroundedRunnerError(f"posthoc {leaf} binding changed")
    source_root = safe_path_v1(
        Path(str(manifest.get("source_receipt_root"))), label="source receipt root"
    )
    if source_root != PHYSICS_ROOT.resolve() or not source_root.is_dir():
        raise GroundedRunnerError("source receipt root changed")

    rgb_document = _read_bound_json_file_v1(
        authority["input_bindings"]["posthoc_rgb_manifest"], label="RGB manifest"
    )
    values = rgb_document.get("artifacts")
    if rgb_document.get("schema") != consumer.RGB_MANIFEST_SCHEMA or not isinstance(
        values, list
    ) or len(values) != 2 * STATE_COUNT * (CONTEXT_COUNT + ACTION_COUNT):
        raise GroundedRunnerError("RGB manifest changed")
    artifacts: dict[str, Any] = {}
    for value in values:
        if not isinstance(value, Mapping):
            raise GroundedRunnerError("RGB artifact entry changed")
        artifact_id = value.get("artifact_id")
        relative = PurePosixPath(str(value.get("path")))
        if (
            not isinstance(artifact_id, str)
            or not artifact_id
            or artifact_id in artifacts
            or relative.is_absolute()
            or ".." in relative.parts
            or value.get("width") != 224
            or value.get("height") != 224
            or value.get("mode") != "RGB"
            or value.get("format") != "PNG"
            or value.get("camera_valid") is not True
        ):
            raise GroundedRunnerError("RGB artifact metadata changed")
        _reject_protected(source_root.joinpath(*relative.parts), label="RGB artifact")
        artifacts[artifact_id] = consumer.RGBArtifactV1(
            artifact_id=artifact_id,
            frame_identity=str(value["frame_identity"]),
            relative_path=relative.as_posix(),
            byte_count=int(value["byte_count"]),
            file_sha256=str(value["file_sha256"]),
            pixel_sha256=str(value["pixel_sha256"]),
            low_information=bool(value["low_information"]),
            low_info_reasons=tuple(str(item) for item in value["low_info_reasons"]),
        )
    try:
        _excluded, tolerances = consumer._validate_calibration_contract(  # noqa: SLF001
            manifest["calibration_contract"], textured_v03=True
        )
        requested_blocks = tuple(
            consumer._validate_tape(item["requested_block"], name="requested action")  # noqa: SLF001
            for item in manifest["action_catalog"]
        )
    except Exception as error:
        raise GroundedRunnerError("posthoc calibration metadata changed") from error
    if len(requested_blocks) != ACTION_COUNT:
        raise GroundedRunnerError("posthoc action catalog changed")

    physics = _read_bound_json_file_v1(
        authority["input_bindings"]["physics_result"], label="physics result"
    )
    raw_bindings = physics.get("state_receipt_bindings")
    if physics.get("status") != "PHYSICS_COMPLETE" or not isinstance(
        raw_bindings, list
    ) or len(raw_bindings) != 2 * STATE_COUNT:
        raise GroundedRunnerError("physics receipt closure changed")
    selected: dict[str, list[Mapping[str, Any]]] = {"train": [], "eval": []}
    ordered_roles: list[str] = []
    for index, value in enumerate(raw_bindings):
        if not isinstance(value, Mapping):
            raise GroundedRunnerError("state receipt binding changed")
        relative = PurePosixPath(str(value.get("path")))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise GroundedRunnerError("state receipt path is not canonical relative POSIX")
        roles = [role for role in ("train", "eval") if ("scenes", role) == tuple(relative.parts[:2])]
        if not roles:
            # Some receipts prefix an output folder before ``scenes``.
            roles = [
                role
                for role in ("train", "eval")
                if any(
                    relative.parts[offset : offset + 2] == ("scenes", role)
                    for offset in range(max(0, len(relative.parts) - 1))
                )
            ]
        if len(roles) != 1:
            raise GroundedRunnerError(f"state receipt {index} role is ambiguous")
        role = roles[0]
        absolute = safe_path_v1(
            PHYSICS_ROOT.joinpath(*relative.parts),
            label=f"{role} state receipt",
        )
        normalized = {
            "path": str(absolute),
            "sha256": value.get("sha256", value.get("file_sha256")),
            "byte_count": value.get("byte_count"),
        }
        _require_binding(normalized, label=f"{role} receipt declaration", rehash=False)
        selected[role].append(normalized)
        ordered_roles.append(role)
    if (
        ordered_roles != ["train"] * STATE_COUNT + ["eval"] * STATE_COUNT
        or any(len(selected[role]) != STATE_COUNT for role in selected)
    ):
        raise GroundedRunnerError("state receipts are not ordered train then eval")
    return SharedRoleMetadataV1(
        source_root=source_root,
        manifest=MappingProxyType(dict(manifest)),
        artifacts=MappingProxyType(artifacts),
        tolerances=MappingProxyType(dict(tolerances)),
        requested_blocks=requested_blocks,
        receipt_bindings=MappingProxyType(
            {role: tuple(values) for role, values in selected.items()}
        ),
    )


def _role_identity_v1(
    role: str,
    plan: Any,
    physical_inputs: torch.Tensor,
    targets: torch.Tensor,
    history_commands: torch.Tensor,
    candidate_commands: torch.Tensor,
) -> str:
    payload = {
        "role": role,
        "plan": str(plan.identity_sha256),
        "physical_inputs": _tensor_sha256(physical_inputs),
        "targets": _tensor_sha256(targets),
        "history_commands": _tensor_sha256(history_commands),
        "candidate_commands": _tensor_sha256(candidate_commands),
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def load_role_runtime_data_v1(
    authority: Mapping[str, Any],
    shared: SharedRoleMetadataV1,
    *,
    role: str,
    ledger: AccessLedgerV1,
) -> RoleRuntimeDataV1:
    """Open and derive exactly one role's rows and raw state receipts."""

    from lewm.benchmarks import go2_matched_branch_physical_outcome_screen_v1 as physical
    from lewm.datasets import go2_world_model_counterfactual_pilot_v1 as consumer

    if role not in {"train", "eval"}:
        raise GroundedRunnerError("unknown role")
    role_binding = authority["input_bindings"][f"posthoc_{role}_rows"]
    ledger.open_role_index(role, str(role_binding["path"]))
    rows = _read_role_jsonl_v1(authority, role=role)
    receipts: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for index, binding in enumerate(shared.receipt_bindings[role]):
        ledger.open_state_receipt(role, str(binding["path"]))
        document = _read_bound_json_file_v1(binding, label=f"{role} state receipt {index}")
        compatible = _consumer_compatible_sync_document_v1(document)
        state = compatible.get("state")
        state_id = state.get("state_id") if isinstance(state, Mapping) else None
        if not isinstance(state_id, str) or not state_id or state_id in by_id:
            raise GroundedRunnerError(f"{role} state receipt identity changed")
        receipts.append(compatible)
        by_id[state_id] = compatible
    groups = []
    for row in rows:
        state_id = row.get("state_id")
        if not isinstance(state_id, str) or state_id not in by_id:
            raise GroundedRunnerError(f"{role} joined row has no state receipt")
        groups.append(
            consumer._parse_group(  # noqa: SLF001
                _consumer_compatible_sync_document_v1(row),
                role=role,
                artifacts=shared.artifacts,
                tolerances=shared.tolerances,
                requested_blocks=shared.requested_blocks,
                collection_state=by_id[state_id],
            )
        )
    if len(groups) != STATE_COUNT:
        raise GroundedRunnerError(f"{role} group count changed")
    physical_groups, physical_by_id = physical._groups_from_receipts(  # noqa: SLF001
        receipts, role=role
    )
    plan = physical.prior.build_role_feature_plan_v1(physical_groups, role=role)
    expected_plan = (
        physical.EXPECTED_TRAIN_PLAN_IDENTITY
        if role == "train"
        else physical.EXPECTED_EVAL_PLAN_IDENTITY
    )
    if plan.identity_sha256 != expected_plan:
        raise GroundedRunnerError(f"{role} feature-plan identity changed")
    physical_inputs, targets = physical._role_arrays(plan, physical_by_id)  # noqa: SLF001
    histories: list[torch.Tensor] = []
    candidates: list[torch.Tensor] = []
    for state in plan.states:
        receipt = physical_by_id[state.state_id]
        history_blocks = receipt["context"].get("history_executed_blocks")
        branches = receipt.get("branches")
        if not isinstance(history_blocks, list) or len(history_blocks) != 2:
            raise GroundedRunnerError("history command blocks changed")
        if not isinstance(branches, list) or len(branches) != ACTION_COUNT:
            raise GroundedRunnerError("candidate command blocks changed")
        histories.append(
            torch.stack([command_tape_channel_major_v1(block) for block in history_blocks])
        )
        candidates.append(
            torch.stack(
                [
                    command_tape_channel_major_v1(branch.get("requested_block"))
                    for branch in branches
                ]
            )
        )
    history_commands = torch.stack(histories)
    candidate_commands = torch.stack(candidates)
    goals = torch.tensor(
        [state.relative_target_xy_body_m for state in plan.states], dtype=torch.float32
    )
    ranks = torch.tensor([state.dense_ranks for state in plan.states], dtype=torch.long)
    context_ids = tuple(
        tuple(plan.artifact_ids[index] for index in state.context_artifact_indices)
        for state in plan.states
    )
    successor_ids = tuple(
        tuple(plan.artifact_ids[index] for index in state.target_artifact_indices)
        for state in plan.states
    )
    parsed_ids = {group.state_id for group in groups}
    if parsed_ids != {state.state_id for state in plan.states}:
        raise GroundedRunnerError(f"{role} joined rows and receipts disagree")
    bundle = consumer.CounterfactualPilotBundleV1(
        root=shared.source_root,
        manifest_binding=MappingProxyType(dict(authority["input_bindings"]["posthoc_manifest"])),
        manifest=shared.manifest,
        rgb_manifest_binding=MappingProxyType(
            dict(authority["input_bindings"]["posthoc_rgb_manifest"])
        ),
        artifacts=shared.artifacts,
        groups_by_role=MappingProxyType({role: tuple(groups)}),
        role_bindings=MappingProxyType(
            {role: MappingProxyType(dict(authority["input_bindings"][f"posthoc_{role}_rows"]))}
        ),
        calibration_receipt=MappingProxyType({}),
        calibration_tolerances=shared.tolerances,
        access_audit=MappingProxyType(
            {
                "role": role,
                "role_index_open_count": 1,
                "state_receipt_open_count": STATE_COUNT,
                "rgb_leaf_open_count": 0,
                "role_filtered": True,
            }
        ),
    )
    identity = _role_identity_v1(
        role,
        plan,
        physical_inputs,
        targets,
        history_commands,
        candidate_commands,
    )
    return RoleRuntimeDataV1(
        role=role,
        bundle=bundle,
        plan=plan,
        physical_inputs=physical_inputs,
        targets=targets,
        history_commands=history_commands,
        candidate_commands=candidate_commands,
        relative_goals=goals,
        dense_ranks=ranks,
        context_artifact_ids=context_ids,
        successor_artifact_ids=successor_ids,
        identity_sha256=identity,
    )


def command_tape_channel_major_v1(command_block: object) -> torch.Tensor:
    """Convert one exact [5,time x 3,channel] command block to 15 channels."""

    value = torch.as_tensor(command_block, dtype=torch.float32)
    if tuple(value.shape) != (COMMAND_STEPS, COMMAND_CHANNELS):
        raise GroundedRunnerError("command block must have exact shape [5,3]")
    if not bool(torch.isfinite(value).all()):
        raise GroundedRunnerError("command block is nonfinite")
    return value.transpose(0, 1).contiguous().reshape(COMMAND_DIM)


def optimizer_microbatches_v1(
    *,
    state_count: int = STATE_COUNT,
    updates: int = MAX_UPDATES,
    seed: int = SAMPLER_SEED,
) -> tuple[tuple[MicrobatchSelectionV1, ...], ...]:
    """Return deterministic complete-state microbatches grouped by update."""

    if (
        type(state_count) is not int
        or state_count < EFFECTIVE_BATCH_STATES
        or state_count % MICROBATCH_STATES != 0
        or type(updates) is not int
        or updates < 1
        or type(seed) is not int
    ):
        raise GroundedRunnerError("sampler configuration is invalid")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    pending: list[int] = []
    result: list[tuple[MicrobatchSelectionV1, ...]] = []
    for _update in range(updates):
        while len(pending) < EFFECTIVE_BATCH_STATES:
            pending.extend(torch.randperm(state_count, generator=generator).tolist())
        selected = pending[:EFFECTIVE_BATCH_STATES]
        del pending[:EFFECTIVE_BATCH_STATES]
        if len(set(selected)) != EFFECTIVE_BATCH_STATES:
            # The registered 128/8 geometry never crosses an epoch boundary;
            # fail rather than silently allow duplicate states in one update.
            raise GroundedRunnerError("one optimizer update contains duplicate states")
        microbatches = []
        for offset in range(0, EFFECTIVE_BATCH_STATES, MICROBATCH_STATES):
            states = torch.tensor(
                selected[offset : offset + MICROBATCH_STATES], dtype=torch.long
            )
            candidates = torch.stack(
                [torch.randperm(ACTION_COUNT, generator=generator) for _ in range(MICROBATCH_STATES)]
            )
            microbatches.append(
                MicrobatchSelectionV1(
                    state_indices=states,
                    candidate_action_ids=candidates,
                )
            )
        result.append(tuple(microbatches))
    return tuple(result)


def optimizer_schedule_identity_v1(
    schedule: Sequence[Sequence[MicrobatchSelectionV1]],
) -> str:
    """Hash the exact state and candidate permutation tape used by both arms."""

    digest = hashlib.sha256()
    digest.update(b"lewm_go2_grounded_dense_dino_optimizer_schedule_v1\0")
    for update, microbatches in enumerate(schedule, start=1):
        digest.update(np.asarray([update, len(microbatches)], dtype="<i8").tobytes())
        for selection in microbatches:
            for value in (selection.state_indices, selection.candidate_action_ids):
                canonical = value.detach().cpu().to(torch.int64).contiguous().numpy()
                digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
                digest.update(canonical.astype("<i8", copy=False).tobytes())
    return digest.hexdigest()


def train_only_futility_v1(trace_0: Mapping[str, Any], trace_400: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the exact conjunctive update-400 joint-arm futility rule."""

    required = {
        "normalized_physical_rank_regret",
        "branch_retrieval_accuracy",
        "successor_cosine_error",
        "persistence_cosine_error",
        "all_finite",
    }
    if set(trace_0) != required or set(trace_400) != required:
        raise GroundedRunnerError("futility trace fields changed")
    values = [
        float(trace_0[name])
        for name in required - {"all_finite"}
    ] + [float(trace_400[name]) for name in required - {"all_finite"}]
    if not all(math.isfinite(value) for value in values):
        raise GroundedRunnerError("futility trace is nonfinite")
    persistence = float(trace_400["persistence_cosine_error"])
    ratio = (
        float(trace_400["successor_cosine_error"]) / persistence
        if persistence > 0.0
        else math.inf
    )
    criteria = {
        "all_finite": trace_0["all_finite"] is True and trace_400["all_finite"] is True,
        "rank_regret_improvement_at_least_0_03": (
            float(trace_0["normalized_physical_rank_regret"])
            - float(trace_400["normalized_physical_rank_regret"])
            >= 0.03
        ),
        "retrieval_at_least_0_35": float(trace_400["branch_retrieval_accuracy"]) >= 0.35,
        "retrieval_gain_at_least_0_15": (
            float(trace_400["branch_retrieval_accuracy"])
            - float(trace_0["branch_retrieval_accuracy"])
            >= 0.15
        ),
        "cosine_to_persistence_ratio_at_most_0_90": ratio <= 0.90,
    }
    return {
        "criteria": criteria,
        "successor_to_persistence_ratio": ratio,
        "continue_to_update_800": all(criteria.values()),
    }


def preprocess_rgb_bytes_v1(raw: bytes) -> torch.Tensor:
    if not isinstance(raw, bytes) or not raw:
        raise GroundedRunnerError("bound RGB reader returned no bytes")
    try:
        with Image.open(BytesIO(raw)) as image:
            if image.format != "PNG":
                raise GroundedRunnerError("bound RGB artifact is not PNG")
            rgb = image.convert("RGB")
            if rgb.size != (224, 224):
                raise GroundedRunnerError("bound RGB artifact is not 224x224")
            array = np.asarray(rgb, dtype=np.uint8).copy()
    except GroundedRunnerError:
        raise
    except Exception as error:
        raise GroundedRunnerError("bound RGB artifact cannot be decoded") from error
    tensor = torch.from_numpy(array).permute(2, 0, 1).to(torch.float32).div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32)[:, None, None]
    result = (tensor - mean) / std
    if tuple(result.shape) != (3, 224, 224) or not bool(torch.isfinite(result).all()):
        raise GroundedRunnerError("preprocessed RGB tensor is invalid")
    return result


class FrozenDINOTrunkV1:
    """The exact frozen DINO patch/position/blocks-0--9 boundary."""

    def __init__(self, dino: torch.nn.Module, *, device: torch.device) -> None:
        if not hasattr(dino, "prepare_tokens_with_masks") or not hasattr(dino, "blocks"):
            raise GroundedRunnerError("DINO model lacks the required trunk interface")
        blocks = tuple(dino.blocks)
        if len(blocks) != 12 or not hasattr(dino, "norm"):
            raise GroundedRunnerError("DINO vits14 must expose exactly twelve blocks and norm")
        self.dino = dino.to(device).eval()
        self.device = device
        self.dino.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 224, 224):
            raise GroundedRunnerError("DINO trunk input shape changed")
        hidden = self.dino.prepare_tokens_with_masks(images.to(self.device, torch.float32))
        for block in tuple(self.dino.blocks)[:10]:
            hidden = block(hidden)
        if (
            tuple(hidden.shape[1:]) != (FULL_TOKEN_COUNT, FEATURE_DIM)
            or hidden.dtype != torch.float32
            or not bool(torch.isfinite(hidden).all())
        ):
            raise GroundedRunnerError("DINO frozen trunk output changed")
        return hidden.detach().cpu()

    def fresh_tail(self) -> tuple[list[torch.nn.Module], torch.nn.Module]:
        return (
            [deepcopy(block).cpu() for block in tuple(self.dino.blocks)[10:12]],
            deepcopy(self.dino.norm).cpu(),
        )


def load_dino_trunk_v1(
    repository: Path,
    checkpoint: Path,
    *,
    device: torch.device,
) -> FrozenDINOTrunkV1:
    repo = safe_path_v1(repository, label="DINO repository")
    weights_path = safe_path_v1(checkpoint, label="DINO checkpoint")
    if not repo.is_dir() or not weights_path.is_file():
        raise GroundedRunnerError("DINO source paths have the wrong type")
    observed_commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed_commit != DINO_REPOSITORY_COMMIT:
        raise GroundedRunnerError("DINO repository commit changed")
    repository_status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if repository_status:
        raise GroundedRunnerError("DINO repository working tree is not clean")
    if file_binding_v1(weights_path) != _binding(
        weights_path, DINO_CHECKPOINT_SHA256, DINO_CHECKPOINT_BYTE_COUNT
    ):
        raise GroundedRunnerError("DINO checkpoint binding changed")
    model = torch.hub.load(str(repo), "dinov2_vits14", source="local", pretrained=False)
    payload = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(payload, Mapping) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, Mapping):
        raise GroundedRunnerError("DINO checkpoint state is malformed")
    model.load_state_dict(payload, strict=True)
    return FrozenDINOTrunkV1(model, device=device)


def precompute_trunks_v1(
    artifact_ids: Sequence[str],
    *,
    role: str,
    kind: str,
    ledger: AccessLedgerV1,
    bound_reader: Callable[[str], bytes],
    trunk: FrozenDINOTrunkV1,
    batch_size: int = 16,
) -> torch.Tensor:
    if not artifact_ids or len(set(artifact_ids)) != len(artifact_ids):
        raise GroundedRunnerError("trunk artifact list is empty or contains duplicates")
    outputs: list[torch.Tensor] = []
    for start in range(0, len(artifact_ids), batch_size):
        selected = artifact_ids[start : start + batch_size]
        images = []
        for artifact_id in selected:
            ledger.open_rgb(role, kind, artifact_id)
            images.append(preprocess_rgb_bytes_v1(bound_reader(artifact_id)))
        outputs.append(trunk.encode(torch.stack(images)))
    result = torch.cat(outputs, dim=0)
    if (
        tuple(result.shape) != (len(artifact_ids), FULL_TOKEN_COUNT, FEATURE_DIM)
        or result.dtype != torch.float32
        or not bool(torch.isfinite(result).all())
    ):
        raise GroundedRunnerError("attempt-local trunk tensor is invalid")
    return result


def _state_dict_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu() for name, tensor in module.state_dict().items()}


def _tensor_sha256(value: torch.Tensor) -> str:
    selected = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(selected.dtype).encode("ascii"))
    digest.update(np.asarray(selected.shape, dtype=np.int64).tobytes())
    digest.update(selected.numpy().tobytes())
    return digest.hexdigest()


def model_state_identity_v1(module: torch.nn.Module) -> str:
    """Canonical identity for paired arm initialization and replay checks."""

    digest = hashlib.sha256()
    state = module.state_dict()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint_v1(
    path: Path,
    *,
    arm: str,
    update: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_identity: str,
    trace: Sequence[Mapping[str, Any]],
    input_statistics: Mapping[str, object],
    outcome_statistics: Mapping[str, object],
    initial_model_identity_sha256: str,
) -> dict[str, Any]:
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "arm": arm,
        "update": update,
        "model_seed": MODEL_SEED,
        "sampler_seed": SAMPLER_SEED,
        "config": runner_config_v1(),
        "train_identity_sha256": train_identity,
        "initial_model_identity_sha256": initial_model_identity_sha256,
        "input_statistics": dict(input_statistics),
        "outcome_statistics": dict(outcome_statistics),
        "model_state_dict": _state_dict_cpu(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "trace": [dict(item) for item in trace],
    }
    _save_torch_exclusive(path, payload)
    binding = file_binding_v1(path)
    reopened = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(reopened, Mapping)
        or reopened.get("schema") != CHECKPOINT_SCHEMA
        or reopened.get("arm") != arm
        or reopened.get("update") != update
        or reopened.get("train_identity_sha256") != train_identity
    ):
        raise GroundedRunnerError("checkpoint round-trip validation failed")
    return binding


def _optimizer_v1(model: torch.nn.Module) -> torch.optim.AdamW:
    tail = list(model.online_tail.parameters())
    core = list(model.predictor.parameters()) + list(model.physical_head.parameters())
    target = {id(parameter) for parameter in model.target_tail.parameters()}
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    supplied = {id(parameter) for parameter in (*tail, *core)}
    if (
        not tail
        or not core
        or len(supplied) != len(tail) + len(core)
        or supplied != trainable
        or supplied & target
    ):
        raise GroundedRunnerError("optimizer parameter inventory changed")
    return torch.optim.AdamW(
        [
            {"params": tail, "lr": 3.0e-5},
            {"params": core, "lr": 3.0e-4},
        ],
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=1.0e-4,
        amsgrad=False,
        foreach=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )


@torch.no_grad()
def ema_last_frame_persistence_v1(
    model: torch.nn.Module,
    context_trunk_tokens: torch.Tensor,
    *,
    action_count: int = ACTION_COUNT,
) -> torch.Tensor:
    """Encode last-frame persistence in the same detached EMA target space."""

    if (
        not isinstance(context_trunk_tokens, torch.Tensor)
        or context_trunk_tokens.ndim != 4
        or tuple(context_trunk_tokens.shape[1:])
        != (CONTEXT_COUNT, FULL_TOKEN_COUNT, FEATURE_DIM)
        or type(action_count) is not int
        or action_count < 1
    ):
        raise GroundedRunnerError("persistence context geometry changed")
    encoded = model.encode_target(context_trunk_tokens[:, -1:].contiguous())
    expected = (
        int(context_trunk_tokens.shape[0]),
        1,
        PATCH_TOKEN_COUNT,
        FEATURE_DIM,
    )
    if tuple(encoded.shape) != expected or encoded.requires_grad:
        raise GroundedRunnerError("EMA persistence encoding contract changed")
    return encoded.expand(-1, action_count, -1, -1)


@torch.no_grad()
def evaluate_train_trace_v1(
    *,
    model: torch.nn.Module,
    role: RoleRuntimeDataV1,
    context_trunks: torch.Tensor,
    successor_trunks: torch.Tensor | None,
    input_statistics: Mapping[str, object],
    outcome_statistics: Mapping[str, object],
    benchmark: Any,
    device: torch.device,
    batch_size: int = 2,
) -> dict[str, Any]:
    """Evaluate train capacity without selecting or mutating a checkpoint."""

    model.eval()
    decoded_rows: list[torch.Tensor] = []
    cosine_sum = 0.0
    persistence_sum = 0.0
    retrieval_sum = 0.0
    jepa_states = 0
    for start in range(0, STATE_COUNT, batch_size):
        stop = min(start + batch_size, STATE_COUNT)
        context = context_trunks[start:stop].to(device)
        history = role.history_commands[start:stop].to(device)
        candidates = role.candidate_commands[start:stop].to(device)
        raw_physical = role.physical_inputs[start:stop].to(device)
        normalized = benchmark.normalize_physical_inputs_v1(
            raw_physical, input_statistics
        )
        prediction = model(context, history, candidates, normalized)
        decoded = benchmark.decode_standardized_outcomes_v1(
            prediction.standardized_physical_residuals, outcome_statistics
        )
        decoded_rows.append(decoded.cpu())
        if successor_trunks is not None:
            target = model.encode_target(successor_trunks[start:stop].to(device))
            count = stop - start
            cosine_sum += float(
                benchmark.dense_patch_cosine_loss_v1(
                    prediction.successor_tokens, target
                )
            ) * count
            retrieval_sum += float(
                benchmark.true_successor_branch_retrieval_v1(
                    prediction.successor_tokens, target
                )
            ) * count
            persistence = ema_last_frame_persistence_v1(model, context)
            persistence_sum += float(
                benchmark.dense_patch_cosine_loss_v1(persistence, target)
            ) * count
            jepa_states += count
    outcomes = torch.cat(decoded_rows)
    scores = benchmark.physical_score_matrix_v1(role.plan, outcomes)
    report = benchmark.report_physical_scores_v1(role.plan, scores)
    regret = float(report["summary"]["normalized_rank_regret"])
    if successor_trunks is None:
        cosine = persistence = retrieval = 0.0
    else:
        if jepa_states != STATE_COUNT:
            raise GroundedRunnerError("train JEPA trace state count changed")
        cosine = cosine_sum / jepa_states
        persistence = persistence_sum / jepa_states
        retrieval = retrieval_sum / jepa_states
    values = (regret, cosine, persistence, retrieval)
    return {
        "normalized_physical_rank_regret": regret,
        "branch_retrieval_accuracy": retrieval,
        "successor_cosine_error": cosine,
        "persistence_cosine_error": persistence,
        "all_finite": all(math.isfinite(value) for value in values),
    }


def train_arm_v1(
    *,
    arm: str,
    model: torch.nn.Module,
    role: RoleRuntimeDataV1,
    context_trunks: torch.Tensor,
    successor_trunks: torch.Tensor | None,
    input_statistics: Mapping[str, object],
    outcome_statistics: Mapping[str, object],
    optimizer_microbatches: Sequence[Sequence[MicrobatchSelectionV1]],
    benchmark: Any,
    device: torch.device,
) -> dict[str, Any]:
    if arm not in ARM_ORDER:
        raise GroundedRunnerError("unknown learned arm")
    joint = arm == "joint_jepa_grounded"
    if joint != (successor_trunks is not None):
        raise GroundedRunnerError("successor trunk access does not match arm")
    if len(optimizer_microbatches) != MAX_UPDATES:
        raise GroundedRunnerError("optimizer sampler length changed")
    optimizer = _optimizer_v1(model)
    standardized_targets = benchmark.standardize_outcome_residuals_v1(
        role.targets, outcome_statistics
    )
    trace = [
        {
            "update": 0,
            **evaluate_train_trace_v1(
                model=model,
                role=role,
                context_trunks=context_trunks,
                successor_trunks=successor_trunks,
                input_statistics=input_statistics,
                outcome_statistics=outcome_statistics,
                benchmark=benchmark,
                device=device,
            ),
        }
    ]
    last_objective: dict[str, float] = {}
    started = time.perf_counter()
    final_update = MAX_UPDATES
    futility: dict[str, Any] | None = None
    for update, microbatches in enumerate(optimizer_microbatches, start=1):
        if len(microbatches) != ACCUMULATION_STEPS:
            raise GroundedRunnerError("gradient accumulation count changed")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        components = {
            "physical_mse": 0.0,
            "physical_rank": 0.0,
            "dense_cosine": 0.0,
            "infonce": 0.0,
            "total": 0.0,
        }
        for selection in microbatches:
            indices = selection.state_indices
            action_ids = selection.candidate_action_ids
            if (
                tuple(indices.shape) != (MICROBATCH_STATES,)
                or tuple(action_ids.shape) != (MICROBATCH_STATES, ACTION_COUNT)
                or not torch.equal(
                    torch.sort(action_ids, dim=1).values,
                    torch.arange(ACTION_COUNT).expand(MICROBATCH_STATES, -1),
                )
            ):
                raise GroundedRunnerError("microbatch state geometry changed")
            context = context_trunks[indices].to(device)
            history = role.history_commands[indices].to(device)
            row = torch.arange(MICROBATCH_STATES)[:, None]
            candidates = role.candidate_commands[indices][row, action_ids].to(device)
            raw_physical = role.physical_inputs[indices][row, action_ids].to(device)
            device_action_ids = action_ids.to(device)
            normalized = benchmark.normalize_physical_inputs_v1(
                raw_physical, input_statistics
            )
            prediction = model(context, history, candidates, normalized)
            target_residual = standardized_targets[indices][row, action_ids].to(device)
            physical_mse = torch.mean(
                (prediction.standardized_physical_residuals - target_residual).square()
            )
            decoded = benchmark.decode_standardized_outcomes_v1(
                prediction.standardized_physical_residuals,
                outcome_statistics,
                action_ids=device_action_ids,
            )
            costs = benchmark.predicted_physical_cost_v1(
                decoded, role.relative_goals[indices].to(device)
            )
            rank_loss = benchmark.strict_rank_pairwise_softplus_loss_v1(
                costs, role.dense_ranks[indices][row, action_ids].to(device)
            )
            dense = torch.zeros((), device=device)
            infonce = torch.zeros((), device=device)
            if joint:
                target = model.encode_target(
                    successor_trunks[indices][row, action_ids].to(device)
                )
                dense = benchmark.dense_patch_cosine_loss_v1(
                    prediction.successor_tokens, target
                )
                infonce = benchmark.within_state_action_infonce_loss_v1(
                    prediction.successor_tokens, target, temperature=0.10
                )
            total = physical_mse + 0.25 * rank_loss + dense + 0.10 * infonce
            if not bool(torch.isfinite(total)):
                raise GroundedRunnerError(f"{arm} loss became nonfinite")
            (total / ACCUMULATION_STEPS).backward()
            for name, value in (
                ("physical_mse", physical_mse),
                ("physical_rank", rank_loss),
                ("dense_cosine", dense),
                ("infonce", infonce),
                ("total", total),
            ):
                components[name] += float(value.detach()) / ACCUMULATION_STEPS
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(grad_norm)):
            raise GroundedRunnerError(f"{arm} gradient became nonfinite")
        optimizer.step()
        model.update_target_ema(EMA_MOMENTUM)
        if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
            raise GroundedRunnerError(f"{arm} parameter became nonfinite")
        last_objective = {
            **components,
            "gradient_norm_before_clip": float(grad_norm.detach()),
        }
        if update in {400, 800}:
            row = {
                "update": update,
                **evaluate_train_trace_v1(
                    model=model,
                    role=role,
                    context_trunks=context_trunks,
                    successor_trunks=successor_trunks,
                    input_statistics=input_statistics,
                    outcome_statistics=outcome_statistics,
                    benchmark=benchmark,
                    device=device,
                ),
                "objective": last_objective,
            }
            trace.append(row)
            if joint and update == 400:
                futility_keys = (
                    "normalized_physical_rank_regret",
                    "branch_retrieval_accuracy",
                    "successor_cosine_error",
                    "persistence_cosine_error",
                    "all_finite",
                )
                futility = train_only_futility_v1(
                    {key: trace[0][key] for key in futility_keys},
                    {key: row[key] for key in futility_keys},
                )
                if not futility["continue_to_update_800"]:
                    final_update = 400
                    break
    elapsed = time.perf_counter() - started
    return {
        "arm": arm,
        "updates": final_update,
        "trace": trace,
        "futility": futility,
        "training_seconds": elapsed,
        "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "optimizer": optimizer,
    }


@torch.no_grad()
def predict_role_outcomes_v1(
    *,
    model: torch.nn.Module,
    role: RoleRuntimeDataV1,
    context_trunks: torch.Tensor,
    input_statistics: Mapping[str, object],
    outcome_statistics: Mapping[str, object],
    benchmark: Any,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    rows = []
    for start in range(0, STATE_COUNT, MICROBATCH_STATES):
        stop = start + MICROBATCH_STATES
        normalized = benchmark.normalize_physical_inputs_v1(
            role.physical_inputs[start:stop].to(device), input_statistics
        )
        prediction = model(
            context_trunks[start:stop].to(device),
            role.history_commands[start:stop].to(device),
            role.candidate_commands[start:stop].to(device),
            normalized,
        )
        rows.append(
            benchmark.decode_standardized_outcomes_v1(
                prediction.standardized_physical_residuals, outcome_statistics
            ).cpu()
        )
    result = torch.cat(rows)
    if tuple(result.shape) != (STATE_COUNT, ACTION_COUNT, PHYSICAL_OUTPUT_DIM):
        raise GroundedRunnerError("role prediction shape changed")
    return result


def _load_model_from_checkpoint_v1(
    binding: Mapping[str, Any],
    *,
    expected_arm: str,
    dino: FrozenDINOTrunkV1,
    model_class: type[torch.nn.Module],
    device: torch.device,
) -> torch.nn.Module:
    _require_binding(binding, label=f"{expected_arm} checkpoint", rehash=True)
    payload = torch.load(Path(str(binding["path"])), map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != CHECKPOINT_SCHEMA
        or payload.get("arm") != expected_arm
    ):
        raise GroundedRunnerError("learned checkpoint contract changed")
    blocks, norm = dino.fresh_tail()
    model = model_class(
        blocks,
        norm,
        initialization_seed=MODEL_SEED,
        ema_momentum=EMA_MOMENTUM,
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device)


def report_group_results_v1(
    report: Mapping[str, Any], *, label: str
) -> list[Mapping[str, Any]]:
    """Select the existing evaluator's exact per-group bootstrap payload."""

    rows = report.get("group_results")
    if (
        not isinstance(rows, list)
        or not rows
        or any(not isinstance(row, Mapping) for row in rows)
    ):
        raise GroundedRunnerError(f"{label} group_results are absent")
    return rows


def assert_role_disjointness_v1(train_plan: Any, evaluation_plan: Any) -> dict[str, Any]:
    """Fail if any state, scene, or artifact identity crosses the role boundary."""

    train_states = {str(state.state_id) for state in train_plan.states}
    eval_states = {str(state.state_id) for state in evaluation_plan.states}
    train_scenes = {str(state.scene_id) for state in train_plan.states}
    eval_scenes = {str(state.scene_id) for state in evaluation_plan.states}
    train_artifacts = {str(value) for value in train_plan.artifact_ids}
    eval_artifacts = {str(value) for value in evaluation_plan.artifact_ids}
    expected_artifacts = STATE_COUNT * (CONTEXT_COUNT + ACTION_COUNT)
    if (
        getattr(train_plan, "role", None) != "train"
        or getattr(evaluation_plan, "role", None) != "eval"
        or len(train_states) != STATE_COUNT
        or len(eval_states) != STATE_COUNT
        or len(train_scenes) != SCENE_COUNT
        or len(eval_scenes) != SCENE_COUNT
        or len(train_artifacts) != expected_artifacts
        or len(eval_artifacts) != expected_artifacts
        or train_states & eval_states
        or train_scenes & eval_scenes
        or train_artifacts & eval_artifacts
    ):
        raise GroundedRunnerError("train and evaluation identities are not disjoint")
    return {
        "state_ids_disjoint": True,
        "scene_ids_disjoint": True,
        "artifact_ids_disjoint": True,
        "train_state_count": len(train_states),
        "eval_state_count": len(eval_states),
        "train_scene_count": len(train_scenes),
        "eval_scene_count": len(eval_scenes),
        "train_artifact_count": len(train_artifacts),
        "eval_artifact_count": len(eval_artifacts),
    }


def evaluate_roles_v1(
    *,
    train: RoleRuntimeDataV1,
    evaluation: RoleRuntimeDataV1,
    eval_context_trunks: torch.Tensor,
    input_statistics: Mapping[str, object],
    outcome_statistics: Mapping[str, object],
    physical_checkpoint: Mapping[str, Any],
    joint_checkpoint: Mapping[str, Any],
    dino: FrozenDINOTrunkV1,
    model_class: type[torch.nn.Module],
    benchmark: Any,
    device: torch.device,
) -> dict[str, Any]:
    from lewm.benchmarks import go2_dinov2_dense_shared_spatial_readout_calibration_v1 as dense
    from lewm.benchmarks import go2_matched_branch_physical_outcome_screen_v1 as physical

    disjointness = assert_role_disjointness_v1(train.plan, evaluation.plan)
    models = {
        "physical_only_matched": _load_model_from_checkpoint_v1(
            physical_checkpoint,
            expected_arm="physical_only_matched",
            dino=dino,
            model_class=model_class,
            device=device,
        ),
        "joint_jepa_grounded": _load_model_from_checkpoint_v1(
            joint_checkpoint,
            expected_arm="joint_jepa_grounded",
            dino=dino,
            model_class=model_class,
            device=device,
        ),
    }
    predictions: dict[str, torch.Tensor] = {}
    reports: dict[str, Any] = {}
    for arm, model in models.items():
        first = predict_role_outcomes_v1(
            model=model,
            role=evaluation,
            context_trunks=eval_context_trunks,
            input_statistics=input_statistics,
            outcome_statistics=outcome_statistics,
            benchmark=benchmark,
            device=device,
        )
        second = predict_role_outcomes_v1(
            model=model,
            role=evaluation,
            context_trunks=eval_context_trunks,
            input_statistics=input_statistics,
            outcome_statistics=outcome_statistics,
            benchmark=benchmark,
            device=device,
        )
        if not torch.equal(first, second):
            raise GroundedRunnerError(f"{arm} deterministic repeat changed")
        predictions[arm] = first
        reports[arm] = benchmark.report_physical_scores_v1(
            evaluation.plan,
            benchmark.physical_score_matrix_v1(evaluation.plan, first),
        )
    task = dense.fit_task_action_only_v1(train.plan)
    task_scores = dense.score_task_action_only_v1(evaluation.plan, task)
    reports["task_action_only"] = benchmark.report_physical_scores_v1(
        evaluation.plan, task_scores
    )
    task_regret = float(
        reports["task_action_only"]["summary"]["normalized_rank_regret"]
    )
    if task_regret != dense.EXPECTED_TASK_EVAL_REGRET:
        raise GroundedRunnerError("task/action-only evaluation regret changed")
    oracle_scores = np.asarray(
        [state.dense_ranks for state in evaluation.plan.states], dtype=np.float64
    )
    reports["privileged_physical_oracle"] = benchmark.report_physical_scores_v1(
        evaluation.plan, oracle_scores
    )
    reports["random_expected"] = physical.prior._random_expected_report(  # noqa: SLF001
        evaluation.plan
    )
    joint_vs_task = benchmark.paired_family_scene_bootstrap_v1(
        report_group_results_v1(
            reports["joint_jepa_grounded"], label="joint_jepa_grounded"
        ),
        report_group_results_v1(reports["task_action_only"], label="task_action_only"),
    )
    joint_vs_matched = benchmark.paired_family_scene_bootstrap_v1(
        report_group_results_v1(
            reports["joint_jepa_grounded"], label="joint_jepa_grounded"
        ),
        report_group_results_v1(
            reports["physical_only_matched"], label="physical_only_matched"
        ),
    )
    gate = benchmark.fixed_gate_v1(
        joint_report=reports["joint_jepa_grounded"],
        task_report=reports["task_action_only"],
        matched_report=reports["physical_only_matched"],
        random_report=reports["random_expected"],
        oracle_report=reports["privileged_physical_oracle"],
        joint_vs_task=joint_vs_task,
        joint_vs_matched=joint_vs_matched,
        integrity_passed=True,
    )
    return {
        "reports": reports,
        "comparisons": {
            "joint_vs_task_action_only": joint_vs_task,
            "joint_vs_physical_only_matched": joint_vs_matched,
        },
        "prediction_identities": {
            arm: _tensor_sha256(value) for arm, value in predictions.items()
        },
        "integrity_evidence": {
            "role_disjointness": disjointness,
            "task_action_eval_regret": task_regret,
            "expected_task_action_eval_regret": dense.EXPECTED_TASK_EVAL_REGRET,
        },
        "deterministic_repeat_passed": True,
        "gate": gate,
    }


def _source_bindings_unchanged(authority: Mapping[str, Any]) -> None:
    for label, expected in authority["source_bindings"].items():
        if file_binding_v1(Path(str(expected["path"]))) != expected:
            raise GroundedRunnerError(f"source {label} changed during execution")


def _validate_git_commit(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise GroundedRunnerError("reviewed git commit is malformed")
    status = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", value, "HEAD"],
        check=False,
    ).returncode
    if status != 0:
        raise GroundedRunnerError("reviewed git commit is not an ancestor of HEAD")
    return value


def validate_live_device_v1(environment: Mapping[str, Any]) -> str:
    """Require the live device to match the exact preregistered GPU claim."""

    if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
        raise GroundedRunnerError("the preregistered ROCm device is unavailable")
    live_name = torch.cuda.get_device_name(0)
    expected_name = str(environment.get("device_name", ""))
    if "R9700" not in live_name or live_name != expected_name:
        raise GroundedRunnerError(
            "live device 0 does not match the preregistered R9700 authority"
        )
    return live_name


def _load_authority_v1(
    path: Path,
    *,
    expected_sha256: str,
    expected_byte_count: int,
    rehash_inputs: bool = True,
) -> dict[str, Any]:
    authority_binding = file_binding_v1(path)
    if authority_binding["sha256"] != expected_sha256 or authority_binding[
        "byte_count"
    ] != expected_byte_count:
        raise GroundedRunnerError("execution authority caller binding changed")
    document = _strict_json_loads(path.read_bytes(), label="execution authority")
    required = {
        "schema",
        "status",
        "preregistration_binding",
        "source_review_binding",
        "source_bindings",
        "input_bindings",
        "dino",
        "environment",
        "permissions",
        "config",
        "output_root",
        "reviewed_git_commit",
    }
    permissions = {
        "train_receipt_access": True,
        "train_context_rgb_access": True,
        "train_successor_rgb_joint_only": True,
        "eval_receipt_access_after_checkpoints": True,
        "eval_context_rgb_access_after_checkpoints": True,
        "eval_successor_rgb_access": False,
        "data_generation": False,
        "protected_or_sealed_access": False,
        "retry_resume_overwrite": False,
    }
    if (
        set(document) != required
        or document.get("schema") != AUTHORITY_SCHEMA
        or document.get("status") != AUTHORITY_STATUS
        or document.get("permissions") != permissions
        or document.get("config") != runner_config_v1()
        or document.get("output_root") != str(DEFAULT_OUTPUT_ROOT.resolve())
    ):
        raise GroundedRunnerError("execution authority contract changed")
    _validate_git_commit(document["reviewed_git_commit"])
    prereg = _require_binding(document["preregistration_binding"], label="preregistration")
    if prereg != _binding(
        PREREGISTRATION, PREREGISTRATION_SHA256, PREREGISTRATION_BYTE_COUNT
    ):
        raise GroundedRunnerError("authority does not bind the frozen preregistration")
    sources = document["source_bindings"]
    if not isinstance(sources, Mapping) or set(sources) != set(SOURCE_PATHS):
        raise GroundedRunnerError("source closure labels changed")
    for label, expected_path in SOURCE_PATHS.items():
        actual = _require_binding(sources[label], label=f"source {label}")
        if actual["path"] != str(expected_path.resolve()):
            raise GroundedRunnerError(f"source {label} path changed")
    review_binding = _require_binding(document["source_review_binding"], label="source review")
    review = _strict_json_loads(
        Path(str(review_binding["path"])).read_bytes(), label="source review"
    )
    if (
        review.get("schema") != SOURCE_REVIEW_SCHEMA
        or review.get("status") != SOURCE_REVIEW_STATUS
        or review.get("preregistration_binding") != prereg
        or review.get("source_bindings") != sources
        or review.get("findings") != []
        or review.get("protected_material_opened") is not False
        or not isinstance(review.get("checks"), Mapping)
        or not review["checks"]
        or any(value is not True for value in review["checks"].values())
    ):
        raise GroundedRunnerError("independent source review did not pass exactly")
    inputs = document["input_bindings"]
    expected_inputs = fixed_input_bindings_v1()
    if not isinstance(inputs, Mapping) or set(inputs) != set(expected_inputs):
        raise GroundedRunnerError("scientific input binding set is malformed")
    for label, binding in inputs.items():
        # Hashing the evaluation role would itself open it.  Its declared
        # binding is syntax-checked now and rehashed only after both checkpoints.
        late = label in {"posthoc_eval_rows", "eval_role_index"} or label.startswith(
            "eval_state_receipt_"
        )
        normalized = _require_binding(
            binding, label=f"input {label}", rehash=rehash_inputs and not late
        )
        if normalized != expected_inputs[label]:
            raise GroundedRunnerError(f"input {label} identity changed")
    dino = document["dino"]
    if (
        not isinstance(dino, Mapping)
        or set(dino) != {"repository_path", "repository_commit", "checkpoint_binding"}
        or dino.get("repository_commit") != DINO_REPOSITORY_COMMIT
    ):
        raise GroundedRunnerError("DINO authority binding changed")
    checkpoint_binding = _require_binding(dino["checkpoint_binding"], label="DINO checkpoint")
    if (
        checkpoint_binding["sha256"] != DINO_CHECKPOINT_SHA256
        or checkpoint_binding["byte_count"] != DINO_CHECKPOINT_BYTE_COUNT
    ):
        raise GroundedRunnerError("DINO checkpoint identity changed")
    environment = document["environment"]
    if (
        not isinstance(environment, Mapping)
        or environment.get("python") != str(Path(sys.executable).resolve())
        or environment.get("torch") != torch.__version__
        or environment.get("hip") != torch.version.hip
        or environment.get("device_index") != 0
        or "R9700" not in str(environment.get("device_name", ""))
    ):
        raise GroundedRunnerError("execution environment changed")
    validate_live_device_v1(environment)
    return document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", required=True, type=Path)
    parser.add_argument("--expected-authority-sha256", required=True)
    parser.add_argument("--expected-authority-byte-count", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    authority = _load_authority_v1(
        args.authority,
        expected_sha256=args.expected_authority_sha256,
        expected_byte_count=args.expected_authority_byte_count,
    )
    output_root = Path(str(authority["output_root"]))
    existed = output_root.exists()
    try:
        result = execute_v1(authority)
    except Exception as error:
        if not existed and output_root.is_dir() and not (output_root / "terminal.json").exists():
            _write_json_exclusive(
                output_root / "terminal.json",
                {
                    "schema": TERMINAL_SCHEMA,
                    "status": "CONSUMED_TERMINAL_INFRASTRUCTURE_FAILURE",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "retry_authorized": False,
                    "result_binding": None,
                },
            )
        raise
    print(
        json.dumps(
            {"status": result["status"], "closed_loop_eligible": result["closed_loop_eligible"]},
            sort_keys=True,
        )
    )
    return 0


# ``execute_v1`` is appended below after the pure benchmark API is imported.
# Keeping imports lazy lets the focused runner tests exercise custody and
# lifecycle behavior with sentinels without importing Genesis or DINO.
def execute_v1(authority: Mapping[str, Any]) -> dict[str, Any]:
    from lewm.benchmarks import go2_grounded_dense_dino_joint_jepa_v1 as benchmark
    from lewm.models.go2_grounded_dense_dino_joint_jepa_v1 import (
        GroundedDenseDINOJointJEPAV1,
    )
    from lewm.datasets.go2_world_model_counterfactual_pilot_v1 import (
        read_bound_rgb_bytes_v1,
    )

    return _execute_with_components_v1(
        authority,
        benchmark=benchmark,
        model_class=GroundedDenseDINOJointJEPAV1,
        rgb_reader=read_bound_rgb_bytes_v1,
        dino_loader=load_dino_trunk_v1,
    )


def _execute_with_components_v1(
    authority: Mapping[str, Any],
    *,
    benchmark: Any,
    model_class: type[torch.nn.Module],
    rgb_reader: Callable[[Any, str], bytes],
    dino_loader: Callable[..., FrozenDINOTrunkV1],
) -> dict[str, Any]:
    """Dependency-injected production orchestration used by focused tests."""

    required_api = {
        "decode_standardized_outcomes_v1",
        "dense_patch_cosine_loss_v1",
        "extract_dense_trunk_layout_v1",
        "fit_input_statistics_v1",
        "fit_outcome_statistics_v1",
        "fixed_gate_v1",
        "normalize_physical_inputs_v1",
        "paired_family_scene_bootstrap_v1",
        "physical_score_matrix_v1",
        "predicted_physical_cost_v1",
        "report_physical_scores_v1",
        "standardize_outcome_residuals_v1",
        "strict_rank_pairwise_softplus_loss_v1",
        "true_successor_branch_retrieval_v1",
        "within_state_action_infonce_loss_v1",
    }
    missing = sorted(name for name in required_api if not callable(getattr(benchmark, name, None)))
    if missing:
        raise GroundedRunnerError(f"benchmark API is incomplete: {missing}")

    output_root = safe_path_v1(
        Path(str(authority["output_root"])), label="output root", must_exist=False
    )
    output_root.mkdir(parents=True, exist_ok=False)
    checkpoints = output_root / "checkpoints"
    checkpoints.mkdir()
    _write_json_exclusive(
        output_root / "reservation.json",
        {
            "schema": "lewm_go2_grounded_dense_dino_joint_jepa_v1_reservation_v1",
            "status": "CONSUMED_ONE_SHOT_ATTEMPT",
            "authority": dict(authority),
        },
    )
    determinism = configure_determinism_v1()
    ledger = AccessLedgerV1()
    shared = _load_shared_role_metadata_v1(authority)
    ledger.load_receipts("train")
    train = load_role_runtime_data_v1(authority, shared, role="train", ledger=ledger)

    if not torch.cuda.is_available():
        raise GroundedRunnerError("the preregistered ROCm device is unavailable")
    device = torch.device("cuda:0")
    dino = dino_loader(
        Path(str(authority["dino"]["repository_path"])),
        Path(str(authority["dino"]["checkpoint_binding"]["path"])),
        device=device,
    )
    context_ids = tuple(itertools.chain.from_iterable(train.context_artifact_ids))
    context_trunks_flat = precompute_trunks_v1(
        context_ids,
        role="train",
        kind="context",
        ledger=ledger,
        bound_reader=lambda artifact_id: rgb_reader(train.bundle, artifact_id),
        trunk=dino,
    )
    physical_layout = benchmark.extract_dense_trunk_layout_v1(
        train.plan,
        context_ids,
        context_trunks_flat,
        include_successors=False,
    )
    context_trunks = physical_layout.context_trunk_tokens
    if physical_layout.successor_trunk_tokens is not None:
        raise GroundedRunnerError("physical-only layout exposed successor trunks")
    input_statistics = benchmark.fit_input_statistics_v1(train.physical_inputs)
    outcome_statistics = benchmark.fit_outcome_statistics_v1(train.targets)
    optimizer_schedule = optimizer_microbatches_v1()
    optimizer_schedule_identity = optimizer_schedule_identity_v1(optimizer_schedule)
    arm_results: dict[str, Any] = {}

    physical_blocks, physical_norm = dino.fresh_tail()
    physical_model = model_class(
        physical_blocks,
        physical_norm,
        initialization_seed=MODEL_SEED,
        ema_momentum=EMA_MOMENTUM,
    ).to(device)
    physical_initial_identity = model_state_identity_v1(physical_model)
    physical_result = train_arm_v1(
        arm="physical_only_matched",
        model=physical_model,
        role=train,
        context_trunks=context_trunks,
        successor_trunks=None,
        input_statistics=input_statistics,
        outcome_statistics=outcome_statistics,
        optimizer_microbatches=optimizer_schedule,
        benchmark=benchmark,
        device=device,
    )
    if int(physical_result["updates"]) != MAX_UPDATES:
        raise GroundedRunnerError("physical-only arm did not reach update 800")
    physical_checkpoint = save_checkpoint_v1(
        checkpoints / "physical_only_matched_update_800.pt",
        arm="physical_only_matched",
        update=MAX_UPDATES,
        model=physical_model,
        optimizer=physical_result["optimizer"],
        train_identity=str(train.identity_sha256),
        trace=physical_result["trace"],
        input_statistics=input_statistics,
        outcome_statistics=outcome_statistics,
        initial_model_identity_sha256=physical_initial_identity,
    )
    ledger.checkpoint("physical_only_matched")
    arm_results["physical_only_matched"] = {
        key: value for key, value in physical_result.items() if key != "optimizer"
    } | {"checkpoint_binding": physical_checkpoint}
    del physical_model
    torch.cuda.empty_cache()

    successor_ids = tuple(itertools.chain.from_iterable(train.successor_artifact_ids))
    successor_trunks_flat = precompute_trunks_v1(
        successor_ids,
        role="train",
        kind="successor",
        ledger=ledger,
        bound_reader=lambda artifact_id: rgb_reader(train.bundle, artifact_id),
        trunk=dino,
    )
    joint_layout = benchmark.extract_dense_trunk_layout_v1(
        train.plan,
        context_ids + successor_ids,
        torch.cat((context_trunks_flat, successor_trunks_flat), dim=0),
        include_successors=True,
    )
    if joint_layout.successor_trunk_tokens is None:
        raise GroundedRunnerError("joint layout omitted successor trunks")
    if not torch.equal(joint_layout.context_trunk_tokens, context_trunks):
        raise GroundedRunnerError("joint and physical context trunks changed")
    successor_trunks = joint_layout.successor_trunk_tokens
    joint_blocks, joint_norm = dino.fresh_tail()
    joint_model = model_class(
        joint_blocks,
        joint_norm,
        initialization_seed=MODEL_SEED,
        ema_momentum=EMA_MOMENTUM,
    ).to(device)
    joint_initial_identity = model_state_identity_v1(joint_model)
    if joint_initial_identity != physical_initial_identity:
        raise GroundedRunnerError("learned arm initial trainable states differ")
    joint_result = train_arm_v1(
        arm="joint_jepa_grounded",
        model=joint_model,
        role=train,
        context_trunks=context_trunks,
        successor_trunks=successor_trunks,
        input_statistics=input_statistics,
        outcome_statistics=outcome_statistics,
        optimizer_microbatches=optimizer_schedule,
        benchmark=benchmark,
        device=device,
    )
    joint_updates = int(joint_result["updates"])
    if joint_updates not in {400, 800}:
        raise GroundedRunnerError("joint arm terminal update changed")
    joint_checkpoint = save_checkpoint_v1(
        checkpoints / f"joint_jepa_grounded_update_{joint_updates}.pt",
        arm="joint_jepa_grounded",
        update=joint_updates,
        model=joint_model,
        optimizer=joint_result["optimizer"],
        train_identity=str(train.identity_sha256),
        trace=joint_result["trace"],
        input_statistics=input_statistics,
        outcome_statistics=outcome_statistics,
        initial_model_identity_sha256=joint_initial_identity,
    )
    ledger.checkpoint("joint_jepa_grounded")
    arm_results["joint_jepa_grounded"] = {
        key: value for key, value in joint_result.items() if key != "optimizer"
    } | {"checkpoint_binding": joint_checkpoint}
    del joint_model
    torch.cuda.empty_cache()
    runtime_provenance = {
        "determinism": determinism,
        "environment": dict(authority["environment"]),
        "dino": dict(authority["dino"]),
        "frozen_dino_boundary": {
            "input_tokens": FULL_TOKEN_COUNT,
            "feature_dimension": FEATURE_DIM,
            "frozen_blocks": list(range(10)),
            "trainable_blocks": [10, 11],
            "trainable_final_norm": True,
        },
        "optimizer_schedule_identity_sha256": optimizer_schedule_identity,
        "arm_initial_model_identity_sha256": physical_initial_identity,
        "train_role_identity_sha256": train.identity_sha256,
        "input_statistics_identity_sha256": input_statistics["identity_sha256"],
        "outcome_statistics_identity_sha256": outcome_statistics["identity_sha256"],
    }

    if joint_updates == 400:
        access_audit = finalized_access_audit_v1(ledger, evaluation_opened=False)
        _source_bindings_unchanged(authority)
        result = {
            "schema": SCHEMA,
            "status": "COMPLETE_TRAIN_CAPACITY_FUTILITY_STOP",
            "closed_loop_eligible": False,
            "evaluation_opened": False,
            "arms": arm_results,
            "runtime_provenance": runtime_provenance,
            "access_audit": access_audit,
            "authority": dict(authority),
        }
        _write_json_exclusive(output_root / "result.json", result)
        result_binding = file_binding_v1(output_root / "result.json")
        _write_json_exclusive(
            output_root / "terminal.json",
            {
                "schema": TERMINAL_SCHEMA,
                "status": "COMPLETE_TRAIN_CAPACITY_FUTILITY_STOP",
                "closed_loop_eligible": False,
                "retry_authorized": False,
                "result_binding": result_binding,
            },
        )
        return result

    ledger.load_receipts("eval")
    evaluation = load_role_runtime_data_v1(
        authority, shared, role="eval", ledger=ledger
    )
    eval_context_ids = tuple(
        itertools.chain.from_iterable(evaluation.context_artifact_ids)
    )
    eval_context_flat = precompute_trunks_v1(
        eval_context_ids,
        role="eval",
        kind="context",
        ledger=ledger,
        bound_reader=lambda artifact_id: rgb_reader(evaluation.bundle, artifact_id),
        trunk=dino,
    )
    eval_layout = benchmark.extract_dense_trunk_layout_v1(
        evaluation.plan,
        eval_context_ids,
        eval_context_flat,
        include_successors=False,
    )
    if eval_layout.successor_trunk_tokens is not None:
        raise GroundedRunnerError("evaluation layout exposed successor trunks")
    evaluation_result = evaluate_roles_v1(
        train=train,
        evaluation=evaluation,
        eval_context_trunks=eval_layout.context_trunk_tokens,
        input_statistics=input_statistics,
        outcome_statistics=outcome_statistics,
        physical_checkpoint=physical_checkpoint,
        joint_checkpoint=joint_checkpoint,
        dino=dino,
        model_class=model_class,
        benchmark=benchmark,
        device=device,
    )
    gate = evaluation_result["gate"]
    eligible = bool(gate["passed"])
    access_audit = finalized_access_audit_v1(ledger, evaluation_opened=True)
    _source_bindings_unchanged(authority)
    result = {
        "schema": SCHEMA,
        "status": (
            "PASS_CLOSED_LOOP_EXPERIMENT_ELIGIBLE"
            if eligible
            else "FAIL_STOP_GROUNDED_DENSE_DINO_MECHANISM"
        ),
        "closed_loop_eligible": eligible,
        "evaluation_opened": True,
        "arms": arm_results,
        "evaluation": evaluation_result,
        "gate": gate,
        "runtime_provenance": runtime_provenance
        | {"eval_role_identity_sha256": evaluation.identity_sha256},
        "access_audit": access_audit,
        "authority": dict(authority),
    }
    _write_json_exclusive(output_root / "result.json", result)
    result_binding = file_binding_v1(output_root / "result.json")
    _write_json_exclusive(
        output_root / "terminal.json",
        {
            "schema": TERMINAL_SCHEMA,
            "status": result["status"],
            "closed_loop_eligible": eligible,
            "retry_authorized": False,
            "result_binding": result_binding,
        },
    )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
