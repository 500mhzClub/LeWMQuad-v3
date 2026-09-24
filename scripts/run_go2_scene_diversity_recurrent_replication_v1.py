#!/usr/bin/env python3
"""Collect and run the one-shot scene-diversity recurrent replication V1."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
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

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks import (  # noqa: E402
    go2_matched_branch_physical_outcome_screen_v1 as physical,
)
from lewm.benchmarks import (  # noqa: E402
    go2_scene_diversity_recurrent_replication_v1 as benchmark,
)
from scripts import collect_go2_scene_diversity_recurrent_replication_v1 as collector  # noqa: E402
from scripts import build_go2_world_model_bounded_branch_experiment_authority_v1 as bounded_authority  # noqa: E402
from scripts import run_go2_grounded_dense_dino_joint_jepa_v1 as upstream  # noqa: E402
from scripts import run_go2_world_model_bounded_branch_experiment_authorized_v1 as collection_supervisor  # noqa: E402
from scripts import run_go2_world_model_counterfactual_calibration_authorized_v1 as calibration_supervisor  # noqa: E402


AUTHORITY_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_v1_execution_authority_v1"
)
AUTHORITY_STATUS = "AUTHORIZED_ONE_SCENE_DIVERSITY_RECURRENT_REPLICATION_V1"
SOURCE_REVIEW_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_v1_source_review_v1"
)
SOURCE_REVIEW_STATUS = "PASS_INDEPENDENT_SOURCE_REVIEW"
RESULT_SCHEMA = "lewm_go2_scene_diversity_recurrent_replication_v1_result_v1"
TERMINAL_SCHEMA = "lewm_go2_scene_diversity_recurrent_replication_v1_terminal_v1"
RESERVATION_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_v1_attempt_reservation_v1"
)

PREREGISTRATION = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_v1_preregistration_2026-08-04.md"
)
SCENE_PANEL = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_v1_scene_panel_2026-08-04.json"
)
SCENE_PANEL_SHA256 = "df145c2d70d82243b373ef6f6d8750dc231f9de2a4d07d9f698a1831b9b84fa7"
SCENE_PANEL_BYTE_COUNT = 207_218
SOURCE_REVIEW = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_v1_source_review_2026-08-04.json"
)
DEFAULT_ATTEMPT_ROOT = REPO_ROOT / (
    ".generated/dev/go2_scene_diversity_recurrent_replication_v1/attempt_v1"
)
DEFAULT_COLLECTION_ROOT = DEFAULT_ATTEMPT_ROOT / "collection"
DINO_REPOSITORY = Path(
    "/home/andrewknowles/.cache/dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8"
)
DINO_CHECKPOINT = Path(
    "/home/andrewknowles/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth"
)
DINO_REPOSITORY_COMMIT = upstream.DINO_REPOSITORY_COMMIT
DINO_CHECKPOINT_SHA256 = upstream.DINO_CHECKPOINT_SHA256
DINO_CHECKPOINT_BYTE_COUNT = upstream.DINO_CHECKPOINT_BYTE_COUNT

SOURCE_PATHS = {
    **{
        f"collection_runtime_{name}": REPO_ROOT / relative
        for name, relative in bounded_authority.canonical_source_paths_v1().items()
    },
    **{
        f"grounded_upstream_{name}": path
        for name, path in upstream.SOURCE_PATHS.items()
    },
    "recurrent_model": REPO_ROOT
    / "lewm/models/go2_task_coupled_recurrent_dynamics_v1.py",
    "recurrent_benchmark": REPO_ROOT
    / "lewm/benchmarks/go2_task_coupled_recurrent_dynamics_v1.py",
    "replication_benchmark": REPO_ROOT
    / "lewm/benchmarks/go2_scene_diversity_recurrent_replication_v1.py",
    "replication_runner": Path(__file__).resolve(),
    "replication_plan_builder": REPO_ROOT
    / "scripts/build_go2_scene_diversity_recurrent_replication_plan_v1.py",
    "replication_collector": REPO_ROOT
    / "scripts/collect_go2_scene_diversity_recurrent_replication_v1.py",
    "replication_authority_builder": REPO_ROOT
    / "scripts/build_go2_scene_diversity_recurrent_replication_authority_v1.py",
    "collection_supervisor": REPO_ROOT
    / "scripts/run_go2_world_model_bounded_branch_experiment_authorized_v1.py",
    "calibration_supervisor": REPO_ROOT
    / "scripts/run_go2_world_model_counterfactual_calibration_authorized_v1.py",
    "replication_benchmark_test": REPO_ROOT
    / "lewm/tests/test_go2_scene_diversity_recurrent_replication_v1_benchmark.py",
    "replication_runner_test": REPO_ROOT
    / "lewm/tests/test_run_go2_scene_diversity_recurrent_replication_v1.py",
    "replication_plan_test": REPO_ROOT
    / "lewm/tests/test_build_go2_scene_diversity_recurrent_replication_plan_v1.py",
    "replication_collector_test": REPO_ROOT
    / "lewm/tests/test_collect_go2_scene_diversity_recurrent_replication_v1.py",
    "replication_authority_test": REPO_ROOT
    / "lewm/tests/test_build_go2_scene_diversity_recurrent_replication_authority_v1.py",
}


class SceneDiversityRunnerError(RuntimeError):
    """Raised when authority, custody, collection, or output contracts change."""


def canonical_bytes_v1(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def file_binding_v1(path: Path) -> dict[str, object]:
    selected = upstream.safe_path_v1(path, label="bound file")
    if not selected.is_file() or selected.is_symlink():
        raise SceneDiversityRunnerError("bound path is not a regular file")
    digest = hashlib.sha256()
    size = 0
    with selected.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(selected), "sha256": digest.hexdigest(), "byte_count": size}


def expected_dino_v1() -> dict[str, object]:
    return {
        "repository_path": str(DINO_REPOSITORY.resolve()),
        "repository_commit": DINO_REPOSITORY_COMMIT,
        "checkpoint_binding": {
            "path": str(DINO_CHECKPOINT.resolve()),
            "sha256": DINO_CHECKPOINT_SHA256,
            "byte_count": DINO_CHECKPOINT_BYTE_COUNT,
        },
    }


def _validate_dino_source_v1() -> None:
    repository = upstream.safe_path_v1(DINO_REPOSITORY, label="DINO repository")
    if not repository.is_dir() or repository.is_symlink():
        raise SceneDiversityRunnerError("DINO repository path changed")
    try:
        head = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain"],
            check=True, capture_output=True, text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise SceneDiversityRunnerError("cannot validate DINO repository") from error
    if head != DINO_REPOSITORY_COMMIT or status:
        raise SceneDiversityRunnerError("DINO repository commit or cleanliness changed")


def _require_binding(
    value: object, *, label: str, rehash: bool = True
) -> dict[str, object]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"path", "sha256", "byte_count"}
        or not isinstance(value.get("path"), str)
        or not isinstance(value.get("sha256"), str)
        or len(str(value["sha256"])) != 64
        or type(value.get("byte_count")) is not int
        or int(value["byte_count"]) <= 0
    ):
        raise SceneDiversityRunnerError(f"{label} binding is malformed")
    observed = dict(value)
    upstream._reject_protected(Path(str(observed["path"])), label=label)  # noqa: SLF001
    if rehash and file_binding_v1(Path(str(observed["path"]))) != observed:
        raise SceneDiversityRunnerError(f"{label} binding changed")
    return observed


def _read_json_binding(value: object, *, label: str) -> dict[str, Any]:
    binding = _require_binding(value, label=label)
    try:
        document = json.loads(Path(str(binding["path"])).read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SceneDiversityRunnerError(f"{label} is not strict JSON") from error
    if not isinstance(document, dict):
        raise SceneDiversityRunnerError(f"{label} must be a JSON object")
    return document


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _save_checkpoint_exclusive(
    path: Path, checkpoint: Mapping[str, Any]
) -> Mapping[str, Any]:
    if path.exists():
        raise SceneDiversityRunnerError("checkpoint path already exists")
    with path.open("xb") as handle:
        torch.save(dict(checkpoint), handle)
        handle.flush()
        os.fsync(handle.fileno())
    reopened = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(reopened, Mapping)
        or reopened.get("identity_sha256") != benchmark.checkpoint_identity_v1(reopened)
    ):
        raise SceneDiversityRunnerError("checkpoint round trip changed")
    return reopened


def _validate_plan_v1(plan: Mapping[str, Any], authority: Mapping[str, Any]) -> None:
    # Full Genesis-dependent validation intentionally belongs to the bound
    # collector runtime.  These model-process checks are exact but metadata-only.
    if (
        plan.get("schema") != "lewm_go2_world_model_counterfactual_pilot_plan_v1"
        or plan.get("attempt_id") != authority.get("attempt_id")
        or plan.get("output_root") != str(DEFAULT_COLLECTION_ROOT.resolve())
        or plan.get("states_per_scene") != 4
        or plan.get("expected_counts") != collector.EXPECTED_COUNTS
        or not isinstance(plan.get("states"), list)
        or len(plan["states"]) != 256
    ):
        raise SceneDiversityRunnerError("scene-diversity plan changed")
    collector._validate_scene_diversity_plan_v1(plan)  # noqa: SLF001


def _validate_authority_v1(
    authority_path: Path, *, expected_sha256: str, expected_byte_count: int
) -> tuple[dict[str, Any], dict[str, object], dict[str, Any]]:
    """Validate every authority leaf before any output, graphics, or collection."""

    expected = {
        "path": str(upstream.safe_path_v1(authority_path, label="authority")),
        "sha256": expected_sha256,
        "byte_count": expected_byte_count,
    }
    authority = _read_json_binding(expected, label="execution authority")
    if (
        set(authority) != collector.AUTHORITY_FIELDS
        or authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("status") != AUTHORITY_STATUS
        or authority.get("attempt_root") != str(DEFAULT_ATTEMPT_ROOT.resolve())
        or authority.get("collection_root") != str(DEFAULT_COLLECTION_ROOT.resolve())
        or authority.get("config") != benchmark.config_v1()
        or authority.get("caps") != collector.EXPECTED_CAPS
        or authority.get("permissions") != collector.EXPECTED_PERMISSIONS
    ):
        raise SceneDiversityRunnerError("execution authority contract changed")
    plan_binding = _require_binding(authority.get("plan_binding"), label="exact plan")
    plan = _read_json_binding(plan_binding, label="exact plan")
    _validate_plan_v1(plan, authority)
    prereg = _require_binding(authority.get("preregistration_binding"), label="preregistration")
    if prereg["path"] != str(PREREGISTRATION.resolve()):
        raise SceneDiversityRunnerError("preregistration path changed")
    panel_binding = file_binding_v1(SCENE_PANEL)
    if panel_binding != {
        "path": str(SCENE_PANEL.resolve()),
        "sha256": SCENE_PANEL_SHA256,
        "byte_count": SCENE_PANEL_BYTE_COUNT,
    }:
        raise SceneDiversityRunnerError("frozen scene panel changed")
    review_binding = _require_binding(authority.get("source_review_binding"), label="source review")
    if review_binding["path"] != str(SOURCE_REVIEW.resolve()):
        raise SceneDiversityRunnerError("source review path changed")
    sources = authority.get("source_bindings")
    if not isinstance(sources, Mapping) or set(sources) != set(SOURCE_PATHS):
        raise SceneDiversityRunnerError("source closure changed")
    for name, path in SOURCE_PATHS.items():
        binding = _require_binding(sources[name], label=f"source {name}")
        if binding["path"] != str(path.resolve()):
            raise SceneDiversityRunnerError(f"source {name} path changed")
    review = _read_json_binding(review_binding, label="source review")
    if (
        review.get("schema") != SOURCE_REVIEW_SCHEMA
        or review.get("status") != SOURCE_REVIEW_STATUS
        or review.get("protected_material_opened") is not False
        or review.get("preregistration_binding") != prereg
        or review.get("scene_panel_binding") != panel_binding
        or review.get("plan_binding") != plan_binding
        or review.get("source_bindings") != sources
        or review.get("findings") != []
    ):
        raise SceneDiversityRunnerError("independent source review changed")
    if authority.get("dino") != expected_dino_v1():
        raise SceneDiversityRunnerError("frozen DINO binding changed")
    _require_binding(authority["dino"]["checkpoint_binding"], label="DINO checkpoint")
    _validate_dino_source_v1()
    attempt = Path(str(authority["attempt_root"]))
    collection = Path(str(authority["collection_root"]))
    development = (REPO_ROOT / ".generated/dev").resolve(strict=True)
    if (
        attempt != DEFAULT_ATTEMPT_ROOT.resolve()
        or collection != DEFAULT_COLLECTION_ROOT.resolve()
        or collection.parent != attempt
    ):
        raise SceneDiversityRunnerError("attempt roots changed")
    attempt.relative_to(development)
    return authority, expected, plan


@dataclass
class ContextOnlyLedgerV1:
    stage: str = "created"
    checkpoint_durable: bool = False
    receipt_loads: dict[str, int] = field(default_factory=lambda: {"train": 0, "eval": 0})
    role_index_opens: dict[str, int] = field(default_factory=lambda: {"train": 0, "eval": 0})
    state_receipt_opens: dict[str, int] = field(default_factory=lambda: {"train": 0, "eval": 0})
    render_receipt_opens: dict[str, int] = field(default_factory=lambda: {"train": 0, "eval": 0})
    rgb_opens: dict[str, int] = field(default_factory=lambda: {
        "train_context": 0, "train_successor": 0,
        "eval_context": 0, "eval_successor": 0,
    })
    opened_receipts: set[tuple[str, str]] = field(default_factory=set)
    opened_artifacts: set[tuple[str, str]] = field(default_factory=set)

    def load_receipts(self, role: str) -> None:
        if role == "train" and self.stage == "created":
            self.stage = "train"
        elif role == "eval" and self.checkpoint_durable and self.stage == "checkpoint":
            self.stage = "eval"
        else:
            raise SceneDiversityRunnerError("role receipts opened outside custody stage")
        self.receipt_loads[role] = 1

    def open_role_index(self, role: str, path: str) -> None:
        if self.receipt_loads.get(role) != 1 or self.role_index_opens[role] or not path:
            raise SceneDiversityRunnerError("role index opened outside custody stage")
        self.role_index_opens[role] = 1

    def open_state_receipt(self, role: str, path: str) -> None:
        key = (role, path)
        if self.receipt_loads.get(role) != 1 or not path or key in self.opened_receipts:
            raise SceneDiversityRunnerError("state receipt opened outside custody stage")
        self.opened_receipts.add(key)
        self.state_receipt_opens[role] += 1

    def open_render_receipt(self, role: str, path: str) -> None:
        if self.receipt_loads.get(role) != 1 or not path:
            raise SceneDiversityRunnerError("render receipt opened outside custody stage")
        self.render_receipt_opens[role] += 1

    def open_rgb(self, role: str, kind: str, artifact_id: str) -> None:
        if kind != "context":
            raise SceneDiversityRunnerError("successor RGB is structurally forbidden")
        if (role == "train" and self.stage != "train") or (
            role == "eval" and self.stage != "eval"
        ):
            raise SceneDiversityRunnerError("context RGB opened outside custody stage")
        key = (role, artifact_id)
        if not artifact_id or key in self.opened_artifacts:
            raise SceneDiversityRunnerError("context artifact opened more than once")
        self.opened_artifacts.add(key)
        self.rgb_opens[f"{role}_context"] += 1

    def checkpoint(self) -> None:
        if self.stage != "train" or self.checkpoint_durable:
            raise SceneDiversityRunnerError("checkpoint durability order changed")
        self.checkpoint_durable = True
        self.stage = "checkpoint"

    def finalized(self) -> dict[str, Any]:
        audit = {
            "stage": self.stage,
            "checkpoint_durable": self.checkpoint_durable,
            "receipt_loads": dict(self.receipt_loads),
            "role_index_opens": dict(self.role_index_opens),
            "state_receipt_opens": dict(self.state_receipt_opens),
            "render_receipt_opens": dict(self.render_receipt_opens),
            "rgb_opens": dict(self.rgb_opens),
            "unique_context_artifacts": len(self.opened_artifacts),
            "successor_rgb_open_count": self.rgb_opens["train_successor"] + self.rgb_opens["eval_successor"],
        }
        if (
            self.stage != "eval"
            or not self.checkpoint_durable
            or self.receipt_loads != {"train": 1, "eval": 1}
            or self.role_index_opens != {"train": 1, "eval": 1}
            or self.state_receipt_opens != {"train": 128, "eval": 128}
            or self.render_receipt_opens != {"train": 32, "eval": 32}
            or self.rgb_opens != {
                "train_context": 384, "train_successor": 0,
                "eval_context": 384, "eval_successor": 0,
            }
            or audit["unique_context_artifacts"] != 768
        ):
            raise SceneDiversityRunnerError("context-only access accounting changed")
        return audit


@dataclass(frozen=True)
class RoleRuntimeDataV1:
    role: str
    plan: Any
    physical_inputs: torch.Tensor
    targets: torch.Tensor
    history_commands: torch.Tensor
    candidate_commands: torch.Tensor
    relative_goals: torch.Tensor
    dense_ranks: torch.Tensor
    context_artifact_ids: tuple[tuple[str, str, str], ...]
    context_artifacts: Mapping[str, Mapping[str, Any]]
    collection_root: Path
    stored_rgb_bytes: int
    stored_rgb_frames: int
    identity_sha256: str


def _historical_binding(value: object, *, root: Path, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise SceneDiversityRunnerError(f"{label} binding is absent")
    relative = PurePosixPath(str(value.get("path")))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise SceneDiversityRunnerError(f"{label} path changed")
    return _require_binding(
        {
            "path": str(upstream.safe_path_v1(root.joinpath(*relative.parts), label=label)),
            "sha256": value.get("file_sha256", value.get("sha256")),
            "byte_count": value.get("byte_count"),
        },
        label=label,
        rehash=False,
    )


def _read_bound_json_once(
    binding: Mapping[str, object], *, label: str
) -> dict[str, Any]:
    normalized = _require_binding(binding, label=label, rehash=False)
    raw = Path(str(normalized["path"])).read_bytes()
    if len(raw) != normalized["byte_count"] or hashlib.sha256(raw).hexdigest() != normalized["sha256"]:
        raise SceneDiversityRunnerError(f"{label} binding changed")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SceneDiversityRunnerError(f"{label} is not strict JSON") from error
    if not isinstance(value, dict):
        raise SceneDiversityRunnerError(f"{label} must be an object")
    return value


def _load_physics_index_v1(
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, object],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(str(authority["collection_root"])) / "physics_result.json"
    binding = file_binding_v1(path)
    value = _read_json_binding(binding, label="physics result")
    if (
        value.get("schema") != "lewm_go2_world_model_counterfactual_pilot_physics_result_v1"
        or value.get("status") != "PHYSICS_COMPLETE"
        or value.get("physics_validated") is not False
        or value.get("citable_as_scientific_evidence") is not False
        or value.get("authorizes_retry_or_resume") is not False
        or value.get("allows_refill") is not False
        or value.get("allows_overwrite") is not False
        or value.get("failure") is not None
        or value.get("authority_binding") != dict(authority_binding)
        or value.get("source_bindings") != authority.get("source_bindings")
        or value.get("caps") != authority.get("caps")
        or value.get("expected_counts") != collector.EXPECTED_COUNTS
        or value.get("observed_counts") != collector.EXPECTED_COUNTS
        or value.get("plan_binding") != {
            "path": str(authority["plan_binding"]["path"]),
            "file_sha256": str(authority["plan_binding"]["sha256"]),
            "byte_count": int(authority["plan_binding"]["byte_count"]),
        }
        or not isinstance(value.get("state_receipt_bindings"), list)
        or len(value["state_receipt_bindings"]) != 256
        or not isinstance(value.get("render_receipt_bindings"), list)
        or len(value["render_receipt_bindings"]) != 64
        or not isinstance(value.get("scene_metrics"), list)
        or len(value["scene_metrics"]) != 64
        or not math.isfinite(float(value.get("collection_wall_seconds", math.nan)))
        or float(value["collection_wall_seconds"]) <= 0.0
    ):
        raise SceneDiversityRunnerError("physics result contract changed")
    value["_binding"] = binding
    value["_plan_state_ids"] = [str(row["state_id"]) for row in plan["states"]]
    return value


def _load_role_runtime_data_v1(
    authority: Mapping[str, Any],
    plan_document: Mapping[str, Any],
    physics_index: Mapping[str, Any],
    *,
    role: str,
    ledger: ContextOnlyLedgerV1,
) -> RoleRuntimeDataV1:
    collection_root = Path(str(authority["collection_root"]))
    declarations = [row for row in plan_document["states"] if row["role"] == role]
    raw_state_bindings = physics_index["state_receipt_bindings"]
    role_state_bindings = [
        value for value in raw_state_bindings
        if ("scenes", role) in zip(
            PurePosixPath(str(value.get("path"))).parts,
            PurePosixPath(str(value.get("path"))).parts[1:],
        )
    ]
    if len(declarations) != 128 or len(role_state_bindings) != 128:
        raise SceneDiversityRunnerError(f"{role} receipt declarations changed")
    ledger.open_role_index(role, str(physics_index["_binding"]["path"]))
    receipts = []
    for index, (declared, raw_binding) in enumerate(zip(declarations, role_state_bindings, strict=True)):
        binding = _historical_binding(raw_binding, root=collection_root, label=f"{role} state receipt")
        ledger.open_state_receipt(role, str(binding["path"]))
        receipt = _read_bound_json_once(binding, label=f"{role} state receipt {index}")
        state = receipt.get("state")
        if (
            receipt.get("status") != "PHYSICS_COMPLETE"
            or not isinstance(state, Mapping)
            or any(state.get(name) != declared.get(name) for name in (
                "role", "state_id", "scene_id", "family", "group_index", "state_index_in_scene"
            ))
        ):
            raise SceneDiversityRunnerError(f"{role} state receipt identity changed")
        context = receipt.get("context")
        branches = receipt.get("branches")
        action_catalog = plan_document.get("action_catalog")
        if (
            not isinstance(context, Mapping)
            or context.get("history_action_ids") != declared.get("history_action_ids")
            or state.get("target_xy_m") != declared.get("target_xy_m")
            or state.get("scene_manifest_binding") != declared.get("scene_manifest_binding")
            or state.get("scene_genesis_binding") != declared.get("scene_genesis_binding")
            or not isinstance(branches, list)
            or [branch.get("action_id") for branch in branches]
            != declared.get("candidate_action_ids")
            or not isinstance(action_catalog, list)
            or len(action_catalog) != 9
            or [branch.get("requested_block") for branch in branches]
            != [action_catalog[action_id].get("requested_block") for action_id in range(9)]
        ):
            raise SceneDiversityRunnerError(
                f"{role} receipt disagrees with planned history/action/source"
            )
        receipts.append(receipt)
    groups, receipt_by_id = physical._groups_from_receipts(receipts, role=role)  # noqa: SLF001
    role_plan = benchmark.build_role_feature_plan_v1(groups, role=role)
    physical_inputs, targets = physical._role_arrays(role_plan, receipt_by_id)  # noqa: SLF001
    histories = []
    candidates = []
    for state in role_plan.states:
        receipt = receipt_by_id[state.state_id]
        history_blocks = receipt["context"].get("history_executed_blocks")
        branches = receipt.get("branches")
        if not isinstance(history_blocks, list) or len(history_blocks) != 2 or not isinstance(branches, list) or len(branches) != 9:
            raise SceneDiversityRunnerError("command tape geometry changed")
        histories.append(torch.stack([upstream.command_tape_channel_major_v1(block) for block in history_blocks]))
        candidates.append(torch.stack([
            upstream.command_tape_channel_major_v1(branch.get("requested_block"))
            for branch in branches
        ]))
    history_commands = torch.stack(histories)
    candidate_commands = torch.stack(candidates)
    goals = torch.tensor([state.relative_target_xy_body_m for state in role_plan.states], dtype=torch.float32)
    ranks = torch.tensor([state.dense_ranks for state in role_plan.states], dtype=torch.long)
    context_ids = tuple(
        tuple(role_plan.artifact_ids[index] for index in state.context_artifact_indices)
        for state in role_plan.states
    )
    wanted = set(itertools.chain.from_iterable(context_ids))
    raw_render_bindings = physics_index["render_receipt_bindings"]
    role_render_bindings = [
        value for value in raw_render_bindings
        if ("scenes", role) in zip(
            PurePosixPath(str(value.get("path"))).parts,
            PurePosixPath(str(value.get("path"))).parts[1:],
        )
    ]
    if len(role_render_bindings) != 32:
        raise SceneDiversityRunnerError(f"{role} render receipt count changed")
    context_artifacts: dict[str, Mapping[str, Any]] = {}
    stored_rgb_bytes = 0
    stored_rgb_frames = 0
    for index, raw_binding in enumerate(role_render_bindings):
        binding = _historical_binding(raw_binding, root=collection_root, label=f"{role} render receipt")
        ledger.open_render_receipt(role, str(binding["path"]))
        render = _read_bound_json_once(binding, label=f"{role} render receipt {index}")
        frames = render.get("frame_receipts")
        scene = render.get("scene")
        if (
            render.get("status") != "RENDER_COMPLETE"
            or not isinstance(scene, Mapping)
            or scene.get("role") != role
            or not isinstance(frames, list)
            or len(frames) != 48
        ):
            raise SceneDiversityRunnerError(f"{role} render receipt changed")
        for frame in frames:
            if (
                not isinstance(frame, Mapping)
                or type(frame.get("byte_count")) is not int
                or int(frame["byte_count"]) <= 0
            ):
                raise SceneDiversityRunnerError("RGB frame byte count changed")
            stored_rgb_bytes += int(frame["byte_count"])
            stored_rgb_frames += 1
            artifact_id = frame.get("artifact_id") if isinstance(frame, Mapping) else None
            if artifact_id not in wanted:
                continue
            relative = PurePosixPath(str(frame.get("path")))
            if (
                artifact_id in context_artifacts
                or relative.is_absolute()
                or ".." in relative.parts
                or frame.get("width") != 224
                or frame.get("height") != 224
                or frame.get("mode") != "RGB"
                or frame.get("format") != "PNG"
                or frame.get("camera_valid") is not True
            ):
                raise SceneDiversityRunnerError("context RGB metadata changed")
            context_artifacts[str(artifact_id)] = MappingProxyType(dict(frame))
    if set(context_artifacts) != wanted or len(context_artifacts) != 384:
        raise SceneDiversityRunnerError(f"{role} context RGB closure changed")
    identity = upstream._role_identity_v1(  # noqa: SLF001
        role, role_plan, physical_inputs, targets, history_commands, candidate_commands
    )
    result = RoleRuntimeDataV1(
        role=role,
        plan=role_plan,
        physical_inputs=physical_inputs,
        targets=targets,
        history_commands=history_commands,
        candidate_commands=candidate_commands,
        relative_goals=goals,
        dense_ranks=ranks,
        context_artifact_ids=context_ids,
        context_artifacts=MappingProxyType(context_artifacts),
        collection_root=collection_root,
        stored_rgb_bytes=stored_rgb_bytes,
        stored_rgb_frames=stored_rgb_frames,
        identity_sha256=identity,
    )
    benchmark.validate_role_scene_geometry_v1(result)
    return result


def _read_context_rgb_v1(role: RoleRuntimeDataV1, artifact_id: str) -> bytes:
    frame = role.context_artifacts.get(artifact_id)
    if frame is None:
        raise SceneDiversityRunnerError("context artifact is not admitted")
    relative = PurePosixPath(str(frame["path"]))
    path = upstream.safe_path_v1(
        role.collection_root.joinpath(*relative.parts), label="context RGB"
    )
    raw = path.read_bytes()
    if len(raw) != int(frame["byte_count"]) or hashlib.sha256(raw).hexdigest() != frame["file_sha256"]:
        raise SceneDiversityRunnerError("context RGB binding changed")
    return raw


@torch.inference_mode()
def _full_dino_context_tokens_v1(
    role: RoleRuntimeDataV1,
    *,
    ledger: ContextOnlyLedgerV1,
    dino: upstream.FrozenDINOTrunkV1,
) -> torch.Tensor:
    artifact_ids = tuple(itertools.chain.from_iterable(role.context_artifact_ids))
    trunks = upstream.precompute_trunks_v1(
        artifact_ids,
        role=role.role,
        kind="context",
        ledger=ledger,
        bound_reader=lambda artifact_id: _read_context_rgb_v1(role, artifact_id),
        trunk=dino,
    )
    rows = []
    for start in range(0, len(artifact_ids), 16):
        hidden = trunks[start : start + 16].to(dino.device)
        for block in tuple(dino.dino.blocks)[10:12]:
            hidden = block(hidden)
        hidden = dino.dino.norm(hidden)
        rows.append(F.normalize(hidden[:, 1:], dim=-1).cpu())
    result = torch.cat(rows).reshape(128, 3, 256, 384).contiguous()
    return benchmark.validate_context_tokens_v1(result, role=role.role)


def assert_role_disjointness_v1(train_plan: Any, eval_plan: Any) -> dict[str, Any]:
    train_states = {str(state.state_id) for state in train_plan.states}
    eval_states = {str(state.state_id) for state in eval_plan.states}
    train_scenes = {str(state.scene_id) for state in train_plan.states}
    eval_scenes = {str(state.scene_id) for state in eval_plan.states}
    train_artifacts = set(map(str, train_plan.artifact_ids))
    eval_artifacts = set(map(str, eval_plan.artifact_ids))
    if (
        len(train_states) != 128 or len(eval_states) != 128
        or len(train_scenes) != 32 or len(eval_scenes) != 32
        or len(train_artifacts) != 1536 or len(eval_artifacts) != 1536
        or train_states & eval_states or train_scenes & eval_scenes
        or train_artifacts & eval_artifacts
    ):
        raise SceneDiversityRunnerError("train and evaluation roles are not disjoint")
    return {
        "state_ids_disjoint": True, "scene_ids_disjoint": True,
        "artifact_ids_disjoint": True, "train_state_count": 128,
        "eval_state_count": 128, "train_scene_count": 32,
        "eval_scene_count": 32, "train_artifact_count": 1536,
        "eval_artifact_count": 1536,
    }


def _reserve_attempt_v1(
    authority: Mapping[str, Any], authority_binding: Mapping[str, object]
) -> None:
    attempt = Path(str(authority["attempt_root"]))
    namespace = attempt.parent
    development = (REPO_ROOT / ".generated/dev").resolve(strict=True)
    if namespace.parent != development or namespace.is_symlink():
        raise SceneDiversityRunnerError("attempt namespace changed")
    if not namespace.exists():
        try:
            os.mkdir(namespace, mode=0o700)
        except OSError as error:
            raise SceneDiversityRunnerError("could not create attempt namespace") from error
    if not namespace.is_dir() or namespace.is_symlink():
        raise SceneDiversityRunnerError("attempt namespace is not a directory")
    if attempt.exists() or attempt.is_symlink():
        raise SceneDiversityRunnerError("attempt root is not fresh")
    try:
        os.mkdir(attempt, mode=0o700)
    except OSError as error:
        raise SceneDiversityRunnerError("could not reserve fresh attempt root") from error
    _write_json_exclusive(attempt / "reservation.json", {
        "schema": RESERVATION_SCHEMA,
        "status": "CONSUMED_ONE_SHOT_ATTEMPT",
        "authority_binding": dict(authority_binding),
        "plan_binding": dict(authority["plan_binding"]),
        "retry_resume_overwrite_authorized": False,
    })


def _collect_if_absent_v1(
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, object],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    attempt = Path(str(authority["attempt_root"]))
    collection = Path(str(authority["collection_root"]))
    physics = collection / "physics_result.json"
    if physics.exists():
        raise SceneDiversityRunnerError("fresh authority unexpectedly has collection output")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise SceneDiversityRunnerError("authorized ROCm device is unavailable")
    expected_device = plan.get("execution_contract", {}).get(
        "graphics_preflight", {}
    ).get("vulkan_device_name")
    if not isinstance(expected_device, str) or torch.cuda.get_device_name(0) != expected_device:
        raise SceneDiversityRunnerError("ROCm and bound graphics device disagree")
    invocation = str(calibration_supervisor._validate_python_invocation(plan))  # noqa: SLF001
    child_env = calibration_supervisor._child_environment(plan)  # noqa: SLF001
    wall_started = time.monotonic()
    graphics = calibration_supervisor._run_graphics_preflight(  # noqa: SLF001
        plan,
        child_env=child_env,
        wall_started=wall_started,
        wall_ceiling=float(authority["caps"]["wall_seconds"]),
    )
    used, total, vendor, device = calibration_supervisor._selected_gpu_memory_files(plan)  # noqa: SLF001
    sampler = calibration_supervisor._GlobalVramSampler(  # noqa: SLF001
        used, total, vendor_id=vendor, device_id=device, interval_seconds=0.02
    )
    ceiling = int(authority["caps"]["selected_device_vram_byte_ceiling"])
    if sampler.baseline_used_bytes > ceiling:
        raise SceneDiversityRunnerError("selected-device VRAM baseline exceeds cap")
    # Reserve only after every static and graphics/device preflight succeeds,
    # immediately before the one authorized collector process is launched.
    _reserve_attempt_v1(authority, authority_binding)
    enforcement = {
        "enabled": True,
        "scope": "selected_device_global_vram_not_process_attributed",
        "ceiling_bytes": ceiling,
        "sample_interval_seconds": sampler.interval_seconds,
        "collector_started": False, "collector_pid": None,
        "collector_exit_code": None, "collector_terminated": False,
        "termination_reason": None,
        "peak_observed_during_collector_bytes": sampler.baseline_used_bytes,
    }
    argv = [
        invocation, str(Path(collector.__file__).resolve()),
        "--plan", str(authority["plan_binding"]["path"]),
        "--expected-plan-byte-count", str(authority["plan_binding"]["byte_count"]),
        "--expected-plan-sha256", str(authority["plan_binding"]["sha256"]),
        "--authority", str(authority_binding["path"]),
        "--expected-authority-byte-count", str(authority_binding["byte_count"]),
        "--expected-authority-sha256", str(authority_binding["sha256"]),
    ]
    sampler.start()
    try:
        phase = collection_supervisor._run_collector_once_with_vram_ceiling(  # noqa: SLF001
            argv,
            timeout=max(0.001, float(authority["caps"]["wall_seconds"]) - (time.monotonic() - wall_started)),
            env=child_env,
            sampler=sampler,
            ceiling_bytes=ceiling,
            enforcement=enforcement,
        )
    finally:
        measurement = sampler.stop()
    if measurement["read_errors"] != 0 or measurement["peak_used_bytes"] > ceiling:
        raise SceneDiversityRunnerError("selected-device VRAM enforcement failed")
    if not physics.is_file():
        raise SceneDiversityRunnerError("collector did not emit physics result")
    return {"graphics_preflight": graphics, "collector": phase, "vram": measurement, "vram_enforcement": enforcement}


def _result_identity_v1(value: Mapping[str, Any]) -> str:
    document = dict(value)
    document.pop("result_identity_sha256", None)
    return hashlib.sha256(canonical_bytes_v1(document)).hexdigest()


def execute_v1(
    authority: Mapping[str, Any],
    *,
    authority_binding: Mapping[str, object],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    collection_run = _collect_if_absent_v1(authority, authority_binding, plan)
    physics_index = _load_physics_index_v1(authority, authority_binding, plan)
    determinism = upstream.configure_determinism_v1()
    ledger = ContextOnlyLedgerV1()
    ledger.load_receipts("train")
    train = _load_role_runtime_data_v1(authority, plan, physics_index, role="train", ledger=ledger)
    if not torch.cuda.is_available():
        raise SceneDiversityRunnerError("authorized ROCm device is unavailable")
    device = torch.device("cuda:0")
    dino = upstream.load_dino_trunk_v1(
        Path(str(authority["dino"]["repository_path"])),
        Path(str(authority["dino"]["checkpoint_binding"]["path"])),
        device=device,
    )
    train_context = _full_dino_context_tokens_v1(train, ledger=ledger, dino=dino)
    checkpoint = benchmark.fit_checkpoint_v1(train, train_context, device=device)
    checkpoint_path = Path(str(authority["attempt_root"])) / "checkpoint.pt"
    # Drop the in-memory training object at this boundary.  Both evaluation
    # replays consume only the checkpoint durably reopened from disk.
    checkpoint = _save_checkpoint_exclusive(checkpoint_path, checkpoint)
    checkpoint_binding = file_binding_v1(checkpoint_path)
    ledger.checkpoint()
    del train_context
    torch.cuda.empty_cache()
    ledger.load_receipts("eval")
    evaluation = _load_role_runtime_data_v1(authority, plan, physics_index, role="eval", ledger=ledger)
    disjointness = assert_role_disjointness_v1(train.plan, evaluation.plan)
    eval_context = _full_dino_context_tokens_v1(evaluation, ledger=ledger, dino=dino)
    first = benchmark.evaluate_checkpoint_v1(
        checkpoint, train, evaluation, eval_context, device=device, integrity_passed=True
    )
    second = benchmark.evaluate_checkpoint_v1(
        checkpoint, train, evaluation, eval_context, device=device, integrity_passed=True
    )
    if canonical_bytes_v1(first) != canonical_bytes_v1(second):
        raise SceneDiversityRunnerError("repeat evaluation was not bitwise identical")
    access_audit = ledger.finalized()
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": first["status"],
        "citable_as_scientific_evidence": False,
        "development_only": True,
        "authority_binding": dict(authority_binding),
        "preregistration_binding": dict(authority["preregistration_binding"]),
        "source_review_binding": dict(authority["source_review_binding"]),
        "source_bindings": dict(authority["source_bindings"]),
        "plan_binding": dict(authority["plan_binding"]),
        "physics_result_binding": dict(physics_index["_binding"]),
        "checkpoint_binding": checkpoint_binding,
        "checkpoint_summary": {
            "identity_sha256": checkpoint["identity_sha256"],
            "frozen_config": checkpoint["config"],
            "arms": {
                arm: [{
                    "seed": member["seed"],
                    "initial_state_identity_sha256": member["initial_state_identity_sha256"],
                    "state_identity_sha256": member["state_identity_sha256"],
                    "updates": member["updates"],
                    "trace": member["trace"],
                    "training_seconds": member["training_seconds"],
                } for member in checkpoint["arms"][arm]]
                for arm in benchmark.ARM_ORDER
            },
        },
        "evaluation": first,
        "repeat_evaluation_exact": True,
        "role_disjointness": disjointness,
        "access_audit": access_audit,
        "collection_summary": {
            "collection_wall_seconds": physics_index["collection_wall_seconds"],
            "stored_rgb_bytes": train.stored_rgb_bytes + evaluation.stored_rgb_bytes,
            "stored_rgb_frames": train.stored_rgb_frames + evaluation.stored_rgb_frames,
            "observed_counts": physics_index["observed_counts"],
            "scenes": [{
                name: row[name]
                for name in (
                    "role", "family", "scene_id", "states", "stored_rgb_frames",
                    "scene_total_wall_seconds", "physics_simulation_wall_seconds",
                    "native_render_wall_seconds",
                )
            } for row in physics_index["scene_metrics"]],
        },
        "collection_run": collection_run,
        "runtime": {
            "determinism": determinism,
            "torch": torch.__version__, "hip": torch.version.hip,
            "device": torch.cuda.get_device_name(device),
            "frozen_dino": {"blocks": list(range(12)), "final_norm": True,
                            "l2_normalized_patch_tokens": True, "trainable": False},
        },
        "successor_observations_opened": 0,
        "authorizes_navigation_claim": False,
        "authorizes_blind_rollout_preregistration": first["authorizes_blind_rollout_preregistration"],
    }
    result["result_identity_sha256"] = _result_identity_v1(result)
    attempt = Path(str(authority["attempt_root"]))
    _write_json_exclusive(attempt / "result.json", result)
    _write_json_exclusive(attempt / "terminal.json", {
        "schema": TERMINAL_SCHEMA,
        "status": result["status"],
        "authorizes_retry_or_resume": False,
        "authorizes_navigation_claim": False,
        "authorizes_blind_rollout_preregistration": result["authorizes_blind_rollout_preregistration"],
        "result_binding": file_binding_v1(attempt / "result.json"),
        "failure": None,
    })
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--expected-authority-sha256", required=True)
    parser.add_argument("--expected-authority-byte-count", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    authority: Mapping[str, Any] | None = None
    try:
        authority, binding, plan = _validate_authority_v1(
            args.authority,
            expected_sha256=args.expected_authority_sha256,
            expected_byte_count=args.expected_authority_byte_count,
        )
        result = execute_v1(authority, authority_binding=binding, plan=plan)
        print(json.dumps({"status": result["status"], "attempt_root": authority["attempt_root"]}))
        return 0
    except Exception as error:
        if authority is not None:
            attempt = Path(str(authority["attempt_root"]))
            terminal = attempt / "terminal.json"
            if attempt.is_dir() and not terminal.exists():
                try:
                    _write_json_exclusive(terminal, {
                        "schema": TERMINAL_SCHEMA,
                        "status": "FAIL_INFRASTRUCTURE_NO_SCIENTIFIC_DECISION",
                        "authorizes_retry_or_resume": False,
                        "authorizes_navigation_claim": False,
                        "authorizes_blind_rollout_preregistration": False,
                        "result_binding": None,
                        "failure": {"type": type(error).__name__, "message": str(error)},
                    })
                except Exception:
                    pass
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORITY_SCHEMA", "AUTHORITY_STATUS", "DINO_CHECKPOINT",
    "DINO_CHECKPOINT_BYTE_COUNT", "DINO_CHECKPOINT_SHA256", "DINO_REPOSITORY",
    "DINO_REPOSITORY_COMMIT", "SOURCE_PATHS", "ContextOnlyLedgerV1",
    "RoleRuntimeDataV1", "SceneDiversityRunnerError", "assert_role_disjointness_v1",
    "expected_dino_v1", "execute_v1", "file_binding_v1",
]
