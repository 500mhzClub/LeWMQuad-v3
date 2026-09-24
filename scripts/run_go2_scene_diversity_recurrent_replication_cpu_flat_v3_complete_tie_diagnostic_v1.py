#!/usr/bin/env python3
"""Run the post-hoc CPU-flat V3 complete-tie evaluation diagnostic.

This successor is evaluation-only and non-confirmatory.  It rehashes the
consumed CPU-flat V3 collection and checkpoint, reconstructs both frozen data
roles, opens only evaluation context RGB for DINO, and evaluates the unchanged
checkpoint twice on CPU.  Its sole scientific-domain adaptation is to admit an
evaluation state whose nine dense ranks are all zero; such a state uses
``max(1, max_dense_rank)`` as its regret denominator.  No state is excluded and
no training, rendering, collection, successor-image access, retry, or repair is
available from this module.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import itertools
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
import time
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for _package_root in (REPO_ROOT, REPO_ROOT / "lewm_genesis", REPO_ROOT / "lewm_worlds"):
    if str(_package_root) not in sys.path:
        sys.path.insert(0, str(_package_root))

from lewm.benchmarks import go2_dinov2_physical_readout_calibration_v1 as calibration  # noqa: E402
from lewm.benchmarks import go2_grounded_dense_dino_joint_jepa_v1 as grounded  # noqa: E402
from lewm.benchmarks import go2_matched_branch_physical_outcome_screen_v1 as physical  # noqa: E402
from lewm.benchmarks import go2_scene_diversity_recurrent_replication_v1 as benchmark  # noqa: E402
from lewm.benchmarks import go2_task_coupled_recurrent_dynamics_v1 as frozen  # noqa: E402
from scripts import run_go2_grounded_dense_dino_joint_jepa_v1 as upstream  # noqa: E402
from scripts import run_go2_scene_diversity_recurrent_replication_v1 as frozen_runner  # noqa: E402


STEM = "go2_scene_diversity_recurrent_replication_cpu_flat_v3_complete_tie_diagnostic_v1"
ATTEMPT_ID = f"{STEM}_attempt_v1"
ATTEMPT_ROOT = REPO_ROOT / ".generated" / "dev" / STEM / "attempt_v1"
RESULT_PATH = ATTEMPT_ROOT / "diagnostic_result.json"
TERMINAL_PATH = ATTEMPT_ROOT / "terminal.json"
RESERVATION_PATH = ATTEMPT_ROOT / "reservation.json"
PLAN_PATH = REPO_ROOT / "docs" / f"lewm_{STEM}_exact_plan_2026-08-05.json"
SOURCE_REVIEW_PATH = (
    REPO_ROOT / "docs" / f"lewm_{STEM}_source_review_2026-08-05.json"
)
BUILDER_PATH = REPO_ROOT / "scripts" / f"build_{STEM}_plan.py"
FOCUSED_TEST_PATH = REPO_ROOT / "lewm" / "tests" / f"test_run_{STEM}.py"

PREDECESSOR_ATTEMPT_ROOT = (
    REPO_ROOT
    / ".generated/dev/go2_scene_diversity_recurrent_replication_genesis_cpu_flat_"
    "development_v3/attempt_v1"
)
PREDECESSOR_COLLECTION_ROOT = PREDECESSOR_ATTEMPT_ROOT / "collection"
PREDECESSOR_RESULT_PATH = PREDECESSOR_ATTEMPT_ROOT / "result.json"
PREDECESSOR_PLAN_PATH = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_"
    "development_v3_scientific_exact_plan_2026-08-05.json"
)
PREDECESSOR_TERMINAL_PATH = PREDECESSOR_ATTEMPT_ROOT / "terminal.json"
PREDECESSOR_PHYSICS_PATH = PREDECESSOR_COLLECTION_ROOT / "physics_result.json"
PREDECESSOR_CHECKPOINT_PATH = PREDECESSOR_ATTEMPT_ROOT / "checkpoint.pt"
PREDECESSOR_TERMINAL_REVIEW_PATH = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_"
    "development_v3_scientific_terminal_review_2026-08-05.json"
)
PREDECESSOR_SOURCE_REVIEW_PATH = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_"
    "development_v3_scientific_source_review_2026-08-05.json"
)

PLAN_SCHEMA = f"lewm_{STEM}_exact_plan_v1"
PLAN_STATUS = "FROZEN_POST_HOC_DEVELOPMENT_DIAGNOSTIC_PLAN"
SOURCE_REVIEW_SCHEMA = f"lewm_{STEM}_source_review_v1"
SOURCE_REVIEW_STATUS = "PASS_INDEPENDENT_COMPLETE_TIE_DIAGNOSTIC_SOURCE_REVIEW"
RESERVATION_SCHEMA = f"lewm_{STEM}_reservation_v1"
RESULT_SCHEMA = f"lewm_{STEM}_result_v1"
TERMINAL_SCHEMA = f"lewm_{STEM}_terminal_v1"
EVALUATION_SCHEMA = f"lewm_{STEM}_evaluation_v1"
COMPLETE_STATUS = "COMPLETE_POST_HOC_NONCONFIRMATORY_DIAGNOSTIC"
FAIL_STATUS = "FAIL_DIAGNOSTIC_INFRASTRUCTURE_NO_DECISION"

EXPECTED_BINDINGS = {
    "scientific_plan_binding": {
        "path": str(PREDECESSOR_PLAN_PATH),
        "sha256": "0ad79cc46cead469d6532cd0be04c5d7623fffe18ddafc737c32855d6c9a8f29",
        "byte_count": 359_692,
    },
    "terminal_binding": {
        "path": str(PREDECESSOR_TERMINAL_PATH),
        "sha256": "a4da81177d77372923b72775f69cfe58b596a651017ef6ebc5988df05d390327",
        "byte_count": 1_273,
    },
    "physics_result_binding": {
        "path": str(PREDECESSOR_PHYSICS_PATH),
        "sha256": "711b8722c11dbae663ad1b004268b77c64ff3d2e818f2c895851c547240e3ed0",
        "byte_count": 369_067,
    },
    "checkpoint_binding": {
        "path": str(PREDECESSOR_CHECKPOINT_PATH),
        "sha256": "6c16f97ae5748e1d230244b4588f3efc11330a2673bd15e2ff83aa2f2392844e",
        "byte_count": 167_423,
    },
    "terminal_review_binding": {
        "path": str(PREDECESSOR_TERMINAL_REVIEW_PATH),
        "sha256": "7218c78387871e82280f96fe746acb047f46d1a2836b7638b12ce9c1514a81dd",
        "byte_count": 17_379,
    },
}
EXPECTED_PREDECESSOR_SOURCE_REVIEW_BINDING = {
    "path": str(PREDECESSOR_SOURCE_REVIEW_PATH),
    "sha256": "43aad17a51cfcd4177de6e6c6f10b4eed0656836482f517668ab9a879ae2ce93",
    "byte_count": 10_544,
}
EXPECTED_EVAL_STATE_COUNT = 128
EXPECTED_COMPLETE_TIE_STATE_COUNT = 4


class CompleteTieDiagnosticError(RuntimeError):
    """Fail-closed diagnostic contract error."""


def canonical_bytes_v1(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _standard_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(value["path"]),
        "sha256": str(value.get("sha256", value.get("file_sha256"))),
        "byte_count": int(value["byte_count"]),
    }


def file_binding_v1(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise CompleteTieDiagnosticError(f"bound path is not a file: {resolved}")
    raw = resolved.read_bytes()
    return {
        "path": str(resolved),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
    }


def _require_exact_binding(
    value: object, *, expected: Mapping[str, Any], label: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CompleteTieDiagnosticError(f"{label} binding is absent")
    try:
        declared = _standard_binding(value)
    except (KeyError, TypeError, ValueError) as error:
        raise CompleteTieDiagnosticError(f"{label} binding is malformed") from error
    wanted = _standard_binding(expected)
    if declared != wanted or file_binding_v1(Path(declared["path"])) != wanted:
        raise CompleteTieDiagnosticError(f"{label} binding changed")
    return wanted


def _read_bound_json(binding: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    normalized = _standard_binding(binding)
    if file_binding_v1(Path(normalized["path"])) != normalized:
        raise CompleteTieDiagnosticError(f"{label} binding changed")
    raw = Path(normalized["path"]).read_bytes()
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CompleteTieDiagnosticError(f"{label} is not strict JSON") from error
    if not isinstance(document, dict):
        raise CompleteTieDiagnosticError(f"{label} must be a JSON object")
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


def _expected_evaluation_contract() -> dict[str, Any]:
    return {
        "evaluation_only": True,
        "training_authorized": False,
        "rendering_authorized": False,
        "collection_authorized": False,
        "checkpoint_reuse_mode": "read_only_exact_rehash",
        "collection_reuse_mode": "read_only_exact_rehash",
        "roles_reconstructed": ["train", "eval"],
        "train_role_use": "live_task_action_only_control_metadata_only",
        "train_context_rgb_open_count": 0,
        "eval_context_rgb_open_count": 384,
        "successor_rgb_open_count": 0,
        "eval_state_count": EXPECTED_EVAL_STATE_COUNT,
        "expected_eval_complete_tie_state_count": EXPECTED_COMPLETE_TIE_STATE_COUNT,
        "eval_state_exclusion_authorized": False,
        "complete_tie_rule": "all_actions_oracle_equivalent",
        "random_expected_denominator": "max(1,max_dense_rank)",
        "rank_tolerance_m": float(physical.RANK_TOLERANCE_M),
        "evaluation_repetitions": 2,
        "repeat_evaluation_exact_required": True,
        "compute_device": "cpu",
        "frozen_recurrent_config": benchmark.config_v1(),
        "model_seeds": list(benchmark.MODEL_SEEDS),
        "sampler_seed": benchmark.SAMPLER_SEED,
        "bootstrap_resamples": grounded.BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": grounded.BOOTSTRAP_SEED,
        "frozen_thresholds": frozen.config_v1()["frozen_h1_thresholds"],
    }


def _expected_source_bindings() -> dict[str, dict[str, Any]]:
    paths = {
        "diagnostic_plan_builder": BUILDER_PATH,
        "diagnostic_runner": Path(__file__).resolve(),
        "focused_test": FOCUSED_TEST_PATH,
        "frozen_replication_runner": Path(frozen_runner.__file__).resolve(),
        "frozen_replication_benchmark": Path(benchmark.__file__).resolve(),
        "frozen_recurrent_benchmark": Path(frozen.__file__).resolve(),
    }
    return {name: file_binding_v1(path) for name, path in paths.items()}


def read_and_validate_plan_v1(
    path: Path, *, expected_sha256: str, expected_byte_count: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_path = PLAN_PATH.resolve(strict=True)
    binding = {
        "path": str(path.resolve(strict=True)),
        "sha256": expected_sha256,
        "byte_count": expected_byte_count,
    }
    if Path(binding["path"]) != expected_path:
        raise CompleteTieDiagnosticError("diagnostic plan path changed")
    if file_binding_v1(expected_path) != binding:
        raise CompleteTieDiagnosticError("diagnostic plan binding changed")
    plan = _read_bound_json(binding, label="diagnostic plan")
    predecessor = plan.get("predecessor")
    if not isinstance(predecessor, Mapping):
        raise CompleteTieDiagnosticError("predecessor contract is absent")
    for name, expected in EXPECTED_BINDINGS.items():
        _require_exact_binding(predecessor.get(name), expected=expected, label=name)
    expected_top = {
        "schema": PLAN_SCHEMA,
        "status": PLAN_STATUS,
        "attempt_id": ATTEMPT_ID,
        "attempt_root": str(ATTEMPT_ROOT.resolve(strict=False)),
        "result_path": str(RESULT_PATH.resolve(strict=False)),
        "terminal_path": str(TERMINAL_PATH.resolve(strict=False)),
        "development_only": True,
        "post_hoc_nonconfirmatory": True,
        "citable_as_scientific_evidence": False,
        "fresh_root_required": True,
    }
    if any(plan.get(name) != value for name, value in expected_top.items()):
        raise CompleteTieDiagnosticError("diagnostic plan identity or claim scope changed")
    expected_predecessor_paths = {
        "attempt_root": str(PREDECESSOR_ATTEMPT_ROOT.resolve(strict=True)),
        "collection_root": str(PREDECESSOR_COLLECTION_ROOT.resolve(strict=True)),
        "result_path": str(PREDECESSOR_RESULT_PATH.resolve(strict=False)),
        "result_must_be_absent": True,
    }
    if any(
        predecessor.get(name) != value
        for name, value in expected_predecessor_paths.items()
    ):
        raise CompleteTieDiagnosticError("predecessor path or no-result contract changed")
    if PREDECESSOR_RESULT_PATH.exists() or PREDECESSOR_RESULT_PATH.is_symlink():
        raise CompleteTieDiagnosticError("predecessor scientific result must remain absent")
    evaluation_contract = plan.get("evaluation_contract")
    expected_evaluation_contract = _expected_evaluation_contract()
    if not isinstance(evaluation_contract, Mapping) or any(
        evaluation_contract.get(name) != value
        for name, value in expected_evaluation_contract.items()
    ):
        raise CompleteTieDiagnosticError("evaluation-only contract changed")
    if plan.get("dino") != frozen_runner.expected_dino_v1():
        raise CompleteTieDiagnosticError("frozen DINO contract changed")
    return plan, binding


def read_and_validate_source_review_v1(
    path: Path,
    *,
    expected_sha256: str,
    expected_byte_count: int,
    plan_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = {
        "path": str(path.resolve(strict=True)),
        "sha256": expected_sha256,
        "byte_count": expected_byte_count,
    }
    if Path(binding["path"]) != SOURCE_REVIEW_PATH.resolve(strict=True):
        raise CompleteTieDiagnosticError("diagnostic source-review path changed")
    if file_binding_v1(Path(binding["path"])) != binding:
        raise CompleteTieDiagnosticError("diagnostic source-review binding changed")
    review = _read_bound_json(binding, label="diagnostic source review")
    expected_decision = {
        "source_review_passed": True,
        "post_hoc_nonconfirmatory_scope_verified": True,
        "exact_predecessor_checkpoint_and_collection_rehash_verified": True,
        "evaluation_only_no_training_rendering_or_collection_verified": True,
        "all_128_eval_states_retained_verified": True,
        "sole_complete_tie_domain_change_verified": True,
        "exact_frozen_model_seeds_bootstrap_thresholds_verified": True,
        "context_only_cpu_dino_and_zero_successor_access_verified": True,
        "fresh_diagnostic_root_verified": True,
        "exactly_one_diagnostic_invocation_eligible_under_user_authorization": True,
        "scientific_execution_authority_created_by_review": False,
        "retry_resume_overwrite_repair_or_second_invocation_authorized": False,
    }
    if (
        review.get("schema") != SOURCE_REVIEW_SCHEMA
        or review.get("status") != SOURCE_REVIEW_STATUS
        or review.get("protected_material_opened") is not False
        or review.get("findings") != []
        or review.get("plan_binding") != dict(plan_binding)
        or review.get("source_bindings") != _expected_source_bindings()
        or review.get("predecessor_terminal_review_binding")
        != EXPECTED_BINDINGS["terminal_review_binding"]
        or review.get("decision") != expected_decision
    ):
        raise CompleteTieDiagnosticError("diagnostic source review changed")
    return review, binding


def _validate_predecessor_terminal_review(plan: Mapping[str, Any]) -> dict[str, Any]:
    bindings = plan["predecessor"]
    review = _read_bound_json(
        bindings["terminal_review_binding"], label="predecessor terminal review"
    )
    review_bindings = review.get("bindings")
    decision = review.get("decision")
    localization = review.get("failure_localization")
    terminal_chain = review.get("terminal_chain")
    recommendation = review.get("successor_recommendation")
    if (
        review.get("schema")
        != (
            "lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_"
            "development_v3_scientific_terminal_review_v1"
        )
        or review.get("status") != "FAIL_CLOSED_NO_SCIENTIFIC_DECISION"
        or review.get("protected_material_opened") is not False
        or not isinstance(review_bindings, Mapping)
        or review_bindings.get("terminal") != bindings["terminal_binding"]
        or review_bindings.get("physics_result") != bindings["physics_result_binding"]
        or review_bindings.get("checkpoint") != bindings["checkpoint_binding"]
        or review_bindings.get("scientific_exact_plan")
        != bindings["scientific_plan_binding"]
        or review_bindings.get("scientific_result") is not None
        or not isinstance(decision, Mapping)
        or decision.get("terminal_and_no_result_state_exact") is not True
        or decision.get("checkpoint_identity_passed") is not True
        or decision.get("four_all_tie_evaluation_states_exactly_localized") is not True
        or decision.get("scientific_decision") is not False
        or decision.get("successor_authority_created") is not False
        or not isinstance(localization, Mapping)
        or localization.get("rank_invalid_all_tie_states") != 4
        or localization.get("rank_invalid_by_role") != {"train": 0, "eval": 4}
        or localization.get("rank_tolerance_m") != float(physical.RANK_TOLERANCE_M)
        or not isinstance(terminal_chain, Mapping)
        or terminal_chain.get("scientific_result_file_present") is not False
        or terminal_chain.get("attempt_consumed_and_terminal") is not True
        or not isinstance(recommendation, Mapping)
        or recommendation.get("successor_execution_authorized") is not False
        or recommendation.get(
            "same_consumed_artifacts_could_only_support_post_hoc_diagnostic_not_preregistered_v3_decision"
        )
        is not True
    ):
        raise CompleteTieDiagnosticError("predecessor terminal review contract changed")
    return review


def _validate_predecessor_terminal(plan: Mapping[str, Any]) -> dict[str, Any]:
    terminal = _read_bound_json(
        plan["predecessor"]["terminal_binding"], label="predecessor terminal"
    )
    failure = terminal.get("failure")
    if (
        terminal.get("schema")
        != (
            "lewm_go2_scene_diversity_recurrent_replication_genesis_cpu_flat_"
            "development_v3_scientific_terminal_v1"
        )
        or terminal.get("status") != "FAIL_INFRASTRUCTURE_NO_SCIENTIFIC_DECISION"
        or terminal.get("result_binding") is not None
        or terminal.get("authorizes_retry_or_resume") is not False
        or terminal.get("authorizes_navigation_claim") is not False
        or not isinstance(failure, Mapping)
        or failure.get("message") != "dense ranks are invalid"
    ):
        raise CompleteTieDiagnosticError("predecessor terminal contract changed")
    if PREDECESSOR_RESULT_PATH.exists() or PREDECESSOR_RESULT_PATH.is_symlink():
        raise CompleteTieDiagnosticError("predecessor result appeared")
    return terminal


def build_eval_role_feature_plan_complete_ties_v1(
    groups: Sequence[Any], *, role: str
) -> calibration.RoleFeaturePlanV1:
    """Build the frozen scene-diversity eval plan while admitting rank-zero ties."""

    if role != "eval" or len(groups) != benchmark.STATE_COUNT:
        raise CompleteTieDiagnosticError("complete-tie adapter is evaluation-only")
    try:
        ordered = tuple(
            sorted(groups, key=lambda group: (int(group.group_index), str(group.state_id)))
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise CompleteTieDiagnosticError("eval group ordering is malformed") from error
    artifact_ids: list[str] = []
    artifact_index_by_id: dict[str, int] = {}
    states: list[calibration.RoleStateIndexV1] = []
    seen_states: set[str] = set()
    seen_group_indices: set[int] = set()

    def append_artifact(value: object) -> int:
        try:
            artifact_id = calibration._text(value, name="RGB artifact ID")  # noqa: SLF001
        except Exception as error:
            raise CompleteTieDiagnosticError("RGB artifact ID changed") from error
        if artifact_id in artifact_index_by_id:
            raise CompleteTieDiagnosticError("artifact is reused across state slots")
        index = len(artifact_ids)
        artifact_ids.append(artifact_id)
        artifact_index_by_id[artifact_id] = index
        return index

    for role_state_index, group in enumerate(ordered):
        if getattr(group, "role", None) != role:
            raise CompleteTieDiagnosticError("group crossed role boundary")
        try:
            state_id = calibration._text(group.state_id, name="state ID")  # noqa: SLF001
            family = calibration._text(group.family, name="family")  # noqa: SLF001
            scene_id = calibration._text(group.scene_id, name="scene ID")  # noqa: SLF001
            group_index = int(group.group_index)
            state_index_in_scene = int(group.state_index_in_scene)
        except (AttributeError, TypeError, ValueError) as error:
            raise CompleteTieDiagnosticError("state identity is malformed") from error
        if (
            state_id in seen_states
            or group_index in seen_group_indices
            or group_index < 0
            or state_index_in_scene < 0
        ):
            raise CompleteTieDiagnosticError("state identity repeats")
        seen_states.add(state_id)
        seen_group_indices.add(group_index)
        target = np.asarray(group.relative_target_xy_body_m, dtype=np.float64)
        contexts = tuple(group.context_rgb_artifact_ids)
        branches = tuple(group.branches)
        if (
            target.shape != (2,)
            or not np.isfinite(target).all()
            or len(contexts) != calibration.CONTEXT_FRAME_COUNT
            or len(branches) != benchmark.ACTION_COUNT
        ):
            raise CompleteTieDiagnosticError("eval group geometry changed")
        try:
            branches = tuple(sorted(branches, key=lambda branch: int(branch.action_id)))
        except (AttributeError, TypeError, ValueError) as error:
            raise CompleteTieDiagnosticError("branch actions are malformed") from error
        if tuple(branch.action_id for branch in branches) != tuple(
            range(benchmark.ACTION_COUNT)
        ):
            raise CompleteTieDiagnosticError("branches must contain the exact nine actions")
        context_indices = tuple(append_artifact(value) for value in contexts)
        target_indices = tuple(
            append_artifact(branch.target_rgb_artifact_id) for branch in branches
        )
        ranks = tuple(branch.oracle_dense_rank for branch in branches)
        # Sole domain adaptation: unlike the predecessor builder, max(ranks)==0
        # is valid for eval.  Integer, nonnegative, nine-action ranks are unchanged.
        if any(type(rank) is not int or rank < 0 for rank in ranks):
            raise CompleteTieDiagnosticError("dense ranks are invalid")
        try:
            labels = tuple(calibration._labels(branch) for branch in branches)  # noqa: SLF001
        except Exception as error:
            raise CompleteTieDiagnosticError("physical labels are malformed") from error
        states.append(
            calibration.RoleStateIndexV1(
                role_state_index=role_state_index,
                state_id=state_id,
                role=role,
                family=family,
                scene_id=scene_id,
                group_index=group_index,
                state_index_in_scene=state_index_in_scene,
                relative_target_xy_body_m=(float(target[0]), float(target[1])),
                context_artifact_indices=context_indices,  # type: ignore[arg-type]
                target_artifact_indices=target_indices,
                dense_ranks=ranks,
                target_progress_m=tuple(item[0] for item in labels),
                physical_fell=tuple(item[1] for item in labels),
                physical_tipped=tuple(item[2] for item in labels),
            )
        )
    families = Counter(state.family for state in states)
    scenes = {(state.family, state.scene_id) for state in states}
    scenes_by_family = Counter(family for family, _scene in scenes)
    if (
        set(families) != set(calibration.FAMILIES)
        or any(families[family] != 16 for family in calibration.FAMILIES)
        or len(scenes) != 32
        or any(scenes_by_family[family] != 4 for family in calibration.FAMILIES)
        or len(artifact_ids) != benchmark.STATE_COUNT * (3 + benchmark.ACTION_COUNT)
    ):
        raise CompleteTieDiagnosticError("eval role balance changed")
    identity_document = {
        "role": role,
        "artifact_ids": artifact_ids,
        "states": [
            {
                "state_id": state.state_id,
                "family": state.family,
                "scene_id": state.scene_id,
                "group_index": state.group_index,
                "state_index_in_scene": state.state_index_in_scene,
                "target": list(state.relative_target_xy_body_m),
                "contexts": list(state.context_artifact_indices),
                "targets": list(state.target_artifact_indices),
                "dense_ranks": list(state.dense_ranks),
            }
            for state in states
        ],
    }
    return calibration.RoleFeaturePlanV1(
        role=role,
        artifact_ids=tuple(artifact_ids),
        artifact_index_by_id=MappingProxyType(artifact_index_by_id),
        states=tuple(states),
        groups=ordered,
        identity_sha256=hashlib.sha256(
            frozen.canonical_bytes_v1(identity_document)
        ).hexdigest(),
    )


@dataclass
class EvaluationOnlyLedgerV1:
    """Account for metadata-only train reconstruction and eval-only DINO RGB."""

    stage: str = "created"
    checkpoint_rehashed: bool = False
    receipt_loads: dict[str, int] = field(
        default_factory=lambda: {"train": 0, "eval": 0}
    )
    role_index_opens: dict[str, int] = field(
        default_factory=lambda: {"train": 0, "eval": 0}
    )
    state_receipt_opens: dict[str, int] = field(
        default_factory=lambda: {"train": 0, "eval": 0}
    )
    render_receipt_opens: dict[str, int] = field(
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
    opened_receipts: set[tuple[str, str]] = field(default_factory=set)
    opened_artifacts: set[tuple[str, str]] = field(default_factory=set)

    def load_receipts(self, role: str) -> None:
        if role == "train" and self.stage == "created":
            self.stage = "train"
        elif role == "eval" and self.stage == "checkpoint" and self.checkpoint_rehashed:
            self.stage = "eval"
        else:
            raise CompleteTieDiagnosticError("role receipts opened outside diagnostic stage")
        self.receipt_loads[role] = 1

    def open_role_index(self, role: str, path: str) -> None:
        if self.receipt_loads.get(role) != 1 or self.role_index_opens[role] or not path:
            raise CompleteTieDiagnosticError("role index opened outside diagnostic stage")
        self.role_index_opens[role] = 1

    def open_state_receipt(self, role: str, path: str) -> None:
        key = (role, path)
        if self.receipt_loads.get(role) != 1 or not path or key in self.opened_receipts:
            raise CompleteTieDiagnosticError("state receipt opened outside diagnostic stage")
        self.opened_receipts.add(key)
        self.state_receipt_opens[role] += 1

    def open_render_receipt(self, role: str, path: str) -> None:
        if self.receipt_loads.get(role) != 1 or not path:
            raise CompleteTieDiagnosticError("render receipt opened outside diagnostic stage")
        self.render_receipt_opens[role] += 1

    def open_rgb(self, role: str, kind: str, artifact_id: str) -> None:
        if role != "eval" or kind != "context" or self.stage != "eval":
            raise CompleteTieDiagnosticError("only evaluation context RGB is authorized")
        key = (role, artifact_id)
        if not artifact_id or key in self.opened_artifacts:
            raise CompleteTieDiagnosticError("evaluation context artifact repeated")
        self.opened_artifacts.add(key)
        self.rgb_opens["eval_context"] += 1

    def checkpoint(self) -> None:
        if self.stage != "train" or self.checkpoint_rehashed:
            raise CompleteTieDiagnosticError("checkpoint reuse order changed")
        self.checkpoint_rehashed = True
        self.stage = "checkpoint"

    def finalized(self) -> dict[str, Any]:
        audit = {
            "stage": self.stage,
            "checkpoint_rehashed": self.checkpoint_rehashed,
            "receipt_loads": dict(self.receipt_loads),
            "role_index_opens": dict(self.role_index_opens),
            "state_receipt_opens": dict(self.state_receipt_opens),
            "render_receipt_opens": dict(self.render_receipt_opens),
            "rgb_opens": dict(self.rgb_opens),
            "unique_context_artifacts": len(self.opened_artifacts),
            "successor_rgb_open_count": (
                self.rgb_opens["train_successor"] + self.rgb_opens["eval_successor"]
            ),
        }
        if (
            self.stage != "eval"
            or not self.checkpoint_rehashed
            or self.receipt_loads != {"train": 1, "eval": 1}
            or self.role_index_opens != {"train": 1, "eval": 1}
            or self.state_receipt_opens != {"train": 128, "eval": 128}
            or self.render_receipt_opens != {"train": 32, "eval": 32}
            or self.rgb_opens
            != {
                "train_context": 0,
                "train_successor": 0,
                "eval_context": 384,
                "eval_successor": 0,
            }
            or len(self.opened_artifacts) != 384
        ):
            raise CompleteTieDiagnosticError("evaluation-only access accounting changed")
        return audit


def _load_eval_role_runtime_data_complete_ties_v1(
    authority: Mapping[str, Any],
    plan_document: Mapping[str, Any],
    physics_index: Mapping[str, Any],
    *,
    ledger: EvaluationOnlyLedgerV1,
) -> frozen_runner.RoleRuntimeDataV1:
    """Reconstruct all 128 eval states without the predecessor max-rank guard."""

    role = "eval"
    collection_root = Path(str(authority["collection_root"]))
    declarations = [row for row in plan_document["states"] if row["role"] == role]
    raw_state_bindings = physics_index["state_receipt_bindings"]
    role_state_bindings = [
        value
        for value in raw_state_bindings
        if ("scenes", role)
        in zip(
            PurePosixPath(str(value.get("path"))).parts,
            PurePosixPath(str(value.get("path"))).parts[1:],
        )
    ]
    if len(declarations) != 128 or len(role_state_bindings) != 128:
        raise CompleteTieDiagnosticError("eval receipt declarations changed")
    ledger.open_role_index(role, str(physics_index["_binding"]["path"]))
    receipts = []
    for index, (declared, raw_binding) in enumerate(
        zip(declarations, role_state_bindings, strict=True)
    ):
        binding = frozen_runner._historical_binding(  # noqa: SLF001
            raw_binding, root=collection_root, label="eval state receipt"
        )
        ledger.open_state_receipt(role, str(binding["path"]))
        receipt = frozen_runner._read_bound_json_once(  # noqa: SLF001
            binding, label=f"eval state receipt {index}"
        )
        state = receipt.get("state")
        if (
            receipt.get("status") != "PHYSICS_COMPLETE"
            or not isinstance(state, Mapping)
            or any(
                state.get(name) != declared.get(name)
                for name in (
                    "role",
                    "state_id",
                    "scene_id",
                    "family",
                    "group_index",
                    "state_index_in_scene",
                )
            )
        ):
            raise CompleteTieDiagnosticError("eval state receipt identity changed")
        context = receipt.get("context")
        branches = receipt.get("branches")
        action_catalog = plan_document.get("action_catalog")
        if (
            not isinstance(context, Mapping)
            or context.get("history_action_ids") != declared.get("history_action_ids")
            or state.get("target_xy_m") != declared.get("target_xy_m")
            or state.get("scene_manifest_binding")
            != declared.get("scene_manifest_binding")
            or state.get("scene_genesis_binding")
            != declared.get("scene_genesis_binding")
            or not isinstance(branches, list)
            or [branch.get("action_id") for branch in branches]
            != declared.get("candidate_action_ids")
            or not isinstance(action_catalog, list)
            or len(action_catalog) != 9
            or [branch.get("requested_block") for branch in branches]
            != [
                action_catalog[action_id].get("requested_block")
                for action_id in range(9)
            ]
        ):
            raise CompleteTieDiagnosticError(
                "eval receipt disagrees with planned history/action/source"
            )
        receipts.append(receipt)
    groups, receipt_by_id = physical._groups_from_receipts(  # noqa: SLF001
        receipts, role=role
    )
    role_plan = build_eval_role_feature_plan_complete_ties_v1(groups, role=role)
    physical_inputs, targets = physical._role_arrays(  # noqa: SLF001
        role_plan, receipt_by_id
    )
    histories = []
    candidates = []
    for state in role_plan.states:
        receipt = receipt_by_id[state.state_id]
        history_blocks = receipt["context"].get("history_executed_blocks")
        branches = receipt.get("branches")
        if (
            not isinstance(history_blocks, list)
            or len(history_blocks) != 2
            or not isinstance(branches, list)
            or len(branches) != 9
        ):
            raise CompleteTieDiagnosticError("eval command tape geometry changed")
        histories.append(
            torch.stack(
                [upstream.command_tape_channel_major_v1(block) for block in history_blocks]
            )
        )
        candidates.append(
            torch.stack(
                [
                    upstream.command_tape_channel_major_v1(
                        branch.get("requested_block")
                    )
                    for branch in branches
                ]
            )
        )
    history_commands = torch.stack(histories)
    candidate_commands = torch.stack(candidates)
    goals = torch.tensor(
        [state.relative_target_xy_body_m for state in role_plan.states],
        dtype=torch.float32,
    )
    ranks = torch.tensor(
        [state.dense_ranks for state in role_plan.states], dtype=torch.long
    )
    context_ids = tuple(
        tuple(role_plan.artifact_ids[index] for index in state.context_artifact_indices)
        for state in role_plan.states
    )
    wanted = set(itertools.chain.from_iterable(context_ids))
    raw_render_bindings = physics_index["render_receipt_bindings"]
    role_render_bindings = [
        value
        for value in raw_render_bindings
        if ("scenes", role)
        in zip(
            PurePosixPath(str(value.get("path"))).parts,
            PurePosixPath(str(value.get("path"))).parts[1:],
        )
    ]
    if len(role_render_bindings) != 32:
        raise CompleteTieDiagnosticError("eval render receipt count changed")
    context_artifacts: dict[str, Mapping[str, Any]] = {}
    stored_rgb_bytes = 0
    stored_rgb_frames = 0
    for index, raw_binding in enumerate(role_render_bindings):
        binding = frozen_runner._historical_binding(  # noqa: SLF001
            raw_binding, root=collection_root, label="eval render receipt"
        )
        ledger.open_render_receipt(role, str(binding["path"]))
        render = frozen_runner._read_bound_json_once(  # noqa: SLF001
            binding, label=f"eval render receipt {index}"
        )
        frames = render.get("frame_receipts")
        scene = render.get("scene")
        if (
            render.get("status") != "RENDER_COMPLETE"
            or not isinstance(scene, Mapping)
            or scene.get("role") != role
            or not isinstance(frames, list)
            or len(frames) != 48
        ):
            raise CompleteTieDiagnosticError("eval render receipt changed")
        for frame in frames:
            if (
                not isinstance(frame, Mapping)
                or type(frame.get("byte_count")) is not int
                or int(frame["byte_count"]) <= 0
            ):
                raise CompleteTieDiagnosticError("RGB frame byte count changed")
            stored_rgb_bytes += int(frame["byte_count"])
            stored_rgb_frames += 1
            artifact_id = frame.get("artifact_id")
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
                raise CompleteTieDiagnosticError("eval context RGB metadata changed")
            context_artifacts[str(artifact_id)] = MappingProxyType(dict(frame))
    if set(context_artifacts) != wanted or len(context_artifacts) != 384:
        raise CompleteTieDiagnosticError("eval context RGB closure changed")
    identity = upstream._role_identity_v1(  # noqa: SLF001
        role,
        role_plan,
        physical_inputs,
        targets,
        history_commands,
        candidate_commands,
    )
    result = frozen_runner.RoleRuntimeDataV1(
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


def random_expected_report_complete_ties_v1(plan: Any) -> dict[str, object]:
    """Frozen random expectation with a total denominator on all-zero ranks."""

    rows = []
    for state in plan.states:
        ranks = np.asarray(state.dense_ranks, dtype=np.float64)
        if ranks.shape != (benchmark.ACTION_COUNT,) or not np.isfinite(ranks).all():
            raise CompleteTieDiagnosticError("random-expected ranks changed")
        denominator = max(1.0, float(ranks.max()))
        rows.append(
            {
                "state_id": state.state_id,
                "scene_id": state.scene_id,
                "family": state.family,
                "selected_action_id": "NOT_APPLICABLE",
                "normalized_rank_regret": float(ranks.mean() / denominator),
                "oracle_equivalent_selection_rate": float((ranks == ranks.min()).mean()),
                "physical_target_progress_m": "NOT_APPLICABLE",
                "physical_path_length_m": "NOT_APPLICABLE",
            }
        )

    def summarize(selected: Sequence[Mapping[str, object]]) -> dict[str, object]:
        return {
            "states": len(selected),
            "normalized_rank_regret": float(
                np.mean([row["normalized_rank_regret"] for row in selected])
            ),
            "oracle_equivalent_selection_rate": float(
                np.mean([row["oracle_equivalent_selection_rate"] for row in selected])
            ),
            "physical_target_progress_m": "NOT_APPLICABLE",
            "physical_path_length_m": "NOT_APPLICABLE",
            "chosen_action_histogram": "NOT_APPLICABLE",
        }

    return {
        "selection_policy": "uniform_random_expectation_no_realized_action",
        "summary": summarize(rows),
        "group_results": rows,
        "per_family": {
            family: summarize([row for row in rows if row["family"] == family])
            for family in calibration.FAMILIES
        },
        "per_scene": [
            {
                "scene_id": scene,
                "family": next(
                    str(row["family"]) for row in rows if row["scene_id"] == scene
                ),
                **summarize([row for row in rows if row["scene_id"] == scene]),
            }
            for scene in sorted({str(row["scene_id"]) for row in rows})
        ],
    }


def _complete_tie_summary(eval_role: Any) -> dict[str, Any]:
    rows = []
    for state in eval_role.plan.states:
        ranks = tuple(int(rank) for rank in state.dense_ranks)
        if max(ranks) == 0:
            rows.append(
                {
                    "state_id": state.state_id,
                    "scene_id": state.scene_id,
                    "family": state.family,
                    "group_index": state.group_index,
                    "dense_ranks": list(ranks),
                }
            )
    if len(eval_role.plan.states) != 128 or len(rows) != 4:
        raise CompleteTieDiagnosticError("expected four complete-tie states among 128 eval states")
    return {
        "eval_state_count": 128,
        "retained_eval_state_count": 128,
        "excluded_eval_state_count": 0,
        "complete_tie_state_count": 4,
        "complete_tie_states": rows,
        "all_actions_oracle_equivalent_for_complete_ties": True,
        "normalized_regret_denominator": "max(1,max_dense_rank)",
    }


def evaluate_checkpoint_complete_ties_v1(
    checkpoint: Mapping[str, object],
    train_role: Any,
    eval_role: Any,
    eval_context_tokens: torch.Tensor,
    *,
    device: torch.device,
    integrity_passed: bool,
) -> dict[str, object]:
    """Evaluate the unchanged checkpoint with only the complete-tie rule added."""

    if checkpoint.get("identity_sha256") != benchmark.checkpoint_identity_v1(checkpoint):
        raise CompleteTieDiagnosticError("checkpoint identity changed")
    if checkpoint.get("config") != frozen.config_v1():
        raise CompleteTieDiagnosticError("checkpoint protocol changed")
    if (
        checkpoint.get("train_plan_identity_sha256")
        != str(train_role.plan.identity_sha256)
        or checkpoint.get("train_role_identity_sha256") != str(train_role.identity_sha256)
    ):
        raise CompleteTieDiagnosticError("checkpoint training role changed")
    train_geometry = benchmark.validate_role_scene_geometry_v1(train_role)
    eval_geometry = benchmark.validate_role_scene_geometry_v1(eval_role)
    tie_summary = _complete_tie_summary(eval_role)
    if train_role.role != "train" or eval_role.role != "eval" or device.type != "cpu":
        raise CompleteTieDiagnosticError("role or CPU evaluation contract changed")
    frozen.validate_context_tokens_v1(eval_context_tokens, role="eval")
    projected_eval_context = frozen.project_context_tokens_v1(
        eval_context_tokens, checkpoint["visual_projection"], role="eval"
    )
    reports: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    for arm in benchmark.ARM_ORDER:
        predictions: list[torch.Tensor] = []
        members: list[dict[str, object]] = []
        for member in checkpoint["arms"][arm]:
            outcomes = frozen.predict_member_v1(
                member["state_dict"],
                eval_role,
                projected_eval_context,
                checkpoint["input_statistics"],
                checkpoint["outcome_statistics"],
                arm=arm,
                device=device,
            )
            scores = grounded.physical_score_matrix_v1(eval_role.plan, outcomes)
            report = grounded.report_physical_scores_v1(eval_role.plan, scores)
            predictions.append(outcomes)
            members.append(
                {
                    "seed": member["seed"],
                    "outcome_identity_sha256": frozen._tensor_sha256(outcomes),  # noqa: SLF001
                    "score_identity_sha256": physical._array_identity_v1(scores),  # noqa: SLF001
                    "report": report,
                }
            )
        ensemble = torch.stack(predictions).mean(dim=0)
        scores = grounded.physical_score_matrix_v1(eval_role.plan, ensemble)
        reports[arm] = grounded.report_physical_scores_v1(eval_role.plan, scores)
        artifacts[arm] = {
            "ensemble_outcomes": ensemble.tolist(),
            "ensemble_outcome_identity_sha256": frozen._tensor_sha256(ensemble),  # noqa: SLF001
            "ensemble_scores": scores.tolist(),
            "ensemble_score_identity_sha256": physical._array_identity_v1(scores),  # noqa: SLF001
            "members": members,
        }
        diagnostics[arm] = benchmark._prediction_diagnostics_v1(  # noqa: SLF001
            ensemble,
            eval_role.targets,
            checkpoint["outcome_statistics"]["residual_scales"],
        )
    task = frozen._fit_task_control_v1(train_role.plan)  # noqa: SLF001
    task_scores = frozen._score_task_control_v1(eval_role.plan, task)  # noqa: SLF001
    reports["task_action_only"] = grounded.report_physical_scores_v1(
        eval_role.plan, task_scores
    )
    task_regret = float(
        reports["task_action_only"]["summary"]["normalized_rank_regret"]
    )
    oracle_scores = np.asarray(
        [state.dense_ranks for state in eval_role.plan.states], dtype=np.float64
    )
    reports["privileged_physical_oracle"] = grounded.report_physical_scores_v1(
        eval_role.plan, oracle_scores
    )
    reports["random_expected"] = random_expected_report_complete_ties_v1(
        eval_role.plan
    )
    comparisons = {
        "visual_vs_task_action_only": benchmark.paired_family_scene_bootstrap_v1(
            reports[benchmark.VISUAL_ARM]["group_results"],
            reports["task_action_only"]["group_results"],
        ),
        "visual_vs_no_vision": benchmark.paired_family_scene_bootstrap_v1(
            reports[benchmark.VISUAL_ARM]["group_results"],
            reports[benchmark.NO_VISION_ARM]["group_results"],
        ),
        "no_vision_vs_task_action_only": benchmark.paired_family_scene_bootstrap_v1(
            reports[benchmark.NO_VISION_ARM]["group_results"],
            reports["task_action_only"]["group_results"],
        ),
    }
    frozen_gate = grounded.fixed_gate_v1(
        joint_report=reports[benchmark.VISUAL_ARM],
        task_report=reports["task_action_only"],
        matched_report=reports[benchmark.NO_VISION_ARM],
        random_report=reports["random_expected"],
        oracle_report=reports["privileged_physical_oracle"],
        joint_vs_task=comparisons["visual_vs_task_action_only"],
        joint_vs_matched=comparisons["visual_vs_no_vision"],
        integrity_passed=integrity_passed,
    )
    passed = bool(frozen_gate["passed"])
    gate = {
        "schema": "lewm_go2_scene_diversity_recurrent_replication_fixed_gate_v1",
        "passed": passed,
        "status": benchmark.PASS_STATUS if passed else benchmark.STOP_STATUS,
        "gates": {
            "1_integrity_and_oracle": frozen_gate["gates"]["1_integrity_and_oracle"],
            "2_absolute_regret": frozen_gate["gates"]["2_absolute_regret"],
            "3_visual_beats_task_action_only": frozen_gate["gates"][
                "3_joint_beats_task_action_only"
            ],
            "4_visual_beats_no_vision": frozen_gate["gates"][
                "4_joint_beats_matched_physical_only"
            ],
            "5_visual_beats_random": frozen_gate["gates"]["5_joint_beats_random"],
        },
        "threshold_source": frozen_gate["schema"],
        "thresholds": frozen.config_v1()["frozen_h1_thresholds"],
    }
    return {
        "schema": EVALUATION_SCHEMA,
        "status": gate["status"],
        "claim_scope": "POST_HOC_DEVELOPMENT_DIAGNOSTIC_NONCONFIRMATORY",
        "checkpoint_identity_sha256": checkpoint["identity_sha256"],
        "train_role_identity_sha256": str(train_role.identity_sha256),
        "eval_role_identity_sha256": str(eval_role.identity_sha256),
        "role_geometry": {"train": train_geometry, "eval": eval_geometry},
        "complete_tie_domain": tie_summary,
        "reports": reports,
        "comparisons": comparisons,
        "gate": gate,
        "prediction_artifacts": artifacts,
        "prediction_diagnostics": diagnostics,
        "task_control": {
            "live_identity_sha256": task.identity_sha256,
            "behavioral_eval_regret": task_regret,
            "historical_exact_regret_assertion": {
                "status": benchmark.TASK_CONTROL_ASSERTION_STATUS,
                "historical_value": physical.EXPECTED_TASK_EVAL_REGRET,
                "reason": "the assertion binds the historical V1 role, not fresh scenes",
            },
            "same_task_relative_gate_thresholds_applied": True,
        },
        "successor_observations_opened": 0,
        "authorizes_blind_rollout_preregistration": False,
        "authorizes_navigation_claim": False,
        "citable_as_scientific_evidence": False,
    }


def _load_predecessor_inputs(
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Mapping[str, Any]]:
    _validate_predecessor_terminal(plan)
    _validate_predecessor_terminal_review(plan)
    predecessor = plan["predecessor"]
    predecessor_plan = _read_bound_json(
        predecessor["scientific_plan_binding"], label="predecessor scientific plan"
    )
    physics = _read_bound_json(
        predecessor["physics_result_binding"], label="predecessor physics result"
    )
    source_review_binding = _standard_binding(physics.get("authority_binding", {}))
    _require_exact_binding(
        source_review_binding,
        expected=EXPECTED_PREDECESSOR_SOURCE_REVIEW_BINDING,
        label="predecessor scientific source review",
    )
    if (
        physics.get("status") != "PHYSICS_COMPLETE"
        or physics.get("failure") is not None
        or physics.get("allows_refill") is not False
        or physics.get("allows_overwrite") is not False
        or physics.get("authorizes_retry_or_resume") is not False
    ):
        raise CompleteTieDiagnosticError("predecessor physics result changed")
    if not isinstance(physics.get("source_bindings"), Mapping):
        raise CompleteTieDiagnosticError("predecessor source closure is absent")
    for name, binding in physics["source_bindings"].items():
        normalized = _standard_binding(binding)
        if file_binding_v1(Path(normalized["path"])) != normalized:
            raise CompleteTieDiagnosticError(f"predecessor source changed: {name}")
    dino = frozen_runner.expected_dino_v1()
    frozen_runner._require_binding(dino["checkpoint_binding"], label="frozen DINO checkpoint")  # noqa: SLF001
    frozen_runner._validate_dino_source_v1()  # noqa: SLF001
    authority = {
        "attempt_id": predecessor_plan["attempt_id"],
        "attempt_root": str(PREDECESSOR_ATTEMPT_ROOT.resolve(strict=True)),
        "collection_root": str(PREDECESSOR_COLLECTION_ROOT.resolve(strict=True)),
        "caps": physics["caps"],
        "source_bindings": physics["source_bindings"],
        "plan_binding": predecessor["scientific_plan_binding"],
        "source_review_binding": source_review_binding,
        "dino": dino,
    }
    physics_index = frozen_runner._load_physics_index_v1(  # noqa: SLF001
        authority, source_review_binding, predecessor_plan
    )
    if _standard_binding(physics_index["_binding"]) != _standard_binding(
        predecessor["physics_result_binding"]
    ):
        raise CompleteTieDiagnosticError("predecessor physics binding changed")
    checkpoint_binding = _require_exact_binding(
        predecessor["checkpoint_binding"],
        expected=EXPECTED_BINDINGS["checkpoint_binding"],
        label="predecessor checkpoint",
    )
    checkpoint = torch.load(
        Path(checkpoint_binding["path"]), map_location="cpu", weights_only=True
    )
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("identity_sha256")
        != benchmark.checkpoint_identity_v1(checkpoint)
        or checkpoint.get("config") != frozen.config_v1()
    ):
        raise CompleteTieDiagnosticError("predecessor checkpoint content changed")
    return authority, physics_index, checkpoint


def _reserve_attempt(
    *, plan_binding: Mapping[str, Any], source_review_binding: Mapping[str, Any]
) -> dict[str, Any]:
    if ATTEMPT_ROOT.exists() or ATTEMPT_ROOT.is_symlink():
        raise CompleteTieDiagnosticError("fresh diagnostic root is not fresh")
    ATTEMPT_ROOT.mkdir(mode=0o700, parents=True)
    reservation = {
        "schema": RESERVATION_SCHEMA,
        "attempt_id": ATTEMPT_ID,
        "plan_binding": dict(plan_binding),
        "source_review_binding": dict(source_review_binding),
        "fresh_root_reserved": True,
        "post_hoc_nonconfirmatory": True,
        "authorizes_retry_or_resume": False,
    }
    _write_json_exclusive(RESERVATION_PATH, reservation)
    return file_binding_v1(RESERVATION_PATH)


def execute_diagnostic_v1(
    *,
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    source_review_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute one fresh read-only diagnostic; callers must clear source first."""

    reservation_binding = _reserve_attempt(
        plan_binding=plan_binding, source_review_binding=source_review_binding
    )
    started = time.monotonic()
    authority, physics_index, checkpoint = _load_predecessor_inputs(plan)
    if (
        torch.cuda.is_available() is not False
        or torch.cuda.device_count() != 0
        or torch.version.hip is not None
    ):
        raise CompleteTieDiagnosticError("bound diagnostic interpreter is not CPU-only")
    device = torch.device("cpu")
    determinism = upstream.configure_determinism_v1()
    ledger = EvaluationOnlyLedgerV1()
    ledger.load_receipts("train")
    train = frozen_runner._load_role_runtime_data_v1(  # noqa: SLF001
        authority, plan_document=_read_bound_json(
            plan["predecessor"]["scientific_plan_binding"],
            label="predecessor scientific plan",
        ), physics_index=physics_index, role="train", ledger=ledger
    )
    if (
        checkpoint.get("train_plan_identity_sha256") != str(train.plan.identity_sha256)
        or checkpoint.get("train_role_identity_sha256") != str(train.identity_sha256)
    ):
        raise CompleteTieDiagnosticError("reconstructed train role changed")
    ledger.checkpoint()
    ledger.load_receipts("eval")
    predecessor_plan = _read_bound_json(
        plan["predecessor"]["scientific_plan_binding"],
        label="predecessor scientific plan",
    )
    evaluation = _load_eval_role_runtime_data_complete_ties_v1(
        authority, predecessor_plan, physics_index, ledger=ledger
    )
    disjointness = frozen_runner.assert_role_disjointness_v1(
        train.plan, evaluation.plan
    )
    tie_summary = _complete_tie_summary(evaluation)
    dino = upstream.load_dino_trunk_v1(
        Path(str(authority["dino"]["repository_path"])),
        Path(str(authority["dino"]["checkpoint_binding"]["path"])),
        device=device,
    )
    if dino.device.type != "cpu":
        raise CompleteTieDiagnosticError("DINO device changed")
    eval_context = frozen_runner._full_dino_context_tokens_v1(  # noqa: SLF001
        evaluation, ledger=ledger, dino=dino
    )
    first = evaluate_checkpoint_complete_ties_v1(
        checkpoint,
        train,
        evaluation,
        eval_context,
        device=device,
        integrity_passed=True,
    )
    second = evaluate_checkpoint_complete_ties_v1(
        checkpoint,
        train,
        evaluation,
        eval_context,
        device=device,
        integrity_passed=True,
    )
    if canonical_bytes_v1(first) != canonical_bytes_v1(second):
        raise CompleteTieDiagnosticError("repeat evaluation was not bitwise exact")
    access_audit = ledger.finalized()
    result = {
        "schema": RESULT_SCHEMA,
        "status": COMPLETE_STATUS,
        "development_only": True,
        "post_hoc_nonconfirmatory": True,
        "citable_as_scientific_evidence": False,
        "authorizes_navigation_claim": False,
        "authorizes_blind_rollout_preregistration": False,
        "authorizes_retry_or_resume": False,
        "plan_binding": dict(plan_binding),
        "source_review_binding": dict(source_review_binding),
        "reservation_binding": reservation_binding,
        "predecessor_bindings": dict(plan["predecessor"]),
        "physics_result_binding": _standard_binding(physics_index["_binding"]),
        "checkpoint_binding": dict(plan["predecessor"]["checkpoint_binding"]),
        "checkpoint_identity_sha256": checkpoint["identity_sha256"],
        "domain_change": {
            "sole_change": "eval_dense_rank_domain_admits_max_rank_zero",
            "random_expected_denominator": "max(1,max_dense_rank)",
            "scene_exclusion": False,
            "rank_tolerance_changed": False,
            "actions_changed": False,
            "model_changed": False,
            "seeds_changed": False,
            "bootstrap_changed": False,
            "thresholds_changed": False,
        },
        "complete_tie_summary": tie_summary,
        "evaluation": first,
        "repeat_evaluation_exact": True,
        "role_disjointness": disjointness,
        "access_audit": access_audit,
        "runtime": {
            "determinism": determinism,
            "torch": torch.__version__,
            "compute_device": "cpu",
            "dino_context_role": "eval_only",
        },
        "successor_observations_opened": 0,
        "wall_seconds": time.monotonic() - started,
    }
    _write_json_exclusive(RESULT_PATH, result)
    result_binding = file_binding_v1(RESULT_PATH)
    _write_json_exclusive(
        TERMINAL_PATH,
        {
            "schema": TERMINAL_SCHEMA,
            "status": COMPLETE_STATUS,
            "result_binding": result_binding,
            "fixed_gate_status": first["gate"]["status"],
            "post_hoc_nonconfirmatory": True,
            "citable_as_scientific_evidence": False,
            "authorizes_navigation_claim": False,
            "authorizes_retry_or_resume": False,
            "failure": None,
        },
    )
    return result


def _write_failure_terminal(error: Exception) -> None:
    if not ATTEMPT_ROOT.is_dir() or TERMINAL_PATH.exists() or TERMINAL_PATH.is_symlink():
        return
    try:
        _write_json_exclusive(
            TERMINAL_PATH,
            {
                "schema": TERMINAL_SCHEMA,
                "status": FAIL_STATUS,
                "result_binding": None,
                "post_hoc_nonconfirmatory": True,
                "citable_as_scientific_evidence": False,
                "authorizes_navigation_claim": False,
                "authorizes_retry_or_resume": False,
                "failure": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            },
        )
    except Exception:
        return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--expected-plan-byte-count", type=int, required=True)
    parser.add_argument("--source-review", type=Path, required=True)
    parser.add_argument("--expected-source-review-sha256", required=True)
    parser.add_argument("--expected-source-review-byte-count", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan, plan_binding = read_and_validate_plan_v1(
            args.plan,
            expected_sha256=args.expected_plan_sha256,
            expected_byte_count=args.expected_plan_byte_count,
        )
        _review, source_review_binding = read_and_validate_source_review_v1(
            args.source_review,
            expected_sha256=args.expected_source_review_sha256,
            expected_byte_count=args.expected_source_review_byte_count,
            plan_binding=plan_binding,
        )
        result = execute_diagnostic_v1(
            plan=plan,
            plan_binding=plan_binding,
            source_review_binding=source_review_binding,
        )
    except Exception as error:
        _write_failure_terminal(error)
        print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ATTEMPT_ID",
    "ATTEMPT_ROOT",
    "CompleteTieDiagnosticError",
    "EvaluationOnlyLedgerV1",
    "build_eval_role_feature_plan_complete_ties_v1",
    "canonical_bytes_v1",
    "evaluate_checkpoint_complete_ties_v1",
    "random_expected_report_complete_ties_v1",
    "read_and_validate_plan_v1",
    "read_and_validate_source_review_v1",
]
