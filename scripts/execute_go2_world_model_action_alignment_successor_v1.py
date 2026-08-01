#!/usr/bin/env python3
"""Execute the final science-identical action-alignment integrity replacement.

The worker consumes the immutable V3 frame pack and one frozen spatial
predecessor.  It never opens RGB, sealed, held-out, or protected material and
never creates data. Both fresh arms execute the same row-stable action-candidate
route; their only scientific difference is the frozen alignment-loss
coefficient.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks import (  # noqa: E402
    go2_rgb_recurrent_patch_memory_temporal_jepa_v1 as temporal_metrics,
)
from lewm.benchmarks import (  # noqa: E402
    go2_world_model_action_alignment_successor_v1 as successor_metrics,
)
from lewm.benchmarks import (  # noqa: E402
    go2_world_model_existing_pool_three_arm_v1 as three_arm_metrics,
)
from lewm.datasets import (  # noqa: E402
    go2_explicit_plan_discounted_successor_state_v27 as h6,
)
from lewm.models.rgb_single_frame_multiblock_masked_spatial_jepa_v1 import (  # noqa: E402
    _gather_spatial_tokens,
    normalized_half_squared_token_energy_v1,
)
from scripts import dev_train_temporal_jepa_scaled as scaled  # noqa: E402
from scripts import execute_go2_world_model_existing_pool_three_arm_v1 as base  # noqa: E402
from scripts import (  # noqa: E402
    extract_go2_world_model_existing_pool_three_arm_v1_action_localization_v1 as custody,
)


SCHEMA_PREFIX = (
    "lewm_go2_world_model_action_alignment_successor_v1_"
    "integrity_replacement_v1"
)
AUTHORITY_SCHEMA = f"{SCHEMA_PREFIX}_execution_authority_v1"
AUTHORITY_STATUS = (
    "AUTHORIZED_ONE_EXACT_ACTION_ALIGNMENT_INTEGRITY_REPLACEMENT_V1_ATTEMPT"
)
RESERVATION_SCHEMA = f"{SCHEMA_PREFIX}_reservation_v1"
RESULT_SCHEMA = f"{SCHEMA_PREFIX}_result_v1"
SNAPSHOT_SCHEMA = f"{SCHEMA_PREFIX}_snapshot_v1"
METRIC_BUNDLE_SCHEMA = f"{SCHEMA_PREFIX}_metric_bundle_v1"
FAILURE_SCHEMA = f"{SCHEMA_PREFIX}_failure_v1"
REVIEW_SCHEMA = f"{SCHEMA_PREFIX}_independent_source_review_v1"
REVIEW_STATUS = "PASS_SOURCE_ONLY_NOT_AUTHORITY"

ATTEMPT_ID = (
    "world_model_action_alignment_successor_v1_integrity_replacement_v1/"
    "attempt_v1"
)
ATTEMPT_ROOT = (
    REPO_ROOT
    / ".generated/dev/world_model_action_alignment_successor_v1_"
    "integrity_replacement_v1/attempt_v1"
)
ORIGINAL_ATTEMPT_ROOT = (
    REPO_ROOT / ".generated/dev/world_model_action_alignment_successor_v1/attempt_v1"
)
AUTHORITY_PATH = (
    REPO_ROOT
    / "docs/lewm_go2_world_model_action_alignment_successor_v1_"
    "integrity_replacement_v1_execution_authority_2026-08-01.json"
)
PREREGISTRATION_PATH = (
    REPO_ROOT
    / "docs/lewm_go2_world_model_action_alignment_successor_v1_"
    "integrity_replacement_v1_preregistration_2026-08-01.md"
)
PLAN_PATH = (
    REPO_ROOT
    / "docs/lewm_go2_world_model_action_alignment_successor_v1_"
    "integrity_replacement_v1_plan_2026-08-01.json"
)
REVIEW_PATH = (
    REPO_ROOT
    / "docs/lewm_go2_world_model_action_alignment_successor_v1_"
    "integrity_replacement_v1_independent_source_review_2026-08-01.json"
)
WORKER_PATH = Path(__file__).resolve()
CHECKER_PATH = (
    REPO_ROOT / "scripts/check_go2_world_model_action_alignment_successor_v1.py"
)
SUPERVISOR_PATH = (
    REPO_ROOT / "scripts/run_go2_world_model_action_alignment_successor_authorized_v1.py"
)

ARM_NAMES = successor_metrics.ARM_NAMES
ARM_COEFFICIENTS = {"baseline": 0.0, "alignment": 1.0}
TRAINING_UPDATES = 700
BATCH_SIZE = 256
MICROBATCH_SIZE = 32
TAIL_UPDATES = successor_metrics.TAIL_UPDATES
EVALUATION_BATCH_SIZE = 64
ALIGNMENT_MARGIN = 0.01
CANDIDATE_SCAN_BATCH_ROWS = 32
MAXIMUM_WALL_SECONDS = 9_000
MAXIMUM_GPU_SECONDS = 7_200
EXPECTED_TRAIN_ROWS = 16_000
EXPECTED_VALIDATION_ROWS = 2_048
ACTION_COUNT = 9

V3_ATTEMPT_ROOT = (
    REPO_ROOT
    / ".generated/dev/world_model_existing_pool_three_arm_v1_integrity_replacement_v3/attempt_v1"
)
PACK_ROOT = V3_ATTEMPT_ROOT / "pack"
PREDECESSOR_PATH = base.PREDECESSOR


def _absolute_binding(path: Path, digest: str, count: int) -> dict[str, Any]:
    return {"path": str(path), "file_sha256": digest, "byte_count": count}


EXPECTED_INPUT_BINDINGS = {
    "predecessor_checkpoint": _absolute_binding(
        PREDECESSOR_PATH, base.PREDECESSOR_SHA256, base.PREDECESSOR_BYTE_COUNT
    ),
    "train_index": _absolute_binding(
        REPO_ROOT / h6.TRAIN_INDEX,
        h6.TRAIN_INDEX_SHA256,
        h6.TRAIN_INDEX_BYTES,
    ),
    "validation_index": _absolute_binding(
        REPO_ROOT / h6.VALIDATION_INDEX,
        h6.VALIDATION_INDEX_SHA256,
        h6.VALIDATION_INDEX_BYTES,
    ),
    "pack_manifest": _absolute_binding(
        PACK_ROOT / "manifest.json",
        "22364f911ab5d3e2956ea9a3fc2d92e2869830cd858ef2d2269379dfc6041bae",
        5_297,
    ),
    "pack_train_frames": _absolute_binding(
        PACK_ROOT / "train_frames.u8",
        "df9a5982370f4ba7c5d1c492f080d44f9900d889877ddb73f08454ba151a5a74",
        2_408_448_000,
    ),
    "pack_train_actions": _absolute_binding(
        PACK_ROOT / "train_actions.npy",
        "11bfcd0724397be8fc84969a32c01b71d41fdedb34c75bbc7a9e4d481a934a78",
        384_128,
    ),
    "pack_train_metadata": _absolute_binding(
        PACK_ROOT / "train_meta.json",
        "2f265eaa57979f2e9c49956ab7bf83df29bcbc75d6b2f274f4d9b7b5d9635265",
        928_029,
    ),
    "pack_validation_frames": _absolute_binding(
        PACK_ROOT / "val_frames.u8",
        "e457d244c07516947ffb8005e2477d9a7f48c5e6a03b8701cf994debb06f6d66",
        308_281_344,
    ),
    "pack_validation_actions": _absolute_binding(
        PACK_ROOT / "val_actions.npy",
        "ad1b33d6ff4839736e27d37114bb1c01ca1cae693b5317c055dc9e776a8be6a1",
        49_280,
    ),
    "pack_validation_metadata": _absolute_binding(
        PACK_ROOT / "val_meta.json",
        "6ef0d194c45a60d9cc28806dd8158360ae4ea6da55caf8685bdcdda9cfeff2a4",
        118_813,
    ),
}

EXPECTED_EVIDENCE_BINDINGS = {
    "v3_result": _absolute_binding(
        V3_ATTEMPT_ROOT / "result.json",
        "764ee61b7bb8b7e1221f01fc34ba0554d0ca681fde21e99b1a9f5585b3360bd4",
        26_054,
    ),
    "v3_terminal_review": _absolute_binding(
        REPO_ROOT
        / "docs/lewm_go2_world_model_existing_pool_three_arm_v1_integrity_replacement_v3_terminal_review_2026-08-01.json",
        "457ca867f406fb6cf4db48bbe9d70340be792b4ee79c38902de112c857b091d2",
        24_635,
    ),
    "localization_result": _absolute_binding(
        REPO_ROOT
        / ".generated/dev/world_model_existing_pool_three_arm_v1_action_localization_v1/attempt_v1/localization.json",
        "eed91ff582fa2ecfc83a740b481986129396029cc5b66bce1f141f6e7e8cfea9",
        66_005,
    ),
    "localization_receipt_check": _absolute_binding(
        REPO_ROOT
        / ".generated/dev/world_model_existing_pool_three_arm_v1_action_localization_v1/attempt_v1/receipt_check.json",
        "71b00064e34e11044418221fc204a0f7ad1d48bd60c85e36dd6f1d0901bf5299",
        1_582,
    ),
    "localization_terminal": _absolute_binding(
        REPO_ROOT
        / ".generated/dev/world_model_existing_pool_three_arm_v1_action_localization_v1/attempt_v1/terminal_supervision.json",
        "c97cabc54bacb902a48b3880646fdecdfc83ba772e335e91f08c0cce902c058c",
        4_765,
    ),
    "original_successor_preregistration": _absolute_binding(
        REPO_ROOT
        / "docs/lewm_go2_world_model_action_alignment_successor_v1_"
        "preregistration_2026-08-01.md",
        "9ef74b866314d84c72e3f125ccccd6a3d7827964176c46b7c5ac16775a17dfa1",
        7_122,
    ),
    "original_successor_plan": _absolute_binding(
        REPO_ROOT
        / "docs/lewm_go2_world_model_action_alignment_successor_v1_"
        "plan_2026-08-01.json",
        "ec3bd3987d3cdb3a5611053e3ba057fc7bb637b6d71efebe41687ac4a34f73db",
        2_110,
    ),
    "original_successor_authority": _absolute_binding(
        REPO_ROOT
        / "docs/lewm_go2_world_model_action_alignment_successor_v1_"
        "execution_authority_2026-08-01.json",
        "366cad5821ce68ea7ee8106b15f428d2859134a7e57b1567d7a1d7306b37ff58",
        20_092,
    ),
    "original_successor_reservation": _absolute_binding(
        ORIGINAL_ATTEMPT_ROOT / "reservation.json",
        "f3786cffa9dd840b6b14fbd47ceb931bd9041f91f62408e1c612036c75768540",
        19_792,
    ),
    "original_successor_failure": _absolute_binding(
        ORIGINAL_ATTEMPT_ROOT / "failure.json",
        "d9849b84b3707e650973d048ca8d7ce2e83ee9409ddea9e62fed1ead1f9563d6",
        1_329,
    ),
    "original_successor_terminal": _absolute_binding(
        ORIGINAL_ATTEMPT_ROOT / "terminal_supervision.json",
        "99f7d217739087327108363ff4fe9e3dc7b1cdac9373d48678b51275a198b6ab",
        7_305,
    ),
    "original_successor_failure_audit": _absolute_binding(
        REPO_ROOT
        / "docs/lewm_go2_world_model_action_alignment_successor_v1_"
        "terminal_preupdate_source_integrity_failure_result_2026-08-01.json",
        "3f8350528c4985b792d22b5d4002b3cc34d926c7a2d8a84431009d6668bd63ed",
        7_173,
    ),
}

PUBLIC_V3_BASELINE_ANCHORS = {
    "balanced_accuracy": 0.2469343816883539,
    "balanced_accuracy_lower": 0.23014452836846072,
    "hardest_margin": -0.009453551490358742,
    "hardest_margin_lower": -0.01138311990101325,
    "persistence_point": -0.14645548512800682,
    "persistence_lower": -0.1829122861354923,
    "wrong_history_point": 0.12255093276460897,
    "wrong_history_lower": 0.11766703087321294,
}

EXACT_CHILD_ENVIRONMENT = {
    "PATH": "/usr/bin:/bin",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "HIP_VISIBLE_DEVICES": "0",
    "ROCR_VISIBLE_DEVICES": "0",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONSAFEPATH": "1",
    "OMP_NUM_THREADS": "1",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_OPTIONAL_LOCKS": "0",
}
EXPECTED_RUNTIME = {
    "python_invocation_path": str(
        REPO_ROOT / ".generated/venvs/world_model_rocm_7_2_1_v1/bin/python"
    ),
    "environment": EXACT_CHILD_ENVIRONMENT,
    "bindings": {
        "git_executable": _absolute_binding(
            Path("/usr/bin/git"),
            "2a8c18fbf43da9f692d75474c72bea9dfd796c260b0f3dfe456376abc3bbd668",
            4_066_232,
        ),
        "python_environment_config": _absolute_binding(
            REPO_ROOT / ".generated/venvs/world_model_rocm_7_2_1_v1/pyvenv.cfg",
            "49222cc65a628e83d00d99da60f1dea8d59bc01a3ea9616227f330e2ecd50577",
            223,
        ),
        "python_executable_target": _absolute_binding(
            Path("/usr/bin/python3.12"),
            "1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118",
            8_020_928,
        ),
    },
}

_BASE_SOURCE_PATHS = {
    f"base_{name}": str(path)
    for name, path in base.REQUIRED_SOURCE_PATHS.items()
    if name not in {"worker", "checker", "external_supervisor"}
}
REQUIRED_SOURCE_PATHS = {
    **_BASE_SOURCE_PATHS,
    "base_three_arm_worker": "scripts/execute_go2_world_model_existing_pool_three_arm_v1.py",
    "localization_custody_worker": (
        "scripts/extract_go2_world_model_existing_pool_three_arm_v1_"
        "action_localization_v1.py"
    ),
    "authorized_device_guard": (
        "scripts/run_go2_world_model_existing_pool_three_arm_authorized_v1.py"
    ),
    "localization_metrics": "lewm/benchmarks/go2_world_model_v3_action_localization_v1.py",
    "alignment_metrics": "lewm/benchmarks/go2_world_model_action_alignment_successor_v1.py",
    "worker": "scripts/execute_go2_world_model_action_alignment_successor_v1.py",
    "checker": "scripts/check_go2_world_model_action_alignment_successor_v1.py",
    "external_supervisor": "scripts/run_go2_world_model_action_alignment_successor_authorized_v1.py",
}
REQUIRED_TEST_PATHS = {
    "alignment_metric_tests": "lewm/tests/test_go2_world_model_action_alignment_successor_v1.py",
    "alignment_worker_checker_tests": "lewm/tests/test_execute_go2_world_model_action_alignment_successor_v1.py",
    "alignment_supervisor_tests": "lewm/tests/test_run_go2_world_model_action_alignment_successor_authorized_v1.py",
}

CLAIM_BOUNDARY = [
    "changed-objective exploratory development comparison only",
    "no requested-versus-executed equivalence or untaken-action causal claim",
    "no fresh blind or shuffled confirmation role",
    "no planning, navigation, WM-A, WM-D, promotion, deployment, or production claim",
]
EXPECTED_SUCCESS_FILES_BEFORE_CHECKER = {
    "reservation.json",
    "baseline_update_000700.pt",
    "alignment_update_000700.pt",
    "metrics.pt",
    "result.json",
}


class AlignmentWorkerError(RuntimeError):
    """The exact authority, custody, or experiment contract failed closed."""


def canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    try:
        return custody.canonical_json_bytes(value, pretty=pretty)
    except Exception as error:
        raise AlignmentWorkerError("document is not finite canonical JSON") from error


def strict_json_bytes(raw: bytes) -> Any:
    try:
        return custody.strict_json_bytes(raw)
    except Exception as error:
        raise AlignmentWorkerError("JSON document is invalid") from error


def file_binding(path: Path) -> dict[str, Any]:
    try:
        return custody.file_binding(Path(path))
    except Exception as error:
        raise AlignmentWorkerError(f"could not bind regular file {path}") from error


def write_immutable_json(path: Path, value: Any) -> dict[str, Any]:
    try:
        return custody.write_immutable_json(Path(path), value)
    except Exception as error:
        raise AlignmentWorkerError(f"could not write immutable JSON {path}") from error


def exact_root_inventory(expected: set[str]) -> list[str]:
    try:
        root_stat = ATTEMPT_ROOT.lstat()
    except FileNotFoundError as error:
        raise AlignmentWorkerError("attempt root is absent") from error
    if not stat.S_ISDIR(root_stat.st_mode) or ATTEMPT_ROOT.is_symlink():
        raise AlignmentWorkerError("attempt root is not a real directory")
    observed: list[str] = []
    with os.scandir(ATTEMPT_ROOT) as entries:
        for entry in entries:
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                raise AlignmentWorkerError("attempt root contains a non-file")
            observed.append(entry.name)
    if set(observed) != expected or len(observed) != len(expected):
        raise AlignmentWorkerError(f"attempt inventory changed: {sorted(observed)}")
    return sorted(observed)


def _binding_is_exact(path: Path, binding: Mapping[str, Any]) -> bool:
    try:
        return file_binding(path) == dict(binding)
    except AlignmentWorkerError:
        return False


def _git(*arguments: str, binary: bool = False) -> str | bytes:
    completed = subprocess.run(
        ["/usr/bin/git", *arguments],
        cwd=REPO_ROOT,
        env={**EXACT_CHILD_ENVIRONMENT, "PATH": "/usr/bin:/bin"},
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout if binary else completed.stdout.decode("utf-8").strip()


def _commit(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AlignmentWorkerError(f"{label} is not a full commit")
    return value


def _require_ancestor(left: str, right: str, *, label: str) -> None:
    result = subprocess.run(
        ["/usr/bin/git", "merge-base", "--is-ancestor", left, right],
        cwd=REPO_ROOT,
        env={**EXACT_CHILD_ENVIRONMENT, "PATH": "/usr/bin:/bin"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0 or left == right:
        raise AlignmentWorkerError(f"{label} ancestry is not strict")


def _require_binding_at_commit(
    binding: Mapping[str, Any], commit: str, *, label: str
) -> None:
    path = Path(str(binding.get("path", "")))
    try:
        relative = path.resolve(strict=True).relative_to(REPO_ROOT.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise AlignmentWorkerError(f"{label} is not a repository path") from error
    raw = _git("show", f"{commit}:{relative.as_posix()}", binary=True)
    assert isinstance(raw, bytes)
    if (
        len(raw) != binding.get("byte_count")
        or hashlib.sha256(raw).hexdigest() != binding.get("file_sha256")
    ):
        raise AlignmentWorkerError(f"{label} differs at frozen commit")


def _read_bound_json(binding: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    try:
        raw = custody._read_absolute_regular_once(binding, label=label)
    except Exception as error:
        raise AlignmentWorkerError(f"could not read bound {label}") from error
    result = strict_json_bytes(raw)
    if type(result) is not dict:
        raise AlignmentWorkerError(f"{label} must be a JSON object")
    return result


def _source_binding(path: str) -> dict[str, Any]:
    return file_binding(REPO_ROOT / path)


def _validate_binding_map(
    observed: Any,
    expected_paths: Mapping[str, str],
    *,
    frozen_commit: str,
    label: str,
) -> dict[str, dict[str, Any]]:
    if type(observed) is not dict or set(observed) != set(expected_paths):
        raise AlignmentWorkerError(f"{label} inventory changed")
    result: dict[str, dict[str, Any]] = {}
    for name, relative in expected_paths.items():
        expected_path = str((REPO_ROOT / relative).resolve(strict=True))
        binding = observed[name]
        if (
            type(binding) is not dict
            or binding.get("path") != expected_path
            or file_binding(Path(expected_path)) != binding
        ):
            raise AlignmentWorkerError(f"{label}.{name} binding changed")
        _require_binding_at_commit(binding, frozen_commit, label=f"{label}.{name}")
        result[name] = dict(binding)
    return result


def _validate_replacement_plan(plan: Any) -> None:
    expected_keys = {
        "schema", "purpose", "development_only",
        "citable_as_original_factual_learnability_claim",
        "authorizes_execution", "route", "direct_predecessor_failure_audit",
        "original_plan_binding", "integrity_replacement", "arms", "objective",
        "action_margin", "head_row_presentations_per_arm_per_training_row",
        "training", "paired_decision", "reuse", "attempt", "caps",
        "science_identity", "forbidden",
    }
    expected_integrity = {
        "version": 1,
        "maximum_integrity_replacements_after_this": 0,
        "sole_functional_change": (
            "row_stable_b32_gradient_enabled_detached_wrong_action_scan"
        ),
        "original_scan": {
            "autograd_enabled": False,
            "batch_rows": 128,
            "flattened_row_action_pairs": True,
        },
        "replacement_scan": {
            "slots": 8,
            "batch_rows": CANDIDATE_SCAN_BATCH_ROWS,
            "row_order": "original_microbatch_order",
            "wrong_id_order_per_row": (
                "ascending_absolute_action_id_excluding_factual"
            ),
            "autograd_enabled": True,
            "energy_detached_before_scan_assignment": True,
            "temporary_graph_discarded_after_each_slot": True,
            "selected_recomputation_batch_rows": MICROBATCH_SIZE,
            "maximum_consistency_error": 1.0e-6,
        },
        "source_probe": {
            "objective_atol": 1.0e-6,
            "gradient_rtol": 1.0e-5,
            "gradient_atol": 1.0e-6,
            "state_hash_exact": True,
            "rng_state_exact": True,
        },
        "scientific_fields_changed": False,
        "tolerance_relaxed": False,
        "failed_attempt_state_reused": False,
    }
    expected_science_identity = {
        "model": True, "predecessor": True, "data": True, "pack": True,
        "initialization": True, "seed": True, "schedule": True,
        "optimizer": True, "updates": True, "coefficients": True,
        "objective": True, "evaluations": True, "metrics": True,
        "thresholds": True, "decision_precedence": True, "caps": True,
        "claim_boundary": True,
    }
    expected_forbidden = [
        "sealed_or_heldout_access", "protected_runtime_access", "rgb_access",
        "alternate_pack_or_checkpoint", "failed_attempt_runtime_state_reuse",
        "architecture_change", "objective_change", "schedule_change",
        "coefficient_search", "tolerance_relaxation", "validation_gradient",
        "automatic_follow_on", "further_integrity_replacement",
    ]
    if (
        type(plan) is not dict
        or set(plan) != expected_keys
        or plan.get("schema") != f"{SCHEMA_PREFIX}_plan_v1"
        or plan.get("purpose")
        != (
            "science_identical_integrity_replacement_of_matched_existing_pool_"
            "global_action_alignment_comparison"
        )
        or plan.get("development_only") is not True
        or plan.get("citable_as_original_factual_learnability_claim") is not False
        or plan.get("authorizes_execution") is not False
        or plan.get("route") != "TEST_GLOBAL_ALIGNMENT_HYPOTHESIS"
        or plan.get("direct_predecessor_failure_audit")
        != EXPECTED_EVIDENCE_BINDINGS["original_successor_failure_audit"]
        or plan.get("original_plan_binding")
        != EXPECTED_EVIDENCE_BINDINGS["original_successor_plan"]
        or plan.get("integrity_replacement") != expected_integrity
        or plan.get("arms")
        != [
            {"name": "baseline", "alignment_coefficient": 0.0},
            {"name": "alignment", "alignment_coefficient": 1.0},
        ]
        or plan.get("objective")
        != (
            "mean(E_factual) + coefficient * mean(relu(0.01 + E_factual - "
            "min_wrong_E))"
        )
        or plan.get("action_margin") != ALIGNMENT_MARGIN
        or plan.get("head_row_presentations_per_arm_per_training_row") != 10
        or plan.get("training")
        != {
            "rows": EXPECTED_TRAIN_ROWS,
            "validation_rows": EXPECTED_VALIDATION_ROWS,
            "updates": TRAINING_UPDATES,
            "batch_size": BATCH_SIZE,
            "microbatch_size": MICROBATCH_SIZE,
            "presentations_per_arm": TRAINING_UPDATES * BATCH_SIZE,
            "seed": 20260731,
            "warmup_updates": 150,
            "schedule_horizon_updates": 3000,
            "observation_updates": list(TAIL_UPDATES),
            "checkpoint_selection": False,
            "early_stopping": False,
        }
        or plan.get("paired_decision")
        != {
            "seed": successor_metrics.PAIRED_BOOTSTRAP_SEED,
            "replicates": 10_000,
            "quantile_indices": [500, 5000, 9499],
            "meaningful_point_threshold": (
                successor_metrics.MEANINGFUL_POINT_THRESHOLD
            ),
            "stall_upper_threshold": successor_metrics.STALL_UPPER_THRESHOLD,
            "absolute_repair_precedence": True,
            "retention_failure_overrides_meaningful": True,
        }
        or plan.get("reuse")
        != {
            "v3_pack_read_only": True,
            "fresh_pack": False,
            "rgb_open_count": 0,
            "data_generation": False,
            "network_access": False,
        }
        or plan.get("attempt")
        != {
            "id": ATTEMPT_ID,
            "maximum_attempts": 1,
            "reservation_consumes_attempt": True,
            "retry": False,
            "resume": False,
            "refill": False,
            "overwrite": False,
            "original_attempt_runtime_reuse": False,
        }
        or plan.get("caps")
        != {
            "maximum_wall_seconds": MAXIMUM_WALL_SECONDS,
            "maximum_gpu_seconds": MAXIMUM_GPU_SECONDS,
            "maximum_training_updates": TRAINING_UPDATES,
        }
        or plan.get("science_identity") != expected_science_identity
        or plan.get("forbidden") != expected_forbidden
    ):
        raise AlignmentWorkerError("bound replacement plan changed")


def load_and_validate_authority(
    authority_path: Path,
    *,
    expected_sha256: str,
    expected_byte_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = Path(authority_path).resolve(strict=True)
    expected_path = AUTHORITY_PATH.resolve(strict=True)
    if selected != expected_path:
        raise AlignmentWorkerError("authority path changed")
    authority_binding = file_binding(selected)
    if authority_binding != {
        "path": str(expected_path),
        "file_sha256": expected_sha256,
        "byte_count": expected_byte_count,
    }:
        raise AlignmentWorkerError("caller-bound authority identity changed")
    authority = _read_bound_json(authority_binding, label="execution authority")
    required_keys = {
        "schema", "status", "authority_granted",
        "citable_as_scientific_evidence", "source_commit", "review_commit",
        "preregistration_binding", "plan_binding", "review_binding", "attempt",
        "input_bindings", "evidence_bindings", "source_bindings", "test_bindings",
        "runtime", "caps", "claim_boundary", "execution", "authorized_command",
        "external_supervisor",
    }
    if set(authority) != required_keys:
        raise AlignmentWorkerError("authority fields changed")
    if (
        authority["schema"] != AUTHORITY_SCHEMA
        or authority["status"] != AUTHORITY_STATUS
        or authority["authority_granted"] is not True
        or authority["citable_as_scientific_evidence"] is not False
        or authority["claim_boundary"] != CLAIM_BOUNDARY
    ):
        raise AlignmentWorkerError("authority grant or claim boundary changed")
    source_commit = _commit(authority["source_commit"], label="source commit")
    review_commit = _commit(authority["review_commit"], label="review commit")
    execution_head = _commit(_git("rev-parse", "HEAD"), label="execution head")
    _require_ancestor(source_commit, review_commit, label="source/review")
    _require_ancestor(review_commit, execution_head, label="review/execution")
    relative_authority = selected.relative_to(REPO_ROOT).as_posix()
    committed_authority = _git("show", f"{execution_head}:{relative_authority}", binary=True)
    assert isinstance(committed_authority, bytes)
    if committed_authority != selected.read_bytes():
        raise AlignmentWorkerError("authority is not the exact execution-HEAD blob")

    expected_documents = {
        "preregistration_binding": PREREGISTRATION_PATH,
        "plan_binding": PLAN_PATH,
        "review_binding": REVIEW_PATH,
    }
    for key, path in expected_documents.items():
        binding = authority[key]
        if type(binding) is not dict or binding.get("path") != str(path.resolve(strict=True)):
            raise AlignmentWorkerError(f"{key} path changed")
        if file_binding(path) != binding:
            raise AlignmentWorkerError(f"{key} bytes changed")
        _require_binding_at_commit(
            binding,
            review_commit if key == "review_binding" else source_commit,
            label=key,
        )
    plan = _read_bound_json(authority["plan_binding"], label="plan")
    _validate_replacement_plan(plan)
    sources = _validate_binding_map(
        authority["source_bindings"], REQUIRED_SOURCE_PATHS,
        frozen_commit=source_commit, label="source bindings"
    )
    tests = _validate_binding_map(
        authority["test_bindings"], REQUIRED_TEST_PATHS,
        frozen_commit=source_commit, label="test bindings"
    )
    review = _read_bound_json(authority["review_binding"], label="independent review")
    if (
        review.get("schema") != REVIEW_SCHEMA
        or review.get("status") != REVIEW_STATUS
        or review.get("authority_granted_by_this_document") is not False
        or review.get("reviewed_source_commit") != source_commit
        or review.get("reviewed_source_bindings") != sources
        or review.get("reviewed_test_bindings") != tests
        or review.get("reviewed_preregistration_binding")
        != authority["preregistration_binding"]
        or review.get("reviewed_plan_binding") != authority["plan_binding"]
        or review.get("route") != "TEST_GLOBAL_ALIGNMENT_HYPOTHESIS"
        or review.get("remaining_findings") != []
        or review.get("authority_granted_by_this_document") is not False
    ):
        raise AlignmentWorkerError("independent source review is not a bound PASS")
    reviewer = review.get("reviewer")
    verification = review.get("verification")
    synthetic_probe = (
        verification.get("exact_rocm_synthetic_probe")
        if type(verification) is dict
        else None
    )
    focused_tests = (
        verification.get("focused_tests")
        if type(verification) is dict
        else None
    )
    expected_synthetic_probe = {
        "passed": True,
        "uses_real_checkpoint_snapshot_pack_index_or_rgb_payload": False,
        "synthetic_predecessor_state_entries": 187,
        "maximum_live_scan_graphs": 1,
        "scan_dispatch": {
            "batch_rows": CANDIDATE_SCAN_BATCH_ROWS,
            "wrong_scan_calls": 8,
            "head_row_presentations": 10,
            "maximum_scan_recompute_error": 0.0,
            "tolerance": 1.0e-6,
            "scan_requires_grad": False,
            "scan_has_grad_fn": False,
            "pre_backward_parameter_grad_count": 0,
            "post_backward_parameter_grad_count": 36,
            "peak_memory_bytes": 1_481_360_896,
        },
        "real_all_parameter_reference": {
            "batch_rows": CANDIDATE_SCAN_BATCH_ROWS,
            "wrong_scan_calls": 8,
            "head_row_presentations": 10,
            "objective_absolute_error": 0.0,
            "objective_atol": 1.0e-6,
            "gradient_parameter_count": 36,
            "gradient_allclose": True,
            "gradient_rtol": 1.0e-5,
            "gradient_atol": 1.0e-6,
            "maximum_gradient_absolute_error": 6.752088665962219e-09,
            "global_gradient_relative_l2_error": 9.253475062389835e-07,
            "minimum_unique_wrong_energy_gap": 7.748603820800781e-06,
            "corrected_state_hash_exact": True,
            "reference_state_hash_exact": True,
            "corrected_cpu_rng_exact": True,
            "corrected_cuda_rng_exact": True,
            "reference_cpu_rng_exact": True,
            "reference_cuda_rng_exact": True,
            "corrected_forward_peak_memory_bytes": 1_013_770_752,
            "reference_forward_peak_memory_bytes": 4_025_879_040,
        },
    }
    if (
        type(reviewer) is not dict
        or type(reviewer.get("identity")) is not str
        or not reviewer["identity"].strip()
        or type(verification) is not dict
        or verification.get("all_focused_tests_passed") is not True
        or type(focused_tests) is not dict
        or focused_tests.get("passed") != 24
        or focused_tests.get("failed") != 0
        or verification.get("normalized_scientific_plan_differences") != []
        or synthetic_probe != expected_synthetic_probe
        or type(review.get("custody")) is not dict
        or review["custody"].get("runtime_payloads_opened") is not False
        or review["custody"].get("sealed_or_heldout_opened") is not False
    ):
        raise AlignmentWorkerError("independent review evidence is incomplete")

    if authority["input_bindings"] != EXPECTED_INPUT_BINDINGS:
        raise AlignmentWorkerError("runtime input bindings changed")
    if authority["evidence_bindings"] != EXPECTED_EVIDENCE_BINDINGS:
        raise AlignmentWorkerError("predecessor evidence bindings changed")
    if ORIGINAL_ATTEMPT_ROOT.is_symlink() or not ORIGINAL_ATTEMPT_ROOT.is_dir():
        raise AlignmentWorkerError("closed original attempt root changed")
    with os.scandir(ORIGINAL_ATTEMPT_ROOT) as entries:
        original_inventory = []
        for entry in entries:
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                raise AlignmentWorkerError("closed original attempt contains a non-file")
            original_inventory.append(entry.name)
    if set(original_inventory) != {
        "reservation.json", "failure.json", "terminal_supervision.json"
    } or len(original_inventory) != 3:
        raise AlignmentWorkerError("closed original attempt inventory changed")
    for name, binding in EXPECTED_EVIDENCE_BINDINGS.items():
        if not _binding_is_exact(Path(binding["path"]), binding):
            raise AlignmentWorkerError(f"pre-reservation evidence changed: {name}")
    if authority["runtime"] != EXPECTED_RUNTIME:
        raise AlignmentWorkerError("runtime contract changed")
    for name, binding in EXPECTED_RUNTIME["bindings"].items():
        if file_binding(Path(binding["path"])) != binding:
            raise AlignmentWorkerError(f"runtime binding changed: {name}")
    if authority["caps"] != {
        "maximum_wall_seconds": MAXIMUM_WALL_SECONDS,
        "maximum_gpu_seconds": MAXIMUM_GPU_SECONDS,
        "maximum_training_updates": TRAINING_UPDATES,
    }:
        raise AlignmentWorkerError("resource caps changed")
    if authority["attempt"] != {
        "id": ATTEMPT_ID,
        "root": str(ATTEMPT_ROOT),
        "maximum_attempts": 1,
        "must_be_absent": True,
        "reservation_consumes_attempt": True,
        "retry": False,
        "resume": False,
        "refill": False,
        "overwrite": False,
    }:
        raise AlignmentWorkerError("attempt contract changed")
    if authority["execution"] != {
        "worker_path": str(WORKER_PATH),
        "checker_path": str(CHECKER_PATH),
        "supervisor_path": str(SUPERVISOR_PATH),
    }:
        raise AlignmentWorkerError("execution paths changed")
    expected_command = [
        EXPECTED_RUNTIME["python_invocation_path"], str(SUPERVISOR_PATH),
        "--authority", str(AUTHORITY_PATH),
        "--expected-authority-sha256", "<CALLER_BOUND_AUTHORITY_SHA256>",
        "--expected-authority-byte-count", "<CALLER_BOUND_AUTHORITY_BYTE_COUNT>",
    ]
    if authority["authorized_command"] != {"argv_template": expected_command}:
        raise AlignmentWorkerError("authorized command changed")
    if authority["external_supervisor"] != {
        "source_binding": sources["external_supervisor"],
        "terminal_reviewer": "Codex primary agent receipt-only terminal review",
    }:
        raise AlignmentWorkerError("external supervisor contract changed")
    return {**authority, "execution_head": execution_head}, authority_binding


def expected_reservation(
    authority: Mapping[str, Any], authority_binding: Mapping[str, Any], *, supervisor_nonce: str
) -> dict[str, Any]:
    worker_command = [
        EXPECTED_RUNTIME["python_invocation_path"], str(WORKER_PATH),
        "--authority", str(AUTHORITY_PATH),
        "--expected-authority-sha256", authority_binding["file_sha256"],
        "--expected-authority-byte-count", str(authority_binding["byte_count"]),
        "--expected-reservation-sha256", "<SUPERVISOR_BOUND_RESERVATION_SHA256>",
        "--expected-reservation-byte-count", "<SUPERVISOR_BOUND_RESERVATION_BYTE_COUNT>",
    ]
    checker_command = [
        EXPECTED_RUNTIME["python_invocation_path"], str(CHECKER_PATH),
        "--manifest", str(ATTEMPT_ROOT / "result.json"),
        "--expected-file-sha256", "<WORKER_RESULT_SHA256>",
        "--expected-byte-count", "<WORKER_RESULT_BYTE_COUNT>",
        "--output", str(ATTEMPT_ROOT / "receipt_check.json"),
    ]
    return {
        "schema": RESERVATION_SCHEMA,
        "status": "RESERVED_ATTEMPT_CONSUMED",
        "attempt_id": ATTEMPT_ID,
        "attempt_root": str(ATTEMPT_ROOT),
        "authority_binding": dict(authority_binding),
        "source_commit": authority["source_commit"],
        "review_commit": authority["review_commit"],
        "execution_head": authority["execution_head"],
        "plan_binding": authority["plan_binding"],
        "review_binding": authority["review_binding"],
        "source_bindings": authority["source_bindings"],
        "test_bindings": authority["test_bindings"],
        "input_bindings": authority["input_bindings"],
        "evidence_bindings": authority["evidence_bindings"],
        "runtime": authority["runtime"],
        "caps": authority["caps"],
        "maximum_attempts": 1,
        "retry": False, "resume": False, "refill": False, "overwrite": False,
        "supervisor_nonce": supervisor_nonce,
        "worker_command_template": worker_command,
        "checker_command_template": checker_command,
        "authorized_device_idle_preflight_passed": True,
    }


def validate_reservation(
    authority: Mapping[str, Any], authority_binding: Mapping[str, Any],
    *, expected_sha256: str, expected_byte_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = file_binding(ATTEMPT_ROOT / "reservation.json")
    if binding != {
        "path": str((ATTEMPT_ROOT / "reservation.json").resolve(strict=True)),
        "file_sha256": expected_sha256,
        "byte_count": expected_byte_count,
    }:
        raise AlignmentWorkerError("reservation binding changed")
    reservation = _read_bound_json(binding, label="reservation")
    nonce = reservation.get("supervisor_nonce")
    if type(nonce) is not str or len(nonce) != 64:
        raise AlignmentWorkerError("reservation nonce is invalid")
    if reservation != expected_reservation(
        authority, authority_binding, supervisor_nonce=nonce
    ):
        raise AlignmentWorkerError("reservation contract changed")
    exact_root_inventory({"reservation.json"})
    return reservation, binding


def validate_exact_child_environment() -> None:
    if dict(os.environ) != EXACT_CHILD_ENVIRONMENT:
        raise AlignmentWorkerError("worker environment differs from exact allowlist")


@dataclass(frozen=True)
class ObjectiveTerms:
    total: torch.Tensor
    factual: torch.Tensor
    hinge: torch.Tensor
    selected_wrong_action_ids: torch.Tensor
    scan_energy: torch.Tensor


def _action_objective_two_pass(
    *,
    arm: base.ArmCore,
    encoded_history: torch.Tensor,
    factual_actions: torch.Tensor,
    target_indices: torch.Tensor,
    target: torch.Tensor,
    coefficient: float,
) -> ObjectiveTerms:
    """Compute the exact minimum-wrong hinge with bounded activation memory.

    Every row is scanned under each of its eight wrong actions using the same
    B32 autograd dispatch as recomputation, but each temporary scan graph is
    detached and discarded immediately. The selected wrong action and factual
    action are then recomputed with a graph. Exact ties select the lowest
    action ID through ``torch.argmin``.
    """

    if coefficient not in (0.0, 1.0):
        raise AlignmentWorkerError("alignment coefficient changed")
    batch = encoded_history.shape[0]
    if (
        batch != CANDIDATE_SCAN_BATCH_ROWS
        or factual_actions.shape != (batch, 3)
        or target_indices.shape != (batch, 64)
        or target.shape != (batch, 64, 192)
    ):
        raise AlignmentWorkerError("objective input shape changed")
    if not torch.is_grad_enabled():
        raise AlignmentWorkerError("wrong-action scan requires autograd dispatch")
    factual_ids = factual_actions[:, 2]
    scan = torch.full(
        (batch, ACTION_COUNT),
        math.inf,
        dtype=torch.float32,
        device=encoded_history.device,
    )
    row_indices = torch.arange(batch, device=encoded_history.device)
    candidate_grid = torch.arange(
        ACTION_COUNT, device=encoded_history.device
    )[None, :].expand(batch, -1)
    wrong_mask = candidate_grid != factual_ids[:, None]
    wrong_ids = candidate_grid[wrong_mask].reshape(batch, ACTION_COUNT - 1)
    if (
        wrong_ids.shape != (batch, ACTION_COUNT - 1)
        or bool((wrong_ids == factual_ids[:, None]).any())
        or not bool((wrong_ids[:, 1:] > wrong_ids[:, :-1]).all())
    ):
        raise AlignmentWorkerError("wrong-action scan row construction changed")
    for slot in range(ACTION_COUNT - 1):
        candidate_ids = wrong_ids[:, slot]
        candidate_actions = factual_actions.clone()
        candidate_actions[:, 2] = candidate_ids
        prediction = base.predict_from_shared_encoding(
            arm,
            encoded_history,
            candidate_actions,
            target_indices,
            candidate_blind=False,
        )
        energy = normalized_half_squared_token_energy_v1(
            prediction.raw, target
        )
        detached_energy = energy.detach()
        if (
            detached_energy.shape != (batch,)
            or detached_energy.requires_grad
            or detached_energy.grad_fn is not None
            or not bool(torch.isfinite(detached_energy).all())
        ):
            raise AlignmentWorkerError("detached scan energy changed")
        scan[row_indices, candidate_ids] = detached_energy
        del prediction, energy, detached_energy, candidate_actions
    if scan.requires_grad or scan.grad_fn is not None:
        raise AlignmentWorkerError("wrong-action scan retained a gradient graph")
    finite_count = torch.isfinite(scan).sum(dim=1)
    if not bool((finite_count == ACTION_COUNT - 1).all()):
        raise AlignmentWorkerError("wrong-action scan did not evaluate exactly eight actions")
    if not bool(torch.isinf(scan.gather(1, factual_ids[:, None])).all()):
        raise AlignmentWorkerError("wrong-action scan included the factual action")
    selected_wrong = torch.argmin(scan, dim=1)
    if bool((selected_wrong == factual_ids).any()):
        raise AlignmentWorkerError("wrong-action selection returned the factual action")

    factual_prediction = base.predict_from_shared_encoding(
        arm,
        encoded_history,
        factual_actions,
        target_indices,
        candidate_blind=False,
    )
    wrong_actions = factual_actions.clone()
    wrong_actions[:, 2] = selected_wrong
    wrong_prediction = base.predict_from_shared_encoding(
        arm,
        encoded_history,
        wrong_actions,
        target_indices,
        candidate_blind=False,
    )
    factual_energy = normalized_half_squared_token_energy_v1(
        factual_prediction.raw, target
    )
    wrong_energy = normalized_half_squared_token_energy_v1(
        wrong_prediction.raw, target
    )
    selected_scan = scan.gather(1, selected_wrong[:, None]).squeeze(1)
    maximum_error = float((wrong_energy.detach() - selected_scan).abs().max())
    if not math.isfinite(maximum_error) or maximum_error > 1.0e-6:
        raise AlignmentWorkerError(
            f"selected-wrong recomputation changed energy by {maximum_error}"
        )
    hinge_rows = torch.relu(ALIGNMENT_MARGIN + factual_energy - wrong_energy)
    factual_loss = factual_energy.mean()
    hinge_loss = hinge_rows.mean()
    # Keep the c=0 autograd route bit-identical to V3's factual-loss baseline;
    # candidate scans and selected-wrong recomputation still execute in both
    # arms, but a mathematically zero hinge is not attached to its graph.
    total = factual_loss if coefficient == 0.0 else factual_loss + hinge_loss
    if not bool(torch.isfinite(total)):
        raise AlignmentWorkerError("alignment objective became nonfinite")
    return ObjectiveTerms(
        total=total,
        factual=factual_loss,
        hinge=hinge_loss,
        selected_wrong_action_ids=selected_wrong.detach(),
        scan_energy=scan.detach(),
    )


def _build_two_arms(
    predecessor_state: Mapping[str, torch.Tensor], *, device: torch.device
) -> tuple[
    torch.nn.Module,
    dict[str, base.ArmCore],
    dict[str, torch.optim.AdamW],
    dict[str, base.ArmPartition],
    dict[str, Any],
]:
    substrate, old_arms, old_optimizers, old_partitions, receipt = (
        base.build_frozen_substrate_and_arms(predecessor_state, device=device)
    )
    arms = {
        "baseline": old_arms.pop("conditioned"),
        "alignment": old_arms.pop("shuffled"),
    }
    optimizers = {
        "baseline": old_optimizers.pop("conditioned"),
        "alignment": old_optimizers.pop("shuffled"),
    }
    partitions = {
        "baseline": old_partitions.pop("conditioned"),
        "alignment": old_partitions.pop("shuffled"),
    }
    old_arms.clear()
    old_optimizers.clear()
    old_partitions.clear()
    torch.cuda.empty_cache()
    hashes = {name: base.module_state_sha256(arm) for name, arm in arms.items()}
    parameter_ids = {
        name: {id(parameter) for parameter in arm.parameters()}
        for name, arm in arms.items()
    }
    if len(set(hashes.values())) != 1 or parameter_ids["baseline"] & parameter_ids["alignment"]:
        raise AlignmentWorkerError("two arms are not identical and disjoint at update zero")
    return substrate, arms, optimizers, partitions, {
        **receipt,
        "two_arm_initial_state_sha256": hashes,
        "two_arm_parameters_disjoint": True,
        "alignment_coefficients": dict(ARM_COEFFICIENTS),
    }


def _train_one_update(
    *,
    update: int,
    batch_rows_cpu: torch.Tensor,
    substrate: torch.nn.Module,
    arms: Mapping[str, base.ArmCore],
    optimizers: Mapping[str, torch.optim.AdamW],
    partitions: Mapping[str, base.ArmPartition],
    train_frames: torch.Tensor,
    train_actions: torch.Tensor,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    if batch_rows_cpu.shape != (BATCH_SIZE,) or batch_rows_cpu.dtype != torch.long:
        raise AlignmentWorkerError("training schedule batch changed")
    fraction = base.learning_rate_fraction(update)
    rates = {
        name: base._set_optimizer_learning_rates(optimizer, fraction=fraction)
        for name, optimizer in optimizers.items()
    }
    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)
    totals = {
        name: {"total": 0.0, "factual": 0.0, "hinge": 0.0}
        for name in ARM_NAMES
    }
    for start in range(0, BATCH_SIZE, MICROBATCH_SIZE):
        micro_cpu = batch_rows_cpu[start : start + MICROBATCH_SIZE]
        micro = micro_cpu.to(device=train_frames.device)
        normalized = scaled.to_float(train_frames[micro])
        target_indices, _ = temporal_metrics.batched_mask_indices(
            "train", micro_cpu.tolist(), device=train_frames.device
        )
        encoded, target = base._encode_context_and_future(
            substrate, normalized, target_indices
        )
        factual_actions = train_actions[micro]
        for arm_name in ARM_NAMES:
            terms = _action_objective_two_pass(
                arm=arms[arm_name],
                encoded_history=encoded,
                factual_actions=factual_actions,
                target_indices=target_indices,
                target=target,
                coefficient=ARM_COEFFICIENTS[arm_name],
            )
            (terms.total * (MICROBATCH_SIZE / BATCH_SIZE)).backward()
            totals[arm_name]["total"] += (
                float(terms.total.detach()) * MICROBATCH_SIZE / BATCH_SIZE
            )
            totals[arm_name]["factual"] += (
                float(terms.factual.detach()) * MICROBATCH_SIZE / BATCH_SIZE
            )
            totals[arm_name]["hinge"] += (
                float(terms.hinge.detach()) * MICROBATCH_SIZE / BATCH_SIZE
            )
    for arm_name in ARM_NAMES:
        partition = partitions[arm_name]
        if not any(parameter.grad is not None for parameter in partition.all):
            raise AlignmentWorkerError(f"{arm_name} produced no gradients")
        norm = torch.nn.utils.clip_grad_norm_(partition.all, base.GRADIENT_CLIP)
        if not bool(torch.isfinite(norm)):
            raise AlignmentWorkerError(f"{arm_name} gradient norm is nonfinite")
        optimizers[arm_name].step()
        if any(not bool(torch.isfinite(parameter).all()) for parameter in partition.all):
            raise AlignmentWorkerError(f"{arm_name} parameters became nonfinite")
        optimizers[arm_name].zero_grad(set_to_none=True)
    return totals, {
        "fraction": fraction,
        "predictor": rates["baseline"]["predictor"],
        "memory": rates["baseline"]["memory"],
    }


@dataclass
class EvaluationVectors:
    factual: dict[str, torch.Tensor]
    candidates: dict[str, torch.Tensor]
    prediction_tokens: dict[str, torch.Tensor]
    target_tokens: torch.Tensor
    persistence: torch.Tensor | None
    wrong_history: dict[str, torch.Tensor]


@torch.no_grad()
def _evaluate_validation(
    *,
    substrate: torch.nn.Module,
    arms: Mapping[str, base.ArmCore],
    frames: torch.Tensor,
    actions: torch.Tensor,
    wrong_history_donors: Sequence[int],
    include_controls: bool,
) -> EvaluationVectors:
    for arm in arms.values():
        arm.eval()
    encoded_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    for start in range(0, EXPECTED_VALIDATION_ROWS, EVALUATION_BATCH_SIZE):
        indices = tuple(range(start, min(start + EVALUATION_BATCH_SIZE, EXPECTED_VALIDATION_ROWS)))
        row_device = torch.tensor(indices, dtype=torch.long, device=frames.device)
        target_indices, _ = temporal_metrics.batched_mask_indices(
            "val", indices, device=frames.device
        )
        encoded, target = base._encode_context_and_future(
            substrate, scaled.to_float(frames[row_device]), target_indices
        )
        encoded_parts.append(encoded)
        target_parts.append(target)
    encoded_cache = torch.cat(encoded_parts)
    target_cache = torch.cat(target_parts)
    factual: dict[str, list[torch.Tensor]] = {name: [] for name in ARM_NAMES}
    candidates: dict[str, list[torch.Tensor]] = {name: [] for name in ARM_NAMES}
    predictions: dict[str, list[torch.Tensor]] = {name: [] for name in ARM_NAMES}
    wrong_history: dict[str, list[torch.Tensor]] = {name: [] for name in ARM_NAMES}
    persistence: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for start in range(0, EXPECTED_VALIDATION_ROWS, EVALUATION_BATCH_SIZE):
        indices = tuple(range(start, min(start + EVALUATION_BATCH_SIZE, EXPECTED_VALIDATION_ROWS)))
        row_device = torch.tensor(indices, dtype=torch.long, device=frames.device)
        target_indices, _ = temporal_metrics.batched_mask_indices(
            "val", indices, device=frames.device
        )
        encoded = encoded_cache[start : start + len(indices)]
        target = target_cache[start : start + len(indices)]
        factual_actions = actions[row_device]
        targets.append(F.normalize(target, p=2.0, dim=-1, eps=1.0e-8).cpu())
        for arm_name in ARM_NAMES:
            prediction = base.predict_from_shared_encoding(
                arms[arm_name], encoded, factual_actions, target_indices,
                candidate_blind=False,
            )
            factual[arm_name].append(base._energy(prediction.raw, target))
            predictions[arm_name].append(prediction.normalized.cpu())
            candidate_parts = []
            for candidate_id in range(ACTION_COUNT):
                intervention = factual_actions.clone()
                intervention[:, 2] = candidate_id
                candidate_prediction = base.predict_from_shared_encoding(
                    arms[arm_name], encoded, intervention, target_indices,
                    candidate_blind=False,
                )
                candidate_parts.append(base._energy(candidate_prediction.raw, target))
            candidates[arm_name].append(torch.stack(candidate_parts, dim=1))
        if include_controls:
            current = _gather_spatial_tokens(encoded[:, 2], target_indices)
            persistence.append(base._energy(current, target))
            donor_ids = torch.tensor(
                [int(wrong_history_donors[index]) for index in indices],
                dtype=torch.long, device=frames.device,
            )
            wrong_encoded = torch.cat((encoded_cache[donor_ids, :2], encoded[:, 2:3]), dim=1)
            wrong_actions = torch.cat((actions[donor_ids, :2], factual_actions[:, 2:3]), dim=1)
            for arm_name in ARM_NAMES:
                wrong_prediction = base.predict_from_shared_encoding(
                    arms[arm_name], wrong_encoded, wrong_actions, target_indices,
                    candidate_blind=False,
                )
                wrong_history[arm_name].append(base._energy(wrong_prediction.raw, target))
    for arm in arms.values():
        arm.train()
    return EvaluationVectors(
        factual={name: torch.cat(factual[name]) for name in ARM_NAMES},
        candidates={name: torch.cat(candidates[name]) for name in ARM_NAMES},
        prediction_tokens={name: torch.cat(predictions[name]) for name in ARM_NAMES},
        target_tokens=torch.cat(targets),
        persistence=torch.cat(persistence) if persistence else None,
        wrong_history={
            name: torch.cat(wrong_history[name])
            for name in ARM_NAMES if wrong_history[name]
        },
    )


@torch.no_grad()
def _evaluate_train_factual(
    *, substrate: torch.nn.Module, arms: Mapping[str, base.ArmCore],
    frames: torch.Tensor, actions: torch.Tensor,
) -> dict[str, torch.Tensor]:
    for arm in arms.values():
        arm.eval()
    values: dict[str, list[torch.Tensor]] = {name: [] for name in ARM_NAMES}
    for start in range(0, EXPECTED_TRAIN_ROWS, EVALUATION_BATCH_SIZE):
        indices = tuple(range(start, min(start + EVALUATION_BATCH_SIZE, EXPECTED_TRAIN_ROWS)))
        row_device = torch.tensor(indices, dtype=torch.long, device=frames.device)
        target_indices, _ = temporal_metrics.batched_mask_indices(
            "train", indices, device=frames.device
        )
        encoded, target = base._encode_context_and_future(
            substrate, scaled.to_float(frames[row_device]), target_indices
        )
        for arm_name in ARM_NAMES:
            prediction = base.predict_from_shared_encoding(
                arms[arm_name], encoded, actions[row_device], target_indices,
                candidate_blind=False,
            )
            values[arm_name].append(base._energy(prediction.raw, target))
    for arm in arms.values():
        arm.train()
    return {name: torch.cat(values[name]) for name in ARM_NAMES}


def _tail_receipt(
    vectors: EvaluationVectors,
    *, rows: Sequence[h6.H6V2Row],
) -> tuple[dict[str, Any], dict[str, float]]:
    actions = [int(row.actions[2]) for row in rows]
    scenes = [row.scene_id for row in rows]
    families = [row.family for row in rows]
    target_rank, _ = scaled.effective_rank(vectors.target_tokens)
    rank_ratios: dict[str, float] = {}
    by_arm: dict[str, Any] = {}
    for arm_name in ARM_NAMES:
        summary = three_arm_metrics.summarize_nine_way_action_identification(
            vectors.candidates[arm_name], actions, scenes, families
        )
        prediction_rank, _ = scaled.effective_rank(vectors.prediction_tokens[arm_name])
        rank_ratio = prediction_rank / target_rank if target_rank > 0.0 else 0.0
        rank_ratios[arm_name] = rank_ratio
        by_arm[arm_name] = {
            "factual_mean_energy": float(vectors.factual[arm_name].mean()),
            "balanced_accuracy": float(summary.scene_family_balanced_accuracy),
            "balanced_accuracy_lower": float(summary.balanced_accuracy_bootstrap_lower_95),
            "hardest_margin": float(summary.hardest_action_margin),
            "hardest_margin_lower": float(summary.hardest_margin_bootstrap_lower_95),
            "prediction_effective_rank": prediction_rank,
            "target_effective_rank": target_rank,
            "rank_ratio": rank_ratio,
        }
    paired = successor_metrics.paired_minimum_action_margin_delta(
        baseline_candidate_energy=vectors.candidates["baseline"],
        treatment_candidate_energy=vectors.candidates["alignment"],
        validation_rows=rows,
    )
    return {"arms": by_arm, "paired_alignment_delta": paired}, rank_ratios


def _absolute_snapshot_binding(binding: Mapping[str, Any], path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve(strict=True)),
        "file_sha256": binding["file_sha256"],
        "byte_count": binding["byte_count"],
    }


def _save_snapshot(
    path: Path, payload: Mapping[str, Any]
) -> dict[str, Any]:
    relative = base.save_immutable_snapshot(path, payload, attempt_root=ATTEMPT_ROOT)
    return _absolute_snapshot_binding(relative, path)


def _validate_reused_pack_binding(role: str, observed: Mapping[str, Any]) -> None:
    prefix = "pack_train" if role == "train" else "pack_validation"
    if observed.get("manifest_sha256") != EXPECTED_INPUT_BINDINGS["pack_manifest"]["file_sha256"]:
        raise AlignmentWorkerError("pack manifest identity changed")
    for field, suffix in (("frames", "frames"), ("actions", "actions"), ("metadata", "metadata")):
        item = observed.get(field)
        expected = EXPECTED_INPUT_BINDINGS[f"{prefix}_{suffix}"]
        if (
            type(item) is not dict
            or item.get("byte_count") != expected["byte_count"]
            or item.get("sha256") != expected["file_sha256"]
        ):
            raise AlignmentWorkerError(f"reused pack {role} {field} changed")


def _baseline_anchor_audit(decision: Mapping[str, Any]) -> dict[str, Any]:
    baseline = decision["localizations"]["baseline"]
    action = baseline["action_identification"]
    controls = baseline["registered_control_reproduction"]
    observed = {
        "balanced_accuracy": action["scene_family_balanced_accuracy"],
        "balanced_accuracy_lower": action["balanced_accuracy_bootstrap_lower_95"],
        "hardest_margin": action["hardest_action_margin"],
        "hardest_margin_lower": action["hardest_margin_bootstrap_lower_95"],
        "persistence_point": controls["persistence"]["macro_log_advantage"],
        "persistence_lower": controls["persistence"]["bootstrap_lower_95"],
        "wrong_history_point": controls["wrong_history"]["macro_log_advantage"],
        "wrong_history_lower": controls["wrong_history"]["bootstrap_lower_95"],
    }
    checks = {
        name: math.isclose(
            float(observed[name]), expected, rel_tol=0.0, abs_tol=1.0e-15
        )
        for name, expected in PUBLIC_V3_BASELINE_ANCHORS.items()
    }
    return {
        "expected": dict(PUBLIC_V3_BASELINE_ANCHORS),
        "observed": observed,
        "checks": checks,
        "exact_within_1e_15": all(checks.values()),
    }


def execute(
    *,
    authority_path: Path,
    expected_authority_sha256: str,
    expected_authority_byte_count: int,
    expected_reservation_sha256: str,
    expected_reservation_byte_count: int,
) -> dict[str, Any]:
    wall_started = time.monotonic()
    validate_exact_child_environment()
    authority, authority_binding = load_and_validate_authority(
        authority_path,
        expected_sha256=expected_authority_sha256,
        expected_byte_count=expected_authority_byte_count,
    )
    reservation, reservation_binding = validate_reservation(
        authority,
        authority_binding,
        expected_sha256=expected_reservation_sha256,
        expected_byte_count=expected_reservation_byte_count,
    )
    try:
        base._validate_runtime_identity({"runtime": authority["runtime"]})
    except Exception as error:
        raise AlignmentWorkerError("exact runtime/device identity changed") from error
    for name, binding in EXPECTED_EVIDENCE_BINDINGS.items():
        if not _binding_is_exact(Path(binding["path"]), binding):
            raise AlignmentWorkerError(f"predecessor evidence changed: {name}")

    train_rows, train_index_audit = h6.load_bound_index(REPO_ROOT, role="train")
    val_rows, val_index_audit = h6.load_bound_index(REPO_ROOT, role="val")
    if len(train_rows) != EXPECTED_TRAIN_ROWS or len(val_rows) != EXPECTED_VALIDATION_ROWS:
        raise AlignmentWorkerError("bound H6 row counts changed")
    if (
        train_index_audit["file_sha256"]
        != EXPECTED_INPUT_BINDINGS["train_index"]["file_sha256"]
        or val_index_audit["file_sha256"]
        != EXPECTED_INPUT_BINDINGS["validation_index"]["file_sha256"]
    ):
        raise AlignmentWorkerError("bound H6 index identity changed")
    schedule, schedule_audit = base.build_bound_training_schedule()
    if (
        tuple(schedule.shape) != (TRAINING_UPDATES, BATCH_SIZE)
        or schedule_audit["presentations"] != TRAINING_UPDATES * BATCH_SIZE
        or schedule_audit["seed"] != base.TRAINING_SEED
    ):
        raise AlignmentWorkerError("bound schedule changed")

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    gpu_started = time.monotonic()

    predecessor_state = base.load_predecessor_state(
        EXPECTED_INPUT_BINDINGS["predecessor_checkpoint"]
    )
    substrate, arms, optimizers, partitions, substrate_receipt = _build_two_arms(
        predecessor_state, device=device
    )
    del predecessor_state
    train_frames, train_actions, train_pack_binding = scaled.load_pack(
        PACK_ROOT, "train", device
    )
    val_frames, val_actions, val_pack_binding = scaled.load_pack(
        PACK_ROOT, "val", device
    )
    _validate_reused_pack_binding("train", train_pack_binding)
    _validate_reused_pack_binding("val", val_pack_binding)
    pack_bindings = {"train": train_pack_binding, "val": val_pack_binding}
    val_donors = temporal_metrics.build_wrong_history_donor_indices(
        base._to_temporal_metrics_rows(val_rows)
    )

    losses: dict[str, dict[str, float]] = {
        name: {"total": math.nan, "factual": math.nan, "hinge": math.nan}
        for name in ARM_NAMES
    }
    learning_rate = {"fraction": 0.0, "predictor": 0.0, "memory": 0.0}
    tail_receipts: dict[int, dict[str, Any]] = {}
    rank_ratio_by_update: dict[int, dict[str, float]] = {}
    final_vectors: EvaluationVectors | None = None

    for update in range(1, TRAINING_UPDATES + 1):
        losses, learning_rate = _train_one_update(
            update=update,
            batch_rows_cpu=schedule[update - 1],
            substrate=substrate,
            arms=arms,
            optimizers=optimizers,
            partitions=partitions,
            train_frames=train_frames,
            train_actions=train_actions,
        )
        base.assert_frozen_substrate_unchanged(
            substrate,
            encoder_sha256=substrate_receipt["encoder_sha256"],
            target_sha256=substrate_receipt["target_sha256"],
        )
        if time.monotonic() - gpu_started > MAXIMUM_GPU_SECONDS:
            raise TimeoutError("authorized GPU cap exceeded")
        if update in TAIL_UPDATES:
            vectors = _evaluate_validation(
                substrate=substrate,
                arms=arms,
                frames=val_frames,
                actions=val_actions,
                wrong_history_donors=val_donors,
                include_controls=update == TRAINING_UPDATES,
            )
            receipt, ranks = _tail_receipt(vectors, rows=val_rows)
            receipt["update"] = update
            receipt["training_loss"] = copy.deepcopy(losses)
            receipt["learning_rate"] = dict(learning_rate)
            tail_receipts[update] = receipt
            rank_ratio_by_update[update] = ranks
            if time.monotonic() - gpu_started > MAXIMUM_GPU_SECONDS:
                raise TimeoutError("authorized GPU cap exceeded after validation panel")
            print(
                json.dumps(
                    {
                        "update": update,
                        "baseline_margin": receipt["arms"]["baseline"]["hardest_margin"],
                        "alignment_margin": receipt["arms"]["alignment"]["hardest_margin"],
                        "paired_delta": receipt["paired_alignment_delta"]["point"],
                        "paired_q05": receipt["paired_alignment_delta"]["one_sided_95_lower_quantile"],
                        "alignment_ba_lower": receipt["arms"]["alignment"]["balanced_accuracy_lower"],
                        "alignment_rank_ratio": ranks["alignment"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if update == TRAINING_UPDATES:
                final_vectors = vectors
            else:
                del vectors
                torch.cuda.empty_cache()

    if final_vectors is None or final_vectors.persistence is None or set(final_vectors.wrong_history) != set(ARM_NAMES):
        raise AlignmentWorkerError("terminal validation controls are absent")
    training_factual = _evaluate_train_factual(
        substrate=substrate,
        arms=arms,
        frames=train_frames,
        actions=train_actions,
    )
    if time.monotonic() - gpu_started > MAXIMUM_GPU_SECONDS:
        raise TimeoutError("authorized GPU cap exceeded after train-fit panel")
    train_means = {
        name: float(training_factual[name].mean()) for name in ARM_NAMES
    }
    train_fit_checks = {
        f"{name}_full_train_factual_energy_finite_positive_below_two": (
            math.isfinite(train_means[name]) and 0.0 < train_means[name] < 2.0
        )
        for name in ARM_NAMES
    }
    train_fit_checks.update(
        {
            f"{name}_terminal_total_loss_finite": math.isfinite(losses[name]["total"])
            for name in ARM_NAMES
        }
    )
    contract_checks = {
        "authority_exact": True,
        "reservation_exact": True,
        "source_and_test_closure_exact": True,
        "runtime_exact": True,
        "input_bindings_exact": True,
        "reused_pack_exact": True,
        "schedule_exact": True,
        "identical_disjoint_initialization": True,
        "shared_candidate_route_exact": True,
        "alignment_coefficients_exact": True,
        "frozen_substrate_exact": True,
        "validation_no_gradient": True,
        "finiteness_exact": True,
        "no_rgb_or_data_generation": True,
        # This optimistic value permits one computation of the full metric
        # object from which the independently checkable anchor audit is
        # derived.  A mismatch is immediately replaced with False and the
        # decision is recomputed before any artifact is published.
        "baseline_v3_reproduction_exact": True,
    }
    decision_arguments = dict(
        baseline_candidate_energy=final_vectors.candidates["baseline"],
        baseline_factual_energy=final_vectors.factual["baseline"],
        baseline_persistence_energy=final_vectors.persistence,
        baseline_wrong_history_energy=final_vectors.wrong_history["baseline"],
        treatment_candidate_energy=final_vectors.candidates["alignment"],
        treatment_factual_energy=final_vectors.factual["alignment"],
        treatment_persistence_energy=final_vectors.persistence,
        treatment_wrong_history_energy=final_vectors.wrong_history["alignment"],
        validation_rows=val_rows,
        treatment_rank_ratio_by_update={
            update: rank_ratio_by_update[update]["alignment"]
            for update in TAIL_UPDATES
        },
        contract_checks=contract_checks,
        train_fit_checks=train_fit_checks,
    )
    decision = successor_metrics.decide_alignment_successor(**decision_arguments)
    baseline_anchor_audit = _baseline_anchor_audit(decision)
    contract_checks["baseline_v3_reproduction_exact"] = bool(
        baseline_anchor_audit["exact_within_1e_15"]
    )
    if not contract_checks["baseline_v3_reproduction_exact"]:
        decision_arguments["contract_checks"] = contract_checks
        decision = successor_metrics.decide_alignment_successor(**decision_arguments)
        repeated_audit = _baseline_anchor_audit(decision)
        if repeated_audit != baseline_anchor_audit:
            raise AlignmentWorkerError("baseline anchor audit changed on recomputation")

    metric_bundle_payload = {
        "schema": METRIC_BUNDLE_SCHEMA,
        "status": "COMPLETE",
        "authority_binding": dict(authority_binding),
        "reservation_binding": dict(reservation_binding),
        "validation_row_indices": torch.arange(EXPECTED_VALIDATION_ROWS, dtype=torch.long),
        "baseline_candidate_energy": final_vectors.candidates["baseline"],
        "baseline_factual_energy": final_vectors.factual["baseline"],
        "baseline_persistence_energy": final_vectors.persistence,
        "baseline_wrong_history_energy": final_vectors.wrong_history["baseline"],
        "alignment_candidate_energy": final_vectors.candidates["alignment"],
        "alignment_factual_energy": final_vectors.factual["alignment"],
        "alignment_persistence_energy": final_vectors.persistence,
        "alignment_wrong_history_energy": final_vectors.wrong_history["alignment"],
        "alignment_rank_ratio_tail": torch.tensor(
            [rank_ratio_by_update[update]["alignment"] for update in TAIL_UPDATES],
            dtype=torch.float64,
        ),
        "training_factual_energy": {
            name: training_factual[name] for name in ARM_NAMES
        },
        "contract_checks": dict(contract_checks),
        "train_fit_checks": dict(train_fit_checks),
    }
    metric_bundle_path = ATTEMPT_ROOT / "metrics.pt"
    metric_bundle_binding = _save_snapshot(metric_bundle_path, metric_bundle_payload)
    snapshot_bindings: dict[str, dict[str, Any]] = {}
    for arm_name in ARM_NAMES:
        snapshot_path = ATTEMPT_ROOT / f"{arm_name}_update_000700.pt"
        snapshot_bindings[arm_name] = _save_snapshot(
            snapshot_path,
            {
                "schema": SNAPSHOT_SCHEMA,
                "status": "COMPLETE",
                "arm": arm_name,
                "alignment_coefficient": ARM_COEFFICIENTS[arm_name],
                "update": TRAINING_UPDATES,
                "authority_binding": dict(authority_binding),
                "reservation_binding": dict(reservation_binding),
                "substrate": substrate_receipt,
                "schedule": schedule_audit,
                "arm_state_dict": base._clone_cpu(arms[arm_name].state_dict()),
                "optimizer_state_dict": base._clone_cpu(optimizers[arm_name].state_dict()),
            },
        )

    scaled._assert_pack_bindings_unchanged(PACK_ROOT, pack_bindings)
    for name, relative in REQUIRED_SOURCE_PATHS.items():
        if file_binding(REPO_ROOT / relative) != authority["source_bindings"][name]:
            raise AlignmentWorkerError(f"source changed during run: {name}")
    for name, relative in REQUIRED_TEST_PATHS.items():
        if file_binding(REPO_ROOT / relative) != authority["test_bindings"][name]:
            raise AlignmentWorkerError(f"test changed during run: {name}")
    base.assert_frozen_substrate_unchanged(
        substrate,
        encoder_sha256=substrate_receipt["encoder_sha256"],
        target_sha256=substrate_receipt["target_sha256"],
    )
    torch.cuda.synchronize(device)
    gpu_elapsed = time.monotonic() - gpu_started
    wall_elapsed = time.monotonic() - wall_started
    if gpu_elapsed > MAXIMUM_GPU_SECONDS or wall_elapsed > MAXIMUM_WALL_SECONDS:
        raise TimeoutError("authorized runtime cap exceeded at terminal sync")
    result = {
        "schema": RESULT_SCHEMA,
        "status": "COMPLETE_PENDING_TERMINAL_REVIEW",
        "development_evidence_complete": True,
        "citable_as_original_factual_learnability_claim": False,
        "authority_binding": dict(authority_binding),
        "reservation_binding": dict(reservation_binding),
        "source_commit": authority["source_commit"],
        "review_commit": authority["review_commit"],
        "execution_head": authority["execution_head"],
        "plan_binding": authority["plan_binding"],
        "review_binding": authority["review_binding"],
        "attempt": {
            **authority["attempt"],
            "consumed": True,
            "retry_authorized": False,
            "resume_authorized": False,
        },
        "input_bindings": authority["input_bindings"],
        "evidence_bindings": authority["evidence_bindings"],
        "metric_bundle_binding": metric_bundle_binding,
        "snapshot_bindings": snapshot_bindings,
        "schedule": schedule_audit,
        "substrate": substrate_receipt,
        "tail_measurements": [tail_receipts[update] for update in TAIL_UPDATES],
        "train_fit": {
            "full_train_factual_mean_energy": train_means,
            "terminal_training_loss": losses,
            "checks": train_fit_checks,
        },
        "baseline_v3_reproduction": baseline_anchor_audit,
        "decision": decision,
        "runtime": {
            "authorized": authority["runtime"],
            "observed": {
                "python_version": sys.version.split()[0],
                "torch_version": torch.__version__,
                "torch_hip": torch.version.hip,
                "numpy_version": np.__version__,
                "device_name": torch.cuda.get_device_name(device),
                "device_arch": str(
                    getattr(torch.cuda.get_device_properties(device), "gcnArchName", "")
                ),
                "gpu_elapsed_seconds": gpu_elapsed,
                "wall_elapsed_seconds": wall_elapsed,
                "maximum_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
            },
        },
        "accounting": {
            "training_updates": TRAINING_UPDATES,
            "optimizer_steps_per_arm": TRAINING_UPDATES,
            "total_optimizer_steps": TRAINING_UPDATES * len(ARM_NAMES),
            "schedule_presentations_per_arm": TRAINING_UPDATES * BATCH_SIZE,
            "training_head_row_presentations_per_arm": TRAINING_UPDATES * BATCH_SIZE * 10,
            "training_head_row_presentations_total": TRAINING_UPDATES * BATCH_SIZE * 10 * len(ARM_NAMES),
            "training_shared_frame_encodings": TRAINING_UPDATES * BATCH_SIZE * 4,
            "validation_updates": list(TAIL_UPDATES),
            "full_train_fit_rows_per_arm": EXPECTED_TRAIN_ROWS,
            "pack_reused_read_only": True,
            "rgb_open_count": 0,
            "data_generation_count": 0,
            "network_access_count": 0,
            "sealed_open_count": 0,
            "heldout_open_count": 0,
        },
        "forbidden_access": {
            "sealed_material_opened": False,
            "heldout_material_opened": False,
            "protected_runtime_material_opened": False,
            "rgb_opened": False,
            "network_access_used": False,
            "validation_used_for_gradient_updates": False,
            "existing_pack_modified": False,
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    write_immutable_json(ATTEMPT_ROOT / "result.json", result)
    exact_root_inventory(EXPECTED_SUCCESS_FILES_BEFORE_CHECKER)
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": decision["status"],
                "paired_delta": decision["paired_alignment_delta"]["point"],
                "paired_q05": decision["paired_alignment_delta"]["one_sided_95_lower_quantile"],
                "paired_q95": decision["paired_alignment_delta"]["one_sided_95_upper_quantile"],
                "gpu_minutes": gpu_elapsed / 60.0,
                "wall_minutes": wall_elapsed / 60.0,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--expected-authority-sha256", required=True)
    parser.add_argument("--expected-authority-byte-count", type=int, required=True)
    parser.add_argument("--expected-reservation-sha256", required=True)
    parser.add_argument("--expected-reservation-byte-count", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    try:
        execute(
            authority_path=arguments.authority,
            expected_authority_sha256=arguments.expected_authority_sha256,
            expected_authority_byte_count=arguments.expected_authority_byte_count,
            expected_reservation_sha256=arguments.expected_reservation_sha256,
            expected_reservation_byte_count=arguments.expected_reservation_byte_count,
        )
        return 0
    except BaseException as error:
        failure_path = ATTEMPT_ROOT / "failure.json"
        if ATTEMPT_ROOT.is_dir() and not failure_path.exists():
            try:
                write_immutable_json(
                    failure_path,
                    {
                        "schema": FAILURE_SCHEMA,
                        "status": "ATTEMPT_CONSUMED_WORKER_FAILURE",
                        "attempt_id": ATTEMPT_ID,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                        "traceback": traceback.format_exc(),
                        "retry": False,
                        "resume": False,
                    },
                )
            except BaseException:
                pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
