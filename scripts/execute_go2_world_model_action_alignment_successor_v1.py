#!/usr/bin/env python3
"""Execute one bounded fixed same-mechanism action-alignment continuation.

The worker consumes the immutable V3 frame pack, one frozen spatial
predecessor, and both exact completed u700 arm/AdamW snapshots. It never opens
RGB, sealed, held-out, or protected material and never creates data. Both arms
continue through absolute updates 701--900 on the unchanged row-stable
action-candidate route. Further training is possible only after a pace-level
absolute gain and a new, separately reviewed preregistration.
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
    go2_world_model_action_alignment_continuation_v1 as continuation_metrics,
)
from lewm.benchmarks import (  # noqa: E402
    go2_world_model_v3_action_localization_v1 as localization_metrics,
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
    "fixed_same_mechanism_continuation_v1"
)
AUTHORITY_SCHEMA = f"{SCHEMA_PREFIX}_execution_authority_v1"
AUTHORITY_STATUS = (
    "AUTHORIZED_ONE_EXACT_FIXED_SAME_MECHANISM_CONTINUATION_V1_ATTEMPT"
)
RESERVATION_SCHEMA = f"{SCHEMA_PREFIX}_reservation_v1"
RESULT_SCHEMA = f"{SCHEMA_PREFIX}_result_v1"
SNAPSHOT_SCHEMA = f"{SCHEMA_PREFIX}_snapshot_v1"
METRIC_BUNDLE_SCHEMA = f"{SCHEMA_PREFIX}_metric_bundle_v1"
FAILURE_SCHEMA = f"{SCHEMA_PREFIX}_failure_v1"
REVIEW_SCHEMA = f"{SCHEMA_PREFIX}_independent_source_review_v1"
REVIEW_STATUS = "PASS_SOURCE_ONLY_NOT_AUTHORITY"

ATTEMPT_ID = (
    "world_model_action_alignment_successor_v1_"
    "fixed_same_mechanism_continuation_v1/attempt_v1"
)
ATTEMPT_ROOT = (
    REPO_ROOT
    / ".generated/dev/world_model_action_alignment_successor_v1_"
    "fixed_same_mechanism_continuation_v1/attempt_v1"
)
PREDECESSOR_ATTEMPT_ROOT = (
    REPO_ROOT
    / ".generated/dev/world_model_action_alignment_successor_v1_"
    "integrity_replacement_v1/attempt_v1"
)
AUTHORITY_PATH = (
    REPO_ROOT
    / "docs/lewm_go2_world_model_action_alignment_successor_v1_"
    "fixed_same_mechanism_continuation_v1_execution_authority_2026-08-01.json"
)
PREREGISTRATION_PATH = (
    REPO_ROOT
    / "docs/lewm_go2_world_model_action_alignment_successor_v1_"
    "fixed_same_mechanism_continuation_v1_preregistration_2026-08-01.md"
)
PLAN_PATH = (
    REPO_ROOT
    / "docs/lewm_go2_world_model_action_alignment_successor_v1_"
    "fixed_same_mechanism_continuation_v1_plan_2026-08-01.json"
)
REVIEW_PATH = (
    REPO_ROOT
    / "docs/lewm_go2_world_model_action_alignment_successor_v1_"
    "fixed_same_mechanism_continuation_v1_independent_source_review_2026-08-01.json"
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
START_UPDATE = 700
TRAINING_UPDATES = 900
ADDITIONAL_TRAINING_UPDATES = TRAINING_UPDATES - START_UPDATE
BATCH_SIZE = 256
MICROBATCH_SIZE = 32
OBSERVATION_UPDATES = (700, 800, 900)
EVALUATION_BATCH_SIZE = 64
ALIGNMENT_MARGIN = 0.01
CANDIDATE_SCAN_BATCH_ROWS = 32
MAXIMUM_WALL_SECONDS = 9_000
MAXIMUM_GPU_SECONDS = 7_200
EXPECTED_TRAIN_ROWS = 16_000
EXPECTED_VALIDATION_ROWS = 2_048
ACTION_COUNT = 9
RANK_TOKEN_COUNT = 64
RANK_FEATURE_DIMENSION = 192
ABSOLUTE_PROGRESS_BOOTSTRAP_SEED = continuation_metrics.PAIRED_BOOTSTRAP_SEED
ABSOLUTE_PROGRESS_THRESHOLD = continuation_metrics.ABSOLUTE_GAIN_THRESHOLD
RECOVERY_DIAGNOSTIC_THRESHOLD = (
    continuation_metrics.RECOVERY_GAIN_THRESHOLD_DIAGNOSTIC_ONLY
)
U700_GUARDS = {
    "balanced_accuracy_lower": 0.34701964075333114,
    "rank_ratio": 0.47287848726118314,
    "persistence_lower": -0.22601831547011703,
    "wrong_history_lower": 0.1406183675693852,
    "hardest_margin": -0.00660276124845185,
    "hardest_margin_lower": -0.0078111838906331724,
}

V3_ATTEMPT_ROOT = (
    REPO_ROOT
    / ".generated/dev/world_model_existing_pool_three_arm_v1_integrity_replacement_v3/attempt_v1"
)
PACK_ROOT = V3_ATTEMPT_ROOT / "pack"
PREDECESSOR_PATH = base.PREDECESSOR

PREDECESSOR_SNAPSHOT_SCHEMA = (
    "lewm_go2_world_model_action_alignment_successor_v1_"
    "integrity_replacement_v1_snapshot_v1"
)
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
    "baseline_u700_snapshot": _absolute_binding(
        PREDECESSOR_ATTEMPT_ROOT / "baseline_update_000700.pt",
        "613693d06309f90b87a7ac3e836d6817eed8c1e473ed0063006eb88960bce770",
        10_909_343,
    ),
    "alignment_u700_snapshot": _absolute_binding(
        PREDECESSOR_ATTEMPT_ROOT / "alignment_update_000700.pt",
        "41435888521041aaa262db9a26eaa656d33a339998372ffd0b068d7c75679731",
        10_909_343,
    ),
}

EXPECTED_EVIDENCE_BINDINGS = {
    "completed_successor_authority": _absolute_binding(
        REPO_ROOT
        / "docs/lewm_go2_world_model_action_alignment_successor_v1_"
        "integrity_replacement_v1_execution_authority_2026-08-01.json",
        "88b2c43264a9ee0fb46cf032d4323281d57f84148ff47a1dce98cde403b55cac",
        22_455,
    ),
    "completed_successor_reservation": _absolute_binding(
        PREDECESSOR_ATTEMPT_ROOT / "reservation.json",
        "34d54c60ef4916eb942574de90f620a0bf15be337497e29fe73277c5abf26787",
        22_188,
    ),
    "completed_successor_result": _absolute_binding(
        PREDECESSOR_ATTEMPT_ROOT / "result.json",
        "57d9cfc2bcfa946805255bfdd1144faaf40290f7a921f6d201e770e114d7dd9b",
        164_670,
    ),
    "completed_successor_receipt_check": _absolute_binding(
        PREDECESSOR_ATTEMPT_ROOT / "receipt_check.json",
        "46b157fb685b41ce05a347f34d4e1a67ad38485350fdfe69412b21fbe4048ec4",
        1_849,
    ),
    "completed_successor_terminal": _absolute_binding(
        PREDECESSOR_ATTEMPT_ROOT / "terminal_supervision.json",
        "098d42503da1255ddc7b5a0c49cbbb746e6c41e8aca64627cc9d25a3cc0824b7",
        6_764,
    ),
    "completed_successor_terminal_review": _absolute_binding(
        REPO_ROOT
        / "docs/lewm_go2_world_model_action_alignment_successor_v1_"
        "integrity_replacement_v1_terminal_review_2026-08-01.json",
        "51e760bc868f4cc2307dcc98c2778f97502e7643c866067eb0d47f8b35de3f45",
        13_967,
    ),
    "preauthority_identity_read_disclosure": _absolute_binding(
        REPO_ROOT
        / "docs/lewm_go2_world_model_action_alignment_successor_v1_"
        "fixed_same_mechanism_continuation_v1_preauthority_identity_read_"
        "disclosure_2026-08-01.json",
        "a602a28b0cf4d9af34318dad98507a911e576cfdbd41e655c3e851bf1dbccc7c",
        5_303,
    ),
    "continuation_governance_correction": _absolute_binding(
        REPO_ROOT
        / "docs/lewm_go2_world_model_action_alignment_successor_v1_"
        "fixed_same_mechanism_continuation_v1_governance_correction_"
        "2026-08-01.json",
        "1a13d90b567c0e25bb459848bdc5b818568226160f5b1b80467d9df5e15cb341",
        3_431,
    ),
}

# Public JSON-only witnesses from the independently checked completed u700
# result. They are deliberately scalar/vector anchors. The predecessor metric
# bundle is explicitly excluded from inputs/evidence and is never opened by
# the continuation worker or the independent continuation source reviewer.
PUBLIC_U700_REPLAY_ANCHORS = {
    "baseline": {
        "factual_mean_energy": 0.12548826619149622,
        "balanced_accuracy": 0.2469343816883539,
        "balanced_accuracy_lower": 0.23014452836846072,
        "hardest_margin": -0.009453551490358742,
        "hardest_margin_lower": -0.01138311990101325,
        "rank_ratio": 0.46826675978556964,
        "per_action_points": [
            -0.006170692834744324,
            -0.005279141936414382,
            -0.005601519050314481,
            -0.00799588780500926,
            -0.007148324499695852,
            -0.007509375523243631,
            -0.009453551490358742,
            -0.003459775001156837,
            -0.007492055966883283,
        ],
        "per_action_q05": [
            -0.007607474729023559,
            -0.007305245818655399,
            -0.008961245139171573,
            -0.010201082615119542,
            -0.008537367256800316,
            -0.009700872771402906,
            -0.011105889591432121,
            -0.005668186473008009,
            -0.009848080127799131,
        ],
        "persistence_lower": -0.1829122861354923,
        "wrong_history_lower": 0.11766703087321294,
    },
    "alignment": {
        "factual_mean_energy": 0.13033405333044357,
        "balanced_accuracy": 0.362969689539191,
        "balanced_accuracy_lower": 0.34701964075333114,
        "hardest_margin": -0.00660276124845185,
        "hardest_margin_lower": -0.0078111838906331724,
        "rank_ratio": 0.47287848726118314,
        "per_action_points": [
            0.0012027149883560013,
            -0.0035527010051377506,
            -0.0015738389923428305,
            -0.005862178591625633,
            0.0011691098445800988,
            -0.00660276124845185,
            -0.004179590968378184,
            0.0014289161780315412,
            -0.0007765471389516971,
        ],
        "per_action_q05": [
            0.00027890959805907734,
            -0.004443377358722731,
            -0.0026513533635927015,
            -0.00682322353805956,
            0.00027973971696981705,
            -0.007808230967274414,
            -0.004987031431587844,
            0.00013842324334251095,
            -0.001866197634443366,
        ],
        "persistence_lower": -0.22601831547011703,
        "wrong_history_lower": 0.1406183675693852,
    },
    "concurrent_delta": {
        "point": 0.0028507902419068927,
        "lower": 0.0013970169908673067,
        "median": 0.003062303631657816,
        "upper": 0.004941227499907773,
    },
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
    "continuation_metrics": "lewm/benchmarks/go2_world_model_action_alignment_continuation_v1.py",
    "worker": "scripts/execute_go2_world_model_action_alignment_successor_v1.py",
    "checker": "scripts/check_go2_world_model_action_alignment_successor_v1.py",
    "external_supervisor": "scripts/run_go2_world_model_action_alignment_successor_authorized_v1.py",
}
REQUIRED_TEST_PATHS = {
    "alignment_metric_tests": "lewm/tests/test_go2_world_model_action_alignment_successor_v1.py",
    "continuation_metric_tests": "lewm/tests/test_go2_world_model_action_alignment_continuation_v1.py",
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
    "baseline_update_000900.pt",
    "alignment_update_000900.pt",
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


def _validate_continuation_plan(plan: Any) -> None:
    """Validate the machine plan's scientifically operative fields.

    The exact field inventory and every operative restoration, decision,
    finality, and custody value are frozen.
    """

    required_keys = {
        "schema", "purpose", "route", "development_only",
        "authorizes_execution", "citable_as_original_factual_learnability_claim",
        "citable_as_planning_usefulness_evidence", "predecessor_evidence",
        "arms", "objective", "action_margin",
        "head_row_presentations_per_arm_per_training_row", "continuation",
        "training", "absolute_progress_decision", "u700_descriptive_anchors",
        "continuation_retention", "terminal_precedence",
        "action_alignment_repair", "post_alignment_persistence_routing",
        "meaningful_progress_incomplete", "reuse", "attempt", "caps",
        "forbidden", "finality", "bootstrap_claim_boundary",
    }
    expected_arms = [
        {
            "name": name,
            "alignment_coefficient": ARM_COEFFICIENTS[name],
            "u700_snapshot": EXPECTED_INPUT_BINDINGS[f"{name}_u700_snapshot"],
        }
        for name in ARM_NAMES
    ]
    if (
        type(plan) is not dict
        or set(plan) != required_keys
        or plan.get("schema") != f"{SCHEMA_PREFIX}_plan_v1"
        or plan.get("purpose") != "bounded_u700_to_u900_absolute_treatment_progress_gate"
        or plan.get("route") != "FIXED_SAME_MECHANISM_CONTINUATION_V1"
        or plan.get("development_only") is not True
        or plan.get("authorizes_execution") is not False
        or plan.get("citable_as_original_factual_learnability_claim") is not False
        or plan.get("citable_as_planning_usefulness_evidence") is not False
        or plan.get("predecessor_evidence") != {
            "execution_authority": EXPECTED_EVIDENCE_BINDINGS["completed_successor_authority"],
            "reservation": EXPECTED_EVIDENCE_BINDINGS["completed_successor_reservation"],
            "result": EXPECTED_EVIDENCE_BINDINGS["completed_successor_result"],
            "receipt_check": EXPECTED_EVIDENCE_BINDINGS["completed_successor_receipt_check"],
            "terminal_supervision": EXPECTED_EVIDENCE_BINDINGS["completed_successor_terminal"],
            "terminal_review": EXPECTED_EVIDENCE_BINDINGS["completed_successor_terminal_review"],
            "preauthority_identity_read_disclosure": EXPECTED_EVIDENCE_BINDINGS["preauthority_identity_read_disclosure"],
            "continuation_governance_correction": EXPECTED_EVIDENCE_BINDINGS["continuation_governance_correction"],
        }
        or plan.get("arms") != expected_arms
        or plan.get("objective") != (
            "mean(E_factual) + coefficient * mean(relu(0.01 + E_factual - "
            "min_wrong_E))"
        )
        or plan.get("action_margin") != ALIGNMENT_MARGIN
        or plan.get("head_row_presentations_per_arm_per_training_row") != 10
        or plan.get("continuation") != {
            "source_global_update": START_UPDATE,
            "terminal_global_update": TRAINING_UPDATES,
            "additional_updates": ADDITIONAL_TRAINING_UPDATES,
            "load_both_arm_state_dicts": True,
            "load_each_arms_own_adamw_state": True,
            "optimizer_step_at_source": START_UPDATE,
            "optimizer_step_at_terminal": TRAINING_UPDATES,
            "warmup_reset": False,
            "optimizer_reset": False,
            "schedule_replay": False,
            "schedule_prefix_exact": True,
            "pretraining_u700_validation_replay": True,
            "u700_public_anchor_absolute_tolerance": 1.0e-12,
            "u700_public_anchor_relative_tolerance": 0.0,
        }
        or plan.get("training") != {
            "rows": EXPECTED_TRAIN_ROWS,
            "validation_rows": EXPECTED_VALIDATION_ROWS,
            "global_schedule_updates": TRAINING_UPDATES,
            "trained_global_updates_inclusive": [START_UPDATE + 1, TRAINING_UPDATES],
            "additional_updates": ADDITIONAL_TRAINING_UPDATES,
            "batch_size": BATCH_SIZE,
            "microbatch_size": MICROBATCH_SIZE,
            "additional_presentations_per_arm": ADDITIONAL_TRAINING_UPDATES * BATCH_SIZE,
            "seed": 20260731,
            "warmup_updates": 150,
            "schedule_horizon_updates": 3000,
            "observation_updates": list(OBSERVATION_UPDATES),
            "checkpoint_selection": False,
            "early_stopping": False,
            "validation_gradient": False,
        }
        or plan.get("absolute_progress_decision") != {
            "definition": "hardest_action_margin_u900_minus_u700_within_alignment_arm",
            "bootstrap_algorithm": continuation_metrics.BOOTSTRAP_ALGORITHM,
            "seed": ABSOLUTE_PROGRESS_BOOTSTRAP_SEED,
            "replicates": 10_000,
            "quantile_indices": [500, 5000, 9499],
            "u700_residual": 0.00660276124845185,
            "u701_u900_learning_rate_fraction_sum": 175.22190359794223,
            "u701_u3000_remaining_learning_rate_fraction_sum": 891.0844401632202,
            "u701_u900_remaining_learning_rate_mass_share": 0.19663894430234558,
            "on_trajectory_gain_threshold": ABSOLUTE_PROGRESS_THRESHOLD,
            "u500_u700_loss_recovery_diagnostic": RECOVERY_DIAGNOSTIC_THRESHOLD,
            "requires_q05_positive": True,
            "concurrent_baseline_relative_delta_is_diagnostic_only": True,
        }
        or plan.get("u700_descriptive_anchors") != {
            "balanced_accuracy_q05": U700_GUARDS["balanced_accuracy_lower"],
            "rank_ratio": U700_GUARDS["rank_ratio"],
            "persistence_q05": U700_GUARDS["persistence_lower"],
            "wrong_history_q05": U700_GUARDS["wrong_history_lower"],
            "worst_action_margin_point": U700_GUARDS["hardest_margin"],
            "shared_minimum_margin_q05": U700_GUARDS["hardest_margin_lower"],
        }
        or plan.get("continuation_retention") != {
            "balanced_accuracy_q05_strictly_above": 1.0 / ACTION_COUNT,
            "wrong_history_q05_strictly_above": 0.0,
            "rank_ratio_at_least": continuation_metrics.RANK_RATIO_RETENTION_MINIMUM,
            "rank_minimum_passing_observation_count": continuation_metrics.RANK_RETENTION_PASS_COUNT,
            "rank_observation_updates": list(OBSERVATION_UPDATES),
            "preserve_positive_action_margin_point_ids": list(continuation_metrics.PRESERVED_POSITIVE_ACTION_IDS),
            "preserve_positive_action_margin_q05_ids": list(continuation_metrics.PRESERVED_POSITIVE_ACTION_IDS),
            "all_contract_checks": True,
            "all_train_fit_checks": True,
            "persistence_is_post_alignment_routing_not_retention": True,
        }
        or plan.get("terminal_precedence") != [
            "FAIL_CONTRACT_CLOSE_ALIGNMENT_BRANCH",
            "FAIL_RETENTION_CLOSE_ALIGNMENT_BRANCH",
            "PASS_ACTION_ALIGNMENT_PROXY_REPAIR_PERSISTENCE_SYSTEMIC",
            "PASS_ACTION_ALIGNMENT_PROXY_REPAIR_PLANNING_WITH_PROXY_CAVEAT",
            "PASS_EXPLORATORY_ACTION_ALIGNMENT_AND_PREDICTOR_USEFULNESS_PROXY",
            "MEANINGFUL_ABSOLUTE_PROGRESS_INCOMPLETE_CONTINUE_SAME_MECHANISM",
            "POSITIVE_BUT_INSUFFICIENT_RATE_CLOSE_ALIGNMENT_BRANCH",
            "INCONCLUSIVE_ABSOLUTE_CHANGE_CLOSE_ALIGNMENT_BRANCH",
            "STALLED_OR_HARMFUL_CLOSE_ALIGNMENT_BRANCH",
        ]
        or plan.get("action_alignment_repair") != {
            "all_nine_action_margin_points_positive": True,
            "all_nine_action_margin_q05_positive": True,
            "shared_minimum_margin_q05_positive": True,
            "balanced_accuracy_q05_strictly_above_chance": True,
            "wrong_history_q05_positive": True,
            "rank_gate_passed": True,
            "all_retention_contract_and_train_fit_checks_passed": True,
        }
        or plan.get("post_alignment_persistence_routing") != {
            "systemic_failure_minimum_nonpositive_per_action_q05_count": 5,
            "systemic_next_step": "PERSISTENCE_RESIDUAL_VS_MATCHED_BASELINE",
            "localized_or_aggregate_unrepaired_next_step": "PLANNING_USEFULNESS_GATE_WITH_PROXY_CAVEAT",
            "passed_requires_zero_nonpositive_per_action_q05_and_positive_aggregate_q05": True,
            "passed_next_step": "PROCEED_TO_PLANNING_USEFULNESS_GATE",
            "automatic_execution_authority": False,
        }
        or plan.get("meaningful_progress_incomplete") != {
            "absolute_gain_at_least": ABSOLUTE_PROGRESS_THRESHOLD,
            "absolute_gain_q05_positive": True,
            "continuation_retention_passed": True,
            "permits_separate_same_mechanism_preregistration": True,
            "automatic_execution_authority": False,
            "selected_next_step": "PREREGISTER_NEXT_FIXED_SAME_MECHANISM_BLOCK",
        }
        or plan.get("reuse") != {
            "v3_pack_read_only": True,
            "fresh_pack": False,
            "rgb_open_count": 0,
            "data_generation": False,
            "network_access": False,
            "predecessor_metric_bundle_input": False,
            "prior_attempt_root_write_count": 0,
        }
        or plan.get("attempt") != {
            "id": ATTEMPT_ID,
            "maximum_attempts": 1,
            "reservation_consumes_attempt": True,
            "retry": False,
            "resume": False,
            "refill": False,
            "overwrite": False,
            "recovery": False,
            "integrity_replacement": False,
            "further_continuation": False,
            "identical_replication": False,
        }
        or plan.get("caps") != {
            "maximum_wall_seconds": MAXIMUM_WALL_SECONDS,
            "maximum_gpu_seconds": MAXIMUM_GPU_SECONDS,
            "maximum_additional_training_updates": ADDITIONAL_TRAINING_UPDATES,
            "maximum_global_training_update": TRAINING_UPDATES,
        }
        or plan.get("forbidden") != [
            "sealed_or_heldout_access", "protected_runtime_access", "rgb_access",
            "new_data_generation", "alternate_pack_or_checkpoint",
            "cross_arm_snapshot_load", "optimizer_reset", "warmup_reset",
            "schedule_replay_or_change", "architecture_or_objective_change",
            "coefficient_or_threshold_search", "validation_gradient",
            "automatic_follow_on", "retry_resume_refill_overwrite_or_recovery",
            "integrity_replacement",
            "unregistered_further_alignment_continuation_or_replication",
        ]
        or plan.get("finality") != {
            "meaningful_progress_requires_separate_preregistration": True,
            "all_other_alignment_unrepaired_outcomes_close_training": True,
            "proxy_pass_does_not_authorize_planning_execution": True,
            "automatic_follow_on": False,
        }
        or plan.get("bootstrap_claim_boundary") != {
            "conditional_on_adaptively_selected_continuation": True,
            "validation_scene_reweighting_only": True,
            "training_seed_uncertainty_measured": False,
            "fresh_scene_generalization_measured": False,
        }
    ):
        raise AlignmentWorkerError("bound continuation plan changed")


INDEPENDENT_SOURCE_REVIEWER_IDENTITY = "/root/continuation_code_audit"
PREAUTHORITY_REVIEW_EXCLUDED_IDENTITIES = {
    "/root",
    "/root/continuation_runtime",
    "/root/localization_result_audit",
}


def _validate_independent_source_reviewer_evidence(review: Mapping[str, Any]) -> None:
    reviewer = review.get("reviewer")
    verification = review.get("verification")
    focused_tests = (
        verification.get("focused_tests")
        if type(verification) is dict
        else None
    )
    if (
        type(reviewer) is not dict
        or reviewer.get("identity") != INDEPENDENT_SOURCE_REVIEWER_IDENTITY
        or reviewer.get("identity") in PREAUTHORITY_REVIEW_EXCLUDED_IDENTITIES
        or type(verification) is not dict
        or verification.get("all_focused_tests_passed") is not True
        or type(focused_tests) is not dict
        or type(focused_tests.get("passed")) is not int
        or focused_tests.get("passed") < 24
        or focused_tests.get("failed") != 0
        or verification.get("restoration_contract_reviewed") is not True
        or verification.get("absolute_progress_decision_reviewed") is not True
        or verification.get("schedule_prefix_and_absolute_update_reviewed") is not True
        or verification.get(
            "preauthority_identity_read_disclosure_and_exclusions_reviewed"
        ) is not True
        or verification.get("governance_correction_reviewed") is not True
        or verification.get("no_real_runtime_payload_opened") is not True
        or type(review.get("custody")) is not dict
        or review["custody"].get("runtime_payloads_opened") is not False
        or review["custody"].get("sealed_or_heldout_opened") is not False
    ):
        raise AlignmentWorkerError("independent review evidence is incomplete")


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
    _validate_continuation_plan(plan)
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
        or review.get("route") != "FIXED_SAME_MECHANISM_CONTINUATION_V1"
        or review.get("remaining_findings") != []
        or review.get("authority_granted_by_this_document") is not False
    ):
        raise AlignmentWorkerError("independent source review is not a bound PASS")
    _validate_independent_source_reviewer_evidence(review)

    if authority["input_bindings"] != EXPECTED_INPUT_BINDINGS:
        raise AlignmentWorkerError("runtime input bindings changed")
    if authority["evidence_bindings"] != EXPECTED_EVIDENCE_BINDINGS:
        raise AlignmentWorkerError("predecessor evidence bindings changed")
    if PREDECESSOR_ATTEMPT_ROOT.is_symlink() or not PREDECESSOR_ATTEMPT_ROOT.is_dir():
        raise AlignmentWorkerError("completed predecessor attempt root changed")
    with os.scandir(PREDECESSOR_ATTEMPT_ROOT) as entries:
        predecessor_inventory = []
        for entry in entries:
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                raise AlignmentWorkerError("completed predecessor contains a non-file")
            predecessor_inventory.append(entry.name)
    if set(predecessor_inventory) != {
        "reservation.json", "baseline_update_000700.pt",
        "alignment_update_000700.pt", "metrics.pt", "result.json",
        "receipt_check.json", "terminal_supervision.json",
    } or len(predecessor_inventory) != 7:
        raise AlignmentWorkerError("completed predecessor attempt inventory changed")
    for name, binding in EXPECTED_INPUT_BINDINGS.items():
        path = Path(binding["path"])
        if name in {"baseline_u700_snapshot", "alignment_u700_snapshot"}:
            try:
                metadata = path.lstat()
            except OSError as error:
                raise AlignmentWorkerError(
                    f"pre-reservation snapshot input changed: {name}"
                ) from error
            if (
                path.is_symlink()
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_size != binding["byte_count"]
            ):
                raise AlignmentWorkerError(
                    f"pre-reservation snapshot input changed: {name}"
                )
            # The exact digest is checked during the one and only bound read
            # immediately before weights-only deserialization by the worker.
        elif not _binding_is_exact(path, binding):
            raise AlignmentWorkerError(f"pre-reservation input changed: {name}")
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
        "maximum_additional_training_updates": ADDITIONAL_TRAINING_UPDATES,
        "maximum_global_training_update": TRAINING_UPDATES,
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
        "recovery": False,
        "integrity_replacement": False,
        "further_continuation": False,
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


def _load_and_restore_u700_snapshot(
    *,
    arm_name: str,
    arm: base.ArmCore,
    optimizer: torch.optim.AdamW,
    substrate_receipt: Mapping[str, Any],
    schedule_u700_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Load one exact bound u700 arm/AdamW snapshot and validate it strictly."""

    if arm_name not in ARM_NAMES:
        raise AlignmentWorkerError("snapshot selected an unknown arm")
    binding = EXPECTED_INPUT_BINDINGS[f"{arm_name}_u700_snapshot"]
    try:
        raw = custody._read_absolute_regular_once(
            binding, label=f"{arm_name} u700 continuation snapshot"
        )
        snapshot = torch.load(
            io.BytesIO(raw), map_location="cpu", weights_only=True
        )
    except Exception as error:
        raise AlignmentWorkerError(
            f"could not load bound {arm_name} u700 snapshot"
        ) from error
    finally:
        if "raw" in locals():
            del raw
    required = {
        "schema", "status", "arm", "alignment_coefficient", "update",
        "authority_binding", "reservation_binding", "substrate", "schedule",
        "arm_state_dict", "optimizer_state_dict",
    }
    if (
        type(snapshot) is not dict
        or set(snapshot) != required
        or snapshot.get("schema") != PREDECESSOR_SNAPSHOT_SCHEMA
        or snapshot.get("status") != "COMPLETE"
        or snapshot.get("arm") != arm_name
        or snapshot.get("alignment_coefficient") != ARM_COEFFICIENTS[arm_name]
        or snapshot.get("update") != START_UPDATE
        or snapshot.get("authority_binding")
        != EXPECTED_EVIDENCE_BINDINGS["completed_successor_authority"]
        or snapshot.get("reservation_binding")
        != EXPECTED_EVIDENCE_BINDINGS["completed_successor_reservation"]
        or snapshot.get("substrate") != dict(substrate_receipt)
        or snapshot.get("schedule") != dict(schedule_u700_audit)
    ):
        raise AlignmentWorkerError(f"{arm_name} u700 snapshot envelope changed")

    observed_model = snapshot["arm_state_dict"]
    expected_model = arm.state_dict()
    if type(observed_model) is not dict or set(observed_model) != set(expected_model):
        raise AlignmentWorkerError(f"{arm_name} model-state inventory changed")
    for name, expected in expected_model.items():
        value = observed_model[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.layout != torch.strided
            or value.dtype != expected.dtype
            or tuple(value.shape) != tuple(expected.shape)
            or not bool(torch.isfinite(value).all())
        ):
            raise AlignmentWorkerError(
                f"{arm_name} model-state tensor changed: {name}"
            )
    model_sha256 = base.tensor_inventory_sha256(observed_model)

    observed_optimizer = snapshot["optimizer_state_dict"]
    fresh_optimizer = optimizer.state_dict()
    if (
        type(observed_optimizer) is not dict
        or set(observed_optimizer) != {"state", "param_groups"}
        or type(observed_optimizer["state"]) is not dict
        or type(observed_optimizer["param_groups"]) is not list
        or len(observed_optimizer["param_groups"])
        != len(fresh_optimizer["param_groups"])
    ):
        raise AlignmentWorkerError(f"{arm_name} optimizer envelope changed")
    expected_lr = {
        "predictor": base.PREDICTOR_BASE_LR
        * base.LR_SCALE
        * base.learning_rate_fraction(START_UPDATE),
        "memory": base.MEMORY_BASE_LR
        * base.LR_SCALE
        * base.learning_rate_fraction(START_UPDATE),
    }
    flat_parameters: list[torch.nn.Parameter] = []
    expected_parameter_ids: list[int] = []
    for group_index, (observed_group, fresh_group) in enumerate(
        zip(
            observed_optimizer["param_groups"],
            fresh_optimizer["param_groups"],
            strict=True,
        )
    ):
        if (
            type(observed_group) is not dict
            or set(observed_group) != set(fresh_group)
            or observed_group.get("group_name") not in expected_lr
            or observed_group.get("params") != fresh_group.get("params")
        ):
            raise AlignmentWorkerError(
                f"{arm_name} optimizer group {group_index} identity changed"
            )
        for key, value in fresh_group.items():
            if key == "lr":
                if observed_group[key] != expected_lr[observed_group["group_name"]]:
                    raise AlignmentWorkerError(
                        f"{arm_name} optimizer group learning rate changed"
                    )
            elif key != "params" and observed_group[key] != value:
                raise AlignmentWorkerError(
                    f"{arm_name} optimizer group hyperparameter changed: {key}"
                )
        expected_parameter_ids.extend(fresh_group["params"])
        flat_parameters.extend(optimizer.param_groups[group_index]["params"])
    if (
        len(expected_parameter_ids) != len(set(expected_parameter_ids))
        or set(observed_optimizer["state"]) != set(expected_parameter_ids)
        or len(flat_parameters) != len(expected_parameter_ids)
    ):
        raise AlignmentWorkerError(f"{arm_name} optimizer state coverage changed")
    parameter_by_id = dict(zip(expected_parameter_ids, flat_parameters, strict=True))
    moment_inventory: dict[str, torch.Tensor] = {}
    for parameter_id in expected_parameter_ids:
        state = observed_optimizer["state"][parameter_id]
        parameter = parameter_by_id[parameter_id]
        if type(state) is not dict or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise AlignmentWorkerError(
                f"{arm_name} optimizer state fields changed for {parameter_id}"
            )
        step = state["step"]
        if (
            not isinstance(step, torch.Tensor)
            or step.device.type != "cpu"
            or step.dtype != torch.float32
            or tuple(step.shape) != ()
            or not bool(torch.isfinite(step))
            or float(step) != float(START_UPDATE)
        ):
            raise AlignmentWorkerError(
                f"{arm_name} optimizer step is not exactly {START_UPDATE}"
            )
        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = state[moment_name]
            if (
                not isinstance(moment, torch.Tensor)
                or moment.device.type != "cpu"
                or moment.layout != torch.strided
                or moment.dtype != parameter.dtype
                or tuple(moment.shape) != tuple(parameter.shape)
                or not bool(torch.isfinite(moment).all())
            ):
                raise AlignmentWorkerError(
                    f"{arm_name} optimizer moment changed: {parameter_id}.{moment_name}"
                )
            moment_inventory[f"{parameter_id}.{moment_name}"] = moment
    optimizer_moment_sha256 = base.tensor_inventory_sha256(moment_inventory)

    try:
        arm.load_state_dict(observed_model, strict=True)
        optimizer.load_state_dict(observed_optimizer)
    except Exception as error:
        raise AlignmentWorkerError(
            f"{arm_name} u700 state restoration failed"
        ) from error
    if base.module_state_sha256(arm) != model_sha256:
        raise AlignmentWorkerError(f"{arm_name} restored model state changed")
    post_load_moments: dict[str, torch.Tensor] = {}
    parameter_offset = 0
    for group_index, (loaded_group, snapshot_group) in enumerate(
        zip(optimizer.param_groups, observed_optimizer["param_groups"], strict=True)
    ):
        if set(loaded_group) != set(snapshot_group):
            raise AlignmentWorkerError(
                f"{arm_name} loaded optimizer group fields changed"
            )
        for key, value in snapshot_group.items():
            if key != "params" and loaded_group[key] != value:
                raise AlignmentWorkerError(
                    f"{arm_name} loaded optimizer hyperparameter changed: {key}"
                )
        group_count = len(snapshot_group["params"])
        expected_loaded_parameters = flat_parameters[
            parameter_offset : parameter_offset + group_count
        ]
        parameter_offset += group_count
        if [id(value) for value in loaded_group["params"]] != [
            id(value) for value in expected_loaded_parameters
        ]:
            raise AlignmentWorkerError(
                f"{arm_name} loaded optimizer parameter order changed"
            )
    if parameter_offset != len(flat_parameters):
        raise AlignmentWorkerError(f"{arm_name} loaded optimizer coverage changed")
    for parameter_id, parameter in parameter_by_id.items():
        state = optimizer.state.get(parameter)
        if (
            type(state) is not dict
            or set(state) != {"step", "exp_avg", "exp_avg_sq"}
            or float(state["step"].detach().cpu()) != float(START_UPDATE)
        ):
            raise AlignmentWorkerError(f"{arm_name} restored optimizer step changed")
        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = state[moment_name]
            if (
                moment.dtype != parameter.dtype
                or tuple(moment.shape) != tuple(parameter.shape)
                or not bool(torch.isfinite(moment).all())
            ):
                raise AlignmentWorkerError(
                    f"{arm_name} loaded optimizer moment changed"
                )
            post_load_moments[f"{parameter_id}.{moment_name}"] = moment
    if base.tensor_inventory_sha256(post_load_moments) != optimizer_moment_sha256:
        raise AlignmentWorkerError(
            f"{arm_name} loaded optimizer moment hash changed"
        )
    del snapshot, observed_model, observed_optimizer
    return {
        "input_binding": dict(binding),
        "schema": PREDECESSOR_SNAPSHOT_SCHEMA,
        "arm": arm_name,
        "update": START_UPDATE,
        "model_state_sha256": model_sha256,
        "optimizer_moment_sha256": optimizer_moment_sha256,
        "optimizer_parameter_count": len(expected_parameter_ids),
        "optimizer_step": START_UPDATE,
        "loaded_once": True,
        "model_and_own_adamw_restored": True,
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


def _rank_covariance(tokens: torch.Tensor) -> torch.Tensor:
    """Return the exact float64 covariance sufficient statistic for rank."""

    if (
        not isinstance(tokens, torch.Tensor)
        or tuple(tokens.shape)
        != (EXPECTED_VALIDATION_ROWS, RANK_TOKEN_COUNT, RANK_FEATURE_DIMENSION)
        or not bool(torch.isfinite(tokens).all())
    ):
        raise AlignmentWorkerError("rank token tensor changed")
    values = tokens.detach().to("cpu", torch.float64)
    centered = values - values.mean(dim=0, keepdim=True)
    flat = centered.reshape(-1, RANK_FEATURE_DIMENSION)
    covariance = flat.T.mm(flat) / (flat.shape[0] - 1)
    covariance = 0.5 * (covariance + covariance.T)
    if (
        covariance.dtype != torch.float64
        or tuple(covariance.shape)
        != (RANK_FEATURE_DIMENSION, RANK_FEATURE_DIMENSION)
        or not bool(torch.isfinite(covariance).all())
    ):
        raise AlignmentWorkerError("rank covariance changed")
    return covariance


def _effective_rank_from_covariance(covariance: torch.Tensor) -> float:
    """Compute entropy effective rank from a validated covariance matrix."""

    if (
        not isinstance(covariance, torch.Tensor)
        or covariance.dtype != torch.float64
        or tuple(covariance.shape)
        != (RANK_FEATURE_DIMENSION, RANK_FEATURE_DIMENSION)
        or not bool(torch.isfinite(covariance).all())
    ):
        raise AlignmentWorkerError("effective-rank covariance changed")
    eigenvalues = torch.linalg.eigvalsh(
        0.5 * (covariance + covariance.T)
    ).clamp_min(0.0)
    total = float(eigenvalues.sum())
    if total <= 0.0:
        return 0.0
    probabilities = eigenvalues / eigenvalues.sum()
    return float(
        (-(probabilities * probabilities.clamp_min(1.0e-12).log()).sum()).exp()
    )


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
) -> tuple[dict[str, Any], dict[str, float], dict[str, torch.Tensor]]:
    actions = [int(row.actions[2]) for row in rows]
    scenes = [row.scene_id for row in rows]
    families = [row.family for row in rows]
    rank_covariances = {
        "target": _rank_covariance(vectors.target_tokens),
        **{
            arm_name: _rank_covariance(vectors.prediction_tokens[arm_name])
            for arm_name in ARM_NAMES
        },
    }
    target_rank = _effective_rank_from_covariance(rank_covariances["target"])
    rank_ratios: dict[str, float] = {}
    by_arm: dict[str, Any] = {}
    for arm_name in ARM_NAMES:
        summary = three_arm_metrics.summarize_nine_way_action_identification(
            vectors.candidates[arm_name], actions, scenes, families
        )
        prediction_rank = _effective_rank_from_covariance(
            rank_covariances[arm_name]
        )
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
    return (
        {"arms": by_arm, "paired_alignment_delta": paired},
        rank_ratios,
        rank_covariances,
    )


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


def _panel_localizations(
    vectors: EvaluationVectors, *, rows: Sequence[h6.H6V2Row]
) -> dict[str, Any]:
    if vectors.persistence is None or set(vectors.wrong_history) != set(ARM_NAMES):
        raise AlignmentWorkerError("full continuation control panel is absent")
    return {
        name: localization_metrics.localize_action_and_controls(
            candidate_energies=vectors.candidates[name],
            factual_energy=vectors.factual[name],
            persistence_energy=vectors.persistence,
            wrong_history_energy=vectors.wrong_history[name],
            validation_rows=rows,
        )
        for name in ARM_NAMES
    }


def _u700_replay_anchor_audit(
    *,
    vectors: EvaluationVectors,
    rows: Sequence[h6.H6V2Row],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    localizations = _panel_localizations(vectors, rows=rows)
    relative = successor_metrics.paired_minimum_action_margin_delta(
        baseline_candidate_energy=vectors.candidates["baseline"],
        treatment_candidate_energy=vectors.candidates["alignment"],
        validation_rows=rows,
    )
    observed: dict[str, Any] = {}
    for name in ARM_NAMES:
        action = localizations[name]["action_identification"]
        margin = localizations[name]["action_margin_localization"]
        controls = localizations[name]["registered_control_reproduction"]
        observed[name] = {
            "factual_mean_energy": receipt["arms"][name]["factual_mean_energy"],
            "balanced_accuracy": action["scene_family_balanced_accuracy"],
            "balanced_accuracy_lower": action["balanced_accuracy_bootstrap_lower_95"],
            "hardest_margin": action["hardest_action_margin"],
            "hardest_margin_lower": action["hardest_margin_bootstrap_lower_95"],
            "rank_ratio": receipt["arms"][name]["rank_ratio"],
            "per_action_points": [
                row["family_equal_scene_macro_point"] for row in margin["per_action"]
            ],
            "per_action_q05": [
                row["one_sided_95_lower_quantile"] for row in margin["per_action"]
            ],
            "persistence_lower": controls["persistence"]["bootstrap_lower_95"],
            "wrong_history_lower": controls["wrong_history"]["bootstrap_lower_95"],
        }
    observed["concurrent_delta"] = {
        "point": relative["point"],
        "lower": relative["one_sided_95_lower_quantile"],
        "median": relative["median_quantile"],
        "upper": relative["one_sided_95_upper_quantile"],
    }
    checks: dict[str, bool] = {}
    for section, expected_section in PUBLIC_U700_REPLAY_ANCHORS.items():
        for name, expected in expected_section.items():
            value = observed[section][name]
            if isinstance(expected, list):
                checks[f"{section}.{name}"] = (
                    len(value) == len(expected)
                    and all(
                        math.isclose(
                            float(item), float(anchor), rel_tol=0.0, abs_tol=1.0e-12
                        )
                        for item, anchor in zip(value, expected, strict=True)
                    )
                )
            else:
                checks[f"{section}.{name}"] = math.isclose(
                    float(value), float(expected), rel_tol=0.0, abs_tol=1.0e-12
                )
    return {
        "expected": copy.deepcopy(PUBLIC_U700_REPLAY_ANCHORS),
        "observed": observed,
        "checks": checks,
        "absolute_tolerance": 1.0e-12,
        "relative_tolerance": 0.0,
        "passed": all(checks.values()),
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
    schedule, schedule_audit = base.build_bound_training_schedule(
        updates=TRAINING_UPDATES
    )
    schedule_u700, schedule_u700_audit = base.build_bound_training_schedule(
        updates=START_UPDATE
    )
    if (
        tuple(schedule.shape) != (TRAINING_UPDATES, BATCH_SIZE)
        or schedule_audit["presentations"] != TRAINING_UPDATES * BATCH_SIZE
        or schedule_audit["seed"] != base.TRAINING_SEED
        or tuple(schedule_u700.shape) != (START_UPDATE, BATCH_SIZE)
        or schedule_u700_audit["presentations"] != START_UPDATE * BATCH_SIZE
        or not torch.equal(schedule[:START_UPDATE], schedule_u700)
    ):
        raise AlignmentWorkerError("bound schedule changed")
    schedule_prefix_receipt = {
        "source_updates": START_UPDATE,
        "terminal_updates": TRAINING_UPDATES,
        "source_schedule": schedule_u700_audit,
        "terminal_schedule": schedule_audit,
        "prefix_tensor_exact": True,
        "trained_slice_start_zero_based": START_UPDATE,
        "trained_slice_stop_exclusive": TRAINING_UPDATES,
    }

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
    restoration_receipts = {
        name: _load_and_restore_u700_snapshot(
            arm_name=name,
            arm=arms[name],
            optimizer=optimizers[name],
            substrate_receipt=substrate_receipt,
            schedule_u700_audit=schedule_u700_audit,
        )
        for name in ARM_NAMES
    }
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
    observation_receipts: dict[int, dict[str, Any]] = {}
    rank_ratio_by_update: dict[int, dict[str, float]] = {}
    rank_covariance_by_update: dict[int, dict[str, torch.Tensor]] = {}
    replay_vectors: EvaluationVectors | None = None
    final_vectors: EvaluationVectors | None = None

    restored_panel = _evaluate_validation(
        substrate=substrate,
        arms=arms,
        frames=val_frames,
        actions=val_actions,
        wrong_history_donors=val_donors,
        include_controls=True,
    )
    restored_receipt, restored_ranks, restored_rank_covariances = _tail_receipt(
        restored_panel, rows=val_rows
    )
    restored_receipt["update"] = START_UPDATE
    restored_receipt["restored_pretraining_replay"] = True
    replay_anchor_audit = _u700_replay_anchor_audit(
        vectors=restored_panel, rows=val_rows, receipt=restored_receipt
    )
    if not replay_anchor_audit["passed"]:
        raise AlignmentWorkerError("restored u700 public anchors did not reproduce")
    observation_receipts[START_UPDATE] = restored_receipt
    rank_ratio_by_update[START_UPDATE] = restored_ranks
    rank_covariance_by_update[START_UPDATE] = restored_rank_covariances
    replay_vectors = EvaluationVectors(
        factual=restored_panel.factual,
        candidates=restored_panel.candidates,
        prediction_tokens={},
        target_tokens=torch.empty(0),
        persistence=restored_panel.persistence,
        wrong_history=restored_panel.wrong_history,
    )
    del restored_panel
    torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "update": START_UPDATE,
                "restored_replay": True,
                "anchor_audit": "PASS",
                "baseline_margin": restored_receipt["arms"]["baseline"]["hardest_margin"],
                "alignment_margin": restored_receipt["arms"]["alignment"]["hardest_margin"],
                "alignment_rank_ratio": restored_ranks["alignment"],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for update in range(START_UPDATE + 1, TRAINING_UPDATES + 1):
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
        if update in OBSERVATION_UPDATES[1:]:
            vectors = _evaluate_validation(
                substrate=substrate,
                arms=arms,
                frames=val_frames,
                actions=val_actions,
                wrong_history_donors=val_donors,
                include_controls=update == TRAINING_UPDATES,
            )
            receipt, ranks, rank_covariances = _tail_receipt(
                vectors, rows=val_rows
            )
            receipt["update"] = update
            receipt["training_loss"] = copy.deepcopy(losses)
            receipt["learning_rate"] = dict(learning_rate)
            observation_receipts[update] = receipt
            rank_ratio_by_update[update] = ranks
            rank_covariance_by_update[update] = rank_covariances
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

    if (
        replay_vectors is None
        or final_vectors is None
        or final_vectors.persistence is None
        or set(final_vectors.wrong_history) != set(ARM_NAMES)
    ):
        raise AlignmentWorkerError("terminal validation controls are absent")
    for arm_name, optimizer in optimizers.items():
        if (
            len(optimizer.state) != restoration_receipts[arm_name]["optimizer_parameter_count"]
            or any(
                float(state["step"].detach().cpu()) != float(TRAINING_UPDATES)
                for state in optimizer.state.values()
            )
        ):
            raise AlignmentWorkerError(
                f"{arm_name} optimizer did not reach exact global step {TRAINING_UPDATES}"
            )
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
        "schedule_900_and_u700_prefix_exact": True,
        "both_exact_u700_snapshots_loaded_once": all(
            receipt["loaded_once"] for receipt in restoration_receipts.values()
        ),
        "both_arm_and_own_adamw_states_restored": all(
            receipt["model_and_own_adamw_restored"]
            for receipt in restoration_receipts.values()
        ),
        "u700_public_anchor_replay_exact_within_1e_12": replay_anchor_audit["passed"],
        "absolute_updates_701_through_900_only": True,
        "both_optimizers_reached_exact_step_900": True,
        "shared_candidate_route_exact": True,
        "alignment_coefficients_exact": True,
        "frozen_substrate_exact": True,
        "validation_no_gradient": True,
        "finiteness_exact": True,
        "no_rgb_or_data_generation": True,
    }
    decision = continuation_metrics.decide_alignment_continuation(
        baseline_candidate_energy_u700=replay_vectors.candidates["baseline"],
        baseline_candidate_energy_u900=final_vectors.candidates["baseline"],
        treatment_candidate_energy_u700=replay_vectors.candidates["alignment"],
        treatment_factual_energy_u700=replay_vectors.factual["alignment"],
        treatment_persistence_energy_u700=replay_vectors.persistence,
        treatment_wrong_history_energy_u700=replay_vectors.wrong_history["alignment"],
        treatment_candidate_energy_u900=final_vectors.candidates["alignment"],
        treatment_factual_energy_u900=final_vectors.factual["alignment"],
        treatment_persistence_energy_u900=final_vectors.persistence,
        treatment_wrong_history_energy_u900=final_vectors.wrong_history["alignment"],
        validation_rows=val_rows,
        treatment_rank_ratio_by_update={
            update: rank_ratio_by_update[update]["alignment"]
            for update in OBSERVATION_UPDATES
        },
        contract_checks=contract_checks,
        train_fit_checks=train_fit_checks,
    )

    metric_bundle_payload = {
        "schema": METRIC_BUNDLE_SCHEMA,
        "status": "COMPLETE",
        "authority_binding": dict(authority_binding),
        "reservation_binding": dict(reservation_binding),
        "validation_row_indices": torch.arange(EXPECTED_VALIDATION_ROWS, dtype=torch.long),
        "u700_baseline_candidate_energy": replay_vectors.candidates["baseline"],
        "u700_baseline_factual_energy": replay_vectors.factual["baseline"],
        "u700_baseline_persistence_energy": replay_vectors.persistence,
        "u700_baseline_wrong_history_energy": replay_vectors.wrong_history["baseline"],
        "u700_alignment_candidate_energy": replay_vectors.candidates["alignment"],
        "u700_alignment_factual_energy": replay_vectors.factual["alignment"],
        "u700_alignment_persistence_energy": replay_vectors.persistence,
        "u700_alignment_wrong_history_energy": replay_vectors.wrong_history["alignment"],
        "u900_baseline_candidate_energy": final_vectors.candidates["baseline"],
        "u900_baseline_factual_energy": final_vectors.factual["baseline"],
        "u900_baseline_persistence_energy": final_vectors.persistence,
        "u900_baseline_wrong_history_energy": final_vectors.wrong_history["baseline"],
        "u900_alignment_candidate_energy": final_vectors.candidates["alignment"],
        "u900_alignment_factual_energy": final_vectors.factual["alignment"],
        "u900_alignment_persistence_energy": final_vectors.persistence,
        "u900_alignment_wrong_history_energy": final_vectors.wrong_history["alignment"],
        "alignment_rank_ratio_observations": torch.tensor(
            [
                rank_ratio_by_update[update]["alignment"]
                for update in OBSERVATION_UPDATES
            ],
            dtype=torch.float64,
        ),
        "baseline_rank_ratio_observations": torch.tensor(
            [
                rank_ratio_by_update[update]["baseline"]
                for update in OBSERVATION_UPDATES
            ],
            dtype=torch.float64,
        ),
        "rank_covariance_by_update": {
            update: {
                name: rank_covariance_by_update[update][name]
                for name in ("target", *ARM_NAMES)
            }
            for update in OBSERVATION_UPDATES
        },
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
        snapshot_path = ATTEMPT_ROOT / f"{arm_name}_update_000900.pt"
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
        "citable_as_planning_usefulness_evidence": False,
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
        "schedule": schedule_prefix_receipt,
        "substrate": substrate_receipt,
        "restoration": restoration_receipts,
        "u700_replay_anchor_audit": replay_anchor_audit,
        "observation_measurements": [
            observation_receipts[update] for update in OBSERVATION_UPDATES
        ],
        "train_fit": {
            "full_train_factual_mean_energy": train_means,
            "terminal_training_loss": losses,
            "checks": train_fit_checks,
        },
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
            "source_global_update": START_UPDATE,
            "terminal_global_update": TRAINING_UPDATES,
            "additional_training_updates": ADDITIONAL_TRAINING_UPDATES,
            "additional_optimizer_steps_per_arm": ADDITIONAL_TRAINING_UPDATES,
            "additional_total_optimizer_steps": ADDITIONAL_TRAINING_UPDATES * len(ARM_NAMES),
            "additional_schedule_presentations_per_arm": ADDITIONAL_TRAINING_UPDATES * BATCH_SIZE,
            "additional_training_head_row_presentations_per_arm": ADDITIONAL_TRAINING_UPDATES * BATCH_SIZE * 10,
            "additional_training_head_row_presentations_total": ADDITIONAL_TRAINING_UPDATES * BATCH_SIZE * 10 * len(ARM_NAMES),
            "additional_training_shared_frame_encodings": ADDITIONAL_TRAINING_UPDATES * BATCH_SIZE * 4,
            "validation_updates": list(OBSERVATION_UPDATES),
            "full_train_fit_rows_per_arm": EXPECTED_TRAIN_ROWS,
            "u700_snapshot_byte_read_count": 2,
            "u700_snapshot_deserialization_count": 2,
            "predecessor_metric_bundle_byte_read_count": 0,
            "predecessor_metric_bundle_deserialization_count": 0,
            "bound_non_snapshot_input_identity_hash_reads_performed": True,
            "pack_payloads_opened_for_training_and_evaluation": True,
            "prior_attempt_write_count": 0,
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
                "absolute_progress": decision["absolute_treatment_hardest_margin_gain"]["point"],
                "absolute_q05": decision["absolute_treatment_hardest_margin_gain"]["one_sided_95_lower_quantile"],
                "absolute_q95": decision["absolute_treatment_hardest_margin_gain"]["one_sided_95_upper_quantile"],
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
                        "refill": False,
                        "overwrite": False,
                        "recovery": False,
                        "integrity_replacement": False,
                        "further_continuation": False,
                    },
                )
            except BaseException:
                pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
