#!/usr/bin/env python3
"""Versioned scorer-fit oracle-v1.3 data workflow.

This successor is intentionally narrower than the historical V2 pipeline.  It
does four data operations and nothing else:

* adopt the 1,422 already-valid V2 labels under the v1.3 equivalence clause;
* replay exactly the diagnosed eighteen graph-boundary branches into a new
  per-tick label overlay (the V2 rows and RGB files remain immutable);
* freeze, then execute, 24 fresh scene-disjoint calibration identities; and
* compose the one 1,440-row scorer training view.

No encoder, scorer trainer, predictor, final-evaluation selector, or final
200-state benchmark is imported or exposed by this module.  Runtime stages are
CPU-only and require an explicit, validated preregistration authority.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import stat
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for _extra in (ROOT, ROOT / "lewm_genesis", ROOT / "lewm_worlds", ROOT / "scripts"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import run_go2_oracle_branch_pilot_v1 as V1
import run_go2_oracle_branch_pilot_v1_2 as V12
from lewm.oracle.go2_textured_v03_renderer import (
    BasePose,
    TexturedV03Renderer,
    capture_base_pose,
    validate_camera_pack,
)
from lewm.oracle import go2_scorer_fit_oracle_v1_3_contract as V13_CONTRACT
from scripts import build_go2_branch_corpus_v1_2 as B
from scripts import diagnose_go2_scorer_fit_v2_graph_label_failures as DIAG


STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
OUT_ROOT = ROOT / ".generated/go2_scorer_fit_oracle_v1_3"
V2_ROOT = B.OUT_ROOT / "scorer_fit"

AUTHORITY_SCHEMA = "go2_scorer_fit_oracle_v1_3_preregistration_authority_v1"
AUTHORITY_SELF_KEY = "authority_digest"
AUTHORITY_PATH = OUT_ROOT / "authority.json"
SUPERSEDED_TRANSITION_SCHEMA = (
    "go2_scorer_fit_oracle_v1_3_superseded_preattempt_transition_v1"
)
SUPERSEDED_TRANSITION_SELF_KEY = "superseded_preattempt_transition_digest"
SUPERSEDED_TRANSITION_PATH = OUT_ROOT / "superseded_preattempt_transition.json"
REPLAY_PLAN_SCHEMA = "go2_scorer_fit_oracle_v1_3_exact_replay_plan_v1"
REPLAY_PLAN_SELF_KEY = "replay_plan_digest"
REPLAY_PLAN_PATH = OUT_ROOT / "replay_plan.json"
EQUIVALENCE_SCHEMA = "go2_scorer_fit_oracle_v1_3_equivalence_receipt_v1"
EQUIVALENCE_SELF_KEY = "equivalence_receipt_digest"
EQUIVALENCE_PATH = OUT_ROOT / "equivalence_receipt.json"
REPLAY_OVERLAY_SCHEMA = "go2_scorer_fit_oracle_v1_3_replay_overlay_v1"
REPLAY_OVERLAY_SELF_KEY = "replay_overlay_digest"
REPLAY_OVERLAY_MANIFEST_SCHEMA = (
    "go2_scorer_fit_oracle_v1_3_replay_overlay_manifest_v1"
)
REPLAY_OVERLAY_MANIFEST_SELF_KEY = "replay_overlay_manifest_digest"
REPLAY_OVERLAY_MANIFEST_PATH = OUT_ROOT / "replay_overlay_manifest.json"

FRESH_STATE_MANIFEST_SCHEMA = (
    "go2_scorer_fit_oracle_v1_3_fresh_calibration_state_manifest_v1"
)
FRESH_STATE_MANIFEST_SELF_KEY = "fresh_calibration_state_manifest_digest"
FRESH_STATE_MANIFEST_PATH = OUT_ROOT / "fresh_calibration/state_manifest.json"
FRESH_BRANCH_ROW_SCHEMA = "go2_scorer_fit_oracle_v1_3_fresh_calibration_branch_row_v1"
FRESH_BRANCH_ROW_SELF_KEY = "fresh_calibration_branch_row_digest"
FRESH_CORPUS_SCHEMA = "go2_scorer_fit_oracle_v1_3_fresh_calibration_corpus_v1"
FRESH_CORPUS_SELF_KEY = "fresh_calibration_corpus_digest"
FRESH_CORPUS_PATH = OUT_ROOT / "fresh_calibration/corpus_receipt.json"

SELECTOR_INTEGRITY_REPLACEMENT_PATH = (
    OUT_ROOT / "fresh_calibration/selector_integrity_replacement_v1.json"
)
FRESH_SELECTION_ATTEMPT_PATH = (
    OUT_ROOT / "fresh_calibration/selection_attempt.json"
)
FRESH_SELECTION_TASKS_ROOT = OUT_ROOT / "fresh_calibration/selection_tasks"
FRESH_SELECTION_RESULTS_ROOT = OUT_ROOT / "fresh_calibration/selection_results"
FRESH_SELECTION_TERMINAL_PATH = (
    OUT_ROOT / "fresh_calibration/selection_terminal.json"
)
FRESH_SELECTION_ATTEMPT_SCHEMA = (
    "go2_scorer_fit_oracle_v1_3_fresh_selection_attempt_v1"
)
FRESH_SELECTION_ATTEMPT_SELF_KEY = "fresh_selection_attempt_digest"
FRESH_SELECTION_TASK_SCHEMA = (
    "go2_scorer_fit_oracle_v1_3_fresh_selection_scene_task_v1"
)
FRESH_SELECTION_TASK_SELF_KEY = "fresh_selection_scene_task_digest"
FRESH_SELECTION_LAUNCH_SCHEMA = (
    "go2_scorer_fit_oracle_v1_3_fresh_selection_scene_launch_v1"
)
FRESH_SELECTION_LAUNCH_SELF_KEY = "fresh_selection_scene_launch_digest"
FRESH_SELECTION_RESULT_SCHEMA = (
    "go2_scorer_fit_oracle_v1_3_fresh_selection_scene_result_v1"
)
FRESH_SELECTION_RESULT_SELF_KEY = "fresh_selection_scene_result_digest"
FRESH_SELECTION_TERMINAL_SCHEMA = (
    "go2_scorer_fit_oracle_v1_3_fresh_selection_terminal_v1"
)
FRESH_SELECTION_TERMINAL_SELF_KEY = "fresh_selection_terminal_digest"

TRAINING_VIEW_SCHEMA = "go2_scorer_fit_oracle_v1_3_training_view_v1"
TRAINING_VIEW_ROW_SCHEMA = "go2_scorer_fit_oracle_v1_3_training_view_row_v1"
TRAINING_VIEW_SELF_KEY = "training_view_digest"
TRAINING_VIEW_PATH = OUT_ROOT / "training_view.json"

DIAGNOSTIC_PATH = (
    ROOT / "docs/lewm_go2_scorer_fit_v2_graph_label_failure_diagnostic_2026-08-15.json"
)
EXPECTED_DIAGNOSTIC_AUDIT_DIGEST = (
    "90dda36b7e85a650a75d1efb5d21faf3f3ed40f0860f3bdb3f6a4e69b8bd3741"
)
EXPECTED_V2_CORPUS_DIGEST = (
    "5216e2182a4e165a673714fcccbd6b769d01fa565a69a466b3cab066ab01ccc3"
)
EXPECTED_V2_ATTEMPTED = 1_440
EXPECTED_V2_VALID = 1_422
EXPECTED_V2_INVALID = 18
EXPECTED_OLD_STATE_COUNT = 120
EXPECTED_OLD_FIT_STATES = 96
EXPECTED_OLD_CALIBRATION_STATES = 24
EXPECTED_OLD_FIT_VALID = 1_146
EXPECTED_OLD_FIT_OVERLAYS = 6
EXPECTED_OLD_CALIBRATION_BRANCHES = 288
EXPECTED_FRESH_CALIBRATION_STATES = 24
EXPECTED_FRESH_CALIBRATION_BRANCHES = 288
EXPECTED_TRAINING_STATES = 120
EXPECTED_TRAINING_ROWS = 1_440
EXPECTED_CANDIDATES = tuple(range(12))

FAMILIES = (
    "large_enclosed_maze",
    "local_composite_motifs",
    "loop_alias_stress",
    "medium_enclosed_maze",
    "open_obstacle_field",
    "rough_local_dynamics",
    "small_enclosed_maze",
    "visual_sensor_stress",
)
STRATA = ("general", "safety_enriched", "completion_enriched")
FRESH_WARMUP_BLOCKS_MIN = 40
FRESH_WARMUP_BLOCKS_MAX = 120
HORIZON_POSE_ATOL = 1e-5
ACTION_ATOL = 1e-6

LABEL_KEYS = (
    "start_geodesic_m",
    "final_geodesic_m",
    "progress",
    "contact_fraction",
    "clearance_cost",
    "stuck_fraction",
    "fall",
    "safety",
    "completion",
    "utility",
    "min_clearance_m",
    "evaluation_points",
)

SOURCE_KIND_V2_VALID = "V2_VALID_ADOPTION"
SOURCE_KIND_REPLAY = "V13_REPLAY_OVERLAY"
SOURCE_KIND_FRESH = "V13_FRESH_CALIBRATION"


class WorkflowError(RuntimeError):
    """A fail-closed v1.3 workflow contract violation."""


class FreshSelectionChildUnresolved(WorkflowError):
    """One isolated selector child failed before publishing an exact result."""

    def __init__(self, task_digest: str, return_code: int | None) -> None:
        super().__init__(
            "isolated fresh-selection child has no valid durable result; "
            "selector attempt is terminal"
        )
        self.task_digest = task_digest
        self.return_code = None if return_code is None else int(return_code)


def _v13() -> Any:
    """Import the pure oracle lazily; importing this workflow starts no runtime."""

    from lewm.oracle import go2_branch_oracle_v1_3 as module
    return module


def canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise WorkflowError("payload is not finite canonical JSON") from exc


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def require_genesis_runtime() -> dict[str, Any]:
    """Require the exact CPU runtime frozen before physical branch execution."""

    import torch

    expected = V13_CONTRACT.GENESIS_RUNTIME_CONTRACT
    interpreter = Path(sys.executable).absolute()
    expected_interpreter = (
        ROOT / expected["interpreter_relative_path"]
    ).absolute()
    pyvenv = ROOT / expected["pyvenv_config_relative_path"]
    if interpreter != expected_interpreter:
        raise WorkflowError("v1.3 Genesis interpreter differs from the frozen runtime")
    if (
        not pyvenv.is_file()
        or pyvenv.is_symlink()
        or pyvenv.stat().st_size != expected["pyvenv_config_byte_count"]
        or file_sha256(pyvenv) != expected["pyvenv_config_sha256"]
    ):
        raise WorkflowError("v1.3 Genesis pyvenv binding changed")
    actual = {
        "interpreter_relative_path": _repo_relative(interpreter),
        "pyvenv_config_relative_path": _repo_relative(pyvenv),
        "pyvenv_config_sha256": file_sha256(pyvenv),
        "pyvenv_config_byte_count": pyvenv.stat().st_size,
        "python_version": sys.version.split()[0],
        "genesis_version": importlib.metadata.version("genesis-world"),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "backend": "cpu",
        "gstaichi_version": importlib.metadata.version("gstaichi"),
    }
    if actual != expected:
        raise WorkflowError("v1.3 Genesis runtime package binding changed")
    return actual


def _with_self_digest(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = copy.deepcopy(dict(payload))
    if key in result:
        raise WorkflowError(f"payload already contains {key}")
    result[key] = digest(result)
    return result


def _validate_self_digest(payload: Mapping[str, Any], key: str) -> str:
    body = copy.deepcopy(dict(payload))
    expected = body.pop(key, None)
    actual = digest(body)
    if expected != actual:
        raise WorkflowError(f"{key} self-digest mismatch")
    return actual


def _sha256_string(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _repo_relative(path: Path, *, root: Path = ROOT) -> str:
    try:
        return str(path.absolute().relative_to(root.absolute()))
    except ValueError as exc:
        raise WorkflowError(f"path escapes repository root: {path}") from exc


def guarded_output_path(relative: str | Path, *, out_root: Path = OUT_ROOT) -> Path:
    """Resolve one lexical descendant of the dedicated v1.3 output root."""

    relative = Path(relative)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise WorkflowError("v1.3 output path must be a non-empty relative descendant")
    root = out_root.absolute()
    if root == OUT_ROOT.absolute():
        if not root.is_symlink():
            raise WorkflowError("registered v1.3 output-root alias is missing")
        raw_target = root.readlink()
        target = raw_target if raw_target.is_absolute() else root.parent / raw_target
        registered = V13_CONTRACT.REGISTERED_GENERATED_TARGET_ROOT
        if (
            target != registered
            or registered.is_symlink()
            or not registered.is_dir()
            or registered.name != root.name
        ):
            raise WorkflowError("registered v1.3 output-root alias changed")
    elif root.is_symlink():
        raise WorkflowError("only the registered v1.3 output root may be a symlink")
    candidate = (root / relative).absolute()
    if root not in candidate.parents:
        raise WorkflowError("v1.3 output path escaped its dedicated root")
    cursor = candidate.parent
    while cursor != root.parent and cursor.exists():
        if cursor.is_symlink() and cursor != root:
            raise WorkflowError(f"v1.3 output ancestor is a symlink: {cursor}")
        if cursor == root:
            break
        cursor = cursor.parent
    return candidate


def _atomic_json(path: Path, payload: Mapping[str, Any], *, out_root: Path = OUT_ROOT,
                 exact_idempotence: bool = True) -> None:
    """Write one guarded artifact; never replace differing bytes."""

    try:
        relative = path.absolute().relative_to(out_root.absolute())
    except ValueError as exc:
        raise WorkflowError("write target is outside the v1.3 output root") from exc
    target = guarded_output_path(relative, out_root=out_root)
    encoded = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    if target.exists():
        if target.is_symlink() or not target.is_file():
            raise WorkflowError(f"existing output is not a regular file: {target}")
        if exact_idempotence and target.read_bytes() == encoded:
            return
        raise WorkflowError(f"refusing to replace existing v1.3 artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink() and target.parent != out_root.absolute():
        raise WorkflowError("v1.3 output parent must not be a symlink")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # A second process may have won after the existence check.
        if target.exists():
            raise WorkflowError(f"concurrent write refused for {target}")
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _read_regular_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise WorkflowError(f"required JSON is absent or not regular: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise WorkflowError(f"JSON root is not a mapping: {path}")
    return value


def load_diagnostic(path: Path = DIAGNOSTIC_PATH) -> dict[str, Any]:
    report = _read_regular_json(path)
    DIAG.validate_report(report)
    if (
        report.get("audit_digest") != EXPECTED_DIAGNOSTIC_AUDIT_DIGEST
        or report.get("corpus", {}).get("corpus_digest")
        != EXPECTED_V2_CORPUS_DIGEST
        or len(report.get("failure_inventory", [])) != EXPECTED_V2_INVALID
    ):
        raise WorkflowError("the frozen graph diagnostic identity changed")
    return report


def build_superseded_preattempt_transition() -> dict[str, Any]:
    payload = {
        "schema": SUPERSEDED_TRANSITION_SCHEMA,
        "status": "SUPERSEDED_PREATTEMPT_IMPLEMENTATION_CORRECTION",
        "superseded": copy.deepcopy(
            V13_CONTRACT.SUPERSEDED_PREATTEMPT_EXECUTION_AUTHORITY
        ),
        "archive_root": str(V13_CONTRACT.SUPERSEDED_PREATTEMPT_ARCHIVE_ROOT),
        "registered_successor_root": str(
            V13_CONTRACT.REGISTERED_GENERATED_TARGET_ROOT
        ),
        "old_bytes_preserved": True,
        "candidate_branch_execution_started": False,
        "retry_or_replacement": False,
    }
    return _with_self_digest(payload, SUPERSEDED_TRANSITION_SELF_KEY)


def validate_superseded_preattempt_archive() -> dict[str, Any]:
    """Validate the exact three-file zero-attempt predecessor after archival."""

    archive = V13_CONTRACT.SUPERSEDED_PREATTEMPT_ARCHIVE_ROOT
    if not archive.is_dir() or archive.is_symlink():
        raise WorkflowError("superseded preattempt archive is absent or not regular")
    expected = {
        "authority.json": V13_CONTRACT.SUPERSEDED_PREATTEMPT_EXECUTION_AUTHORITY[
            "authority"
        ],
        "replay_plan.json": V13_CONTRACT.SUPERSEDED_PREATTEMPT_EXECUTION_AUTHORITY[
            "replay_plan"
        ],
        "equivalence_receipt.json":
            V13_CONTRACT.SUPERSEDED_PREATTEMPT_EXECUTION_AUTHORITY[
                "equivalence_receipt"
            ],
    }
    if {path.name for path in archive.iterdir()} != set(expected):
        raise WorkflowError("superseded preattempt archive inventory changed")
    for name, binding in expected.items():
        path = archive / name
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != binding["byte_count"]
            or file_sha256(path) != binding["raw_sha256"]
        ):
            raise WorkflowError(f"superseded preattempt archive changed: {name}")
    return build_superseded_preattempt_transition()


def _source_binding(root: Path = ROOT) -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        text=True,
    ).strip()
    if dirty:
        raise WorkflowError("preregistration authority requires a clean tracked source tree")
    bindings = V13_CONTRACT.source_bindings(root)
    preregistration_path = root / V13_CONTRACT.PREREGISTRATION_PATH
    preregistration = _read_regular_json(preregistration_path)
    V13_CONTRACT.validate_preregistration(preregistration, root=root)
    return {
        "source_repository_commit": commit,
        "source_repository_clean": True,
        "source_files": bindings,
        "source_files_digest": digest(bindings),
        "preregistration_path": _repo_relative(preregistration_path, root=root),
        "preregistration_sha256": file_sha256(preregistration_path),
        "preregistration_byte_count": preregistration_path.stat().st_size,
        "preregistration": preregistration,
    }


def _diagnostic_replay_projection(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = []
    for row in report.get("failure_inventory", []):
        projection = {
            "branch_identity_digest": row.get("branch_identity_digest"),
            "branch_row_digest": row.get("branch_row_digest"),
            "state_id": row.get("state_id"),
            "state_identity_digest": row.get("state_identity_digest"),
            "assignment_identity_digest": row.get("assignment_identity_digest"),
            "candidate_index": row.get("candidate_index"),
            "scene_id": row.get("scene_id"),
            "split_role": row.get("split_role"),
            "primary_category": row.get("primary_category"),
        }
        if (
            not all(_sha256_string(projection[key]) for key in (
                "branch_identity_digest", "branch_row_digest",
                "state_identity_digest", "assignment_identity_digest",
            ))
            or not isinstance(projection["candidate_index"], int)
            or projection["candidate_index"] not in EXPECTED_CANDIDATES
            or projection["split_role"] not in {"fit", "calibration"}
        ):
            raise WorkflowError("diagnostic contains a malformed replay identity")
        entries.append(projection)
    entries.sort(key=lambda value: value["branch_identity_digest"])
    if len(entries) != EXPECTED_V2_INVALID or len({
            row["branch_identity_digest"] for row in entries}) != EXPECTED_V2_INVALID:
        raise WorkflowError("diagnostic exact-replay identity set is not eighteen unique rows")
    return entries


def build_authority(
    source_binding: Mapping[str, Any], report: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the outcome-independent execution authority from frozen diagnosis."""

    DIAG.validate_report(report)
    replay_entries = _diagnostic_replay_projection(report)
    oracle = _v13()
    contract_digest = V13_CONTRACT.contract_digest()
    if {
        row["branch_identity_digest"] for row in replay_entries
    } != set(V13_CONTRACT.FAILED_BRANCH_ALLOWLIST):
        raise WorkflowError("diagnostic replay set differs from the frozen v1.3 contract")
    payload = {
        "schema": AUTHORITY_SCHEMA,
        "status": STATUS,
        "source_binding": copy.deepcopy(dict(source_binding)),
        "oracle_v1_3_digest": oracle.oracle_digest(),
        "scorer_fit_oracle_v1_3_contract_digest": contract_digest,
        "superseded_preattempt_transition_digest":
            build_superseded_preattempt_transition()[
                SUPERSEDED_TRANSITION_SELF_KEY
            ],
        "oracle_contract": oracle.oracle_contract(),
        "diagnostic": {
            "path": _repo_relative(DIAGNOSTIC_PATH),
            "audit_digest": report["audit_digest"],
            "audited_source_commit": report["audited_source_commit"],
        },
        "v2_corpus": {
            "corpus_digest": EXPECTED_V2_CORPUS_DIGEST,
            "attempted_branch_count": EXPECTED_V2_ATTEMPTED,
            "valid_branch_count": EXPECTED_V2_VALID,
            "invalid_branch_count": EXPECTED_V2_INVALID,
            "state_count": EXPECTED_OLD_STATE_COUNT,
        },
        "exact_replay": {
            "identity_count": EXPECTED_V2_INVALID,
            "identity_set_digest": digest(sorted(
                row["branch_identity_digest"] for row in replay_entries)),
            "identity_projection_digest": digest(replay_entries),
            "selection_source": "diagnostic inventory only",
            "attempt_policy": "AT_MOST_ONCE_PER_EXACT_IDENTITY_NO_RETRY_OR_REPLACEMENT",
            "rgb_policy": "REUSE_IMMUTABLE_V2_FRAMES_NO_RERENDER",
            "horizon_pose_atol": HORIZON_POSE_ATOL,
            "action_atol": ACTION_ATOL,
            "source_snapshot_digest_preserved_as_lineage": True,
            "source_snapshot_digest_equality_required": False,
            "prebranch_physical_witness_required": True,
        },
        "genesis_runtime": dict(V13_CONTRACT.GENESIS_RUNTIME_CONTRACT),
        "legacy_equivalence": {
            "adopted_valid_count": EXPECTED_V2_VALID,
            "fit_adopted_count": EXPECTED_OLD_FIT_VALID,
            "historical_calibration_adopted_but_not_training_eligible": 276,
            "comparison": "exact canonical legacy-label projection",
        },
        "fresh_calibration_selection": {
            "contract_selector_digest":
                V13_CONTRACT.fresh_calibration_selector_digest(),
            "families": list(FAMILIES),
            "strata": list(STRATA),
            "states_per_family_per_stratum": 1,
            "state_count": EXPECTED_FRESH_CALIBRATION_STATES,
            "candidate_indices": list(EXPECTED_CANDIDATES),
            "branch_count": EXPECTED_FRESH_CALIBRATION_BRANCHES,
            "scene_order": "lexical within family after frozen exclusions",
            "one_state_per_scene": True,
            "warmup_blocks_inclusive": [
                FRESH_WARMUP_BLOCKS_MIN, FRESH_WARMUP_BLOCKS_MAX,
            ],
            "drive_seed": "v1.2 V1._drive_seed(scene_id)",
            "freeze_all_states_before_any_fresh_branch": True,
            "worker_or_infrastructure_failure": "terminal; never a scene rejection",
            "outcome_dependent_replacement": False,
        },
        "training_view": {
            "fit": {
                "state_count": EXPECTED_OLD_FIT_STATES,
                "v2_valid_rows": EXPECTED_OLD_FIT_VALID,
                "v1_3_replay_overlays": EXPECTED_OLD_FIT_OVERLAYS,
                "branch_count": 1_152,
            },
            "calibration": {
                "state_count": EXPECTED_FRESH_CALIBRATION_STATES,
                "branch_count": EXPECTED_FRESH_CALIBRATION_BRANCHES,
            },
            "row_count": EXPECTED_TRAINING_ROWS,
            "historical_calibration": "DEVELOPMENT_ONLY_AND_EXCLUDED",
        },
        "prohibited": {
            "modify_or_replace_v2_rows_or_frames": True,
            "retry_or_replace_executed_branch": True,
            "train_with_missing_label": True,
            "encode_or_train_in_this_workflow": True,
            "open_predictor_or_utility_shard": True,
            "generate_final_200_state_benchmark": True,
        },
        "stage_order": [
            "issue-authority", "issue-replay-plan", "adopt-valid",
            "replay-failures", "select-calibration", "generate-calibration",
            "compose-training-view",
        ],
    }
    return _with_self_digest(payload, AUTHORITY_SELF_KEY)


def validate_authority(payload: Mapping[str, Any]) -> None:
    _validate_self_digest(payload, AUTHORITY_SELF_KEY)
    if (
        payload.get("schema") != AUTHORITY_SCHEMA
        or payload.get("status") != STATUS
        or payload.get("v2_corpus", {}).get("corpus_digest")
        != EXPECTED_V2_CORPUS_DIGEST
        or payload.get("exact_replay", {}).get("identity_count")
        != EXPECTED_V2_INVALID
        or payload.get("fresh_calibration_selection", {}).get("state_count")
        != EXPECTED_FRESH_CALIBRATION_STATES
        or payload.get("fresh_calibration_selection", {}).get("branch_count")
        != EXPECTED_FRESH_CALIBRATION_BRANCHES
        or payload.get("training_view", {}).get("row_count")
        != EXPECTED_TRAINING_ROWS
        or payload.get("oracle_v1_3_digest") != _v13().oracle_digest()
        or payload.get("scorer_fit_oracle_v1_3_contract_digest")
        != V13_CONTRACT.contract_digest()
        or payload.get("genesis_runtime")
        != V13_CONTRACT.GENESIS_RUNTIME_CONTRACT
        or payload.get("superseded_preattempt_transition_digest")
        != build_superseded_preattempt_transition()[SUPERSEDED_TRANSITION_SELF_KEY]
    ):
        raise WorkflowError("v1.3 preregistration authority changed")
    prohibited = payload.get("prohibited")
    if not isinstance(prohibited, Mapping) or not prohibited or not all(
            value is True for value in prohibited.values()):
        raise WorkflowError("v1.3 prohibited-operation gate is open")
    source_binding = payload.get("source_binding")
    if not isinstance(source_binding, Mapping):
        raise WorkflowError("v1.3 source/preregistration binding is absent")
    preregistration = source_binding.get("preregistration")
    if isinstance(preregistration, Mapping):
        V13_CONTRACT.validate_original_preregistration(preregistration)


def issue_authority(*, root: Path = ROOT, out_root: Path = OUT_ROOT) -> dict[str, Any]:
    report = load_diagnostic(root / DIAGNOSTIC_PATH.relative_to(ROOT))
    authority = build_authority(_source_binding(root), report)
    if out_root.absolute() == OUT_ROOT.absolute():
        transition = validate_superseded_preattempt_archive()
        _atomic_json(
            out_root / SUPERSEDED_TRANSITION_PATH.name,
            transition,
            out_root=out_root,
        )
    _atomic_json(out_root / "authority.json", authority, out_root=out_root)
    return authority


def load_authority(path: Path = AUTHORITY_PATH) -> dict[str, Any]:
    guarded_output_path("authority.json", out_root=path.parent)
    if path.parent.absolute() == OUT_ROOT.absolute():
        transition = _read_regular_json(
            path.parent / SUPERSEDED_TRANSITION_PATH.name
        )
        _validate_self_digest(transition, SUPERSEDED_TRANSITION_SELF_KEY)
        if transition != validate_superseded_preattempt_archive():
            raise WorkflowError("superseded preattempt transition changed")
    authority = _read_regular_json(path)
    validate_authority(authority)
    live_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT, text=True,
    ).strip()
    if (
        dirty
        or live_commit
        != authority["source_binding"].get("source_repository_commit")
    ):
        raise WorkflowError("v1.3 runtime source commit is no longer exact and clean")
    preregistration = authority["source_binding"].get("preregistration")
    if isinstance(preregistration, Mapping):
        V13_CONTRACT.validate_original_preregistration(preregistration, root=ROOT)
        preregistration_path = ROOT / V13_CONTRACT.ORIGINAL_PREREGISTRATION_PATH
        committed = _read_regular_json(preregistration_path)
        if (
            committed != preregistration
            or authority["source_binding"].get("preregistration_path")
            != _repo_relative(preregistration_path)
            or authority["source_binding"].get("preregistration_sha256")
            != file_sha256(preregistration_path)
            or authority["source_binding"].get("preregistration_byte_count")
            != preregistration_path.stat().st_size
        ):
            raise WorkflowError("committed v1.3 preregistration binding changed")
    return authority


def _contract_output_path(path: str, *, out_root: Path) -> Path:
    try:
        relative = Path(path).relative_to(V13_CONTRACT.GENERATED_ROOT)
    except ValueError as exc:
        raise WorkflowError("selector-integrity path escaped the output root") from exc
    return guarded_output_path(relative, out_root=out_root)


def _mode_string(path: Path) -> str:
    return f"{stat.S_IMODE(path.stat().st_mode):04o}"


def _binding_set(root: Path, *, out_root: Path) -> list[dict[str, Any]]:
    if not root.is_dir() or root.is_symlink():
        raise WorkflowError(f"preserved artifact root changed: {root}")
    records = []
    for path in sorted(root.iterdir(), key=lambda value: value.name):
        if not path.is_file() or path.is_symlink():
            raise WorkflowError(f"preserved artifact inventory changed: {root}")
        relative = path.absolute().relative_to(out_root.absolute())
        records.append({
            "path": str(V13_CONTRACT.GENERATED_ROOT / relative),
            "sha256": file_sha256(path),
            "byte_count": path.stat().st_size,
        })
    return records


def _validate_preserved_selector_predecessor(
    *, out_root: Path = OUT_ROOT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the frozen aa7c authority and its completed upstream outputs."""

    replacement = V13_CONTRACT.SELECTOR_INTEGRITY_REPLACEMENT
    expected_artifacts = replacement["preserved_predecessor_artifacts"]
    validated: dict[str, Any] = {}
    base_authority: dict[str, Any] | None = None
    for name in (
        "authority", "replay_plan", "equivalence_receipt",
        "replay_overlay_manifest",
    ):
        binding = expected_artifacts[name]
        path = _contract_output_path(binding["path"], out_root=out_root)
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != binding["byte_count"]
            or file_sha256(path) != binding["raw_sha256"]
            or _mode_string(path) != binding["file_mode"]
        ):
            raise WorkflowError(f"preserved selector predecessor changed: {name}")
        payload = _read_regular_json(path)
        _validate_self_digest(payload, binding["self_key"])
        if payload.get(binding["self_key"]) != binding["self_digest"]:
            raise WorkflowError(f"preserved selector predecessor digest changed: {name}")
        for count_key in (
            "entry_count", "compared_count", "mismatch_count", "overlay_count",
            "fit_overlay_count", "historical_calibration_overlay_count",
        ):
            if count_key in binding and payload.get(count_key) != binding[count_key]:
                raise WorkflowError(
                    f"preserved selector predecessor count changed: {name}/{count_key}"
                )
        validated[name] = {
            "path": binding["path"],
            "self_digest": binding["self_digest"],
            "raw_sha256": binding["raw_sha256"],
            "byte_count": binding["byte_count"],
        }
        if name == "authority":
            base_authority = payload

    for name in ("replay_attempt_markers", "replay_overlays"):
        binding = expected_artifacts[name]
        root = _contract_output_path(binding["root"], out_root=out_root)
        records = _binding_set(root, out_root=out_root)
        if (
            len(records) != binding["count"]
            or any(
                _mode_string(root / Path(record["path"]).name)
                != binding["file_mode"]
                for record in records
            )
            or digest(records) != binding["file_binding_set_digest"]
        ):
            raise WorkflowError(f"preserved selector predecessor set changed: {name}")
        validated[name] = {
            "root": binding["root"],
            "count": binding["count"],
            "file_binding_set_digest": binding["file_binding_set_digest"],
        }

    if base_authority is None:
        raise WorkflowError("preserved selector predecessor authority is absent")
    if (
        base_authority.get(AUTHORITY_SELF_KEY)
        != replacement["predecessor_authority_digest"]
        or base_authority.get("scorer_fit_oracle_v1_3_contract_digest")
        != replacement["predecessor_contract_digest"]
        or base_authority.get("source_binding", {}).get("source_repository_commit")
        != replacement["predecessor_source_commit"]
        or base_authority.get("source_binding", {}).get("source_files_digest")
        != replacement["predecessor_source_files_digest"]
    ):
        raise WorkflowError("preserved selector predecessor authority lineage changed")
    return base_authority, validated


def build_selector_integrity_replacement_authority(
    *, base_authority: Mapping[str, Any], source_binding: Mapping[str, Any],
    predecessor_validation: Mapping[str, Any],
) -> dict[str, Any]:
    replacement = V13_CONTRACT.SELECTOR_INTEGRITY_REPLACEMENT
    preregistration = source_binding.get("preregistration")
    if not isinstance(preregistration, Mapping):
        raise WorkflowError("selector-integrity successor preregistration is absent")
    if (
        base_authority.get(AUTHORITY_SELF_KEY)
        != replacement["predecessor_authority_digest"]
        or source_binding.get("preregistration_path")
        != replacement["successor_preregistration_path"]
    ):
        raise WorkflowError("selector-integrity authority lineage changed")
    payload = {
        "schema": V13_CONTRACT.SELECTOR_INTEGRITY_REPLACEMENT_SCHEMA,
        "status": "FROZEN_SELECTOR_ONLY_INTEGRITY_REPLACEMENT_AUTHORITY",
        "complete": True,
        "selector_integrity_replacement": copy.deepcopy(replacement),
        "selector_integrity_replacement_digest":
            V13_CONTRACT.selector_integrity_replacement_digest(),
        "predecessor_authority_digest": base_authority[AUTHORITY_SELF_KEY],
        "predecessor_artifact_validation": copy.deepcopy(
            dict(predecessor_validation)
        ),
        "failed_selector_outputs_verified": copy.deepcopy(
            replacement["failed_selector_outputs"]
        ),
        "successor_preregistration_path": source_binding["preregistration_path"],
        "successor_preregistration_digest": preregistration[
            replacement["successor_preregistration_digest_key"]
        ],
        "successor_contract_digest": V13_CONTRACT.contract_digest(),
        "successor_source_binding": copy.deepcopy(dict(source_binding)),
        "one_shot_selector_attempt_authorised": True,
        "candidate_branch_execution_authorised": False,
    }
    return _with_self_digest(
        payload, V13_CONTRACT.SELECTOR_INTEGRITY_REPLACEMENT_SELF_KEY
    )


def validate_selector_integrity_replacement_authority(
    correction: Mapping[str, Any], *, base_authority: Mapping[str, Any],
    predecessor_validation: Mapping[str, Any], root: Path | None = None,
) -> None:
    replacement = V13_CONTRACT.SELECTOR_INTEGRITY_REPLACEMENT
    _validate_self_digest(
        correction, V13_CONTRACT.SELECTOR_INTEGRITY_REPLACEMENT_SELF_KEY
    )
    source_binding = correction.get("successor_source_binding")
    preregistration = (
        source_binding.get("preregistration")
        if isinstance(source_binding, Mapping) else None
    )
    if (
        correction.get("schema")
        != V13_CONTRACT.SELECTOR_INTEGRITY_REPLACEMENT_SCHEMA
        or correction.get("status")
        != "FROZEN_SELECTOR_ONLY_INTEGRITY_REPLACEMENT_AUTHORITY"
        or correction.get("complete") is not True
        or correction.get("selector_integrity_replacement") != replacement
        or correction.get("selector_integrity_replacement_digest")
        != V13_CONTRACT.selector_integrity_replacement_digest()
        or correction.get("predecessor_authority_digest")
        != base_authority.get(AUTHORITY_SELF_KEY)
        or correction.get("predecessor_artifact_validation")
        != predecessor_validation
        or correction.get("failed_selector_outputs_verified")
        != replacement["failed_selector_outputs"]
        or correction.get("successor_preregistration_path")
        != replacement["successor_preregistration_path"]
        or not isinstance(preregistration, Mapping)
        or correction.get("successor_preregistration_digest")
        != preregistration.get(replacement["successor_preregistration_digest_key"])
        or correction.get("successor_contract_digest")
        != V13_CONTRACT.contract_digest()
        or correction.get("one_shot_selector_attempt_authorised") is not True
        or correction.get("candidate_branch_execution_authorised") is not False
    ):
        raise WorkflowError("selector-integrity replacement authority changed")
    V13_CONTRACT.validate_preregistration(preregistration)
    if root is not None:
        live = _source_binding(root)
        if source_binding != live:
            raise WorkflowError("selector-integrity successor source binding changed")


def issue_selector_integrity_replacement(
    *, root: Path = ROOT, out_root: Path = OUT_ROOT,
) -> dict[str, Any]:
    base, predecessor = _validate_preserved_selector_predecessor(
        out_root=out_root
    )
    correction_path = _contract_output_path(
        str(V13_CONTRACT.SELECTOR_INTEGRITY_REPLACEMENT_AUTHORITY_PATH),
        out_root=out_root,
    )
    if correction_path.exists():
        correction = _read_regular_json(correction_path)
        validate_selector_integrity_replacement_authority(
            correction, base_authority=base,
            predecessor_validation=predecessor, root=root,
        )
        return correction
    fresh_root = _contract_output_path(
        str(V13_CONTRACT.FRESH_CALIBRATION_ROOT), out_root=out_root
    )
    if fresh_root.exists():
        raise WorkflowError(
            "selector-integrity replacement requires the recorded absent fresh root"
        )
    source_binding = _source_binding(root)
    correction = build_selector_integrity_replacement_authority(
        base_authority=base, source_binding=source_binding,
        predecessor_validation=predecessor,
    )
    validate_selector_integrity_replacement_authority(
        correction, base_authority=base,
        predecessor_validation=predecessor, root=root,
    )
    _atomic_json(correction_path, correction, out_root=out_root)
    return correction


def load_selector_integrity_replacement_authority(
    *, root: Path = ROOT, out_root: Path = OUT_ROOT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base, predecessor = _validate_preserved_selector_predecessor(
        out_root=out_root
    )
    correction = _read_regular_json(_contract_output_path(
        str(V13_CONTRACT.SELECTOR_INTEGRITY_REPLACEMENT_AUTHORITY_PATH),
        out_root=out_root,
    ))
    validate_selector_integrity_replacement_authority(
        correction, base_authority=base,
        predecessor_validation=predecessor, root=root,
    )
    return base, correction


# ---------------------------------------------------------------- V2 inputs --
def load_v2_corpus(*, v2_root: Path = V2_ROOT) -> dict[str, Any]:
    """Open the finite V2 manifest/ledger/row surface; never mutate it."""

    state_manifest = _read_regular_json(v2_root / "state_manifest_v2.json")
    assignment_manifest = _read_regular_json(
        v2_root / "full_bank_assignment_manifest_v2.json"
    )
    receipt = _read_regular_json(v2_root / "corpus_receipt_v2.json")
    def validate_v2_digest(payload: Mapping[str, Any], key: str) -> None:
        body = dict(payload)
        expected = body.pop(key, None)
        if expected != B.canonical_digest(body):
            raise WorkflowError(f"V2 {key} self-digest changed")

    validate_v2_digest(state_manifest, "state_manifest_digest")
    validate_v2_digest(
        assignment_manifest, "full_bank_assignment_manifest_digest"
    )
    identity_payload = receipt.get("corpus_digest_payload")
    if (
        not isinstance(identity_payload, Mapping)
        or receipt.get("corpus_digest") != B.canonical_digest(identity_payload)
        or receipt.get("corpus_digest") != EXPECTED_V2_CORPUS_DIGEST
        or state_manifest.get("state_manifest_digest")
        != V13_CONTRACT.FROZEN_STATE_MANIFEST_DIGEST
        or assignment_manifest.get("full_bank_assignment_manifest_digest")
        != V13_CONTRACT.FROZEN_ASSIGNMENT_MANIFEST_DIGEST
        or receipt.get("state_manifest_digest")
        != V13_CONTRACT.FROZEN_STATE_MANIFEST_DIGEST
        or receipt.get("full_bank_assignment_manifest_digest")
        != V13_CONTRACT.FROZEN_ASSIGNMENT_MANIFEST_DIGEST
        or identity_payload.get("state_manifest_digest")
        != V13_CONTRACT.FROZEN_STATE_MANIFEST_DIGEST
        or identity_payload.get("full_bank_assignment_manifest_digest")
        != V13_CONTRACT.FROZEN_ASSIGNMENT_MANIFEST_DIGEST
        or receipt.get("branch_rows_sha256")
        != V13_CONTRACT.FROZEN_BRANCH_ROWS_SHA256
        or identity_payload.get("branch_rows_sha256")
        != V13_CONTRACT.FROZEN_BRANCH_ROWS_SHA256
        or identity_payload.get("branch_identity_set_digest")
        != V13_CONTRACT.FROZEN_BRANCH_IDENTITY_SET_DIGEST
        or receipt.get("attempted_branches") != EXPECTED_V2_ATTEMPTED
        or receipt.get("valid_branches") != EXPECTED_V2_VALID
        or receipt.get("invalid_branches") != EXPECTED_V2_INVALID
    ):
        raise WorkflowError("V2 corpus receipt changed")
    ledger_path = v2_root / "branch_rows_v2.jsonl"
    if not ledger_path.is_file() or ledger_path.is_symlink():
        raise WorkflowError("V2 branch ledger is unavailable")
    if file_sha256(ledger_path) != receipt.get("branch_rows_sha256"):
        raise WorkflowError("V2 branch ledger raw digest changed")
    rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    if (
        len(rows) != EXPECTED_V2_ATTEMPTED
        or sum(row.get("valid") is True for row in rows) != EXPECTED_V2_VALID
        or sum(row.get("valid") is False for row in rows) != EXPECTED_V2_INVALID
        or [row.get("branch_row_digest") for row in rows]
        != identity_payload.get("branch_row_digests")
        or B.canonical_digest(sorted(
            str(row.get("branch_identity_digest", "")) for row in rows
        )) != V13_CONTRACT.FROZEN_BRANCH_IDENTITY_SET_DIGEST
    ):
        raise WorkflowError("V2 branch ledger inventory changed")
    row_paths: dict[str, Path] = {}
    for row in rows:
        body = dict(row)
        expected = body.pop("branch_row_digest", None)
        if expected != B.canonical_digest(body):
            raise WorkflowError("V2 row self-digest changed")
        identity = str(row.get("branch_identity_digest", ""))
        path = v2_root / "row_records_v2" / f"{identity}.json"
        if _read_regular_json(path) != row:
            raise WorkflowError("V2 row record differs from the ledger")
        row_paths[identity] = path
    states = state_manifest.get("states", [])
    roles = Counter(state.get("split_role") for state in states)
    if (
        len(states) != EXPECTED_OLD_STATE_COUNT
        or roles != Counter({
            "fit": EXPECTED_OLD_FIT_STATES,
            "calibration": EXPECTED_OLD_CALIBRATION_STATES,
        })
        or len({state.get("scene_id") for state in states}) != len(states)
    ):
        raise WorkflowError("V2 state identity/role inventory changed")
    return {
        "state_manifest": state_manifest,
        "assignment_manifest": assignment_manifest,
        "receipt": receipt,
        "rows": rows,
        "row_paths": row_paths,
    }


# -------------------------------------------------------------- replay plan --
def _validate_v2_row_inventory(v2_rows: Sequence[Mapping[str, Any]]) -> None:
    identities = [str(row.get("branch_identity_digest", "")) for row in v2_rows]
    valid_count = sum(row.get("valid") is True for row in v2_rows)
    invalid = [row for row in v2_rows if row.get("valid") is False]
    if (
        len(v2_rows) != EXPECTED_V2_ATTEMPTED
        or valid_count != EXPECTED_V2_VALID
        or len(invalid) != EXPECTED_V2_INVALID
        or any(row.get("invalid_reason")
               != "unlocatable_or_unreachable_geodesic" for row in invalid)
        or len(set(identities)) != len(identities)
        or any(not _sha256_string(identity) for identity in identities)
    ):
        raise WorkflowError("V2 row inventory is not exact 1440/1422/18")


def build_replay_plan(
    authority: Mapping[str, Any], report: Mapping[str, Any],
    v2_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    validate_authority(authority)
    _validate_v2_row_inventory(v2_rows)
    diagnostic_entries = _diagnostic_replay_projection(report)
    row_by_identity = {
        str(row.get("branch_identity_digest")): row for row in v2_rows
    }
    entries = []
    for expected in diagnostic_entries:
        row = row_by_identity.get(expected["branch_identity_digest"])
        if row is None:
            raise WorkflowError("diagnosed replay identity is absent from V2")
        for key in (
            "branch_identity_digest", "branch_row_digest", "state_id",
            "state_identity_digest", "assignment_identity_digest",
            "candidate_index", "scene_id", "split_role",
        ):
            if row.get(key) != expected[key]:
                raise WorkflowError(f"diagnostic/V2 replay binding mismatch: {key}")
        if (
            row.get("valid") is not False
            or row.get("invalid_reason") != "unlocatable_or_unreachable_geodesic"
        ):
            raise WorkflowError("exact replay identity is not one diagnosed V2 refusal")
        source_context_poses = [
            frame.get("camera_pose_world")
            for frame in row.get("context_frames", [])
        ]
        source_horizon_poses = [
            frame.get("camera_pose_world")
            for frame in row.get("horizon_frames", [])
        ]
        source_prebranch_witness = _normalised_prebranch_witness(
            proprio=row.get("proprio"), control=row.get("control"),
            action_context_blocks=row.get("action_context_blocks"),
            previous_applied_command=row.get("previous_applied_command"),
        )
        entries.append({
            **expected,
            "snapshot_digest": row.get("snapshot_digest"),
            "candidate": row.get("candidate"),
            "primitives": row.get("primitives"),
            "goal": row.get("goal"),
            "requested": row.get("requested"),
            "post_slew": row.get("post_slew"),
            "source_context_camera_poses": source_context_poses,
            "source_context_pose_digest": digest(source_context_poses),
            "source_prebranch_witness": source_prebranch_witness,
            "source_prebranch_witness_digest":
                digest(source_prebranch_witness),
            "horizon_camera_poses": source_horizon_poses,
            "source_horizon_pose_digest": digest(source_horizon_poses),
        })
    entries.sort(key=lambda value: value["branch_identity_digest"])
    payload = {
        "schema": REPLAY_PLAN_SCHEMA,
        "status": STATUS,
        "authority_digest": authority[AUTHORITY_SELF_KEY],
        "oracle_v1_3_digest": authority["oracle_v1_3_digest"],
        "scorer_fit_oracle_v1_3_contract_digest": authority[
            "scorer_fit_oracle_v1_3_contract_digest"
        ],
        "v2_corpus_digest": EXPECTED_V2_CORPUS_DIGEST,
        "diagnostic_audit_digest": report["audit_digest"],
        "entry_count": len(entries),
        "identity_set_digest": digest(sorted(
            row["branch_identity_digest"] for row in entries)),
        "entries": entries,
        "attempt_policy": authority["exact_replay"]["attempt_policy"],
        "writes": "v1.3 overlay root only",
    }
    return _with_self_digest(payload, REPLAY_PLAN_SELF_KEY)


def validate_replay_plan(
    plan: Mapping[str, Any], authority: Mapping[str, Any],
    report: Mapping[str, Any], v2_rows: Sequence[Mapping[str, Any]],
) -> None:
    _validate_self_digest(plan, REPLAY_PLAN_SELF_KEY)
    expected = build_replay_plan(authority, report, v2_rows)
    if dict(plan) != expected:
        raise WorkflowError("exact-eighteen replay plan changed")
    if (
        plan.get("entry_count") != EXPECTED_V2_INVALID
        or len(plan.get("entries", [])) != EXPECTED_V2_INVALID
        or plan.get("identity_set_digest")
        != authority["exact_replay"]["identity_set_digest"]
    ):
        raise WorkflowError("replay plan is not the exact diagnosed eighteen")


def issue_replay_plan(*, out_root: Path = OUT_ROOT,
                      v2_root: Path = V2_ROOT) -> dict[str, Any]:
    authority = load_authority(out_root / "authority.json")
    report = load_diagnostic()
    corpus = load_v2_corpus(v2_root=v2_root)
    plan = build_replay_plan(authority, report, corpus["rows"])
    _atomic_json(out_root / "replay_plan.json", plan, out_root=out_root)
    return plan


def load_replay_plan(*, out_root: Path = OUT_ROOT,
                     v2_root: Path = V2_ROOT,
                     authority: Mapping[str, Any] | None = None) -> dict[str, Any]:
    plan = _read_regular_json(out_root / "replay_plan.json")
    if authority is None:
        authority = load_authority(out_root / "authority.json")
    corpus = load_v2_corpus(v2_root=v2_root)
    validate_replay_plan(plan, authority, load_diagnostic(), corpus["rows"])
    return plan


# ---------------------------------------------------------- valid adoption --
def _label_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    projection = {key: row.get(key) for key in LABEL_KEYS}
    required_finite = (
        "start_geodesic_m", "progress",
        "contact_fraction", "clearance_cost", "stuck_fraction", "fall",
        "safety", "completion", "utility", "min_clearance_m",
    )
    if (
        any(isinstance(projection[key], bool) or not isinstance(
            projection[key], (int, float)) or not math.isfinite(
                float(projection[key])) for key in required_finite)
        or not (
            projection["final_geodesic_m"] is None
            or (
                not isinstance(projection["final_geodesic_m"], bool)
                and isinstance(projection["final_geodesic_m"], (int, float))
                and math.isfinite(float(projection["final_geodesic_m"]))
                and float(projection["final_geodesic_m"]) >= 0.0
            )
        )
        or isinstance(projection["evaluation_points"], bool)
        or not isinstance(projection["evaluation_points"], int)
        or projection["evaluation_points"] <= 0
    ):
        raise WorkflowError("branch label projection is incomplete or non-finite")
    return projection


def _validate_legacy_formula(projection: Mapping[str, Any]) -> None:
    """Recompute every aggregate that is recoverable from a valid V2 row."""

    if projection["final_geodesic_m"] is None:
        raise WorkflowError("legacy-valid equivalence requires a finite final distance")
    progress = V12.progress_from_distances(
        float(projection["start_geodesic_m"]),
        float(projection["final_geodesic_m"]),
    )
    graded = (
        float(projection["contact_fraction"])
        + float(projection["clearance_cost"])
        + float(projection["stuck_fraction"])
    ) / 3.0
    safety = max(float(projection["fall"]), graded)
    utility = V12.composite_utility(
        float(projection["progress"]), float(projection["safety"]),
        float(projection["completion"]),
    )
    if (
        progress is None
        or progress != projection["progress"]
        or safety != projection["safety"]
        or utility != projection["utility"]
        or float(projection["completion"]) not in (0.0, 1.0)
        or any(not 0.0 <= float(projection[key]) <= 1.0 for key in (
            "contact_fraction", "clearance_cost", "stuck_fraction", "fall",
            "safety",
        ))
    ):
        raise WorkflowError("legacy-valid aggregate formula does not recompute exactly")


def build_equivalence_receipt(
    authority: Mapping[str, Any], v2_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    validate_authority(authority)
    _validate_v2_row_inventory(v2_rows)
    oracle = _v13()
    pairs = []
    role_counts: Counter[str] = Counter()
    for row in v2_rows:
        if row.get("valid") is not True:
            continue
        legacy = _label_projection(row)
        _validate_legacy_formula(legacy)
        adopted = oracle.legacy_label_projection(row)
        if adopted != legacy:
            raise WorkflowError("v1.3 legacy projection differs from V2 label bytes")
        role = str(row.get("split_role"))
        role_counts[role] += 1
        pairs.append({
            "branch_identity_digest": row["branch_identity_digest"],
            "branch_row_digest": row["branch_row_digest"],
            "split_role": role,
            "v1_2_label_digest": digest(legacy),
            "v1_3_label_digest": digest(adopted),
            "exact_equal": True,
        })
    pairs.sort(key=lambda value: value["branch_identity_digest"])
    if (
        len(pairs) != EXPECTED_V2_VALID
        or role_counts != Counter({"fit": EXPECTED_OLD_FIT_VALID,
                                  "calibration": 276})
    ):
        raise WorkflowError("V2 valid adoption role/count inventory changed")
    payload = {
        "schema": EQUIVALENCE_SCHEMA,
        "status": STATUS,
        "authority_digest": authority[AUTHORITY_SELF_KEY],
        "oracle_v1_3_digest": authority["oracle_v1_3_digest"],
        "scorer_fit_oracle_v1_3_contract_digest": authority[
            "scorer_fit_oracle_v1_3_contract_digest"
        ],
        "v2_corpus_digest": EXPECTED_V2_CORPUS_DIGEST,
        "compared_branch_count": len(pairs),
        "fit_branch_count": role_counts["fit"],
        "historical_calibration_branch_count": role_counts["calibration"],
        "mismatch_count": 0,
        "comparison": "exact canonical legacy-label projection",
        "pairs_digest": digest(pairs),
        "pairs": pairs,
    }
    return _with_self_digest(payload, EQUIVALENCE_SELF_KEY)


def validate_equivalence_receipt(
    receipt: Mapping[str, Any], authority: Mapping[str, Any],
    v2_rows: Sequence[Mapping[str, Any]],
) -> None:
    _validate_self_digest(receipt, EQUIVALENCE_SELF_KEY)
    if dict(receipt) != build_equivalence_receipt(authority, v2_rows):
        raise WorkflowError("1,422-row v1.3 equivalence receipt changed")


def issue_equivalence_receipt(*, out_root: Path = OUT_ROOT,
                              v2_root: Path = V2_ROOT) -> dict[str, Any]:
    authority = load_authority(out_root / "authority.json")
    corpus = load_v2_corpus(v2_root=v2_root)
    receipt = build_equivalence_receipt(authority, corpus["rows"])
    _atomic_json(out_root / "equivalence_receipt.json", receipt, out_root=out_root)
    return receipt


# --------------------------------------------------------------- tick trace --
def _to_float_list(value: Any, width: int, label: str) -> list[float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (1, width):
        array = array[0]
    if array.shape != (width,) or not np.all(np.isfinite(array)):
        raise WorkflowError(f"{label} must be {width} finite values")
    return [float(item) for item in array]


def _trace_sample(
    ctx: V1.BranchContext, label_computer: Any, pose_step_cls: Any,
    *, episode_id: int, episode_step: int, stamp_ns: int,
    requested_cmd: Sequence[float] | None, executed_cmd: Sequence[float],
    goal_cell: int, field: Any, topology: Mapping[str, Any],
    global_tick: int | None, block_index: int | None, tick_in_block: int | None,
) -> dict[str, Any]:
    oracle = _v13()
    runner = ctx.runner
    robot = ctx.build.robot
    position = _to_float_list(runner._as_np(robot.get_pos()), 3, "base position")
    quaternion = _to_float_list(
        runner._as_np(robot.get_quat()), 4, "base quaternion WXYZ"
    )
    (x, y), yaw, z = ctx.pose()
    if not np.allclose(position, [x, y, z], rtol=0.0, atol=1e-9):
        raise WorkflowError("BranchContext pose differs from raw robot position")
    label = label_computer.step(pose_step_cls(
        timestamp_ns=int(stamp_ns), env_idx=0, episode_id=int(episode_id),
        episode_step=int(episode_step), position_xy_world=(float(x), float(y)),
        yaw_world_rad=float(yaw),
        last_command=tuple(float(value) for value in executed_cmd),
    ))
    flags = {key: bool(value) for key, value in V1._termination_flags(ctx).items()}
    hit = ctx.scene_graph.locate((float(x), float(y)))
    locate_distance = float(hit.distance_m)
    cell = int(hit.cell_id)
    blocked = getattr(ctx.scene_graph, "nav_blocked_cells", frozenset())
    raw_bfs = ctx.scene_graph.bfs_distance(cell, int(goal_cell))
    masked_bfs = ctx.scene_graph.bfs_distance(
        cell, int(goal_cell), transit_blocked=blocked
    )
    located = locate_distance <= oracle.LOCATE_MAX_DISTANCE_M
    remaining = field.remaining_distance((float(x), float(y)), cell) if located else math.inf
    graph_status = oracle.classify_graph_status(
        locate_distance, remaining if math.isfinite(remaining) else None,
        pose_finite=not flags["nan"],
    )
    if located and ((masked_bfs is None) == math.isfinite(remaining)):
        raise WorkflowError("masked BFS and metric geodesic finiteness disagree")
    from lewm_genesis.rollout import _roll_from_quat_wxyz, _pitch_from_quat_wxyz
    qw, qx, qy, qz = quaternion
    roll = float(_roll_from_quat_wxyz(qw, qx, qy, qz))
    pitch = float(_pitch_from_quat_wxyz(qw, qx, qy, qz))
    clearance = float(label.clearance_m)
    contact_count = int(V12._contact_count(ctx, dict(topology)))
    terminated = bool(flags["fall"] or flags["out_of_bounds"] or flags["tipped"])
    return {
        "global_tick": global_tick,
        "block_index": block_index,
        "tick_in_block": tick_in_block,
        "episode_id": int(episode_id),
        "episode_step": int(episode_step),
        "timestamp_ns": int(stamp_ns),
        "requested_command": (
            None if requested_cmd is None else [float(value) for value in requested_cmd]
        ),
        "post_slew_command": [float(value) for value in executed_cmd],
        "position_world_xyz_m": position,
        "quaternion_world_wxyz": quaternion,
        "rpy_world_rad": [roll, pitch, float(yaw)],
        "xy": [float(x), float(y)],
        "yaw": float(yaw),
        "z": float(z),
        "nearest_cell_id": cell,
        "nearest_cell_distance_m": locate_distance,
        "located": bool(located),
        "accepted_cell_id": cell if located else None,
        "cell_id": cell,
        "goal_cell_id": int(goal_cell),
        "raw_bfs_to_goal": None if raw_bfs is None else int(raw_bfs),
        "masked_bfs_to_goal": None if masked_bfs is None else int(masked_bfs),
        "geodesic_m": float(remaining) if math.isfinite(remaining) else None,
        "graph_status": graph_status,
        # v1.3 preserves the v1.2 nearest-cell completion definition exactly;
        # locate/reachability status is deliberately not a conjunct here.
        "at_goal_cell": bool(cell == int(goal_cell)),
        "clearance_m": clearance,
        "clearance_deficit": float(
            max(0.0, min(1.0, (V12.CLEARANCE_SAFE_M - clearance)
                             / V12.CLEARANCE_SAFE_M))
        ),
        "stuck": bool(label.stuck_label),
        "disallowed_contacts": contact_count,
        "disallowed_contact": bool(contact_count > 0),
        "termination": flags,
        "terminated": terminated,
        "nan": flags["nan"],
    }


def execute_branch_trace_v13(
    ctx: V1.BranchContext, snapshot: Any,
    candidate: tuple[str, tuple[str, ...]], *, field: Any,
    topology: Mapping[str, Any],
    on_block_end: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Execute one branch and retain every 10 Hz label-evidence sample.

    Physical termination is recorded but does not truncate the evidence
    horizon.  A non-finite solver state is terminal and cannot be labelled.
    """

    from lewm_worlds.labels.derived import DerivedLabelComputer, DerivedLabelConfig, PoseStep
    oracle = _v13()
    V1.restore_branch_state(ctx, snapshot)
    runner = ctx.runner
    goal_cell = int(snapshot.goal["landmark_cell"])
    steps_per_tick = int(runner._policy_steps_per_command_tick)
    label_computer = DerivedLabelComputer(ctx.manifest, config=DerivedLabelConfig())
    episode_id = int(runner.episode_states[0].episode_id)
    state = {
        "episode_step": int(runner.episode_states[0].episode_step),
        "stamp_ns": int(runner._sim_time_ns),
    }
    start = _trace_sample(
        ctx, label_computer, PoseStep, episode_id=episode_id,
        episode_step=state["episode_step"], stamp_ns=state["stamp_ns"],
        requested_cmd=None,
        executed_cmd=np.asarray(runner._last_executed, dtype=np.float64)[0],
        goal_cell=goal_cell, field=field, topology=topology,
        global_tick=None, block_index=None, tick_in_block=None,
    )
    requested_all: list[Any] = []
    executed_all: list[Any] = []
    ticks: list[dict[str, Any]] = []
    nan_seen = False
    for block_index, primitive in enumerate(candidate[1]):
        requested = V1.block_for(primitive)[None, ...]
        executed = np.asarray(
            runner._clip_block(np.asarray(requested, dtype=np.float32)).executed,
            dtype=np.float64,
        )

        def after_policy_step(tick_index: int, step_index: int,
                              _requested=requested, _executed=executed,
                              _block=block_index) -> None:
            if step_index != steps_per_tick - 1:
                return
            state["episode_step"] += 1
            state["stamp_ns"] += int(runner._command_dt_ns)
            row = _trace_sample(
                ctx, label_computer, PoseStep, episode_id=episode_id,
                episode_step=state["episode_step"], stamp_ns=state["stamp_ns"],
                requested_cmd=_requested[0, tick_index],
                executed_cmd=_executed[0, tick_index], goal_cell=goal_cell,
                field=field, topology=topology,
                global_tick=_block * int(runner._block_size) + int(tick_index),
                block_index=_block, tick_in_block=int(tick_index),
            )
            ticks.append(row)

        block = runner.execute_requested_block(
            requested, after_policy_step=after_policy_step
        )
        requested_all.append(np.asarray(block.requested)[0].tolist())
        executed_all.append(np.asarray(block.executed)[0].tolist())
        ctx.ticks_executed += runner._block_size
        ctx.episode_ticks += runner._block_size
        ctx.policy_steps += runner._block_size * steps_per_tick
        ctx.last_block_executed = np.asarray(block.executed, dtype=np.float32).copy()
        if on_block_end is not None:
            on_block_end(block_index)
        if ticks and ticks[-1]["nan"]:
            nan_seen = True
            break
    completion = False
    for row in ticks:
        completion = completion or bool(row["at_goal_cell"])
        row["completion_latched"] = bool(completion)
    branch = {
        "schema": oracle.TRACE_SCHEMA,
        "candidate": candidate[0],
        "primitives": list(candidate[1]),
        "requested": requested_all,
        "post_slew": executed_all,
        "blocks_completed": len(executed_all),
        "nan": nan_seen,
        "start": start,
        "ticks": ticks,
    }
    if not nan_seen and (
        len(ticks) != oracle.HORIZON_TICKS
        or [row["global_tick"] for row in ticks]
        != list(range(oracle.HORIZON_TICKS))
    ):
        raise WorkflowError("v1.3 branch did not preserve exactly twenty ordered ticks")
    canonical_json(branch)
    return branch


def _pose_to_camera_pose(ctx: V1.BranchContext, pose: BasePose) -> dict[str, list[float]]:
    # This calls only the historical pure camera transform; it does not create
    # a renderer or render a new RGB frame.
    from lewm_genesis.render_replay import _camera_pose_from_payload
    value = _camera_pose_from_payload(pose.replay_payload(), validate_camera_pack(ctx.pack))
    return {
        key: [float(item) for item in value[key]]
        for key in ("position", "lookat", "up")
    }


def _camera_pose_sequence_error(
    old_poses: Sequence[Mapping[str, Any]],
    new_poses: Sequence[Mapping[str, Any]], *, expected_count: int,
    pose_atol: float,
) -> float:
    if len(old_poses) != expected_count or len(new_poses) != expected_count:
        raise WorkflowError(f"exact replay lacks {expected_count} old/new poses")
    deviations: list[float] = []
    for old, new in zip(old_poses, new_poses):
        for key in ("position", "lookat", "up"):
            old_values = np.asarray(old.get(key), dtype=np.float64)
            new_values = np.asarray(new.get(key), dtype=np.float64)
            if old_values.shape != (3,) or new_values.shape != (3,):
                raise WorkflowError("replay camera pose is malformed")
            deviation = float(np.max(np.abs(old_values - new_values)))
            deviations.append(deviation)
            if deviation > pose_atol:
                raise WorkflowError("exact replay camera pose exceeds frozen tolerance")
    return max(deviations, default=0.0)


def _normalised_prebranch_witness(
    *, proprio: Any, control: Any, action_context_blocks: Any,
    previous_applied_command: Any,
) -> dict[str, Any]:
    proprio_array = np.asarray(proprio, dtype=np.float32)
    control_array = np.asarray(control, dtype=np.float32)
    action_context_array = np.asarray(action_context_blocks, dtype=np.float64)
    previous_array = np.asarray(previous_applied_command, dtype=np.float64)
    if (
        proprio_array.shape != (B.PROPRIO_HISTORY, 30)
        or control_array.shape != (B.PROPRIO_HISTORY, 2)
        or action_context_array.shape != (B.CONTEXT_SLOTS, 10)
        or previous_array.shape != (3,)
        or not all(np.all(np.isfinite(value)) for value in (
            proprio_array, control_array, action_context_array, previous_array,
        ))
    ):
        raise WorkflowError("exact replay prebranch physical witness is malformed")
    return {
        "proprio": proprio_array.tolist(),
        "control": control_array.tolist(),
        "action_context_blocks": action_context_array.tolist(),
        "previous_applied_command": previous_array.tolist(),
    }


def validate_replay_preexecution(
    *, old_row: Mapping[str, Any], replay_snapshot_digest: str,
    context_camera_poses: Sequence[Mapping[str, Any]],
    proprio: Sequence[Sequence[float]], control: Sequence[Sequence[float]],
    action_context_blocks: Sequence[Sequence[float]],
    previous_applied_command: Sequence[float],
    pose_atol: float = HORIZON_POSE_ATOL,
    action_atol: float = ACTION_ATOL,
) -> dict[str, Any]:
    """Validate the persisted physical pre-branch witness before an attempt."""

    source_snapshot_digest = old_row.get("snapshot_digest")
    if not _sha256_string(source_snapshot_digest) or not _sha256_string(
            replay_snapshot_digest):
        raise WorkflowError("exact replay snapshot lineage is malformed")
    old_context_poses = [
        frame.get("camera_pose_world")
        for frame in old_row.get("context_frames", [])
    ]
    max_error = _camera_pose_sequence_error(
        old_context_poses, context_camera_poses,
        expected_count=3, pose_atol=pose_atol,
    )
    source_witness = _normalised_prebranch_witness(
        proprio=old_row.get("proprio"), control=old_row.get("control"),
        action_context_blocks=old_row.get("action_context_blocks"),
        previous_applied_command=old_row.get("previous_applied_command"),
    )
    replay_witness = _normalised_prebranch_witness(
        proprio=proprio, control=control,
        action_context_blocks=action_context_blocks,
        previous_applied_command=previous_applied_command,
    )
    old_proprio = np.asarray(source_witness["proprio"], dtype=np.float32)
    new_proprio = np.asarray(replay_witness["proprio"], dtype=np.float32)
    old_control = np.asarray(source_witness["control"], dtype=np.float32)
    new_control = np.asarray(replay_witness["control"], dtype=np.float32)
    old_action_context = np.asarray(
        source_witness["action_context_blocks"], dtype=np.float64
    )
    new_action_context = np.asarray(
        replay_witness["action_context_blocks"], dtype=np.float64
    )
    old_previous = np.asarray(
        source_witness["previous_applied_command"], dtype=np.float64
    )
    new_previous = np.asarray(
        replay_witness["previous_applied_command"], dtype=np.float64
    )
    if not np.array_equal(old_proprio, new_proprio):
        raise WorkflowError("exact replay proprio history changed")
    if not np.array_equal(old_control, new_control):
        raise WorkflowError("exact replay control history changed")
    action_context_error = float(np.max(np.abs(
        old_action_context - new_action_context
    )))
    previous_error = float(np.max(np.abs(old_previous - new_previous)))
    if action_context_error > action_atol:
        raise WorkflowError("exact replay action context changed")
    if previous_error > action_atol:
        raise WorkflowError("exact replay previous applied command changed")
    return {
        "source_snapshot_digest": source_snapshot_digest,
        "replay_snapshot_digest": replay_snapshot_digest,
        "source_snapshot_digest_preserved_as_lineage": True,
        "snapshot_digest_equality_required": False,
        "snapshot_digest_limitation": (
            "source snapshot bytes and original process-global RNG history were not "
            "persisted"
        ),
        "context_pose_atol": float(pose_atol),
        "max_context_camera_pose_abs_error": max_error,
        "three_context_poses_within_tolerance": True,
        "source_context_pose_digest": digest(old_context_poses),
        "replay_context_pose_digest": digest(list(context_camera_poses)),
        "replay_context_camera_poses": copy.deepcopy(list(context_camera_poses)),
        "source_prebranch_witness_digest": digest(source_witness),
        "replay_prebranch_witness_digest": digest(replay_witness),
        "replay_prebranch_witness": replay_witness,
        "proprio_history_exact": True,
        "control_history_exact": True,
        "prebranch_action_atol": float(action_atol),
        "max_action_context_abs_error": action_context_error,
        "action_context_within_tolerance": True,
        "max_previous_applied_command_abs_error": previous_error,
        "previous_applied_command_within_tolerance": True,
    }


def validate_replay_equality(
    *, old_row: Mapping[str, Any], branch: Mapping[str, Any],
    horizon_camera_poses: Sequence[Mapping[str, Any]],
    preexecution: Mapping[str, Any], pose_atol: float = HORIZON_POSE_ATOL,
    action_atol: float = ACTION_ATOL,
) -> dict[str, Any]:
    if (
        preexecution.get("source_snapshot_digest")
        != old_row.get("snapshot_digest")
        or preexecution.get("source_snapshot_digest_preserved_as_lineage") is not True
        or preexecution.get("snapshot_digest_equality_required") is not False
        or preexecution.get("three_context_poses_within_tolerance") is not True
        or preexecution.get("proprio_history_exact") is not True
        or preexecution.get("control_history_exact") is not True
        or preexecution.get("action_context_within_tolerance") is not True
        or preexecution.get("previous_applied_command_within_tolerance") is not True
    ):
        raise WorkflowError("exact replay preexecution witness changed")
    if branch.get("requested") != old_row.get("realised_requested_prefix"):
        raise WorkflowError("exact replay requested action prefix changed")
    if not np.allclose(
        np.asarray(branch.get("post_slew"), dtype=np.float64),
        np.asarray(old_row.get("post_slew"), dtype=np.float64),
        rtol=0.0, atol=action_atol,
    ):
        raise WorkflowError("exact replay post-slew actions changed")
    old_poses = [
        frame.get("camera_pose_world") for frame in old_row.get("horizon_frames", [])
    ]
    max_error = _camera_pose_sequence_error(
        old_poses, horizon_camera_poses,
        expected_count=4, pose_atol=pose_atol,
    )
    return {
        **copy.deepcopy(dict(preexecution)),
        "requested_actions_exact": True,
        "post_slew_action_atol": float(action_atol),
        "post_slew_actions_within_tolerance": True,
        "horizon_pose_atol": float(pose_atol),
        "max_horizon_camera_pose_abs_error": max_error,
        "four_horizon_poses_within_tolerance": True,
        "source_horizon_pose_digest": digest(old_poses),
        "replay_horizon_pose_digest": digest(list(horizon_camera_poses)),
        "replay_horizon_camera_poses":
            copy.deepcopy(list(horizon_camera_poses)),
    }


def _attempt_path(kind: str, identity: str, *, out_root: Path) -> Path:
    if kind not in {"replay", "fresh_calibration"} or not _sha256_string(identity):
        raise WorkflowError("attempt marker kind/identity is malformed")
    if kind == "replay":
        return out_root / "replay_attempts" / f"{identity}.json"
    return out_root / "fresh_calibration/attempts" / f"{identity}.json"


def begin_attempt_once(
    kind: str, identity: str, bindings: Mapping[str, Any], *, out_root: Path = OUT_ROOT
) -> dict[str, Any]:
    path = _attempt_path(kind, identity, out_root=out_root)
    if path.exists():
        raise WorkflowError(
            f"branch {identity} already has an attempt marker; retry/replacement refused"
        )
    marker = _with_self_digest({
        "schema": "go2_scorer_fit_oracle_v1_3_branch_attempt_v1",
        "status": "ATTEMPT_STARTED_NO_RETRY_AUTHORITY",
        "kind": kind,
        "identity_digest": identity,
        "bindings": copy.deepcopy(dict(bindings)),
    }, "attempt_digest")
    _atomic_json(path, marker, out_root=out_root, exact_idempotence=False)
    return marker


def _replay_overlay_path(identity: str, *, out_root: Path = OUT_ROOT) -> Path:
    if not _sha256_string(identity):
        raise WorkflowError("replay overlay identity is malformed")
    return out_root / "replay_overlays" / f"{identity}.json"


def _make_replay_overlay(
    *, authority: Mapping[str, Any], plan: Mapping[str, Any],
    old_row: Mapping[str, Any], branch: Mapping[str, Any], score: Mapping[str, Any],
    replay_equality: Mapping[str, Any], attempt: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema": REPLAY_OVERLAY_SCHEMA,
        "status": STATUS,
        "authority_digest": authority[AUTHORITY_SELF_KEY],
        "replay_plan_digest": plan[REPLAY_PLAN_SELF_KEY],
        "oracle_v1_3_digest": authority["oracle_v1_3_digest"],
        "scorer_fit_oracle_v1_3_contract_digest": authority[
            "scorer_fit_oracle_v1_3_contract_digest"
        ],
        "source_v2_corpus_digest": EXPECTED_V2_CORPUS_DIGEST,
        "source_branch_identity_digest": old_row["branch_identity_digest"],
        "source_branch_row_digest": old_row["branch_row_digest"],
        "state_id": old_row["state_id"],
        "state_identity_digest": old_row["state_identity_digest"],
        "assignment_identity_digest": old_row["assignment_identity_digest"],
        "scene_id": old_row["scene_id"],
        "split_role": old_row["split_role"],
        "family": old_row["family"],
        "stratum": old_row["stratum"],
        "candidate_index": old_row["candidate_index"],
        "candidate": old_row["candidate"],
        "goal": old_row["goal"],
        "attempt_digest": attempt["attempt_digest"],
        "replay_equality": copy.deepcopy(dict(replay_equality)),
        "trace": copy.deepcopy(dict(branch)),
        "labels": _label_projection(score),
        "v1_3_graph_evidence": {
            key: score.get(key) for key in (
                "final_graph_status", "first_graph_boundary_tick",
                "last_finite_geodesic_m", "last_finite_geodesic_tick",
            ) if key in score
        },
        "v2_row_modified": False,
        "v2_rgb_rerendered": False,
    }
    return _with_self_digest(payload, REPLAY_OVERLAY_SELF_KEY)


def generate_replay_overlays(*, backend: str = "cpu",
                             out_root: Path = OUT_ROOT) -> dict[str, Any]:
    """Execute only the eighteen preregistered V2 identities."""

    if backend != "cpu":
        raise WorkflowError("v1.3 branch runtime is frozen to CPU")
    authority = load_authority(out_root / "authority.json")
    require_genesis_runtime()
    plan = load_replay_plan(out_root=out_root)
    corpus = load_v2_corpus()
    equivalence = _read_regular_json(out_root / "equivalence_receipt.json")
    validate_equivalence_receipt(equivalence, authority, corpus["rows"])
    # The corpus loader has already verified the immutable raw state manifest.
    # Those raw rows are also the exact projection consumed by the corrected
    # V2 redrive check.  Reopening the historical *live* V2 runtime authority
    # here would incorrectly require its old source commit to equal this
    # committed v1.3 successor.
    state_by_id = {
        row["state_id"]: row for row in corpus["state_manifest"]["states"]
    }
    redrive_by_state = state_by_id
    old_by_identity = {
        row["branch_identity_digest"]: row for row in corpus["rows"]
    }
    entries_by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in plan["entries"]:
        entries_by_state[entry["state_id"]].append(entry)
    shared = V1._load_shared(backend)
    overlays: list[dict[str, Any]] = []
    for state_id in sorted(entries_by_state):
        state = state_by_id[state_id]
        scene_dir = Path(state["scene_dir"])
        manifest_path = scene_dir / "manifest.json"
        if (
            B.file_sha256(manifest_path) != state["scene_manifest_sha256"]
            or manifest_path.stat().st_size != state["scene_manifest_byte_count"]
        ):
            raise WorkflowError("replay scene manifest changed")
        ctx = V1.build_context(
            scene_dir, seed=int(state["drive_seed"]), backend=backend, shared=shared
        )
        topology = V12.link_topology(ctx)
        ctx.begin_episode()
        proprio_log: list[list[float]] = []
        control_log: list[list[float]] = []
        action_context_blocks: list[list[float]] = []
        context_poses: list[BasePose] = []
        warmup_blocks = int(state["warmup_blocks"])

        def probe(_tick_idx: int, previous_applied: Sequence[float]) -> None:
            proprio_log.append(B.proprio_sample(ctx))
            control_log.append(B.control_sample(previous_applied))

        for block_index in range(warmup_blocks):
            driven = B.drive_block_with_probe(ctx, probe)
            if block_index >= warmup_blocks - B.CONTEXT_SLOTS:
                action_context_blocks.append(B.action_block_10d(
                    np.asarray(driven.executed, dtype=np.float64)[0]
                ))
                context_poses.append(capture_base_pose(ctx))
        verdict = B.classify_state(ctx, topology, requested_stratum=state["stratum"])
        if isinstance(verdict, str):
            raise WorkflowError(f"exact replay state redrive failed: {verdict}")
        record, field, _strata = verdict
        mismatch = B._redrive_mismatch(
            redrive_by_state[state_id], record, ctx, full_bank_v2=True
        )
        if mismatch is not None:
            raise WorkflowError(f"exact replay state mismatch: {mismatch}")
        snapshot = V1.capture_branch_state(
            ctx, goal=state["goal"], identity={
                "state_id": state_id,
                "state_identity_digest": state["state_identity_digest"],
                "scene_id": state["scene_id"], "family": state["family"],
                "split": state["split"], "block_index": state["warmup_blocks"],
                "source_step": state["source_step"], "episode_id": state["episode_id"],
            },
        )
        context_camera_poses = [
            _pose_to_camera_pose(ctx, pose) for pose in context_poses
        ]
        proprio = np.asarray(
            proprio_log[-B.PROPRIO_HISTORY:], dtype=np.float32
        )
        control = np.asarray(
            control_log[-B.PROPRIO_HISTORY:], dtype=np.float32
        )
        previous_applied = np.asarray(
            ctx.runner._last_executed, dtype=np.float64
        )[0].tolist()
        preexecution_by_identity = {
            entry["branch_identity_digest"]: validate_replay_preexecution(
                old_row=old_by_identity[entry["branch_identity_digest"]],
                replay_snapshot_digest=snapshot.digest,
                context_camera_poses=context_camera_poses,
                proprio=proprio.tolist(),
                control=control.tolist(),
                action_context_blocks=action_context_blocks,
                previous_applied_command=previous_applied,
            )
            for entry in entries_by_state[state_id]
        }
        for entry in sorted(entries_by_state[state_id], key=lambda row: row["candidate_index"]):
            identity = entry["branch_identity_digest"]
            output_path = _replay_overlay_path(identity, out_root=out_root)
            if output_path.exists():
                overlay = _read_regular_json(output_path)
                _validate_replay_overlay_binding(
                    overlay, authority=authority, plan=plan,
                    planned=entry, out_root=out_root,
                )
                overlays.append(overlay)
                continue
            attempt = begin_attempt_once(
                "replay", identity,
                {"authority_digest": authority[AUTHORITY_SELF_KEY],
                 "replay_plan_digest": plan[REPLAY_PLAN_SELF_KEY]},
                out_root=out_root,
            )
            horizon_poses: list[BasePose] = []
            branch = execute_branch_trace_v13(
                ctx, snapshot, V1.CANDIDATE_BANK[int(entry["candidate_index"])],
                field=field, topology=topology,
                on_block_end=lambda _index: horizon_poses.append(capture_base_pose(ctx)),
            )
            old_row = old_by_identity[identity]
            equality = validate_replay_equality(
                old_row=old_row, branch=branch,
                horizon_camera_poses=[_pose_to_camera_pose(ctx, pose)
                                      for pose in horizon_poses],
                preexecution=preexecution_by_identity[identity],
            )
            score = _v13().score_branch_v13(branch)
            if score is None:
                raise WorkflowError("exact replay produced no v1.3 label; retry forbidden")
            overlay = _make_replay_overlay(
                authority=authority, plan=plan, old_row=old_row, branch=branch,
                score=score, replay_equality=equality, attempt=attempt,
            )
            _atomic_json(output_path, overlay, out_root=out_root)
            overlays.append(overlay)
        del ctx
        gc.collect()
    return compile_replay_overlay_manifest(
        authority=authority, plan=plan, overlays=overlays, out_root=out_root
    )


def _validate_replay_overlay_binding(
    overlay: Mapping[str, Any], *, authority: Mapping[str, Any],
    plan: Mapping[str, Any], planned: Mapping[str, Any], out_root: Path,
) -> None:
    _validate_self_digest(overlay, REPLAY_OVERLAY_SELF_KEY)
    identity = str(overlay.get("source_branch_identity_digest", ""))
    score = _v13().score_branch_v13(overlay.get("trace", {}))
    equality = overlay.get("replay_equality", {})
    trace = overlay.get("trace", {})
    source_context_poses = planned.get("source_context_camera_poses", [])
    source_witness = planned.get("source_prebranch_witness", {})
    replay_context_poses = equality.get("replay_context_camera_poses", [])
    replay_witness = equality.get("replay_prebranch_witness", {})
    replay_horizon_poses = equality.get("replay_horizon_camera_poses", [])
    try:
        source_normalised = _normalised_prebranch_witness(**source_witness)
        replay_normalised = _normalised_prebranch_witness(**replay_witness)
        context_error = _camera_pose_sequence_error(
            source_context_poses, replay_context_poses,
            expected_count=3, pose_atol=HORIZON_POSE_ATOL,
        )
        horizon_error = _camera_pose_sequence_error(
            planned.get("horizon_camera_poses", []), replay_horizon_poses,
            expected_count=4, pose_atol=HORIZON_POSE_ATOL,
        )
        source_proprio = np.asarray(source_normalised["proprio"], dtype=np.float32)
        replay_proprio = np.asarray(replay_normalised["proprio"], dtype=np.float32)
        source_control = np.asarray(source_normalised["control"], dtype=np.float32)
        replay_control = np.asarray(replay_normalised["control"], dtype=np.float32)
        source_action_context = np.asarray(
            source_normalised["action_context_blocks"], dtype=np.float64
        )
        replay_action_context = np.asarray(
            replay_normalised["action_context_blocks"], dtype=np.float64
        )
        source_previous = np.asarray(
            source_normalised["previous_applied_command"], dtype=np.float64
        )
        replay_previous = np.asarray(
            replay_normalised["previous_applied_command"], dtype=np.float64
        )
        action_context_error = float(np.max(np.abs(
            source_action_context - replay_action_context
        )))
        previous_error = float(np.max(np.abs(source_previous - replay_previous)))
        physical_witness_matches = bool(
            np.array_equal(source_proprio, replay_proprio)
            and np.array_equal(source_control, replay_control)
            and action_context_error <= ACTION_ATOL
            and previous_error <= ACTION_ATOL
        )
        post_slew_matches_plan = bool(np.allclose(
            np.asarray(trace.get("post_slew"), dtype=np.float64),
            np.asarray(planned.get("post_slew"), dtype=np.float64),
            rtol=0.0, atol=ACTION_ATOL,
        ))
    except (TypeError, ValueError, WorkflowError):
        context_error = math.inf
        horizon_error = math.inf
        action_context_error = math.inf
        previous_error = math.inf
        physical_witness_matches = False
        post_slew_matches_plan = False

    def finite_at_most(value: Any, limit: float) -> bool:
        return bool(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            and 0.0 <= float(value) <= limit
        )

    marker_path = _attempt_path("replay", identity, out_root=out_root)
    marker = _read_regular_json(marker_path)
    _validate_self_digest(marker, "attempt_digest")
    if (
        overlay.get("schema") != REPLAY_OVERLAY_SCHEMA
        or overlay.get("authority_digest") != authority[AUTHORITY_SELF_KEY]
        or overlay.get("replay_plan_digest") != plan[REPLAY_PLAN_SELF_KEY]
        or overlay.get("oracle_v1_3_digest") != authority["oracle_v1_3_digest"]
        or overlay.get("scorer_fit_oracle_v1_3_contract_digest")
        != authority["scorer_fit_oracle_v1_3_contract_digest"]
        or identity != planned.get("branch_identity_digest")
        or overlay.get("source_branch_row_digest") != planned.get("branch_row_digest")
        or overlay.get("state_id") != planned.get("state_id")
        or overlay.get("state_identity_digest") != planned.get("state_identity_digest")
        or overlay.get("assignment_identity_digest")
        != planned.get("assignment_identity_digest")
        or overlay.get("scene_id") != planned.get("scene_id")
        or overlay.get("split_role") != planned.get("split_role")
        or overlay.get("candidate_index") != planned.get("candidate_index")
        or overlay.get("candidate") != planned.get("candidate")
        or overlay.get("goal") != planned.get("goal")
        or trace.get("candidate") != planned.get("candidate")
        or trace.get("primitives") != planned.get("primitives")
        or marker.get("kind") != "replay"
        or marker.get("identity_digest") != identity
        or overlay.get("attempt_digest") != marker.get("attempt_digest")
        or score is None
        or _label_projection(score) != _label_projection(overlay.get("labels", {}))
        or equality.get("source_snapshot_digest") != planned.get("snapshot_digest")
        or not _sha256_string(equality.get("replay_snapshot_digest"))
        or equality.get("source_snapshot_digest_preserved_as_lineage") is not True
        or equality.get("snapshot_digest_equality_required") is not False
        or equality.get("source_context_pose_digest")
        != planned.get("source_context_pose_digest")
        or planned.get("source_context_pose_digest")
        != digest(source_context_poses)
        or equality.get("replay_context_pose_digest")
        != digest(replay_context_poses)
        or equality.get("source_prebranch_witness_digest")
        != planned.get("source_prebranch_witness_digest")
        or planned.get("source_prebranch_witness_digest")
        != digest(source_witness)
        or equality.get("replay_prebranch_witness_digest")
        != digest(replay_witness)
        or not physical_witness_matches
        or equality.get("three_context_poses_within_tolerance") is not True
        or equality.get("proprio_history_exact") is not True
        or equality.get("control_history_exact") is not True
        or equality.get("action_context_within_tolerance") is not True
        or equality.get("previous_applied_command_within_tolerance") is not True
        or equality.get("context_pose_atol") != HORIZON_POSE_ATOL
        or not finite_at_most(
            equality.get("max_context_camera_pose_abs_error"), HORIZON_POSE_ATOL
        )
        or equality.get("max_context_camera_pose_abs_error") != context_error
        or equality.get("prebranch_action_atol") != ACTION_ATOL
        or not finite_at_most(
            equality.get("max_action_context_abs_error"), ACTION_ATOL
        )
        or equality.get("max_action_context_abs_error") != action_context_error
        or not finite_at_most(
            equality.get("max_previous_applied_command_abs_error"), ACTION_ATOL
        )
        or equality.get("max_previous_applied_command_abs_error") != previous_error
        or equality.get("requested_actions_exact") is not True
        or equality.get("post_slew_actions_within_tolerance") is not True
        or trace.get("requested") != planned.get("requested")
        or not post_slew_matches_plan
        or equality.get("post_slew_action_atol") != ACTION_ATOL
        or equality.get("four_horizon_poses_within_tolerance") is not True
        or equality.get("source_horizon_pose_digest")
        != planned.get("source_horizon_pose_digest")
        or planned.get("source_horizon_pose_digest")
        != digest(planned.get("horizon_camera_poses", []))
        or equality.get("replay_horizon_pose_digest")
        != digest(replay_horizon_poses)
        or equality.get("horizon_pose_atol") != HORIZON_POSE_ATOL
        or not finite_at_most(
            equality.get("max_horizon_camera_pose_abs_error"), HORIZON_POSE_ATOL
        )
        or equality.get("max_horizon_camera_pose_abs_error") != horizon_error
        or overlay.get("v2_row_modified") is not False
        or overlay.get("v2_rgb_rerendered") is not False
    ):
        raise WorkflowError("replay overlay scientific binding changed")


def compile_replay_overlay_manifest(
    *, authority: Mapping[str, Any], plan: Mapping[str, Any],
    overlays: Sequence[Mapping[str, Any]], out_root: Path = OUT_ROOT,
) -> dict[str, Any]:
    plan_by_identity = {
        row["branch_identity_digest"]: row for row in plan["entries"]
    }
    by_identity: dict[str, Mapping[str, Any]] = {}
    for overlay in overlays:
        identity = str(overlay.get("source_branch_identity_digest", ""))
        if identity in by_identity:
            raise WorkflowError("duplicate replay overlay identity")
        planned = plan_by_identity.get(identity)
        if planned is None:
            raise WorkflowError("replay overlay is absent from the exact plan")
        _validate_replay_overlay_binding(
            overlay, authority=authority, plan=plan, planned=planned,
            out_root=out_root,
        )
        by_identity[identity] = overlay
    expected = {entry["branch_identity_digest"] for entry in plan["entries"]}
    if set(by_identity) != expected or len(by_identity) != EXPECTED_V2_INVALID:
        raise WorkflowError("replay overlay set is not the exact eighteen")
    rows = [{
        "source_branch_identity_digest": identity,
        "replay_overlay_digest": by_identity[identity][REPLAY_OVERLAY_SELF_KEY],
        "path": _repo_relative(_replay_overlay_path(identity, out_root=out_root)),
        "sha256": file_sha256(_replay_overlay_path(identity, out_root=out_root)),
    } for identity in sorted(by_identity)]
    payload = {
        "schema": REPLAY_OVERLAY_MANIFEST_SCHEMA,
        "status": STATUS,
        "authority_digest": authority[AUTHORITY_SELF_KEY],
        "replay_plan_digest": plan[REPLAY_PLAN_SELF_KEY],
        "oracle_v1_3_digest": authority["oracle_v1_3_digest"],
        "scorer_fit_oracle_v1_3_contract_digest": authority[
            "scorer_fit_oracle_v1_3_contract_digest"
        ],
        "overlay_count": len(rows),
        "fit_overlay_count": sum(
            overlay["split_role"] == "fit" for overlay in by_identity.values()),
        "historical_calibration_overlay_count": sum(
            overlay["split_role"] == "calibration" for overlay in by_identity.values()),
        "rows": rows,
    }
    if (payload["fit_overlay_count"], payload["historical_calibration_overlay_count"]) \
            != (EXPECTED_OLD_FIT_OVERLAYS, 12):
        raise WorkflowError("replay overlay fit/calibration distribution changed")
    manifest = _with_self_digest(payload, REPLAY_OVERLAY_MANIFEST_SELF_KEY)
    _atomic_json(out_root / "replay_overlay_manifest.json", manifest, out_root=out_root)
    return manifest


# ------------------------------------------------------ fresh state selector --
def build_fresh_calibration_manifest(
    *, authority: Mapping[str, Any], v2_state_manifest: Mapping[str, Any],
    states: Sequence[Mapping[str, Any]], exclusion_binding: Mapping[str, Any],
    selector_integrity_replacement_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if (
        authority.get(AUTHORITY_SELF_KEY)
        == V13_CONTRACT.SELECTOR_INTEGRITY_REPLACEMENT[
            "predecessor_authority_digest"
        ]
    ):
        _validate_self_digest(authority, AUTHORITY_SELF_KEY)
    else:
        validate_authority(authority)
    old_states = v2_state_manifest.get("states", [])
    old_scene_ids = {str(row.get("scene_id")) for row in old_states}
    if len(old_scene_ids) != EXPECTED_OLD_STATE_COUNT:
        raise WorkflowError("old scorer-fit scene set changed")
    ordered = sorted(
        (copy.deepcopy(dict(row)) for row in states),
        key=lambda row: (FAMILIES.index(row["family"]),
                         STRATA.index(row["stratum"]), row["scene_id"]),
    )
    expected_slots = {(family, stratum) for family in FAMILIES for stratum in STRATA}
    actual_slots = {(row.get("family"), row.get("stratum")) for row in ordered}
    scene_ids = [str(row.get("scene_id")) for row in ordered]
    if (
        len(ordered) != EXPECTED_FRESH_CALIBRATION_STATES
        or actual_slots != expected_slots
        or len(set(scene_ids)) != len(scene_ids)
        or old_scene_ids.intersection(scene_ids)
    ):
        raise WorkflowError("fresh calibration is not an exact disjoint 8x3 panel")
    normalised = []
    for index, state in enumerate(ordered):
        state = dict(state)
        state["state_index"] = index
        state["split_role"] = "calibration"
        state["candidate_indices"] = list(EXPECTED_CANDIDATES)
        scientific = dict(state)
        scientific.pop("state_identity_digest", None)
        scientific["authority_digest"] = authority[AUTHORITY_SELF_KEY]
        state["state_identity_digest"] = digest(scientific)
        normalised.append(state)
    if len({row["state_identity_digest"] for row in normalised}) != len(normalised):
        raise WorkflowError("fresh calibration state identities are not unique")
    payload = {
        "schema": FRESH_STATE_MANIFEST_SCHEMA,
        "status": STATUS,
        "complete": True,
        "authority_digest": authority[AUTHORITY_SELF_KEY],
        "selector_integrity_replacement_authority_digest": (
            selector_integrity_replacement_authority.get(
                V13_CONTRACT.SELECTOR_INTEGRITY_REPLACEMENT_SELF_KEY
            )
            if selector_integrity_replacement_authority is not None else None
        ),
        "oracle_v1_3_digest": authority["oracle_v1_3_digest"],
        "scorer_fit_oracle_v1_3_contract_digest": authority[
            "scorer_fit_oracle_v1_3_contract_digest"
        ],
        "source_v2_state_manifest_digest": v2_state_manifest["state_manifest_digest"],
        "selection_contract": authority["fresh_calibration_selection"],
        "exclusion_binding": copy.deepcopy(dict(exclusion_binding)),
        "old_scorer_fit_scene_ids_digest": digest(sorted(old_scene_ids)),
        "state_count": len(normalised),
        "candidate_count": EXPECTED_FRESH_CALIBRATION_BRANCHES,
        "states": normalised,
        "candidate_outcomes_consumed": False,
        "all_identities_frozen_before_branch_execution": True,
    }
    return _with_self_digest(payload, FRESH_STATE_MANIFEST_SELF_KEY)


def validate_fresh_calibration_manifest(
    manifest: Mapping[str, Any], *, authority: Mapping[str, Any],
    v2_state_manifest: Mapping[str, Any],
    selector_integrity_replacement_authority: Mapping[str, Any] | None = None,
) -> None:
    _validate_self_digest(manifest, FRESH_STATE_MANIFEST_SELF_KEY)
    rebuilt = build_fresh_calibration_manifest(
        authority=authority, v2_state_manifest=v2_state_manifest,
        states=manifest.get("states", []),
        exclusion_binding=manifest.get("exclusion_binding", {}),
        selector_integrity_replacement_authority=
            selector_integrity_replacement_authority,
    )
    if dict(manifest) != rebuilt:
        raise WorkflowError("fresh calibration state manifest changed")


def build_fresh_selection_attempt(
    *, authority: Mapping[str, Any], correction: Mapping[str, Any],
    v2_state_manifest: Mapping[str, Any], plan: Mapping[str, Any],
    equivalence: Mapping[str, Any], overlay_manifest: Mapping[str, Any],
    pool: Mapping[str, Sequence[Path]], exclusion: Mapping[str, Any],
) -> dict[str, Any]:
    old_scene_ids = sorted(
        str(row["scene_id"]) for row in v2_state_manifest.get("states", [])
    )
    inventory = {
        family: [path.name for path in sorted(pool[family], key=lambda p: p.name)]
        for family in FAMILIES
    }
    scene_bindings = {
        family: {
            path.name: {
                "scene_dir": str(path.resolve()),
                "scene_manifest_sha256": B.file_sha256(path / "manifest.json"),
                "scene_manifest_byte_count": (path / "manifest.json").stat().st_size,
            }
            for path in sorted(pool[family], key=lambda p: p.name)
        }
        for family in FAMILIES
    }
    payload = {
        "schema": FRESH_SELECTION_ATTEMPT_SCHEMA,
        "status": "ONE_SHOT_SELECTOR_INTEGRITY_REPLACEMENT_STARTED",
        "complete": False,
        "authority_digest": authority[AUTHORITY_SELF_KEY],
        "selector_integrity_replacement_authority_digest": correction[
            V13_CONTRACT.SELECTOR_INTEGRITY_REPLACEMENT_SELF_KEY
        ],
        "selector_integrity_replacement_digest":
            V13_CONTRACT.selector_integrity_replacement_digest(),
        "selector_digest": V13_CONTRACT.fresh_calibration_selector_digest(),
        "v2_state_manifest_digest": v2_state_manifest["state_manifest_digest"],
        "replay_plan_digest": plan[REPLAY_PLAN_SELF_KEY],
        "equivalence_receipt_digest": equivalence[EQUIVALENCE_SELF_KEY],
        "replay_overlay_manifest_digest": overlay_manifest[
            REPLAY_OVERLAY_MANIFEST_SELF_KEY
        ],
        "families": list(FAMILIES),
        "strata": list(STRATA),
        "scene_inventory": inventory,
        "scene_inventory_digest": digest(inventory),
        "scene_bindings": scene_bindings,
        "scene_bindings_digest": digest(scene_bindings),
        "old_scene_ids": old_scene_ids,
        "old_scene_ids_digest": digest(old_scene_ids),
        "manifest_exclusion_binding": {
            **copy.deepcopy(dict(exclusion)),
            "old_scorer_fit_scene_ids": old_scene_ids,
            "old_scorer_fit_scene_ids_digest": digest(old_scene_ids),
            "lexical_selection": True,
            "replay_plan_digest": plan[REPLAY_PLAN_SELF_KEY],
        },
        "warmup_blocks_inclusive": [
            FRESH_WARMUP_BLOCKS_MIN, FRESH_WARMUP_BLOCKS_MAX,
        ],
        "backend": "cpu",
        "per_scene_process_isolation": True,
        "candidate_outcomes_consumed": False,
    }
    return _with_self_digest(payload, FRESH_SELECTION_ATTEMPT_SELF_KEY)


def validate_fresh_selection_attempt(
    attempt: Mapping[str, Any], *, authority: Mapping[str, Any],
    correction: Mapping[str, Any], v2_state_manifest: Mapping[str, Any],
    plan: Mapping[str, Any], equivalence: Mapping[str, Any],
    overlay_manifest: Mapping[str, Any],
    pool: Mapping[str, Sequence[Path]], exclusion: Mapping[str, Any],
) -> None:
    _validate_self_digest(attempt, FRESH_SELECTION_ATTEMPT_SELF_KEY)
    expected = build_fresh_selection_attempt(
        authority=authority, correction=correction,
        v2_state_manifest=v2_state_manifest, plan=plan,
        equivalence=equivalence, overlay_manifest=overlay_manifest,
        pool=pool, exclusion=exclusion,
    )
    if dict(attempt) != expected:
        raise WorkflowError("fresh-selection one-shot attempt changed")


def build_fresh_selection_task(
    *, attempt: Mapping[str, Any], correction: Mapping[str, Any],
    family: str, stratum: str, slot_index: int, scene_ordinal: int,
    scene_dir: Path, used_scene_ids: Sequence[str],
) -> dict[str, Any]:
    manifest_path = scene_dir / "manifest.json"
    if (
        family not in FAMILIES or stratum not in STRATA
        or not manifest_path.is_file() or manifest_path.is_symlink()
    ):
        raise WorkflowError("fresh-selection task input is malformed")
    payload = {
        "schema": FRESH_SELECTION_TASK_SCHEMA,
        "status": "DURABLE_BEFORE_ISOLATED_SCENE_PROCESS",
        "attempt_digest": attempt[FRESH_SELECTION_ATTEMPT_SELF_KEY],
        "selector_integrity_replacement_authority_digest": correction[
            V13_CONTRACT.SELECTOR_INTEGRITY_REPLACEMENT_SELF_KEY
        ],
        "selector_digest": V13_CONTRACT.fresh_calibration_selector_digest(),
        "backend": "cpu",
        "slot_index": int(slot_index),
        "scene_ordinal": int(scene_ordinal),
        "family": family,
        "stratum": stratum,
        "state_id": f"oracle_v1_3-calibration-{family}-{stratum}",
        "scene_id": scene_dir.name,
        "scene_dir": str(scene_dir.resolve()),
        "scene_manifest_sha256": B.file_sha256(manifest_path),
        "scene_manifest_byte_count": manifest_path.stat().st_size,
        "drive_seed": int(V1._drive_seed(scene_dir.name)),
        "used_scene_ids_before": sorted(str(value) for value in used_scene_ids),
        "used_scene_ids_before_digest": digest(sorted(
            str(value) for value in used_scene_ids
        )),
        "warmup_blocks_inclusive": [
            FRESH_WARMUP_BLOCKS_MIN, FRESH_WARMUP_BLOCKS_MAX,
        ],
        "candidate_outcomes_consumed": False,
    }
    return _with_self_digest(payload, FRESH_SELECTION_TASK_SELF_KEY)


def validate_fresh_selection_task(
    task: Mapping[str, Any], *, attempt: Mapping[str, Any],
    correction: Mapping[str, Any],
) -> None:
    _validate_self_digest(task, FRESH_SELECTION_TASK_SELF_KEY)
    family = task.get("family")
    stratum = task.get("stratum")
    scene_id = task.get("scene_id")
    slot_index = task.get("slot_index")
    scene_ordinal = task.get("scene_ordinal")
    used_scene_ids = task.get("used_scene_ids_before")
    scene_dir = Path(str(task.get("scene_dir", "")))
    manifest_path = scene_dir / "manifest.json"
    inventory = attempt.get("scene_inventory", {})
    scene_binding = attempt.get("scene_bindings", {}).get(family, {}).get(
        scene_id, {}
    )
    if (
        family not in FAMILIES
        or stratum not in STRATA
        or not isinstance(scene_id, str)
        or isinstance(slot_index, bool) or not isinstance(slot_index, int)
        or isinstance(scene_ordinal, bool) or not isinstance(scene_ordinal, int)
        or slot_index < 0 or scene_ordinal < 0
        or not isinstance(used_scene_ids, list)
        or any(not isinstance(value, str) for value in used_scene_ids)
        or used_scene_ids != sorted(set(used_scene_ids))
        or scene_id not in inventory.get(family, [])
        or scene_dir.name != scene_id
        or task.get("scene_dir") != scene_binding.get("scene_dir")
        or task.get("scene_manifest_sha256")
        != scene_binding.get("scene_manifest_sha256")
        or task.get("scene_manifest_byte_count")
        != scene_binding.get("scene_manifest_byte_count")
        or not manifest_path.is_file()
        or manifest_path.is_symlink()
        or manifest_path.stat().st_size != task.get("scene_manifest_byte_count")
        or B.file_sha256(manifest_path) != task.get("scene_manifest_sha256")
    ):
        raise WorkflowError("fresh-selection scene task changed")
    expected = build_fresh_selection_task(
        attempt=attempt, correction=correction, family=family,
        stratum=stratum, slot_index=slot_index, scene_ordinal=scene_ordinal,
        scene_dir=scene_dir, used_scene_ids=used_scene_ids,
    )
    if dict(task) != expected:
        raise WorkflowError("fresh-selection scene task changed")


def build_fresh_selection_result(
    *, task: Mapping[str, Any], selected_state: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload = {
        "schema": FRESH_SELECTION_RESULT_SCHEMA,
        "status": "SCENE_EVALUATION_PUBLISHED_BEFORE_PROCESS_TEARDOWN",
        "task_digest": task[FRESH_SELECTION_TASK_SELF_KEY],
        "attempt_digest": task["attempt_digest"],
        "selector_integrity_replacement_authority_digest": task[
            "selector_integrity_replacement_authority_digest"
        ],
        "family": task["family"],
        "stratum": task["stratum"],
        "scene_id": task["scene_id"],
        "eligible": selected_state is not None,
        "selected_state": (
            copy.deepcopy(dict(selected_state))
            if selected_state is not None else None
        ),
        "candidate_outcomes_consumed": False,
    }
    return _with_self_digest(payload, FRESH_SELECTION_RESULT_SELF_KEY)


def validate_fresh_selection_result(
    result: Mapping[str, Any], *, task: Mapping[str, Any],
) -> None:
    _validate_self_digest(result, FRESH_SELECTION_RESULT_SELF_KEY)
    selected = result.get("selected_state")
    eligible = result.get("eligible")
    if (
        result.get("schema") != FRESH_SELECTION_RESULT_SCHEMA
        or result.get("status")
        != "SCENE_EVALUATION_PUBLISHED_BEFORE_PROCESS_TEARDOWN"
        or result.get("task_digest") != task.get(FRESH_SELECTION_TASK_SELF_KEY)
        or result.get("attempt_digest") != task.get("attempt_digest")
        or result.get("selector_integrity_replacement_authority_digest")
        != task.get("selector_integrity_replacement_authority_digest")
        or any(result.get(key) != task.get(key) for key in (
            "family", "stratum", "scene_id",
        ))
        or not isinstance(eligible, bool)
        or (eligible and not isinstance(selected, Mapping))
        or (not eligible and selected is not None)
        or result.get("candidate_outcomes_consumed") is not False
    ):
        raise WorkflowError("fresh-selection scene result changed")
    if eligible:
        required = {
            "state_id", "family", "scene_id", "scene_dir",
            "scene_manifest_sha256", "scene_manifest_byte_count", "split",
            "drive_seed", "stratum", "warmup_blocks", "source_step",
            "episode_id", "episode_cluster_id", "cell_id", "boundary", "goal",
            "goal_type", "body_clearance_m", "clearance_m",
            "previous_applied_command",
        }
        allowed = required | {
            "completion_rotation_eligibility_vector", "snapshot_task_status",
        }
        if (
            not required.issubset(selected)
            or not set(selected).issubset(allowed)
            or selected.get("state_id") != task.get("state_id")
            or selected.get("family") != task.get("family")
            or selected.get("stratum") != task.get("stratum")
            or selected.get("scene_id") != task.get("scene_id")
            or selected.get("scene_dir") != task.get("scene_dir")
            or selected.get("scene_manifest_sha256")
            != task.get("scene_manifest_sha256")
            or selected.get("scene_manifest_byte_count")
            != task.get("scene_manifest_byte_count")
            or selected.get("drive_seed") != task.get("drive_seed")
            or isinstance(selected.get("warmup_blocks"), bool)
            or not isinstance(selected.get("warmup_blocks"), int)
            or not FRESH_WARMUP_BLOCKS_MIN
            <= selected["warmup_blocks"] <= FRESH_WARMUP_BLOCKS_MAX
        ):
            raise WorkflowError("fresh-selection selected state changed")
    expected = build_fresh_selection_result(
        task=task, selected_state=selected if eligible else None,
    )
    if dict(result) != expected:
        raise WorkflowError("fresh-selection scene result changed")


def _fresh_selection_artifact_path(
    root: Path, identity: str, *, out_root: Path,
) -> Path:
    if not _sha256_string(identity):
        raise WorkflowError("fresh-selection artifact identity is malformed")
    relative_root = root.absolute().relative_to(out_root.absolute())
    return guarded_output_path(relative_root / f"{identity}.json", out_root=out_root)


def _publish_exact_selection_task(
    task: Mapping[str, Any], *, out_root: Path,
) -> tuple[Path, bool]:
    path = _fresh_selection_artifact_path(
        out_root / "fresh_calibration/selection_tasks",
        task[FRESH_SELECTION_TASK_SELF_KEY], out_root=out_root,
    )
    existed = path.exists()
    _atomic_json(path, task, out_root=out_root)
    return path, not existed


def _selection_result_path(task_digest: str, *, out_root: Path) -> Path:
    return _fresh_selection_artifact_path(
        out_root / "fresh_calibration/selection_results",
        task_digest, out_root=out_root,
    )


def _selection_launch_path(task_digest: str, *, out_root: Path) -> Path:
    if not _sha256_string(task_digest):
        raise WorkflowError("fresh-selection launch identity is malformed")
    relative = Path(
        "fresh_calibration/selection_tasks"
    ) / f"{task_digest}.launch.json"
    return guarded_output_path(relative, out_root=out_root)


def build_fresh_selection_launch_marker(
    *, task: Mapping[str, Any], attempt: Mapping[str, Any],
    correction: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema": FRESH_SELECTION_LAUNCH_SCHEMA,
        "status": "EXCLUSIVE_CHILD_LAUNCH_CLAIMED_BEFORE_GENESIS",
        "task_digest": task[FRESH_SELECTION_TASK_SELF_KEY],
        "attempt_digest": attempt[FRESH_SELECTION_ATTEMPT_SELF_KEY],
        "selector_integrity_replacement_authority_digest": correction[
            V13_CONTRACT.SELECTOR_INTEGRITY_REPLACEMENT_SELF_KEY
        ],
        "selector_digest": V13_CONTRACT.fresh_calibration_selector_digest(),
        "candidate_outcomes_consumed": False,
    }
    return _with_self_digest(payload, FRESH_SELECTION_LAUNCH_SELF_KEY)


def validate_fresh_selection_launch_marker(
    marker: Mapping[str, Any], *, task: Mapping[str, Any],
    attempt: Mapping[str, Any], correction: Mapping[str, Any],
) -> None:
    _validate_self_digest(marker, FRESH_SELECTION_LAUNCH_SELF_KEY)
    expected = build_fresh_selection_launch_marker(
        task=task, attempt=attempt, correction=correction,
    )
    if dict(marker) != expected:
        raise WorkflowError("fresh-selection launch marker changed")


def _claim_fresh_selection_launch(
    *, task: Mapping[str, Any], attempt: Mapping[str, Any],
    correction: Mapping[str, Any], out_root: Path,
) -> tuple[Path, dict[str, Any]]:
    marker = build_fresh_selection_launch_marker(
        task=task, attempt=attempt, correction=correction,
    )
    path = _selection_launch_path(
        task[FRESH_SELECTION_TASK_SELF_KEY], out_root=out_root,
    )
    _atomic_json(path, marker, out_root=out_root, exact_idempotence=False)
    return path, marker


def _selected_state_from_scene(
    *, task: Mapping[str, Any], ctx: Any, record: Mapping[str, Any], block: int,
) -> dict[str, Any]:
    scene_dir = Path(task["scene_dir"])
    episode_id = int(ctx.runner.episode_states[0].episode_id)
    selected = {
        "state_id": task["state_id"],
        "family": task["family"],
        "scene_id": task["scene_id"],
        "scene_dir": task["scene_dir"],
        "scene_manifest_sha256": task["scene_manifest_sha256"],
        "scene_manifest_byte_count": task["scene_manifest_byte_count"],
        "split": scene_dir.parents[1].name,
        "drive_seed": task["drive_seed"],
        "stratum": task["stratum"],
        "warmup_blocks": block,
        "source_step": int(record["boundary"]["source_step"]),
        "episode_id": episode_id,
        "episode_cluster_id": f"{task['scene_id']}/env0/ep{episode_id}",
        "cell_id": int(record["cell_id"]),
        "boundary": record["boundary"],
        "goal": record["goal"],
        "goal_type": record["goal"]["material_id"],
        "body_clearance_m": float(record["body_clearance_m"]),
        "clearance_m": float(record["clearance_m"]),
        "previous_applied_command": np.asarray(
            ctx.runner._last_executed, dtype=np.float64
        )[0].tolist(),
    }
    for key in (
        "completion_rotation_eligibility_vector",
        "snapshot_task_status", "previous_applied_command",
    ):
        if key in record:
            selected[key] = record[key]
    return selected


def execute_fresh_selection_task(
    *, task: Mapping[str, Any], result_path: Path, backend: str,
    out_root: Path,
) -> dict[str, Any]:
    """Evaluate one scene and publish its exact result before runtime teardown."""

    if backend != "cpu" or task.get("backend") != "cpu":
        raise WorkflowError("fresh-selection child is CPU-only")
    if result_path.exists():
        raise WorkflowError("fresh-selection result already exists; rerun refused")
    require_genesis_runtime()
    scene_dir = Path(task["scene_dir"])
    ctx: Any | None = None
    try:
        shared = V1._load_shared(backend)
        ctx = V1.build_context(
            scene_dir, seed=int(task["drive_seed"]), backend=backend,
            shared=shared,
        )
        topology = V12.link_topology(ctx)
        ctx.begin_episode()
        selected: dict[str, Any] | None = None
        for block in range(1, FRESH_WARMUP_BLOCKS_MAX + 1):
            B.drive_block_with_probe(ctx, lambda _tick, _previous: None)
            if block < FRESH_WARMUP_BLOCKS_MIN:
                continue
            verdict = B.classify_state(
                ctx, topology, requested_stratum=task["stratum"]
            )
            if isinstance(verdict, str):
                continue
            record, _field, _strata = verdict
            selected = _selected_state_from_scene(
                task=task, ctx=ctx, record=record, block=block,
            )
            break
        result = build_fresh_selection_result(
            task=task, selected_state=selected,
        )
        _atomic_json(result_path, result, out_root=out_root)
        return result
    finally:
        if ctx is not None:
            del ctx
        gc.collect()


def _validate_fresh_selection_sequence(
    *, attempt: Mapping[str, Any], correction: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(tasks) != len(results):
        raise WorkflowError("fresh-selection task/result inventory differs")
    slots = [(family, stratum) for family in FAMILIES for stratum in STRATA]
    old_scene_ids = set(attempt["old_scene_ids"])
    used_scene_ids: set[str] = set()
    selected_states: list[dict[str, Any]] = []
    slot_index = 0
    scene_ordinal = 0
    for task, result in zip(tasks, results):
        if slot_index >= len(slots):
            raise WorkflowError("fresh-selection continued after all slots were filled")
        validate_fresh_selection_task(
            task, attempt=attempt, correction=correction,
        )
        validate_fresh_selection_result(result, task=task)
        family, stratum = slots[slot_index]
        candidates = [
            scene_id for scene_id in attempt["scene_inventory"][family]
            if scene_id not in old_scene_ids and scene_id not in used_scene_ids
        ]
        if (
            task.get("slot_index") != slot_index
            or task.get("scene_ordinal") != scene_ordinal
            or task.get("family") != family
            or task.get("stratum") != stratum
            or scene_ordinal >= len(candidates)
            or task.get("scene_id") != candidates[scene_ordinal]
            or task.get("used_scene_ids_before") != sorted(used_scene_ids)
        ):
            raise WorkflowError("fresh-selection lexical sequence changed")
        if result["eligible"]:
            selected = copy.deepcopy(dict(result["selected_state"]))
            selected_states.append(selected)
            used_scene_ids.add(selected["scene_id"])
            slot_index += 1
            scene_ordinal = 0
        else:
            scene_ordinal += 1
    if slot_index != len(slots):
        raise WorkflowError("fresh-selection sequence is incomplete")
    return selected_states


def build_fresh_selection_terminal(
    *, attempt: Mapping[str, Any], correction: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    states = _validate_fresh_selection_sequence(
        attempt=attempt, correction=correction, tasks=tasks, results=results,
    )
    rows = []
    for task, result in zip(tasks, results):
        launch = build_fresh_selection_launch_marker(
            task=task, attempt=attempt, correction=correction,
        )
        task_digest = task[FRESH_SELECTION_TASK_SELF_KEY]
        rows.append({
            "task_digest": task_digest,
            "launch_digest": launch[FRESH_SELECTION_LAUNCH_SELF_KEY],
            "result_digest": result[FRESH_SELECTION_RESULT_SELF_KEY],
            "task_path": str(Path("selection_tasks") / f"{task_digest}.json"),
            "launch_path": str(
                Path("selection_tasks") / f"{task_digest}.launch.json"
            ),
            "result_path": str(
                Path("selection_results") / f"{task_digest}.json"
            ),
            "eligible": result["eligible"],
        })
    payload = {
        "schema": FRESH_SELECTION_TERMINAL_SCHEMA,
        "status": "COMPLETE_EXACT_24_STATE_SELECTOR_TERMINAL",
        "complete": True,
        "attempt_digest": attempt[FRESH_SELECTION_ATTEMPT_SELF_KEY],
        "selector_integrity_replacement_authority_digest": correction[
            V13_CONTRACT.SELECTOR_INTEGRITY_REPLACEMENT_SELF_KEY
        ],
        "selector_digest": V13_CONTRACT.fresh_calibration_selector_digest(),
        "task_result_count": len(rows),
        "selected_state_count": len(states),
        "selected_scene_count": len({row["scene_id"] for row in states}),
        "states": states,
        "states_digest": digest(states),
        "rows": rows,
        "candidate_outcomes_consumed": False,
        "state_manifest_published": False,
    }
    return _with_self_digest(payload, FRESH_SELECTION_TERMINAL_SELF_KEY)


def build_failed_fresh_selection_terminal(
    *, attempt: Mapping[str, Any], correction: Mapping[str, Any],
    failed_task: Mapping[str, Any], return_code: int | None,
    failure_kind: str = "ISOLATED_CHILD_RETURNED_WITHOUT_VALID_RESULT",
) -> dict[str, Any]:
    payload = {
        "schema": FRESH_SELECTION_TERMINAL_SCHEMA,
        "status": "TERMINAL_UNRESOLVED_ISOLATED_CHILD",
        "complete": False,
        "attempt_digest": attempt[FRESH_SELECTION_ATTEMPT_SELF_KEY],
        "selector_integrity_replacement_authority_digest": correction[
            V13_CONTRACT.SELECTOR_INTEGRITY_REPLACEMENT_SELF_KEY
        ],
        "selector_digest": V13_CONTRACT.fresh_calibration_selector_digest(),
        "failed_task_digest": failed_task[FRESH_SELECTION_TASK_SELF_KEY],
        "failed_scene_id": failed_task["scene_id"],
        "failure_kind": failure_kind,
        "return_code": None if return_code is None else int(return_code),
        "valid_result_published": False,
        "retry_or_scene_replacement_authorised": False,
        "candidate_branch_execution_started": False,
    }
    return _with_self_digest(payload, FRESH_SELECTION_TERMINAL_SELF_KEY)


def _selection_terminal_rows(
    terminal: Mapping[str, Any], *, attempt: Mapping[str, Any],
    correction: Mapping[str, Any], out_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    fresh_root = out_root / "fresh_calibration"
    for row in terminal.get("rows", []):
        task_relative = Path(str(row.get("task_path", "")))
        launch_relative = Path(str(row.get("launch_path", "")))
        result_relative = Path(str(row.get("result_path", "")))
        if (
            task_relative.is_absolute() or launch_relative.is_absolute()
            or result_relative.is_absolute() or ".." in task_relative.parts
            or ".." in launch_relative.parts or ".." in result_relative.parts
        ):
            raise WorkflowError("fresh-selection terminal path is unsafe")
        task = _read_regular_json(fresh_root / task_relative)
        launch = _read_regular_json(fresh_root / launch_relative)
        result = _read_regular_json(fresh_root / result_relative)
        if (
            task.get(FRESH_SELECTION_TASK_SELF_KEY) != row.get("task_digest")
            or launch.get(FRESH_SELECTION_LAUNCH_SELF_KEY)
            != row.get("launch_digest")
            or result.get(FRESH_SELECTION_RESULT_SELF_KEY)
            != row.get("result_digest")
            or result.get("eligible") != row.get("eligible")
        ):
            raise WorkflowError("fresh-selection terminal row changed")
        validate_fresh_selection_task(
            task, attempt=attempt, correction=correction,
        )
        validate_fresh_selection_launch_marker(
            launch, task=task, attempt=attempt, correction=correction,
        )
        validate_fresh_selection_result(result, task=task)
        tasks.append(task)
        results.append(result)
    return tasks, results


def validate_fresh_selection_terminal(
    terminal: Mapping[str, Any], *, attempt: Mapping[str, Any],
    correction: Mapping[str, Any], out_root: Path,
) -> list[dict[str, Any]]:
    _validate_self_digest(terminal, FRESH_SELECTION_TERMINAL_SELF_KEY)
    if terminal.get("complete") is not True:
        failure_kind = terminal.get("failure_kind")
        return_code = terminal.get("return_code")
        failed_task_digest = terminal.get("failed_task_digest")
        if (
            terminal.get("schema") != FRESH_SELECTION_TERMINAL_SCHEMA
            or terminal.get("status") != "TERMINAL_UNRESOLVED_ISOLATED_CHILD"
            or terminal.get("attempt_digest")
            != attempt.get(FRESH_SELECTION_ATTEMPT_SELF_KEY)
            or terminal.get("selector_integrity_replacement_authority_digest")
            != correction.get(V13_CONTRACT.SELECTOR_INTEGRITY_REPLACEMENT_SELF_KEY)
            or terminal.get("valid_result_published") is not False
            or terminal.get("retry_or_scene_replacement_authorised") is not False
            or terminal.get("candidate_branch_execution_started") is not False
            or not _sha256_string(failed_task_digest)
            or failure_kind not in {
                "ISOLATED_CHILD_RETURNED_WITHOUT_VALID_RESULT",
                "PREEXISTING_TASK_WITHOUT_VALID_RESULT",
            }
            or (
                failure_kind == "PREEXISTING_TASK_WITHOUT_VALID_RESULT"
                and return_code is not None
            )
            or (
                failure_kind == "ISOLATED_CHILD_RETURNED_WITHOUT_VALID_RESULT"
                and (isinstance(return_code, bool)
                     or not isinstance(return_code, int))
            )
        ):
            raise WorkflowError("fresh-selection failure terminal changed")
        failed_task_path = _fresh_selection_artifact_path(
            out_root / "fresh_calibration/selection_tasks",
            failed_task_digest, out_root=out_root,
        )
        failed_task = _read_regular_json(failed_task_path)
        validate_fresh_selection_task(
            failed_task, attempt=attempt, correction=correction,
        )
        expected = build_failed_fresh_selection_terminal(
            attempt=attempt, correction=correction,
            failed_task=failed_task, return_code=return_code,
            failure_kind=failure_kind,
        )
        if dict(terminal) != expected:
            raise WorkflowError("fresh-selection failure terminal changed")
        raise WorkflowError("fresh-selection attempt is terminal after unresolved child")
    tasks, results = _selection_terminal_rows(
        terminal, attempt=attempt, correction=correction, out_root=out_root,
    )
    expected = build_fresh_selection_terminal(
        attempt=attempt, correction=correction, tasks=tasks, results=results,
    )
    if dict(terminal) != expected:
        raise WorkflowError("fresh-selection success terminal changed")
    return copy.deepcopy(expected["states"])


def _run_fresh_selection_child(
    *, task_path: Path, result_path: Path, backend: str,
) -> int:
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--fresh-selection-worker-task", str(task_path),
        "--fresh-selection-worker-result", str(result_path),
        "--backend", backend,
    ]
    try:
        completed = subprocess.run(command, cwd=ROOT, check=False)
    except OSError:
        return 127
    return int(completed.returncode)


def run_fresh_selection_worker(
    *, task_path: Path, result_path: Path, backend: str,
) -> dict[str, Any]:
    out_root = result_path.absolute().parents[2]
    expected_task_root = out_root / "fresh_calibration/selection_tasks"
    expected_result_root = out_root / "fresh_calibration/selection_results"
    terminal_path = out_root / "fresh_calibration/selection_terminal.json"
    if (
        task_path.absolute().parent != expected_task_root.absolute()
        or result_path.absolute().parent != expected_result_root.absolute()
        or task_path.name != result_path.name
    ):
        raise WorkflowError("fresh-selection worker paths changed")
    launch_path = _selection_launch_path(task_path.stem, out_root=out_root)
    if terminal_path.exists() or terminal_path.is_symlink():
        raise WorkflowError("fresh-selection worker refused an existing terminal")
    if result_path.exists() or result_path.is_symlink():
        raise WorkflowError("fresh-selection worker refused an existing result")
    if launch_path.exists() or launch_path.is_symlink():
        raise WorkflowError("fresh-selection worker refused an existing launch marker")
    authority, correction = load_selector_integrity_replacement_authority(
        out_root=out_root
    )
    del authority
    task = _read_regular_json(task_path)
    attempt = _read_regular_json(
        out_root / "fresh_calibration/selection_attempt.json"
    )
    _validate_self_digest(attempt, FRESH_SELECTION_ATTEMPT_SELF_KEY)
    validate_fresh_selection_task(
        task, attempt=attempt, correction=correction,
    )
    if (
        task.get(FRESH_SELECTION_TASK_SELF_KEY) != task_path.stem
    ):
        raise WorkflowError("fresh-selection worker task authority changed")
    claimed_path, marker = _claim_fresh_selection_launch(
        task=task, attempt=attempt, correction=correction, out_root=out_root,
    )
    if claimed_path != launch_path:
        raise WorkflowError("fresh-selection launch marker path changed")
    validate_fresh_selection_launch_marker(
        marker, task=task, attempt=attempt, correction=correction,
    )
    if terminal_path.exists() or terminal_path.is_symlink():
        raise WorkflowError("fresh-selection worker refused an existing terminal")
    return execute_fresh_selection_task(
        task=task, result_path=result_path, backend=backend, out_root=out_root,
    )


def select_fresh_calibration_states(*, backend: str = "cpu",
                                    out_root: Path = OUT_ROOT) -> dict[str, Any]:
    """Run the one-shot selector as durable, per-scene isolated tasks."""

    if backend != "cpu":
        raise WorkflowError("fresh calibration selection is CPU-only")
    authority, correction = load_selector_integrity_replacement_authority(
        out_root=out_root
    )
    require_genesis_runtime()
    corpus = load_v2_corpus()
    plan = load_replay_plan(out_root=out_root, authority=authority)
    equivalence = _read_regular_json(out_root / "equivalence_receipt.json")
    validate_equivalence_receipt(equivalence, authority, corpus["rows"])
    overlay_manifest = _read_regular_json(out_root / "replay_overlay_manifest.json")
    _validate_self_digest(overlay_manifest, REPLAY_OVERLAY_MANIFEST_SELF_KEY)
    if (
        overlay_manifest.get("overlay_count") != EXPECTED_V2_INVALID
        or overlay_manifest.get("authority_digest") != authority[AUTHORITY_SELF_KEY]
    ):
        raise WorkflowError("fresh selection requires all exact replay overlays")
    pool, exclusion = B.scene_pool("scorer_fit")
    if tuple(sorted(pool)) != FAMILIES:
        raise WorkflowError("fresh calibration family inventory changed")
    old_scene_ids = {
        str(row["scene_id"]) for row in corpus["state_manifest"]["states"]
    }
    expected_attempt = build_fresh_selection_attempt(
        authority=authority, correction=correction,
        v2_state_manifest=corpus["state_manifest"], plan=plan,
        equivalence=equivalence, overlay_manifest=overlay_manifest,
        pool=pool, exclusion=exclusion,
    )
    attempt_path = out_root / "fresh_calibration/selection_attempt.json"
    if attempt_path.exists():
        attempt = _read_regular_json(attempt_path)
        validate_fresh_selection_attempt(
            attempt, authority=authority, correction=correction,
            v2_state_manifest=corpus["state_manifest"], plan=plan,
            equivalence=equivalence, overlay_manifest=overlay_manifest,
            pool=pool, exclusion=exclusion,
        )
    else:
        attempt = expected_attempt
        _atomic_json(attempt_path, attempt, out_root=out_root)

    terminal_path = out_root / "fresh_calibration/selection_terminal.json"
    state_manifest_path = out_root / "fresh_calibration/state_manifest.json"
    if terminal_path.exists():
        terminal = _read_regular_json(terminal_path)
        chosen = validate_fresh_selection_terminal(
            terminal, attempt=attempt, correction=correction, out_root=out_root,
        )
    else:
        if state_manifest_path.exists():
            raise WorkflowError("fresh state manifest exists without selector terminal")
        tasks: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        used_scenes: set[str] = set()
        for slot_index, (family, stratum) in enumerate(
            (family, stratum) for family in FAMILIES for stratum in STRATA
        ):
            selected: dict[str, Any] | None = None
            scene_ordinal = 0
            for scene_dir in sorted(pool[family], key=lambda path: path.name):
                if scene_dir.name in used_scenes or scene_dir.name in old_scene_ids:
                    continue
                task = build_fresh_selection_task(
                    attempt=attempt, correction=correction, family=family,
                    stratum=stratum, slot_index=slot_index,
                    scene_ordinal=scene_ordinal, scene_dir=scene_dir,
                    used_scene_ids=sorted(used_scenes),
                )
                task_path, task_created = _publish_exact_selection_task(
                    task, out_root=out_root
                )
                result_path = _selection_result_path(
                    task[FRESH_SELECTION_TASK_SELF_KEY], out_root=out_root,
                )
                return_code = 0
                if not task_created and not result_path.exists():
                    failed = build_failed_fresh_selection_terminal(
                        attempt=attempt, correction=correction,
                        failed_task=task, return_code=None,
                        failure_kind="PREEXISTING_TASK_WITHOUT_VALID_RESULT",
                    )
                    _atomic_json(terminal_path, failed, out_root=out_root)
                    raise FreshSelectionChildUnresolved(
                        task[FRESH_SELECTION_TASK_SELF_KEY], None
                    )
                if not result_path.exists():
                    return_code = _run_fresh_selection_child(
                        task_path=task_path, result_path=result_path,
                        backend=backend,
                    )
                if not result_path.exists():
                    failed = build_failed_fresh_selection_terminal(
                        attempt=attempt, correction=correction,
                        failed_task=task, return_code=return_code,
                    )
                    _atomic_json(terminal_path, failed, out_root=out_root)
                    raise FreshSelectionChildUnresolved(
                        task[FRESH_SELECTION_TASK_SELF_KEY], return_code
                    )
                try:
                    launch = _read_regular_json(_selection_launch_path(
                        task[FRESH_SELECTION_TASK_SELF_KEY], out_root=out_root,
                    ))
                    validate_fresh_selection_launch_marker(
                        launch, task=task, attempt=attempt,
                        correction=correction,
                    )
                    result = _read_regular_json(result_path)
                    validate_fresh_selection_result(result, task=task)
                except WorkflowError as exc:
                    failure_kind = (
                        "ISOLATED_CHILD_RETURNED_WITHOUT_VALID_RESULT"
                        if task_created
                        else "PREEXISTING_TASK_WITHOUT_VALID_RESULT"
                    )
                    failure_code = return_code if task_created else None
                    failed = build_failed_fresh_selection_terminal(
                        attempt=attempt, correction=correction,
                        failed_task=task, return_code=failure_code,
                        failure_kind=failure_kind,
                    )
                    _atomic_json(terminal_path, failed, out_root=out_root)
                    raise FreshSelectionChildUnresolved(
                        task[FRESH_SELECTION_TASK_SELF_KEY], failure_code
                    ) from exc
                tasks.append(task)
                results.append(result)
                if result["eligible"]:
                    selected = copy.deepcopy(dict(result["selected_state"]))
                    used_scenes.add(selected["scene_id"])
                    break
                scene_ordinal += 1
            if selected is None:
                raise WorkflowError(
                    f"no frozen fresh calibration identity for {family}/{stratum}"
                )
        terminal = build_fresh_selection_terminal(
            attempt=attempt, correction=correction, tasks=tasks, results=results,
        )
        _atomic_json(terminal_path, terminal, out_root=out_root)
        chosen = validate_fresh_selection_terminal(
            terminal, attempt=attempt, correction=correction, out_root=out_root,
        )

    manifest = build_fresh_calibration_manifest(
        authority=authority, v2_state_manifest=corpus["state_manifest"],
        states=chosen,
        exclusion_binding=attempt["manifest_exclusion_binding"],
        selector_integrity_replacement_authority=correction,
    )
    _atomic_json(state_manifest_path, manifest, out_root=out_root)
    validate_fresh_calibration_manifest(
        manifest, authority=authority,
        v2_state_manifest=corpus["state_manifest"],
        selector_integrity_replacement_authority=correction,
    )
    return manifest


def load_validated_fresh_selection(
    *, out_root: Path = OUT_ROOT, v2_root: Path = V2_ROOT,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load the paired authority only after terminal and manifest validation."""

    authority, correction = load_selector_integrity_replacement_authority(
        out_root=out_root
    )
    corpus = load_v2_corpus(v2_root=v2_root)
    plan = load_replay_plan(
        out_root=out_root, v2_root=v2_root, authority=authority,
    )
    equivalence = _read_regular_json(out_root / "equivalence_receipt.json")
    validate_equivalence_receipt(equivalence, authority, corpus["rows"])
    overlay_manifest = _read_regular_json(
        out_root / "replay_overlay_manifest.json"
    )
    _validate_self_digest(overlay_manifest, REPLAY_OVERLAY_MANIFEST_SELF_KEY)
    pool, exclusion = B.scene_pool("scorer_fit")
    attempt = _read_regular_json(
        out_root / "fresh_calibration/selection_attempt.json"
    )
    validate_fresh_selection_attempt(
        attempt, authority=authority, correction=correction,
        v2_state_manifest=corpus["state_manifest"], plan=plan,
        equivalence=equivalence, overlay_manifest=overlay_manifest,
        pool=pool, exclusion=exclusion,
    )
    terminal = _read_regular_json(
        out_root / "fresh_calibration/selection_terminal.json"
    )
    states = validate_fresh_selection_terminal(
        terminal, attempt=attempt, correction=correction, out_root=out_root,
    )
    manifest = _read_regular_json(
        out_root / "fresh_calibration/state_manifest.json"
    )
    validate_fresh_calibration_manifest(
        manifest, authority=authority,
        v2_state_manifest=corpus["state_manifest"],
        selector_integrity_replacement_authority=correction,
    )
    expected_manifest = build_fresh_calibration_manifest(
        authority=authority, v2_state_manifest=corpus["state_manifest"],
        states=states,
        exclusion_binding=attempt["manifest_exclusion_binding"],
        selector_integrity_replacement_authority=correction,
    )
    continuation = V13_CONTRACT.SELECTOR_INTEGRITY_REPLACEMENT[
        "preserved_predecessor_authority_continuation"
    ]
    if (
        manifest != expected_manifest
        or continuation["replacement_grants_new_candidate_branch_authority"]
        is not False
        or continuation["predecessor_candidate_branch_authority_preserved"]
        is not True
        or continuation["continuation_requires_valid_selector_terminal"]
        is not True
        or continuation["continuation_requires_exact_24_state_manifest"]
        is not True
        or continuation[
            "continuation_requires_current_successor_preregistration_binding"
        ] is not True
        or continuation["fresh_branch_attempts_before_replacement"] != 0
        or continuation["fresh_branch_budget"]
        != EXPECTED_FRESH_CALIBRATION_BRANCHES
    ):
        raise WorkflowError("fresh branch continuation authority changed")
    return authority, correction, corpus, manifest


# ------------------------------------------------ fresh branch generation --
def _fresh_branch_identity(
    state: Mapping[str, Any], candidate_index: int,
    state_manifest: Mapping[str, Any], authority: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = V1.CANDIDATE_BANK[candidate_index]
    payload = {
        "schema": "go2_scorer_fit_oracle_v1_3_fresh_calibration_branch_identity_v1",
        "state_id": state["state_id"],
        "state_identity_digest": state["state_identity_digest"],
        "scene_id": state["scene_id"],
        "candidate_index": candidate_index,
        "candidate": candidate[0],
        "primitives": list(candidate[1]),
        "goal": state["goal"],
        "candidate_bank_digest": V1.bank_digest(),
        "oracle_v1_3_digest": authority["oracle_v1_3_digest"],
        "authority_digest": authority[AUTHORITY_SELF_KEY],
        "fresh_calibration_state_manifest_digest": state_manifest[
            FRESH_STATE_MANIFEST_SELF_KEY
        ],
    }
    return {**payload, "branch_identity_digest": digest(payload)}


def _fresh_row_path(identity: str, *, out_root: Path = OUT_ROOT) -> Path:
    if not _sha256_string(identity):
        raise WorkflowError("fresh row identity is malformed")
    return out_root / "fresh_calibration/rows" / f"{identity}.json"


def _fresh_frame_receipt(
    result: Any, path: Path, *, fresh_root: Path, index_key: str, index: int,
) -> dict[str, Any]:
    try:
        relative = path.absolute().relative_to(fresh_root.absolute())
    except ValueError as exc:
        raise WorkflowError("fresh RGB frame escaped its dedicated root") from exc
    # Revalidate the registered top-level storage alias immediately before
    # each potentially large frame write.  Treating ``fresh_root`` as an
    # independent custom root would bypass the production alias contract.
    out_root = fresh_root.parent
    guarded_relative = path.absolute().relative_to(out_root.absolute())
    path = guarded_output_path(guarded_relative, out_root=out_root)
    from PIL import Image
    image = np.asarray(result.image)
    if image.shape != (224, 224, 3) or image.dtype != np.uint8:
        raise WorkflowError("fresh calibration renderer returned an invalid RGB array")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    Image.fromarray(image).save(temporary, format="PNG")
    digest_value = file_sha256(temporary)
    byte_count = temporary.stat().st_size
    if path.exists():
        if (
            path.is_file() and not path.is_symlink()
            and path.stat().st_size == byte_count
            and file_sha256(path) == digest_value
        ):
            temporary.unlink()
        else:
            temporary.unlink(missing_ok=True)
            raise WorkflowError("refusing to replace a differing fresh RGB frame")
    else:
        os.link(temporary, path)
        temporary.unlink()
    return {
        index_key: index,
        "path": str(relative),
        "sha256": digest_value,
        "byte_count": byte_count,
        "shape": [224, 224, 3],
        "dtype": "uint8",
        "camera_pose_world": result.camera_pose_world,
    }


def _fresh_redrive_matches(state: Mapping[str, Any], record: Mapping[str, Any],
                           ctx: V1.BranchContext) -> bool:
    keys = ("boundary", "cell_id", "goal", "body_clearance_m", "clearance_m")
    previous = np.asarray(ctx.runner._last_executed, dtype=np.float64)[0].tolist()
    return bool(
        all(state.get(key) == record.get(key) for key in keys)
        and int(ctx.runner.episode_states[0].episode_id) == int(state["episode_id"])
        and int(record["boundary"]["source_step"]) == int(state["source_step"])
        and state.get("previous_applied_command") == previous
        and all(
            state.get(key) == record.get(key)
            for key in ("completion_rotation_eligibility_vector",
                        "snapshot_task_status")
            if key in state or key in record
        )
    )


def _validate_fresh_branch_row(
    row: Mapping[str, Any], *, state: Mapping[str, Any],
    state_manifest: Mapping[str, Any], authority: Mapping[str, Any],
    out_root: Path,
) -> None:
    _validate_self_digest(row, FRESH_BRANCH_ROW_SELF_KEY)
    candidate_index = row.get("candidate_index")
    if (
        isinstance(candidate_index, bool)
        or not isinstance(candidate_index, int)
        or candidate_index not in EXPECTED_CANDIDATES
    ):
        raise WorkflowError("fresh calibration row candidate index is malformed")
    expected = _fresh_branch_identity(
        state, candidate_index, state_manifest, authority
    )
    score = _v13().score_branch_v13(row.get("trace", {}))
    marker_path = _attempt_path(
        "fresh_calibration", expected["branch_identity_digest"], out_root=out_root
    )
    marker = _read_regular_json(marker_path)
    _validate_self_digest(marker, "attempt_digest")
    identity_matches = all(
        row.get(key) == value for key, value in expected.items() if key != "schema"
    )
    if (
        row.get("schema") != FRESH_BRANCH_ROW_SCHEMA
        or row.get("branch_identity_schema") != expected["schema"]
        or row.get("status") != STATUS
        or row.get("record_complete") is not True
        or row.get("valid") is not True
        or not identity_matches
        or row.get("authority_digest") != authority[AUTHORITY_SELF_KEY]
        or row.get("oracle_v1_3_digest") != authority["oracle_v1_3_digest"]
        or row.get("scorer_fit_oracle_v1_3_contract_digest")
        != authority["scorer_fit_oracle_v1_3_contract_digest"]
        or row.get("fresh_calibration_state_manifest_digest")
        != state_manifest[FRESH_STATE_MANIFEST_SELF_KEY]
        or row.get("state_index") != state["state_index"]
        or row.get("split_role") != "calibration"
        or marker.get("kind") != "fresh_calibration"
        or marker.get("identity_digest") != expected["branch_identity_digest"]
        or marker.get("attempt_digest") != row.get("attempt_digest")
        or score is None
        or _label_projection(score) != _label_projection(row)
        or len(row.get("context_frames", [])) != 3
        or len(row.get("horizon_frames", [])) != 4
    ):
        raise WorkflowError("fresh calibration branch row binding changed")
    candidate = V1.CANDIDATE_BANK[candidate_index]
    requested, post_slew, action_blocks = B.candidate_planning_trajectory(
        candidate, row.get("previous_applied_command", [])
    )
    if (
        row.get("requested") != requested
        or row.get("action_blocks") != action_blocks
        or not np.allclose(
            np.asarray(row.get("post_slew"), dtype=np.float64),
            np.asarray(post_slew, dtype=np.float64), rtol=0.0, atol=ACTION_ATOL,
        )
    ):
        raise WorkflowError("fresh calibration action trajectory changed")
    fresh_root = out_root / "fresh_calibration"
    for frame in [*row["context_frames"], *row["horizon_frames"]]:
        relative = Path(str(frame.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise WorkflowError("fresh calibration frame path is unsafe")
        path = fresh_root / relative
        if (
            not path.is_file() or path.is_symlink()
            or path.stat().st_size != frame.get("byte_count")
            or file_sha256(path) != frame.get("sha256")
            or frame.get("shape") != [224, 224, 3]
            or frame.get("dtype") != "uint8"
        ):
            raise WorkflowError("fresh calibration frame receipt changed")


def generate_fresh_calibration(*, backend: str = "cpu",
                               out_root: Path = OUT_ROOT) -> dict[str, Any]:
    if backend != "cpu":
        raise WorkflowError("fresh calibration branch generation is CPU-only")
    authority, correction, corpus, manifest = load_validated_fresh_selection(
        out_root=out_root
    )
    require_genesis_runtime()
    del correction, corpus
    fresh_root = out_root / "fresh_calibration"
    shared = V1._load_shared(backend)
    completed: list[dict[str, Any]] = []
    for state in manifest["states"]:
        scene_dir = Path(state["scene_dir"])
        manifest_path = scene_dir / "manifest.json"
        if (
            B.file_sha256(manifest_path) != state["scene_manifest_sha256"]
            or manifest_path.stat().st_size != state["scene_manifest_byte_count"]
        ):
            raise WorkflowError("fresh calibration scene manifest changed")
        identities = [
            _fresh_branch_identity(state, index, manifest, authority)
            for index in EXPECTED_CANDIDATES
        ]
        existing = []
        for identity in identities:
            path = _fresh_row_path(identity["branch_identity_digest"], out_root=out_root)
            if path.exists():
                row = _read_regular_json(path)
                _validate_fresh_branch_row(
                    row, state=state, state_manifest=manifest,
                    authority=authority, out_root=out_root,
                )
                existing.append(row)
        if len(existing) == len(identities):
            completed.extend(existing)
            continue
        # Any orphaned attempt is an interrupted/failed scientific execution;
        # it cannot be silently retried or replaced.
        for identity in identities:
            path = _fresh_row_path(identity["branch_identity_digest"], out_root=out_root)
            attempt_path = _attempt_path(
                "fresh_calibration", identity["branch_identity_digest"], out_root=out_root
            )
            if attempt_path.exists() and not path.exists():
                raise WorkflowError("orphaned fresh branch attempt; retry refused")
        ctx = V1.build_context(
            scene_dir, seed=int(state["drive_seed"]), backend=backend, shared=shared
        )
        topology = V12.link_topology(ctx)
        ctx.begin_episode()
        proprio_log: list[list[float]] = []
        control_log: list[list[float]] = []
        context_poses: list[BasePose] = []
        action_context_blocks: list[list[float]] = []

        def probe(_tick: int, previous: Sequence[float]) -> None:
            proprio_log.append(B.proprio_sample(ctx))
            control_log.append(B.control_sample(previous))

        for block_index in range(int(state["warmup_blocks"])):
            driven = B.drive_block_with_probe(ctx, probe)
            if block_index >= int(state["warmup_blocks"]) - B.CONTEXT_SLOTS:
                action_context_blocks.append(B.action_block_10d(
                    np.asarray(driven.executed, dtype=np.float64)[0]
                ))
                context_poses.append(capture_base_pose(ctx))
        verdict = B.classify_state(ctx, topology, requested_stratum=state["stratum"])
        if isinstance(verdict, str):
            raise WorkflowError(f"fresh state redrive failed: {verdict}")
        record, field, _strata = verdict
        if not _fresh_redrive_matches(state, record, ctx):
            raise WorkflowError("fresh calibration state redrive changed")
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        import genesis as gs
        renderer = TexturedV03Renderer(ctx, gs=gs, raw_manifest=raw_manifest)
        context_frames = []
        for slot, pose in enumerate(context_poses):
            result = renderer.render_pose(pose)
            path = fresh_root / "frames" / state["family"] / (
                f"{state['state_identity_digest']}_ctx{slot}.png"
            )
            context_frames.append(_fresh_frame_receipt(
                result, path, fresh_root=fresh_root, index_key="slot", index=slot
            ))
        proprio = np.asarray(proprio_log[-B.PROPRIO_HISTORY:], dtype=np.float32)
        control = np.asarray(control_log[-B.PROPRIO_HISTORY:], dtype=np.float32)
        if proprio.shape != (B.PROPRIO_HISTORY, 30) or control.shape != (B.PROPRIO_HISTORY, 2):
            raise WorkflowError("fresh calibration planning history changed")
        previous = np.asarray(ctx.runner._last_executed, dtype=np.float64)[0].tolist()
        snapshot = V1.capture_branch_state(
            ctx, goal=state["goal"], identity={
                "state_id": state["state_id"],
                "state_identity_digest": state["state_identity_digest"],
                "scene_id": state["scene_id"], "family": state["family"],
                "split": state["split"], "block_index": state["warmup_blocks"],
                "source_step": state["source_step"], "episode_id": state["episode_id"],
            },
        )
        for identity in identities:
            output = _fresh_row_path(identity["branch_identity_digest"], out_root=out_root)
            if output.exists():
                retained = _read_regular_json(output)
                _validate_fresh_branch_row(
                    retained, state=state, state_manifest=manifest,
                    authority=authority, out_root=out_root,
                )
                completed.append(retained)
                continue
            attempt = begin_attempt_once(
                "fresh_calibration", identity["branch_identity_digest"],
                {"authority_digest": authority[AUTHORITY_SELF_KEY],
                 "state_manifest_digest": manifest[FRESH_STATE_MANIFEST_SELF_KEY]},
                out_root=out_root,
            )
            candidate = V1.CANDIDATE_BANK[identity["candidate_index"]]
            requested_plan, post_slew_plan, action_blocks = B.candidate_planning_trajectory(
                candidate, previous
            )
            horizon_poses: list[BasePose] = []
            branch = execute_branch_trace_v13(
                ctx, snapshot, candidate, field=field, topology=topology,
                on_block_end=lambda _index: horizon_poses.append(capture_base_pose(ctx)),
            )
            if not np.allclose(
                np.asarray(branch["post_slew"], dtype=np.float64),
                np.asarray(post_slew_plan, dtype=np.float64),
                rtol=0.0, atol=ACTION_ATOL,
            ):
                raise WorkflowError("fresh calibration post-slew plan changed")
            score = _v13().score_branch_v13(branch)
            if score is None:
                raise WorkflowError("fresh calibration branch lacks a label; retry refused")
            horizon_frames = []
            for horizon, pose in enumerate(horizon_poses, start=1):
                result = renderer.render_pose(pose)
                path = fresh_root / "frames" / state["family"] / (
                    f"{identity['branch_identity_digest']}_h{horizon}.png"
                )
                horizon_frames.append(_fresh_frame_receipt(
                    result, path, fresh_root=fresh_root,
                    index_key="horizon", index=horizon,
                ))
            if len(horizon_frames) != 4:
                raise WorkflowError("fresh calibration row lacks four target frames")
            row_payload = {
                **{key: value for key, value in identity.items() if key != "schema"},
                "schema": FRESH_BRANCH_ROW_SCHEMA,
                "branch_identity_schema": identity["schema"],
                "status": STATUS,
                "record_complete": True,
                "valid": True,
                "authority_digest": authority[AUTHORITY_SELF_KEY],
                "oracle_v1_3_digest": authority["oracle_v1_3_digest"],
                "scorer_fit_oracle_v1_3_contract_digest": authority[
                    "scorer_fit_oracle_v1_3_contract_digest"
                ],
                "fresh_calibration_state_manifest_digest": manifest[
                    FRESH_STATE_MANIFEST_SELF_KEY
                ],
                "state_index": state["state_index"],
                "split_role": "calibration",
                "stratum": state["stratum"], "scene_id": state["scene_id"],
                "family": state["family"], "split": state["split"],
                "episode_cluster_id": state["episode_cluster_id"],
                "episode_id": state["episode_id"], "source_step": state["source_step"],
                "requested": requested_plan,
                "realised_requested_prefix": branch["requested"],
                "post_slew": branch["post_slew"],
                "candidate_post_slew_plan": post_slew_plan,
                "action_blocks": action_blocks,
                "action_context_blocks": action_context_blocks,
                "previous_applied_command": previous,
                "goal_binding_input": [
                    math.sin(float(state["goal"]["bearing_body_rad"])),
                    math.cos(float(state["goal"]["bearing_body_rad"])),
                    float(state["goal"]["range_m"]),
                ],
                "context_frames": context_frames,
                "horizon_frames": horizon_frames,
                "context_paths": [frame["path"] for frame in context_frames],
                "horizon_paths": [frame["path"] for frame in horizon_frames],
                "proprio": proprio.tolist(), "control": control.tolist(),
                "masks": {
                    "context_rgb_valid": [True] * 3,
                    "observed_proprio_valid": [True] * B.PROPRIO_HISTORY,
                    "observed_control_valid": [True] * B.PROPRIO_HISTORY,
                    "future_proprio_available": [False] * 4,
                    "target_rgb_valid": [True] * 4,
                },
                "timing": {
                    "command_hz": 10, "ticks_per_block": 5,
                    "seconds_per_block": 0.5,
                    "context_boundary_offsets_blocks": [-2, -1, 0],
                    "target_horizons_blocks": [1, 2, 3, 4],
                },
                "snapshot_digest": snapshot.digest,
                "attempt_digest": attempt["attempt_digest"],
                "trace": branch,
                **_label_projection(score),
            }
            row = _with_self_digest(row_payload, FRESH_BRANCH_ROW_SELF_KEY)
            _atomic_json(output, row, out_root=out_root)
            completed.append(row)
        del renderer, ctx
        gc.collect()
    return compile_fresh_calibration_corpus(
        authority=authority, state_manifest=manifest, rows=completed,
        out_root=out_root,
    )


def compile_fresh_calibration_corpus(
    *, authority: Mapping[str, Any], state_manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]], out_root: Path = OUT_ROOT,
) -> dict[str, Any]:
    by_identity: dict[str, Mapping[str, Any]] = {}
    state_by_id = {state["state_id"]: state for state in state_manifest["states"]}
    for row in rows:
        state = state_by_id.get(row.get("state_id"))
        if state is None:
            raise WorkflowError("fresh calibration row has an unknown state")
        _validate_fresh_branch_row(
            row, state=state, state_manifest=state_manifest,
            authority=authority, out_root=out_root,
        )
        identity = str(row.get("branch_identity_digest", ""))
        if identity in by_identity or row.get("valid") is not True:
            raise WorkflowError("fresh calibration corpus has duplicate/invalid row")
        _label_projection(row)
        by_identity[identity] = row
    expected = {
        _fresh_branch_identity(state, index, state_manifest, authority)[
            "branch_identity_digest"
        ]
        for state in state_manifest["states"] for index in EXPECTED_CANDIDATES
    }
    if set(by_identity) != expected or len(by_identity) != EXPECTED_FRESH_CALIBRATION_BRANCHES:
        raise WorkflowError("fresh calibration corpus is not exact 24x12")
    inventory = []
    for identity in sorted(by_identity):
        path = _fresh_row_path(identity, out_root=out_root)
        inventory.append({
            "branch_identity_digest": identity,
            "fresh_calibration_branch_row_digest": by_identity[identity][
                FRESH_BRANCH_ROW_SELF_KEY
            ],
            "path": _repo_relative(path),
            "sha256": file_sha256(path),
        })
    payload = {
        "schema": FRESH_CORPUS_SCHEMA,
        "status": STATUS,
        "complete": True,
        "authority_digest": authority[AUTHORITY_SELF_KEY],
        "oracle_v1_3_digest": authority["oracle_v1_3_digest"],
        "scorer_fit_oracle_v1_3_contract_digest": authority[
            "scorer_fit_oracle_v1_3_contract_digest"
        ],
        "fresh_calibration_state_manifest_digest": state_manifest[
            FRESH_STATE_MANIFEST_SELF_KEY
        ],
        "state_count": EXPECTED_FRESH_CALIBRATION_STATES,
        "branch_count": len(inventory),
        "rows": inventory,
    }
    receipt = _with_self_digest(payload, FRESH_CORPUS_SELF_KEY)
    _atomic_json(
        out_root / "fresh_calibration/corpus_receipt.json", receipt,
        out_root=out_root,
    )
    return receipt


# ------------------------------------------------------------ training view --
def _historical_calibration_disposition(
    states: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    calibration_states = sorted(
        (row for row in states if row.get("split_role") == "calibration"),
        key=lambda row: row["state_identity_digest"],
    )
    identities = [row["state_identity_digest"] for row in calibration_states]
    scenes = sorted(str(row["scene_id"]) for row in calibration_states)
    branch_count = sum(row.get("split_role") == "calibration" for row in rows)
    payload = {
        "state_count": len(calibration_states),
        "branch_count": branch_count,
        "status": "DEVELOPMENT_ONLY",
        "qualification_eligible": False,
        "discarded": False,
        "state_identity_digests": identities,
        "scene_ids": scenes,
    }
    if len(identities) != 24 or branch_count != 288:
        raise WorkflowError("historical calibration disposition inventory changed")
    return _with_self_digest(payload, "disposition_digest")


def _artifact_reference(path: Path, self_digest: str) -> dict[str, Any]:
    return {
        "path": _repo_relative(path),
        "sha256": file_sha256(path),
        "self_digest": self_digest,
    }


def _training_view_row(
    *, role: str, source_kind: str, state: Mapping[str, Any],
    input_path: Path, input_row: Mapping[str, Any], frame_root: Path,
    label_path: Path, label_source: Mapping[str, Any], label_self_digest: str,
) -> dict[str, Any]:
    labels = (
        _label_projection(label_source["labels"])
        if source_kind == SOURCE_KIND_REPLAY else _label_projection(label_source)
    )
    payload = {
        "schema": TRAINING_VIEW_ROW_SCHEMA,
        "role": role,
        "source_kind": source_kind,
        "state_id": state["state_id"],
        "state_identity_digest": state["state_identity_digest"],
        "scene_id": state["scene_id"],
        "family": state["family"],
        "stratum": state["stratum"],
        "candidate_index": input_row["candidate_index"],
        "branch_identity_digest": input_row["branch_identity_digest"],
        "input": _artifact_reference(
            input_path,
            str(input_row.get("branch_row_digest")
                or input_row.get(FRESH_BRANCH_ROW_SELF_KEY)),
        ),
        "frame_root": _repo_relative(frame_root),
        "label_source": _artifact_reference(label_path, label_self_digest),
        "label_projection": labels,
    }
    return _with_self_digest(payload, "training_view_row_digest")


def compose_training_view(*, out_root: Path = OUT_ROOT,
                          v2_root: Path = V2_ROOT) -> dict[str, Any]:
    authority, correction, corpus, fresh_manifest = \
        load_validated_fresh_selection(out_root=out_root, v2_root=v2_root)
    del correction
    equivalence = _read_regular_json(out_root / "equivalence_receipt.json")
    validate_equivalence_receipt(equivalence, authority, corpus["rows"])
    replay_manifest = _read_regular_json(out_root / "replay_overlay_manifest.json")
    _validate_self_digest(replay_manifest, REPLAY_OVERLAY_MANIFEST_SELF_KEY)
    if replay_manifest.get("overlay_count") != 18:
        raise WorkflowError("training view requires all eighteen replay overlays")
    overlays: dict[str, tuple[dict[str, Any], Path]] = {}
    for entry in replay_manifest["rows"]:
        path = ROOT / entry["path"]
        overlay = _read_regular_json(path)
        _validate_self_digest(overlay, REPLAY_OVERLAY_SELF_KEY)
        if file_sha256(path) != entry["sha256"]:
            raise WorkflowError("replay overlay raw digest changed")
        overlays[overlay["source_branch_identity_digest"]] = (overlay, path)
    fresh_receipt = _read_regular_json(out_root / "fresh_calibration/corpus_receipt.json")
    _validate_self_digest(fresh_receipt, FRESH_CORPUS_SELF_KEY)
    if fresh_receipt.get("branch_count") != EXPECTED_FRESH_CALIBRATION_BRANCHES:
        raise WorkflowError("fresh calibration corpus is incomplete")
    old_state_by_id = {
        row["state_id"]: row for row in corpus["state_manifest"]["states"]
    }
    view_rows = []
    for row in corpus["rows"]:
        if row["split_role"] != "fit":
            continue
        input_path = corpus["row_paths"][row["branch_identity_digest"]]
        if row["valid"] is True:
            view_rows.append(_training_view_row(
                role="fit", source_kind=SOURCE_KIND_V2_VALID,
                state=old_state_by_id[row["state_id"]], input_path=input_path,
                input_row=row, frame_root=v2_root, label_path=input_path,
                label_source=row, label_self_digest=row["branch_row_digest"],
            ))
        else:
            overlay_pair = overlays.get(row["branch_identity_digest"])
            if overlay_pair is None:
                raise WorkflowError("fit refusal lacks its exact replay overlay")
            overlay, overlay_path = overlay_pair
            view_rows.append(_training_view_row(
                role="fit", source_kind=SOURCE_KIND_REPLAY,
                state=old_state_by_id[row["state_id"]], input_path=input_path,
                input_row=row, frame_root=v2_root, label_path=overlay_path,
                label_source=overlay,
                label_self_digest=overlay[REPLAY_OVERLAY_SELF_KEY],
            ))
    fresh_state_by_id = {row["state_id"]: row for row in fresh_manifest["states"]}
    for entry in fresh_receipt["rows"]:
        path = ROOT / entry["path"]
        row = _read_regular_json(path)
        _validate_self_digest(row, FRESH_BRANCH_ROW_SELF_KEY)
        if file_sha256(path) != entry["sha256"]:
            raise WorkflowError("fresh calibration row raw digest changed")
        view_rows.append(_training_view_row(
            role="calibration", source_kind=SOURCE_KIND_FRESH,
            state=fresh_state_by_id[row["state_id"]], input_path=path,
            input_row=row, frame_root=out_root / "fresh_calibration",
            label_path=path, label_source=row,
            label_self_digest=row[FRESH_BRANCH_ROW_SELF_KEY],
        ))
    view_rows.sort(key=lambda row: (
        0 if row["role"] == "fit" else 1,
        row["state_identity_digest"], row["candidate_index"],
    ))
    role_counts = Counter(row["role"] for row in view_rows)
    source_counts = Counter(row["source_kind"] for row in view_rows)
    state_counts = {
        role: len({row["state_identity_digest"] for row in view_rows
                   if row["role"] == role})
        for role in ("fit", "calibration")
    }
    if (
        len(view_rows) != EXPECTED_TRAINING_ROWS
        or role_counts != Counter({"fit": 1_152, "calibration": 288})
        or state_counts != {"fit": 96, "calibration": 24}
        or source_counts != Counter({
            SOURCE_KIND_V2_VALID: EXPECTED_OLD_FIT_VALID,
            SOURCE_KIND_REPLAY: EXPECTED_OLD_FIT_OVERLAYS,
            SOURCE_KIND_FRESH: EXPECTED_FRESH_CALIBRATION_BRANCHES,
        })
        or len({row["branch_identity_digest"] for row in view_rows})
        != len(view_rows)
    ):
        raise WorkflowError("training view is not exact 1146+6+288")
    disposition = _historical_calibration_disposition(
        corpus["state_manifest"]["states"], corpus["rows"]
    )
    old_calibration_states = set(disposition["state_identity_digests"])
    old_calibration_scenes = set(disposition["scene_ids"])
    if any(
        row["state_identity_digest"] in old_calibration_states
        or row["scene_id"] in old_calibration_scenes
        for row in view_rows
    ):
        raise WorkflowError("historical calibration leaked into training view")
    payload = {
        "schema": TRAINING_VIEW_SCHEMA,
        "status": STATUS,
        "complete": True,
        "authority_digest": authority[AUTHORITY_SELF_KEY],
        "oracle_v1_3_digest": authority["oracle_v1_3_digest"],
        "scorer_fit_oracle_v1_3_contract_digest": authority[
            "scorer_fit_oracle_v1_3_contract_digest"
        ],
        "v2_corpus_digest": EXPECTED_V2_CORPUS_DIGEST,
        "equivalence_receipt_digest": equivalence[EQUIVALENCE_SELF_KEY],
        "replay_overlay_manifest_digest": replay_manifest[
            REPLAY_OVERLAY_MANIFEST_SELF_KEY
        ],
        "fresh_calibration_state_manifest_digest": fresh_manifest[
            FRESH_STATE_MANIFEST_SELF_KEY
        ],
        "fresh_calibration_corpus_digest": fresh_receipt[FRESH_CORPUS_SELF_KEY],
        "fit_state_count": 96, "fit_branch_count": 1_152,
        "calibration_state_count": 24, "calibration_branch_count": 288,
        "row_count": len(view_rows),
        "source_kind_counts": dict(sorted(source_counts.items())),
        "historical_calibration_disposition": disposition,
        "rows": view_rows,
        "missing_label_count": 0,
    }
    view = _with_self_digest(payload, TRAINING_VIEW_SELF_KEY)
    _atomic_json(out_root / "training_view.json", view, out_root=out_root)
    return view


def _validate_training_view_shape(view: Mapping[str, Any]) -> None:
    _validate_self_digest(view, TRAINING_VIEW_SELF_KEY)
    if (
        view.get("schema") != TRAINING_VIEW_SCHEMA
        or view.get("status") != STATUS
        or view.get("complete") is not True
        or view.get("row_count") != EXPECTED_TRAINING_ROWS
        or view.get("fit_state_count") != 96
        or view.get("fit_branch_count") != 1_152
        or view.get("calibration_state_count") != 24
        or view.get("calibration_branch_count") != 288
        or view.get("missing_label_count") != 0
        or view.get("scorer_fit_oracle_v1_3_contract_digest")
        != V13_CONTRACT.contract_digest()
        or len(view.get("rows", [])) != EXPECTED_TRAINING_ROWS
    ):
        raise WorkflowError("training view shape/count contract changed")
    disposition = view.get("historical_calibration_disposition", {})
    _validate_self_digest(disposition, "disposition_digest")
    if {
        key: disposition.get(key) for key in (
            "state_count", "branch_count", "status", "qualification_eligible",
            "discarded",
        )
    } != {
        "state_count": 24, "branch_count": 288,
        "status": "DEVELOPMENT_ONLY", "qualification_eligible": False,
        "discarded": False,
    }:
        raise WorkflowError("historical calibration disposition changed")
    rows = view.get("rows", [])
    for row in rows:
        if not isinstance(row, Mapping):
            raise WorkflowError("training-view row is malformed")
        _validate_self_digest(row, "training_view_row_digest")
        _label_projection(row.get("label_projection", {}))
    role_counts = Counter(row.get("role") for row in rows)
    source_counts = Counter(row.get("source_kind") for row in rows)
    state_counts = {
        role: len({row.get("state_identity_digest") for row in rows
                   if row.get("role") == role})
        for role in ("fit", "calibration")
    }
    if (
        role_counts != Counter({"fit": 1_152, "calibration": 288})
        or source_counts != Counter({
            SOURCE_KIND_V2_VALID: EXPECTED_OLD_FIT_VALID,
            SOURCE_KIND_REPLAY: EXPECTED_OLD_FIT_OVERLAYS,
            SOURCE_KIND_FRESH: EXPECTED_FRESH_CALIBRATION_BRANCHES,
        })
        or state_counts != {"fit": 96, "calibration": 24}
        or len({row.get("branch_identity_digest") for row in rows}) != len(rows)
        or any(
            row.get("state_identity_digest")
            in set(disposition["state_identity_digests"])
            or row.get("scene_id") in set(disposition["scene_ids"])
            for row in rows
        )
    ):
        raise WorkflowError("training-view row identity/source distribution changed")


def _resolve_bound_input(reference: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    path = ROOT / str(reference.get("path", ""))
    if not path.is_file() or path.is_symlink() or ROOT.absolute() not in path.absolute().parents:
        raise WorkflowError("training-view source path is unavailable or escapes repository")
    if file_sha256(path) != reference.get("sha256"):
        raise WorkflowError("training-view source raw digest changed")
    payload = _read_regular_json(path)
    expected = reference.get("self_digest")
    if payload.get("branch_row_digest") == expected:
        body = dict(payload)
        body.pop("branch_row_digest", None)
        if B.canonical_digest(body) != expected:
            raise WorkflowError("V2 training-view row self digest changed")
    elif payload.get(FRESH_BRANCH_ROW_SELF_KEY) == expected:
        _validate_self_digest(payload, FRESH_BRANCH_ROW_SELF_KEY)
    elif payload.get(REPLAY_OVERLAY_SELF_KEY) == expected:
        _validate_self_digest(payload, REPLAY_OVERLAY_SELF_KEY)
    else:
        raise WorkflowError("training-view source self digest changed")
    return payload, path


def _normalise_frame_records(records: Sequence[Mapping[str, Any]],
                             frame_root: Path) -> list[dict[str, Any]]:
    result = []
    for record in records:
        row = copy.deepcopy(dict(record))
        relative = Path(str(row.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise WorkflowError("frame path is not a safe relative path")
        path = (frame_root / relative).absolute()
        if ROOT.absolute() not in path.parents:
            raise WorkflowError("frame path escapes repository")
        row["path"] = _repo_relative(path)
        result.append(row)
    return result


def load_training_view_for_consumption(
    path: Path = TRAINING_VIEW_PATH, *, materialize: bool = True,
) -> dict[str, Any]:
    """Load the exact 1,440-row view; optionally materialise model inputs."""

    if path.absolute() != TRAINING_VIEW_PATH.absolute():
        raise WorkflowError("training view must be loaded from its registered path")
    view = _read_regular_json(path)
    _validate_training_view_shape(view)
    authority, correction, _corpus, manifest = load_validated_fresh_selection()
    del correction
    if (
        view.get("authority_digest") != authority[AUTHORITY_SELF_KEY]
        or view.get("oracle_v1_3_digest") != authority["oracle_v1_3_digest"]
        or view.get("scorer_fit_oracle_v1_3_contract_digest")
        != authority["scorer_fit_oracle_v1_3_contract_digest"]
        or view.get("fresh_calibration_state_manifest_digest")
        != manifest[FRESH_STATE_MANIFEST_SELF_KEY]
    ):
        raise WorkflowError("training view is bound to another v1.3 authority")
    if not materialize:
        return view
    material_rows = []
    for reference_row in view["rows"]:
        _validate_self_digest(reference_row, "training_view_row_digest")
        input_row, _input_path = _resolve_bound_input(reference_row["input"])
        label_source, _label_path = _resolve_bound_input(reference_row["label_source"])
        labels = (
            _label_projection(label_source["labels"])
            if reference_row["source_kind"] == SOURCE_KIND_REPLAY
            else _label_projection(label_source)
        )
        if labels != reference_row["label_projection"]:
            raise WorkflowError("materialised label differs from training-view projection")
        frame_root = ROOT / reference_row["frame_root"]
        normalised = {
            **{key: reference_row[key] for key in (
                "role", "source_kind", "state_id", "state_identity_digest",
                "scene_id", "family", "stratum", "candidate_index",
                "branch_identity_digest", "training_view_row_digest",
            )},
            "frame_root": reference_row["frame_root"],
            "context_frames": _normalise_frame_records(
                input_row["context_frames"], frame_root
            ),
            "horizon_frames": _normalise_frame_records(
                input_row["horizon_frames"], frame_root
            ),
            "action_blocks": input_row["action_blocks"],
            "action_context_blocks": input_row["action_context_blocks"],
            "previous_applied_command": input_row["previous_applied_command"],
            "goal_binding_input": input_row["goal_binding_input"],
            "proprio": input_row["proprio"],
            "control": input_row["control"],
            "masks": input_row["masks"],
            "timing": input_row["timing"],
            **labels,
        }
        material_rows.append(normalised)
    result = copy.deepcopy(view)
    result["reference_rows"] = result.pop("rows")
    result["rows"] = material_rows
    return result


# --------------------------------------------------------------------- CLI --
STAGES = (
    "issue-authority",
    "issue-replay-plan",
    "adopt-valid",
    "replay-failures",
    "issue-selector-integrity-replacement",
    "select-calibration",
    "generate-calibration",
    "compose-training-view",
    "status",
)
RUNTIME_STAGES = {
    "replay-failures", "select-calibration", "generate-calibration",
}


def status(*, out_root: Path = OUT_ROOT) -> dict[str, Any]:
    paths = {
        "authority": out_root / "authority.json",
        "replay_plan": out_root / "replay_plan.json",
        "equivalence": out_root / "equivalence_receipt.json",
        "replay_overlays": out_root / "replay_overlay_manifest.json",
        "selector_integrity_replacement":
            out_root / "fresh_calibration/selector_integrity_replacement_v1.json",
        "fresh_selection_attempt":
            out_root / "fresh_calibration/selection_attempt.json",
        "fresh_selection_terminal":
            out_root / "fresh_calibration/selection_terminal.json",
        "fresh_states": out_root / "fresh_calibration/state_manifest.json",
        "fresh_corpus": out_root / "fresh_calibration/corpus_receipt.json",
        "training_view": out_root / "training_view.json",
    }
    return {
        "schema": "go2_scorer_fit_oracle_v1_3_workflow_status_v1",
        "status": STATUS,
        "artifacts": {
            key: bool(path.is_file() and not path.is_symlink())
            for key, path in paths.items()
        },
        "runtime_running": False,
        "encoder_or_trainer_exposed": False,
        "predictor_or_final_benchmark_exposed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--backend", default="cpu", choices=("cpu",))
    parser.add_argument(
        "--fresh-selection-worker-task", type=Path, help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--fresh-selection-worker-result", type=Path, help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--execute-authorized", action="store_true",
        help="required only for the three simulator-backed stages",
    )
    args = parser.parse_args(argv)
    worker_paths = (
        args.fresh_selection_worker_task,
        args.fresh_selection_worker_result,
    )
    if any(path is not None for path in worker_paths):
        if (
            not all(path is not None for path in worker_paths)
            or args.stage is not None
            or args.execute_authorized
        ):
            raise WorkflowError("fresh-selection worker invocation is malformed")
        result = run_fresh_selection_worker(
            task_path=args.fresh_selection_worker_task,
            result_path=args.fresh_selection_worker_result,
            backend=args.backend,
        )
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.stage is None:
        parser.error("--stage is required")
    if args.stage in RUNTIME_STAGES and not args.execute_authorized:
        raise WorkflowError(
            f"{args.stage} is simulator-backed and requires --execute-authorized"
        )
    if args.stage not in RUNTIME_STAGES and args.execute_authorized:
        raise WorkflowError("--execute-authorized is invalid for a source/data-only stage")
    if args.stage == "issue-authority":
        result = issue_authority()
    elif args.stage == "issue-replay-plan":
        result = issue_replay_plan()
    elif args.stage == "adopt-valid":
        result = issue_equivalence_receipt()
    elif args.stage == "replay-failures":
        result = generate_replay_overlays(backend=args.backend)
    elif args.stage == "issue-selector-integrity-replacement":
        result = issue_selector_integrity_replacement()
    elif args.stage == "select-calibration":
        result = select_fresh_calibration_states(backend=args.backend)
    elif args.stage == "generate-calibration":
        result = generate_fresh_calibration(backend=args.backend)
    elif args.stage == "compose-training-view":
        result = compose_training_view()
    else:
        result = status()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
