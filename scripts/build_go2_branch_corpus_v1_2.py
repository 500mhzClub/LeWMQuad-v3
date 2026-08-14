#!/usr/bin/env python3
"""Branch corpora for the v1.2 utility scorer and the final evaluation.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.  No predictor checkpoint is opened here.

Two pools, one generator:

* ``scorer_fit``  120 states (15/family), 6 candidates each  -> 720 branches
* ``final_eval``  200 states (25/family), 12 candidates each -> 2400 branches

The snapshot mechanism, the canonical boundary, the candidate bank and oracle
v1.2 are imported unchanged.  What is new here is only the corpus: state
selection with strata, the frozen candidate rotation, the proprioceptive and
control history, and the textured_v03 renders at the three context slots and the
four horizons.

Stages::

    --stage states    resolve + freeze the identities (no branch, no render)
    --stage branches  execute, render and label every allocated branch
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from fnmatch import fnmatchcase
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "lewm_genesis", ROOT / "lewm_worlds", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import run_go2_oracle_branch_pilot_v1 as V1
import run_go2_oracle_branch_pilot_v1_2 as V12
from lewm.oracle.go2_branch_oracle_v1_2 import (
    GeodesicField, HORIZON_S, V_MAX_MPS,
    progress_digest, safety_digest, oracle_digest as v12_oracle_digest,
)
from lewm.oracle import go2_candidate_allocation_v1_2 as ALLOC
from lewm.oracle import go2_invalid_scorer_identity_exclusion_v1_2 as INVALID_IDS
from lewm.oracle import go2_parallel_small_completion_search_v1 as PARALLEL_SEARCH
from lewm.oracle import go2_scorer_projection_fix_interruption_v1 as INTERRUPTION
from lewm.oracle import go2_scorer_small_search_performance_interruption_v1 as PERFORMANCE_INTERRUPTION
from lewm.oracle import go2_scorer_state_selector_amendment_v2 as STATE_SELECTOR
from lewm.oracle.go2_textured_v03_renderer import (
    BasePose,
    TexturedV03Renderer,
    capture_base_pose,
    renderer_contract_digest as textured_v03_renderer_contract_digest,
)
from lewm.oracle.go2_scorer_contract_v1_2 import (
    CORPUS_SELECTION_CONTRACT,
    SUPERSEDED_PRE_RUN_CONTRACT_ARTIFACT,
    TARGET_ENCODER,
    clean_source_binding,
    contract as scorer_contract,
    contract_digest as scorer_contract_digest,
    preprocess_contract_digest,
    render_contract_digest,
    target_encoder_digest,
)
from scripts import dev_action_slew_reconstruction_v1 as SLEW
from scripts import build_dev_v03_proprio_action_manifest_v1 as M

STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
OUT_ROOT = ROOT / ".generated/go2_branch_corpus_v1_2"
CORPUS = V1.CORPUS
SPLITS = ("train", "val", "test_id", "test_hard")

CONTEXT_SLOTS = 3
SAMPLES_PER_SLOT = M.SAMPLES_PER_SLOT           # 5
PROPRIO_HISTORY = CONTEXT_SLOTS * SAMPLES_PER_SLOT   # 15 trailing 10 Hz samples
HORIZONS = 4
EXPECTED_FAMILIES = 8
FACTORIAL_ROOT = Path(
    "/home/andrewknowles/.cache/lewm_go2_temporal_v03/proprio_v1"
)
FACTORIAL_MANIFEST = FACTORIAL_ROOT / "factorial_manifest.json"
FACTORIAL_MANIFEST_FILE_SHA256 = (
    "8bf59020d24e02fdb11948f3732220df839aa1c3bc8612392ce6baab6b8d629c"
)
FACTORIAL_MANIFEST_DIGEST = (
    "6ff053033475debd3d8bb415080efb15adfaefc31f01295b956bd85c12b6dac0"
)
FACTORIAL_ROWS = FACTORIAL_ROOT / "proprio_rows.jsonl"
FACTORIAL_ROWS_SHA256 = (
    "7b79d12830f12175c591a87982a20e5df7a8d64cfc40e99dd9cee2dc1ae2543e"
)
V11_IDENTITY_MANIFEST_DIGEST = (
    "015eb0bb4ccb9da28ce4b055771975fc68ac0c986e462d9c3af0a61ef45a9ea2"
)
V12_IDENTITY_MANIFEST_DIGEST = (
    "5f380bf7f49ef10437c7d9644f04dbef065f0550dfd30d0ec36208cda25d08cf"
)

# ---- frozen strata (scorer-fit only), snapshot-time geometry only -------------
STRATA = ("general", "safety_enriched", "completion_enriched")
SAFETY_ENRICHED_MAX_BODY_CLEARANCE_M = 0.10
COMPLETION_ENRICHED_MAX_GEODESIC_M = STATE_SELECTOR.COMPLETION_MAX_GEODESIC_M
COMPLETION_ENRICHED_MAX_BEARING_RAD = math.radians(
    STATE_SELECTOR.COMPLETION_MAX_ABS_BEARING_DEG)
COMPLETION_HORIZON_S = STATE_SELECTOR.HORIZON_S
COMPLETION_MAX_TRANSLATION_M = STATE_SELECTOR.MAX_TRANSLATION_M
if (COMPLETION_HORIZON_S != HORIZON_S
        or STATE_SELECTOR.V_MAX_MPS != V_MAX_MPS
        or tuple(STATE_SELECTOR.SCORER_FIT_SELECTION_PRIORITY) != STRATA):
    raise RuntimeError("state-selector amendment changed a preserved oracle binding")

POOLS = {
    "scorer_fit": {
        "states_per_family": 15, "candidates_per_state": 6,
        "strata": {"general": 5, "safety_enriched": 5, "completion_enriched": 5},
        "calibration_per_stratum_per_family": 1,
    },
    "final_eval": {
        "states_per_family": 25, "candidates_per_state": 12,
        "strata": None,
        "calibration_per_stratum_per_family": 0,
    },
}

SELECTION = dict(CORPUS_SELECTION_CONTRACT)
WARMUP_BLOCKS_MIN, WARMUP_BLOCKS_MAX = SELECTION["warmup_blocks"]
PRE_IDENTITY_VALIDATION_NAME = "pre_identity_allocation_validation.json"
LAUNCH_RECEIPT_NAME = "clean_source_launch_receipt.json"
SELECTOR_FEASIBILITY_SCHEMA = (
    "go2_scorer_fit_state_selector_feasibility_receipt_v1"
)
SELECTOR_FEASIBILITY_RECEIPT_NAME = "state_selector_feasibility_receipt.json"
SELECTOR_FEASIBILITY_PASS_STATUS = "PASS_OUTCOME_FREE_ALL_SCENE_FEASIBILITY"
SELECTOR_FEASIBILITY_REDUCER_VERSION = (
    "go2_scorer_fit_state_selector_feasibility_scene_isolated_reducer_v1"
)
SELECTOR_FEASIBILITY_TASK_CENSUS_SCHEMA = (
    "go2_scorer_fit_state_selector_feasibility_task_census_v1"
)
SELECTOR_FEASIBILITY_TASK_CENSUS_NAME = (
    "state_selector_feasibility_task_census.json"
)
SELECTOR_FEASIBILITY_SCENE_SHARD_SCHEMA = (
    "go2_scorer_fit_state_selector_feasibility_scene_shard_v1"
)
SELECTOR_FEASIBILITY_SCENE_SHARD_STATUS = (
    "COMPLETE_OUTCOME_FREE_SCENE_CENSUS_NO_ELIGIBILITY_VERDICT"
)
SELECTOR_FEASIBILITY_SCENE_SHARD_ROOT = (
    "state_selector_feasibility_scene_shards"
)
# The exhaustive v1 census is frozen evidence.  The final, prospective
# reachability amendment never overwrites it: seven families can be certified
# from their already-inside-radius completion rows, while only the one family
# whose old completion count was zero is redriven to recover the snapshot-time
# inputs that v1 did not retain.
REACHABILITY_FEASIBILITY_SCHEMA = STATE_SELECTOR.STATE_SELECTOR_FEASIBILITY_SCHEMA
REACHABILITY_FEASIBILITY_RECEIPT_NAME = (
    STATE_SELECTOR.STATE_SELECTOR_FEASIBILITY_RECEIPT_NAME
)
REACHABILITY_FEASIBILITY_PASS_STATUS = (
    STATE_SELECTOR.STATE_SELECTOR_FEASIBILITY_PASS_STATUS
)
REACHABILITY_FEASIBILITY_SCENE_SHARD_SCHEMA = (
    "go2_scorer_fit_state_selector_reachability_scene_shard_v2"
)
REACHABILITY_FEASIBILITY_SCENE_SHARD_STATUS = (
    "COMPLETE_OUTCOME_FREE_REACHABILITY_SCENE_EVIDENCE_NO_IDENTITY"
)
REACHABILITY_FEASIBILITY_SCENE_SHARD_ROOT = (
    "state_selector_reachability_feasibility_scene_shards_v2"
)
REACHABILITY_REDRIVE_FAMILY = "small_enclosed_maze"
SMALL_COMPLETION_SEARCH_FAILURE_SCHEMA = (
    "go2_scorer_fit_small_completion_joint_search_preoutcome_failure_v2"
)
SMALL_COMPLETION_SEARCH_FAILURE_STATUS = (
    "FAIL_PREOUTCOME_SMALL_COMPLETION_JOINT_SEARCH"
)
SMALL_COMPLETION_SEARCH_FAILURE_NAME = (
    "small_completion_joint_search_preoutcome_failure_v2.json"
)
PARALLEL_SMALL_SEARCH_PLAN_NAME = (
    "small_completion_parallel_search_plan_v1.json"
)
PARALLEL_SMALL_BENCHMARK_NAME = (
    "small_completion_parallel_prefix_benchmark_v1.json"
)
PARALLEL_SMALL_CHECKPOINT_ROOT = "small_completion_parallel_search_v1"
PARALLEL_SMALL_TERMINAL_RESULT_NAME = (
    "small_completion_parallel_terminal_result_v1.json"
)
PARALLEL_SMALL_JOINT_RECEIPT_NAME = (
    "small_completion_parallel_joint_receipt_v1.json"
)
PARALLEL_SMALL_TERMINAL_FAILURE_NAME = (
    "small_completion_parallel_terminal_failure_v1.json"
)
PARALLEL_SMALL_JOINT_RECEIPT_SCHEMA = (
    "go2_branch_corpus_v1_2_parallel_small_completion_joint_receipt_v1"
)
PARALLEL_SMALL_FAILURE_SCHEMA = (
    "go2_branch_corpus_v1_2_parallel_small_completion_terminal_failure_v1"
)
PARALLEL_SMALL_WORKER_COUNT = 32
PARALLEL_SMALL_ACTIVE_RANK_WINDOW = 3
PARALLEL_SMALL_BENCHMARK_MAXIMUM_FRACTION = 0.5
STATE_RESOLUTION_SCENE_REQUEST_SCHEMA = (
    "go2_branch_corpus_v1_2_state_resolution_scene_request_v1"
)
STATE_RESOLUTION_SCENE_CAPTURE_SCHEMA = (
    "go2_branch_corpus_v1_2_state_resolution_scene_capture_v1"
)
STATE_RESOLUTION_SCENE_REQUEST_ROOT = "state_resolution_scene_requests_v1"
STATE_RESOLUTION_SCENE_CAPTURE_ROOT = "state_resolution_scene_captures_v1"
STATE_RESOLUTION_REDUCER_CONTRACT = {
    "schema": "go2_branch_corpus_v1_2_state_resolution_reducer_contract_v1",
    "scene_order": "frozen scene_pool order (lexical scene id)",
    "block_order": "ascending block 1 through WARMUP_BLOCKS_MAX",
    "eligible_block_start": "WARMUP_BLOCKS_MIN",
    "stratum_priority": list(STRATA),
    "dynamic_quota_rule": "skip strata whose family quota is already full",
    "within_scene_rule": (
        "at each eligible block select the first passing unmet stratum; "
        "select at most one state per scene"
    ),
    "rejection_ledger_rule": (
        "only a block with no selected stratum contributes the ordered "
        "stratum:reason-key conjunction"
    ),
    "candidate_outcomes_consumed": False,
}
STATE_RESOLUTION_SELECTION_SEMANTICS = (
    "lexical scene order; first eligible block; first unmet frozen stratum "
    "in general,safety_enriched,completion_enriched order; at most one state "
    "per scene"
)
FROZEN_FEASIBILITY_RECEIPT_DIGEST = STATE_SELECTOR.FROZEN_FAILED_CENSUS_RECEIPT[
    "state_selector_feasibility_receipt_digest"]
FROZEN_FEASIBILITY_RECEIPT_RAW_SHA256 = (
    STATE_SELECTOR.FROZEN_FAILED_CENSUS_RECEIPT["raw_sha256"]
)
FROZEN_FEASIBILITY_TASK_CENSUS_DIGEST = (
    "0ee5fb6d073e6e8db33b0f63ce9b70b8346ba12f29f729f06c06de5982fbe109"
)
FROZEN_FEASIBILITY_SCENE_SHARD_COUNT = int(
    STATE_SELECTOR.FROZEN_FAILED_CENSUS_RECEIPT["scene_shards_expected"])
FROZEN_FEASIBILITY_FAILURE_REPORT_PATH = ROOT / STATE_SELECTOR.FAILURE_REPORT_PATH
FROZEN_FEASIBILITY_FAILURE_REPORT_DIGEST = STATE_SELECTOR.FAILURE_REPORT_DIGEST
FROZEN_FEASIBILITY_FAILURE_REPORT_RAW_SHA256 = (
    STATE_SELECTOR.FAILURE_REPORT_RAW_SHA256
)
SELECTOR_FEASIBILITY_FORBIDDEN_FIELDS = (
    "selected_state_identities_created", "candidate_outcomes_loaded",
    "branch_identities_created", "branches_attempted", "frames_rendered",
    "target_latents_encoded", "scorer_training_started",
)
SCORER_CONTRACT_ARTIFACT_PATH = (
    ROOT / ".generated/go2_utility_scorer_v1_2/scorer_contract_v1_2.json"
)
LAUNCH_BINDING_KEYS = (
    "clean_source_launch_receipt_digest",
    "source_repository_commit",
    "clean_source_binding_digest",
    "bound_implementations_digest",
    "scorer_contract_artifact_digest",
    "mixed_precontract_disposition_receipt_digest",
)
ACTIVE_SELECTOR_BINDING_KEYS = tuple(STATE_SELECTOR.ACTIVE_SELECTOR_BINDING_KEYS)
STATE_SHARD_COMMON_KEYS = (
    "selection_digest", "scorer_fit_allocation_design_digest",
    "candidate_allocator_contract_digest",
    "candidate_allocation_amendment_digest",
    "pre_identity_allocation_validation_digest",
    "invalid_scorer_identity_exclusion_digest",
    "state_selector_amendment_digest",
    "state_selector_feasibility_receipt_digest", "candidate_bank_digest",
    *LAUNCH_BINDING_KEYS,
    "progress_contract_digest", "safety_contract_digest",
    "oracle_v1_2_digest", "scorer_contract_v1_2_digest", "boundary_digest",
    "render_contract_digest", "preprocess_contract_digest",
    "textured_v03_renderer_contract_digest", "preprocessing_digest",
    "target_encoder_digest", "target_encoder_checkpoint_sha256",
    "genesis_backend",
)

MIXED_ACTIVE_STATE_SHARD_SCHEMA = (
    "go2_branch_corpus_v1_2_mixed_active_state_shard_v2"
)
MIXED_ACTIVE_STATE_SHARD_NAME = (
    "active_mixed_state_shard_{family}_reachability_v2.json"
)
MIXED_REPLACEMENT_TRANSPORT_SCHEMA = (
    "go2_scorer_fit_mixed_preoutcome_replacement_transport_v2"
)
MIXED_REPLACEMENT_SCENE_REQUEST_SCHEMA = (
    "go2_scorer_fit_mixed_preoutcome_replacement_scene_request_v2"
)
MIXED_REPLACEMENT_SCENE_CAPTURE_SCHEMA = (
    "go2_scorer_fit_mixed_preoutcome_replacement_scene_capture_v2"
)
MIXED_REPLACEMENT_SCENE_REQUEST_ROOT = (
    "mixed_preoutcome_replacement_scene_requests_v2"
)
MIXED_REPLACEMENT_SCENE_CAPTURE_ROOT = (
    "mixed_preoutcome_replacement_scene_captures_v2"
)
MIXED_REPLACEMENT_FAILURE_SCHEMA = (
    "go2_scorer_fit_mixed_preoutcome_replacement_failure_v2"
)
MIXED_REPLACEMENT_FAILURE_STATUS = (
    "FAIL_PREOUTCOME_MIXED_REPLACEMENT_INTERVAL_EXHAUSTED"
)
MIXED_REPLACEMENT_FAILURE_NAME = (
    "mixed_preoutcome_replacement_failure_v2.json"
)


class SmallCompletionJointSearchInfeasible(RuntimeError):
    """The frozen one-pass completion pool cannot yield five valid identities."""

    def __init__(self, reason: str, *, attempt_count: int,
                 allocator_infeasible_count: int,
                 candidate_scene_ids: Sequence[str]):
        super().__init__(reason)
        self.reason = str(reason)
        self.attempt_count = int(attempt_count)
        self.allocator_infeasible_count = int(allocator_infeasible_count)
        self.candidate_scene_ids = [str(value) for value in candidate_scene_ids]


def selection_digest() -> str:
    # The scorer contract, not a mutually generated artifact, freezes selection.
    return str(scorer_contract()["corpus_selection_digest"])


def canonical_digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(V1._jsonable(payload), sort_keys=True).encode()
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(V1._jsonable(payload), indent=2,
                                    sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def _assert_unsealed_path(path: Path) -> None:
    """Reject sealed custody names and symlinks before path traversal."""

    if any(part == ".." or part == "sealed_test.json" or part == "sealed"
           or part.startswith("sealed_") for part in path.parts):
        raise RuntimeError("sealed benchmark paths are inaccessible")
    absolute = path if path.is_absolute() else Path.cwd() / path
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise RuntimeError("symlinked corpus paths are inaccessible")


def _frozen_generated_artifact_path(
        path: Path, *, generated_root: Path | None = None) -> Path:
    """Return one custody-checked canonical frozen-artifact path.

    Large generated outputs may be relocated behind the single managed
    ``OUT_ROOT`` alias.  That storage alias is not a corpus/scene traversal
    authority: this helper permits exactly that one root symlink, returns the
    resolved target path so a later alias swap cannot redirect the read, and
    rejects every symlink below it.  The ordinary corpus guard above remains
    deliberately stricter and is never relaxed by this helper.
    """

    root = OUT_ROOT if generated_root is None else Path(generated_root)
    absolute_root = root if root.is_absolute() else Path.cwd() / root
    absolute_path = path if path.is_absolute() else Path.cwd() / path
    for label, value in (("generated root", absolute_root),
                         ("generated artifact", absolute_path)):
        if any(part == ".." or part == "sealed_test.json" or part == "sealed"
               or part.startswith("sealed_") for part in value.parts):
            raise RuntimeError(
                f"{label} crosses an inaccessible custody component")
    try:
        relative = absolute_path.relative_to(absolute_root)
    except ValueError as exc:
        raise RuntimeError(
            "frozen generated artifact escaped the managed output root") from exc
    if not relative.parts:
        raise RuntimeError("frozen generated artifact path names only its root")

    # No alias is allowed before the one exact generated-output root.
    _assert_unsealed_path(absolute_root.parent)
    if absolute_root.is_symlink():
        raw_target = absolute_root.readlink()
        target = (raw_target if raw_target.is_absolute()
                  else absolute_root.parent / raw_target)
        if (target.name != absolute_root.name
                or any(part == ".." or part == "sealed_test.json"
                       or part == "sealed" or part.startswith("sealed_")
                       for part in target.parts)):
            raise RuntimeError(
                "managed generated-output alias target identity is inaccessible")
        _assert_unsealed_path(target)
        try:
            canonical_root = target.resolve(strict=True)
        except OSError as exc:
            raise RuntimeError(
                "managed generated-output alias target is missing") from exc
    else:
        if not absolute_root.is_dir():
            raise RuntimeError("managed generated-output root is missing")
        canonical_root = absolute_root.resolve(strict=True)
    if (not canonical_root.is_dir()
            or canonical_root.name != absolute_root.name):
        raise RuntimeError("managed generated-output root identity changed")
    _assert_unsealed_path(canonical_root)

    canonical_path = canonical_root.joinpath(*relative.parts)
    # This checks every descendant component without following an alias and
    # therefore rejects a symlinked receipt, census, family directory, or shard.
    _assert_unsealed_path(canonical_path)
    return canonical_path


def _load_pre_identity_allocation_validation() -> dict[str, Any]:
    raw_path = OUT_ROOT / "scorer_fit" / PRE_IDENTITY_VALIDATION_NAME
    path = _pin_generated_path(raw_path, raw_path)
    if not path.is_file():
        raise RuntimeError(
            "state identity selection is gated on the frozen pre-identity "
            "allocation validation artifact"
        )
    artifact = json.loads(path.read_text())
    ALLOC.validate_pre_identity_structural_validation(artifact)
    return artifact


def _issued_scorer_contract_path() -> Path:
    """Pin the exact registered utility-scorer artifact root.

    The corpus and utility-scorer generated roots are two distinct managed
    aliases.  Callers must name this exact artifact and retain the returned
    canonical path for every subsequent byte operation so an alias swap cannot
    redirect a later read or digest.
    """

    return _pin_generated_path(
        SCORER_CONTRACT_ARTIFACT_PATH, SCORER_CONTRACT_ARTIFACT_PATH,
        generated_root=SCORER_CONTRACT_ARTIFACT_PATH.parent)


def _load_issued_scorer_contract_at_path(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("clean-source scorer contract must be issued before preflight")
    artifact = json.loads(path.read_text())
    _verify_self_digest(artifact, "contract_artifact_digest", "scorer contract artifact")
    if (artifact.get("complete") is not True
            or artifact.get("scorer_contract_v1_2_digest")
            != scorer_contract_digest()
            or artifact.get("source_repository_clean") is not True):
        raise RuntimeError("issued scorer contract is not the current clean-source contract")
    current_source = clean_source_binding()
    if (artifact.get("clean_source_binding") != current_source
            or artifact.get("clean_source_binding_digest")
            != canonical_digest(current_source)):
        raise RuntimeError("issued scorer contract source binding differs from current HEAD")
    interruption = INTERRUPTION.load_and_validate_interruption_receipt(
        expected_source_repository_commit=str(
            current_source["source_repository_commit"]),
        expected_clean_source_binding_digest=canonical_digest(current_source),
        expected_bound_implementations_digest=str(
            current_source["bound_implementations_digest"]),
        root=ROOT,
    )
    if artifact.get("preoutcome_projection_fix_interruption") != \
            INTERRUPTION.receipt_binding(interruption, root=ROOT):
        raise RuntimeError(
            "issued scorer contract lost projection-fix interruption lineage"
        )
    performance_interruption = (
        PERFORMANCE_INTERRUPTION
        .load_and_validate_performance_interruption_receipt(
            expected_source_repository_commit=str(
                current_source["source_repository_commit"]),
            expected_clean_source_binding_digest=canonical_digest(current_source),
            expected_bound_implementations_digest=str(
                current_source["bound_implementations_digest"]),
            root=ROOT,
        )
    )
    if artifact.get("preoutcome_small_search_performance_interruption") != \
            PERFORMANCE_INTERRUPTION.receipt_binding(
                performance_interruption, root=ROOT):
        raise RuntimeError(
            "issued scorer contract lost small-search performance interruption lineage"
        )
    return artifact


def _load_issued_scorer_contract() -> dict[str, Any]:
    return _load_issued_scorer_contract_at_path(
        _issued_scorer_contract_path())


def _build_clean_source_launch_receipt(
        pre_identity: dict[str, Any]) -> dict[str, Any]:
    scorer_artifact_path = _issued_scorer_contract_path()
    scorer_artifact = _load_issued_scorer_contract_at_path(
        scorer_artifact_path)
    source = scorer_artifact["clean_source_binding"]
    selector_receipts = _load_state_selector_preconditions(
        source_commit=str(source["source_repository_commit"]),
        successor_selection_digest=selection_digest(),
        clean_source_binding_digest=scorer_artifact["clean_source_binding_digest"],
        bound_implementations_digest=source["bound_implementations_digest"],
    )
    if (scorer_artifact.get("state_selector_feasibility_receipt_digest")
            != selector_receipts["state_selector_feasibility_receipt_digest"]
            or scorer_artifact.get(
                "mixed_precontract_disposition_receipt_digest")
            != selector_receipts[
                "mixed_precontract_disposition_receipt_digest"]):
        raise RuntimeError(
            "issued scorer contract differs from active selector receipts"
        )
    receipt = {
        "schema": "go2_utility_scorer_v1_2_clean_source_launch_receipt",
        "status": STATUS,
        "complete": True,
        "source_repository_commit": source["source_repository_commit"],
        "source_repository_clean": True,
        "clean_source_binding_digest":
            scorer_artifact["clean_source_binding_digest"],
        "bound_implementations_digest": source["bound_implementations_digest"],
        "scorer_contract_v1_2_digest": scorer_contract_digest(),
        "scorer_contract_artifact_digest":
            scorer_artifact["contract_artifact_digest"],
        "scorer_contract_artifact_sha256":
            file_sha256(scorer_artifact_path),
        "candidate_allocation_amendment_digest":
            ALLOC.allocation_amendment_digest(),
        "invalid_scorer_identity_exclusion_digest":
            INVALID_IDS.invalid_identity_exclusion_digest(),
        "state_selector_amendment_digest":
            STATE_SELECTOR.state_selector_amendment_digest(),
        "preoutcome_projection_fix_interruption": dict(
            scorer_artifact["preoutcome_projection_fix_interruption"]),
        "preoutcome_small_search_performance_interruption": dict(
            scorer_artifact[
                "preoutcome_small_search_performance_interruption"]),
        **selector_receipts,
        "pre_identity_allocation_validation_digest":
            pre_identity["pre_identity_validation_digest"],
    }
    receipt["clean_source_launch_receipt_digest"] = canonical_digest(receipt)
    return receipt


def _load_clean_source_launch_receipt() -> dict[str, Any]:
    raw_path = OUT_ROOT / "scorer_fit" / LAUNCH_RECEIPT_NAME
    path = _pin_generated_path(raw_path, raw_path)
    if not path.is_file():
        raise RuntimeError("state identity selection requires a clean-source launch receipt")
    receipt = json.loads(path.read_text())
    _verify_self_digest(
        receipt, "clean_source_launch_receipt_digest", "clean-source launch receipt")
    expected = _build_clean_source_launch_receipt(
        _load_pre_identity_allocation_validation())
    if receipt != expected:
        raise RuntimeError("clean-source launch receipt differs from current clean HEAD")
    return receipt


def issue_pre_identity_allocation_validation(out: Path) -> int:
    """Issue the deterministic structural table before any state identity."""

    if out.name != "scorer_fit":
        raise RuntimeError("allocation preflight is defined only for scorer_fit")
    # Verifies clean git HEAD, exact bound source bytes and the issued contract
    # before any pre-identity artifact is retained or created.
    _load_issued_scorer_contract()
    amendment_path = ROOT / ALLOC.AMENDMENT_ARTIFACT_PATH
    ALLOC.validate_allocation_amendment_artifact(
        json.loads(amendment_path.read_text())
    )
    artifact = ALLOC.build_pre_identity_structural_validation()
    path = out / PRE_IDENTITY_VALIDATION_NAME
    retained = False
    if path.is_file():
        try:
            existing = json.loads(path.read_text())
            ALLOC.validate_pre_identity_structural_validation(existing)
            if existing == artifact:
                retained = True
        except Exception:
            pass
        if not retained and _outcome_generation_started(out):
            raise RuntimeError(
                "pre-identity allocation validation changed after outcomes started"
            )
        if not retained:
            _preserve_invalid(path, out, "pre-identity-validation-mismatch")
    if not retained:
        atomic_json(path, artifact)

    launch = _build_clean_source_launch_receipt(artifact)
    launch_path = out / LAUNCH_RECEIPT_NAME
    if launch_path.is_file():
        existing_launch = json.loads(launch_path.read_text())
        if existing_launch != launch:
            if _outcome_generation_started(out):
                raise RuntimeError("clean-source launch binding changed after outcomes started")
            _preserve_invalid(launch_path, out, "clean-source-launch-mismatch")
    atomic_json(launch_path, launch)
    print(json.dumps({
        "recovery": ("retained_valid_pre_identity_validation" if retained
                     else "issued_pre_identity_validation"),
        "path": str(path),
        "clean_source_launch_receipt_path": str(launch_path),
        "clean_source_launch_receipt_digest":
            launch["clean_source_launch_receipt_digest"],
        "source_repository_commit": launch["source_repository_commit"],
        "pre_identity_validation_digest":
            artifact["pre_identity_validation_digest"],
        "state_slots": artifact["global"]["state_slot_count"],
        "candidate_slots": artifact["global"]["candidate_slot_count"],
        "goal_type_validation_status":
            artifact["goal_type_validation"]["status"],
    }, indent=2, sort_keys=True))
    return 0


def _verify_self_digest(payload: dict[str, Any], key: str, label: str) -> None:
    expected = canonical_digest({name: value for name, value in payload.items()
                                 if name != key})
    if payload.get(key) != expected:
        raise RuntimeError(f"{label} self digest mismatch")


def _load_state_selector_preconditions(
        *, source_commit: str, successor_selection_digest: str,
        clean_source_binding_digest: str | None = None,
        bound_implementations_digest: str | None = None,
        ) -> dict[str, str]:
    """Load and validate the outcome-free pre-identity feasibility gate."""

    raw_feasibility_path = (
        ROOT / STATE_SELECTOR.STATE_SELECTOR_FEASIBILITY_RECEIPT_PATH)
    raw_disposition_path = (
        ROOT
        / STATE_SELECTOR.PRESERVED_STATE_MIXED_PRECONTRACT_DISPOSITION_RECEIPT_PATH
    )
    feasibility_path = _pin_generated_path(
        raw_feasibility_path, raw_feasibility_path)
    disposition_path = _pin_generated_path(
        raw_disposition_path, raw_disposition_path)
    if not feasibility_path.is_file():
        raise RuntimeError("state-selector all-family feasibility receipt is missing")
    if not disposition_path.is_file():
        raise RuntimeError(
            "preserved-state mixed precontract disposition is missing"
        )
    feasibility = STATE_SELECTOR.validate_frozen_reachability_feasibility_pass(
        root=ROOT)
    if json.loads(feasibility_path.read_text()) != feasibility:
        raise RuntimeError("active feasibility bytes differ from frozen PASS")
    disposition = json.loads(disposition_path.read_text())
    STATE_SELECTOR.validate_preserved_state_mixed_precontract_disposition_receipt(
        disposition,
        expected_source_commit=source_commit,
        expected_successor_selection_digest=successor_selection_digest,
        expected_clean_source_binding_digest=clean_source_binding_digest,
        expected_bound_implementations_digest=bound_implementations_digest,
        root=ROOT,
    )
    feasibility_digest = str(
        feasibility["state_selector_feasibility_receipt_digest"]
    )
    return {
        "state_selector_feasibility_receipt_digest": feasibility_digest,
        "mixed_precontract_disposition_receipt_digest": str(
            disposition["mixed_precontract_disposition_receipt_digest"]
        ),
    }


@lru_cache(maxsize=1)
def _preserved_states_by_digest() -> dict[str, dict[str, Any]]:
    shards = STATE_SELECTOR.load_preserved_state_shards(ROOT)
    states = {
        str(state["state_identity_digest"]): dict(state)
        for shard in shards.values() for state in shard["states"]
    }
    if len(states) != 45:
        raise RuntimeError("preserved predecessor state identity count changed")
    return states


def _state_identity_matches_active_or_preserved(state: dict[str, Any]) -> bool:
    digest = str(state.get("state_identity_digest", ""))
    preserved = _preserved_states_by_digest().get(digest)
    if preserved is not None:
        return _state_identity_payload(state) == _state_identity_payload(preserved)
    return _state_identity_digest(state) == digest


def _load_active_mixed_disposition() -> dict[str, Any]:
    """Reopen the active 37/8 authority under the current clean source."""

    raw_path = (
        ROOT
        / STATE_SELECTOR.PRESERVED_STATE_MIXED_PRECONTRACT_DISPOSITION_RECEIPT_PATH
    )
    path = _pin_generated_path(raw_path, raw_path)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("active mixed precontract disposition is missing")
    payload = json.loads(path.read_text())
    source = clean_source_binding()
    STATE_SELECTOR.validate_preserved_state_mixed_precontract_disposition_receipt(
        payload,
        expected_source_commit=str(source["source_repository_commit"]),
        expected_successor_selection_digest=selection_digest(),
        expected_clean_source_binding_digest=canonical_digest(source),
        expected_bound_implementations_digest=str(
            source["bound_implementations_digest"]),
        root=ROOT,
    )
    return payload


def _mixed_disposition_sets() -> tuple[
        dict[str, dict[str, Any]], dict[str, dict[str, Any]],
        dict[str, dict[str, Any]]]:
    receipt = _load_active_mixed_disposition()
    retained = {
        str(row["state_identity_digest"]): dict(row)
        for row in receipt["retained_predecessor_identities"]
    }
    rejected = {
        str(row["state_identity_digest"]): dict(row)
        for row in receipt["rejected_predecessor_identities"]
    }
    slots = {str(row["state_id"]): dict(row)
             for row in receipt["replacement_slots"]}
    if len(retained) != 37 or len(rejected) != 8 or len(slots) != 8:
        raise RuntimeError("active mixed disposition count changed")
    return retained, rejected, slots


def _completion_state_ordinal(state_id: str) -> int:
    prefix, separator, ordinal = str(state_id).rpartition("-")
    if (not separator or not prefix.endswith("-completion_enriched")
            or len(ordinal) != 2 or not ordinal.isdigit()):
        raise RuntimeError(f"invalid completion state slot {state_id!r}")
    value = int(ordinal)
    if not 0 <= value < 5:
        raise RuntimeError(f"completion state slot is out of range: {state_id!r}")
    return value


def _mixed_family_replacement_plan(family: str) -> dict[str, Any]:
    """Derive exact vacant ordinal intervals from retained lexical anchors."""

    preserved = STATE_SELECTOR.load_preserved_state_shards(ROOT)
    if family not in preserved:
        raise RuntimeError(f"family {family!r} has no predecessor state shard")
    retained_rows, rejected_rows, slot_rows = _mixed_disposition_sets()
    source_states = {
        str(state["state_identity_digest"]): dict(state)
        for state in preserved[family]["states"]
    }
    retained_states = [
        source_states[identity] for identity in retained_rows
        if identity in source_states
    ]
    family_slots = sorted(
        (dict(row) for row in slot_rows.values() if row["family"] == family),
        key=lambda row: _completion_state_ordinal(str(row["state_id"])),
    )
    family_rejected = {
        identity: row for identity, row in rejected_rows.items()
        if row["family"] == family
    }
    if not family_slots:
        raise RuntimeError(f"family {family!r} has no authorized replacement slots")
    completion_by_ordinal = {
        _completion_state_ordinal(str(state["state_id"])): state
        for state in retained_states
        if state["stratum"] == "completion_enriched"
    }
    if len(completion_by_ordinal) + len(family_slots) != 5:
        raise RuntimeError(
            f"family {family!r} replacement and retained completion slots do not total five"
        )
    retained_anchor_rows = [{
        "ordinal": ordinal,
        "state_id": str(state["state_id"]),
        "scene_id": str(state["scene_id"]),
        "state_identity_digest": str(state["state_identity_digest"]),
    } for ordinal, state in sorted(completion_by_ordinal.items())]
    anchor_scene_ids = [row["scene_id"] for row in retained_anchor_rows]
    if anchor_scene_ids != sorted(anchor_scene_ids):
        raise RuntimeError(
            f"family {family!r} retained completion anchors are not lexical"
        )

    groups: list[dict[str, Any]] = []
    vacant_ordinals = sorted(
        _completion_state_ordinal(str(row["state_id"])) for row in family_slots
    )
    for _unused, ordinal_group in itertools.groupby(
            enumerate(vacant_ordinals), key=lambda pair: pair[1] - pair[0]):
        ordinals = [pair[1] for pair in ordinal_group]
        lower_ordinals = [value for value in completion_by_ordinal
                          if value < ordinals[0]]
        upper_ordinals = [value for value in completion_by_ordinal
                          if value > ordinals[-1]]
        lower_scene = (None if not lower_ordinals else str(
            completion_by_ordinal[max(lower_ordinals)]["scene_id"]))
        upper_scene = (None if not upper_ordinals else str(
            completion_by_ordinal[min(upper_ordinals)]["scene_id"]))
        slots_by_ordinal = {
            _completion_state_ordinal(str(row["state_id"])): row
            for row in family_slots
        }
        groups.append({
            "lower_scene_id_exclusive": lower_scene,
            "upper_scene_id_exclusive": upper_scene,
            "vacant_ordinals": ordinals,
            "replacement_slots": [slots_by_ordinal[value] for value in ordinals],
        })
    if sum(len(row["replacement_slots"]) for row in groups) != len(family_slots):
        raise RuntimeError("mixed replacement interval grouping lost a slot")
    return {
        "family": family,
        "retained_states": sorted(retained_states, key=lambda row: row["state_id"]),
        "retained_state_count": len(retained_states),
        "retained_scene_ids": sorted(str(row["scene_id"])
                                     for row in retained_states),
        "retained_anchor_rows": retained_anchor_rows,
        "rejected_identity_digests": sorted(family_rejected),
        "rejected_identity_rows": sorted(
            family_rejected.values(), key=lambda row: row["state_id"]),
        "replacement_slots": family_slots,
        "interval_groups": groups,
    }


def _replacement_reuses_rejected_snapshot(
        state: dict[str, Any], slot: dict[str, Any]) -> bool:
    predecessor = _preserved_states_by_digest().get(str(
        slot["predecessor_state_identity_digest"]))
    if predecessor is None:
        raise RuntimeError("replacement slot predecessor identity is unknown")
    snapshot_keys = (
        "scene_id", "episode_cluster_id", "episode_id", "source_step",
        "warmup_blocks", "cell_id", "boundary",
    )
    return all(state.get(key) == predecessor.get(key) for key in snapshot_keys)


def _replacement_reuses_any_rejected_snapshot(
        state: dict[str, Any], rejected_identity_digests: Sequence[str]) -> bool:
    preserved = _preserved_states_by_digest()
    missing = [identity for identity in rejected_identity_digests
               if identity not in preserved]
    if missing:
        raise RuntimeError("replacement request contains an unknown rejected identity")
    return any(_replacement_reuses_rejected_snapshot(
        state, {"predecessor_state_identity_digest": identity})
        for identity in rejected_identity_digests)


def _preserve_invalid(path: Path, out: Path, reason: str) -> Path:
    invalid_root = out / "invalid_attempts"
    invalid_root.mkdir(parents=True, exist_ok=True)
    digest = file_sha256(path) if path.is_file() else "not-a-file"
    target = invalid_root / f"{path.name}.{digest[:16]}.{reason}.invalid"
    counter = 0
    while target.exists():
        counter += 1
        target = invalid_root / (
            f"{path.name}.{digest[:16]}.{reason}.{counter}.invalid")
    path.rename(target)
    return target


def _factorial_scene_exclusions() -> tuple[set[str], dict[str, Any]]:
    if file_sha256(FACTORIAL_MANIFEST) != FACTORIAL_MANIFEST_FILE_SHA256:
        raise RuntimeError("frozen factorial manifest file digest changed")
    if file_sha256(FACTORIAL_ROWS) != FACTORIAL_ROWS_SHA256:
        raise RuntimeError("frozen factorial base-row digest changed")
    factorial = json.loads(FACTORIAL_MANIFEST.read_text())
    _verify_self_digest(factorial, "digest", "factorial manifest")
    if factorial["digest"] != FACTORIAL_MANIFEST_DIGEST:
        raise RuntimeError("frozen factorial manifest identity changed")
    if factorial.get("base_manifest_rows_sha256") != FACTORIAL_ROWS_SHA256:
        raise RuntimeError("factorial manifest does not bind the expected base rows")
    base_rows = [json.loads(line) for line in FACTORIAL_ROWS.read_text().splitlines()
                 if line.strip()]
    scenes: set[str] = set()
    for row in factorial["rows"]:
        index = int(row["manifest_row_index"])
        if not 0 <= index < len(base_rows):
            raise RuntimeError("factorial manifest row index is out of bounds")
        scenes.add(str(base_rows[index]["scene"]))
    if len(scenes) != 80:
        raise RuntimeError(f"expected 80 factorial scenes, recovered {len(scenes)}")
    binding = {
        "factorial_manifest_digest": FACTORIAL_MANIFEST_DIGEST,
        "factorial_manifest_file_sha256": FACTORIAL_MANIFEST_FILE_SHA256,
        "factorial_rows_sha256": FACTORIAL_ROWS_SHA256,
        "scene_count": 80,
        "scene_ids_digest": canonical_digest(sorted(scenes)),
    }
    return scenes, binding


def _pilot_scene_exclusions() -> tuple[set[str], dict[str, Any]]:
    v11_path = V1.OUT_DIR / "identity_manifest.json"
    v11 = json.loads(v11_path.read_text())
    _verify_self_digest(v11, "identity_manifest_digest", "oracle-v1.1 identity manifest")
    if v11["identity_manifest_digest"] != V11_IDENTITY_MANIFEST_DIGEST:
        raise RuntimeError("oracle-v1.1 pilot identity manifest changed")
    v12_path = V12.OUT_DIR / "state_manifest.json"
    v12 = json.loads(v12_path.read_text())
    _verify_self_digest(v12, "state_manifest_digest", "oracle-v1.2 state manifest")
    if v12["state_manifest_digest"] != V12_IDENTITY_MANIFEST_DIGEST:
        raise RuntimeError("oracle-v1.2 pilot identity manifest changed")
    scenes = ({str(row["scene_id"]) for row in v11["pilot_states"]}
              | {str(row["scene_id"]) for row in v11["replay_states"]}
              | {str(row["scene_id"]) for row in v12["states"]})
    binding = {
        "oracle_v1_1_identity_manifest_digest": V11_IDENTITY_MANIFEST_DIGEST,
        "oracle_v1_2_identity_manifest_digest": V12_IDENTITY_MANIFEST_DIGEST,
        "scene_count": len(scenes),
        "scene_ids_digest": canonical_digest(sorted(scenes)),
    }
    return scenes, binding


def _state_identity_payload(state: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in state.items()
            if key not in {"state_identity_digest", "state_index",
                           "candidate_indices", "candidate_rotation_index",
                           "branch_identities"}}


def _state_identity_digest(state: dict[str, Any]) -> str:
    return canonical_digest({
        "schema": "go2_branch_state_identity_v1_2",
        "selection_digest": selection_digest(),
        "scorer_contract_v1_2_digest": scorer_contract_digest(),
        "state": _state_identity_payload(state),
    })


# ------------------------------------------------------------------ rendering --
def write_png_atomic(array: np.ndarray, path: Path, out: Path) -> tuple[str, int]:
    """Write an exact 224-square PNG without overwriting a differing artifact."""

    from PIL import Image
    image = np.asarray(array)
    if image.shape != (224, 224, 3) or image.dtype != np.uint8:
        raise RuntimeError(f"invalid historical RGB array {image.shape}/{image.dtype}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    Image.fromarray(image).save(temporary, format="PNG")
    digest = file_sha256(temporary)
    byte_count = temporary.stat().st_size
    if path.exists():
        if (path.is_file() and path.stat().st_size == byte_count
                and file_sha256(path) == digest):
            temporary.unlink()
            return digest, byte_count
        _preserve_invalid(path, out, "frame-mismatch")
    os.replace(temporary, path)
    return digest, byte_count


# ------------------------------------------------------------- proprioception --
def proprio_sample(ctx: V1.BranchContext) -> list[float]:
    """The frozen 30-D sensed-state vector, in the corpus channel order."""

    from lewm_genesis.rollout import _roll_from_quat_wxyz, _pitch_from_quat_wxyz
    runner = ctx.runner
    robot = ctx.build.robot
    quat_wxyz = np.asarray(runner._as_np(robot.get_quat()), dtype=np.float64)
    ang_world = np.asarray(runner._as_np(robot.get_ang()), dtype=np.float64)
    if quat_wxyz.ndim == 1:
        quat_wxyz = quat_wxyz[None, :]
    if ang_world.ndim == 1:
        ang_world = ang_world[None, :]
    qw, qx, qy, qz = (float(v) for v in quat_wxyz[0])
    quat_xyzw = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    gyro = runner._world_to_body(ang_world[0], quat_xyzw)
    gravity = M.projected_gravity(_roll_from_quat_wxyz(qw, qx, qy, qz),
                                  _pitch_from_quat_wxyz(qw, qx, qy, qz))
    feature = [g - o for g, o in zip(gravity, M.GRAVITY_OFFSET)]
    joint_pos = np.asarray(runner._as_np(
        robot.get_dofs_position(runner._leg_dof_idx.tolist())), dtype=np.float64)
    joint_vel = np.asarray(runner._as_np(
        robot.get_dofs_velocity(runner._leg_dof_idx.tolist())), dtype=np.float64)
    if joint_pos.ndim == 2:
        joint_pos, joint_vel = joint_pos[0], joint_vel[0]
    return ([float(v) for v in feature] + [float(v) for v in gyro]
            + [float(v) for v in joint_pos] + [float(v) for v in joint_vel])


def control_sample(previous_applied: Sequence[float]) -> list[float]:
    """Efference copy: the applied command at the tick BEFORE this sample."""

    return [float(previous_applied[c]) for c in SLEW.ACTIVE_CHANNELS]


def action_block_10d(executed_block: np.ndarray) -> list[float]:
    """The frozen 10-D five-tick post-slew action for one block."""

    return SLEW.flatten([[float(v) for v in tick] for tick in executed_block])


# ------------------------------------------------------------------- driving ---
def drive_block_with_probe(ctx: V1.BranchContext,
                           probe: Callable[[int, Sequence[float]], None]) -> Any:
    """V1.drive_one_block, with a per-command-tick probe.

    Replicated rather than patched so the frozen driver stays untouched.
    """

    runner = ctx.runner
    requested, _choices = runner._collect_block()
    executed = np.asarray(
        runner._clip_block(np.asarray(requested, dtype=np.float32)).executed,
        dtype=np.float64)
    steps_per_tick = int(runner._policy_steps_per_command_tick)
    previous = np.asarray(runner._last_executed, dtype=np.float64)[0].copy()
    carry = {"prev": previous}

    def after_policy_step(tick_idx: int, step_idx: int) -> None:
        if step_idx != steps_per_tick - 1:
            return
        probe(int(tick_idx), carry["prev"])
        carry["prev"] = executed[0, tick_idx].copy()

    block = runner.execute_requested_block(requested,
                                           after_policy_step=after_policy_step)
    for _ in range(runner._block_size):
        for state in runner.episode_states:
            state.step()
    runner._blocks_in_episode += 1
    ctx.ticks_executed += runner._block_size
    ctx.episode_ticks += runner._block_size
    ctx.policy_steps += runner._block_size * steps_per_tick
    ctx.last_block_executed = np.asarray(block.executed, dtype=np.float32).copy()

    before = int(runner.episode_states[0].reset_count)
    from lewm_genesis import ros_msg_adapter as adapter
    runner._check_and_reset_fallen_envs(V1._NullWriter(), adapter)
    runner._check_and_reset_completed_envs(V1._NullWriter(), adapter)
    after = int(runner.episode_states[0].reset_count)
    ctx.reset_in_last_block = after != before
    if ctx.reset_in_last_block:
        ctx.episode_ticks = 0
        ctx.last_block_executed = None
        ctx.episode_start_reset_count = after
    return block


# ------------------------------------------------------------------ eligible --
def landmark_bearing_range(ctx: V1.BranchContext, landmark_xy: Sequence[float]
                           ) -> tuple[float, float]:
    """The planning-time observable goal binding: body bearing and range."""

    from lewm_worlds.scene_graph import wrap_angle_pi
    (x, y), yaw, _z = ctx.pose()
    dx = float(landmark_xy[0]) - x
    dy = float(landmark_xy[1]) - y
    return float(wrap_angle_pi(math.atan2(dy, dx) - yaw)), float(math.hypot(dx, dy))


_FIELD_CACHE: dict[tuple[int, int], GeodesicField] = {}


def geodesic_field(ctx: V1.BranchContext, landmark_cell: int,
                   blocked: frozenset) -> GeodesicField:
    """Dijkstra depends only on the scene and the goal, so cache it per scene."""

    key = (id(ctx.scene_graph), int(landmark_cell))
    field = _FIELD_CACHE.get(key)
    if field is None:
        field = GeodesicField(ctx.scene_graph, int(landmark_cell),
                              transit_blocked=blocked)
        _FIELD_CACHE[key] = field
    return field


def _snapshot_task_status(ctx: V1.BranchContext, goal_cell: int) -> dict[str, Any]:
    """Return the production task flags at one pre-branch canonical boundary.

    The active collector is the sole owner of route claims.  Truncation is
    false by construction: this selector runs between complete production
    command blocks, before a candidate branch or invocation-level limit exists.
    """

    runner = ctx.runner
    active = runner._scheduler.policy_for(0)
    visited = getattr(active, "visited_landmark_cells", None)
    claimed = (frozenset(int(cell) for cell in visited(0))
               if callable(visited) else frozenset())
    # Match RolloutRunner._check_and_reset_completed_envs exactly: its route
    # completion universe is the runner's landmark-cell lookup, not a newly
    # inferred or filtered goal list.
    all_goal_cells = frozenset(
        int(cell) for cell in runner._landmark_cell_to_id
    )
    from lewm_genesis.rollout import _MIN_BLOCKS_BEFORE_COMPLETE_RESET
    reset_evidence = {
        "minimum_block_guard_pass": (
            int(runner._blocks_in_episode[0])
            >= _MIN_BLOCKS_BEFORE_COMPLETE_RESET),
        "scene_graph_available": runner._scene_graph is not None,
        "active_collector_route_like": callable(visited),
        "active_collector_non_revisit": not bool(
            getattr(active, "revisit_after_arrival", False)),
        "scene_landmark_cells_nonempty": bool(all_goal_cells),
        "all_scene_landmark_cells_claimed": bool(
            all_goal_cells and all_goal_cells.issubset(claimed)),
    }
    task_completed = all(reset_evidence.values())
    termination_flags = {
        str(key): bool(value) for key, value in V1._termination_flags(ctx).items()
    }
    return {
        "task_completed": task_completed,
        "goal_claimed": int(goal_cell) in claimed,
        "production_claim_evidence": {
            "active_collector_visited_accessor_callable": callable(visited),
            "active_collector_claimed_cells": sorted(claimed),
            "designated_goal_cell": int(goal_cell),
        },
        "production_task_completion_reset_evidence": reset_evidence,
        "terminated": any(termination_flags.values()),
        "truncated": False,
        "termination_flags": termination_flags,
    }


def _predecessor_start_radius_eligibility(
        *, graph_hops: int, reachable: bool, continuous_geodesic_m: float,
        bearing_body_rad: float, task_status: dict[str, Any]) -> dict[str, Any]:
    """Superseded V1 predicate retained solely for frozen lineage tests.

    ``graph_hops`` is retained as diagnostic evidence only.  In particular,
    zero hops is not itself completion: the production task claim/completion
    flags remain authoritative and must both be false.
    """

    reasons: list[str] = []
    if not bool(reachable) or not math.isfinite(float(continuous_geodesic_m)):
        reasons.append("completion_unreachable")
    elif float(continuous_geodesic_m) > COMPLETION_ENRICHED_MAX_GEODESIC_M:
        reasons.append("completion_geodesic_gt_0_75m")
    if (not math.isfinite(float(bearing_body_rad))
            or abs(float(bearing_body_rad)) > COMPLETION_ENRICHED_MAX_BEARING_RAD):
        reasons.append("completion_bearing_gt_75deg")
    for key in ("task_completed", "goal_claimed", "terminated", "truncated"):
        if key not in task_status or not isinstance(task_status[key], bool):
            reasons.append(f"completion_snapshot_{key}_unavailable")
        elif task_status[key]:
            reasons.append(f"completion_snapshot_{key}")
    return {
        "eligible": not reasons,
        "rejection_reasons": reasons,
        "reachable": bool(reachable),
        "continuous_geodesic_m": float(continuous_geodesic_m),
        "max_geodesic_m": COMPLETION_ENRICHED_MAX_GEODESIC_M,
        "bearing_body_rad": float(bearing_body_rad),
        "abs_bearing_rad": abs(float(bearing_body_rad)),
        "max_abs_bearing_rad": COMPLETION_ENRICHED_MAX_BEARING_RAD,
        "horizon_s": COMPLETION_HORIZON_S,
        "max_translation_m": COMPLETION_MAX_TRANSLATION_M,
        "graph_hops_diagnostic": int(graph_hops),
        "task_status": dict(task_status),
    }


def _oracle_completion_target_unchanged() -> bool:
    """Verify the frozen future completion label, not a snapshot task flag."""

    return bool(
        v12_oracle_digest() == STATE_SELECTOR.ORACLE_V1_2_DIGEST
        and HORIZONS == 4
        and COMPLETION_HORIZON_S == HORIZON_S == 2.0
    )


def _snapshot_claim_semantics_unchanged(task_status: dict[str, Any]) -> bool:
    evidence = task_status.get("production_claim_evidence", {})
    cells = evidence.get("active_collector_claimed_cells")
    goal_cell = evidence.get("designated_goal_cell")
    return bool(
        isinstance(evidence.get(
            "active_collector_visited_accessor_callable"), bool)
        and isinstance(cells, list)
        and isinstance(goal_cell, int)
        and isinstance(task_status.get("goal_claimed"), bool)
        and task_status["goal_claimed"] is (goal_cell in cells)
    )


def _production_task_reset_semantics_unchanged(
        task_status: dict[str, Any]) -> bool:
    evidence = task_status.get("production_task_completion_reset_evidence", {})
    required = (
        "minimum_block_guard_pass", "scene_graph_available",
        "active_collector_route_like", "active_collector_non_revisit",
        "scene_landmark_cells_nonempty", "all_scene_landmark_cells_claimed",
    )
    return bool(
        all(isinstance(evidence.get(key), bool) for key in required)
        and task_status.get("task_completed") is all(evidence[key] for key in required)
    )


def _goal_material(ctx: V1.BranchContext, name: str) -> str | None:
    materials = sorted({str(obj.material_id) for obj in ctx.pack.static_objects
                        if str(obj.object_id) == str(name)})
    if len(materials) != 1 or not materials[0].startswith("landmark_"):
        return None
    return materials[0]


def _state_record(*, boundary: dict[str, Any], cell: int, name: str,
                  landmark_cell: int, goal_type: str, hops: int, distance: float,
                  bearing: float, range_m: float, centre: Sequence[float],
                  body_clearance: float, clearance: float,
                  completion_eligibility: dict[str, Any] | None = None,
                  completion_rotation_eligibility_vector:
                      dict[str, Any] | None = None,
                  snapshot_task_status: dict[str, Any] | None = None
                  ) -> dict[str, Any]:
    record = {
        "boundary": boundary, "cell_id": int(cell),
        "goal": {"landmark_id": str(name), "landmark_cell": int(landmark_cell),
                 "material_id": str(goal_type),
                 "graph_edges": int(hops), "start_geodesic_m": float(distance),
                 "bearing_body_rad": float(bearing), "range_m": float(range_m),
                 "landmark_xy_m": [float(centre[0]), float(centre[1])]},
        "body_clearance_m": float(body_clearance),
        "clearance_m": float(clearance),
    }
    if completion_eligibility is not None:
        record["completion_eligibility"] = completion_eligibility
        record["snapshot_task_status"] = completion_eligibility["task_status"]
    if completion_rotation_eligibility_vector is not None:
        record["completion_rotation_eligibility_vector"] = \
            completion_rotation_eligibility_vector
        rotations = completion_rotation_eligibility_vector["rotations"]
        if not rotations:
            raise RuntimeError("completion rotation vector is empty")
        record["snapshot_task_status"] = (
            dict(snapshot_task_status) if snapshot_task_status is not None
            else dict(rotations[0]["task_status"])
        )
        record["previous_applied_command"] = list(
            rotations[0]["previous_applied_command"])
    return record


def classify_state(ctx: V1.BranchContext, topology: dict[str, Any], *,
                   requested_stratum: str | None = None,
                   diagnostics: dict[str, int] | None = None
                   ) -> tuple[dict[str, Any], GeodesicField, set[str]] | str:
    """Classify one snapshot, optionally binding the goal for one stratum.

    General and safety retain the original nearest-landmark ``hops >= 2``
    semantics.  The successor changes only completion goal enumeration: its
    hop count is diagnostic, while unchanged continuous geometry and exact
    production task-status guards determine eligibility.
    """

    if requested_stratum not in (None, *STRATA):
        raise ValueError(f"unknown requested stratum {requested_stratum!r}")

    def reject(reason: str) -> str:
        if diagnostics is not None:
            diagnostics[reason] = diagnostics.get(reason, 0) + 1
        return reason

    try:
        boundary = V1.assert_canonical_boundary(ctx)
    except V1.BoundaryRefused as exc:
        return reject(f"boundary_refused: {str(exc)[:50]}")
    if ctx.episode_ticks < PROPRIO_HISTORY - 1:
        return reject("insufficient_proprioceptive_history")
    (x, y), yaw, _z = ctx.pose()
    hit = ctx.scene_graph.locate((x, y))
    if float(hit.distance_m) > V1.LOCATE_MAX_DISTANCE_M:
        return reject("locate_distance_gt_2m")
    if V12._contact_count(ctx, topology) > 0:
        return reject("already_in_disallowed_contact")

    graph = ctx.scene_graph
    blocked = getattr(graph, "nav_blocked_cells", frozenset())
    cell = int(hit.cell_id)
    from analyze_go2_closed_loop_quality import _body_probe_configuration_clearance_m
    body_clearance = float(_body_probe_configuration_clearance_m(
        ctx.grid, [x, y], yaw,
        body_forward_m=V1.CONTACT_BODY_FORWARD_M,
        body_half_width_m=V1.CONTACT_BODY_HALF_WIDTH_M,
        body_probe_margin_m=V1.CONTACT_BODY_PROBE_MARGIN_M))
    clearance = float(graph.clearance_to_walls((x, y)))

    # Completion uses a dedicated designation pass.  Unlike the general/safety
    # pass below, no hop floor is applied before the unchanged reachability,
    # continuous-distance, bearing, horizon and task-state checks.
    if requested_stratum == "completion_enriched":
        eligible: list[tuple[tuple[float, str, int, int], dict[str, Any],
                                  GeodesicField]] = []
        saw_reachable = False
        for name, landmark_cell in sorted(graph.landmark_cells,
                                          key=lambda kv: str(kv[0])):
            hops = graph.bfs_distance(cell, int(landmark_cell),
                                      transit_blocked=blocked)
            if hops is None:
                if diagnostics is not None:
                    diagnostics["completion_unreachable"] = diagnostics.get(
                        "completion_unreachable", 0) + 1
                continue
            saw_reachable = True
            field = geodesic_field(ctx, int(landmark_cell), blocked)
            distance = field.remaining_distance((x, y), cell)
            centre = graph.cell_center(int(landmark_cell))
            bearing, range_m = landmark_bearing_range(ctx, centre)
            task_status = _snapshot_task_status(ctx, int(landmark_cell))
            previous_applied = np.asarray(
                ctx.runner._last_executed, dtype=np.float64)[0].tolist()
            vector = STATE_SELECTOR.completion_rotation_eligibility_vector(
                graph_hops=int(hops), reachable=math.isfinite(distance),
                continuous_geodesic_m=float(distance),
                bearing_body_rad=float(bearing), task_status=task_status,
                previous_applied_command=previous_applied)
            if diagnostics is not None:
                reasons = sorted({
                    reason for row in vector["rotations"]
                    for reason in row["rejection_reasons"]
                    if reason != (
                        "completion_geodesic_gap_gt_allocated_subset_l_max")
                })
                if (not vector["eligible_under_at_least_one_rotation"]
                        and not reasons):
                    reasons.append(
                        "completion_gap_exceeds_every_allowed_subset_l_max")
                for reason in reasons:
                    diagnostics[reason] = diagnostics.get(reason, 0) + 1
            goal_type = _goal_material(ctx, str(name))
            if goal_type is None:
                if diagnostics is not None:
                    diagnostics["bound_landmark_material_missing_or_ambiguous"] = \
                        diagnostics.get(
                            "bound_landmark_material_missing_or_ambiguous", 0) + 1
                continue
            if not vector["eligible_under_at_least_one_rotation"]:
                continue
            record = _state_record(
                boundary=boundary, cell=cell, name=str(name),
                landmark_cell=int(landmark_cell), goal_type=goal_type,
                hops=int(hops), distance=float(distance), bearing=float(bearing),
                range_m=float(range_m), centre=centre,
                body_clearance=body_clearance, clearance=clearance,
                completion_rotation_eligibility_vector=vector,
                snapshot_task_status=task_status)
            key = (float(distance), str(name), int(landmark_cell), int(hops))
            eligible.append((key, record, field))
        if not eligible:
            return reject("no_completion_enriched_goal" if saw_reachable
                          else "no_reachable_landmark")
        _key, record, field = min(eligible, key=lambda row: row[0])
        return record, field, {"completion_enriched"}

    # Original general/safety designation and ordering, byte-for-byte in
    # substance: the closest reachable landmark at one or more graph edges.
    best: tuple[float, str, int, int, GeodesicField] | None = None
    for name, landmark_cell in sorted(graph.landmark_cells, key=lambda kv: str(kv[0])):
        hops = graph.bfs_distance(cell, int(landmark_cell), transit_blocked=blocked)
        if hops is None or int(hops) < 1:
            continue
        field = geodesic_field(ctx, int(landmark_cell), blocked)
        distance = field.remaining_distance((x, y), cell)
        if not math.isfinite(distance):
            continue
        key = (float(distance), str(name), int(landmark_cell), int(hops))
        if best is None or key < best[:4]:
            best = (*key, field)
    if best is None:
        return reject("no_reachable_landmark")
    distance, name, landmark_cell, hops, field = best
    centre = graph.cell_center(int(landmark_cell))
    bearing, range_m = landmark_bearing_range(ctx, centre)
    goal_type = _goal_material(ctx, str(name))
    if goal_type is None:
        return reject("bound_landmark_material_missing_or_ambiguous")

    strata: set[str] = set()
    if hops >= 2:
        strata.add("general")
        if body_clearance <= SAFETY_ENRICHED_MAX_BODY_CLEARANCE_M:
            strata.add("safety_enriched")
    # Preserve the pre-successor default path for final-evaluation callers.
    if (requested_stratum is None
            and distance <= COMPLETION_ENRICHED_MAX_GEODESIC_M
            and abs(bearing) <= COMPLETION_ENRICHED_MAX_BEARING_RAD):
        strata.add("completion_enriched")
    if requested_stratum is not None and requested_stratum not in strata:
        return reject("no_stratum")
    if not strata:
        return reject("no_stratum")
    record = _state_record(
        boundary=boundary, cell=cell, name=str(name),
        landmark_cell=int(landmark_cell), goal_type=goal_type, hops=int(hops),
        distance=float(distance), bearing=float(bearing), range_m=float(range_m),
        centre=centre, body_clearance=body_clearance, clearance=clearance)
    return record, field, strata


# ------------------------------------------------------------------- stage A --
def scene_pool(pool_name: str) -> tuple[dict[str, list[Path]], dict[str, Any]]:
    factorial_scenes, factorial_binding = _factorial_scene_exclusions()
    pilot_scenes, pilot_binding = _pilot_scene_exclusions()
    invalid_identity_index = INVALID_IDS.load_invalid_identity_index()
    excluded = (set(factorial_scenes) | set(pilot_scenes)
                | set(invalid_identity_index.scene_ids))
    scorer_binding: dict[str, Any] | None = None
    if pool_name == "final_eval":
        scorer_path = OUT_ROOT / "scorer_fit/state_manifest.json"
        scorer_manifest = load_active_state_manifest_for_consumption(
            scorer_path)
        if (scorer_manifest.get("pool") != "scorer_fit"
                or len(scorer_manifest.get("states", [])) != 120
                or scorer_manifest.get("scorer_contract_v1_2_digest")
                != scorer_contract_digest()):
            raise RuntimeError("final selection requires the complete current scorer-fit identity manifest")
        scorer_scenes = {str(row["scene_id"]) for row in scorer_manifest["states"]}
        if len(scorer_scenes) != 120:
            raise RuntimeError("scorer-fit identity manifest is not scene-disjoint")
        excluded |= scorer_scenes
        scorer_binding = {
            "state_manifest_digest": scorer_manifest["state_manifest_digest"],
            "scene_count": len(scorer_scenes),
            "scene_ids_digest": canonical_digest(sorted(scorer_scenes)),
        }

    families: dict[str, list[Path]] = {}
    for split in SPLITS:
        root = CORPUS / split
        _assert_unsealed_path(root)
        if not root.is_dir():
            continue
        for family_dir in sorted(root.iterdir()):
            _assert_unsealed_path(family_dir)
            if not family_dir.is_dir():
                continue
            for scene_dir in sorted(family_dir.iterdir()):
                _assert_unsealed_path(scene_dir)
                if scene_dir.name in excluded:
                    continue
                manifest = scene_dir / "manifest.json"
                genesis_scene = scene_dir / "genesis_scene.json"
                _assert_unsealed_path(manifest)
                _assert_unsealed_path(genesis_scene)
                if (not manifest.is_file() or not genesis_scene.is_file()):
                    continue
                families.setdefault(family_dir.name, []).append(scene_dir)
    for family in families:
        families[family].sort(key=lambda path: path.name)
    if len(families) != EXPECTED_FAMILIES:
        raise RuntimeError(f"expected eight scene families, found {sorted(families)}")
    allow_list = {family: [path.name for path in paths]
                  for family, paths in sorted(families.items())}
    exclusion_binding = {
        "factorial": factorial_binding,
        "oracle_pilots": pilot_binding,
        "invalid_scorer_identity_attempt": invalid_identity_index.binding(),
        "scorer_fit": scorer_binding,
        "excluded_scene_count": len(excluded),
        "excluded_scene_ids_digest": canonical_digest(sorted(excluded)),
        "allow_list_scene_count": sum(len(values) for values in allow_list.values()),
        "allow_list_digest": canonical_digest(allow_list),
    }
    return families, exclusion_binding


def _metric_distribution(values: Sequence[float]) -> dict[str, Any]:
    finite = np.asarray([float(value) for value in values
                         if math.isfinite(float(value))], dtype=np.float64)
    if finite.size == 0:
        return {"count": 0, "min": None, "q1": None, "median": None,
                "mean": None, "q3": None, "max": None}
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "q1": float(np.quantile(finite, 0.25)),
        "median": float(np.quantile(finite, 0.5)),
        "mean": float(np.mean(finite)),
        "q3": float(np.quantile(finite, 0.75)),
        "max": float(np.max(finite)),
    }


def build_selector_feasibility_summary(
        *, family: str, allowed_scene_count: int,
        requested_strata: Sequence[str], scene_evidence: Sequence[dict[str, Any]],
        rejection_counts: dict[str, int]) -> dict[str, Any]:
    """Pure reducer for an identity-free, outcome-free family dry-run."""

    strata = tuple(str(value) for value in requested_strata)
    if not strata or any(value not in STRATA for value in strata):
        raise ValueError("dry-run requested unknown or empty strata")
    evidence_by_stratum: dict[str, list[dict[str, Any]]] = {
        stratum: [] for stratum in strata
    }
    seen_pairs: set[tuple[str, str]] = set()
    for row in scene_evidence:
        # Intentionally enumerate the permitted keys: a branch outcome, oracle
        # label or candidate result has no read path through this reducer.
        row_family = str(row["family"])
        scene_id = str(row["scene_id"])
        stratum = str(row["stratum"])
        if row_family != family or stratum not in evidence_by_stratum:
            raise ValueError("dry-run evidence family/stratum mismatch")
        pair = (scene_id, stratum)
        if pair in seen_pairs:
            raise ValueError("dry-run evidence must be first-eligible per scene/stratum")
        seen_pairs.add(pair)
        evidence_by_stratum[stratum].append({
            "scene_id": scene_id,
            "first_eligible_block": int(row["first_eligible_block"]),
            "continuous_geodesic_m": float(row["continuous_geodesic_m"]),
            "abs_bearing_rad": float(row["abs_bearing_rad"]),
            "graph_hops_diagnostic": int(row["graph_hops_diagnostic"]),
            "body_clearance_m": float(row["body_clearance_m"]),
        })

    per_stratum: dict[str, Any] = {}
    for stratum in strata:
        rows = sorted(evidence_by_stratum[stratum],
                      key=lambda row: (row["scene_id"], row["first_eligible_block"]))
        count = len(rows)
        required = int(POOLS["scorer_fit"]["strata"][stratum])
        per_stratum[stratum] = {
            "required_distinct_scenes": required,
            "eligible_distinct_scenes": count,
            "quota_pass": count >= required,
            "distributions": {
                "continuous_geodesic_m": _metric_distribution(
                    [row["continuous_geodesic_m"] for row in rows]),
                "abs_bearing_rad": _metric_distribution(
                    [row["abs_bearing_rad"] for row in rows]),
                "graph_hops_diagnostic": _metric_distribution(
                    [row["graph_hops_diagnostic"] for row in rows]),
                "first_eligible_block": _metric_distribution(
                    [row["first_eligible_block"] for row in rows]),
                "body_clearance_m": _metric_distribution(
                    [row["body_clearance_m"] for row in rows]),
            },
            "scene_evidence": rows,
        }
    return {
        "family": str(family),
        "allowed_scene_count": int(allowed_scene_count),
        "scanned_scene_count": int(allowed_scene_count),
        "requested_strata": list(strata),
        "per_stratum": per_stratum,
        "rejection_counts": {
            str(key): int(value) for key, value in sorted(rejection_counts.items())
        },
        "all_requested_quotas_pass": all(
            row["quota_pass"] for row in per_stratum.values()),
    }


def _selector_scene_evidence(family: str, scene_id: str, stratum: str,
                             block_index: int,
                             record: dict[str, Any]) -> dict[str, Any]:
    goal = record["goal"]
    return {
        "family": str(family),
        "scene_id": str(scene_id),
        "stratum": str(stratum),
        "first_eligible_block": int(block_index),
        "continuous_geodesic_m": float(goal["start_geodesic_m"]),
        "abs_bearing_rad": abs(float(goal["bearing_body_rad"])),
        "graph_hops_diagnostic": int(goal["graph_edges"]),
        "body_clearance_m": float(record["body_clearance_m"]),
    }


def _scan_selector_scene(*, family: str, scene_dir: Path,
                         requested_strata: Sequence[str], ctx: Any
                         ) -> dict[str, Any]:
    """Scan one exact scene; a native crash cannot yield a result object."""

    evidence: list[dict[str, Any]] = []
    rejections: dict[str, int] = {}
    topology = V12.link_topology(ctx)
    ctx.begin_episode()
    found_in_scene: set[str] = set()
    for block_idx in range(WARMUP_BLOCKS_MAX):
        ctx.drive_one_block()
        if block_idx + 1 < WARMUP_BLOCKS_MIN:
            continue
        for stratum in requested_strata:
            if stratum in found_in_scene:
                continue
            local_diagnostics: dict[str, int] = {}
            verdict = classify_state(
                ctx, topology, requested_stratum=stratum,
                diagnostics=local_diagnostics)
            for reason, count in local_diagnostics.items():
                key = reason.split(":")[0]
                rejections[key] = rejections.get(key, 0) + int(count)
            if isinstance(verdict, str):
                continue
            record, _field, _eligible = verdict
            evidence.append(_selector_scene_evidence(
                family, scene_dir.name, stratum, block_idx + 1, record))
            found_in_scene.add(stratum)
        if len(found_in_scene) == len(requested_strata):
            break
    return {
        "family": family,
        "scene_id": scene_dir.name,
        "scene_evidence": evidence,
        "rejection_counts": {
            str(key): int(value) for key, value in sorted(rejections.items())
        },
    }


def _selector_feasibility_scene_task(
        family: str, scene_dir: Path, task_index: int) -> dict[str, Any]:
    manifest = scene_dir / "manifest.json"
    genesis_scene = scene_dir / "genesis_scene.json"
    _assert_unsealed_path(scene_dir)
    _assert_unsealed_path(manifest)
    _assert_unsealed_path(genesis_scene)
    payload = {
        "schema": "go2_scorer_fit_selector_feasibility_scene_task_v1",
        "family": family,
        "task_index_within_family": int(task_index),
        "scene_id": scene_dir.name,
        "scene_dir": str(scene_dir.resolve()),
        "split": scene_dir.parent.parent.name,
        "drive_seed": int(V1._drive_seed(scene_dir.name)),
        "scene_manifest_sha256": file_sha256(manifest),
        "scene_manifest_byte_count": manifest.stat().st_size,
        "genesis_scene_sha256": file_sha256(genesis_scene),
        "genesis_scene_byte_count": genesis_scene.stat().st_size,
        "requested_strata": list(STRATA),
    }
    payload["scene_task_digest"] = canonical_digest(payload)
    return payload


def build_selector_feasibility_task_census(
        *, pool: dict[str, Sequence[Path]], source: dict[str, Any],
        successor_selection_digest: str,
        exclusion_binding_digest: str) -> dict[str, Any]:
    """Freeze every allowed scene task before any isolated worker starts."""

    if set(pool) != set(STATE_SELECTOR.REQUIRED_FAMILIES):
        raise RuntimeError("selector-feasibility task census family set changed")
    families: list[dict[str, Any]] = []
    all_task_digests: list[str] = []
    for family in STATE_SELECTOR.REQUIRED_FAMILIES:
        scenes = sorted(pool[family], key=lambda path: path.name)
        if len({scene.name for scene in scenes}) != len(scenes):
            raise RuntimeError(
                f"selector-feasibility task census repeats a scene in {family}")
        tasks = [
            _selector_feasibility_scene_task(family, scene, index)
            for index, scene in enumerate(scenes)
        ]
        all_task_digests.extend(task["scene_task_digest"] for task in tasks)
        families.append({
            "family": family,
            "allowed_scene_count": len(tasks),
            "tasks": tasks,
            "family_task_set_digest": canonical_digest(
                [task["scene_task_digest"] for task in tasks]),
        })
    payload = {
        "schema": SELECTOR_FEASIBILITY_TASK_CENSUS_SCHEMA,
        "status": "FROZEN_OUTCOME_FREE_EXHAUSTIVE_SCENE_TASK_CENSUS",
        "complete": True,
        "source_repository_commit": source["source_repository_commit"],
        "clean_source_binding_digest": canonical_digest(source),
        "bound_implementations_digest": source["bound_implementations_digest"],
        "successor_selection_digest": successor_selection_digest,
        "state_selector_amendment_digest":
            STATE_SELECTOR.state_selector_amendment_digest(),
        "exclusion_binding_digest": exclusion_binding_digest,
        "family_count": len(families),
        "scene_task_count": len(all_task_digests),
        "families": families,
        "scene_task_set_digest": canonical_digest(all_task_digests),
        "selected_state_identities_created": False,
        "candidate_outcomes_loaded": False,
        "branch_identities_created": False,
        "branches_attempted": 0,
        "frames_rendered": 0,
        "target_latents_encoded": 0,
        "scorer_training_started": False,
    }
    payload["state_selector_feasibility_task_census_digest"] = \
        canonical_digest(payload)
    return payload


def _validate_selector_feasibility_task_census(
        census: dict[str, Any], *, pool: dict[str, Sequence[Path]],
        source: dict[str, Any], successor_selection_digest: str,
        exclusion_binding_digest: str) -> None:
    _verify_self_digest(
        census, "state_selector_feasibility_task_census_digest",
        "state-selector feasibility task census")
    expected = build_selector_feasibility_task_census(
        pool=pool, source=source,
        successor_selection_digest=successor_selection_digest,
        exclusion_binding_digest=exclusion_binding_digest)
    if census != expected:
        raise RuntimeError(
            "selector-feasibility task census differs from the exact allow-list")


def _issue_selector_feasibility_task_census(
        *, out: Path, pool: dict[str, Sequence[Path]], source: dict[str, Any],
        successor_selection_digest: str,
        exclusion_binding_digest: str) -> dict[str, Any]:
    path = out / SELECTOR_FEASIBILITY_TASK_CENSUS_NAME
    expected = build_selector_feasibility_task_census(
        pool=pool, source=source,
        successor_selection_digest=successor_selection_digest,
        exclusion_binding_digest=exclusion_binding_digest)
    if path.is_file():
        try:
            existing = json.loads(path.read_text())
            _validate_selector_feasibility_task_census(
                existing, pool=pool, source=source,
                successor_selection_digest=successor_selection_digest,
                exclusion_binding_digest=exclusion_binding_digest)
        except Exception:
            if _outcome_generation_started(out):
                raise RuntimeError(
                    "selector-feasibility task census changed after outcomes")
            _preserve_invalid(path, out, "selector-feasibility-task-census-invalid")
        else:
            return existing
    elif path.exists():
        if _outcome_generation_started(out):
            raise RuntimeError(
                "selector-feasibility task census path changed after outcomes")
        _preserve_invalid(path, out, "selector-feasibility-task-census-invalid")
    atomic_json(path, expected)
    return expected


def _selector_feasibility_family_tasks(
        census: dict[str, Any], family: str) -> list[dict[str, Any]]:
    matches = [row for row in census["families"] if row["family"] == family]
    if len(matches) != 1:
        raise RuntimeError(f"task census family lookup is ambiguous for {family}")
    return list(matches[0]["tasks"])


def _selector_feasibility_scene_shard_path(
        out: Path, task: dict[str, Any]) -> Path:
    return (out / SELECTOR_FEASIBILITY_SCENE_SHARD_ROOT / task["family"]
            / f"{task['scene_task_digest']}.json")


def _build_selector_feasibility_scene_shard(
        *, task: dict[str, Any], scene_result: dict[str, Any],
        task_census_digest: str, source: dict[str, Any],
        successor_selection_digest: str, exclusion_binding_digest: str,
        runtime_s: float) -> dict[str, Any]:
    payload = {
        "schema": SELECTOR_FEASIBILITY_SCENE_SHARD_SCHEMA,
        "status": SELECTOR_FEASIBILITY_SCENE_SHARD_STATUS,
        "complete": True,
        "binding_receipt": False,
        "eligibility_verdict_inferred_from_process_exit": False,
        "source_repository_commit": source["source_repository_commit"],
        "clean_source_binding_digest": canonical_digest(source),
        "bound_implementations_digest": source["bound_implementations_digest"],
        "successor_selection_digest": successor_selection_digest,
        "state_selector_amendment_digest":
            STATE_SELECTOR.state_selector_amendment_digest(),
        "state_selector_feasibility_task_census_digest": task_census_digest,
        "exclusion_binding_digest": exclusion_binding_digest,
        "task": task,
        "scene_result": scene_result,
        "runtime_s": round(float(runtime_s), 6),
        "selected_state_identities_created": False,
        "candidate_outcomes_loaded": False,
        "branch_identities_created": False,
        "branches_attempted": 0,
        "frames_rendered": 0,
        "target_latents_encoded": 0,
        "scorer_training_started": False,
    }
    payload["state_selector_feasibility_scene_shard_digest"] = \
        canonical_digest(payload)
    return payload


def _validate_selector_feasibility_scene_shard(
        shard: dict[str, Any], *, expected_task: dict[str, Any],
        expected_task_census_digest: str, source: dict[str, Any],
        expected_successor_selection_digest: str,
        expected_exclusion_binding_digest: str) -> None:
    _verify_self_digest(
        shard, "state_selector_feasibility_scene_shard_digest",
        f"selector-feasibility scene shard {expected_task['scene_id']}")
    if (shard.get("schema") != SELECTOR_FEASIBILITY_SCENE_SHARD_SCHEMA
            or shard.get("status") != SELECTOR_FEASIBILITY_SCENE_SHARD_STATUS
            or shard.get("complete") is not True
            or shard.get("binding_receipt") is not False
            or shard.get("eligibility_verdict_inferred_from_process_exit") is not False
            or shard.get("source_repository_commit")
            != source["source_repository_commit"]
            or shard.get("clean_source_binding_digest") != canonical_digest(source)
            or shard.get("bound_implementations_digest")
            != source["bound_implementations_digest"]
            or shard.get("successor_selection_digest")
            != expected_successor_selection_digest
            or shard.get("state_selector_amendment_digest")
            != STATE_SELECTOR.state_selector_amendment_digest()
            or shard.get("state_selector_feasibility_task_census_digest")
            != expected_task_census_digest
            or shard.get("exclusion_binding_digest")
            != expected_exclusion_binding_digest
            or shard.get("task") != expected_task
            or any(shard.get(key) not in (False, 0)
                   for key in SELECTOR_FEASIBILITY_FORBIDDEN_FIELDS)):
        raise RuntimeError(
            f"selector-feasibility scene shard {expected_task['scene_id']} binding failed")
    runtime_s = shard.get("runtime_s")
    if (isinstance(runtime_s, bool) or not isinstance(runtime_s, (int, float))
            or not math.isfinite(float(runtime_s)) or float(runtime_s) < 0.0):
        raise RuntimeError("selector-feasibility scene runtime is invalid")
    result = shard.get("scene_result")
    if (not isinstance(result, dict)
            or set(result) != {
                "family", "scene_id", "scene_evidence", "rejection_counts"}
            or result.get("family") != expected_task["family"]
            or result.get("scene_id") != expected_task["scene_id"]
            or not isinstance(result.get("scene_evidence"), list)
            or not isinstance(result.get("rejection_counts"), dict)):
        raise RuntimeError("selector-feasibility scene result is malformed")
    seen: set[str] = set()
    evidence_keys = {
        "family", "scene_id", "stratum", "first_eligible_block",
        "continuous_geodesic_m", "abs_bearing_rad", "graph_hops_diagnostic",
        "body_clearance_m",
    }
    for evidence in result["scene_evidence"]:
        if (not isinstance(evidence, dict)
                or set(evidence) != evidence_keys
                or evidence.get("family") != expected_task["family"]
                or evidence.get("scene_id") != expected_task["scene_id"]
                or evidence.get("stratum") not in STRATA
                or evidence["stratum"] in seen):
            raise RuntimeError("selector-feasibility scene evidence is malformed")
        seen.add(str(evidence["stratum"]))
    if any(not isinstance(key, str)
           or isinstance(value, bool) or not isinstance(value, int) or value < 0
           for key, value in result["rejection_counts"].items()):
        raise RuntimeError("selector-feasibility scene rejections are malformed")


def _load_completed_selector_feasibility_scene_shard(
        path: Path, *, expected_task: dict[str, Any],
        task_census_digest: str, source: dict[str, Any],
        successor_selection_digest: str,
        exclusion_binding_digest: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        shard = json.loads(path.read_text())
        _validate_selector_feasibility_scene_shard(
            shard, expected_task=expected_task,
            expected_task_census_digest=task_census_digest, source=source,
            expected_successor_selection_digest=successor_selection_digest,
            expected_exclusion_binding_digest=exclusion_binding_digest)
        return shard
    except (OSError, TypeError, ValueError, RuntimeError, json.JSONDecodeError):
        return None


def _execute_selector_feasibility_scene_worker(
        *, args: argparse.Namespace, task: dict[str, Any], path: Path,
        task_census_digest: str, source: dict[str, Any],
        successor_selection_digest: str,
        exclusion_binding_digest: str) -> dict[str, Any]:
    """Scan and atomically receipt one scene before releasing native state."""

    started = time.time()
    shared = V1._load_shared(args.backend)
    scene_dir = Path(task["scene_dir"])
    ctx = V1.build_context(
        scene_dir, seed=int(task["drive_seed"]), backend=args.backend,
        shared=shared)
    try:
        result = _scan_selector_scene(
            family=str(task["family"]), scene_dir=scene_dir,
            requested_strata=STRATA, ctx=ctx)
        payload = _build_selector_feasibility_scene_shard(
            task=task, scene_result=result,
            task_census_digest=task_census_digest, source=source,
            successor_selection_digest=successor_selection_digest,
            exclusion_binding_digest=exclusion_binding_digest,
            runtime_s=time.time() - started)
        _validate_selector_feasibility_scene_shard(
            payload, expected_task=task,
            expected_task_census_digest=task_census_digest, source=source,
            expected_successor_selection_digest=successor_selection_digest,
            expected_exclusion_binding_digest=exclusion_binding_digest)
        atomic_json(path, payload)
        return payload
    finally:
        # Native teardown has historically SIGSEGV'd.  A complete scene census
        # must be durable before either reference is released or GC is forced.
        _FIELD_CACHE.clear()
        del ctx
        gc.collect()


def _selector_feasibility_family_row(
        summary: dict[str, Any], requested_strata: Sequence[str]
        ) -> dict[str, Any]:
    """Project one exhaustive family scan into the binding receipt schema."""

    strata: dict[str, Any] = {}
    for stratum in requested_strata:
        evidence = summary["per_stratum"][stratum]
        strata[stratum] = {
            "required_distinct_scenes": evidence["required_distinct_scenes"],
            "eligible_distinct_scenes": evidence["eligible_distinct_scenes"],
            "verdict": "PASS" if evidence["quota_pass"] else "FAIL",
            "distributions": evidence["distributions"],
            "scene_evidence": evidence["scene_evidence"],
        }
    return {
        "family": summary["family"],
        "allowed_scene_count": summary["allowed_scene_count"],
        "scanned_scene_count": summary["scanned_scene_count"],
        "all_allowed_scenes_scanned": (
            summary["scanned_scene_count"] == summary["allowed_scene_count"]),
        "verdict": "PASS" if summary["all_requested_quotas_pass"] else "FAIL",
        "strata": strata,
        "rejection_counts": summary["rejection_counts"],
    }


def build_selector_feasibility_receipt_from_family_reductions(
        *, reductions: Sequence[dict[str, Any]], source: dict[str, Any],
        successor_selection_digest: str,
        exclusion_binding_digest: str,
        task_census: dict[str, Any]) -> dict[str, Any]:
    """Pure deterministic reducer over eight validated exhaustive censuses."""

    _verify_self_digest(
        task_census, "state_selector_feasibility_task_census_digest",
        "state-selector feasibility task census")
    if (task_census.get("source_repository_commit")
            != source["source_repository_commit"]
            or task_census.get("clean_source_binding_digest")
            != canonical_digest(source)
            or task_census.get("bound_implementations_digest")
            != source["bound_implementations_digest"]
            or task_census.get("successor_selection_digest")
            != successor_selection_digest
            or task_census.get("state_selector_amendment_digest")
            != STATE_SELECTOR.state_selector_amendment_digest()
            or task_census.get("exclusion_binding_digest")
            != exclusion_binding_digest):
        raise RuntimeError("selector-feasibility reducer task census binding failed")
    by_family: dict[str, dict[str, Any]] = {}
    task_census_digest = str(
        task_census["state_selector_feasibility_task_census_digest"])
    for reduction in reductions:
        family = str(reduction.get("family", ""))
        if family in by_family:
            raise RuntimeError("selector-feasibility reducer received a repeated family")
        _verify_self_digest(
            reduction, "family_reduction_digest",
            f"selector-feasibility family reduction {family}")
        tasks = _selector_feasibility_family_tasks(task_census, family)
        scene_bindings = reduction.get("scene_shards")
        exact_scene_coverage = (
            isinstance(scene_bindings, list)
            and len(scene_bindings) == len(tasks)
            and all(
                isinstance(row, dict)
                and set(row) == {
                    "family", "scene_id", "scene_task_digest",
                    "scene_shard_digest"}
                and row["family"] == family
                and row["scene_id"] == task["scene_id"]
                and row["scene_task_digest"] == task["scene_task_digest"]
                and _is_sha256(row["scene_shard_digest"])
                for row, task in zip(scene_bindings, tasks)))
        family_result = reduction.get("family_result", {})
        if (reduction.get("schema")
                != "go2_scorer_fit_selector_feasibility_family_reduction_v1"
                or reduction.get("task_census_digest") != task_census_digest
                or reduction.get("scene_task_count") != len(tasks)
                or not exact_scene_coverage
                or family_result.get("family") != family
                or family_result.get("allowed_scene_count") != len(tasks)
                or family_result.get("scanned_scene_count") != len(tasks)
                or family_result.get("all_allowed_scenes_scanned") is not True
                or isinstance(reduction.get("runtime_s"), bool)
                or not isinstance(reduction.get("runtime_s"), (int, float))
                or not math.isfinite(float(reduction["runtime_s"]))
                or float(reduction["runtime_s"]) < 0.0):
            raise RuntimeError(
                f"selector-feasibility family reduction {family} is malformed")
        by_family[family] = reduction
    required_families = tuple(STATE_SELECTOR.REQUIRED_FAMILIES)
    if set(by_family) != set(required_families):
        raise RuntimeError("selector-feasibility reducer requires all eight families")
    ordered = [by_family[family] for family in required_families]
    family_rows = [reduction["family_result"] for reduction in ordered]
    scene_shard_lineage = [
        row for reduction in ordered for row in reduction["scene_shards"]
    ]
    passed = all(row["verdict"] == "PASS" for row in family_rows)
    payload = {
        "schema": SELECTOR_FEASIBILITY_SCHEMA,
        "status": (SELECTOR_FEASIBILITY_PASS_STATUS if passed
                   else "FAIL_OUTCOME_FREE_SELECTOR_FEASIBILITY"),
        "complete": True,
        "binding_receipt": True,
        "source_repository_commit": source["source_repository_commit"],
        "clean_source_binding_digest": canonical_digest(source),
        "bound_implementations_digest": source["bound_implementations_digest"],
        "successor_selection_digest": successor_selection_digest,
        "state_selector_amendment_digest":
            STATE_SELECTOR.state_selector_amendment_digest(),
        "state_selector_feasibility_task_census_digest": task_census_digest,
        "scene_task_count": task_census["scene_task_count"],
        "scene_shard_count": len(scene_shard_lineage),
        "scene_shard_lineage": scene_shard_lineage,
        "scene_shard_lineage_digest": canonical_digest(scene_shard_lineage),
        "family_count": len(family_rows),
        "strata": list(STRATA),
        "required_distinct_scenes_per_stratum": 5,
        "families": family_rows,
        "exclusion_binding_digest": exclusion_binding_digest,
        "runtime_s": round(math.fsum(
            float(reduction["runtime_s"]) for reduction in ordered), 6),
        "reducer_version": SELECTOR_FEASIBILITY_REDUCER_VERSION,
        "family_reduction_digests": {
            family: by_family[family][
                "family_reduction_digest"]
            for family in required_families
        },
        "scene_subprocess_isolation": True,
        "resume_reuses_only_valid_complete_scene_shards": True,
        "selected_state_identities_created": False,
        "candidate_outcomes_loaded": False,
        "branch_identities_created": False,
        "branches_attempted": 0,
        "frames_rendered": 0,
        "target_latents_encoded": 0,
        "scorer_training_started": False,
    }
    payload["state_selector_feasibility_receipt_digest"] = canonical_digest(payload)
    return payload


def _run_selector_feasibility_scene_subprocess(
        args: argparse.Namespace, task: dict[str, Any]) -> int:
    """Run exactly one pre-bound scene task in a fresh native process."""

    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--pool", "scorer_fit", "--stage", "selector-feasibility",
        "--family", str(task["family"]),
        "--selector-scene-id", str(task["scene_id"]),
        "--backend", str(args.backend),
    ]
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    completed = subprocess.run(
        command, cwd=ROOT, env=environment, check=False)
    return int(completed.returncode)


def _collect_selector_feasibility_scene_shards(
        *, args: argparse.Namespace, out: Path,
        tasks: Sequence[dict[str, Any]], task_census_digest: str,
        source: dict[str, Any], successor_selection_digest: str,
        exclusion_binding_digest: str) -> list[dict[str, Any]]:
    """Resume exact complete scenes; a process exit is never eligibility data."""

    shards: list[dict[str, Any]] = []
    for task in tasks:
        path = _selector_feasibility_scene_shard_path(out, task)
        shard = _load_completed_selector_feasibility_scene_shard(
            path, expected_task=task, task_census_digest=task_census_digest,
            source=source,
            successor_selection_digest=successor_selection_digest,
            exclusion_binding_digest=exclusion_binding_digest)
        if shard is None:
            if _outcome_generation_started(out):
                state = "invalid" if path.exists() else "missing"
                raise RuntimeError(
                    f"selector-feasibility scene shard is {state} after outcomes")
            if path.exists():
                preserved = _preserve_invalid(
                    path, out, "selector-feasibility-scene-invalid")
                print(f"[recovery] preserved invalid scene census {preserved}",
                      flush=True)
            print(
                "[selector-feasibility] isolated exhaustive scene census: "
                f"{task['family']}/{task['scene_id']}", flush=True)
            return_code = _run_selector_feasibility_scene_subprocess(args, task)
            shard = _load_completed_selector_feasibility_scene_shard(
                path, expected_task=task, task_census_digest=task_census_digest,
                source=source,
                successor_selection_digest=successor_selection_digest,
                exclusion_binding_digest=exclusion_binding_digest)
            if shard is None:
                raise RuntimeError(
                    "isolated selector-feasibility scene "
                    f"{task['family']}/{task['scene_id']} exited {return_code} "
                    "without a valid durable census; no eligibility conclusion "
                    "was recorded")
            if return_code != 0:
                print(
                    "[recovery] retained valid atomic scene census despite "
                    f"worker return code {return_code}: "
                    f"{task['family']}/{task['scene_id']}", flush=True)
        else:
            print(
                "[selector-feasibility] retained valid exhaustive scene census: "
                f"{task['family']}/{task['scene_id']}", flush=True)
        shards.append(shard)
    return shards


def _reduce_selector_feasibility_family_scene_shards(
        *, family: str, tasks: Sequence[dict[str, Any]],
        shards: Sequence[dict[str, Any]], task_census_digest: str,
        source: dict[str, Any], successor_selection_digest: str,
        exclusion_binding_digest: str) -> dict[str, Any]:
    """Deterministically reduce the exact scene census for one family."""

    if len(tasks) != len(shards):
        raise RuntimeError(f"selector-feasibility family {family} scene count changed")
    by_task_digest: dict[str, dict[str, Any]] = {}
    for shard in shards:
        digest = str(shard.get("task", {}).get("scene_task_digest", ""))
        if digest in by_task_digest:
            raise RuntimeError(
                f"selector-feasibility family {family} repeats a scene shard")
        by_task_digest[digest] = shard
    expected_task_digests = [str(task["scene_task_digest"]) for task in tasks]
    if set(by_task_digest) != set(expected_task_digests):
        raise RuntimeError(
            f"selector-feasibility family {family} scene task coverage changed")
    scene_evidence: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = {}
    scene_bindings: list[dict[str, Any]] = []
    runtime_values: list[float] = []
    for task in tasks:
        shard = by_task_digest[str(task["scene_task_digest"])]
        _validate_selector_feasibility_scene_shard(
            shard, expected_task=task,
            expected_task_census_digest=task_census_digest, source=source,
            expected_successor_selection_digest=successor_selection_digest,
            expected_exclusion_binding_digest=exclusion_binding_digest)
        result = shard["scene_result"]
        scene_evidence.extend(result["scene_evidence"])
        for reason, count in result["rejection_counts"].items():
            rejection_counts[reason] = rejection_counts.get(reason, 0) + int(count)
        runtime_values.append(float(shard["runtime_s"]))
        scene_bindings.append({
            "family": family,
            "scene_id": task["scene_id"],
            "scene_task_digest": task["scene_task_digest"],
            "scene_shard_digest":
                shard["state_selector_feasibility_scene_shard_digest"],
        })
    summary = build_selector_feasibility_summary(
        family=family, allowed_scene_count=len(tasks),
        requested_strata=STRATA, scene_evidence=scene_evidence,
        rejection_counts=rejection_counts)
    payload = {
        "schema": "go2_scorer_fit_selector_feasibility_family_reduction_v1",
        "family": family,
        "task_census_digest": task_census_digest,
        "scene_task_count": len(tasks),
        "scene_shards": scene_bindings,
        "family_result": _selector_feasibility_family_row(summary, STRATA),
        "runtime_s": round(math.fsum(runtime_values), 6),
    }
    payload["family_reduction_digest"] = canonical_digest(payload)
    return payload


def _reduce_selector_feasibility_families(
        *, args: argparse.Namespace, out: Path, source: dict[str, Any],
        successor_selection_digest: str,
        exclusion_binding_digest: str,
        task_census: dict[str, Any]) -> list[dict[str, Any]]:
    """Reduce exact scene shards family-by-family without loading Genesis."""

    reductions: list[dict[str, Any]] = []
    task_census_digest = str(
        task_census["state_selector_feasibility_task_census_digest"])
    for family in STATE_SELECTOR.REQUIRED_FAMILIES:
        tasks = _selector_feasibility_family_tasks(task_census, family)
        scene_shards = _collect_selector_feasibility_scene_shards(
            args=args, out=out, tasks=tasks,
            task_census_digest=task_census_digest, source=source,
            successor_selection_digest=successor_selection_digest,
            exclusion_binding_digest=exclusion_binding_digest)
        reductions.append(_reduce_selector_feasibility_family_scene_shards(
            family=family, tasks=tasks, shards=scene_shards,
            task_census_digest=task_census_digest, source=source,
            successor_selection_digest=successor_selection_digest,
            exclusion_binding_digest=exclusion_binding_digest))
    return reductions


def _load_completed_selector_feasibility(
        path: Path, *, source: dict[str, Any],
        successor_selection_digest: str,
        exclusion_binding_digest: str,
        task_census: dict[str, Any]) -> dict[str, Any] | None:
    """Rebuild the binding receipt from every current durable scene shard."""

    if not path.is_file():
        return None
    try:
        existing = json.loads(path.read_text())
        task_census_digest = str(
            task_census["state_selector_feasibility_task_census_digest"])
        reductions: list[dict[str, Any]] = []
        for family in STATE_SELECTOR.REQUIRED_FAMILIES:
            tasks = _selector_feasibility_family_tasks(task_census, family)
            shards: list[dict[str, Any]] = []
            for task in tasks:
                shard = _load_completed_selector_feasibility_scene_shard(
                    _selector_feasibility_scene_shard_path(path.parent, task),
                    expected_task=task,
                    task_census_digest=task_census_digest, source=source,
                    successor_selection_digest=successor_selection_digest,
                    exclusion_binding_digest=exclusion_binding_digest)
                if shard is None:
                    return None
                shards.append(shard)
            reductions.append(_reduce_selector_feasibility_family_scene_shards(
                family=family, tasks=tasks, shards=shards,
                task_census_digest=task_census_digest, source=source,
                successor_selection_digest=successor_selection_digest,
                exclusion_binding_digest=exclusion_binding_digest))
        expected = build_selector_feasibility_receipt_from_family_reductions(
            reductions=reductions, source=source,
            successor_selection_digest=successor_selection_digest,
            exclusion_binding_digest=exclusion_binding_digest,
            task_census=task_census)
        if existing != expected:
            return None
        if existing["status"] == SELECTOR_FEASIBILITY_PASS_STATUS:
            STATE_SELECTOR.PREDECESSOR.validate_state_selector_feasibility_receipt(
                existing,
                expected_source_commit=str(source["source_repository_commit"]),
                expected_successor_selection_digest=successor_selection_digest,
            )
        return existing
    except (OSError, TypeError, ValueError, RuntimeError, json.JSONDecodeError):
        return None


def stage_selector_feasibility(args: argparse.Namespace) -> int:
    """Run the outcome-free all-scene feasibility gate or a scoped diagnostic."""

    if args.pool != "scorer_fit":
        raise RuntimeError("selector feasibility is defined only for scorer_fit")
    if args.backend != "cpu":
        raise RuntimeError("the frozen selector feasibility backend is cpu")
    STATE_SELECTOR.validate_authority_artifacts()
    source = clean_source_binding()
    if source.get("source_repository_clean") is not True:
        raise RuntimeError("selector feasibility requires a clean source repository")
    successor_digest = selection_digest()
    selector_scene_id = getattr(args, "selector_scene_id", None)
    if (selector_scene_id is not None
            and (args.family is None or args.stratum is not None)):
        raise RuntimeError(
            "--selector-scene-id requires one --family and no --stratum")
    requested_families = ([str(args.family)] if args.family is not None
                          else list(STATE_SELECTOR.REQUIRED_FAMILIES))
    requested_strata = ([str(args.stratum)] if args.stratum is not None
                        else list(STATE_SELECTOR.REQUIRED_STRATA))
    binding_run = (
        args.family is None and args.stratum is None
        and selector_scene_id is None)
    scene_worker_run = selector_scene_id is not None
    out = OUT_ROOT / "scorer_fit"
    out.mkdir(parents=True, exist_ok=True)
    if binding_run:
        path = out / SELECTOR_FEASIBILITY_RECEIPT_NAME
    elif scene_worker_run:
        path = None
    else:
        path = out / ("state_selector_feasibility_diagnostic_"
                      + "-".join(requested_families) + "_"
                      + "-".join(requested_strata) + ".json")

    pool, exclusion = scene_pool("scorer_fit")
    unknown = sorted(set(requested_families) - set(pool))
    if unknown:
        raise RuntimeError(f"unknown selector-feasibility families: {unknown}")
    exclusion_digest = canonical_digest(exclusion)
    census: dict[str, Any] | None = None
    if not scene_worker_run:
        census = _issue_selector_feasibility_task_census(
            out=out, pool=pool, source=source,
            successor_selection_digest=successor_digest,
            exclusion_binding_digest=exclusion_digest)
    elif scene_worker_run:
        census_path = out / SELECTOR_FEASIBILITY_TASK_CENSUS_NAME
        if not census_path.is_file():
            raise RuntimeError(
                "isolated scene worker requires the frozen task census")
        census = json.loads(census_path.read_text())
        _validate_selector_feasibility_task_census(
            census, pool=pool, source=source,
            successor_selection_digest=successor_digest,
            exclusion_binding_digest=exclusion_digest)

    if binding_run:
        assert census is not None and path is not None
        existing = _load_completed_selector_feasibility(
            path, source=source,
            successor_selection_digest=successor_digest,
            exclusion_binding_digest=exclusion_digest,
            task_census=census)
        if existing is not None:
            print(json.dumps(existing, indent=2, sort_keys=True))
            return (0 if existing.get("status")
                    == SELECTOR_FEASIBILITY_PASS_STATUS else 1)

    if binding_run:
        assert census is not None and path is not None
        reductions = _reduce_selector_feasibility_families(
            args=args, out=out, source=source,
            successor_selection_digest=successor_digest,
            exclusion_binding_digest=exclusion_digest,
            task_census=census)
        payload = build_selector_feasibility_receipt_from_family_reductions(
            reductions=reductions, source=source,
            successor_selection_digest=successor_digest,
            exclusion_binding_digest=exclusion_digest,
            task_census=census)
        if path.exists():
            if _outcome_generation_started(out):
                raise RuntimeError(
                    "selector-feasibility receipt changed after outcomes started")
            _preserve_invalid(path, out, "selector-feasibility-superseded")
        atomic_json(path, payload)
        passed = payload["status"] == SELECTOR_FEASIBILITY_PASS_STATUS
        if passed:
            STATE_SELECTOR.PREDECESSOR.validate_state_selector_feasibility_receipt(
                payload,
                expected_source_commit=str(source["source_repository_commit"]),
                expected_successor_selection_digest=successor_digest)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if passed else 1

    if scene_worker_run:
        assert census is not None
        family = requested_families[0]
        matches = [
            task for task in _selector_feasibility_family_tasks(census, family)
            if task["scene_id"] == selector_scene_id
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"scene task lookup is ambiguous for {family}/{selector_scene_id}")
        task = matches[0]
        path = _selector_feasibility_scene_shard_path(out, task)
        census_digest = str(
            census["state_selector_feasibility_task_census_digest"])
        existing = _load_completed_selector_feasibility_scene_shard(
            path, expected_task=task, task_census_digest=census_digest,
            source=source, successor_selection_digest=successor_digest,
            exclusion_binding_digest=exclusion_digest)
        if existing is not None:
            print(json.dumps(existing, indent=2, sort_keys=True))
            return 0
        if _outcome_generation_started(out):
            state = "invalid" if path.exists() else "missing"
            raise RuntimeError(
                f"selector-feasibility scene shard is {state} after outcomes")
        if path.exists():
            _preserve_invalid(path, out, "selector-feasibility-scene-invalid")
        payload = _execute_selector_feasibility_scene_worker(
            args=args, task=task, path=path,
            task_census_digest=census_digest, source=source,
            successor_selection_digest=successor_digest,
            exclusion_binding_digest=exclusion_digest)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    # Scoped diagnostics reuse the same exact per-scene workers and merely
    # reduce a requested view; they cannot satisfy the binding all-family gate.
    assert census is not None
    census_digest = str(
        census["state_selector_feasibility_task_census_digest"])
    families: list[dict[str, Any]] = []
    runtime_values: list[float] = []
    for family in requested_families:
        tasks = _selector_feasibility_family_tasks(census, family)
        scene_shards = _collect_selector_feasibility_scene_shards(
            args=args, out=out, tasks=tasks,
            task_census_digest=census_digest, source=source,
            successor_selection_digest=successor_digest,
            exclusion_binding_digest=exclusion_digest)
        evidence = [
            row for shard in scene_shards
            for row in shard["scene_result"]["scene_evidence"]
            if row["stratum"] in requested_strata
        ]
        rejections: dict[str, int] = {}
        for shard in scene_shards:
            runtime_values.append(float(shard["runtime_s"]))
            for reason, count in shard["scene_result"]["rejection_counts"].items():
                rejections[reason] = rejections.get(reason, 0) + int(count)
        families.append(build_selector_feasibility_summary(
            family=family, allowed_scene_count=len(tasks),
            requested_strata=requested_strata, scene_evidence=evidence,
            rejection_counts=rejections))
    family_rows = [
        _selector_feasibility_family_row(summary, requested_strata)
        for summary in families
    ]
    passed = all(row["verdict"] == "PASS" for row in family_rows)
    payload = {
        "schema": "go2_scorer_fit_state_selector_feasibility_diagnostic_v1",
        "status": ("PASS_OUTCOME_FREE_SCOPED_FEASIBILITY" if passed
                   else "FAIL_OUTCOME_FREE_SELECTOR_FEASIBILITY"),
        "complete": True,
        "binding_receipt": False,
        "source_repository_commit": source["source_repository_commit"],
        "clean_source_binding_digest": canonical_digest(source),
        "bound_implementations_digest": source["bound_implementations_digest"],
        "successor_selection_digest": successor_digest,
        "state_selector_amendment_digest":
            STATE_SELECTOR.state_selector_amendment_digest(),
        "family_count": len(family_rows),
        "strata": list(requested_strata),
        "required_distinct_scenes_per_stratum": 5,
        "families": family_rows,
        "exclusion_binding_digest": exclusion_digest,
        "runtime_s": round(math.fsum(runtime_values), 6),
        "selected_state_identities_created": False,
        "candidate_outcomes_loaded": False,
        "branch_identities_created": False,
        "branches_attempted": 0,
        "frames_rendered": 0,
        "target_latents_encoded": 0,
        "scorer_training_started": False,
    }
    payload["state_selector_feasibility_diagnostic_digest"] = \
        canonical_digest(payload)
    assert path is not None
    if path.exists():
        _preserve_invalid(path, out, "selector-feasibility-superseded")
    atomic_json(path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 1


def _load_frozen_selector_feasibility_lineage(
        out: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Open the frozen v1 failure and exact census without reinterpreting it.

    This intentionally does not validate the old artifacts against the current
    source tree.  It validates their frozen raw/self digests and their complete
    1,284-scene lineage instead.  Requiring current-source equality would turn
    evidence reuse into an accidental request to rerun every scene.
    """

    failure_path = FROZEN_FEASIBILITY_FAILURE_REPORT_PATH
    _assert_unsealed_path(failure_path)
    receipt_path = _frozen_generated_artifact_path(
        out / SELECTOR_FEASIBILITY_RECEIPT_NAME)
    census_path = _frozen_generated_artifact_path(
        out / SELECTOR_FEASIBILITY_TASK_CENSUS_NAME)
    for label, path in (("frozen failure report", failure_path),
                        ("frozen feasibility receipt", receipt_path),
                        ("frozen task census", census_path)):
        if not path.is_file():
            raise RuntimeError(f"{label} is missing")
    if file_sha256(failure_path) != FROZEN_FEASIBILITY_FAILURE_REPORT_RAW_SHA256:
        raise RuntimeError("frozen feasibility failure report raw digest changed")
    failure = json.loads(failure_path.read_text())
    _verify_self_digest(failure, "failure_report_digest",
                        "frozen feasibility failure report")
    if (failure.get("failure_report_digest")
            != FROZEN_FEASIBILITY_FAILURE_REPORT_DIGEST):
        raise RuntimeError("frozen feasibility failure report binding changed")

    if file_sha256(receipt_path) != FROZEN_FEASIBILITY_RECEIPT_RAW_SHA256:
        raise RuntimeError("frozen feasibility receipt raw digest changed")
    receipt = json.loads(receipt_path.read_text())
    _verify_self_digest(receipt, "state_selector_feasibility_receipt_digest",
                        "frozen feasibility receipt")
    if (receipt.get("status") != "FAIL_OUTCOME_FREE_SELECTOR_FEASIBILITY"
            or receipt.get("state_selector_feasibility_receipt_digest")
            != FROZEN_FEASIBILITY_RECEIPT_DIGEST
            or receipt.get("scene_task_count")
            != FROZEN_FEASIBILITY_SCENE_SHARD_COUNT
            or receipt.get("scene_shard_count")
            != FROZEN_FEASIBILITY_SCENE_SHARD_COUNT):
        raise RuntimeError("frozen feasibility receipt lineage changed")

    census = json.loads(census_path.read_text())
    _verify_self_digest(census, "state_selector_feasibility_task_census_digest",
                        "frozen selector task census")
    if (census.get("state_selector_feasibility_task_census_digest")
            != FROZEN_FEASIBILITY_TASK_CENSUS_DIGEST
            or census.get("scene_task_count")
            != FROZEN_FEASIBILITY_SCENE_SHARD_COUNT):
        raise RuntimeError("frozen selector task census changed")
    return failure, receipt, census


def _frozen_selector_scene_shards(
        *, out: Path, receipt: dict[str, Any], census: dict[str, Any]
        ) -> dict[tuple[str, str], dict[str, Any]]:
    """Reopen all old shards by their receipt lineage, honoring custody."""

    lineage = receipt.get("scene_shard_lineage")
    if (not isinstance(lineage, list)
            or len(lineage) != FROZEN_FEASIBILITY_SCENE_SHARD_COUNT
            or canonical_digest(lineage)
            != receipt.get("scene_shard_lineage_digest")):
        raise RuntimeError("frozen selector scene lineage is malformed")
    tasks = {
        (str(task["family"]), str(task["scene_id"])): task
        for family_row in census["families"] for task in family_row["tasks"]
    }
    if len(tasks) != FROZEN_FEASIBILITY_SCENE_SHARD_COUNT:
        raise RuntimeError("frozen selector task census repeats a scene")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for binding in lineage:
        key = (str(binding.get("family", "")),
               str(binding.get("scene_id", "")))
        task = tasks.get(key)
        if (task is None
                or binding.get("scene_task_digest")
                != task.get("scene_task_digest")):
            raise RuntimeError("frozen selector shard/task lineage mismatch")
        path = _frozen_generated_artifact_path(
            _selector_feasibility_scene_shard_path(out, task))
        if not path.is_file():
            raise RuntimeError(f"frozen selector shard is missing: {key}")
        shard = json.loads(path.read_text())
        _verify_self_digest(shard, "state_selector_feasibility_scene_shard_digest",
                            f"frozen selector scene shard {key}")
        if (shard.get("state_selector_feasibility_scene_shard_digest")
                != binding.get("scene_shard_digest")
                or shard.get("task") != task
                or shard.get("complete") is not True
                or any(shard.get(field) not in (False, 0)
                       for field in SELECTOR_FEASIBILITY_FORBIDDEN_FIELDS)):
            raise RuntimeError(f"frozen selector scene shard binding failed: {key}")
        result[key] = shard
    if set(result) != set(tasks):
        raise RuntimeError("frozen selector scene shard coverage is incomplete")
    return result


def _reachability_scene_shard_path(out: Path, task: dict[str, Any]) -> Path:
    return (out / REACHABILITY_FEASIBILITY_SCENE_SHARD_ROOT / task["family"]
            / f"{task['scene_task_digest']}.json")


def _completion_rotation_evidence(
        *, graph_hops: int, distance: float, bearing: float,
        task_status: dict[str, Any], previous_applied_command: Sequence[float]
        ) -> list[dict[str, Any]]:
    """Evaluate every unchanged allowed six-candidate subset, without a branch."""

    rows: list[dict[str, Any]] = []
    for rotation_index, candidate_indices in enumerate(ALLOC.ROTATION_BLOCKS):
        evidence = STATE_SELECTOR.completion_enriched_eligibility(
            graph_hops=int(graph_hops), reachable=math.isfinite(float(distance)),
            continuous_geodesic_m=float(distance),
            bearing_body_rad=float(bearing), task_status=task_status,
            candidate_indices=list(candidate_indices),
            previous_applied_command=list(previous_applied_command))
        rows.append({
            "rotation_index": int(rotation_index),
            "candidate_indices": list(candidate_indices),
            "eligible": bool(evidence["eligible"]),
            "continuous_geodesic_m": float(evidence[
                "continuous_geodesic_m"]),
            "completion_radius_m": float(evidence["completion_radius_m"]),
            "continuous_geodesic_gap_m": float(evidence[
                "continuous_geodesic_gap_m"]),
            "l_max_m": float(evidence["l_max_m"]),
            "l_max_candidate_indices": list(evidence[
                "l_max_candidate_indices"]),
            "candidate_path_lengths_m": [dict(row) for row in evidence[
                "candidate_path_lengths_m"]],
            "rejection_reasons": list(evidence["rejection_reasons"]),
        })
    if len(rows) != 12 or any(
            row["candidate_indices"] != list(ALLOC.ROTATION_BLOCKS[index])
            for index, row in enumerate(rows)):
        raise RuntimeError("reachability census changed the frozen allocation blocks")
    return rows


def _scan_reachability_selector_scene(
        *, family: str, scene_dir: Path, ctx: Any) -> dict[str, Any]:
    """Redrive one small-maze scene and retain pre-outcome rotation evidence."""

    if family != REACHABILITY_REDRIVE_FAMILY:
        raise RuntimeError("reachability redrive is restricted to small_enclosed_maze")
    topology = V12.link_topology(ctx)
    ctx.begin_episode()
    rejections: dict[str, int] = {}
    evidence: dict[str, Any] | None = None
    for block_idx in range(WARMUP_BLOCKS_MAX):
        ctx.drive_one_block()
        if block_idx + 1 < WARMUP_BLOCKS_MIN:
            continue
        local: dict[str, int] = {}
        try:
            boundary = V1.assert_canonical_boundary(ctx)
        except V1.BoundaryRefused:
            local["boundary_refused"] = 1
        else:
            if ctx.episode_ticks < PROPRIO_HISTORY - 1:
                local["insufficient_proprioceptive_history"] = 1
            else:
                (x, y), yaw, _z = ctx.pose()
                hit = ctx.scene_graph.locate((x, y))
                if float(hit.distance_m) > V1.LOCATE_MAX_DISTANCE_M:
                    local["locate_distance_gt_2m"] = 1
                elif V12._contact_count(ctx, topology) > 0:
                    local["already_in_disallowed_contact"] = 1
                else:
                    graph = ctx.scene_graph
                    blocked = getattr(graph, "nav_blocked_cells", frozenset())
                    cell = int(hit.cell_id)
                    from analyze_go2_closed_loop_quality import \
                        _body_probe_configuration_clearance_m
                    body_clearance = float(_body_probe_configuration_clearance_m(
                        ctx.grid, [x, y], yaw,
                        body_forward_m=V1.CONTACT_BODY_FORWARD_M,
                        body_half_width_m=V1.CONTACT_BODY_HALF_WIDTH_M,
                        body_probe_margin_m=V1.CONTACT_BODY_PROBE_MARGIN_M))
                    clearance = float(graph.clearance_to_walls((x, y)))
                    previous = np.asarray(
                        ctx.runner._last_executed, dtype=np.float64)[0].tolist()
                    eligible_goals: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
                    saw_reachable = False
                    for name, goal_cell in sorted(
                            graph.landmark_cells, key=lambda row: str(row[0])):
                        hops = graph.bfs_distance(
                            cell, int(goal_cell), transit_blocked=blocked)
                        if hops is None:
                            local["completion_unreachable"] = local.get(
                                "completion_unreachable", 0) + 1
                            continue
                        saw_reachable = True
                        field = geodesic_field(ctx, int(goal_cell), blocked)
                        distance = float(field.remaining_distance((x, y), cell))
                        centre = graph.cell_center(int(goal_cell))
                        bearing, range_m = landmark_bearing_range(ctx, centre)
                        task_status = _snapshot_task_status(ctx, int(goal_cell))
                        rotation_vector = \
                            STATE_SELECTOR.completion_rotation_eligibility_vector(
                            graph_hops=int(hops),
                            reachable=math.isfinite(distance),
                            continuous_geodesic_m=distance,
                            bearing_body_rad=bearing, task_status=task_status,
                            previous_applied_command=previous)
                        rotations = [dict(row) for row in
                                     rotation_vector["rotations"]]
                        invariant_reasons = sorted({
                            reason for row in rotations
                            for reason in row["rejection_reasons"]
                            if reason != (
                                "completion_geodesic_gap_gt_allocated_subset_l_max"
                            )
                        })
                        for reason in invariant_reasons:
                            local[reason] = local.get(reason, 0) + 1
                        passing = [row for row in rotations if row["eligible"]]
                        if not passing:
                            if not invariant_reasons:
                                local[
                                    "completion_gap_exceeds_every_allowed_subset_l_max"
                                ] = local.get(
                                    "completion_gap_exceeds_every_allowed_subset_l_max",
                                    0) + 1
                            continue
                        goal_type = _goal_material(ctx, str(name))
                        if goal_type is None:
                            local[
                                "bound_landmark_material_missing_or_ambiguous"
                            ] = local.get(
                                "bound_landmark_material_missing_or_ambiguous", 0
                            ) + 1
                            continue
                        row = {
                            "family": family,
                            "scene_id": scene_dir.name,
                            "stratum": "completion_enriched",
                            "first_eligible_block": int(block_idx + 1),
                            "source_step": int(boundary["source_step"]),
                            "boundary": boundary,
                            "cell_id": int(cell),
                            "episode_id": int(
                                ctx.runner.episode_states[0].episode_id),
                            "episode_cluster_id": (
                                f"{scene_dir.name}/env0/ep"
                                f"{int(ctx.runner.episode_states[0].episode_id)}"
                            ),
                            "goal_landmark_id": str(name),
                            "goal_landmark_cell": int(goal_cell),
                            "goal_material_id": str(goal_type),
                            "continuous_geodesic_m": distance,
                            "completion_radius_m": float(
                                STATE_SELECTOR.COMPLETION_RADIUS_M),
                            "continuous_geodesic_gap_m": max(
                                distance - float(STATE_SELECTOR.COMPLETION_RADIUS_M),
                                0.0),
                            "abs_bearing_rad": abs(float(bearing)),
                            "bearing_body_rad": float(bearing),
                            "range_m": float(range_m),
                            "goal_landmark_xy_m": [
                                float(centre[0]), float(centre[1])],
                            "graph_hops_diagnostic": int(hops),
                            "body_clearance_m": body_clearance,
                            "clearance_m": clearance,
                            "previous_applied_command": [float(v) for v in previous],
                            "allocation_rotation_evidence": rotations,
                            "completion_rotation_eligibility_vector":
                                rotation_vector,
                            "eligible_rotation_indices": [
                                row["candidate_rotation_index"]
                                for row in passing],
                            "passes_any_allowed_allocation": True,
                            "passes_every_allowed_allocation": len(passing) == 12,
                            "admitted_by_horizon_reachability_amendment": bool(
                                distance > STATE_SELECTOR.COMPLETION_RADIUS_M),
                            "snapshot_task_status": task_status,
                        }
                        eligible_goals.append((
                            (distance, str(name), int(goal_cell), int(hops)), row))
                    if eligible_goals:
                        evidence = min(eligible_goals, key=lambda item: item[0])[1]
                        evidence["eligible_designated_goal_count_at_first_eligible_snapshot"] = \
                            len(eligible_goals)
                    elif saw_reachable:
                        local["no_completion_enriched_goal"] = local.get(
                            "no_completion_enriched_goal", 0) + 1
                    else:
                        local["no_reachable_landmark"] = local.get(
                            "no_reachable_landmark", 0) + 1
        for reason, count in local.items():
            rejections[reason] = rejections.get(reason, 0) + int(count)
        if evidence is not None:
            break
    return {
        "family": family,
        "scene_id": scene_dir.name,
        "completion_scene_evidence": ([] if evidence is None else [evidence]),
        "rejection_counts": {
            str(key): int(value) for key, value in sorted(rejections.items())
        },
    }


def _build_reachability_scene_shard(
        *, task: dict[str, Any], predecessor_shard_digest: str,
        scene_result: dict[str, Any], source: dict[str, Any], runtime_s: float
        ) -> dict[str, Any]:
    payload = {
        "schema": REACHABILITY_FEASIBILITY_SCENE_SHARD_SCHEMA,
        "status": REACHABILITY_FEASIBILITY_SCENE_SHARD_STATUS,
        "complete": True,
        "binding_receipt": False,
        "source_repository_commit": source["source_repository_commit"],
        "clean_source_binding_digest": canonical_digest(source),
        "bound_implementations_digest": source["bound_implementations_digest"],
        "successor_selection_digest": selection_digest(),
        "state_selector_amendment_digest":
            STATE_SELECTOR.state_selector_amendment_digest(),
        "candidate_allocation_amendment_digest":
            ALLOC.allocation_amendment_digest(),
        "frozen_predecessor_feasibility_receipt_digest":
            FROZEN_FEASIBILITY_RECEIPT_DIGEST,
        "frozen_predecessor_scene_shard_digest": predecessor_shard_digest,
        "task": task,
        "scene_result": scene_result,
        "runtime_s": round(float(runtime_s), 6),
        "selected_state_identities_created": False,
        "candidate_outcomes_loaded": False,
        "branch_identities_created": False,
        "branches_attempted": 0,
        "frames_rendered": 0,
        "target_latents_encoded": 0,
        "scorer_training_started": False,
    }
    payload["state_selector_reachability_scene_shard_digest"] = \
        canonical_digest(payload)
    return payload


def _validate_reachability_scene_shard(
        shard: dict[str, Any], *, expected_task: dict[str, Any],
        predecessor_shard_digest: str, source: dict[str, Any]) -> None:
    _verify_self_digest(
        shard, "state_selector_reachability_scene_shard_digest",
        f"reachability scene shard {expected_task['scene_id']}")
    if (shard.get("schema") != REACHABILITY_FEASIBILITY_SCENE_SHARD_SCHEMA
            or shard.get("status") != REACHABILITY_FEASIBILITY_SCENE_SHARD_STATUS
            or shard.get("complete") is not True
            or shard.get("binding_receipt") is not False
            or shard.get("source_repository_commit")
            != source["source_repository_commit"]
            or shard.get("clean_source_binding_digest") != canonical_digest(source)
            or shard.get("bound_implementations_digest")
            != source["bound_implementations_digest"]
            or shard.get("successor_selection_digest") != selection_digest()
            or shard.get("state_selector_amendment_digest")
            != STATE_SELECTOR.state_selector_amendment_digest()
            or shard.get("candidate_allocation_amendment_digest")
            != ALLOC.allocation_amendment_digest()
            or shard.get("frozen_predecessor_feasibility_receipt_digest")
            != FROZEN_FEASIBILITY_RECEIPT_DIGEST
            or shard.get("frozen_predecessor_scene_shard_digest")
            != predecessor_shard_digest
            or shard.get("task") != expected_task
            or any(shard.get(field) not in (False, 0)
                   for field in SELECTOR_FEASIBILITY_FORBIDDEN_FIELDS)):
        raise RuntimeError("reachability scene shard binding failed")
    result = shard.get("scene_result")
    if (not isinstance(result, dict)
            or result.get("family") != REACHABILITY_REDRIVE_FAMILY
            or result.get("scene_id") != expected_task["scene_id"]
            or not isinstance(result.get("completion_scene_evidence"), list)
            or len(result["completion_scene_evidence"]) > 1
            or not isinstance(result.get("rejection_counts"), dict)):
        raise RuntimeError("reachability scene result is malformed")
    for row in result["completion_scene_evidence"]:
        rotations = row.get("allocation_rotation_evidence")
        if (not isinstance(rotations, list) or len(rotations) != 12
                or row.get("passes_any_allowed_allocation") is not True
                or [entry.get("candidate_rotation_index") for entry in rotations]
                != list(range(12))
                or any(entry.get("candidate_indices")
                       != list(ALLOC.ROTATION_BLOCKS[index])
                       for index, entry in enumerate(rotations))):
            raise RuntimeError("reachability rotation evidence is malformed")
        expected_vector = STATE_SELECTOR.completion_rotation_eligibility_vector(
            graph_hops=int(row["graph_hops_diagnostic"]),
            reachable=math.isfinite(float(row["continuous_geodesic_m"])),
            continuous_geodesic_m=float(row["continuous_geodesic_m"]),
            bearing_body_rad=float(row["bearing_body_rad"]),
            task_status=row["snapshot_task_status"],
            previous_applied_command=row["previous_applied_command"])
        if (row.get("completion_rotation_eligibility_vector")
                != expected_vector
                or rotations != expected_vector["rotations"]
                or row.get("eligible_rotation_indices")
                != expected_vector["eligible_rotation_indices"]
                or row.get("passes_any_allowed_allocation")
                != expected_vector["eligible_under_at_least_one_rotation"]
                or row.get("passes_every_allowed_allocation")
                != expected_vector["eligible_under_every_rotation"]):
            raise RuntimeError(
                "reachability scene evidence arithmetic reconstruction failed")


def _load_reachability_scene_shard(
        path: Path, *, expected_task: dict[str, Any],
        predecessor_shard_digest: str, source: dict[str, Any]
        ) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
        _validate_reachability_scene_shard(
            payload, expected_task=expected_task,
            predecessor_shard_digest=predecessor_shard_digest, source=source)
        return payload
    except (OSError, ValueError, TypeError, RuntimeError, json.JSONDecodeError):
        return None


def _reused_predecessor_family_row(
        family_row: dict[str, Any]) -> dict[str, Any]:
    """Reclassify an already-passing family without inventing missing fields."""

    family = str(family_row["family"])
    if family == REACHABILITY_REDRIVE_FAMILY:
        raise ValueError("small family requires its scoped reachability redrive")
    result = json.loads(json.dumps(family_row))
    completion = result["strata"]["completion_enriched"]
    rows = completion["scene_evidence"]
    if (completion["verdict"] != "PASS" or len(rows) < 5
            or any(float(row["continuous_geodesic_m"])
                   > float(STATE_SELECTOR.COMPLETION_RADIUS_M) for row in rows)):
        raise RuntimeError(
            f"cached completion evidence is not mask-independent for {family}")
    result["provenance"] = "REUSED_FROZEN_1284_SCENE_CENSUS"
    for stratum in STATE_SELECTOR.REQUIRED_STRATA:
        result["strata"][stratum]["eligible_designated_goal_count"] = {
            "available": False,
            "reason": "V1_CENSUS_DID_NOT_RETAIN_GOAL_LEVEL_TOTALS",
        }
    completion["reachability_reclassification"] = {
        "status": "PASS_MASK_INDEPENDENT_ALREADY_WITHIN_COMPLETION_RADIUS",
        "candidate_subset_needed_for_verdict": False,
        "eligible_distinct_scenes": len(rows),
        "redrive_performed": False,
        "graph_hops_zero_retained_scene_count": sum(
            int(row["graph_hops_diagnostic"]) == 0 for row in rows),
        "graph_hops_positive_retained_scene_count": sum(
            int(row["graph_hops_diagnostic"]) > 0 for row in rows),
    }
    return result


def _small_reachability_family_row(
        *, predecessor_family_row: dict[str, Any],
        shards: Sequence[dict[str, Any]]) -> dict[str, Any]:
    evidence = [row for shard in shards
                for row in shard["scene_result"]["completion_scene_evidence"]]
    rejections: dict[str, int] = {}
    for shard in shards:
        for reason, count in shard["scene_result"]["rejection_counts"].items():
            rejections[reason] = rejections.get(reason, 0) + int(count)
    d0 = [float(row["continuous_geodesic_m"]) for row in evidence]
    gaps = [float(row["continuous_geodesic_gap_m"]) for row in evidence]
    bearings = [float(row["abs_bearing_rad"]) for row in evidence]
    signed_bearings = [float(row["bearing_body_rad"]) for row in evidence]
    signed_gaps = [float(row["continuous_geodesic_m"])
                   - float(STATE_SELECTOR.COMPLETION_RADIUS_M)
                   for row in evidence]
    graph_hops = [float(row["graph_hops_diagnostic"]) for row in evidence]
    first_blocks = [float(row["first_eligible_block"]) for row in evidence]
    all_lmax = [float(rotation["l_max_m"]) for row in evidence
                for rotation in row["allocation_rotation_evidence"]]
    min_lmax = [min(float(rotation["l_max_m"])
                    for rotation in row["allocation_rotation_evidence"])
                for row in evidence]
    max_lmax = [max(float(rotation["l_max_m"])
                    for rotation in row["allocation_rotation_evidence"])
                for row in evidence]
    completion = {
        "required_distinct_scenes": 5,
        "eligible_distinct_scenes": len(evidence),
        "verdict": "PASS" if len(evidence) >= 5 else "FAIL",
        "actual_allocated_subset_verification":
            "MANDATORY_POST_IDENTITY_PRE_OUTCOME",
        "distributions": {
            "continuous_geodesic_m_d0": _metric_distribution(d0),
            "continuous_geodesic_gap_m_d0_minus_0_75_clamped":
                _metric_distribution(gaps),
            "d0_minus_0_75_m_signed": _metric_distribution(signed_gaps),
            "abs_bearing_rad": _metric_distribution(bearings),
            "bearing_body_rad_signed": _metric_distribution(signed_bearings),
            "graph_hops_diagnostic": _metric_distribution(graph_hops),
            "first_eligible_block": _metric_distribution(first_blocks),
            "l_max_m_all_allowed_rotation_state_pairs":
                _metric_distribution(all_lmax),
            "minimum_l_max_m_per_state_across_allowed_rotations":
                _metric_distribution(min_lmax),
            "maximum_l_max_m_per_state_across_allowed_rotations":
                _metric_distribution(max_lmax),
        },
        "admitted_specifically_by_horizon_reachability_amendment": sum(
            bool(row["admitted_by_horizon_reachability_amendment"])
            for row in evidence),
        "allocation_robust_distinct_scenes": sum(
            bool(row["passes_every_allowed_allocation"]) for row in evidence),
        "retained_first_eligible_scene_row_count": len(evidence),
        "eligible_state_count_semantics": (
            "one retained first-eligible snapshot row per eligible scene; not "
            "an exhaustive count of snapshots or designated goals"
        ),
        "eligible_designated_goal_count_at_retained_first_eligible_snapshots":
            sum(int(row[
                "eligible_designated_goal_count_at_first_eligible_snapshot"])
                for row in evidence),
        "graph_hops_zero_retained_scene_count": sum(
            int(row["graph_hops_diagnostic"]) == 0 for row in evidence),
        "graph_hops_positive_retained_scene_count": sum(
            int(row["graph_hops_diagnostic"]) > 0 for row in evidence),
        "scene_evidence": sorted(
            evidence, key=lambda row: (row["scene_id"],
                                       row["first_eligible_block"])),
    }
    result = {
        "family": REACHABILITY_REDRIVE_FAMILY,
        "allowed_scene_count": len(shards),
        "scanned_scene_count": len(shards),
        "all_allowed_scenes_scanned": len(shards) == 182,
        "verdict": "PASS" if len(evidence) >= 5 and len(shards) == 182 else "FAIL",
        "provenance": "SCOPED_SMALL_FAMILY_REDRIVE_REQUIRED_MISSING_V1_FIELDS",
        "strata": {
            "general": {
                **predecessor_family_row["strata"]["general"],
                "eligible_designated_goal_count": {
                    "available": False,
                    "reason": "V1_CENSUS_DID_NOT_RETAIN_GOAL_LEVEL_TOTALS",
                },
            },
            "safety_enriched": {
                **predecessor_family_row["strata"]["safety_enriched"],
                "eligible_designated_goal_count": {
                    "available": False,
                    "reason": "V1_CENSUS_DID_NOT_RETAIN_GOAL_LEVEL_TOTALS",
                },
            },
            "completion_enriched": completion,
        },
        "rejection_counts": {
            key: int(value) for key, value in sorted(rejections.items())
        },
    }
    return result


def build_reachability_feasibility_receipt(
        *, predecessor_receipt: dict[str, Any],
        small_scene_shards: Sequence[dict[str, Any]], source: dict[str, Any]
        ) -> dict[str, Any]:
    """Pure final amendment reducer; the frozen v1 receipt is never modified."""

    old_by_family = {row["family"]: row for row in predecessor_receipt["families"]}
    small = _small_reachability_family_row(
        predecessor_family_row=old_by_family[REACHABILITY_REDRIVE_FAMILY],
        shards=small_scene_shards)
    families = [
        (small if family == REACHABILITY_REDRIVE_FAMILY
         else _reused_predecessor_family_row(old_by_family[family]))
        for family in STATE_SELECTOR.REQUIRED_FAMILIES
    ]
    passed = all(row["verdict"] == "PASS" for row in families)
    small_lineage = [{
        "family": REACHABILITY_REDRIVE_FAMILY,
        "scene_id": shard["task"]["scene_id"],
        "scene_task_digest": shard["task"]["scene_task_digest"],
        "predecessor_scene_shard_digest":
            shard["frozen_predecessor_scene_shard_digest"],
        "reachability_scene_shard_digest":
            shard["state_selector_reachability_scene_shard_digest"],
    } for shard in small_scene_shards]
    payload = {
        "schema": REACHABILITY_FEASIBILITY_SCHEMA,
        "status": (REACHABILITY_FEASIBILITY_PASS_STATUS if passed else
                   "FAIL_OUTCOME_FREE_REACHABILITY_SELECTOR_FEASIBILITY"),
        "complete": True,
        "binding_receipt": True,
        "source_repository_commit": source["source_repository_commit"],
        "clean_source_binding_digest": canonical_digest(source),
        "bound_implementations_digest": source["bound_implementations_digest"],
        "successor_selection_digest": selection_digest(),
        "state_selector_amendment_digest":
            STATE_SELECTOR.state_selector_amendment_digest(),
        "candidate_allocation_amendment_digest":
            ALLOC.allocation_amendment_digest(),
        "frozen_predecessor": {
            "failure_report_digest": FROZEN_FEASIBILITY_FAILURE_REPORT_DIGEST,
            "failure_report_raw_sha256":
                FROZEN_FEASIBILITY_FAILURE_REPORT_RAW_SHA256,
            "feasibility_receipt_digest": FROZEN_FEASIBILITY_RECEIPT_DIGEST,
            "feasibility_receipt_raw_sha256":
                FROZEN_FEASIBILITY_RECEIPT_RAW_SHA256,
            "task_census_digest": FROZEN_FEASIBILITY_TASK_CENSUS_DIGEST,
            "scene_shard_count": FROZEN_FEASIBILITY_SCENE_SHARD_COUNT,
            "preserved_unchanged": True,
        },
        "reuse_policy": {
            "general_and_safety_criteria_unchanged": True,
            "seven_family_cached_completion_rows_reclassified": True,
            "unrelated_family_redrives": 0,
            "small_enclosed_maze_redrive_scene_count": len(small_scene_shards),
            "small_redrive_reason": (
                "v1 shards omitted previous_applied_command and the frozen "
                "candidate-subset identity needed for L_max"
            ),
            "actual_allocated_mask_check_required_before_manifest": True,
            "actual_allocated_mask_check_status":
                "MANDATORY_DEFERRED_TO_JOINT_SEARCH_AND_PHASE2",
        },
        "family_count": len(families),
        "families": families,
        "small_scene_shard_lineage": small_lineage,
        "small_scene_shard_lineage_digest": canonical_digest(small_lineage),
        "required_distinct_scenes_per_stratum": 5,
        "candidate_allocation_changed": False,
        "selector_completion_radius_m": float(
            STATE_SELECTOR.COMPLETION_RADIUS_M),
        "horizon_s": float(STATE_SELECTOR.HORIZON_S),
        "horizon_ticks": int(STATE_SELECTOR.HORIZON_TICKS),
        "selected_state_identities_created": False,
        "candidate_outcomes_loaded": False,
        "branch_identities_created": False,
        "branches_attempted": 0,
        "frames_rendered": 0,
        "target_latents_encoded": 0,
        "scorer_training_started": False,
    }
    payload["state_selector_feasibility_receipt_digest"] = canonical_digest(payload)
    return payload


def _validate_reachability_feasibility_receipt(
        payload: dict[str, Any], *, predecessor_receipt: dict[str, Any],
        small_scene_shards: Sequence[dict[str, Any]], source: dict[str, Any]
        ) -> None:
    _verify_self_digest(payload, "state_selector_feasibility_receipt_digest",
                        "reachability selector feasibility receipt")
    expected = build_reachability_feasibility_receipt(
        predecessor_receipt=predecessor_receipt,
        small_scene_shards=small_scene_shards, source=source)
    if payload != expected:
        raise RuntimeError("reachability feasibility receipt is not reproducible")


def _run_reachability_scene_subprocess(
        args: argparse.Namespace, task: dict[str, Any]) -> int:
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--pool", "scorer_fit", "--stage",
        "selector-reachability-feasibility",
        "--family", REACHABILITY_REDRIVE_FAMILY,
        "--selector-scene-id", str(task["scene_id"]),
        "--backend", str(args.backend),
    ]
    completed = subprocess.run(
        command, cwd=ROOT, env={**os.environ, "PYTHONUNBUFFERED": "1"},
        check=False)
    return int(completed.returncode)


def stage_selector_reachability_feasibility(args: argparse.Namespace) -> int:
    """Reuse 1,102 old scene scans and redrive exactly 182 small scenes."""

    if args.pool != "scorer_fit" or args.backend != "cpu":
        raise RuntimeError("reachability feasibility is scorer_fit/cpu only")
    source = clean_source_binding()
    if source.get("source_repository_clean") is not True:
        raise RuntimeError("reachability feasibility requires clean frozen source")
    out = OUT_ROOT / "scorer_fit"
    failure, predecessor, census = _load_frozen_selector_feasibility_lineage(out)
    del failure
    old_shards = _frozen_selector_scene_shards(
        out=out, receipt=predecessor, census=census)
    small_tasks = _selector_feasibility_family_tasks(
        census, REACHABILITY_REDRIVE_FAMILY)
    if len(small_tasks) != 182:
        raise RuntimeError("frozen small-family task count changed")
    selector_scene_id = getattr(args, "selector_scene_id", None)
    if selector_scene_id is not None:
        matches = [task for task in small_tasks
                   if task["scene_id"] == selector_scene_id]
        if len(matches) != 1:
            raise RuntimeError("small-family scene worker task is ambiguous")
        task = matches[0]
        predecessor_shard = old_shards[(REACHABILITY_REDRIVE_FAMILY,
                                        str(task["scene_id"]))]
        old_digest = str(predecessor_shard[
            "state_selector_feasibility_scene_shard_digest"])
        path = _reachability_scene_shard_path(out, task)
        existing = _load_reachability_scene_shard(
            path, expected_task=task, predecessor_shard_digest=old_digest,
            source=source)
        if existing is not None:
            print(json.dumps(existing, indent=2, sort_keys=True))
            return 0
        if _outcome_generation_started(out):
            raise RuntimeError("reachability scene evidence cannot regenerate after outcomes")
        if path.exists():
            _preserve_invalid(path, out, "reachability-scene-invalid")
        started = time.time()
        shared = V1._load_shared(args.backend)
        ctx = V1.build_context(
            Path(task["scene_dir"]), seed=int(task["drive_seed"]),
            backend=args.backend, shared=shared)
        try:
            result = _scan_reachability_selector_scene(
                family=REACHABILITY_REDRIVE_FAMILY,
                scene_dir=Path(task["scene_dir"]), ctx=ctx)
            payload = _build_reachability_scene_shard(
                task=task, predecessor_shard_digest=old_digest,
                scene_result=result, source=source,
                runtime_s=time.time() - started)
            _validate_reachability_scene_shard(
                payload, expected_task=task,
                predecessor_shard_digest=old_digest, source=source)
            atomic_json(path, payload)
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        finally:
            _FIELD_CACHE.clear()
            del ctx
            gc.collect()

    if args.family is not None or args.stratum is not None:
        raise RuntimeError("binding reachability feasibility accepts no scope")
    shards: list[dict[str, Any]] = []
    for task in small_tasks:
        old_digest = str(old_shards[(
            REACHABILITY_REDRIVE_FAMILY, str(task["scene_id"]))][
                "state_selector_feasibility_scene_shard_digest"])
        path = _reachability_scene_shard_path(out, task)
        shard = _load_reachability_scene_shard(
            path, expected_task=task, predecessor_shard_digest=old_digest,
            source=source)
        if shard is None:
            if _outcome_generation_started(out):
                raise RuntimeError("missing reachability evidence after outcomes")
            if path.exists():
                _preserve_invalid(path, out, "reachability-scene-invalid")
            print("[selector-reachability] isolated small-family census: "
                  f"{task['scene_id']}", flush=True)
            return_code = _run_reachability_scene_subprocess(args, task)
            shard = _load_reachability_scene_shard(
                path, expected_task=task, predecessor_shard_digest=old_digest,
                source=source)
            if shard is None:
                raise RuntimeError(
                    f"small reachability worker exited {return_code} without "
                    "valid durable evidence")
        shards.append(shard)
    receipt = build_reachability_feasibility_receipt(
        predecessor_receipt=predecessor, small_scene_shards=shards,
        source=source)
    path = out / REACHABILITY_FEASIBILITY_RECEIPT_NAME
    if path.is_file():
        existing = json.loads(path.read_text())
        try:
            _validate_reachability_feasibility_receipt(
                existing, predecessor_receipt=predecessor,
                small_scene_shards=shards, source=source)
        except Exception:
            if _outcome_generation_started(out):
                raise RuntimeError("reachability receipt changed after outcomes")
            _preserve_invalid(path, out, "reachability-feasibility-invalid")
        else:
            print(json.dumps(existing, indent=2, sort_keys=True))
            return 0 if existing["status"] == REACHABILITY_FEASIBILITY_PASS_STATUS else 1
    elif path.exists():
        raise RuntimeError("reachability receipt path is not a regular file")
    atomic_json(path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == REACHABILITY_FEASIBILITY_PASS_STATUS else 1


_PHASE1_MANAGED_GENERATED_ROOTS = (
    Path(".generated/go2_branch_corpus_v1_2"),
    Path(".generated/go2_utility_scorer_v1_2"),
)


def _pin_phase1_managed_roots(*, root: Path) -> dict[Path, Path]:
    """Resolve each permitted generated alias once for one coherent audit."""

    root_path = Path(root)
    absolute_root = root_path if root_path.is_absolute() else Path.cwd() / root_path
    pinned: dict[Path, Path] = {}
    for managed_relative in _PHASE1_MANAGED_GENERATED_ROOTS:
        managed_root = absolute_root / managed_relative
        if managed_root.is_symlink() or managed_root.exists():
            sentinel = managed_root / ".phase1-custody-root-sentinel"
            pinned[managed_relative] = _pin_generated_path(
                sentinel, sentinel, generated_root=managed_root).parent
        else:
            # An entirely absent generated root is valid absence evidence.
            # Prove that no existing ancestor is an alias before freezing its
            # lexical location for this audit.
            _assert_unsealed_path(managed_root)
            pinned[managed_relative] = managed_root
    return pinned


def _pinned_phase1_surface_path(
        *, pinned_roots: Mapping[Path, Path],
        relative: str | Path) -> Path:
    """Derive one registered surface from a previously pinned root map."""

    logical = Path(relative)
    if logical.is_absolute():
        raise RuntimeError("phase-1 surface path must be repository-relative")
    managed_relative = next((candidate for candidate in
                             _PHASE1_MANAGED_GENERATED_ROOTS
                             if logical == candidate
                             or candidate in logical.parents), None)
    if managed_relative is None:
        raise RuntimeError("phase-1 surface escaped registered generated roots")
    if set(pinned_roots) != set(_PHASE1_MANAGED_GENERATED_ROOTS):
        raise RuntimeError("phase-1 pinned-root surface changed")
    tail = logical.relative_to(managed_relative)
    pinned_path = Path(pinned_roots[managed_relative]).joinpath(*tail.parts)
    _assert_unsealed_path(pinned_path)
    return pinned_path


def _guarded_phase1_descendant_artifacts(
        *, pinned_root: Path, relative: str | Path) -> list[str]:
    """Enumerate files below one pinned root without following a symlink."""

    logical_root = Path(relative)
    if not pinned_root.exists():
        return []
    if pinned_root.is_symlink() or not pinned_root.is_dir():
        raise RuntimeError("phase-1 directory root is not a regular directory")
    artifacts: list[str] = []
    stack: list[tuple[Path, Path]] = [(logical_root, pinned_root)]
    while stack:
        logical_directory, pinned_directory = stack.pop()
        for entry in sorted(pinned_directory.iterdir(), key=lambda row: row.name):
            if entry.is_symlink():
                raise RuntimeError("phase-1 descendant is symlinked")
            logical_entry = logical_directory / entry.name
            _assert_unsealed_path(entry)
            if entry.is_file():
                artifacts.append(str(logical_entry))
            elif entry.is_dir():
                stack.append((logical_entry, entry))
            else:
                raise RuntimeError("phase-1 descendant is not a regular node")
    return sorted(artifacts)


def _phase1_outcome_surface_absence_attestation(
        *, root: Path = ROOT) -> dict[str, Any]:
    """Audit exact scorer-output paths once, before the phase-1 redrive.

    The resulting attestation is an issuance-time fact.  Later legitimate
    branch, latent, training, and transfer outputs do not invalidate it, so
    downstream validators verify the frozen receipt rather than reopening this
    live filesystem surface.
    """

    def kind(path: Path) -> str:
        if path.is_symlink():
            return "symlink"
        if not path.exists():
            return "absent"
        if path.is_file():
            return "file"
        if path.is_dir():
            return "directory"
        return "other"

    pinned_roots = _pin_phase1_managed_roots(root=root)

    exact_file_checks: list[dict[str, Any]] = []
    for relative in STATE_SELECTOR.PHASE1_FORBIDDEN_EXACT_FILE_PATHS:
        target = _pinned_phase1_surface_path(
            pinned_roots=pinned_roots, relative=relative)
        target_kind = kind(target)
        absent = target_kind == "absent"
        exact_file_checks.append({
            "path": relative,
            "exists": not absent,
            "kind": target_kind,
            "artifact_absent": absent,
        })

    directory_checks: list[dict[str, Any]] = []
    for relative in STATE_SELECTOR.PHASE1_FORBIDDEN_DIRECTORY_ROOTS:
        target = _pinned_phase1_surface_path(
            pinned_roots=pinned_roots, relative=relative)
        target_kind = kind(target)
        descendants: list[str] = []
        if target_kind == "directory":
            descendants = _guarded_phase1_descendant_artifacts(
                pinned_root=target, relative=relative)
        elif target_kind != "absent":
            descendants = [relative]
        directory_checks.append({
            "path": relative,
            "exists": target_kind != "absent",
            "kind": target_kind,
            "descendant_artifact_count": len(descendants),
            "descendant_artifacts": descendants,
            "artifact_absent": not descendants,
        })

    glob_checks: list[dict[str, Any]] = []
    for pattern in STATE_SELECTOR.PHASE1_FORBIDDEN_GLOB_PATTERNS:
        logical_pattern = Path(pattern)
        logical_parent = logical_pattern.parent
        if any(character in str(logical_parent) for character in "*?["):
            raise RuntimeError("phase-1 glob parent is not an exact path")
        pinned_parent = _pinned_phase1_surface_path(
            pinned_roots=pinned_roots, relative=logical_parent)
        matches: list[str] = []
        if pinned_parent.exists():
            if pinned_parent.is_symlink() or not pinned_parent.is_dir():
                raise RuntimeError("phase-1 glob parent is not a directory")
            for entry in sorted(pinned_parent.iterdir(), key=lambda row: row.name):
                if not fnmatchcase(entry.name, logical_pattern.name):
                    continue
                if entry.is_symlink():
                    raise RuntimeError("phase-1 glob match is symlinked")
                logical_entry = logical_parent / entry.name
                _assert_unsealed_path(entry)
                matches.append(str(logical_entry))
        glob_checks.append({
            "pattern": pattern,
            "match_count": len(matches),
            "matches": matches,
            "artifact_absent": not matches,
        })

    forbidden_count = (
        sum(not row["artifact_absent"] for row in exact_file_checks)
        + sum(row["descendant_artifact_count"] for row in directory_checks)
        + sum(row["match_count"] for row in glob_checks)
    )
    payload = {
        "schema": STATE_SELECTOR.PHASE1_OUTCOME_SURFACE_ATTESTATION_SCHEMA,
        "status": ("PASS_PRE_OUTCOME_SURFACE_ABSENT" if forbidden_count == 0
                   else "FAIL_PRE_OUTCOME_SURFACE_PRESENT"),
        "issuance_gate": (
            "BEFORE_PHASE1_REDRIVE_AND_BEFORE_ANY_SCIENTIFIC_OUTCOME"
        ),
        "live_reopen_after_legitimate_outcomes": False,
        "exact_file_checks": exact_file_checks,
        "directory_root_checks": directory_checks,
        "glob_checks": glob_checks,
        "forbidden_artifact_count": forbidden_count,
        "all_forbidden_artifacts_absent": forbidden_count == 0,
        "candidate_outcomes_loaded": False,
        "branches_attempted": 0,
        "frames_rendered": 0,
        "target_latents_encoded": 0,
        "scorer_training_started": False,
        "predictor_checkpoints_opened": 0,
    }
    payload["attestation_digest"] = canonical_digest(payload)
    return payload


def _phase1_present_outcome_paths(attestation: dict[str, Any]) -> list[str]:
    """Return concise exact paths for a pre-redrive refusal message."""

    paths = [row["path"] for row in attestation["exact_file_checks"]
             if not row["artifact_absent"]]
    paths.extend(
        artifact for row in attestation["directory_root_checks"]
        for artifact in row["descendant_artifacts"]
    )
    paths.extend(
        match for row in attestation["glob_checks"] for match in row["matches"]
    )
    return sorted(set(paths))


def _replay_small_fixed_prefix_pairs(
        pairs: Sequence[Mapping[str, Any]], *,
        successor_bindings: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]:
    """Run the ordinary request/capture validators and exact 5G/5S reducer.

    Historical pairs carry their byte-bound historical nested bindings.  A
    successor replay instead supplies the one current binding object that must
    occur in every projected request.  Neither mode consults an active
    request/capture directory; the complete pair sequence is provided by the
    custody module after it has reopened the exact archived or successor bytes.
    """

    pool, exclusion = scene_pool("scorer_fit")
    family = REACHABILITY_REDRIVE_FAMILY
    args = argparse.Namespace(pool="scorer_fit", family=family, backend="cpu")
    required = {
        "general": 5,
        "safety_enriched": 5,
        "completion_enriched": 0,
    }
    found = {key: 0 for key in required}
    selected: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    rows = [dict(pair) for pair in pairs]
    if len(rows) != PERFORMANCE_INTERRUPTION.SMALL_PREFIX_REQUEST_COUNT:
        raise RuntimeError("small fixed prefix pair count changed")
    for ordinal, pair in enumerate(rows):
        request = dict(pair.get("request", {}))
        capture = dict(pair.get("capture", {}))
        if (pair.get("scene_ordinal") != ordinal
                or request.get("scene_ordinal") != ordinal
                or pair.get("scene_id") != request.get("scene", {}).get("scene_id")):
            raise RuntimeError("small fixed prefix lexical scene order changed")
        embedded = request.get("state_shard_bindings")
        if not isinstance(embedded, Mapping):
            raise RuntimeError("small fixed prefix bindings are malformed")
        expected_bindings = (
            embedded if successor_bindings is None else
            PERFORMANCE_INTERRUPTION.project_successor_state_shard_bindings(
                embedded, successor_bindings)
        )
        _validate_state_resolution_scene_request(
            request, args=args, out=OUT_ROOT / "scorer_fit", pool=pool,
            exclusion=exclusion,
            expected_state_shard_bindings=expected_bindings)
        _validate_state_resolution_scene_capture(
            capture, expected_request=request)
        requested = [name for name in STRATA
                     if found[name] < required[name]]
        if (request.get("required_counts") != required
                or request.get("found_before_scene") != found
                or request.get("requested_strata_in_priority_order")
                != requested
                or capture.get("worker_failure") is not None):
            raise RuntimeError("small fixed prefix reducer input changed")
        chosen = capture.get("chosen_state")
        chosen_stratum = None
        chosen_digest = None
        if chosen is not None:
            chosen = dict(chosen)
            chosen_stratum = str(chosen["stratum"])
            if chosen_stratum not in requested:
                raise RuntimeError("small fixed prefix selected a filled stratum")
            found[chosen_stratum] += 1
            selected.append(chosen)
            chosen_digest = str(chosen["state_identity_digest"])
        quota_full = found == required
        if quota_full != (ordinal == len(rows) - 1):
            raise RuntimeError("small fixed prefix does not stop at first quota")
        trace.append({
            "scene_ordinal": ordinal,
            "scene_id": str(pair["scene_id"]),
            "found_before_scene": request["found_before_scene"],
            "requested_strata_in_priority_order": requested,
            "chosen_stratum": chosen_stratum,
            "chosen_state_identity_digest": chosen_digest,
        })
    if (found != required or len(selected) != 10
            or len({state["scene_id"] for state in selected}) != 10):
        raise RuntimeError("small fixed prefix no longer contains exact 5G/5S")
    return {
        "states": selected,
        "resolver_cursor_scene_id": str(rows[-1]["scene_id"]),
        "reducer_trace_digest": canonical_digest(trace),
        "state_identity_digests": sorted(
            str(state["state_identity_digest"]) for state in selected),
    }


def _revalidate_performance_interrupted_small_prefix(
        pairs: Sequence[Mapping[str, Any]]) -> bool:
    _replay_small_fixed_prefix_pairs(pairs)
    return True


def _revalidate_reissued_small_prefix(
        archived_pairs: Sequence[Mapping[str, Any]],
        successor_pairs: Sequence[Mapping[str, Any]],
        successor_bindings: Mapping[str, Any]) -> bool:
    archived = _replay_small_fixed_prefix_pairs(archived_pairs)
    successor = _replay_small_fixed_prefix_pairs(
        successor_pairs, successor_bindings=successor_bindings)
    if (archived["state_identity_digests"]
            != successor["state_identity_digests"]
            or archived["resolver_cursor_scene_id"]
            != successor["resolver_cursor_scene_id"]):
        raise RuntimeError("small prefix successor changed scientific identities")
    return True


def stage_small_search_performance_interruption() -> int:
    """Archive the exact 24-hour pre-outcome attempt under clean successor source.

    This stage is an authority/custody transition only.  It neither satisfies
    the selector gate nor resolves a state.  Contract issuance and fixed-shard
    reissue occur in later explicit steps after this receipt is reviewed.
    """

    source = clean_source_binding()
    receipt = (
        PERFORMANCE_INTERRUPTION
        .issue_and_archive_performance_interruption_receipt(
            source_repository_commit=str(source["source_repository_commit"]),
            clean_source_binding_digest=canonical_digest(source),
            bound_implementations_digest=str(
                source["bound_implementations_digest"]),
            outcome_surface_absent=lambda:
                _phase1_outcome_surface_absence_attestation(root=ROOT),
            revalidate_small_prefix=
                _revalidate_performance_interrupted_small_prefix,
            root=ROOT,
        )
    )
    print(json.dumps({
        "status": receipt["status"],
        "receipt": PERFORMANCE_INTERRUPTION.receipt_binding(
            receipt, root=ROOT),
        "scientific_gate_input": receipt["scientific_gate_input"],
        "may_satisfy_selector_gate": receipt["may_satisfy_selector_gate"],
    }, indent=2, sort_keys=True))
    return 0


def _phase1_expected_states() -> list[tuple[dict[str, Any], dict[str, Any]]]:
    predecessor = STATE_SELECTOR.load_preserved_state_shards(ROOT)
    return [
        (dict(expected), dict(state))
        for expected in STATE_SELECTOR.PRESERVED_STATE_SHARDS
        for state in predecessor[str(expected["family"])]["states"]
    ]


def _phase1_state_check_shard_path(
        state_identity_digest: str, *, root: Path = ROOT) -> Path:
    if not _is_sha256(state_identity_digest):
        raise RuntimeError("phase-1 state identity digest is invalid")
    return (
        root / STATE_SELECTOR.PHASE1_STATE_CHECK_SHARD_ROOT
        / f"{state_identity_digest}.json"
    )


def _phase1_check_template(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "state_id": str(entry["state_id"]),
        "state_identity_digest": str(entry["state_identity_digest"]),
        "exclusion_checks_pass": False,
        "exact_redrive_pass": False,
        "amended_classification_pass": False,
        "goal_binding_unchanged": False,
        "oracle_completion_target_unchanged": False,
        "snapshot_production_designated_goal_claim_unchanged": False,
        "production_task_completion_reset_unchanged": False,
        "completion_state_task_status_all_false": False,
        "failure_reason": None,
    }


def _build_phase1_state_check_shard(
        *, entry: dict[str, Any], expected_shard: dict[str, Any],
        check: dict[str, Any], source: dict[str, Any],
        successor_digest: str, feasibility_digest: str,
        outcome_surface_attestation_digest: str) -> dict[str, Any]:
    payload = {
        "schema": STATE_SELECTOR.PHASE1_STATE_CHECK_SHARD_SCHEMA,
        "status": STATE_SELECTOR.PHASE1_STATE_CHECK_SHARD_STATUS,
        "complete": True,
        "binding_receipt": False,
        "source_repository_commit": source["source_repository_commit"],
        "clean_source_binding_digest": canonical_digest(source),
        "bound_implementations_digest": source["bound_implementations_digest"],
        "successor_selection_digest": successor_digest,
        "state_selector_amendment_digest":
            STATE_SELECTOR.state_selector_amendment_digest(),
        "state_selector_feasibility_receipt_digest": feasibility_digest,
        "outcome_surface_absence_attestation_digest":
            outcome_surface_attestation_digest,
        "predecessor_state_shard": dict(expected_shard),
        "family": str(entry["family"]),
        "stratum": str(entry["stratum"]),
        "scene_id": str(entry["scene_id"]),
        "state_id": str(entry["state_id"]),
        "state_identity_digest": str(entry["state_identity_digest"]),
        "source_state_digest": canonical_digest(entry),
        "candidate_outcomes_loaded": False,
        "branch_identities_created": False,
        "branches_attempted": 0,
        "frames_rendered": 0,
        "target_latents_encoded": 0,
        "scorer_training_started": False,
        "predictor_checkpoints_opened": 0,
        "state_check": check,
    }
    payload["state_check_shard_digest"] = canonical_digest(payload)
    return payload


def _load_valid_phase1_state_check_shard(
        *, path: Path, entry: dict[str, Any], expected_shard: dict[str, Any],
        source: dict[str, Any], successor_digest: str,
        feasibility_digest: str, outcome_surface_attestation_digest: str,
        root: Path = ROOT) -> dict[str, Any] | None:
    if not path.is_file() or path.is_symlink():
        return None
    try:
        payload = json.loads(path.read_text())
        STATE_SELECTOR.validate_phase1_state_check_shard(
            payload,
            expected_state=entry,
            expected_predecessor_shard=expected_shard,
            expected_source_commit=str(source["source_repository_commit"]),
            expected_successor_selection_digest=successor_digest,
            expected_feasibility_receipt_digest=feasibility_digest,
            expected_outcome_surface_attestation_digest=
                outcome_surface_attestation_digest,
        )
        if (
            payload.get("clean_source_binding_digest")
            != canonical_digest(source)
            or payload.get("bound_implementations_digest")
            != source["bound_implementations_digest"]
            or path.resolve() != _phase1_state_check_shard_path(
                str(entry["state_identity_digest"]), root=root).resolve()
        ):
            return None
    except (OSError, json.JSONDecodeError,
            STATE_SELECTOR.StateSelectorAmendmentError, RuntimeError):
        return None
    return payload


def _execute_phase1_state_check_worker(
        *, entry: dict[str, Any], expected_shard: dict[str, Any],
        source: dict[str, Any], successor_digest: str,
        feasibility_digest: str, outcome_surface_attestation_digest: str,
        backend: str) -> dict[str, Any]:
    """Run and durably write exactly one check before native teardown."""

    path = _phase1_state_check_shard_path(str(entry["state_identity_digest"]))
    existing = _load_valid_phase1_state_check_shard(
        path=path, entry=entry, expected_shard=expected_shard, source=source,
        successor_digest=successor_digest,
        feasibility_digest=feasibility_digest,
        outcome_surface_attestation_digest=outcome_surface_attestation_digest)
    if existing is not None:
        return existing
    if path.exists():
        _preserve_invalid(
            path, OUT_ROOT / "scorer_fit", "phase1-state-check-invalid")

    check = _phase1_check_template(entry)
    ctx = None
    try:
        try:
            pool, _exclusion = scene_pool("scorer_fit")
            allowed = {
                family: {scene.name: scene for scene in scenes}
                for family, scenes in pool.items()
            }
            family = str(entry["family"])
            scene_dir = allowed.get(family, {}).get(str(entry["scene_id"]))
            if scene_dir is None:
                raise RuntimeError(
                    "scene is absent from strict successor allow-list")
            if (
                scene_dir.resolve() != Path(str(entry["scene_dir"])).resolve()
                or scene_dir.parent.parent.name != str(entry["split"])
                or int(V1._drive_seed(scene_dir.name))
                != int(entry["drive_seed"])
            ):
                raise RuntimeError("scene path, split, or drive seed changed")
            INVALID_IDS.assert_disjoint(
                [entry], label=f"preserved revalidation {entry['state_id']}")
            if (
                file_sha256(scene_dir / "manifest.json")
                != entry["scene_manifest_sha256"]
                or (scene_dir / "manifest.json").stat().st_size
                != int(entry["scene_manifest_byte_count"])
            ):
                raise RuntimeError("scene manifest changed")
            check["exclusion_checks_pass"] = True
            shared = V1._load_shared(backend)
            ctx = V1.build_context(
                scene_dir, seed=int(entry["drive_seed"]), backend=backend,
                shared=shared)
            topology = V12.link_topology(ctx)
            ctx.begin_episode()
            for _block_index in range(int(entry["warmup_blocks"])):
                ctx.drive_one_block()
            verdict = classify_state(
                ctx, topology, requested_stratum=str(entry["stratum"]))
            if isinstance(verdict, str):
                raise RuntimeError(f"amended classification failed: {verdict}")
            record, _field, eligible = verdict
            check["amended_classification_pass"] = (
                str(entry["stratum"]) in eligible)
            check["goal_binding_unchanged"] = record["goal"] == entry["goal"]
            mismatch = _redrive_mismatch(entry, record, ctx)
            check["exact_redrive_pass"] = mismatch is None
            semantic_status = _snapshot_task_status(
                ctx, int(entry["goal"]["landmark_cell"]))
            record_status_matches = (
                str(entry["stratum"]) != "completion_enriched"
                or record.get("snapshot_task_status") == semantic_status)
            check["oracle_completion_target_unchanged"] = \
                _oracle_completion_target_unchanged()
            check[
                "snapshot_production_designated_goal_claim_unchanged"
            ] = bool(
                record_status_matches
                and _snapshot_claim_semantics_unchanged(semantic_status))
            check["production_task_completion_reset_unchanged"] = bool(
                record_status_matches
                and _production_task_reset_semantics_unchanged(semantic_status))
            if str(entry["stratum"]) == "completion_enriched":
                check["completion_rotation_eligibility"] = record[
                    "completion_rotation_eligibility_vector"]
                check["completion_state_task_status_all_false"] = all(
                    semantic_status.get(key) is False for key in (
                        "task_completed", "goal_claimed", "terminated",
                        "truncated"))
            else:
                check["completion_state_task_status_all_false"] = True
            if not all(check[key] is True for key in (
                "exclusion_checks_pass", "exact_redrive_pass",
                "amended_classification_pass", "goal_binding_unchanged",
                "oracle_completion_target_unchanged",
                "snapshot_production_designated_goal_claim_unchanged",
                "production_task_completion_reset_unchanged",
                "completion_state_task_status_all_false",
            )):
                raise RuntimeError(
                    mismatch or "one or more revalidation checks failed")
        except Exception as exc:
            check["failure_reason"] = f"{type(exc).__name__}:{str(exc)[:200]}"

        payload = _build_phase1_state_check_shard(
            entry=entry, expected_shard=expected_shard, check=check,
            source=source, successor_digest=successor_digest,
            feasibility_digest=feasibility_digest,
            outcome_surface_attestation_digest=
                outcome_surface_attestation_digest)
        STATE_SELECTOR.validate_phase1_state_check_shard(
            payload, expected_state=entry,
            expected_predecessor_shard=expected_shard,
            expected_source_commit=str(source["source_repository_commit"]),
            expected_successor_selection_digest=successor_digest,
            expected_feasibility_receipt_digest=feasibility_digest,
            expected_outcome_surface_attestation_digest=
                outcome_surface_attestation_digest)
        # This atomic replacement deliberately precedes context deletion and
        # garbage collection.  A later native teardown signal cannot erase a
        # complete scientific check.
        atomic_json(path, payload)
        return payload
    finally:
        _FIELD_CACHE.clear()
        if ctx is not None:
            del ctx
        gc.collect()


def _run_phase1_state_subprocess(
        args: argparse.Namespace, *, state_identity_digest: str,
        outcome_surface_attestation_digest: str) -> int:
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--pool", "scorer_fit", "--stage", "revalidate-preserved",
        "--backend", str(args.backend),
        "--preserved-state-identity-digest", state_identity_digest,
        "--phase1-outcome-surface-attestation-digest",
        outcome_surface_attestation_digest,
    ]
    completed = subprocess.run(
        command, cwd=ROOT, env={**os.environ, "PYTHONUNBUFFERED": "1"},
        check=False)
    return int(completed.returncode)


def _phase1_state_check_provenance(
        shards: Sequence[dict[str, Any]],
        expected_states: Sequence[tuple[dict[str, Any], dict[str, Any]]],
        *, root: Path = ROOT) -> list[dict[str, Any]]:
    if len(shards) != len(expected_states) or len(expected_states) != 45:
        raise RuntimeError("phase-1 state-check shard count changed")
    rows: list[dict[str, Any]] = []
    for payload, (_expected_shard, state) in zip(
            shards, expected_states, strict=True):
        path = _phase1_state_check_shard_path(
            str(state["state_identity_digest"]), root=root)
        rows.append({
            "family": str(state["family"]),
            "state_id": str(state["state_id"]),
            "state_identity_digest": str(state["state_identity_digest"]),
            "path": str(path.relative_to(root)),
            "raw_sha256": file_sha256(path),
            "byte_count": path.stat().st_size,
            "state_check_shard_digest": payload["state_check_shard_digest"],
        })
    return rows


def _build_phase1_aggregate_receipt(
        *, shard_payloads: Sequence[dict[str, Any]],
        expected_states: Sequence[tuple[dict[str, Any], dict[str, Any]]],
        source: dict[str, Any], successor_digest: str,
        feasibility_digest: str,
        outcome_surface_absence: dict[str, Any],
        root: Path = ROOT) -> dict[str, Any]:
    """Deterministically reduce the exact 45 durable state-check shards."""

    STATE_SELECTOR.validate_phase1_outcome_surface_absence_attestation(
        outcome_surface_absence)
    by_identity = {
        str(payload["state_identity_digest"]): payload
        for payload in shard_payloads
    }
    if (len(shard_payloads) != 45 or len(expected_states) != 45
            or len(by_identity) != 45):
        raise RuntimeError("phase-1 reducer did not recover 45 unique checks")
    predecessor_shards = STATE_SELECTOR.load_preserved_state_shards(root)
    shard_rows: list[dict[str, Any]] = []
    all_state_digests: list[str] = []
    global_failures: list[dict[str, Any]] = []
    for expected in STATE_SELECTOR.PRESERVED_STATE_SHARDS:
        family = str(expected["family"])
        shard = predecessor_shards[family]
        state_checks = [
            dict(by_identity[str(entry["state_identity_digest"])]["state_check"])
            for entry in shard["states"]
        ]
        state_digests = sorted(
            str(state["state_identity_digest"]) for state in shard["states"])
        all_state_digests.extend(state_digests)
        failed = [row for row in state_checks
                  if row["failure_reason"] is not None]
        global_failures.extend(dict(row) for row in failed)
        shard_rows.append({
            **dict(expected),
            "revalidated_state_count": len(state_checks),
            "unchanged_state_identity_count": len(state_checks),
            "failed_state_count": len(failed),
            "exact_redrive_pass": not failed and all(
                row["exact_redrive_pass"] for row in state_checks),
            "amended_classification_pass": not failed and all(
                row["amended_classification_pass"] for row in state_checks),
            "completion_state_task_status_all_false": not failed and all(
                row["completion_state_task_status_all_false"]
                for row in state_checks),
            "exclusion_checks_pass": not failed and all(
                row["exclusion_checks_pass"] for row in state_checks),
            "goal_binding_unchanged": not failed and all(
                row["goal_binding_unchanged"] for row in state_checks),
            "oracle_completion_target_unchanged": not failed and all(
                row["oracle_completion_target_unchanged"]
                for row in state_checks),
            "snapshot_production_designated_goal_claim_unchanged":
                not failed and all(
                    row[
                        "snapshot_production_designated_goal_claim_unchanged"
                    ] for row in state_checks),
            "production_task_completion_reset_unchanged": not failed and all(
                row["production_task_completion_reset_unchanged"]
                for row in state_checks),
            "candidate_outcomes_loaded": False,
            "state_identity_digests": state_digests,
            "state_identity_set_digest": canonical_digest(state_digests),
            "state_checks": state_checks,
        })

    state_check_provenance = _phase1_state_check_provenance(
        shard_payloads, expected_states, root=root)
    passed = not global_failures
    payload = {
        "schema": STATE_SELECTOR.PRESERVED_STATE_PRECONTRACT_REVALIDATION_SCHEMA,
        "status": ("PASS_PRECONTRACT_IDENTITY_REVALIDATION" if passed
                   else "FAIL_PRECONTRACT_IDENTITY_REVALIDATION"),
        "complete": True,
        "source_repository_commit": source["source_repository_commit"],
        "clean_source_binding_digest": canonical_digest(source),
        "bound_implementations_digest": source["bound_implementations_digest"],
        "successor_selection_digest": successor_digest,
        "state_selector_amendment_digest":
            STATE_SELECTOR.state_selector_amendment_digest(),
        "state_selector_feasibility_receipt_digest": feasibility_digest,
        "outcome_surface_absence_verified_at_phase1_issuance": True,
        "outcome_surface_absence_attestation_digest":
            outcome_surface_absence["attestation_digest"],
        "outcome_surface_absence_attestation": outcome_surface_absence,
        "state_check_subprocess_transport": {
            "schema": "go2_scorer_fit_phase1_subprocess_transport_v1",
            "state_check_shard_schema":
                STATE_SELECTOR.PHASE1_STATE_CHECK_SHARD_SCHEMA,
            "state_check_shard_root":
                STATE_SELECTOR.PHASE1_STATE_CHECK_SHARD_ROOT,
            "state_count": len(state_check_provenance),
            "one_state_per_subprocess": True,
            "atomic_shard_write_before_native_cleanup": True,
            "return_code_ignored_only_after_valid_shard": True,
            "resume_scope": "MISSING_OR_INVALID_STATE_CHECK_SHARDS_ONLY",
            "candidate_outcomes_loaded": False,
            "state_check_shard_provenance_digest":
                canonical_digest(state_check_provenance),
        },
        "state_check_shard_provenance": state_check_provenance,
        "predecessor_selection_digest":
            STATE_SELECTOR.PREDECESSOR_SELECTION_DIGEST,
        "predecessor_scorer_contract_digest":
            STATE_SELECTOR.PREDECESSOR_SCORER_CONTRACT_DIGEST,
        "candidate_outcomes_loaded": False,
        "candidate_allocation_loaded": False,
        "branch_identities_created": False,
        "branches_attempted": 0,
        "frames_rendered": 0,
        "target_latents_encoded": 0,
        "scorer_training_started": False,
        "preserved_state_count": len(all_state_digests),
        "state_identity_set_digest": canonical_digest(sorted(all_state_digests)),
        "shards": shard_rows,
        "failure_count": len(global_failures),
        "failures": global_failures,
    }
    payload["preserved_state_precontract_revalidation_receipt_digest"] = \
        canonical_digest(payload)
    return payload


def _reconstruct_terminal_phase1_failure(
        *, receipt: dict[str, Any], expected_states: Sequence[
            tuple[dict[str, Any], dict[str, Any]]], source: dict[str, Any],
        successor_digest: str, feasibility_digest: str,
        root: Path = ROOT) -> dict[str, Any]:
    """Reopen all atomic checks before accepting a failed phase-1 terminal."""

    _verify_self_digest(
        receipt, "preserved_state_precontract_revalidation_receipt_digest",
        "failed precontract revalidation receipt")
    absence = receipt.get("outcome_surface_absence_attestation")
    if not isinstance(absence, dict):
        raise RuntimeError(
            "failed precontract receipt lacks its absence attestation")
    STATE_SELECTOR.validate_phase1_outcome_surface_absence_attestation(absence)
    if (
        receipt.get("status") != "FAIL_PRECONTRACT_IDENTITY_REVALIDATION"
        or receipt.get("complete") is not True
        or receipt.get("source_repository_commit")
        != source["source_repository_commit"]
        or receipt.get("clean_source_binding_digest") != canonical_digest(source)
        or receipt.get("bound_implementations_digest")
        != source["bound_implementations_digest"]
        or receipt.get("successor_selection_digest") != successor_digest
        or receipt.get("state_selector_amendment_digest")
        != STATE_SELECTOR.state_selector_amendment_digest()
        or receipt.get("state_selector_feasibility_receipt_digest")
        != feasibility_digest
        or receipt.get("outcome_surface_absence_verified_at_phase1_issuance")
        is not True
        or receipt.get("outcome_surface_absence_attestation_digest")
        != absence["attestation_digest"]
    ):
        raise RuntimeError(
            "failed precontract receipt lineage does not match live inputs")
    shards: list[dict[str, Any]] = []
    for expected_shard, entry in expected_states:
        shard = _load_valid_phase1_state_check_shard(
            path=_phase1_state_check_shard_path(
                str(entry["state_identity_digest"]), root=root),
            entry=entry,
            expected_shard=expected_shard,
            source=source,
            successor_digest=successor_digest,
            feasibility_digest=feasibility_digest,
            outcome_surface_attestation_digest=absence["attestation_digest"],
            root=root,
        )
        if shard is None:
            raise RuntimeError(
                "failed precontract receipt lacks an exact durable state check")
        shards.append(shard)
    reconstructed = _build_phase1_aggregate_receipt(
        shard_payloads=shards,
        expected_states=expected_states,
        source=source,
        successor_digest=successor_digest,
        feasibility_digest=feasibility_digest,
        outcome_surface_absence=absence,
        root=root,
    )
    if reconstructed.get("status") != "FAIL_PRECONTRACT_IDENTITY_REVALIDATION":
        raise RuntimeError(
            "failed precontract terminal does not reproduce from atomic checks")
    if reconstructed != receipt:
        raise RuntimeError(
            "failed precontract terminal differs from its atomic checks")
    return reconstructed


def stage_preserved_state_precontract_revalidation(
        args: argparse.Namespace) -> int:
    """Carry the frozen 45-check FAIL into the active 37/8 disposition.

    The exact failed aggregate and all 45 atomic redrive shards were produced
    under their frozen clean source and are immutable evidence.  This successor
    stage performs no simulator work: it reopens that terminal byte-for-byte,
    attests that scientific outcome surfaces are still absent, and issues the
    distinct active authority for 37 retained identities and eight vacant
    replacement slots.
    """

    if args.pool != "scorer_fit" or args.family is not None or args.stratum is not None:
        raise RuntimeError(
            "preserved-state precontract revalidation is one all-family scorer-fit gate"
        )
    if args.backend != "cpu":
        raise RuntimeError("preserved-state revalidation requires the CPU backend")
    if (getattr(args, "preserved_state_identity_digest", None) is not None
            or getattr(args, "phase1_outcome_surface_attestation_digest", None)
            is not None):
        raise RuntimeError(
            "the frozen 45-state terminal may not be redriven under successor source"
        )
    STATE_SELECTOR.validate_authority_artifacts()
    source = clean_source_binding()
    if source.get("source_repository_clean") is not True:
        raise RuntimeError("mixed precontract disposition requires clean source")
    successor_digest = selection_digest()
    # The prior clean implementation was interrupted before one replacement
    # identity existed.  Preserve its exact mixed authority, contract, launch,
    # and all 31 outcome-free request/capture records before issuing any
    # successor-source authority.  The old records remain byte-bound but are
    # explicitly inactive and cannot be resumed.
    interruption = INTERRUPTION.issue_and_archive_interruption_receipt(
        source_repository_commit=str(source["source_repository_commit"]),
        clean_source_binding_digest=canonical_digest(source),
        bound_implementations_digest=str(
            source["bound_implementations_digest"]),
        root=ROOT,
    )
    INTERRUPTION.validate_interruption_receipt(
        interruption,
        expected_source_repository_commit=str(source["source_repository_commit"]),
        expected_clean_source_binding_digest=canonical_digest(source),
        expected_bound_implementations_digest=str(
            source["bound_implementations_digest"]),
        root=ROOT,
        require_archived=True,
    )
    out = OUT_ROOT / "scorer_fit"
    raw_frozen_failure_path = (
        ROOT / STATE_SELECTOR.PRESERVED_STATE_PRECONTRACT_REVALIDATION_RECEIPT_PATH
    )
    raw_path = (
        ROOT
        / STATE_SELECTOR.PRESERVED_STATE_MIXED_PRECONTRACT_DISPOSITION_RECEIPT_PATH
    )
    if raw_path.parent != out:
        raise RuntimeError("mixed disposition receipt escaped scorer-fit pool")
    frozen_failure_path = _pin_generated_path(
        raw_frozen_failure_path, raw_frozen_failure_path)
    path = _pin_generated_path(raw_path, raw_path)
    # These validators reopen the historical feasibility PASS, failed aggregate,
    # all 45 atomic check shards, and their exact predecessor identity shards.
    STATE_SELECTOR.validate_frozen_reachability_feasibility_pass(root=ROOT)
    STATE_SELECTOR.validate_frozen_preserved_precontract_failure(root=ROOT)
    frozen_binding = STATE_SELECTOR.FROZEN_PRESERVED_PRECONTRACT_FAILURE
    if (not frozen_failure_path.is_file()
            or file_sha256(frozen_failure_path) != frozen_binding["raw_sha256"]
            or frozen_failure_path.stat().st_size != frozen_binding["byte_count"]):
        raise RuntimeError("frozen failed precontract receipt bytes changed")
    if path.is_file():
        existing = json.loads(path.read_text())
        STATE_SELECTOR.validate_preserved_state_mixed_precontract_disposition_receipt(
            existing,
            expected_source_commit=str(source["source_repository_commit"]),
            expected_successor_selection_digest=successor_digest,
            expected_clean_source_binding_digest=canonical_digest(source),
            expected_bound_implementations_digest=str(
                source["bound_implementations_digest"]),
            root=ROOT,
        )
        print(json.dumps(existing, indent=2, sort_keys=True))
        return 0

    outcome_surface_absence = _phase1_outcome_surface_absence_attestation()
    if outcome_surface_absence["all_forbidden_artifacts_absent"] is not True:
        present = _phase1_present_outcome_paths(outcome_surface_absence)
        raise RuntimeError(
            "mixed precontract disposition found a pre-existing scientific "
            f"outcome surface: {present[:20]}"
        )
    STATE_SELECTOR.validate_phase1_outcome_surface_absence_attestation(
        outcome_surface_absence)
    payload = \
        STATE_SELECTOR.build_preserved_state_mixed_precontract_disposition_receipt(
            source_repository_commit=str(source["source_repository_commit"]),
            clean_source_binding_digest=canonical_digest(source),
            bound_implementations_digest=str(
                source["bound_implementations_digest"]),
            successor_selection_digest=successor_digest,
            outcome_surface_absence_attestation=outcome_surface_absence,
            root=ROOT,
        )
    atomic_json(path, payload)
    STATE_SELECTOR.validate_preserved_state_mixed_precontract_disposition_receipt(
        payload,
        expected_source_commit=str(source["source_repository_commit"]),
        expected_successor_selection_digest=successor_digest,
        expected_clean_source_binding_digest=canonical_digest(source),
        expected_bound_implementations_digest=str(
            source["bound_implementations_digest"]),
        root=ROOT,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _state_shard_bindings(args: argparse.Namespace, exclusion: dict[str, Any],
                          family_allow_list: list[str]) -> dict[str, Any]:
    scorer = scorer_contract()
    pre_identity = _load_pre_identity_allocation_validation()
    launch = _load_clean_source_launch_receipt()
    return {
        "selection_digest": selection_digest(),
        "scorer_fit_allocation_design_digest":
            scorer["scorer_fit_allocation_design_digest"],
        "candidate_allocator_contract_digest": ALLOC.allocation_contract_digest(),
        "candidate_allocation_amendment_digest":
            ALLOC.allocation_amendment_digest(),
        "pre_identity_allocation_validation_digest":
            pre_identity["pre_identity_validation_digest"],
        "invalid_scorer_identity_exclusion_digest":
            INVALID_IDS.invalid_identity_exclusion_digest(),
        "state_selector_amendment_digest":
            STATE_SELECTOR.state_selector_amendment_digest(),
        "state_selector_feasibility_receipt_digest":
            launch["state_selector_feasibility_receipt_digest"],
        "clean_source_launch_receipt_digest":
            launch["clean_source_launch_receipt_digest"],
        "source_repository_commit": launch["source_repository_commit"],
        "clean_source_binding_digest": launch["clean_source_binding_digest"],
        "bound_implementations_digest": launch["bound_implementations_digest"],
        "scorer_contract_artifact_digest":
            launch["scorer_contract_artifact_digest"],
        "mixed_precontract_disposition_receipt_digest":
            launch["mixed_precontract_disposition_receipt_digest"],
        "candidate_bank_digest": V1.bank_digest(),
        "progress_contract_digest": progress_digest(),
        "safety_contract_digest": safety_digest(),
        "oracle_v1_2_digest": v12_oracle_digest(),
        "scorer_contract_v1_2_digest": scorer_contract_digest(),
        "boundary_digest": V1.BOUNDARY_DIGEST,
        "render_contract_digest": render_contract_digest(),
        "textured_v03_renderer_contract_digest":
            textured_v03_renderer_contract_digest(),
        "preprocess_contract_digest": preprocess_contract_digest(),
        "preprocessing_digest": TARGET_ENCODER["preprocessing_identity_sha256"],
        "target_encoder_digest": target_encoder_digest(),
        "target_encoder_checkpoint_sha256": TARGET_ENCODER["checkpoint_sha256"],
        "genesis_backend": args.backend,
        "exclusion_binding": exclusion,
        "family_allow_list_digest": canonical_digest(family_allow_list),
    }


def _phase1_completion_rotation_vectors() -> dict[str, dict[str, Any]]:
    """Return seven retained completion vectors from the frozen 45-check FAIL."""

    receipt = STATE_SELECTOR.validate_frozen_preserved_precontract_failure(
        root=ROOT)
    _load_active_mixed_disposition()
    vectors: dict[str, dict[str, Any]] = {}
    for shard in receipt.get("shards", []):
        for check in shard.get("state_checks", []):
            vector = check.get("completion_rotation_eligibility")
            if vector is None:
                continue
            digest = str(check["state_identity_digest"])
            if digest in vectors:
                raise RuntimeError("phase-1 repeats a completion identity")
            vectors[digest] = dict(vector)
    retained, _rejected, _slots = _mixed_disposition_sets()
    expected = {
        identity for identity, row in retained.items()
        if row["stratum"] == "completion_enriched"
    }
    if len(vectors) != 7 or set(vectors) != expected:
        raise RuntimeError("frozen phase-1 does not contain seven retained vectors")
    return vectors


def _state_completion_rotation_vector(
        state: dict[str, Any], preserved_vectors: dict[str, dict[str, Any]]
        ) -> dict[str, Any]:
    vector = state.get("completion_rotation_eligibility_vector")
    if vector is None:
        vector = preserved_vectors.get(str(state["state_identity_digest"]))
    if not isinstance(vector, dict):
        raise RuntimeError(
            f"completion state {state['state_id']} lacks a rotation vector")
    return vector


def _allocation_projection(states: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    return [{
        "state_id": str(state["state_id"]),
        "state_identity_digest": str(state["state_identity_digest"]),
        "family": str(state["family"]),
        "stratum": str(state["stratum"]),
        "split_role": str(state["split_role"]),
        "goal_type": str(state["goal_type"]),
    } for state in states]


def _allocation_assignment_set_digest(allocation: dict[str, Any]) -> str:
    """Digest only state-to-mask assignments, independent of source receipt."""

    rows = sorted([{
        "state_id": str(row["state_id"]),
        "state_identity_digest": str(row["state_identity_digest"]),
        "candidate_rotation_index": int(row["rotation_index"]),
        "candidate_indices": [int(value) for value in row["candidate_indices"]],
    } for row in allocation["assignments"]], key=lambda row: (
        row["state_identity_digest"], row["state_id"]))
    return PARALLEL_SEARCH.canonical_digest(rows)


def _all_completion_masks_pass(
        *, states: Sequence[dict[str, Any]], allocation: dict[str, Any],
        preserved_vectors: dict[str, dict[str, Any]]) -> bool:
    """Recompute all 40 exact predicates; no outcome field is accepted."""

    assigned = {str(row["state_identity_digest"]): row
                for row in allocation["assignments"]}
    checked = 0
    for state in states:
        if state["stratum"] != "completion_enriched":
            continue
        assignment = assigned.get(str(state["state_identity_digest"]))
        if assignment is None:
            raise RuntimeError("joint search allocation omitted a completion state")
        vector = _state_completion_rotation_vector(state, preserved_vectors)
        rotations = vector.get("rotations")
        rotation = int(assignment["rotation_index"])
        if (not isinstance(rotations, list) or len(rotations) != 12
                or rotations[rotation].get("candidate_indices")
                != assignment["candidate_indices"]):
            raise RuntimeError("joint search rotation evidence is malformed")
        selected_evidence = rotations[rotation]
        eligible = selected_evidence.get("eligible")
        if not isinstance(eligible, bool):
            raise RuntimeError(
                "joint search rotation eligibility is not an exact boolean")
        if eligible is False:
            # An ordinary, arithmetically valid ineligible combination advances
            # the lexicographic search.  Malformed evidence must never be
            # converted into apparent scientific infeasibility.
            expected = STATE_SELECTOR.completion_enriched_eligibility(
                graph_hops=int(selected_evidence["graph_hops_diagnostic"]),
                reachable=bool(selected_evidence["reachable"]),
                continuous_geodesic_m=float(
                    selected_evidence["continuous_geodesic_m"]),
                bearing_body_rad=float(selected_evidence["bearing_body_rad"]),
                task_status=selected_evidence["task_status"],
                candidate_indices=assignment["candidate_indices"],
                previous_applied_command=selected_evidence[
                    "previous_applied_command"])
            if selected_evidence != expected:
                raise RuntimeError(
                    "joint search ineligible evidence failed reconstruction")
            return False
        # For an eligible row, central validation is structural and scientific;
        # propagate every defect rather than treating it as the next combo.
        try:
            STATE_SELECTOR.validate_allocated_completion_evidence(
                selected_evidence,
                candidate_indices=assignment["candidate_indices"],
                previous_applied_command=selected_evidence[
                    "previous_applied_command"])
        except STATE_SELECTOR.StateSelectorAmendmentError as exc:
            raise RuntimeError(
                "joint search eligible evidence failed reconstruction") from exc
        checked += 1
    if checked != 40:
        raise RuntimeError(f"joint search checked {checked}, expected 40 states")
    return True


def select_small_completion_combination(
        *, fixed_states: Sequence[dict[str, Any]],
        raw_candidates: Sequence[dict[str, Any]],
        preserved_vectors: dict[str, dict[str, Any]],
        resolver_cursor_scene_id: str | None = None,
        ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Refuse the superseded serial/unbounded implementation explicitly."""

    del fixed_states, raw_candidates, preserved_vectors, resolver_cursor_scene_id
    raise RuntimeError(
        "serial small completion selection is superseded by the measured "
        "bounded parallel coordinator")


def _cursor_restricted_completion_rows(
        evidence: Sequence[dict[str, Any]], *, resolver_cursor_scene_id: str,
        excluded_scene_ids: set[str]) -> list[dict[str, Any]]:
    """Continue one lexical scene pass; never revisit consumed scenes."""

    ordered = sorted(evidence, key=lambda row: (
        str(row["scene_id"]), int(row["first_eligible_block"])))
    return [row for row in ordered
            if (str(row["scene_id"]) > str(resolver_cursor_scene_id)
                and str(row["scene_id"]) not in excluded_scene_ids)]


def _small_completion_candidates_from_feasibility(
        *, out: Path, excluded_scene_ids: set[str],
        resolver_cursor_scene_id: str) -> list[dict[str, Any]]:
    raw_path = out / REACHABILITY_FEASIBILITY_RECEIPT_NAME
    path = _pin_generated_path(raw_path, raw_path)
    if not path.is_file():
        raise RuntimeError("small identity resolution requires reachability receipt")
    receipt = json.loads(path.read_text())
    launch = _load_clean_source_launch_receipt()
    frozen = STATE_SELECTOR.validate_frozen_reachability_feasibility_pass(
        root=ROOT)
    if receipt != frozen:
        raise RuntimeError("active reachability feasibility differs from frozen PASS")
    if receipt.get("state_selector_feasibility_receipt_digest") != launch[
            "state_selector_feasibility_receipt_digest"]:
        raise RuntimeError("reachability feasibility differs from launch binding")
    small = next(row for row in receipt["families"]
                 if row["family"] == REACHABILITY_REDRIVE_FAMILY)
    evidence = _cursor_restricted_completion_rows(
        small["strata"]["completion_enriched"]["scene_evidence"],
        resolver_cursor_scene_id=resolver_cursor_scene_id,
        excluded_scene_ids=excluded_scene_ids)
    _, _, census = _load_frozen_selector_feasibility_lineage(out)
    tasks = {task["scene_id"]: task for task in
             _selector_feasibility_family_tasks(census, REACHABILITY_REDRIVE_FAMILY)}
    candidates: list[dict[str, Any]] = []
    for row in evidence:
        task = tasks[str(row["scene_id"])]
        state = {
            "state_id": "DEFERRED_SMALL_COMPLETION_JOINT_SEARCH",
            "family": REACHABILITY_REDRIVE_FAMILY,
            "scene_id": str(row["scene_id"]),
            "scene_dir": str(task["scene_dir"]),
            "scene_manifest_sha256": str(task["scene_manifest_sha256"]),
            "scene_manifest_byte_count": int(task["scene_manifest_byte_count"]),
            "split": str(task["split"]),
            "drive_seed": int(task["drive_seed"]),
            "stratum": "completion_enriched",
            "split_role": "DEFERRED_SMALL_COMPLETION_JOINT_SEARCH",
            "warmup_blocks": int(row["first_eligible_block"]),
            "source_step": int(row["source_step"]),
            "episode_id": int(row["episode_id"]),
            "episode_cluster_id": str(row["episode_cluster_id"]),
            "cell_id": int(row["cell_id"]),
            "boundary": dict(row["boundary"]),
            "goal": {
                "landmark_id": str(row["goal_landmark_id"]),
                "landmark_cell": int(row["goal_landmark_cell"]),
                "material_id": str(row["goal_material_id"]),
                "graph_edges": int(row["graph_hops_diagnostic"]),
                "start_geodesic_m": float(row["continuous_geodesic_m"]),
                "bearing_body_rad": float(row["bearing_body_rad"]),
                "range_m": float(row["range_m"]),
                "landmark_xy_m": list(row["goal_landmark_xy_m"]),
            },
            "goal_type": str(row["goal_material_id"]),
            "body_clearance_m": float(row["body_clearance_m"]),
            "clearance_m": float(row["clearance_m"]),
            "completion_rotation_eligibility_vector": dict(
                row["completion_rotation_eligibility_vector"]),
            "snapshot_task_status": dict(row["snapshot_task_status"]),
            "previous_applied_command": list(row["previous_applied_command"]),
        }
        candidates.append(state)
    return candidates


def _issue_small_completion_search_failure(
        *, out: Path, error: SmallCompletionJointSearchInfeasible,
        resolver_cursor_scene_id: str) -> dict[str, Any]:
    """Persist one non-overwriting terminal pre-outcome search failure."""

    launch = _load_clean_source_launch_receipt()
    payload = {
        "schema": SMALL_COMPLETION_SEARCH_FAILURE_SCHEMA,
        "status": SMALL_COMPLETION_SEARCH_FAILURE_STATUS,
        "complete": True,
        "source_repository_commit": launch["source_repository_commit"],
        "clean_source_launch_receipt_digest":
            launch["clean_source_launch_receipt_digest"],
        "state_selector_amendment_digest":
            STATE_SELECTOR.state_selector_amendment_digest(),
        "state_selector_feasibility_receipt_digest":
            launch["state_selector_feasibility_receipt_digest"],
        "candidate_allocation_amendment_digest":
            ALLOC.allocation_amendment_digest(),
        "resolver_cursor_scene_id": str(resolver_cursor_scene_id),
        "cursor_restricted_candidate_scene_ids":
            list(error.candidate_scene_ids),
        "cursor_restricted_candidate_scene_ids_digest":
            canonical_digest(list(error.candidate_scene_ids)),
        "cursor_restricted_candidate_scene_count":
            len(error.candidate_scene_ids),
        "combination_attempt_count": int(error.attempt_count),
        "allocator_infeasible_combination_count":
            int(error.allocator_infeasible_count),
        "failure_reason": error.reason,
        "complete_identity_manifest_created": False,
        "candidate_outcomes_loaded": False,
        "branch_identities_created": False,
        "branches_attempted": 0,
        "frames_rendered": 0,
        "target_latents_encoded": 0,
        "scorer_training_started": False,
        "predictor_checkpoints_opened": 0,
    }
    payload["small_completion_joint_search_failure_digest"] = \
        canonical_digest(payload)
    _validate_small_completion_search_failure_static(payload, launch=launch)
    raw_path = out / SMALL_COMPLETION_SEARCH_FAILURE_NAME
    path = _pin_generated_path(raw_path, raw_path)
    if path.exists():
        if not path.is_file():
            raise RuntimeError("small completion failure path is not a file")
        existing = json.loads(path.read_text())
        _verify_self_digest(
            existing, "small_completion_joint_search_failure_digest",
            "small completion joint-search failure")
        if existing != payload:
            raise RuntimeError(
                "refusing to overwrite a different small completion failure")
        return existing
    atomic_json(path, payload)
    return payload


def _state_resolution_request_path(
        out: Path, family: str, request_digest: str) -> Path:
    if not _is_sha256(request_digest) or family not in STATE_SELECTOR.REQUIRED_FAMILIES:
        raise RuntimeError("state-resolution request identity is invalid")
    return out / STATE_RESOLUTION_SCENE_REQUEST_ROOT / family / \
        f"{request_digest}.json"


def _state_resolution_capture_path(
        out: Path, family: str, request_digest: str) -> Path:
    if not _is_sha256(request_digest) or family not in STATE_SELECTOR.REQUIRED_FAMILIES:
        raise RuntimeError("state-resolution capture identity is invalid")
    return out / STATE_RESOLUTION_SCENE_CAPTURE_ROOT / family / \
        f"{request_digest}.json"


def _mixed_replacement_request_path(
        out: Path, family: str, request_digest: str) -> Path:
    if not _is_sha256(request_digest) or family not in {
            row["family"] for row in STATE_SELECTOR.PRESERVED_STATE_SHARDS}:
        raise RuntimeError("mixed replacement request identity is invalid")
    return out / MIXED_REPLACEMENT_SCENE_REQUEST_ROOT / family / \
        f"{request_digest}.json"


def _mixed_replacement_capture_path(
        out: Path, family: str, request_digest: str) -> Path:
    if not _is_sha256(request_digest) or family not in {
            row["family"] for row in STATE_SELECTOR.PRESERVED_STATE_SHARDS}:
        raise RuntimeError("mixed replacement capture identity is invalid")
    return out / MIXED_REPLACEMENT_SCENE_CAPTURE_ROOT / family / \
        f"{request_digest}.json"


def _pin_generated_path(
        raw_path: Path, expected_raw_path: Path, *,
        generated_root: Path | None = None) -> Path:
    """Pin one managed generated path and prove its exact logical identity."""

    pinned = _frozen_generated_artifact_path(
        raw_path, generated_root=generated_root)
    expected = _frozen_generated_artifact_path(
        expected_raw_path, generated_root=generated_root)
    if pinned != expected:
        raise RuntimeError("managed generated artifact path identity changed")
    return pinned


def _scene_in_open_interval(
        scene_id: str, *, lower: str | None, upper: str | None) -> bool:
    return bool((lower is None or scene_id > lower)
                and (upper is None or scene_id < upper))


def _mixed_replacement_candidate_scenes(
        *, scenes: Sequence[Path], interval: dict[str, Any],
        retained_scene_ids: set[str]) -> list[tuple[int, Path]]:
    lower = interval["lower_scene_id_exclusive"]
    upper = interval["upper_scene_id_exclusive"]
    return [
        (ordinal, scene_dir)
        for ordinal, scene_dir in enumerate(scenes)
        if (scene_dir.name not in retained_scene_ids
            and _scene_in_open_interval(
                scene_dir.name, lower=lower, upper=upper))
    ]


def _build_mixed_replacement_scene_request(
        *, args: argparse.Namespace, out: Path, scene_dir: Path,
        scene_ordinal: int, interval: dict[str, Any], slot: dict[str, Any],
        accepted_scene_ids_before: Sequence[str], exclusion: dict[str, Any],
        family_allow_list: Sequence[str], persist: bool = True) -> dict[str, Any]:
    family = str(args.family)
    plan = _mixed_family_replacement_plan(family)
    manifest_path = scene_dir / "manifest.json"
    payload = {
        "schema": MIXED_REPLACEMENT_SCENE_REQUEST_SCHEMA,
        "status": STATUS,
        "complete": True,
        "binding_receipt": False,
        "pool": "scorer_fit",
        "family": family,
        "backend": str(args.backend),
        "scene_ordinal": int(scene_ordinal),
        "scene": {
            "scene_id": str(scene_dir.name),
            "scene_dir": str(scene_dir.resolve()),
            "scene_manifest_sha256": file_sha256(manifest_path),
            "scene_manifest_byte_count": manifest_path.stat().st_size,
            "split": str(scene_dir.parent.parent.name),
            "drive_seed": int(V1._drive_seed(scene_dir.name)),
        },
        "replacement_slot": dict(slot),
        "anchor_interval": {
            "lower_scene_id_exclusive": interval[
                "lower_scene_id_exclusive"],
            "upper_scene_id_exclusive": interval[
                "upper_scene_id_exclusive"],
            "vacant_ordinals": list(interval["vacant_ordinals"]),
        },
        "accepted_scene_ids_before": list(accepted_scene_ids_before),
        "retained_scene_ids_digest": canonical_digest(
            plan["retained_scene_ids"]),
        "rejected_identity_digests": list(plan["rejected_identity_digests"]),
        "rejected_identity_set_digest": canonical_digest(
            plan["rejected_identity_digests"]),
        "family_replacement_plan_digest": canonical_digest(plan),
        "warmup_blocks_min": int(WARMUP_BLOCKS_MIN),
        "warmup_blocks_max": int(WARMUP_BLOCKS_MAX),
        "state_shard_bindings": _state_shard_bindings(
            args, exclusion, list(family_allow_list)),
        "selection_semantics": (
            "completion-only; lexical scene order within the retained-anchor "
            "interval; ascending blocks; first V2-eligible identity distinct "
            "from every rejected predecessor digest"
        ),
        "candidate_outcomes_loaded": False,
        "branch_identities_created": False,
        "branches_attempted": 0,
        "frames_rendered": 0,
        "target_latents_encoded": 0,
        "scorer_training_started": False,
        "predictor_checkpoints_opened": 0,
    }
    payload["mixed_replacement_scene_request_digest"] = canonical_digest(payload)
    raw_path = _mixed_replacement_request_path(
        out, family, payload["mixed_replacement_scene_request_digest"])
    path = _pin_generated_path(raw_path, raw_path)
    if not persist:
        return payload
    if path.is_file():
        if json.loads(path.read_text()) != payload:
            raise RuntimeError("mixed replacement request digest collision")
    elif path.exists():
        raise RuntimeError("mixed replacement request path is not a file")
    else:
        atomic_json(path, payload)
    return payload


def _validate_mixed_replacement_scene_request(
        request: dict[str, Any], *, args: argparse.Namespace, out: Path,
        pool: dict[str, list[Path]], exclusion: dict[str, Any]) -> None:
    _verify_self_digest(
        request, "mixed_replacement_scene_request_digest",
        "mixed replacement scene request")
    expected_keys = {
        "schema", "status", "complete", "binding_receipt", "pool", "family",
        "backend", "scene_ordinal", "scene", "replacement_slot",
        "anchor_interval", "accepted_scene_ids_before",
        "retained_scene_ids_digest", "rejected_identity_digests",
        "rejected_identity_set_digest", "family_replacement_plan_digest",
        "warmup_blocks_min", "warmup_blocks_max", "state_shard_bindings",
        "selection_semantics", "candidate_outcomes_loaded",
        "branch_identities_created", "branches_attempted", "frames_rendered",
        "target_latents_encoded", "scorer_training_started",
        "predictor_checkpoints_opened", "mixed_replacement_scene_request_digest",
    }
    family = str(args.family)
    plan = _mixed_family_replacement_plan(family)
    scenes = pool.get(family, [])
    ordinal = request.get("scene_ordinal")
    if (not isinstance(ordinal, int) or isinstance(ordinal, bool)
            or not 0 <= ordinal < len(scenes)):
        raise RuntimeError("mixed replacement scene ordinal is invalid")
    scene_dir = scenes[ordinal]
    manifest_path = scene_dir / "manifest.json"
    expected_scene = {
        "scene_id": scene_dir.name,
        "scene_dir": str(scene_dir.resolve()),
        "scene_manifest_sha256": file_sha256(manifest_path),
        "scene_manifest_byte_count": manifest_path.stat().st_size,
        "split": scene_dir.parent.parent.name,
        "drive_seed": int(V1._drive_seed(scene_dir.name)),
    }
    interval = request.get("anchor_interval")
    slot = request.get("replacement_slot")
    matching_intervals = [row for row in plan["interval_groups"] if (
        isinstance(interval, dict)
        and interval == {
            "lower_scene_id_exclusive": row["lower_scene_id_exclusive"],
            "upper_scene_id_exclusive": row["upper_scene_id_exclusive"],
            "vacant_ordinals": row["vacant_ordinals"],
        })]
    accepted = request.get("accepted_scene_ids_before")
    expected_slot = None
    if (len(matching_intervals) == 1 and isinstance(accepted, list)
            and len(accepted) < len(matching_intervals[0]["replacement_slots"])):
        expected_slot = matching_intervals[0]["replacement_slots"][len(accepted)]
    if (
        set(request) != expected_keys
        or request.get("schema") != MIXED_REPLACEMENT_SCENE_REQUEST_SCHEMA
        or request.get("status") != STATUS
        or request.get("complete") is not True
        or request.get("binding_receipt") is not False
        or request.get("pool") != "scorer_fit"
        or request.get("family") != family
        or request.get("backend") != "cpu"
        or request.get("scene") != expected_scene
        or len(matching_intervals) != 1
        or slot != expected_slot
        or not isinstance(accepted, list)
        or accepted != sorted(accepted)
        or len(set(accepted)) != len(accepted)
        or (accepted and scene_dir.name <= accepted[-1])
        or scene_dir.name in set(plan["retained_scene_ids"])
        or not _scene_in_open_interval(
            scene_dir.name,
            lower=interval["lower_scene_id_exclusive"],
            upper=interval["upper_scene_id_exclusive"])
        or any(not _scene_in_open_interval(
            value, lower=interval["lower_scene_id_exclusive"],
            upper=interval["upper_scene_id_exclusive"])
               for value in accepted)
        or request.get("retained_scene_ids_digest")
        != canonical_digest(plan["retained_scene_ids"])
        or request.get("rejected_identity_digests")
        != plan["rejected_identity_digests"]
        or request.get("rejected_identity_set_digest")
        != canonical_digest(plan["rejected_identity_digests"])
        or request.get("family_replacement_plan_digest") != canonical_digest(plan)
        or request.get("warmup_blocks_min") != WARMUP_BLOCKS_MIN
        or request.get("warmup_blocks_max") != WARMUP_BLOCKS_MAX
        or request.get("state_shard_bindings") != _state_shard_bindings(
            args, exclusion, [path.name for path in scenes])
        or any(request.get(key) not in (False, 0) for key in (
            "candidate_outcomes_loaded", "branch_identities_created",
            "branches_attempted", "frames_rendered", "target_latents_encoded",
            "scorer_training_started", "predictor_checkpoints_opened"))
    ):
        raise RuntimeError("mixed replacement scene request changed")


def _build_mixed_replacement_scene_capture(
        *, request: dict[str, Any], chosen_state: dict[str, Any] | None,
        rejection_reasons: dict[str, int], worker_failure: str | None,
        blocks_driven: int, attempt_trace: Sequence[dict[str, Any]],
        ) -> dict[str, Any]:
    payload = {
        "schema": MIXED_REPLACEMENT_SCENE_CAPTURE_SCHEMA,
        "status": ("COMPLETE_OUTCOME_FREE_MIXED_REPLACEMENT_SCENE_RESOLUTION"
                   if worker_failure is None
                   else "FAIL_OUTCOME_FREE_MIXED_REPLACEMENT_SCENE_RESOLUTION"),
        "complete": True,
        "binding_receipt": False,
        "mixed_replacement_scene_request_digest": request[
            "mixed_replacement_scene_request_digest"],
        "request": request,
        "family": request["family"],
        "scene_id": request["scene"]["scene_id"],
        "blocks_driven": int(blocks_driven),
        "attempt_trace": list(attempt_trace),
        "chosen_state": chosen_state,
        "scene_rejection_reasons": dict(sorted(rejection_reasons.items())),
        "worker_failure": worker_failure,
        "candidate_outcomes_loaded": False,
        "branch_identities_created": False,
        "branches_attempted": 0,
        "frames_rendered": 0,
        "target_latents_encoded": 0,
        "scorer_training_started": False,
        "predictor_checkpoints_opened": 0,
        "atomic_write_precedes_native_context_cleanup": True,
    }
    payload["mixed_replacement_scene_capture_digest"] = canonical_digest(payload)
    return payload


def _validate_mixed_replacement_scene_capture(
        capture: dict[str, Any], *, expected_request: dict[str, Any]) -> None:
    _verify_self_digest(
        capture, "mixed_replacement_scene_capture_digest",
        "mixed replacement scene capture")
    expected_keys = {
        "schema", "status", "complete", "binding_receipt",
        "mixed_replacement_scene_request_digest", "request", "family",
        "scene_id", "blocks_driven", "attempt_trace", "chosen_state",
        "scene_rejection_reasons", "worker_failure",
        "candidate_outcomes_loaded", "branch_identities_created",
        "branches_attempted", "frames_rendered", "target_latents_encoded",
        "scorer_training_started", "predictor_checkpoints_opened",
        "atomic_write_precedes_native_context_cleanup",
        "mixed_replacement_scene_capture_digest",
    }
    failure = capture.get("worker_failure")
    blocks = capture.get("blocks_driven")
    trace = capture.get("attempt_trace")
    if (
        set(capture) != expected_keys
        or capture.get("schema") != MIXED_REPLACEMENT_SCENE_CAPTURE_SCHEMA
        or capture.get("status") != (
            "COMPLETE_OUTCOME_FREE_MIXED_REPLACEMENT_SCENE_RESOLUTION"
            if failure is None
            else "FAIL_OUTCOME_FREE_MIXED_REPLACEMENT_SCENE_RESOLUTION")
        or capture.get("complete") is not True
        or capture.get("binding_receipt") is not False
        or capture.get("request") != expected_request
        or capture.get("mixed_replacement_scene_request_digest")
        != expected_request["mixed_replacement_scene_request_digest"]
        or capture.get("family") != expected_request["family"]
        or capture.get("scene_id") != expected_request["scene"]["scene_id"]
        or not isinstance(blocks, int) or isinstance(blocks, bool)
        or not 0 <= blocks <= WARMUP_BLOCKS_MAX
        or not isinstance(trace, list)
        or (failure is not None and (not isinstance(failure, str) or not failure))
        or any(capture.get(key) not in (False, 0) for key in (
            "candidate_outcomes_loaded", "branch_identities_created",
            "branches_attempted", "frames_rendered", "target_latents_encoded",
            "scorer_training_started", "predictor_checkpoints_opened"))
        or capture.get("atomic_write_precedes_native_context_cleanup") is not True
    ):
        raise RuntimeError("mixed replacement scene capture is malformed")
    selected_count = 0
    replayed_reasons: dict[str, int] = {}
    for trace_index, row in enumerate(trace):
        expected_block = WARMUP_BLOCKS_MIN + trace_index
        if (
            not isinstance(row, dict)
            or set(row) != {"block_index", "verdict", "reason_key"}
            or row.get("block_index") != expected_block
            or row.get("verdict") not in {"REJECT", "SELECT", "ERROR"}
        ):
            raise RuntimeError("mixed replacement attempt trace is malformed")
        if row["verdict"] == "REJECT":
            reason = row.get("reason_key")
            if not isinstance(reason, str) or not reason:
                raise RuntimeError("mixed replacement rejection reason is missing")
            replayed_reasons[reason] = replayed_reasons.get(reason, 0) + 1
        elif row["verdict"] == "SELECT":
            if row.get("reason_key") is not None or trace_index != len(trace) - 1:
                raise RuntimeError("mixed replacement selection is not terminal")
            selected_count += 1
        else:
            reason = row.get("reason_key")
            if (failure is None or not isinstance(reason, str) or not reason
                    or trace_index != len(trace) - 1
                    or reason != failure):
                raise RuntimeError("mixed replacement error trace is malformed")
    expected_trace_count = max(blocks - WARMUP_BLOCKS_MIN + 1, 0)
    chosen = capture.get("chosen_state")
    if (
        len(trace) != expected_trace_count
        or selected_count != (1 if chosen is not None else 0)
        or (failure is None and chosen is None and blocks != WARMUP_BLOCKS_MAX)
        or (failure is not None and chosen is not None)
        or replayed_reasons != capture.get("scene_rejection_reasons")
    ):
        raise RuntimeError("mixed replacement trace/capture reducer changed")
    if chosen is None:
        return
    if not isinstance(chosen, dict):
        raise RuntimeError("mixed replacement chosen state is malformed")
    slot = expected_request["replacement_slot"]
    expected_chosen_keys = {
        "state_id", "family", "scene_id", "scene_dir",
        "scene_manifest_sha256", "scene_manifest_byte_count", "split",
        "drive_seed", "stratum", "split_role", "warmup_blocks",
        "source_step", "episode_id", "episode_cluster_id", "cell_id",
        "boundary", "goal", "goal_type", "body_clearance_m", "clearance_m",
        "completion_rotation_eligibility_vector", "snapshot_task_status",
        "previous_applied_command", "state_identity_digest",
    }
    if (
        set(chosen) != expected_chosen_keys
        or chosen.get("state_id") != slot["state_id"]
        or chosen.get("family") != slot["family"]
        or chosen.get("stratum") != slot["stratum"]
        or chosen.get("split_role") != slot["split_role"]
        or chosen.get("scene_id") != expected_request["scene"]["scene_id"]
        or chosen.get("scene_dir") != expected_request["scene"]["scene_dir"]
        or chosen.get("scene_manifest_sha256")
        != expected_request["scene"]["scene_manifest_sha256"]
        or chosen.get("scene_manifest_byte_count")
        != expected_request["scene"]["scene_manifest_byte_count"]
        or chosen.get("split") != expected_request["scene"]["split"]
        or chosen.get("drive_seed") != expected_request["scene"]["drive_seed"]
        or chosen.get("warmup_blocks") != blocks
        or chosen.get("state_identity_digest") != _state_identity_digest(chosen)
        or chosen.get("state_identity_digest")
        in set(expected_request["rejected_identity_digests"])
        or _replacement_reuses_any_rejected_snapshot(
            chosen, expected_request["rejected_identity_digests"])
    ):
        raise RuntimeError("mixed replacement chosen identity changed")
    vector = chosen["completion_rotation_eligibility_vector"]
    rotations = vector.get("rotations") if isinstance(vector, dict) else None
    if not isinstance(rotations, list) or len(rotations) != 12:
        raise RuntimeError("mixed replacement rotation vector is malformed")
    first = rotations[0]
    try:
        expected_vector = STATE_SELECTOR.completion_rotation_eligibility_vector(
            graph_hops=int(first["graph_hops_diagnostic"]),
            reachable=bool(first["reachable"]),
            continuous_geodesic_m=float(first["continuous_geodesic_m"]),
            bearing_body_rad=float(first["bearing_body_rad"]),
            task_status=first["task_status"],
            previous_applied_command=first["previous_applied_command"],
        )
    except (KeyError, TypeError, ValueError,
            STATE_SELECTOR.StateSelectorAmendmentError) as exc:
        raise RuntimeError("mixed replacement vector cannot be reconstructed") from exc
    try:
        STATE_SELECTOR.validate_snapshot_task_status_binding(
            chosen["snapshot_task_status"], first["task_status"],
            designated_goal_cell=int(chosen["goal"]["landmark_cell"]))
    except (KeyError, TypeError, ValueError,
            STATE_SELECTOR.StateSelectorAmendmentError) as exc:
        raise RuntimeError(
            "mixed replacement snapshot task status changed"
        ) from exc
    if (
        vector != expected_vector
        or vector.get("eligible_under_at_least_one_rotation") is not True
        or chosen["previous_applied_command"] != first["previous_applied_command"]
        or int(chosen["goal"]["graph_edges"])
        != int(first["graph_hops_diagnostic"])
        or float(chosen["goal"]["start_geodesic_m"])
        != float(first["continuous_geodesic_m"])
        or float(chosen["goal"]["bearing_body_rad"])
        != float(first["bearing_body_rad"])
    ):
        raise RuntimeError("mixed replacement completion evidence changed")


def _load_valid_mixed_replacement_scene_capture(
        *, path: Path, request: dict[str, Any]) -> dict[str, Any] | None:
    expected_raw = _mixed_replacement_capture_path(
        OUT_ROOT / "scorer_fit", str(request["family"]),
        str(request["mixed_replacement_scene_request_digest"]))
    pinned = _pin_generated_path(path, expected_raw)
    if not pinned.is_file() or pinned.is_symlink():
        return None
    try:
        payload = json.loads(pinned.read_text())
        _validate_mixed_replacement_scene_capture(
            payload, expected_request=request)
    except (OSError, json.JSONDecodeError, RuntimeError):
        return None
    return payload


def _build_state_resolution_scene_request(
        *, args: argparse.Namespace, out: Path, scene_dir: Path,
        scene_ordinal: int, found: dict[str, int], need: dict[str, int],
        exclusion: dict[str, Any], family_allow_list: Sequence[str],
        ) -> dict[str, Any]:
    family = str(args.family)
    manifest_path = scene_dir / "manifest.json"
    requested = (
        ["evaluation"] if "evaluation" in need
        and found.get("evaluation", 0) < need["evaluation"]
        else [name for name in STRATA
              if found.get(name, 0) < need.get(name, 0)]
    )
    if not requested:
        raise RuntimeError("state-resolution request has no unmet stratum")
    payload = {
        "schema": STATE_RESOLUTION_SCENE_REQUEST_SCHEMA,
        "status": STATUS,
        "complete": True,
        "binding_receipt": False,
        "pool": str(args.pool),
        "family": family,
        "backend": str(args.backend),
        "scene_ordinal": int(scene_ordinal),
        "scene": {
            "scene_id": str(scene_dir.name),
            "scene_dir": str(scene_dir.resolve()),
            "scene_manifest_sha256": file_sha256(manifest_path),
            "scene_manifest_byte_count": manifest_path.stat().st_size,
            "split": str(scene_dir.parent.parent.name),
            "drive_seed": int(V1._drive_seed(scene_dir.name)),
        },
        "found_before_scene": {
            key: int(found[key]) for key in need
        },
        "required_counts": {key: int(need[key]) for key in need},
        "requested_strata_in_priority_order": requested,
        "stratum_priority": list(STRATA),
        "warmup_blocks_min": int(WARMUP_BLOCKS_MIN),
        "warmup_blocks_max": int(WARMUP_BLOCKS_MAX),
        "state_shard_bindings": _state_shard_bindings(
            args, exclusion, list(family_allow_list)),
        "candidate_outcomes_loaded": False,
        "branch_identities_created": False,
        "branches_attempted": 0,
        "frames_rendered": 0,
        "target_latents_encoded": 0,
        "scorer_training_started": False,
        "predictor_checkpoints_opened": 0,
        "resolver_algorithm_digest":
            canonical_digest(STATE_RESOLUTION_REDUCER_CONTRACT),
        "selection_semantics": STATE_RESOLUTION_SELECTION_SEMANTICS,
    }
    payload["state_resolution_scene_request_digest"] = canonical_digest(payload)
    raw_path = _state_resolution_request_path(
        out, family, payload["state_resolution_scene_request_digest"])
    path = _pin_generated_path(raw_path, raw_path)
    if path.is_file():
        existing = json.loads(path.read_text())
        if existing != payload:
            raise RuntimeError("state-resolution request digest collision")
    elif path.exists():
        raise RuntimeError("state-resolution request path is not a regular file")
    else:
        atomic_json(path, payload)
    return payload


def _validate_state_resolution_scene_request(
        request: dict[str, Any], *, args: argparse.Namespace,
        out: Path, pool: dict[str, list[Path]], exclusion: dict[str, Any],
        expected_state_shard_bindings: Mapping[str, Any] | None = None,
        ) -> None:
    _verify_self_digest(
        request, "state_resolution_scene_request_digest",
        "state-resolution scene request")
    family = str(args.family)
    expected_keys = {
        "schema", "status", "complete", "binding_receipt", "pool", "family",
        "backend", "scene_ordinal", "scene", "found_before_scene",
        "required_counts", "requested_strata_in_priority_order",
        "stratum_priority", "warmup_blocks_min", "warmup_blocks_max",
        "state_shard_bindings", "candidate_outcomes_loaded",
        "branch_identities_created", "branches_attempted", "frames_rendered",
        "target_latents_encoded", "scorer_training_started",
        "predictor_checkpoints_opened", "selection_semantics",
        "resolver_algorithm_digest",
        "state_resolution_scene_request_digest",
    }
    if set(request) != expected_keys:
        raise RuntimeError("state-resolution request key surface changed")
    scenes = pool.get(family, [])
    ordinal = request.get("scene_ordinal")
    if not isinstance(ordinal, int) or isinstance(ordinal, bool) \
            or not (0 <= ordinal < len(scenes)):
        raise RuntimeError("state-resolution request scene ordinal is invalid")
    scene_dir = scenes[ordinal]
    scene = request.get("scene")
    manifest_path = scene_dir / "manifest.json"
    spec = POOLS[str(args.pool)]
    need = ({"evaluation": spec["states_per_family"]}
            if spec["strata"] is None else dict(spec["strata"]))
    if (str(args.pool) == "scorer_fit" and family == REACHABILITY_REDRIVE_FAMILY):
        need["completion_enriched"] = 0
    found = request.get("found_before_scene")
    requested = ((["evaluation"] if "evaluation" in need
                  and found.get("evaluation", 0) < need["evaluation"]
                  else [name for name in STRATA
                        if found.get(name, 0) < need.get(name, 0)])
                 if isinstance(found, dict) else None)
    expected_scene = {
        "scene_id": str(scene_dir.name),
        "scene_dir": str(scene_dir.resolve()),
        "scene_manifest_sha256": file_sha256(manifest_path),
        "scene_manifest_byte_count": manifest_path.stat().st_size,
        "split": str(scene_dir.parent.parent.name),
        "drive_seed": int(V1._drive_seed(scene_dir.name)),
    }
    expected_bindings = (
        _state_shard_bindings(args, exclusion, [path.name for path in scenes])
        if expected_state_shard_bindings is None
        else dict(expected_state_shard_bindings)
    )
    if (
        request.get("schema") != STATE_RESOLUTION_SCENE_REQUEST_SCHEMA
        or request.get("status") != STATUS
        or request.get("complete") is not True
        or request.get("binding_receipt") is not False
        or request.get("pool") != args.pool
        or request.get("family") != family
        or request.get("backend") != args.backend
        or scene != expected_scene
        or request.get("required_counts") != need
        or not isinstance(found, dict)
        or set(found) != set(need)
        or any(not isinstance(value, int) or isinstance(value, bool)
               or value < 0 or value > need[key]
               for key, value in found.items())
        or request.get("requested_strata_in_priority_order") != requested
        or not requested
        or request.get("stratum_priority") != list(STRATA)
        or request.get("warmup_blocks_min") != WARMUP_BLOCKS_MIN
        or request.get("warmup_blocks_max") != WARMUP_BLOCKS_MAX
        or request.get("state_shard_bindings") != expected_bindings
        or request.get("candidate_outcomes_loaded") is not False
        or request.get("branch_identities_created") is not False
        or request.get("branches_attempted") != 0
        or request.get("frames_rendered") != 0
        or request.get("target_latents_encoded") != 0
        or request.get("scorer_training_started") is not False
        or request.get("predictor_checkpoints_opened") != 0
        or request.get("resolver_algorithm_digest")
        != canonical_digest(STATE_RESOLUTION_REDUCER_CONTRACT)
        or request.get("selection_semantics")
        != STATE_RESOLUTION_SELECTION_SEMANTICS
    ):
        raise RuntimeError("state-resolution request does not match live frozen inputs")


def _build_state_resolution_scene_capture(
        *, request: dict[str, Any], chosen_state: dict[str, Any] | None,
        rejection_reasons: dict[str, int], worker_failure: str | None,
        blocks_driven: int, attempt_trace: Sequence[dict[str, Any]]
        ) -> dict[str, Any]:
    payload = {
        "schema": STATE_RESOLUTION_SCENE_CAPTURE_SCHEMA,
        "status": ("COMPLETE_OUTCOME_FREE_SCENE_RESOLUTION"
                   if worker_failure is None
                   else "FAIL_OUTCOME_FREE_SCENE_RESOLUTION"),
        "complete": True,
        "binding_receipt": False,
        "state_resolution_scene_request_digest":
            request["state_resolution_scene_request_digest"],
        "request": request,
        "family": request["family"],
        "scene_id": request["scene"]["scene_id"],
        "blocks_driven": int(blocks_driven),
        "attempt_trace": list(attempt_trace),
        "chosen_state": chosen_state,
        "scene_rejection_reasons": dict(sorted(rejection_reasons.items())),
        "worker_failure": worker_failure,
        "candidate_outcomes_loaded": False,
        "branch_identities_created": False,
        "branches_attempted": 0,
        "frames_rendered": 0,
        "target_latents_encoded": 0,
        "scorer_training_started": False,
        "predictor_checkpoints_opened": 0,
        "atomic_write_precedes_native_context_cleanup": True,
    }
    payload["state_resolution_scene_capture_digest"] = canonical_digest(payload)
    return payload


def _replay_state_resolution_attempt_trace(
        *, request: dict[str, Any], attempt_trace: Any,
        blocks_driven: int, worker_failure: str | None
        ) -> tuple[dict[str, int], str | None]:
    """Replay the frozen in-memory reducer from durable per-block evidence."""

    if not isinstance(attempt_trace, list):
        raise RuntimeError("state-resolution attempt trace is not a list")
    requested = list(request["requested_strata_in_priority_order"])
    expected_first = WARMUP_BLOCKS_MIN
    selected: str | None = None
    rejections: dict[str, int] = {}
    for trace_index, block in enumerate(attempt_trace):
        expected_block = expected_first + trace_index
        if (
            not isinstance(block, dict)
            or set(block) != {"block_index", "attempts"}
            or block.get("block_index") != expected_block
            or not isinstance(block.get("attempts"), list)
            or not block["attempts"]
        ):
            raise RuntimeError("state-resolution block trace is malformed")
        attempts = block["attempts"]
        if len(attempts) > len(requested):
            raise RuntimeError("state-resolution trace attempts too many strata")
        block_selected: str | None = None
        rejected_labels: list[str] = []
        for ordinal, attempt in enumerate(attempts):
            if (
                not isinstance(attempt, dict)
                or set(attempt) != {"stratum", "verdict", "reason_key"}
                or attempt.get("stratum") != requested[ordinal]
                or attempt.get("verdict") not in {"REJECT", "SELECT"}
            ):
                raise RuntimeError("state-resolution stratum trace is malformed")
            if attempt["verdict"] == "REJECT":
                if not isinstance(attempt.get("reason_key"), str) \
                        or not attempt["reason_key"]:
                    raise RuntimeError(
                        "state-resolution rejection trace lacks reason")
                rejected_labels.append(
                    (str(attempt["reason_key"])
                     if attempt["stratum"] == "evaluation"
                     else f"{attempt['stratum']}:{attempt['reason_key']}")
                )
            else:
                if attempt.get("reason_key") is not None \
                        or ordinal != len(attempts) - 1:
                    raise RuntimeError(
                        "state-resolution selected trace is not terminal")
                block_selected = str(attempt["stratum"])
        if block_selected is None:
            if len(attempts) != len(requested):
                raise RuntimeError(
                    "state-resolution no-selection block omitted a stratum")
            key = "|".join(rejected_labels)
            rejections[key] = rejections.get(key, 0) + 1
        else:
            if selected is not None or trace_index != len(attempt_trace) - 1:
                raise RuntimeError(
                    "state-resolution trace selected more than once or not last")
            selected = block_selected
    if attempt_trace and attempt_trace[-1]["block_index"] != blocks_driven:
        raise RuntimeError("state-resolution trace does not reach blocks_driven")
    if worker_failure is None:
        if selected is None:
            if blocks_driven != WARMUP_BLOCKS_MAX:
                raise RuntimeError(
                    "successful no-selection trace did not exhaust the scene")
        elif blocks_driven < WARMUP_BLOCKS_MIN:
            raise RuntimeError("state-resolution selected before warmup")
        expected_trace_count = max(
            blocks_driven - WARMUP_BLOCKS_MIN + 1, 0)
        if len(attempt_trace) != expected_trace_count:
            raise RuntimeError("state-resolution trace skipped an eligible block")
    return dict(sorted(rejections.items())), selected


def _validate_state_resolution_scene_capture(
        capture: dict[str, Any], *, expected_request: dict[str, Any]) -> None:
    _verify_self_digest(
        capture, "state_resolution_scene_capture_digest",
        "state-resolution scene capture")
    expected_keys = {
        "schema", "status", "complete", "binding_receipt",
        "state_resolution_scene_request_digest", "request", "family",
        "scene_id", "blocks_driven", "chosen_state",
        "attempt_trace",
        "scene_rejection_reasons", "worker_failure",
        "candidate_outcomes_loaded", "branch_identities_created",
        "branches_attempted", "frames_rendered", "target_latents_encoded",
        "scorer_training_started", "predictor_checkpoints_opened",
        "atomic_write_precedes_native_context_cleanup",
        "state_resolution_scene_capture_digest",
    }
    failure = capture.get("worker_failure")
    if (
        set(capture) != expected_keys
        or capture.get("schema") != STATE_RESOLUTION_SCENE_CAPTURE_SCHEMA
        or capture.get("status") != (
            "COMPLETE_OUTCOME_FREE_SCENE_RESOLUTION" if failure is None
            else "FAIL_OUTCOME_FREE_SCENE_RESOLUTION")
        or capture.get("complete") is not True
        or capture.get("binding_receipt") is not False
        or capture.get("request") != expected_request
        or capture.get("state_resolution_scene_request_digest")
        != expected_request["state_resolution_scene_request_digest"]
        or capture.get("family") != expected_request["family"]
        or capture.get("scene_id") != expected_request["scene"]["scene_id"]
        or not isinstance(capture.get("blocks_driven"), int)
        or isinstance(capture.get("blocks_driven"), bool)
        or not (0 <= capture["blocks_driven"] <= WARMUP_BLOCKS_MAX)
        or (failure is not None
            and (not isinstance(failure, str) or not failure))
        or capture.get("candidate_outcomes_loaded") is not False
        or capture.get("branch_identities_created") is not False
        or capture.get("branches_attempted") != 0
        or capture.get("frames_rendered") != 0
        or capture.get("target_latents_encoded") != 0
        or capture.get("scorer_training_started") is not False
        or capture.get("predictor_checkpoints_opened") != 0
        or capture.get("atomic_write_precedes_native_context_cleanup") is not True
    ):
        raise RuntimeError("state-resolution scene capture is malformed")
    reasons = capture.get("scene_rejection_reasons")
    if (not isinstance(reasons, dict)
            or any(not isinstance(key, str) or not key
                   or not isinstance(value, int) or isinstance(value, bool)
                   or value < 0 for key, value in reasons.items())):
        raise RuntimeError("state-resolution rejection evidence is malformed")
    replayed_reasons, replayed_selected = _replay_state_resolution_attempt_trace(
        request=expected_request, attempt_trace=capture.get("attempt_trace"),
        blocks_driven=int(capture["blocks_driven"]), worker_failure=failure)
    if replayed_reasons != reasons:
        raise RuntimeError(
            "state-resolution rejection ledger differs from attempt trace")
    chosen = capture.get("chosen_state")
    if failure is not None and chosen is not None:
        raise RuntimeError("failed state-resolution capture selected a state")
    if chosen is None:
        if failure is None and replayed_selected is not None:
            raise RuntimeError("state-resolution trace selected an omitted state")
        return
    if not isinstance(chosen, dict):
        raise RuntimeError("state-resolution chosen state is malformed")
    requested = expected_request["requested_strata_in_priority_order"]
    stratum = chosen.get("stratum")
    if replayed_selected != stratum:
        raise RuntimeError("state-resolution chosen state differs from trace")
    found = expected_request["found_before_scene"]
    ordinal = found.get(stratum)
    expected_id = (
        f"{expected_request['pool']}-{expected_request['family']}-"
        f"{stratum}-{ordinal:02d}"
    )
    expected_split_role = (
        "evaluation" if expected_request["pool"] == "final_eval"
        else ("calibration" if ordinal == 0 else "fit")
    )
    expected_chosen_keys = {
        "state_id", "family", "scene_id", "scene_dir",
        "scene_manifest_sha256", "scene_manifest_byte_count", "split",
        "drive_seed", "stratum", "split_role", "warmup_blocks",
        "source_step", "episode_id", "episode_cluster_id", "cell_id",
        "boundary", "goal", "goal_type", "body_clearance_m", "clearance_m",
        "state_identity_digest",
    }
    if stratum == "completion_enriched":
        expected_chosen_keys.update({
            "completion_rotation_eligibility_vector", "snapshot_task_status",
            "previous_applied_command",
        })
    if (
        set(chosen) != expected_chosen_keys
        or stratum not in requested
        or not isinstance(ordinal, int)
        or chosen.get("state_id") != expected_id
        or chosen.get("split_role") != expected_split_role
        or chosen.get("family") != expected_request["family"]
        or chosen.get("scene_id") != expected_request["scene"]["scene_id"]
        or chosen.get("scene_dir") != expected_request["scene"]["scene_dir"]
        or chosen.get("scene_manifest_sha256")
        != expected_request["scene"]["scene_manifest_sha256"]
        or chosen.get("scene_manifest_byte_count")
        != expected_request["scene"]["scene_manifest_byte_count"]
        or chosen.get("split") != expected_request["scene"]["split"]
        or chosen.get("drive_seed") != expected_request["scene"]["drive_seed"]
        or chosen.get("state_identity_digest") != _state_identity_digest(chosen)
        or not isinstance(chosen.get("warmup_blocks"), int)
        or not (WARMUP_BLOCKS_MIN <= chosen["warmup_blocks"]
                <= WARMUP_BLOCKS_MAX)
        or chosen.get("warmup_blocks") != capture["blocks_driven"]
        or not isinstance(chosen.get("boundary"), dict)
        or not isinstance(chosen.get("goal"), dict)
        or chosen.get("goal_type") != chosen["goal"].get("material_id")
    ):
        raise RuntimeError("state-resolution chosen identity changed")
    if stratum == "completion_enriched":
        vector = chosen["completion_rotation_eligibility_vector"]
        rotations = vector.get("rotations") if isinstance(vector, dict) else None
        if not isinstance(rotations, list) or len(rotations) != 12:
            raise RuntimeError(
                "state-resolution completion vector is malformed")
        first = rotations[0]
        try:
            expected_vector = \
                STATE_SELECTOR.completion_rotation_eligibility_vector(
                    graph_hops=int(first["graph_hops_diagnostic"]),
                    reachable=bool(first["reachable"]),
                    continuous_geodesic_m=float(
                        first["continuous_geodesic_m"]),
                    bearing_body_rad=float(first["bearing_body_rad"]),
                    task_status=first["task_status"],
                    previous_applied_command=first[
                        "previous_applied_command"],
                )
        except (KeyError, TypeError, ValueError,
                STATE_SELECTOR.StateSelectorAmendmentError) as exc:
            raise RuntimeError(
                "state-resolution completion vector cannot be recomputed"
            ) from exc
        try:
            STATE_SELECTOR.validate_snapshot_task_status_binding(
                chosen["snapshot_task_status"], first["task_status"],
                designated_goal_cell=int(chosen["goal"]["landmark_cell"]))
        except (KeyError, TypeError, ValueError,
                STATE_SELECTOR.StateSelectorAmendmentError) as exc:
            raise RuntimeError(
                "state-resolution snapshot task status changed"
            ) from exc
        if (
            vector != expected_vector
            or chosen["previous_applied_command"]
            != first["previous_applied_command"]
            or int(chosen["goal"]["graph_edges"])
            != int(first["graph_hops_diagnostic"])
            or float(chosen["goal"]["start_geodesic_m"])
            != float(first["continuous_geodesic_m"])
            or float(chosen["goal"]["bearing_body_rad"])
            != float(first["bearing_body_rad"])
            or vector.get("eligible_under_at_least_one_rotation") is not True
        ):
            raise RuntimeError(
                "state-resolution completion reachability evidence changed")


def _load_valid_state_resolution_scene_capture(
        *, path: Path, request: dict[str, Any]) -> dict[str, Any] | None:
    expected_raw = _state_resolution_capture_path(
        OUT_ROOT / str(request["pool"]), str(request["family"]),
        str(request["state_resolution_scene_request_digest"]))
    pinned = _pin_generated_path(path, expected_raw)
    if not pinned.is_file() or pinned.is_symlink():
        return None
    try:
        payload = json.loads(pinned.read_text())
        _validate_state_resolution_scene_capture(
            payload, expected_request=request)
    except (OSError, json.JSONDecodeError, RuntimeError):
        return None
    return payload


def _execute_state_resolution_scene_worker(
        *, args: argparse.Namespace, request: dict[str, Any], out: Path,
        pool: dict[str, list[Path]], exclusion: dict[str, Any]
        ) -> dict[str, Any]:
    """Resolve one scene and persist its capture before native teardown."""

    _validate_state_resolution_scene_request(
        request, args=args, out=out, pool=pool, exclusion=exclusion)
    capture_path = _state_resolution_capture_path(
        out, str(args.family), request["state_resolution_scene_request_digest"])
    existing = _load_valid_state_resolution_scene_capture(
        path=capture_path, request=request)
    if existing is not None:
        return existing
    if capture_path.exists():
        _preserve_invalid(
            capture_path, out, "state-resolution-scene-capture-invalid")

    ctx = None
    chosen: dict[str, Any] | None = None
    reasons: dict[str, int] = {}
    attempt_trace: list[dict[str, Any]] = []
    blocks_driven = 0
    worker_failure: str | None = None
    try:
        try:
            scene_dir = pool[str(args.family)][int(request["scene_ordinal"])]
            shared = V1._load_shared(args.backend)
            ctx = V1.build_context(
                scene_dir, seed=int(request["scene"]["drive_seed"]),
                backend=args.backend, shared=shared)
            topology = V12.link_topology(ctx)
            ctx.begin_episode()
            for block_idx in range(WARMUP_BLOCKS_MAX):
                ctx.drive_one_block()
                blocks_driven = block_idx + 1
                if blocks_driven < WARMUP_BLOCKS_MIN:
                    continue
                attempted: list[str] = []
                block_attempts: list[dict[str, Any]] = []
                record = None
                wanted: list[str] = []
                for name in request["requested_strata_in_priority_order"]:
                    verdict = (classify_state(ctx, topology)
                               if name == "evaluation"
                               else classify_state(
                                   ctx, topology, requested_stratum=name))
                    if isinstance(verdict, str):
                        reason_key = verdict.split(":")[0]
                        attempted.append(f"{name}:{reason_key}")
                        block_attempts.append({
                            "stratum": name,
                            "verdict": "REJECT",
                            "reason_key": reason_key,
                        })
                        continue
                    record, _field, _eligible = verdict
                    wanted = [name]
                    block_attempts.append({
                        "stratum": name,
                        "verdict": "SELECT",
                        "reason_key": None,
                    })
                    break
                attempt_trace.append({
                    "block_index": blocks_driven,
                    "attempts": block_attempts,
                })
                if not wanted:
                    key = ("no_requested_stratum" if not attempted
                           else "|".join(attempted))
                    reasons[key] = reasons.get(key, 0) + 1
                    continue
                stratum = wanted[0]
                ordinal = int(request["found_before_scene"][stratum])
                split_role = (
                    "evaluation" if args.pool == "final_eval"
                    else ("calibration" if ordinal == 0 else "fit"))
                state_id = f"{args.pool}-{args.family}-{stratum}-{ordinal:02d}"
                assert record is not None
                chosen = {
                    "state_id": state_id,
                    "family": str(args.family),
                    "scene_id": scene_dir.name,
                    "scene_dir": str(scene_dir.resolve()),
                    "scene_manifest_sha256":
                        request["scene"]["scene_manifest_sha256"],
                    "scene_manifest_byte_count":
                        request["scene"]["scene_manifest_byte_count"],
                    "split": request["scene"]["split"],
                    "drive_seed": int(request["scene"]["drive_seed"]),
                    "stratum": stratum,
                    "split_role": split_role,
                    "warmup_blocks": blocks_driven,
                    "source_step": int(record["boundary"]["source_step"]),
                    "episode_id": int(
                        ctx.runner.episode_states[0].episode_id),
                    "episode_cluster_id": (
                        f"{scene_dir.name}/env0/ep"
                        f"{int(ctx.runner.episode_states[0].episode_id)}"
                    ),
                    "cell_id": int(record["cell_id"]),
                    "boundary": record["boundary"],
                    "goal": record["goal"],
                    "goal_type": record["goal"]["material_id"],
                    "body_clearance_m": float(record["body_clearance_m"]),
                    "clearance_m": float(record["clearance_m"]),
                }
                if stratum == "completion_enriched":
                    chosen["completion_rotation_eligibility_vector"] = record[
                        "completion_rotation_eligibility_vector"]
                    chosen["snapshot_task_status"] = record[
                        "snapshot_task_status"]
                    chosen["previous_applied_command"] = record[
                        "previous_applied_command"]
                chosen["state_identity_digest"] = _state_identity_digest(chosen)
                break
        except Exception as exc:
            worker_failure = f"{type(exc).__name__}:{str(exc)[:500]}"
            chosen = None

        capture = _build_state_resolution_scene_capture(
            request=request, chosen_state=chosen,
            rejection_reasons=reasons, worker_failure=worker_failure,
            blocks_driven=blocks_driven, attempt_trace=attempt_trace)
        _validate_state_resolution_scene_capture(
            capture, expected_request=request)
        atomic_json(capture_path, capture)
        return capture
    finally:
        _FIELD_CACHE.clear()
        if ctx is not None:
            del ctx
        gc.collect()


def _execute_mixed_replacement_scene_worker(
        *, args: argparse.Namespace, request: dict[str, Any], out: Path,
        pool: dict[str, list[Path]], exclusion: dict[str, Any]
        ) -> dict[str, Any]:
    """Resolve one replacement scene and durably write before teardown."""

    _validate_mixed_replacement_scene_request(
        request, args=args, out=out, pool=pool, exclusion=exclusion)
    raw_capture_path = _mixed_replacement_capture_path(
        out, str(args.family), request["mixed_replacement_scene_request_digest"])
    capture_path = _pin_generated_path(raw_capture_path, raw_capture_path)
    existing = _load_valid_mixed_replacement_scene_capture(
        path=raw_capture_path, request=request)
    if existing is not None:
        return existing
    if capture_path.exists():
        _preserve_invalid(
            capture_path, out, "mixed-replacement-scene-capture-invalid")

    ctx = None
    chosen: dict[str, Any] | None = None
    reasons: dict[str, int] = {}
    attempt_trace: list[dict[str, Any]] = []
    blocks_driven = 0
    worker_failure: str | None = None
    try:
        try:
            scene_dir = pool[str(args.family)][int(request["scene_ordinal"])]
            shared = V1._load_shared(args.backend)
            ctx = V1.build_context(
                scene_dir, seed=int(request["scene"]["drive_seed"]),
                backend=args.backend, shared=shared)
            topology = V12.link_topology(ctx)
            ctx.begin_episode()
            for block_index in range(WARMUP_BLOCKS_MAX):
                ctx.drive_one_block()
                blocks_driven = block_index + 1
                if blocks_driven < WARMUP_BLOCKS_MIN:
                    continue
                verdict = classify_state(
                    ctx, topology, requested_stratum="completion_enriched")
                if isinstance(verdict, str):
                    reason = verdict.split(":")[0]
                    reasons[reason] = reasons.get(reason, 0) + 1
                    attempt_trace.append({
                        "block_index": blocks_driven,
                        "verdict": "REJECT",
                        "reason_key": reason,
                    })
                    continue
                record, _field, _eligible = verdict
                slot = request["replacement_slot"]
                candidate = {
                    "state_id": str(slot["state_id"]),
                    "family": str(slot["family"]),
                    "scene_id": scene_dir.name,
                    "scene_dir": str(scene_dir.resolve()),
                    "scene_manifest_sha256":
                        request["scene"]["scene_manifest_sha256"],
                    "scene_manifest_byte_count":
                        request["scene"]["scene_manifest_byte_count"],
                    "split": request["scene"]["split"],
                    "drive_seed": int(request["scene"]["drive_seed"]),
                    "stratum": "completion_enriched",
                    "split_role": str(slot["split_role"]),
                    "warmup_blocks": blocks_driven,
                    "source_step": int(record["boundary"]["source_step"]),
                    "episode_id": int(ctx.runner.episode_states[0].episode_id),
                    "episode_cluster_id": (
                        f"{scene_dir.name}/env0/ep"
                        f"{int(ctx.runner.episode_states[0].episode_id)}"
                    ),
                    "cell_id": int(record["cell_id"]),
                    "boundary": record["boundary"],
                    "goal": record["goal"],
                    "goal_type": record["goal"]["material_id"],
                    "body_clearance_m": float(record["body_clearance_m"]),
                    "clearance_m": float(record["clearance_m"]),
                    "completion_rotation_eligibility_vector": record[
                        "completion_rotation_eligibility_vector"],
                    "snapshot_task_status": record["snapshot_task_status"],
                    "previous_applied_command": record[
                        "previous_applied_command"],
                }
                candidate["state_identity_digest"] = _state_identity_digest(candidate)
                if _replacement_reuses_any_rejected_snapshot(
                        candidate, request["rejected_identity_digests"]):
                    reason = "rejected_predecessor_physical_snapshot_collision"
                    reasons[reason] = reasons.get(reason, 0) + 1
                    attempt_trace.append({
                        "block_index": blocks_driven,
                        "verdict": "REJECT",
                        "reason_key": reason,
                    })
                    continue
                if candidate["state_identity_digest"] in set(
                        request["rejected_identity_digests"]):
                    reason = "rejected_predecessor_identity_digest_collision"
                    reasons[reason] = reasons.get(reason, 0) + 1
                    attempt_trace.append({
                        "block_index": blocks_driven,
                        "verdict": "REJECT",
                        "reason_key": reason,
                    })
                    continue
                chosen = candidate
                attempt_trace.append({
                    "block_index": blocks_driven,
                    "verdict": "SELECT",
                    "reason_key": None,
                })
                break
        except Exception as exc:
            worker_failure = f"{type(exc).__name__}:{str(exc)[:500]}"
            chosen = None
            if blocks_driven >= WARMUP_BLOCKS_MIN:
                expected_trace_count = blocks_driven - WARMUP_BLOCKS_MIN + 1
                if len(attempt_trace) == expected_trace_count - 1:
                    attempt_trace.append({
                        "block_index": blocks_driven,
                        "verdict": "ERROR",
                        "reason_key": worker_failure,
                    })

        capture = _build_mixed_replacement_scene_capture(
            request=request, chosen_state=chosen, rejection_reasons=reasons,
            worker_failure=worker_failure, blocks_driven=blocks_driven,
            attempt_trace=attempt_trace)
        _validate_mixed_replacement_scene_capture(
            capture, expected_request=request)
        atomic_json(capture_path, capture)
        return capture
    finally:
        _FIELD_CACHE.clear()
        if ctx is not None:
            del ctx
        gc.collect()


def _run_mixed_replacement_scene_subprocess(
        args: argparse.Namespace, *, request_digest: str) -> int:
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--pool", "scorer_fit", "--stage", "states",
        "--family", str(args.family), "--backend", str(args.backend),
        "--mixed-replacement-scene-request-digest", request_digest,
    ]
    completed = subprocess.run(
        command, cwd=ROOT, env={**os.environ, "PYTHONUNBUFFERED": "1"},
        check=False)
    return int(completed.returncode)


def _get_or_run_mixed_replacement_scene_capture(
        *, args: argparse.Namespace, request: dict[str, Any], out: Path,
        runner: Any = None) -> dict[str, Any]:
    digest = str(request["mixed_replacement_scene_request_digest"])
    path = _mixed_replacement_capture_path(out, str(args.family), digest)
    capture = _load_valid_mixed_replacement_scene_capture(
        path=path, request=request)
    if capture is None:
        if path.exists():
            _preserve_invalid(
                path, out, "mixed-replacement-scene-capture-invalid")
        run = (_run_mixed_replacement_scene_subprocess
               if runner is None else runner)
        return_code = int(run(args, request_digest=digest))
        capture = _load_valid_mixed_replacement_scene_capture(
            path=path, request=request)
        if capture is None:
            raise RuntimeError(
                f"mixed replacement worker {request['scene']['scene_id']} "
                f"exited {return_code} without a valid durable capture"
            )
        if capture.get("worker_failure") is None and return_code != 0:
            print(
                f"[recovery] mixed replacement worker "
                f"{request['scene']['scene_id']} exited {return_code} after "
                "its valid atomic capture; retaining capture",
                flush=True)
    if capture.get("worker_failure") is not None:
        raise RuntimeError(
            f"mixed replacement worker failed for "
            f"{request['scene']['scene_id']}: {capture['worker_failure']}"
        )
    return capture


def stage_mixed_replacement_scene_worker(args: argparse.Namespace) -> int:
    if (args.pool != "scorer_fit" or args.family is None
            or args.backend != "cpu"):
        raise RuntimeError("mixed replacement worker requires scorer-fit/family/cpu")
    digest = str(args.mixed_replacement_scene_request_digest)
    out = OUT_ROOT / "scorer_fit"
    raw_path = _mixed_replacement_request_path(out, str(args.family), digest)
    path = _pin_generated_path(raw_path, raw_path)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("mixed replacement scene request is missing")
    request = json.loads(path.read_text())
    if request.get("mixed_replacement_scene_request_digest") != digest:
        raise RuntimeError("mixed replacement worker request digest changed")
    _load_clean_source_launch_receipt()
    pool, exclusion = scene_pool("scorer_fit")
    capture = _execute_mixed_replacement_scene_worker(
        args=args, request=request, out=out, pool=pool, exclusion=exclusion)
    print(json.dumps({
        "scene_id": capture["scene_id"],
        "mixed_replacement_scene_capture_digest": capture[
            "mixed_replacement_scene_capture_digest"],
        "chosen_state_id": (None if capture["chosen_state"] is None
                            else capture["chosen_state"]["state_id"]),
        "pass": capture["worker_failure"] is None,
    }, indent=2, sort_keys=True))
    return 0 if capture["worker_failure"] is None else 1


def _run_state_resolution_scene_subprocess(
        args: argparse.Namespace, *, request_digest: str) -> int:
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--pool", str(args.pool), "--stage", "states",
        "--family", str(args.family), "--backend", str(args.backend),
        "--state-resolution-scene-request-digest", request_digest,
    ]
    completed = subprocess.run(
        command, cwd=ROOT, env={**os.environ, "PYTHONUNBUFFERED": "1"},
        check=False)
    return int(completed.returncode)


def _get_or_run_state_resolution_scene_capture(
        *, args: argparse.Namespace, request: dict[str, Any], out: Path,
        runner: Any = None) -> dict[str, Any]:
    request_digest = str(request["state_resolution_scene_request_digest"])
    raw_path = _state_resolution_capture_path(
        out, str(args.family), request_digest)
    path = _pin_generated_path(raw_path, raw_path)
    capture = _load_valid_state_resolution_scene_capture(
        path=raw_path, request=request)
    if capture is None:
        if path.exists():
            _preserve_invalid(path, out, "state-resolution-scene-capture-invalid")
        run = _run_state_resolution_scene_subprocess if runner is None else runner
        return_code = int(run(args, request_digest=request_digest))
        capture = _load_valid_state_resolution_scene_capture(
            path=raw_path, request=request)
        if capture is None:
            raise RuntimeError(
                f"state-resolution worker {request['scene']['scene_id']} exited "
                f"{return_code} without a valid durable capture"
            )
        if capture.get("worker_failure") is None and return_code != 0:
            print(
                f"[recovery] state-resolution worker "
                f"{request['scene']['scene_id']} exited {return_code} after "
                "its valid atomic capture; retaining capture",
                flush=True)
    if capture.get("worker_failure") is not None:
        raise RuntimeError(
            f"state-resolution worker failed for {request['scene']['scene_id']}: "
            f"{capture['worker_failure']}"
        )
    return capture


def stage_state_resolution_scene_worker(args: argparse.Namespace) -> int:
    """Internal one-scene worker; no final state shard is issued here."""

    if args.family is None or args.backend != "cpu":
        raise RuntimeError("state-resolution scene worker requires family/cpu")
    request_digest = str(args.state_resolution_scene_request_digest)
    out = OUT_ROOT / args.pool
    raw_request_path = _state_resolution_request_path(
        out, str(args.family), request_digest)
    request_path = _pin_generated_path(raw_request_path, raw_request_path)
    if not request_path.is_file() or request_path.is_symlink():
        raise RuntimeError("state-resolution scene request is missing")
    request = json.loads(request_path.read_text())
    if request.get("state_resolution_scene_request_digest") != request_digest:
        raise RuntimeError("state-resolution worker request digest changed")
    _load_clean_source_launch_receipt()
    pool, exclusion = scene_pool(args.pool)
    capture = _execute_state_resolution_scene_worker(
        args=args, request=request, out=out, pool=pool, exclusion=exclusion)
    print(json.dumps({
        "scene_id": capture["scene_id"],
        "state_resolution_scene_capture_digest":
            capture["state_resolution_scene_capture_digest"],
        "chosen_state_id": (None if capture["chosen_state"] is None
                            else capture["chosen_state"]["state_id"]),
        "pass": capture["worker_failure"] is None,
    }, indent=2, sort_keys=True))
    return 0 if capture["worker_failure"] is None else 1


def _live_small_completion_search_inputs(
        *, args: argparse.Namespace, out: Path,
        ) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]],
                   dict[str, dict[str, Any]]]:
    """Reconstruct the terminal small-family search without simulator work.

    The ordinary resolver writes one atomic capture per scene.  A terminal
    joint-search receipt is trusted only after those captures are replayed to
    the first point at which the five general and five safety quotas became
    full, the cursor-restricted completion pool is reopened from the live
    feasibility receipt, and the other seven final state shards are reopened.
    """

    if (args.pool != "scorer_fit"
            or args.family != REACHABILITY_REDRIVE_FAMILY
            or args.backend != "cpu"):
        raise RuntimeError(
            "terminal small completion replay requires scorer-fit/small/cpu")
    pool, exclusion = scene_pool("scorer_fit")
    scenes = pool[REACHABILITY_REDRIVE_FAMILY]
    need = {
        "general": 5,
        "safety_enriched": 5,
        "completion_enriched": 0,
    }
    found = {key: 0 for key in need}
    small_fixed: list[dict[str, Any]] = []
    cursor: str | None = None
    raw_request_root = out / STATE_RESOLUTION_SCENE_REQUEST_ROOT / \
        REACHABILITY_REDRIVE_FAMILY
    request_root = _pin_generated_path(
        raw_request_root, raw_request_root)
    raw_capture_root = out / STATE_RESOLUTION_SCENE_CAPTURE_ROOT / \
        REACHABILITY_REDRIVE_FAMILY
    capture_root = _pin_generated_path(
        raw_capture_root, raw_capture_root)
    if not request_root.is_dir() or request_root.is_symlink():
        raise RuntimeError(
            "terminal small completion replay lacks resolver requests")
    requests: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(request_root.iterdir(), key=lambda value: value.name):
        if path.suffix != ".json":
            continue
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(
                "terminal small completion resolver request is not regular")
        try:
            request = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "terminal small completion resolver request is invalid JSON"
            ) from exc
        requests.append((path, request))

    family_allow_list = [path.name for path in scenes]
    for scene_ordinal, scene_dir in enumerate(scenes):
        if all(found[key] >= need[key] for key in need):
            break
        requested = [
            name for name in STRATA
            if found.get(name, 0) < need.get(name, 0)
        ]
        matches = [
            (request_path, request) for request_path, request in requests
            if (request.get("scene_ordinal") == scene_ordinal
                and request.get("found_before_scene") == found
                and request.get("required_counts") == need
                and request.get("requested_strata_in_priority_order")
                == requested)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "terminal small completion replay lacks one exact resolver "
                f"request for scene {scene_dir.name}")
        request_path, request = matches[0]
        _validate_state_resolution_scene_request(
            request, args=args, out=out, pool=pool, exclusion=exclusion)
        request_digest = str(request["state_resolution_scene_request_digest"])
        expected_request_path = request_root / f"{request_digest}.json"
        if (request_path != expected_request_path
                or not request_path.is_file() or request_path.is_symlink()
                or json.loads(request_path.read_text()) != request):
            raise RuntimeError(
                "terminal small completion resolver request bytes changed")
        capture_path = capture_root / f"{request_digest}.json"
        if not capture_path.is_file() or capture_path.is_symlink():
            raise RuntimeError(
                "terminal small completion replay lacks a passing capture for "
                f"scene {scene_dir.name}")
        try:
            capture = json.loads(capture_path.read_text())
            _validate_state_resolution_scene_capture(
                capture, expected_request=request)
        except (OSError, ValueError, TypeError, RuntimeError,
                json.JSONDecodeError) as exc:
            raise RuntimeError(
                "terminal small completion replay lacks a passing capture for "
                f"scene {scene_dir.name}") from exc
        if capture.get("worker_failure") is not None:
            raise RuntimeError(
                "terminal small completion replay lacks a passing capture for "
                f"scene {scene_dir.name}")
        cursor = str(scene_dir.name)
        chosen = capture.get("chosen_state")
        if chosen is not None:
            stratum = str(chosen["stratum"])
            if stratum not in found or found[stratum] >= need[stratum]:
                raise RuntimeError(
                    "terminal small completion capture violates live quota")
            found[stratum] += 1
            small_fixed.append(dict(chosen))
    if cursor is None or found != need or len(small_fixed) != 10:
        raise RuntimeError(
            "terminal small completion replay did not recover exact G/S quotas")

    fixed_states: list[dict[str, Any]] = []
    for family in STATE_SELECTOR.REQUIRED_FAMILIES:
        if family == REACHABILITY_REDRIVE_FAMILY:
            continue
        _shard_path, shard = _load_active_family_state_shard(
            out, family, pool="scorer_fit")
        fixed_states.extend(dict(state) for state in shard["states"])
    fixed_states.extend(small_fixed)
    if len(fixed_states) != 115:
        raise RuntimeError(
            "terminal small completion replay did not recover 115 fixed states")
    raw_candidates = _small_completion_candidates_from_feasibility(
        out=out,
        excluded_scene_ids={str(state["scene_id"]) for state in fixed_states},
        resolver_cursor_scene_id=cursor,
    )
    return (
        cursor,
        fixed_states,
        raw_candidates,
        _phase1_completion_rotation_vectors(),
    )


def _artifact_binding(path: Path, *, self_key: str | None = None,
                      logical_root: Path = ROOT) -> dict[str, Any]:
    """Bind one regular generated JSON artifact without following aliases."""

    pinned = _pin_generated_path(path, path)
    if not pinned.is_file() or pinned.is_symlink():
        raise RuntimeError(f"bound artifact is not a regular file: {path}")
    payload = json.loads(pinned.read_text())
    try:
        logical = str(path.relative_to(logical_root))
    except ValueError:
        logical = str(path)
    binding = {
        "path": logical,
        "raw_sha256": file_sha256(pinned),
        "byte_count": pinned.stat().st_size,
    }
    if self_key is not None:
        expected = PARALLEL_SEARCH.canonical_digest({
            name: value for name, value in payload.items()
            if name != self_key})
        if payload.get(self_key) != expected:
            raise RuntimeError(f"{path.name} parallel self digest mismatch")
        binding["self_digest_key"] = self_key
        binding["self_digest"] = payload[self_key]
    return binding


def _write_or_require_exact_json(path: Path, payload: Mapping[str, Any],
                                 *, label: str) -> dict[str, Any]:
    """Atomically issue once, or reopen exactly; never rewrite evidence."""

    pinned = _pin_generated_path(path, path)
    expected = dict(payload)
    if pinned.exists():
        if (not pinned.is_file() or pinned.is_symlink()
                or json.loads(pinned.read_text()) != expected):
            raise RuntimeError(f"{label} differs from its frozen bytes")
        return expected
    # ``atomic_json`` historically created the parent implicitly.  Preserve
    # that behavior only after the lexical path has been pinned to the managed
    # physical root; the exclusive install below must never traverse an
    # unvalidated alias while creating directories.
    pinned.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(V1._jsonable(expected), indent=2, sort_keys=True)
               + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{pinned.name}.tmp-", dir=str(pinned.parent))
    temporary = Path(temporary_name)
    installed = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, pinned, follow_symlinks=False)
            installed = True
        except FileExistsError:
            pass
        if installed:
            directory_fd = os.open(
                pinned.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    if (not pinned.is_file() or pinned.is_symlink()
            or json.loads(pinned.read_text()) != expected):
        raise RuntimeError(f"{label} atomic reopen changed")
    return expected


def _load_reissued_small_prefix_inputs(out: Path) -> dict[str, Any]:
    """Reopen the active normal-schema 5G/5S prefix and its receipt."""

    performance = _load_current_performance_interruption_receipt()
    bindings = _performance_successor_bindings()
    receipt = (
        PERFORMANCE_INTERRUPTION
        .load_and_validate_small_prefix_reissue_receipt(
            performance_receipt=performance,
            successor_bindings=bindings,
            revalidate_prefix=_revalidate_reissued_small_prefix,
            root=ROOT,
        )
    )
    receipt_binding = (
        PERFORMANCE_INTERRUPTION.small_prefix_reissue_receipt_binding(
            receipt, root=ROOT)
    )
    pairs: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    rejections: dict[str, dict[str, int]] = {}
    for mapping in receipt["mapping_rows"]:
        loaded: dict[str, Any] = {
            "scene_id": str(mapping["scene_id"]),
            "scene_ordinal": int(mapping["scene_ordinal"]),
        }
        for kind in ("request", "capture"):
            row = dict(mapping[f"successor_{kind}"])
            raw_path = ROOT / str(row["path"])
            path = _pin_generated_path(raw_path, raw_path)
            if (not path.is_file() or path.is_symlink()
                    or file_sha256(path) != row["raw_sha256"]
                    or path.stat().st_size != row["byte_count"]):
                raise RuntimeError("reissued small prefix transport changed")
            payload = json.loads(path.read_text())
            loaded[kind] = payload
        request = loaded["request"]
        capture = loaded["capture"]
        digest = str(request["state_resolution_scene_request_digest"])
        if (Path(str(mapping["successor_request"]["path"]))
                != Path(PERFORMANCE_INTERRUPTION.SMALL_PREFIX_ROOTS[
                    "request"]) / f"{digest}.json"
                or Path(str(mapping["successor_capture"]["path"]))
                != Path(PERFORMANCE_INTERRUPTION.SMALL_PREFIX_ROOTS[
                    "capture"]) / f"{digest}.json"):
            raise RuntimeError("reissued small prefix logical paths changed")
        provenance.append({
            "scene_id": str(mapping["scene_id"]),
            "state_resolution_scene_request_digest": digest,
            "state_resolution_scene_capture_digest":
                capture["state_resolution_scene_capture_digest"],
            "request_path": str(mapping["successor_request"]["path"]),
            "request_raw_sha256": mapping["successor_request"]["raw_sha256"],
            "request_byte_count": mapping["successor_request"]["byte_count"],
            "capture_path": str(mapping["successor_capture"]["path"]),
            "capture_raw_sha256": mapping["successor_capture"]["raw_sha256"],
            "capture_byte_count": mapping["successor_capture"]["byte_count"],
        })
        rejections[str(mapping["scene_id"])] = dict(
            capture["scene_rejection_reasons"])
        pairs.append(loaded)
    replay = _replay_small_fixed_prefix_pairs(
        pairs, successor_bindings=bindings)
    if (replay["resolver_cursor_scene_id"]
            != receipt["resolver_cursor_scene_id"]
            or len(replay["states"]) != 10):
        raise RuntimeError("reissued small prefix projection changed")
    return {
        "states": [dict(state) for state in replay["states"]],
        "resolver_cursor_scene_id": replay["resolver_cursor_scene_id"],
        "scene_rejection_reasons": rejections,
        "capture_provenance": provenance,
        "receipt": receipt,
        "receipt_binding": receipt_binding,
        "performance_receipt_binding":
            PERFORMANCE_INTERRUPTION.receipt_binding(performance, root=ROOT),
    }


def _common_state_shard_bindings(
        shards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not shards:
        raise RuntimeError("state-shard common binding input is empty")
    common = {key: shards[0][key] for key in STATE_SHARD_COMMON_KEYS}
    for shard in shards[1:]:
        if any(shard.get(key) != common[key]
               for key in STATE_SHARD_COMMON_KEYS):
            raise RuntimeError("fixed state shards contain mixed bindings")
    return common


def _joint_state_order(states: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted((dict(state) for state in states), key=lambda state: (
        str(state["family"]),
        STRATA.index(str(state["stratum"])),
        str(state["scene_id"]),
    ))


def _pre_allocation_identity_payload(
        *, states: Sequence[Mapping[str, Any]],
        common: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "go2_branch_corpus_v1_2_pre_allocation_identity_manifest",
        "pool": "scorer_fit",
        "spec": POOLS["scorer_fit"],
        **{key: common[key] for key in STATE_SHARD_COMMON_KEYS},
        "state_identities": _allocation_projection(
            _joint_state_order(states)),
    }


def _parallel_small_search_inputs(out: Path) -> dict[str, Any]:
    """Construct the exact 115-state fixed input and cursor-only candidate pool."""

    if out != OUT_ROOT / "scorer_fit":
        raise RuntimeError("parallel small search is scorer-fit only")
    prefix = _load_reissued_small_prefix_inputs(out)
    fixed_shards: list[dict[str, Any]] = []
    fixed_evidence: list[dict[str, Any]] = []
    for family in STATE_SELECTOR.REQUIRED_FAMILIES:
        if family == REACHABILITY_REDRIVE_FAMILY:
            continue
        _path, shard, evidence = _load_active_state_shard_evidence(
            out, family, pool="scorer_fit")
        fixed_shards.append(shard)
        fixed_evidence.append(evidence)
    common = _common_state_shard_bindings(fixed_shards)
    fixed_states = [dict(state) for shard in fixed_shards
                    for state in shard["states"]]
    fixed_states.extend(dict(state) for state in prefix["states"])
    if (len(fixed_states) != 115
            or len({state["scene_id"] for state in fixed_states}) != 115
            or len({state["state_identity_digest"] for state in fixed_states})
            != 115):
        raise RuntimeError("parallel search fixed 115-state identity set changed")
    cursor = str(prefix["resolver_cursor_scene_id"])
    candidates = _small_completion_candidates_from_feasibility(
        out=out,
        excluded_scene_ids={str(state["scene_id"]) for state in fixed_states},
        resolver_cursor_scene_id=cursor,
    )
    candidates = sorted(candidates, key=lambda state: (
        str(state["scene_id"]), int(state["warmup_blocks"])))
    candidate_scene_ids = [str(state["scene_id"]) for state in candidates]
    if (len(candidates) < 5 or candidate_scene_ids != sorted(candidate_scene_ids)
            or len(set(candidate_scene_ids)) != len(candidate_scene_ids)
            or any(scene_id <= cursor for scene_id in candidate_scene_ids)):
        raise RuntimeError("parallel search cursor-restricted candidate pool changed")
    return {
        "fixed_states": fixed_states,
        "raw_candidates": candidates,
        "candidate_scene_ids": candidate_scene_ids,
        "resolver_cursor_scene_id": cursor,
        "preserved_vectors": _phase1_completion_rotation_vectors(),
        "common": common,
        "prefix": prefix,
        "fixed_shard_evidence": fixed_evidence,
    }


def _parallel_selected_completion_states(
        raw_candidates: Sequence[Mapping[str, Any]],
        combination_indices: Sequence[int]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    combination = [dict(raw_candidates[index]) for index in combination_indices]
    for ordinal, raw in enumerate(sorted(
            combination, key=lambda row: str(row["scene_id"]))):
        state = dict(raw)
        state["state_id"] = (
            f"scorer_fit-{REACHABILITY_REDRIVE_FAMILY}-"
            f"completion_enriched-{ordinal:02d}")
        state["split_role"] = "calibration" if ordinal == 0 else "fit"
        state["state_identity_digest"] = _state_identity_digest(state)
        selected.append(state)
    return selected


def _parallel_rank_material(
        inputs: Mapping[str, Any], rank: int,
        combination_indices: Sequence[int]) -> dict[str, Any]:
    expected = PARALLEL_SEARCH.unrank_combination(
        rank, len(inputs["raw_candidates"]), 5)
    if tuple(combination_indices) != expected:
        raise RuntimeError("parallel rank/combination identity changed")
    selected = _parallel_selected_completion_states(
        inputs["raw_candidates"], combination_indices)
    states = _joint_state_order([*inputs["fixed_states"], *selected])
    if len(states) != 120:
        raise RuntimeError("parallel rank does not contain 120 states")
    source_payload = _pre_allocation_identity_payload(
        states=states, common=inputs["common"])
    return {
        "states": states,
        "source_identity_manifest_digest": canonical_digest(source_payload),
        "mask_context": {
            "preserved_vectors": inputs["preserved_vectors"],
        },
    }


def _parallel_mask_classifier(
        states: Sequence[Mapping[str, Any]], allocation: Mapping[str, Any],
        context: Mapping[str, Any]) -> bool:
    vectors = context.get("preserved_vectors")
    if not isinstance(vectors, dict):
        raise RuntimeError("parallel mask context changed")
    return _all_completion_masks_pass(
        states=[dict(state) for state in states], allocation=dict(allocation),
        preserved_vectors=vectors)


def _parallel_winner_validator_zero_solve(
        _rank: int, states: Sequence[Mapping[str, Any]],
        allocation: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
    """Rebuild and check exact allocation bytes without invoking a MILP."""

    manifest = dict(allocation)
    rotations = [int(row["rotation_index"])
                 for row in manifest.get("assignments", [])]
    expected = PARALLEL_SEARCH.materialize_allocation_manifest_single_solve(
        states,
        source_identity_manifest_digest=str(
            manifest.get("source_identity_manifest_digest", "")),
        rotations=rotations,
    )
    if manifest != expected:
        raise RuntimeError("parallel winner allocation bytes changed")
    if _allocation_assignment_set_digest(manifest) != \
            PARALLEL_SEARCH._candidate_assignment_set_digest(manifest):
        raise RuntimeError("parallel winner assignment digest implementation diverged")
    return _parallel_mask_classifier(states, manifest, context)


def _parallel_plan_bindings(inputs: Mapping[str, Any]) -> dict[str, Any]:
    launch = _load_clean_source_launch_receipt()
    return {
        "small_prefix_reissue_receipt":
            dict(inputs["prefix"]["receipt_binding"]),
        "performance_interruption_receipt":
            dict(inputs["prefix"]["performance_receipt_binding"]),
        "fixed_state_active_envelope_set_digest": canonical_digest(
            inputs["fixed_shard_evidence"]),
        "fixed_state_identity_projection_digest": canonical_digest(
            _allocation_projection(inputs["fixed_states"])),
        "state_selector_feasibility_receipt_digest":
            launch["state_selector_feasibility_receipt_digest"],
        "pre_allocation_identity_payload_schema":
            "go2_branch_corpus_v1_2_pre_allocation_identity_manifest",
        "candidate_outcomes_consumed": False,
    }


def _build_parallel_search_plan(
        inputs: Mapping[str, Any], *,
        measured_benchmark_receipt_digest: str | None) -> dict[str, Any]:
    launch = _load_clean_source_launch_receipt()
    return PARALLEL_SEARCH.build_search_plan(
        candidate_scene_ids=inputs["candidate_scene_ids"],
        combination_size=5,
        worker_count=PARALLEL_SMALL_WORKER_COUNT,
        active_rank_window=PARALLEL_SMALL_ACTIVE_RANK_WINDOW,
        source_repository_commit=str(launch["source_repository_commit"]),
        clean_source_launch_receipt_digest=str(
            launch["clean_source_launch_receipt_digest"]),
        state_selector_amendment_digest=
            STATE_SELECTOR.state_selector_amendment_digest(),
        candidate_allocation_amendment_digest=
            ALLOC.allocation_amendment_digest(),
        fixed_state_projection_digest=canonical_digest(
            _allocation_projection(inputs["fixed_states"])),
        resolver_cursor_scene_id=str(inputs["resolver_cursor_scene_id"]),
        solver_identity={
            "implementation": "scipy.optimize.milp/HiGHS",
            "candidate_allocation_algorithm_version": ALLOC.ALGORITHM_VERSION,
            "parallel_search_algorithm_version":
                PARALLEL_SEARCH.ALGORITHM_VERSION,
        },
        solver_options={
            "disp": False,
            "presolve": True,
            "mip_rel_gap": 0.0,
            "threads": 1,
            "time_limit": PARALLEL_SEARCH.DEFAULT_MILP_TIME_LIMIT_S,
        },
        bindings=_parallel_plan_bindings(inputs),
        measured_benchmark_receipt_digest=
            measured_benchmark_receipt_digest,
    )


def _parallel_benchmark_source_binding(
        inputs: Mapping[str, Any], provisional_plan: Mapping[str, Any]) -> str:
    rank_zero = _parallel_rank_material(
        inputs, 0, PARALLEL_SEARCH.unrank_combination(
            0, len(inputs["raw_candidates"]), 5))
    return PARALLEL_SEARCH.canonical_digest({
        "schema": "go2_branch_corpus_v1_2_parallel_small_benchmark_binding_v1",
        "provisional_search_plan_digest": provisional_plan["search_plan_digest"],
        "rank_zero_source_identity_manifest_digest":
            rank_zero["source_identity_manifest_digest"],
        "rank_zero_projection_digest": PARALLEL_SEARCH.canonical_digest(
            ALLOC._normalise_identity_states(
                _allocation_projection(rank_zero["states"]))),
        "small_prefix_reissue_receipt_digest":
            inputs["prefix"]["receipt_binding"]["receipt_digest"],
        "candidate_outcomes_consumed": False,
    })


def _load_parallel_plan_and_benchmark(
        out: Path, inputs: Mapping[str, Any],
        ) -> tuple[dict[str, Any], dict[str, Any], str]:
    raw_benchmark_path = out / PARALLEL_SMALL_BENCHMARK_NAME
    benchmark_path = _pin_generated_path(
        raw_benchmark_path, raw_benchmark_path)
    if not benchmark_path.is_file() or benchmark_path.is_symlink():
        raise RuntimeError("measured parallel prefix benchmark is missing")
    benchmark = json.loads(benchmark_path.read_text())
    provisional = _build_parallel_search_plan(
        inputs, measured_benchmark_receipt_digest=None)
    source_binding = _parallel_benchmark_source_binding(inputs, provisional)
    PARALLEL_SEARCH.require_measured_benchmark_gate(
        benchmark, expected_source_binding_digest=source_binding,
        maximum_parallel_fraction=PARALLEL_SMALL_BENCHMARK_MAXIMUM_FRACTION)
    expected_plan = _build_parallel_search_plan(
        inputs,
        measured_benchmark_receipt_digest=benchmark[
            "benchmark_receipt_digest"])
    raw_plan_path = out / PARALLEL_SMALL_SEARCH_PLAN_NAME
    plan_path = _pin_generated_path(raw_plan_path, raw_plan_path)
    if not plan_path.is_file() or plan_path.is_symlink():
        raise RuntimeError("parallel small search plan is missing")
    plan = PARALLEL_SEARCH.validate_search_plan(
        json.loads(plan_path.read_text()))
    if plan != expected_plan:
        raise RuntimeError("parallel small search plan differs from live inputs")
    return plan, benchmark, source_binding


def stage_parallel_small_completion_benchmark() -> int:
    """Measure the exact three prefix waves and freeze the gated search plan."""

    out = OUT_ROOT / "scorer_fit"
    inputs = _parallel_small_search_inputs(out)
    provisional = _build_parallel_search_plan(
        inputs, measured_benchmark_receipt_digest=None)
    source_binding = _parallel_benchmark_source_binding(inputs, provisional)
    raw_benchmark_path = out / PARALLEL_SMALL_BENCHMARK_NAME
    benchmark_path = _pin_generated_path(
        raw_benchmark_path, raw_benchmark_path)
    if benchmark_path.exists() and (
            not benchmark_path.is_file() or benchmark_path.is_symlink()):
        raise RuntimeError("parallel benchmark path is not a regular file")
    if benchmark_path.is_file():
        benchmark = json.loads(benchmark_path.read_text())
    else:
        rank_zero = _parallel_rank_material(
            inputs, 0, PARALLEL_SEARCH.unrank_combination(
                0, len(inputs["raw_candidates"]), 5))
        benchmark = PARALLEL_SEARCH.run_measured_fixed_rotation_benchmark(
            states=rank_zero["states"], search_plan=provisional,
            source_binding_digest=source_binding,
            sample_prefix_indices=(0, 1, 2),
            maximum_parallel_fraction=
                PARALLEL_SMALL_BENCHMARK_MAXIMUM_FRACTION,
        )
        _write_or_require_exact_json(
            raw_benchmark_path, benchmark,
            label="parallel prefix benchmark")
    PARALLEL_SEARCH.require_measured_benchmark_gate(
        benchmark, expected_source_binding_digest=source_binding,
        maximum_parallel_fraction=PARALLEL_SMALL_BENCHMARK_MAXIMUM_FRACTION)
    plan = _build_parallel_search_plan(
        inputs,
        measured_benchmark_receipt_digest=benchmark[
            "benchmark_receipt_digest"])
    _write_or_require_exact_json(
        out / PARALLEL_SMALL_SEARCH_PLAN_NAME, plan,
        label="parallel small search plan")
    print(json.dumps({
        "status": "PASS_MEASURED_PARALLEL_PREFIX_BENCHMARK",
        "benchmark_receipt_digest": benchmark["benchmark_receipt_digest"],
        "search_plan_digest": plan["search_plan_digest"],
        "candidate_pool_count": plan["candidate_pool_count"],
        "total_rank_count": plan["total_rank_count"],
        "worker_count": plan["worker_count"],
        "active_rank_window": plan["active_rank_window"],
    }, indent=2, sort_keys=True))
    return 0


def _parallel_search_checkpoint_root(out: Path) -> Path:
    raw = out / PARALLEL_SMALL_CHECKPOINT_ROOT
    return _pin_generated_path(raw, raw)


def _parallel_search_callbacks(inputs: Mapping[str, Any]) -> tuple[
        Callable[[int, tuple[int, ...]], Mapping[str, Any]],
        Callable[[Sequence[Mapping[str, Any]], Mapping[str, Any],
                  Mapping[str, Any]], bool],
        Callable[[int, Sequence[Mapping[str, Any]], Mapping[str, Any],
                  Mapping[str, Any]], bool]]:
    def prepare(rank: int, combination: tuple[int, ...]) -> Mapping[str, Any]:
        return _parallel_rank_material(inputs, rank, combination)

    return prepare, _parallel_mask_classifier, \
        _parallel_winner_validator_zero_solve


def _parallel_certificate_binding(
        raw_path: Path, payload: Mapping[str, Any], self_key: str, *,
        pinned_path: Path | None = None) -> dict[str, Any]:
    """Bind a certificate while keeping lexical and pinned paths distinct.

    Production ``OUT_ROOT`` is one permitted managed symlink.  ``raw_path``
    therefore remains the lexical custody identity, while ``pinned_path`` may
    be the already-resolved path used by the executor.  Never feed that
    resolved path back through ``_pin_generated_path`` as though it were a
    second lexical artifact path.
    """

    expected_pinned = _pin_generated_path(raw_path, raw_path)
    pinned = expected_pinned if pinned_path is None else Path(pinned_path)
    if pinned != expected_pinned:
        raise RuntimeError("parallel certificate pinned path changed")
    if (not pinned.is_file() or pinned.is_symlink()
            or json.loads(pinned.read_text()) != dict(payload)):
        raise RuntimeError("parallel certificate bytes changed")
    expected = PARALLEL_SEARCH.canonical_digest({
        key: value for key, value in payload.items() if key != self_key})
    if payload.get(self_key) != expected:
        raise RuntimeError("parallel certificate self digest changed")
    try:
        logical = str(raw_path.relative_to(ROOT))
    except ValueError:
        logical = str(raw_path)
    return {
        "path": logical,
        "raw_sha256": file_sha256(pinned),
        "byte_count": pinned.stat().st_size,
        "self_digest_key": self_key,
        "self_digest": payload[self_key],
    }


def _parallel_rank_prefix_bindings(
        *, plan: Mapping[str, Any], out: Path,
        terminal_rank: int) -> list[dict[str, Any]]:
    raw_checkpoint_root = out / PARALLEL_SMALL_CHECKPOINT_ROOT
    checkpoint_root = _parallel_search_checkpoint_root(out)
    rows: list[dict[str, Any]] = []
    for rank in range(terminal_rank + 1):
        raw_path = raw_checkpoint_root / "ranks" / f"{rank:012d}.json"
        path = checkpoint_root / "ranks" / f"{rank:012d}.json"
        receipt = PARALLEL_SEARCH.load_rank_receipt(
            path, search_plan=plan, expected_rank=rank)
        rows.append({
            "rank": rank,
            "classification": receipt["classification"],
            **_parallel_certificate_binding(
                raw_path, receipt, "rank_receipt_digest", pinned_path=path),
        })
    return rows


def _build_parallel_joint_receipt(
        *, out: Path, inputs: Mapping[str, Any], plan: Mapping[str, Any],
        benchmark: Mapping[str, Any], terminal: Mapping[str, Any],
        allocation: Mapping[str, Any]) -> dict[str, Any]:
    if terminal.get("status") != "PASS":
        raise RuntimeError("joint receipt requires a terminal PASS")
    rank = int(terminal["rank"])
    raw_checkpoint_root = out / PARALLEL_SMALL_CHECKPOINT_ROOT
    checkpoint_root = _parallel_search_checkpoint_root(out)
    rank_rows = _parallel_rank_prefix_bindings(
        plan=plan, out=out, terminal_rank=rank)
    raw_winner_path = raw_checkpoint_root / "winner-objective-validation.json"
    winner_path = checkpoint_root / "winner-objective-validation.json"
    winner = dict(terminal["winner_validation_receipt"])
    receipt = {
        "schema": PARALLEL_SMALL_JOINT_RECEIPT_SCHEMA,
        "status": "PASS_FIRST_LEXICOGRAPHIC_EXACT_MASK_COMBINATION",
        "complete": True,
        "search_plan": _artifact_binding(
            out / PARALLEL_SMALL_SEARCH_PLAN_NAME,
            self_key="search_plan_digest"),
        "measured_benchmark": _artifact_binding(
            out / PARALLEL_SMALL_BENCHMARK_NAME,
            self_key="benchmark_receipt_digest"),
        "small_prefix_reissue_receipt":
            dict(inputs["prefix"]["receipt_binding"]),
        "performance_interruption_receipt":
            dict(inputs["prefix"]["performance_receipt_binding"]),
        "checkpoint_root": str(
            (out / PARALLEL_SMALL_CHECKPOINT_ROOT).relative_to(ROOT)),
        "rank": rank,
        "combination_attempt_count": terminal["combination_attempt_count"],
        "allocator_infeasible_combination_count":
            terminal["allocator_infeasible_combination_count"],
        "rank_prefix_receipts": rank_rows,
        "rank_prefix_receipt_set_digest":
            PARALLEL_SEARCH.canonical_digest(rank_rows),
        "winner_validation_receipt": _parallel_certificate_binding(
            raw_winner_path, winner, "winner_validation_receipt_digest",
            pinned_path=winner_path),
        "resolver_cursor_scene_id": inputs["resolver_cursor_scene_id"],
        "candidate_pool_scene_ids": list(inputs["candidate_scene_ids"]),
        "candidate_pool_scene_ids_digest":
            PARALLEL_SEARCH.canonical_digest(inputs["candidate_scene_ids"]),
        "selected_scene_ids": list(
            terminal["rank_receipt"]["selected_scene_ids"]),
        "provisional_allocation_manifest_digest":
            allocation["allocation_manifest_digest"],
        "provisional_candidate_assignment_set_digest":
            _allocation_assignment_set_digest(dict(allocation)),
        "final_candidate_assignment_set_digest":
            _allocation_assignment_set_digest(dict(allocation)),
        "final_masks_equal_searched_masks": True,
        "terminal_search_result_digest":
            PARALLEL_SEARCH.canonical_digest(terminal),
        "candidate_outcomes_consumed": False,
    }
    receipt["parallel_small_completion_joint_receipt_digest"] = \
        PARALLEL_SEARCH.canonical_digest(receipt)
    return receipt


def _build_parallel_small_terminal_shard(
        *, inputs: Mapping[str, Any], terminal: Mapping[str, Any],
        joint_receipt: Mapping[str, Any]) -> dict[str, Any]:
    selected = _parallel_selected_completion_states(
        inputs["raw_candidates"], terminal["rank_receipt"][
            "combination_indices"])
    states = [*inputs["prefix"]["states"], *selected]
    states.sort(key=lambda state: (
        STRATA.index(str(state["stratum"])), str(state["scene_id"])))
    pool, exclusion = scene_pool("scorer_fit")
    args = argparse.Namespace(
        pool="scorer_fit", family=REACHABILITY_REDRIVE_FAMILY,
        backend="cpu")
    expected_common = _state_shard_bindings(
        args, exclusion,
        [path.name for path in pool[REACHABILITY_REDRIVE_FAMILY]])
    if any(inputs["common"].get(key) != expected_common[key]
           for key in inputs["common"]):
        raise RuntimeError("small terminal shard common bindings changed")
    provenance = list(inputs["prefix"]["capture_provenance"])
    shard = {
        "schema": "go2_branch_corpus_v1_2_state_shard",
        "status": STATUS,
        "complete": True,
        "pool": "scorer_fit",
        "family": REACHABILITY_REDRIVE_FAMILY,
        "spec": POOLS["scorer_fit"],
        "selection": SELECTION,
        **expected_common,
        "states": states,
        "scene_rejection_reasons":
            dict(inputs["prefix"]["scene_rejection_reasons"]),
        "state_resolution_subprocess_transport": {
            "schema": "go2_branch_corpus_v1_2_state_resolution_transport_v1",
            "one_scene_per_subprocess": True,
            "atomic_capture_write_before_native_cleanup": True,
            "return_code_ignored_only_after_valid_capture": True,
            "resume_scope": "REISSUED_EXACT_SMALL_PREFIX_PLUS_PARALLEL_CERTIFICATE",
            "resolver_algorithm_digest":
                canonical_digest(STATE_RESOLUTION_REDUCER_CONTRACT),
            "resolver_cursor_scene_id": inputs["resolver_cursor_scene_id"],
            "scene_capture_count": len(provenance),
            "scene_capture_provenance_digest": canonical_digest(provenance),
            "candidate_outcomes_loaded": False,
        },
        "state_resolution_scene_capture_provenance": provenance,
        "small_prefix_reissue_receipt":
            dict(inputs["prefix"]["receipt_binding"]),
        "small_completion_joint_allocation_search": dict(joint_receipt),
    }
    shard["state_shard_digest"] = canonical_digest(shard)
    return shard


def _parallel_failure_evidence_inventory(
        *, out: Path, plan: Mapping[str, Any],
        prepare_rank: Callable[[int, tuple[int, ...]], Mapping[str, Any]],
        ) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]],
                   list[dict[str, Any]]]:
    """Validate the exact checkpoint namespace before binding fatal evidence."""

    raw_checkpoint_root = out / PARALLEL_SMALL_CHECKPOINT_ROOT
    checkpoint_root = _parallel_search_checkpoint_root(out)
    total = int(plan["total_rank_count"])
    prepared: dict[int, tuple[str, str, list[Mapping[str, Any]]]] = {}
    checkpoint_rows: list[dict[str, Any]] = []
    if checkpoint_root.exists():
        if checkpoint_root.is_symlink() or not checkpoint_root.is_dir():
            raise RuntimeError("parallel checkpoint root is not regular")
        allowed = {
            "ranks": "directory",
            "waves": "directory",
            "winner-objective": "directory",
            "winner-objective-validation.json": "file",
        }
        for entry in sorted(checkpoint_root.iterdir(), key=lambda row: row.name):
            kind = allowed.get(entry.name)
            if kind is None:
                raise RuntimeError("parallel checkpoint entry is noncanonical")
            if entry.is_symlink() or (
                    kind == "directory" and not entry.is_dir()) or (
                    kind == "file" and not entry.is_file()):
                raise RuntimeError("parallel checkpoint entry is not regular")

    def material(rank: int) -> tuple[str, str, list[Mapping[str, Any]]]:
        if rank not in prepared:
            combination = PARALLEL_SEARCH.unrank_combination(
                rank, int(plan["candidate_pool_count"]),
                int(plan["combination_size"]))
            row = dict(prepare_rank(rank, combination))
            if set(row) != {
                    "states", "source_identity_manifest_digest", "mask_context"}:
                raise RuntimeError("failure replay rank preparation changed")
            states = list(row["states"])
            source = row["source_identity_manifest_digest"]
            if (not _is_sha256(source)
                    or not isinstance(row["mask_context"], Mapping)
                    or len(states) != PARALLEL_SEARCH.PREFIX_STATE_COUNT):
                raise RuntimeError("failure replay rank binding changed")
            projection = PARALLEL_SEARCH.canonical_digest(
                PARALLEL_SEARCH.project_allocator_identity_states(states))
            prepared[rank] = (projection, source, states)
        return prepared[rank]

    rank_receipts: dict[int, dict[str, Any]] = {}
    raw_rank_root = raw_checkpoint_root / "ranks"
    rank_root = checkpoint_root / "ranks"
    if rank_root.exists():
        if rank_root.is_symlink() or not rank_root.is_dir():
            raise RuntimeError("parallel rank receipt root is not regular")
        for path in sorted(rank_root.iterdir(), key=lambda row: row.name):
            if path.is_symlink() or not path.is_file():
                raise RuntimeError("parallel rank receipt entry is not regular")
            match = re.fullmatch(r"([0-9]{12})\.json", path.name)
            if match is None:
                raise RuntimeError("parallel rank receipt filename is noncanonical")
            rank = int(match.group(1))
            if rank >= total or rank in rank_receipts:
                raise RuntimeError("parallel rank receipt rank is outside the plan")
            raw_path = raw_rank_root / path.name
            expected_path = _pin_generated_path(raw_path, raw_path)
            if expected_path != path:
                raise RuntimeError("parallel rank receipt path identity changed")
            receipt = PARALLEL_SEARCH.load_rank_receipt(
                path, search_plan=plan, expected_rank=rank)
            projection, source, _states = material(rank)
            if (receipt["projection_digest"] != projection
                    or receipt["source_identity_manifest_digest"] != source):
                raise RuntimeError("parallel rank receipt source binding changed")
            rank_receipts[rank] = receipt
            checkpoint_rows.append({
                "kind": "rank_receipt", "rank": rank,
                **_parallel_certificate_binding(
                    raw_path, receipt, "rank_receipt_digest",
                    pinned_path=path),
            })

    fatal_rows: list[dict[str, Any]] = []
    validated_wave_ranks: set[int] = set()
    raw_wave_root = raw_checkpoint_root / "waves"
    wave_root = checkpoint_root / "waves"
    if wave_root.exists():
        if wave_root.is_symlink() or not wave_root.is_dir():
            raise RuntimeError("parallel wave root is not regular")
        for rank_dir in sorted(wave_root.iterdir(), key=lambda row: row.name):
            if rank_dir.is_symlink() or not rank_dir.is_dir():
                raise RuntimeError("parallel wave rank directory is not regular")
            rank_match = re.fullmatch(r"rank-([0-9]{12})", rank_dir.name)
            if rank_match is None:
                raise RuntimeError("parallel wave rank directory is noncanonical")
            rank = int(rank_match.group(1))
            if rank >= total:
                raise RuntimeError("parallel wave rank is outside the plan")
            validated_wave_ranks.add(rank)
            projection, _source, _states = material(rank)
            indexed: dict[int, Path] = {}
            for path in sorted(rank_dir.iterdir(), key=lambda row: row.name):
                if path.is_symlink() or not path.is_file():
                    raise RuntimeError("parallel wave entry is not regular")
                match = re.fullmatch(r"prefix-([0-9]{3})\.json", path.name)
                if match is None:
                    raise RuntimeError("parallel wave filename is noncanonical")
                state_index = int(match.group(1))
                if (state_index >= PARALLEL_SEARCH.PREFIX_STATE_COUNT
                        or state_index in indexed):
                    raise RuntimeError("parallel wave index is outside the plan")
                indexed[state_index] = path
            if sorted(indexed) != list(range(len(indexed))):
                raise RuntimeError("parallel wave prefix is not contiguous")
            prefix: list[int] = []
            terminal_wave = False
            last_wave_status: str | None = None
            for state_index in range(len(indexed)):
                path = indexed[state_index]
                raw_path = (raw_wave_root / rank_dir.name / path.name)
                expected_path = _pin_generated_path(raw_path, raw_path)
                if expected_path != path:
                    raise RuntimeError("parallel wave path identity changed")
                payload = json.loads(path.read_text())
                receipt = PARALLEL_SEARCH._validate_wave_receipt(
                    payload, plan=plan, rank=rank,
                    projection_digest=projection, expected_prefix=prefix)
                binding = _parallel_certificate_binding(
                    raw_path, receipt, "wave_receipt_digest",
                    pinned_path=path)
                checkpoint_rows.append({
                    "kind": "prefix_wave_receipt", "rank": rank,
                    "state_index": state_index,
                    "wave_status": receipt["wave_status"],
                    **binding,
                })
                last_wave_status = str(receipt["wave_status"])
                if terminal_wave:
                    raise RuntimeError("parallel wave continues after terminal status")
                if receipt["wave_status"] == "SELECTED":
                    prefix.append(int(receipt["selected_rotation"]))
                else:
                    terminal_wave = True
                if receipt["wave_status"] == "FATAL":
                    fatal_rows.append(dict(binding))
            existing_rank = rank_receipts.get(rank)
            if existing_rank is not None:
                classification = existing_rank["classification"]
                if classification == "ALLOCATOR_INFEASIBLE":
                    if (len(indexed) != 1 or not terminal_wave
                            or prefix != []
                            or last_wave_status != "ALLOCATOR_INFEASIBLE"):
                        raise RuntimeError(
                            "allocator-infeasible rank wave evidence changed")
                elif classification in {"PASS", "MASK_FAIL", "NONPASS"}:
                    if (terminal_wave
                            or len(indexed)
                            != PARALLEL_SEARCH.PREFIX_STATE_COUNT
                            or prefix != existing_rank["selected_rotations"]):
                        raise RuntimeError("completed rank wave evidence changed")

    for rank, receipt in rank_receipts.items():
        if receipt["classification"] not in {
                "ALLOCATOR_INFEASIBLE", "MASK_FAIL", "PASS"}:
            raise RuntimeError(
                "persisted scientific rank classification is unreachable")
        if rank not in validated_wave_ranks:
            raise RuntimeError(
                "persisted rank receipt lacks complete wave evidence")

    objective_receipts: dict[int, list[dict[str, Any]]] = {}
    raw_objective_root = raw_checkpoint_root / "winner-objective"
    objective_root = checkpoint_root / "winner-objective"
    if objective_root.exists():
        if objective_root.is_symlink() or not objective_root.is_dir():
            raise RuntimeError("parallel objective root is not regular")
        for rank_dir in sorted(objective_root.iterdir(), key=lambda row: row.name):
            if rank_dir.is_symlink() or not rank_dir.is_dir():
                raise RuntimeError(
                    "parallel objective rank directory is not regular")
            rank_match = re.fullmatch(r"rank-([0-9]{12})", rank_dir.name)
            if rank_match is None:
                raise RuntimeError(
                    "parallel objective rank directory is noncanonical")
            rank = int(rank_match.group(1))
            rank_receipt = rank_receipts.get(rank)
            if (rank >= total or rank_receipt is None
                    or rank_receipt["classification"] != "PASS"):
                raise RuntimeError("parallel objective rank lacks PASS evidence")
            projection, source, _states = material(rank)
            rotations = list(rank_receipt["selected_rotations"])
            indexed: dict[int, Path] = {}
            for path in sorted(rank_dir.iterdir(), key=lambda row: row.name):
                if path.is_symlink() or not path.is_file():
                    raise RuntimeError("parallel objective entry is not regular")
                match = re.fullmatch(r"prefix-([0-9]{3})\.json", path.name)
                if match is None:
                    raise RuntimeError(
                        "parallel objective filename is noncanonical")
                state_index = int(match.group(1))
                if (state_index >= PARALLEL_SEARCH.PREFIX_STATE_COUNT
                        or state_index in indexed):
                    raise RuntimeError(
                        "parallel objective index is outside the plan")
                indexed[state_index] = path
            if sorted(indexed) != list(range(len(indexed))):
                raise RuntimeError("parallel objective prefix is not contiguous")
            prefix: list[int] = []
            terminal_wave = False
            for state_index in range(len(indexed)):
                path = indexed[state_index]
                raw_path = raw_objective_root / rank_dir.name / path.name
                expected_path = _pin_generated_path(raw_path, raw_path)
                if expected_path != path:
                    raise RuntimeError("parallel objective path identity changed")
                receipt = PARALLEL_SEARCH._validate_objective_wave_receipt(
                    json.loads(path.read_text()), plan=plan, rank=rank,
                    projection_digest=projection, source_digest=source,
                    expected_prefix=prefix,
                    certified_rotation=int(rotations[state_index]))
                binding = _parallel_certificate_binding(
                    raw_path, receipt, "objective_wave_receipt_digest",
                    pinned_path=path)
                checkpoint_rows.append({
                    "kind": "objective_wave_receipt", "rank": rank,
                    "state_index": state_index,
                    "solver_status": receipt["solver_status"],
                    **binding,
                })
                objective_receipts.setdefault(rank, []).append(receipt)
                if terminal_wave:
                    raise RuntimeError(
                        "parallel objective continues after fatal status")
                if receipt["solver_status"] == "FEASIBLE":
                    prefix.append(int(receipt["selected_rotation"]))
                else:
                    terminal_wave = True
                    fatal_rows.append(dict(binding))

    raw_winner_path = raw_checkpoint_root / "winner-objective-validation.json"
    winner_path = checkpoint_root / "winner-objective-validation.json"
    if winner_path.exists():
        if winner_path.is_symlink() or not winner_path.is_file():
            raise RuntimeError("parallel winner validation path is not regular")
        winner_payload = json.loads(winner_path.read_text())
        winner_rank = winner_payload.get("rank")
        if (isinstance(winner_rank, bool) or not isinstance(winner_rank, int)
                or winner_rank not in rank_receipts):
            raise RuntimeError("parallel winner validation rank changed")
        rank_receipt = rank_receipts[winner_rank]
        if rank_receipt["classification"] != "PASS":
            raise RuntimeError("parallel winner validation lacks PASS evidence")
        projection, source, _states = material(winner_rank)
        winner_receipt = PARALLEL_SEARCH._validate_winner_validation_receipt(
            winner_payload, plan=plan, rank=winner_rank,
            projection_digest=projection, source_digest=source,
            allocation_digest=rank_receipt[
                "provisional_allocation_manifest_digest"],
            assignment_digest=rank_receipt[
                "provisional_candidate_assignment_set_digest"],
            rotations=rank_receipt["selected_rotations"],
            objective_wave_rows=[{
                "state_index": row["state_index"],
                "objective_wave_receipt_digest":
                    row["objective_wave_receipt_digest"],
            } for row in objective_receipts.get(winner_rank, [])])
        checkpoint_rows.append({
            "kind": "winner_validation_receipt", "rank": winner_rank,
            **_parallel_certificate_binding(
                raw_winner_path, winner_receipt,
                "winner_validation_receipt_digest", pinned_path=winner_path),
        })

    checkpoint_rows.sort(key=lambda row: (str(row["path"]), str(row["kind"])))
    fatal_rows.sort(key=lambda row: str(row["path"]))
    return rank_receipts, fatal_rows, checkpoint_rows


def _parallel_terminal_result_disposition(
        *, out: Path, status: str,
        terminal: Mapping[str, Any] | None) -> dict[str, Any]:
    """Bind terminal-result absence or the exact EXHAUSTED result bytes."""

    raw_path = out / PARALLEL_SMALL_TERMINAL_RESULT_NAME
    path = _pin_generated_path(raw_path, raw_path)
    try:
        logical_path = str(raw_path.relative_to(ROOT))
    except ValueError:
        logical_path = str(raw_path)
    if status == "FATAL":
        if terminal is not None or path.exists():
            raise RuntimeError(
                "FATAL failure contradicts an existing terminal result")
        return {"status": "ABSENT", "path": logical_path}
    if status != "EXHAUSTED" or terminal is None:
        raise RuntimeError("parallel failure terminal disposition changed")
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("EXHAUSTED terminal result is absent or irregular")
    raw = path.read_bytes()
    observed = json.loads(raw)
    if observed != dict(terminal):
        raise RuntimeError("EXHAUSTED terminal result bytes changed")
    return {
        "status": "PRESENT_EXHAUSTED",
        "path": logical_path,
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "terminal_result_digest": PARALLEL_SEARCH.canonical_digest(observed),
    }


def _parallel_failure_receipt(
        *, out: Path, inputs: Mapping[str, Any], plan: Mapping[str, Any],
        benchmark: Mapping[str, Any], status: str, reason: str,
        terminal: Mapping[str, Any] | None = None,
        certified_rank_receipts: Sequence[Mapping[str, Any]] | None = None,
        prepare_rank: Callable[
            [int, tuple[int, ...]], Mapping[str, Any]],
        ) -> dict[str, Any]:
    raw_checkpoint_root = out / PARALLEL_SMALL_CHECKPOINT_ROOT
    checkpoint_root = _parallel_search_checkpoint_root(out)
    total_rank_count = int(plan["total_rank_count"])
    if status == "EXHAUSTED":
        if (terminal is None or certified_rank_receipts is None
                or len(certified_rank_receipts) != total_rank_count):
            raise RuntimeError(
                "EXHAUSTED failure requires a certified complete rank frontier")
    elif certified_rank_receipts is not None:
        raise RuntimeError(
            "certified exhausted ranks cannot bind a non-EXHAUSTED failure")
    validated_ranks, fatal_rows, checkpoint_rows = \
        _parallel_failure_evidence_inventory(
            out=out, plan=plan, prepare_rank=prepare_rank)
    if (status == "FATAL" and len(validated_ranks) == total_rank_count
            and all(receipt["classification"] in {
                "ALLOCATOR_INFEASIBLE", "MASK_FAIL"
            } for receipt in validated_ranks.values())):
        raise RuntimeError(
            "complete ordinary-nonpass frontier requires EXHAUSTED evidence")
    rank_rows: list[dict[str, Any]] = []
    for rank in range(total_rank_count):
        raw_path = raw_checkpoint_root / "ranks" / f"{rank:012d}.json"
        path = checkpoint_root / "ranks" / f"{rank:012d}.json"
        loaded = validated_ranks.get(rank)
        if loaded is None:
            if status == "EXHAUSTED":
                raise RuntimeError("EXHAUSTED rank frontier is incomplete")
            break
        if (certified_rank_receipts is not None
                and loaded != dict(certified_rank_receipts[rank])):
            raise RuntimeError("certified EXHAUSTED rank receipt changed")
        rank_rows.append({
            "rank": rank, "classification": loaded["classification"],
            **_parallel_certificate_binding(
                raw_path, loaded, "rank_receipt_digest", pinned_path=path),
        })
    if status == "EXHAUSTED" and fatal_rows:
        raise RuntimeError("EXHAUSTED search contains fatal wave evidence")
    payload = {
        "schema": PARALLEL_SMALL_FAILURE_SCHEMA,
        "status": status,
        "complete": True,
        "reason": reason,
        "search_plan": _artifact_binding(
            out / PARALLEL_SMALL_SEARCH_PLAN_NAME,
            self_key="search_plan_digest"),
        "measured_benchmark": _artifact_binding(
            out / PARALLEL_SMALL_BENCHMARK_NAME,
            self_key="benchmark_receipt_digest"),
        "small_prefix_reissue_receipt":
            dict(inputs["prefix"]["receipt_binding"]),
        "rank_prefix_receipts": rank_rows,
        "rank_prefix_receipt_set_digest":
            PARALLEL_SEARCH.canonical_digest(rank_rows),
        "fatal_wave_receipts": fatal_rows,
        "fatal_wave_receipt_set_digest":
            PARALLEL_SEARCH.canonical_digest(fatal_rows),
        "checkpoint_evidence_inventory": checkpoint_rows,
        "checkpoint_evidence_inventory_digest":
            PARALLEL_SEARCH.canonical_digest(checkpoint_rows),
        "terminal_result_disposition":
            _parallel_terminal_result_disposition(
                out=out, status=status, terminal=terminal),
        "terminal_result": None if terminal is None else dict(terminal),
        "complete_identity_manifest_created": False,
        "candidate_outcomes_loaded": False,
        "branch_identities_created": False,
        "branches_attempted": 0,
        "frames_rendered": 0,
        "target_latents_encoded": 0,
        "scorer_training_started": False,
        "predictor_checkpoints_opened": 0,
    }
    payload["parallel_small_completion_failure_receipt_digest"] = \
        PARALLEL_SEARCH.canonical_digest(payload)
    return payload


def _load_existing_parallel_terminal_failure(
        *, out: Path, inputs: Mapping[str, Any], plan: Mapping[str, Any],
        benchmark: Mapping[str, Any],
        prepare_rank: Callable[
            [int, tuple[int, ...]], Mapping[str, Any]],
        classify_mask: Callable[[Sequence[Mapping[str, Any]],
                                 Mapping[str, Any], Mapping[str, Any]], bool],
        ) -> dict[str, Any] | None:
    """Reopen a terminal failure exactly; it permanently forbids a retry."""

    raw_path = out / PARALLEL_SMALL_TERMINAL_FAILURE_NAME
    path = _pin_generated_path(raw_path, raw_path)
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("parallel terminal failure path is not regular")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError("parallel terminal failure is not a mapping")
    observed = payload.get(
        "parallel_small_completion_failure_receipt_digest")
    if (not _is_sha256(observed)
            or observed != PARALLEL_SEARCH.canonical_digest({
                key: value for key, value in payload.items()
                if key != "parallel_small_completion_failure_receipt_digest"})):
        raise RuntimeError("parallel terminal failure self digest changed")
    status = payload.get("status")
    reason = payload.get("reason")
    terminal = payload.get("terminal_result")
    if (status not in {"FATAL", "EXHAUSTED"}
            or not isinstance(reason, str) or not reason
            or (status == "FATAL" and terminal is not None)
            or (status == "EXHAUSTED" and not isinstance(terminal, Mapping))):
        raise RuntimeError("parallel terminal failure status/reason changed")
    certified: list[dict[str, Any]] | None = None
    if status == "EXHAUSTED":
        certified = PARALLEL_SEARCH.validate_exhausted_search_result(
            terminal_result=dict(terminal), search_plan=plan,
            checkpoint_root=_parallel_search_checkpoint_root(out),
            prepare_rank=prepare_rank, classify_mask=classify_mask)
    expected = _parallel_failure_receipt(
        out=out, inputs=inputs, plan=plan, benchmark=benchmark,
        status=str(status), reason=reason,
        terminal=None if terminal is None else dict(terminal),
        certified_rank_receipts=certified,
        prepare_rank=prepare_rank)
    if payload != expected:
        raise RuntimeError("parallel terminal failure evidence changed")
    return payload


def stage_parallel_small_completion_search() -> int:
    """Run/resume the shared 32-worker ordered scientific coordinator."""

    out = OUT_ROOT / "scorer_fit"
    inputs = _parallel_small_search_inputs(out)
    plan, benchmark, _source_binding = _load_parallel_plan_and_benchmark(
        out, inputs)
    prepare, classify, validate = _parallel_search_callbacks(inputs)
    existing_failure = _load_existing_parallel_terminal_failure(
        out=out, inputs=inputs, plan=plan, benchmark=benchmark,
        prepare_rank=prepare, classify_mask=classify)
    if existing_failure is not None:
        raise RuntimeError(
            "parallel small completion search already terminated with "
            f"{existing_failure['status']}; retry is forbidden")
    raw_terminal_path = out / PARALLEL_SMALL_TERMINAL_RESULT_NAME
    terminal_path = _pin_generated_path(raw_terminal_path, raw_terminal_path)
    if terminal_path.exists() and (
            not terminal_path.is_file() or terminal_path.is_symlink()):
        raise RuntimeError("parallel terminal result path is not regular")
    try:
        if terminal_path.is_file():
            terminal = json.loads(terminal_path.read_text())
        else:
            terminal = PARALLEL_SEARCH.run_scientific_parallel_search(
                search_plan=plan,
                checkpoint_root=_parallel_search_checkpoint_root(out),
                prepare_rank=prepare,
                classify_mask=classify,
                validate_winner=validate,
            )
            _write_or_require_exact_json(
                raw_terminal_path, terminal,
                label="parallel small terminal search result")
        if terminal.get("status") == "EXHAUSTED":
            certified_exhausted_ranks = \
                PARALLEL_SEARCH.validate_exhausted_search_result(
                    terminal_result=terminal, search_plan=plan,
                    checkpoint_root=_parallel_search_checkpoint_root(out),
                    prepare_rank=prepare, classify_mask=classify,
                )
            failure = _parallel_failure_receipt(
                out=out, inputs=inputs, plan=plan, benchmark=benchmark,
                status="EXHAUSTED", reason=(
                    "no lexicographic cursor-restricted five-scene set passes "
                    "the exact allocator and all 40 allocated masks"),
                terminal=terminal,
                certified_rank_receipts=certified_exhausted_ranks,
                prepare_rank=prepare)
            _write_or_require_exact_json(
                out / PARALLEL_SMALL_TERMINAL_FAILURE_NAME, failure,
                label="parallel small EXHAUSTED receipt")
            raise RuntimeError("parallel small completion search exhausted")
        allocation = PARALLEL_SEARCH.validate_terminal_search_result(
            terminal_result=terminal, search_plan=plan,
            checkpoint_root=_parallel_search_checkpoint_root(out),
            prepare_rank=prepare, classify_mask=classify,
            validate_winner=validate,
        )
    except PARALLEL_SEARCH.ParallelSearchFatal as exc:
        failure = _parallel_failure_receipt(
            out=out, inputs=inputs, plan=plan, benchmark=benchmark,
            status="FATAL", reason=str(exc), prepare_rank=prepare)
        _write_or_require_exact_json(
            out / PARALLEL_SMALL_TERMINAL_FAILURE_NAME, failure,
            label="parallel small FATAL receipt")
        raise RuntimeError(
            "parallel small completion search failed fatally") from exc
    joint = _build_parallel_joint_receipt(
        out=out, inputs=inputs, plan=plan, benchmark=benchmark,
        terminal=terminal, allocation=allocation)
    _write_or_require_exact_json(
        out / PARALLEL_SMALL_JOINT_RECEIPT_NAME, joint,
        label="parallel small joint receipt")
    shard = _build_parallel_small_terminal_shard(
        inputs=inputs, terminal=terminal, joint_receipt=joint)
    shard_path = out / f"state_shard_{REACHABILITY_REDRIVE_FAMILY}.json"
    _write_or_require_exact_json(
        shard_path, shard, label="parallel small terminal state shard")
    _validate_state_shard(shard, shard_path, "scorer_fit")
    print(json.dumps({
        "status": joint["status"],
        "winning_rank": joint["rank"],
        "selected_scene_ids": joint["selected_scene_ids"],
        "allocation_manifest_digest":
            allocation["allocation_manifest_digest"],
        "state_shard_digest": shard["state_shard_digest"],
    }, indent=2, sort_keys=True))
    return 0


def _validate_small_completion_search_failure_static(
        payload: dict[str, Any], *, launch: dict[str, Any]) -> None:
    """Validate the closed schema and reason-dependent arithmetic."""

    exact_keys = {
        "schema", "status", "complete", "source_repository_commit",
        "clean_source_launch_receipt_digest", "state_selector_amendment_digest",
        "state_selector_feasibility_receipt_digest",
        "candidate_allocation_amendment_digest", "resolver_cursor_scene_id",
        "cursor_restricted_candidate_scene_ids",
        "cursor_restricted_candidate_scene_ids_digest",
        "cursor_restricted_candidate_scene_count", "combination_attempt_count",
        "allocator_infeasible_combination_count", "failure_reason",
        "complete_identity_manifest_created", "candidate_outcomes_loaded",
        "branch_identities_created", "branches_attempted", "frames_rendered",
        "target_latents_encoded", "scorer_training_started",
        "predictor_checkpoints_opened",
        "small_completion_joint_search_failure_digest",
    }
    scenes = payload.get("cursor_restricted_candidate_scene_ids")
    count = payload.get("cursor_restricted_candidate_scene_count")
    attempts = payload.get("combination_attempt_count")
    allocator_infeasible = payload.get(
        "allocator_infeasible_combination_count")
    fewer_than_five = (
        "cursor-restricted small completion pool has fewer than five scenes")
    exhausted = (
        "no lexicographic small completion combination passes all 40 exact "
        "allocated-mask reachability predicates")
    exact_ints = (count, attempts, allocator_infeasible)
    if (
        set(payload) != exact_keys
        or payload.get("schema") != SMALL_COMPLETION_SEARCH_FAILURE_SCHEMA
        or payload.get("status") != SMALL_COMPLETION_SEARCH_FAILURE_STATUS
        or payload.get("complete") is not True
        or payload.get("source_repository_commit")
        != launch["source_repository_commit"]
        or payload.get("clean_source_launch_receipt_digest")
        != launch["clean_source_launch_receipt_digest"]
        or payload.get("state_selector_amendment_digest")
        != STATE_SELECTOR.state_selector_amendment_digest()
        or payload.get("state_selector_feasibility_receipt_digest")
        != launch["state_selector_feasibility_receipt_digest"]
        or payload.get("candidate_allocation_amendment_digest")
        != ALLOC.allocation_amendment_digest()
        or not isinstance(payload.get("resolver_cursor_scene_id"), str)
        or not payload.get("resolver_cursor_scene_id")
        or not isinstance(scenes, list) or scenes != sorted(scenes)
        or any(not isinstance(scene, str) or not scene for scene in scenes)
        or len(set(scenes)) != len(scenes)
        or payload.get("cursor_restricted_candidate_scene_ids_digest")
        != canonical_digest(scenes)
        or any(not isinstance(value, int) or isinstance(value, bool)
               or value < 0 for value in exact_ints)
        or count != len(scenes)
        or any(payload.get(key) not in (False, 0) for key in (
            "complete_identity_manifest_created", "candidate_outcomes_loaded",
            "branch_identities_created", "branches_attempted", "frames_rendered",
            "target_latents_encoded", "scorer_training_started",
            "predictor_checkpoints_opened",
        ))
    ):
        raise RuntimeError("small completion terminal failure receipt is malformed")
    reason = payload.get("failure_reason")
    if reason == fewer_than_five:
        valid_reason_arithmetic = (
            count < 5 and attempts == 0 and allocator_infeasible == 0)
    elif reason == exhausted:
        valid_reason_arithmetic = (
            count >= 5
            and attempts == math.comb(count, 5)
            and allocator_infeasible <= attempts)
    else:
        valid_reason_arithmetic = False
    if not valid_reason_arithmetic:
        raise RuntimeError(
            "small completion terminal failure reason/arithmetic mismatch")


def _load_small_completion_search_failure(
        out: Path, *, args: argparse.Namespace | None = None,
        ) -> dict[str, Any] | None:
    """Reopen the superseded serial receipt as lineage only, never replay it."""

    raw_path = out / SMALL_COMPLETION_SEARCH_FAILURE_NAME
    path = _pin_generated_path(raw_path, raw_path)
    if not path.exists():
        return None
    if not path.is_file():
        raise RuntimeError("small completion failure path is not a file")
    payload = json.loads(path.read_text())
    _verify_self_digest(
        payload, "small_completion_joint_search_failure_digest",
        "small completion joint-search failure")
    launch = _load_clean_source_launch_receipt()
    _validate_small_completion_search_failure_static(payload, launch=launch)
    del args
    return payload


def _mixed_active_state_shard_path(out: Path, family: str) -> Path:
    if family not in {row["family"] for row in STATE_SELECTOR.PRESERVED_STATE_SHARDS}:
        raise RuntimeError("mixed active shard family is not a predecessor family")
    return out / MIXED_ACTIVE_STATE_SHARD_NAME.format(family=family)


def _active_state_shard_path(out: Path, family: str, *, pool: str) -> Path:
    if (pool == "scorer_fit"
            and family in {
                row["family"] for row in STATE_SELECTOR.PRESERVED_STATE_SHARDS}):
        return _mixed_active_state_shard_path(out, family)
    return out / f"state_shard_{family}.json"


def _load_current_performance_interruption_receipt() -> dict[str, Any]:
    """Reopen the exact source-bound interruption authority read-only."""

    source = clean_source_binding()
    return (
        PERFORMANCE_INTERRUPTION
        .load_and_validate_performance_interruption_receipt(
            expected_source_repository_commit=str(
                source["source_repository_commit"]),
            expected_clean_source_binding_digest=canonical_digest(source),
            expected_bound_implementations_digest=str(
                source["bound_implementations_digest"]),
            root=ROOT,
        )
    )


def _load_active_state_shard_evidence(
        out: Path, family: str, *, pool: str,
        ) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Load an active shard and retain its exact on-disk custody envelope.

    Seven scorer-fit shards are source-reissue wrappers after the performance
    interruption.  They may be unwrapped only after the archived transport is
    scientifically replayed under the current bindings.  The wrapper bytes,
    its lineage receipt, and the inner successor digest remain distinct
    provenance inputs; callers must never pretend the inner object occupied
    the active path.
    """

    raw_path = _active_state_shard_path(out, family, pool=pool)
    path = _pin_generated_path(raw_path, raw_path)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"active state shard {family} is missing")
    envelope = json.loads(path.read_text())
    try:
        evidence_path = str(raw_path.relative_to(ROOT))
    except ValueError:
        # Synthetic alias-custody tests may place the lexical generated root
        # outside the repository; production paths are always repository
        # relative and the canonical path is independently pinned above.
        evidence_path = str(raw_path)
    is_reissued = (
        envelope.get("schema") == PERFORMANCE_INTERRUPTION.REISSUED_SHARD_SCHEMA)
    if is_reissued:
        if pool != "scorer_fit" or family == REACHABILITY_REDRIVE_FAMILY:
            raise RuntimeError("source-reissued wrapper appeared outside fixed scorer-fit shards")
        receipt = _load_current_performance_interruption_receipt()
        payload = PERFORMANCE_INTERRUPTION.validate_reissued_fixed_state_shard(
            envelope,
            receipt=receipt,
            revalidate_predecessor=
                _revalidate_performance_interrupted_fixed_shard,
            root=ROOT,
        )
        evidence = {
            "envelope_schema": PERFORMANCE_INTERRUPTION.REISSUED_SHARD_SCHEMA,
            "active_path": evidence_path,
            "active_raw_sha256": file_sha256(path),
            "active_byte_count": path.stat().st_size,
            "source_reissued_state_shard_digest": str(
                envelope["source_reissued_state_shard_digest"]),
            "predecessor_state_shard_digest": str(
                envelope["predecessor_state_shard_digest"]),
            "performance_interruption_receipt_digest": str(
                envelope["performance_interruption_receipt_digest"]),
            "successor_state_shard_digest": str(payload["state_shard_digest"]),
        }
    else:
        payload = envelope
        evidence = {
            "envelope_schema": str(payload.get("schema", "")),
            "active_path": evidence_path,
            "active_raw_sha256": file_sha256(path),
            "active_byte_count": path.stat().st_size,
            "source_reissued_state_shard_digest": None,
            "predecessor_state_shard_digest": None,
            "performance_interruption_receipt_digest": None,
            "successor_state_shard_digest": str(
                payload.get("state_shard_digest", "")),
        }
    if not is_reissued:
        if (pool == "scorer_fit"
                and family in {
                    row["family"] for row in
                    STATE_SELECTOR.PRESERVED_STATE_SHARDS}):
            _validate_mixed_active_state_shard(payload, raw_path)
        else:
            _validate_state_shard(payload, raw_path, pool)
    if payload.get("family") != family:
        raise RuntimeError("active state shard family changed")
    if evidence["successor_state_shard_digest"] != payload["state_shard_digest"]:
        raise RuntimeError("active shard evidence and successor payload disagree")
    return path, payload, evidence


def _load_active_family_state_shard(
        out: Path, family: str, *, pool: str) -> tuple[Path, dict[str, Any]]:
    path, payload, _evidence = _load_active_state_shard_evidence(
        out, family, pool=pool)
    return path, payload


def _mixed_capture_provenance(
        *, out: Path, request: dict[str, Any], capture: dict[str, Any],
        interval_index: int) -> dict[str, Any]:
    family = str(request["family"])
    digest = str(request["mixed_replacement_scene_request_digest"])
    request_raw_path = _mixed_replacement_request_path(out, family, digest)
    capture_raw_path = _mixed_replacement_capture_path(out, family, digest)
    request_path = _pin_generated_path(request_raw_path, request_raw_path)
    capture_path = _pin_generated_path(capture_raw_path, capture_raw_path)
    return {
        "interval_index": int(interval_index),
        "scene_id": str(request["scene"]["scene_id"]),
        "replacement_slot_state_id": str(request["replacement_slot"]["state_id"]),
        "mixed_replacement_scene_request_digest": digest,
        "mixed_replacement_scene_capture_digest": str(
            capture["mixed_replacement_scene_capture_digest"]),
        "request_path": str(request_raw_path.relative_to(ROOT)),
        "request_raw_sha256": file_sha256(request_path),
        "request_byte_count": request_path.stat().st_size,
        "capture_path": str(capture_raw_path.relative_to(ROOT)),
        "capture_raw_sha256": file_sha256(capture_path),
        "capture_byte_count": capture_path.stat().st_size,
        "selected": capture["chosen_state"] is not None,
    }


def _build_mixed_replacement_failure(
        *, args: argparse.Namespace, out: Path, plan: dict[str, Any],
        interval_index: int, candidate_scene_ids: Sequence[str],
        accepted_states: Sequence[dict[str, Any]],
        provenance: Sequence[dict[str, Any]]) -> dict[str, Any]:
    launch = _load_clean_source_launch_receipt()
    interval = plan["interval_groups"][interval_index]
    payload = {
        "schema": MIXED_REPLACEMENT_FAILURE_SCHEMA,
        "status": MIXED_REPLACEMENT_FAILURE_STATUS,
        "complete": True,
        "binding_receipt": True,
        "source_repository_commit": launch["source_repository_commit"],
        "clean_source_launch_receipt_digest": launch[
            "clean_source_launch_receipt_digest"],
        "mixed_precontract_disposition_receipt_digest": launch[
            "mixed_precontract_disposition_receipt_digest"],
        "state_selector_amendment_digest":
            STATE_SELECTOR.state_selector_amendment_digest(),
        "candidate_allocation_amendment_digest":
            ALLOC.allocation_amendment_digest(),
        "family": str(args.family),
        "family_replacement_plan_digest": canonical_digest(plan),
        "failed_interval_index": int(interval_index),
        "failed_anchor_interval": {
            "lower_scene_id_exclusive": interval[
                "lower_scene_id_exclusive"],
            "upper_scene_id_exclusive": interval[
                "upper_scene_id_exclusive"],
            "vacant_ordinals": list(interval["vacant_ordinals"]),
            "replacement_slots": list(interval["replacement_slots"]),
        },
        "candidate_scene_ids": list(candidate_scene_ids),
        "candidate_scene_ids_digest": canonical_digest(list(candidate_scene_ids)),
        "required_replacement_count": len(interval["replacement_slots"]),
        "accepted_replacement_count": len(accepted_states),
        "accepted_replacement_identities": [{
            "state_id": str(state["state_id"]),
            "state_identity_digest": str(state["state_identity_digest"]),
            "scene_id": str(state["scene_id"]),
        } for state in accepted_states],
        "scanned_scene_count": len(provenance),
        "scene_capture_provenance": list(provenance),
        "scene_capture_provenance_digest": canonical_digest(list(provenance)),
        "failure_reason": "ANCHOR_INTERVAL_EXHAUSTED_BEFORE_VACANT_SLOTS_FILLED",
        "complete_identity_manifest_created": False,
        "candidate_allocation_loaded": False,
        "candidate_outcomes_loaded": False,
        "branch_identities_created": False,
        "branches_attempted": 0,
        "frames_rendered": 0,
        "target_latents_encoded": 0,
        "scorer_training_started": False,
        "predictor_checkpoints_opened": 0,
    }
    payload["mixed_replacement_failure_receipt_digest"] = canonical_digest(payload)
    return payload


def _issue_mixed_replacement_failure(
        *, args: argparse.Namespace, out: Path, plan: dict[str, Any],
        interval_index: int, candidate_scene_ids: Sequence[str],
        accepted_states: Sequence[dict[str, Any]],
        provenance: Sequence[dict[str, Any]]) -> dict[str, Any]:
    payload = _build_mixed_replacement_failure(
        args=args, out=out, plan=plan, interval_index=interval_index,
        candidate_scene_ids=candidate_scene_ids,
        accepted_states=accepted_states, provenance=provenance)
    raw_path = out / MIXED_REPLACEMENT_FAILURE_NAME
    path = _pin_generated_path(raw_path, raw_path)
    if path.exists():
        if not path.is_file() or path.is_symlink():
            raise RuntimeError("mixed replacement failure path is not a regular file")
        existing = json.loads(path.read_text())
        _verify_self_digest(
            existing, "mixed_replacement_failure_receipt_digest",
            "mixed replacement terminal failure")
        if existing != payload:
            raise RuntimeError("refusing to overwrite a different replacement failure")
        return existing
    atomic_json(path, payload)
    return payload


def _load_mixed_replacement_failure(
        *, args: argparse.Namespace, out: Path,
        pool: dict[str, list[Path]], exclusion: dict[str, Any]
        ) -> dict[str, Any] | None:
    """Accept a terminal only after replaying its exact captured prefix."""

    raw_path = out / MIXED_REPLACEMENT_FAILURE_NAME
    path = _pin_generated_path(raw_path, raw_path)
    if not path.exists():
        return None
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("mixed replacement terminal path is not regular")
    receipt = json.loads(path.read_text())
    _verify_self_digest(
        receipt, "mixed_replacement_failure_receipt_digest",
        "mixed replacement terminal failure")
    failure_family = str(receipt.get("family", ""))
    if failure_family not in pool:
        raise RuntimeError("mixed replacement terminal family is invalid")
    replay_args = argparse.Namespace(
        pool="scorer_fit", family=failure_family, backend="cpu")
    plan = _mixed_family_replacement_plan(failure_family)
    launch = _load_clean_source_launch_receipt()
    interval_index = receipt.get("failed_interval_index")
    if (not isinstance(interval_index, int) or isinstance(interval_index, bool)
            or not 0 <= interval_index < len(plan["interval_groups"])):
        raise RuntimeError("mixed replacement terminal interval is invalid")
    interval = plan["interval_groups"][interval_index]
    scenes = pool[str(receipt["family"])]
    global_retained, _rejected, _slots = _mixed_disposition_sets()
    retained_scene_ids = {
        str(row["scene_id"]) for row in global_retained.values()
    }
    candidates = _mixed_replacement_candidate_scenes(
        scenes=scenes, interval=interval,
        retained_scene_ids=retained_scene_ids)
    candidate_scene_ids = [scene.name for _ordinal, scene in candidates]
    provenance = receipt.get("scene_capture_provenance")
    if not isinstance(provenance, list):
        raise RuntimeError("mixed replacement terminal lacks capture provenance")
    accepted: list[dict[str, Any]] = []
    replayed: list[dict[str, Any]] = []
    if len(provenance) != len(candidates):
        raise RuntimeError("mixed replacement terminal did not exhaust its interval")
    for row, (scene_ordinal, scene_dir) in zip(provenance, candidates, strict=True):
        slot = interval["replacement_slots"][len(accepted)]
        request = _build_mixed_replacement_scene_request(
            args=replay_args, out=out, scene_dir=scene_dir,
            scene_ordinal=scene_ordinal, interval=interval, slot=slot,
            accepted_scene_ids_before=[state["scene_id"] for state in accepted],
            exclusion=exclusion,
            family_allow_list=[path.name for path in scenes], persist=False)
        request_path = _mixed_replacement_request_path(
            out, failure_family, request["mixed_replacement_scene_request_digest"])
        capture_path = _mixed_replacement_capture_path(
            out, failure_family, request["mixed_replacement_scene_request_digest"])
        pinned_request_path = _pin_generated_path(request_path, request_path)
        if (not pinned_request_path.is_file() or pinned_request_path.is_symlink()
                or json.loads(pinned_request_path.read_text()) != request):
            raise RuntimeError(
                "mixed replacement terminal lacks its exact durable request")
        capture = _load_valid_mixed_replacement_scene_capture(
            path=capture_path, request=request)
        if capture is None or capture.get("worker_failure") is not None:
            raise RuntimeError("mixed replacement terminal lacks a valid capture")
        replay_row = _mixed_capture_provenance(
            out=out, request=request, capture=capture,
            interval_index=interval_index)
        if row != replay_row:
            raise RuntimeError("mixed replacement terminal provenance changed")
        replayed.append(replay_row)
        if capture["chosen_state"] is not None:
            accepted.append(dict(capture["chosen_state"]))
            if len(accepted) == len(interval["replacement_slots"]):
                raise RuntimeError(
                    "mixed replacement terminal claims failure after quota passed")
    expected = _build_mixed_replacement_failure(
        args=replay_args, out=out, plan=plan, interval_index=interval_index,
        candidate_scene_ids=candidate_scene_ids, accepted_states=accepted,
        provenance=replayed)
    if receipt != expected or any(
            receipt.get(key) not in (False, 0) for key in (
                "complete_identity_manifest_created", "candidate_allocation_loaded",
                "candidate_outcomes_loaded", "branch_identities_created",
                "branches_attempted", "frames_rendered", "target_latents_encoded",
                "scorer_training_started", "predictor_checkpoints_opened")):
        raise RuntimeError("mixed replacement terminal differs from live replay")
    return receipt


def resolve_mixed_active_family(args: argparse.Namespace) -> dict[str, Any]:
    """Retain exact passing identities and fill only authorized vacant slots."""

    if args.pool != "scorer_fit" or args.backend != "cpu" or args.family is None:
        raise RuntimeError("mixed active resolution requires scorer-fit/family/cpu")
    _load_clean_source_launch_receipt()
    out = OUT_ROOT / "scorer_fit"
    pool, exclusion = scene_pool("scorer_fit")
    family = str(args.family)
    plan = _mixed_family_replacement_plan(family)
    terminal = _load_mixed_replacement_failure(
        args=args, out=out, pool=pool, exclusion=exclusion)
    if terminal is not None:
        raise RuntimeError(
            "terminal pre-outcome mixed replacement failure: "
            f"{terminal['family']} interval {terminal['failed_interval_index']}"
        )
    scenes = pool[family]
    family_allow_list = [path.name for path in scenes]
    retained_all, rejected_all, _slot_rows = _mixed_disposition_sets()
    retained_scene_ids = {str(row["scene_id"]) for row in retained_all.values()}
    rejected_identity_digests = set(rejected_all)
    replacements: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    rejections: dict[str, dict[str, int]] = {}
    for interval_index, interval in enumerate(plan["interval_groups"]):
        candidates = _mixed_replacement_candidate_scenes(
            scenes=scenes, interval=interval,
            retained_scene_ids=retained_scene_ids)
        accepted: list[dict[str, Any]] = []
        interval_provenance: list[dict[str, Any]] = []
        for scene_ordinal, scene_dir in candidates:
            if len(accepted) == len(interval["replacement_slots"]):
                break
            slot = interval["replacement_slots"][len(accepted)]
            request = _build_mixed_replacement_scene_request(
                args=args, out=out, scene_dir=scene_dir,
                scene_ordinal=scene_ordinal, interval=interval, slot=slot,
                accepted_scene_ids_before=[state["scene_id"] for state in accepted],
                exclusion=exclusion, family_allow_list=family_allow_list)
            capture = _get_or_run_mixed_replacement_scene_capture(
                args=args, request=request, out=out)
            row = _mixed_capture_provenance(
                out=out, request=request, capture=capture,
                interval_index=interval_index)
            provenance.append(row)
            interval_provenance.append(row)
            rejections[scene_dir.name] = dict(capture["scene_rejection_reasons"])
            chosen = capture["chosen_state"]
            if chosen is not None:
                chosen = dict(chosen)
                if chosen["state_identity_digest"] in rejected_identity_digests:
                    raise RuntimeError("rejected predecessor identity re-entered")
                accepted.append(chosen)
                replacements.append(chosen)
                print(
                    f"[mixed-replacement] {family[:22]:22s} "
                    f"{chosen['state_id']} {scene_dir.name} "
                    f"blocks={chosen['warmup_blocks']}", flush=True)
        if len(accepted) != len(interval["replacement_slots"]):
            failure = _issue_mixed_replacement_failure(
                args=args, out=out, plan=plan, interval_index=interval_index,
                candidate_scene_ids=[scene.name for _ordinal, scene in candidates],
                accepted_states=accepted, provenance=interval_provenance)
            print(json.dumps(failure, indent=2, sort_keys=True))
            raise RuntimeError(
                "terminal pre-outcome mixed replacement interval exhausted"
            )
        interval_rows.append({
            "interval_index": interval_index,
            "lower_scene_id_exclusive": interval["lower_scene_id_exclusive"],
            "upper_scene_id_exclusive": interval["upper_scene_id_exclusive"],
            "vacant_ordinals": list(interval["vacant_ordinals"]),
            "replacement_slot_state_ids": [
                row["state_id"] for row in interval["replacement_slots"]],
            "candidate_scene_ids": [scene.name for _ordinal, scene in candidates],
            "scanned_scene_ids": [row["scene_id"] for row in interval_provenance],
            "selected_scene_ids": [state["scene_id"] for state in accepted],
            "stopped_at_first_complete_prefix": True,
        })

    states = [dict(row) for row in plan["retained_states"]] + replacements
    states.sort(key=lambda row: (
        STRATA.index(str(row["stratum"])), str(row["state_id"])))
    if (
        len(states) != 15
        or len(replacements) != len(plan["replacement_slots"])
        or len({row["state_id"] for row in states}) != 15
        or len({row["scene_id"] for row in states}) != 15
        or len({row["state_identity_digest"] for row in states}) != 15
        or any(row["state_identity_digest"] in rejected_identity_digests
               for row in states)
    ):
        raise RuntimeError("mixed active family identity set is invalid")
    completion = sorted(
        (row for row in states if row["stratum"] == "completion_enriched"),
        key=lambda row: _completion_state_ordinal(str(row["state_id"])))
    if [row["scene_id"] for row in completion] != sorted(
            row["scene_id"] for row in completion):
        raise RuntimeError("mixed replacement changed lexical completion ordinals")
    INVALID_IDS.assert_disjoint(states, label=f"{family} mixed active state shard")
    source_shard_binding = next(
        dict(row) for row in STATE_SELECTOR.PRESERVED_STATE_SHARDS
        if row["family"] == family)
    bindings = _state_shard_bindings(
        args, exclusion, family_allow_list)
    shard = {
        "schema": MIXED_ACTIVE_STATE_SHARD_SCHEMA,
        "status": STATUS,
        "complete": True,
        "pool": "scorer_fit",
        "family": family,
        "spec": POOLS["scorer_fit"],
        "selection": SELECTION,
        **bindings,
        "predecessor_state_shard_binding": source_shard_binding,
        "retained_predecessor_identity_digests": sorted(
            row["state_identity_digest"] for row in plan["retained_states"]),
        "rejected_predecessor_identity_digests":
            plan["rejected_identity_digests"],
        "replacement_slot_fills": [{
            "state_id": row["state_id"],
            "state_identity_digest": row["state_identity_digest"],
            "scene_id": row["scene_id"],
            "split_role": row["split_role"],
        } for row in sorted(replacements, key=lambda value: value["state_id"])],
        "states": states,
        "scene_rejection_reasons": rejections,
        "mixed_replacement_subprocess_transport": {
            "schema": MIXED_REPLACEMENT_TRANSPORT_SCHEMA,
            "one_scene_per_subprocess": True,
            "atomic_capture_write_before_native_cleanup": True,
            "return_code_ignored_only_after_valid_capture": True,
            "resume_scope": "MISSING_OR_INVALID_REPLACEMENT_SCENE_CAPTURES_ONLY",
            "interval_rows": interval_rows,
            "scene_capture_count": len(provenance),
            "scene_capture_provenance_digest": canonical_digest(provenance),
            "candidate_outcomes_loaded": False,
        },
        "mixed_replacement_scene_capture_provenance": provenance,
    }
    shard["state_shard_digest"] = canonical_digest(shard)
    return shard


def resolve_states(args: argparse.Namespace) -> dict[str, Any]:
    if args.backend != "cpu":
        raise RuntimeError("the frozen branch backend is cpu")
    # This must happen before scene discovery or simulator construction.
    _load_clean_source_launch_receipt()
    if (args.pool == "scorer_fit"
            and args.family == REACHABILITY_REDRIVE_FAMILY):
        raise RuntimeError(
            "serial small-family resolution is superseded; use the prefix "
            "reissue, measured benchmark, and parallel search stages")
    spec = POOLS[args.pool]
    pool, exclusion = scene_pool(args.pool)
    if args.family is None:
        raise RuntimeError("state identity resolution must be sharded by one family")
    if args.family not in pool:
        raise RuntimeError(f"unknown family {args.family!r}")
    family = args.family
    scenes = pool[family]
    small_joint_search = (
        args.pool == "scorer_fit" and family == REACHABILITY_REDRIVE_FAMILY)
    states: list[dict[str, Any]] = []
    rejections: dict[str, dict[str, int]] = {}
    need = ({"evaluation": spec["states_per_family"]}
            if spec["strata"] is None else dict(spec["strata"]))
    if small_joint_search:
        # Completion identities come from the already-scoped 182-scene
        # reachability evidence and are jointly frozen only after the exact
        # unchanged allocation over all 120 identities succeeds.
        need["completion_enriched"] = 0
    found = {key: 0 for key in need}
    resolver_cursor_scene_id: str | None = None

    family_allow_list = [path.name for path in scenes]
    capture_provenance: list[dict[str, Any]] = []
    for scene_ordinal, scene_dir in enumerate(scenes):
        if all(found[key] >= need[key] for key in need):
            break
        resolver_cursor_scene_id = str(scene_dir.name)
        request = _build_state_resolution_scene_request(
            args=args, out=OUT_ROOT / args.pool, scene_dir=scene_dir,
            scene_ordinal=scene_ordinal, found=found, need=need,
            exclusion=exclusion, family_allow_list=family_allow_list)
        capture = _get_or_run_state_resolution_scene_capture(
            args=args, request=request, out=OUT_ROOT / args.pool)
        raw_capture_path = _state_resolution_capture_path(
            OUT_ROOT / args.pool, family,
            request["state_resolution_scene_request_digest"])
        raw_request_path = _state_resolution_request_path(
            OUT_ROOT / args.pool, family,
            request["state_resolution_scene_request_digest"])
        capture_path = _pin_generated_path(
            raw_capture_path, raw_capture_path)
        request_path = _pin_generated_path(
            raw_request_path, raw_request_path)
        capture_provenance.append({
            "scene_id": str(scene_dir.name),
            "state_resolution_scene_request_digest":
                request["state_resolution_scene_request_digest"],
            "state_resolution_scene_capture_digest":
                capture["state_resolution_scene_capture_digest"],
            "request_path": str(raw_request_path.relative_to(ROOT)),
            "request_raw_sha256": file_sha256(request_path),
            "request_byte_count": request_path.stat().st_size,
            "capture_path": str(raw_capture_path.relative_to(ROOT)),
            "capture_raw_sha256": file_sha256(capture_path),
            "capture_byte_count": capture_path.stat().st_size,
        })
        rejections[scene_dir.name] = dict(capture["scene_rejection_reasons"])
        chosen = capture["chosen_state"]
        if chosen is not None:
            chosen = dict(chosen)
            found[str(chosen["stratum"])] += 1
            states.append(chosen)
            print(f"[states] {args.pool} {family[:22]:22s} {chosen['stratum'][:20]:20s} "
                  f"{scene_dir.name} blocks={chosen['warmup_blocks']} "
                  f"edges={chosen['goal']['graph_edges']} "
                  f"d0={chosen['goal']['start_geodesic_m']:.2f}m", flush=True)

    incomplete = {key: [found[key], need[key]] for key in need
                  if found[key] != need[key]}
    if incomplete:
        raise RuntimeError(f"could not resolve frozen state quota for {family}: {incomplete}")
    states.sort(key=lambda state: (
        STRATA.index(state["stratum"]) if state["stratum"] in STRATA else 0,
        state["scene_id"],
    ))
    if len(states) != spec["states_per_family"]:
        raise RuntimeError("state shard count mismatch")
    if len({state["scene_id"] for state in states}) != len(states):
        raise RuntimeError("state shard reuses a scene")
    INVALID_IDS.assert_disjoint(states, label=f"{family} state shard")
    bindings = _state_shard_bindings(args, exclusion, [path.name for path in scenes])
    shard = {
        "schema": "go2_branch_corpus_v1_2_state_shard",
        "status": STATUS,
        "complete": True,
        "pool": args.pool,
        "family": family,
        "spec": spec,
        "selection": SELECTION,
        **bindings,
        "states": states,
        "scene_rejection_reasons": rejections,
        "state_resolution_subprocess_transport": {
            "schema": "go2_branch_corpus_v1_2_state_resolution_transport_v1",
            "one_scene_per_subprocess": True,
            "atomic_capture_write_before_native_cleanup": True,
            "return_code_ignored_only_after_valid_capture": True,
            "resume_scope": "MISSING_OR_INVALID_SCENE_CAPTURES_ONLY",
            "resolver_algorithm_digest":
                canonical_digest(STATE_RESOLUTION_REDUCER_CONTRACT),
            "resolver_cursor_scene_id": resolver_cursor_scene_id,
            "scene_capture_count": len(capture_provenance),
            "scene_capture_provenance_digest":
                canonical_digest(capture_provenance),
            "candidate_outcomes_loaded": False,
        },
        "state_resolution_scene_capture_provenance": capture_provenance,
    }
    shard["state_shard_digest"] = canonical_digest(shard)
    return shard


# ------------------------------------------------------------------- stage B --
_POST_ALLOCATION_STATE_FIELDS = frozenset({
    "state_index", "candidate_indices", "candidate_rotation_index",
    "branch_identities",
})


def _preallocation_state_projection(
        states: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Remove only registered merge-time fields from full scientific states."""

    if isinstance(states, (str, bytes)) or not isinstance(states, Sequence):
        raise RuntimeError("state projection input is not a sequence")
    projected: list[dict[str, Any]] = []
    for state in states:
        if not isinstance(state, Mapping):
            raise RuntimeError("state projection row is not a mapping")
        projected.append({
            key: value for key, value in state.items()
            if key not in _POST_ALLOCATION_STATE_FIELDS
        })
    return _joint_state_order(projected)


def _validate_small_completion_joint_search_receipt(
        *, manifest: dict[str, Any], allocation: dict[str, Any],
        replay_live: bool = True) -> None:
    """Strictly replay durable parallel certificates with zero MILP solves."""

    del replay_live  # Kept for source compatibility; all validation is strict.
    receipt = manifest.get("small_completion_joint_allocation_search")
    if not isinstance(receipt, dict):
        raise RuntimeError("state manifest lacks the parallel small joint receipt")
    if receipt.get("parallel_small_completion_joint_receipt_digest") != \
            PARALLEL_SEARCH.canonical_digest({
                key: value for key, value in receipt.items()
                if key != "parallel_small_completion_joint_receipt_digest"}):
        raise RuntimeError("parallel small joint receipt self digest mismatch")
    out = OUT_ROOT / "scorer_fit"
    inputs = _parallel_small_search_inputs(out)
    plan, benchmark, _source_binding = _load_parallel_plan_and_benchmark(
        out, inputs)
    raw_terminal = out / PARALLEL_SMALL_TERMINAL_RESULT_NAME
    terminal_path = _pin_generated_path(raw_terminal, raw_terminal)
    if not terminal_path.is_file() or terminal_path.is_symlink():
        raise RuntimeError("parallel terminal result is missing")
    terminal = json.loads(terminal_path.read_text())
    prepare, classify, validate = _parallel_search_callbacks(inputs)
    certified = PARALLEL_SEARCH.validate_terminal_search_result(
        terminal_result=terminal, search_plan=plan,
        checkpoint_root=_parallel_search_checkpoint_root(out),
        prepare_rank=prepare, classify_mask=classify,
        validate_winner=validate)
    expected_receipt = _build_parallel_joint_receipt(
        out=out, inputs=inputs, plan=plan, benchmark=benchmark,
        terminal=terminal, allocation=certified)
    if (receipt != expected_receipt or dict(allocation) != certified
            or manifest.get("small_prefix_reissue_receipt")
            not in (None, inputs["prefix"]["receipt_binding"])):
        raise RuntimeError("parallel small terminal certificate changed")
    manifest_states = _preallocation_state_projection(
        manifest.get("states", []))
    certified_states = _parallel_rank_material(
        inputs, int(terminal["rank"]), tuple(
            terminal["rank_receipt"]["combination_indices"]))["states"]
    if manifest_states != certified_states:
        raise RuntimeError("parallel terminal states differ from manifest")


def _certify_parallel_allocation_solve_free(
        out: Path, supplied_allocation: Mapping[str, Any]) -> dict[str, Any]:
    """Replay the complete durable terminal proof and return exact bytes."""

    inputs = _parallel_small_search_inputs(out)
    plan, _benchmark, _source_binding = _load_parallel_plan_and_benchmark(
        out, inputs)
    raw_terminal = out / PARALLEL_SMALL_TERMINAL_RESULT_NAME
    terminal_path = _pin_generated_path(raw_terminal, raw_terminal)
    if not terminal_path.is_file() or terminal_path.is_symlink():
        raise RuntimeError("parallel terminal result is missing")
    terminal = json.loads(terminal_path.read_text())
    prepare, classify, validate = _parallel_search_callbacks(inputs)
    certified = PARALLEL_SEARCH.validate_terminal_search_result(
        terminal_result=terminal, search_plan=plan,
        checkpoint_root=_parallel_search_checkpoint_root(out),
        prepare_rank=prepare, classify_mask=classify,
        validate_winner=validate)
    if dict(supplied_allocation) != certified:
        raise RuntimeError("solve-free certificate binds another allocation")
    return certified


def _validate_state_manifest(manifest: dict[str, Any], pool: str) -> None:
    _verify_self_digest(manifest, "state_manifest_digest", "state manifest")
    expected_states = EXPECTED_FAMILIES * POOLS[pool]["states_per_family"]
    expected_branches = expected_states * POOLS[pool]["candidates_per_state"]
    if (manifest.get("schema") != "go2_branch_corpus_v1_2_state_manifest"
            or manifest.get("complete") is not True
            or manifest.get("pool") != pool
            or manifest.get("scorer_contract_v1_2_digest")
            != scorer_contract_digest()
            or manifest.get("selection_digest") != selection_digest()
            or manifest.get("candidate_allocator_contract_digest")
            != ALLOC.allocation_contract_digest()
            or manifest.get("candidate_allocation_amendment_digest")
            != ALLOC.allocation_amendment_digest()
            or manifest.get("invalid_scorer_identity_exclusion_digest")
            != INVALID_IDS.invalid_identity_exclusion_digest()
            or manifest.get("state_selector_amendment_digest")
            != STATE_SELECTOR.state_selector_amendment_digest()
            or manifest.get("state_selector_feasibility_receipt_digest")
            != _load_clean_source_launch_receipt()[
                "state_selector_feasibility_receipt_digest"]
            or manifest.get("pre_identity_allocation_validation_digest")
            != _load_pre_identity_allocation_validation()[
                "pre_identity_validation_digest"]
            or manifest.get("candidate_bank_digest") != V1.bank_digest()
            or manifest.get("progress_contract_digest") != progress_digest()
            or manifest.get("safety_contract_digest") != safety_digest()
            or manifest.get("oracle_v1_2_digest") != v12_oracle_digest()
            or manifest.get("render_contract_digest") != render_contract_digest()
            or manifest.get("textured_v03_renderer_contract_digest")
            != textured_v03_renderer_contract_digest()
            or manifest.get("preprocess_contract_digest")
            != preprocess_contract_digest()
            or manifest.get("preprocessing_digest")
            != TARGET_ENCODER["preprocessing_identity_sha256"]
            or manifest.get("target_encoder_digest") != target_encoder_digest()
            or manifest.get("target_encoder_checkpoint_sha256")
            != TARGET_ENCODER["checkpoint_sha256"]
            or manifest.get("boundary_digest") != V1.BOUNDARY_DIGEST
            or manifest.get("genesis_backend") != "cpu"
            or len(manifest.get("states", [])) != expected_states
            or manifest.get("attempted_branch_count_registered") != expected_branches):
        raise RuntimeError("state manifest is incomplete or bound to another contract")
    launch = _load_clean_source_launch_receipt()
    for key in LAUNCH_BINDING_KEYS:
        if manifest.get(key) != launch[key]:
            raise RuntimeError(f"state manifest clean-source binding mismatch: {key}")
    _validate_state_shard_provenance(manifest, pool=pool)
    invalid_identity_index = INVALID_IDS.load_invalid_identity_index()
    if manifest.get("exclusion_binding", {}).get(
            "invalid_scorer_identity_attempt") != invalid_identity_index.binding():
        raise RuntimeError("state manifest invalid45 exclusion binding mismatch")
    INVALID_IDS.assert_disjoint(
        manifest["states"], label="state manifest", index=invalid_identity_index)
    identities = [identity for state in manifest["states"]
                  for identity in state["branch_identities"]]
    if (len(identities) != expected_branches
            or canonical_digest(sorted(row["branch_identity_digest"]
                                       for row in identities))
            != manifest["branch_identity_set_digest"]):
        raise RuntimeError("state manifest branch identity set is inconsistent")
    if len({state["scene_id"] for state in manifest["states"]}) != expected_states:
        raise RuntimeError("state manifest is not scene-disjoint")
    if any(not _state_identity_matches_active_or_preserved(state)
           for state in manifest["states"]):
        raise RuntimeError("state manifest contains a changed state identity")
    identity_bindings = {
        "pool": pool,
        **{key: manifest[key] for key in (
            "candidate_bank_digest", "oracle_v1_2_digest",
            "scorer_contract_v1_2_digest", "render_contract_digest",
            "textured_v03_renderer_contract_digest", "preprocess_contract_digest",
            "target_encoder_digest", "candidate_allocation_amendment_digest",
            "candidate_allocation_post_identity_validation_digest",
            "pre_identity_allocation_validation_digest",
            "invalid_scorer_identity_exclusion_digest",
            *ACTIVE_SELECTOR_BINDING_KEYS,
            *LAUNCH_BINDING_KEYS,
        )},
    }
    for state in manifest["states"]:
        for candidate_index in state["candidate_indices"]:
            expected = _branch_identity(state, int(candidate_index), identity_bindings)
            if _identity_for(state, int(candidate_index)) != expected:
                raise RuntimeError("state manifest contains a changed branch identity")
    raw_allocation_path = OUT_ROOT / pool / "candidate_allocation_manifest.json"
    allocation_path = _pin_generated_path(
        raw_allocation_path, raw_allocation_path)
    if not allocation_path.is_file() or allocation_path.is_symlink():
        raise RuntimeError("candidate allocation artifact is not a regular file")
    allocation = json.loads(allocation_path.read_text())
    if allocation.get("allocation_manifest_digest") \
            != manifest["candidate_allocation_manifest_digest"]:
        raise RuntimeError("candidate allocation artifact digest mismatch")
    if pool == "scorer_fit":
        preconditions = _load_state_selector_preconditions(
            source_commit=str(launch["source_repository_commit"]),
            successor_selection_digest=selection_digest(),
            clean_source_binding_digest=str(launch["clean_source_binding_digest"]),
            bound_implementations_digest=str(
                launch["bound_implementations_digest"]))
        if (preconditions["state_selector_feasibility_receipt_digest"]
                != launch["state_selector_feasibility_receipt_digest"]
                or preconditions[
                    "mixed_precontract_disposition_receipt_digest"]
                != launch[
                    "mixed_precontract_disposition_receipt_digest"]):
            raise RuntimeError("live selector preconditions differ from launch")
        # The terminal parallel certificate replays every fixed-r prefix wave,
        # the bounded winner-objective proof, exact allocation bytes, and all
        # completion masks.  It performs no new MILP solve.
        _validate_small_completion_joint_search_receipt(
            manifest=manifest, allocation=allocation)
        if (allocation["post_identity_pre_outcome_validation"][
                "post_identity_validation_digest"]
                != manifest[
                    "candidate_allocation_post_identity_validation_digest"]):
            raise RuntimeError(
                "post-identity allocation validation digest mismatch"
            )
        raw_revalidation_path = (
            ROOT / STATE_SELECTOR.PRESERVED_STATE_REVALIDATION_RECEIPT_PATH
        )
        revalidation_path = _pin_generated_path(
            raw_revalidation_path, raw_revalidation_path)
        if not revalidation_path.is_file():
            raise RuntimeError("post-allocation state revalidation receipt is missing")
        revalidation = json.loads(revalidation_path.read_text())
        launch = _load_clean_source_launch_receipt()
        STATE_SELECTOR.validate_preserved_state_revalidation_receipt_from_solve_free_certified_allocation(
            revalidation,
            allocation_manifest=allocation,
            active_states=manifest["states"],
            certify_allocation_solve_free=lambda supplied:
                _certify_parallel_allocation_solve_free(
                    OUT_ROOT / "scorer_fit", supplied),
            expected_source_commit=str(launch["source_repository_commit"]),
            expected_successor_selection_digest=selection_digest(),
            expected_feasibility_receipt_digest=str(
                launch["state_selector_feasibility_receipt_digest"]),
            expected_mixed_precontract_disposition_receipt_digest=str(
                launch["mixed_precontract_disposition_receipt_digest"]),
            root=ROOT,
        )
        if (manifest.get("preserved_state_revalidation_receipt_digest")
                != revalidation[
                    "preserved_state_revalidation_receipt_digest"]):
            raise RuntimeError("state manifest phase-2 revalidation digest mismatch")
    elif allocation["allocation_manifest_digest"] != canonical_digest({
            key: value for key, value in allocation.items()
            if key != "allocation_manifest_digest"}):
        raise RuntimeError("final candidate allocation self digest mismatch")
    assignment = {row["state_id"]: row for row in allocation["assignments"]}
    if any(list(state["candidate_indices"]) != list(
               assignment.get(state["state_id"], {}).get(
                   "candidate_indices", []))
           for state in manifest["states"]):
        raise RuntimeError("state manifest candidate assignments changed")
    if pool == "scorer_fit" and any(
            isinstance(state.get("candidate_rotation_index"), bool)
            or not isinstance(state.get("candidate_rotation_index"), int)
            or state["candidate_rotation_index"]
            != assignment.get(state["state_id"], {}).get("rotation_index")
            for state in manifest["states"]):
        raise RuntimeError("state manifest candidate rotations changed")


def validate_active_state_manifest_for_consumption(
        manifest: dict[str, Any], pool: str = "scorer_fit") -> None:
    """Reopen the complete live pre-outcome identity chain for consumers.

    This intentionally delegates to the corpus builder's single canonical
    validator.  It opens only identity/allocation/selector/provenance JSON and
    scene-manifest custody evidence: it never opens a frame, latent, scorer
    weight, predictor checkpoint, or scientific branch outcome.  Encoder,
    trainer and transfer consumers must call it before their first scientific
    data or model access.
    """

    if pool != "scorer_fit":
        raise RuntimeError(
            "shared scorer consumers require the active scorer-fit manifest")
    if not isinstance(manifest, dict):
        raise RuntimeError("active scorer-fit manifest must be a mapping")
    _validate_state_manifest(manifest, pool)


def load_active_state_manifest_for_consumption(
        path: Path, pool: str = "scorer_fit") -> dict[str, Any]:
    """Pin, read, and fully validate the one active scorer-fit manifest.

    Consumers must use this loader instead of parsing the lexical generated
    path themselves.  The canonical path returned by the managed-root guard is
    retained for the read, so swapping the root alias after resolution cannot
    redirect it.  Descendant and leaf symlinks remain forbidden.
    """

    if pool != "scorer_fit":
        raise RuntimeError(
            "shared scorer consumers require the active scorer-fit manifest")
    raw_path = Path(path)
    expected_raw_path = OUT_ROOT / pool / "state_manifest.json"
    pinned = _pin_generated_path(raw_path, expected_raw_path)
    if not pinned.is_file() or pinned.is_symlink():
        raise RuntimeError("active scorer-fit state manifest is missing")
    try:
        manifest = json.loads(pinned.read_text())
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "active scorer-fit state manifest is not valid JSON") from exc
    validate_active_state_manifest_for_consumption(manifest, pool)
    return manifest


# Explicit long-form alias for call sites that prefer the operation order in
# the name.  Both names share the same pin-before-read implementation.
load_and_validate_active_state_manifest_for_consumption = \
    load_active_state_manifest_for_consumption


def pin_active_scorer_fit_artifact_for_consumption(
        path: Path, relative_path: str | Path) -> Path:
    """Pin one explicitly registered scorer-fit identity/provenance JSON.

    This deliberately does not grant a generic generated-tree read surface.
    Downstream consumers may reopen only the finite pre-outcome chain that the
    canonical manifest validator already verifies.
    """

    relative = Path(relative_path)
    allowed = {
        Path(PRE_IDENTITY_VALIDATION_NAME),
        Path(LAUNCH_RECEIPT_NAME),
        Path(REACHABILITY_FEASIBILITY_RECEIPT_NAME),
        Path(STATE_SELECTOR.PRESERVED_STATE_MIXED_PRECONTRACT_DISPOSITION_RECEIPT_PATH).name,
        Path(STATE_SELECTOR.PRESERVED_STATE_REVALIDATION_RECEIPT_PATH).name,
        Path("candidate_allocation_manifest.json"),
        Path("state_manifest.json"),
    }
    allowed_paths = {Path(value) for value in allowed}
    if (relative.is_absolute() or relative not in allowed_paths
            or len(relative.parts) != 1):
        raise RuntimeError(
            "artifact is not in the registered scorer-fit consumption set")
    expected = OUT_ROOT / "scorer_fit" / relative
    return _pin_generated_path(Path(path), expected)


def _identity_for(state: dict[str, Any], candidate_index: int) -> dict[str, Any]:
    matches = [row for row in state["branch_identities"]
               if int(row["candidate_index"]) == int(candidate_index)]
    if len(matches) != 1:
        raise RuntimeError("state manifest candidate identity lookup is ambiguous")
    return matches[0]


def _row_path(out: Path, identity: dict[str, Any]) -> Path:
    return out / "row_records" / f"{identity['branch_identity_digest']}.json"


def _resolve_corpus_file(out: Path, relative: str) -> Path:
    path = (out / relative).resolve()
    if out.resolve() not in path.parents:
        raise RuntimeError(f"corpus artifact escapes output root: {relative}")
    return path


def _validate_frame_record(out: Path, record: dict[str, Any]) -> None:
    path = _resolve_corpus_file(out, str(record["path"]))
    if (not path.is_file()
            or path.stat().st_size != int(record["byte_count"])
            or file_sha256(path) != record["sha256"]
            or record.get("shape") != [224, 224, 3]
            or record.get("dtype") != "uint8"):
        raise RuntimeError(f"frame receipt mismatch for {path}")


def _validate_branch_row(row: dict[str, Any], state: dict[str, Any],
                         identity: dict[str, Any], manifest: dict[str, Any],
                         out: Path) -> None:
    _verify_self_digest(row, "branch_row_digest",
                        f"branch row {state['state_id']}|{identity['candidate']}")
    expected = {
        "pool": manifest["pool"],
        "state_id": state["state_id"],
        "state_identity_digest": state["state_identity_digest"],
        "branch_identity_digest": identity["branch_identity_digest"],
        "candidate": identity["candidate"],
        "candidate_index": int(identity["candidate_index"]),
        "state_manifest_digest": manifest["state_manifest_digest"],
        "candidate_allocation_manifest_digest":
            manifest["candidate_allocation_manifest_digest"],
        "candidate_allocator_contract_digest":
            manifest["candidate_allocator_contract_digest"],
        "candidate_allocation_amendment_digest":
            manifest["candidate_allocation_amendment_digest"],
        "candidate_allocation_post_identity_validation_digest":
            manifest["candidate_allocation_post_identity_validation_digest"],
        "pre_identity_allocation_validation_digest":
            manifest["pre_identity_allocation_validation_digest"],
        "invalid_scorer_identity_exclusion_digest":
            manifest["invalid_scorer_identity_exclusion_digest"],
        **{key: manifest[key] for key in ACTIVE_SELECTOR_BINDING_KEYS},
        **{key: manifest[key] for key in LAUNCH_BINDING_KEYS},
        "candidate_bank_digest": manifest["candidate_bank_digest"],
        "progress_contract_digest": manifest["progress_contract_digest"],
        "safety_contract_digest": manifest["safety_contract_digest"],
        "oracle_v1_2_digest": manifest["oracle_v1_2_digest"],
        "scorer_contract_v1_2_digest": manifest["scorer_contract_v1_2_digest"],
        "selection_digest": manifest["selection_digest"],
        "boundary_digest": manifest["boundary_digest"],
        "render_contract_digest": manifest["render_contract_digest"],
        "textured_v03_renderer_contract_digest":
            manifest["textured_v03_renderer_contract_digest"],
        "preprocess_contract_digest": manifest["preprocess_contract_digest"],
        "preprocessing_digest": manifest["preprocessing_digest"],
        "target_encoder_digest": manifest["target_encoder_digest"],
        "target_encoder_checkpoint_sha256":
            manifest["target_encoder_checkpoint_sha256"],
    }
    if row.get("schema") != "go2_branch_corpus_v1_2_branch_row" \
            or row.get("record_complete") is not True:
        raise RuntimeError("branch row is not a completion record")
    for key, value in expected.items():
        if row.get(key) != value:
            raise RuntimeError(f"branch row {key} mismatch")
    for key in ("state_index", "split_role", "stratum", "scene_id", "family",
                "split", "episode_cluster_id", "episode_id", "source_step"):
        if row.get(key) != state[key]:
            raise RuntimeError(f"branch row state field {key} mismatch")
    if row.get("primitives") != identity["primitives"]:
        raise RuntimeError("branch row primitive sequence mismatch")
    if row.get("goal") != state["goal"]:
        raise RuntimeError("branch row goal binding mismatch")
    INVALID_IDS.assert_disjoint(
        [row], label="branch row", index=INVALID_IDS.load_invalid_identity_index())
    goal = state["goal"]
    expected_goal_binding = [
        math.sin(float(goal["bearing_body_rad"])),
        math.cos(float(goal["bearing_body_rad"])),
        float(goal["range_m"]),
    ]
    if not np.allclose(np.asarray(row.get("goal_binding_input"), dtype=np.float64),
                       np.asarray(expected_goal_binding, dtype=np.float64),
                       rtol=0.0, atol=1e-12):
        raise RuntimeError("branch row numeric goal binding mismatch")

    previous = np.asarray(row.get("previous_applied_command"), dtype=np.float64)
    if previous.shape != (3,) or not np.all(np.isfinite(previous)):
        raise RuntimeError("branch row previous applied command is malformed")
    candidate = V1.CANDIDATE_BANK[int(identity["candidate_index"])]
    requested, post_slew_plan, action_blocks = candidate_planning_trajectory(
        candidate, previous.tolist())
    if row.get("requested") != requested:
        raise RuntimeError("branch row requested candidate plan mismatch")
    if not np.allclose(np.asarray(row.get("candidate_post_slew_plan"),
                                  dtype=np.float64),
                       np.asarray(post_slew_plan, dtype=np.float64),
                       rtol=0.0, atol=1e-12):
        raise RuntimeError("branch row post-slew candidate plan mismatch")
    if not np.allclose(np.asarray(row.get("action_blocks"), dtype=np.float64),
                       np.asarray(action_blocks, dtype=np.float64),
                       rtol=0.0, atol=1e-12):
        raise RuntimeError("branch row scorer action blocks mismatch")
    realised_prefix = np.asarray(row.get("post_slew"), dtype=np.float64)
    if (realised_prefix.ndim != 3 or realised_prefix.shape[1:] != (5, 3)
            or realised_prefix.shape[0] > HORIZONS
            or not np.all(np.isfinite(realised_prefix))
            or not np.allclose(realised_prefix,
                               np.asarray(post_slew_plan[:len(realised_prefix)],
                                          dtype=np.float64),
                               rtol=0.0, atol=1e-6)):
        raise RuntimeError("branch row realised post-slew prefix mismatch")
    action_context = np.asarray(row.get("action_context_blocks"), dtype=np.float64)
    proprio = np.asarray(row.get("proprio"), dtype=np.float64)
    control = np.asarray(row.get("control"), dtype=np.float64)
    if (action_context.shape != (CONTEXT_SLOTS, SLEW.ACTION_DIM)
            or proprio.shape != (PROPRIO_HISTORY, 30)
            or control.shape != (PROPRIO_HISTORY, 2)
            or not np.all(np.isfinite(action_context))
            or not np.all(np.isfinite(proprio))
            or not np.all(np.isfinite(control))):
        raise RuntimeError("branch row planning histories are malformed")
    context = row.get("context_frames", [])
    horizons = row.get("horizon_frames", [])
    if row.get("context_paths") != [frame.get("path") for frame in context] \
            or row.get("horizon_paths") != [frame.get("path") for frame in horizons]:
        raise RuntimeError("branch row frame-path projection mismatch")
    if row.get("valid"):
        if len(context) != CONTEXT_SLOTS or len(horizons) != HORIZONS:
            raise RuntimeError("valid branch row lacks exact H=1..4 renders")
        if any(row.get(key) is None for key in ("progress", "safety",
                                                "completion", "utility")):
            raise RuntimeError("valid branch row lacks oracle labels")
    elif not isinstance(row.get("invalid_reason"), str) or not row["invalid_reason"]:
        raise RuntimeError("invalid branch row lacks a reason code")
    for frame in context + horizons:
        _validate_frame_record(out, frame)


def _completed_rows(manifest: dict[str, Any], out: Path) -> dict[tuple[str, int], dict[str, Any]]:
    completed: dict[tuple[str, int], dict[str, Any]] = {}
    for state in manifest["states"]:
        for candidate_index in state["candidate_indices"]:
            identity = _identity_for(state, int(candidate_index))
            path = _row_path(out, identity)
            if not path.exists():
                continue
            try:
                row = json.loads(path.read_text())
                _validate_branch_row(row, state, identity, manifest, out)
            except Exception as exc:
                preserved = _preserve_invalid(path, out, "row-validation-failed")
                print(f"[recovery] preserved invalid row {preserved}: {exc}", flush=True)
                continue
            completed[(state["state_id"], int(candidate_index))] = row
    return completed


def _frame_receipt(result: Any, path: Path, out: Path, *, index_key: str,
                   index_value: int) -> dict[str, Any]:
    digest, byte_count = write_png_atomic(result.image, path, out)
    return {
        index_key: int(index_value),
        "path": str(path.relative_to(out)),
        "sha256": digest,
        "byte_count": byte_count,
        "shape": [224, 224, 3],
        "dtype": "uint8",
        "camera_pose_world": result.camera_pose_world,
        "render_runtime_s": round(float(result.runtime_s), 6),
    }


def _row_bindings(manifest: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "candidate_allocation_manifest_digest", "candidate_allocator_contract_digest",
        "candidate_allocation_amendment_digest",
        "candidate_allocation_post_identity_validation_digest",
        "pre_identity_allocation_validation_digest",
        "invalid_scorer_identity_exclusion_digest", "candidate_bank_digest",
        *ACTIVE_SELECTOR_BINDING_KEYS,
        *LAUNCH_BINDING_KEYS,
        "progress_contract_digest", "safety_contract_digest",
        "oracle_v1_2_digest", "scorer_contract_v1_2_digest", "selection_digest",
        "boundary_digest", "render_contract_digest",
        "textured_v03_renderer_contract_digest", "preprocess_contract_digest",
        "preprocessing_digest", "target_encoder_digest",
        "target_encoder_checkpoint_sha256",
    )
    return {key: manifest[key] for key in keys}


def _redrive_mismatch(entry: dict[str, Any], record: dict[str, Any],
                      ctx: V1.BranchContext) -> str | None:
    comparisons = {
        "source_step": int(record["boundary"]["source_step"]) == int(entry["source_step"]),
        "boundary": record["boundary"] == entry["boundary"],
        "episode_id": int(ctx.runner.episode_states[0].episode_id)
                      == int(entry["episode_id"]),
        "cell_id": int(record["cell_id"]) == int(entry["cell_id"]),
        "goal": record["goal"] == entry["goal"],
        "body_clearance": float(record["body_clearance_m"])
                          == float(entry["body_clearance_m"]),
        "clearance": float(record["clearance_m"])
                     == float(entry["clearance_m"]),
    }
    if entry.get("stratum") == "completion_enriched":
        if "completion_rotation_eligibility_vector" in entry:
            # New V2 successor identities bind the complete twelve-rotation
            # vector plus the unprojected task status and actual previous
            # applied command.  They are not members of the byte-frozen V1
            # preserved set and must be checked against their own evidence.
            comparisons.update({
                "completion_rotation_eligibility_vector":
                    record.get("completion_rotation_eligibility_vector")
                    == entry.get("completion_rotation_eligibility_vector"),
                "snapshot_task_status": record.get("snapshot_task_status")
                                        == entry.get("snapshot_task_status"),
                "previous_applied_command":
                    record.get("previous_applied_command")
                    == entry.get("previous_applied_command"),
            })
        elif "completion_eligibility" in entry:
            # Lineage-only compatibility for a pre-V2 identity form.  Active
            # successor identities never use this field.
            comparisons.update({
                "completion_eligibility": record.get("completion_eligibility")
                                          == entry.get("completion_eligibility"),
                "snapshot_task_status": record.get("snapshot_task_status")
                                        == entry.get("snapshot_task_status"),
            })
        else:
            # The 45 phase-1 predecessor identities predate evidence fields;
            # their exact payloads cannot be changed.  They are admissible only
            # through the byte-bound predecessor set, and current redrive must
            # independently pass the successor predicate with all task flags
            # false before any branch is attempted.
            status = record.get("snapshot_task_status", {})
            vector = record.get("completion_rotation_eligibility_vector", {})
            comparisons.update({
                "preserved_completion_identity":
                    str(entry.get("state_identity_digest"))
                    in _preserved_states_by_digest(),
                "completion_successor_eligible": bool(
                    vector.get("eligible_under_at_least_one_rotation")),
                "completion_task_status": all(
                    status.get(key) is False for key in (
                        "task_completed", "goal_claimed", "terminated", "truncated"
                    )
                ),
            })
    failed = [name for name, passed in comparisons.items() if not passed]
    return None if not failed else "redrive_" + "_".join(failed) + "_mismatch"


def candidate_planning_trajectory(candidate: tuple[str, tuple[str, ...]],
                                  previous_applied: Sequence[float]
                                  ) -> tuple[list[list[list[float]]],
                                             list[list[list[float]]],
                                             list[list[float]]]:
    """Frozen full four-block request/post-slew plan, without future state."""

    previous = tuple(float(value) for value in previous_applied)
    requested: list[list[list[float]]] = []
    post_slew: list[list[list[float]]] = []
    action_blocks: list[list[float]] = []
    for primitive in candidate[1]:
        requested_block = np.asarray(V1.block_for(primitive), dtype=np.float64).tolist()
        reconstructed, previous = SLEW.reconstruct_block(primitive, previous)
        requested.append(requested_block)
        post_slew.append([[float(value) for value in tick]
                          for tick in reconstructed])
        action_blocks.append(action_block_10d(np.asarray(reconstructed,
                                                         dtype=np.float64)))
    if len(action_blocks) != HORIZONS or any(len(block) != SLEW.ACTION_DIM
                                             for block in action_blocks):
        raise RuntimeError("candidate planning action shape changed")
    return requested, post_slew, action_blocks


def _invalid_completed_row(entry: dict[str, Any], identity: dict[str, Any],
                           manifest: dict[str, Any], reason: str,
                           runtime_s: float) -> dict[str, Any]:
    row = {
        "schema": "go2_branch_corpus_v1_2_branch_row",
        "status": STATUS,
        "record_complete": True,
        "pool": manifest["pool"],
        "state_id": entry["state_id"],
        "state_index": int(entry["state_index"]),
        "state_identity_digest": entry["state_identity_digest"],
        "branch_identity_digest": identity["branch_identity_digest"],
        "split_role": entry["split_role"],
        "stratum": entry["stratum"],
        "scene_id": entry["scene_id"],
        "family": entry["family"],
        "split": entry["split"],
        "episode_cluster_id": entry["episode_cluster_id"],
        "episode_id": entry["episode_id"],
        "source_step": entry["source_step"],
        "candidate": identity["candidate"],
        "candidate_index": int(identity["candidate_index"]),
        "primitives": identity["primitives"],
        "goal": entry["goal"],
        "requested": None,
        "post_slew": None,
        "action_blocks": None,
        "action_context_blocks": None,
        "previous_applied_command": None,
        "context_frames": [],
        "horizon_frames": [],
        "context_paths": [],
        "horizon_paths": [],
        "proprio": None,
        "control": None,
        "valid": False,
        "invalid_reason": reason,
        "blocks_completed": 0,
        "truncated_at_block": None,
        "snapshot_digest": None,
        "wall_time_s": round(float(runtime_s), 6),
        "storage_bytes": 0,
        "state_manifest_digest": manifest["state_manifest_digest"],
        **_row_bindings(manifest),
    }
    for key in ("start_geodesic_m", "final_geodesic_m", "progress",
                "contact_fraction", "clearance_cost", "stuck_fraction", "fall",
                "safety", "completion", "utility", "min_clearance_m",
                "evaluation_points"):
        row[key] = None
    row["branch_row_digest"] = canonical_digest(row)
    return row


def _write_row(out: Path, identity: dict[str, Any], row: dict[str, Any]) -> None:
    path = _row_path(out, identity)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing == row:
            return
        _preserve_invalid(path, out, "row-overwrite-refused")
    atomic_json(path, row)


def _write_invalid_attempt_row(out: Path, identity: dict[str, Any],
                               row: dict[str, Any], reason: str) -> Path:
    root = out / "invalid_attempts/redrive_records"
    root.mkdir(parents=True, exist_ok=True)
    stem = f"{identity['branch_identity_digest']}.{reason.replace(':', '-')}.invalid.json"
    path = root / stem
    counter = 0
    while path.exists():
        counter += 1
        path = root / f"{stem}.{counter}"
    atomic_json(path, row)
    return path


def _compiled_receipt(
        manifest: dict[str, Any], out: Path, ordered: list[dict[str, Any]],
        completed_states: int, rows_text: str,
        invocation_runtime_s: float) -> dict[str, Any]:
    """Build the exact derived receipt without mutating the corpus.

    Keeping this calculation independent of the on-disk ledger lets a
    zero-new resume prove that both existing derived artifacts are already
    exact, then retain their bytes rather than rewriting operational timing.
    """

    expected_states = len(manifest["states"])
    expected_branches = int(manifest["attempted_branch_count_registered"])
    valid_count = sum(bool(row["valid"]) for row in ordered)
    complete = len(ordered) == expected_branches and completed_states == expected_states
    rows_bytes = rows_text.encode()
    branch_rows_sha = hashlib.sha256(rows_bytes).hexdigest()
    frame_sizes: dict[str, int] = {}
    for row in ordered:
        for frame in row.get("context_frames", []) + row.get("horizon_frames", []):
            relative = str(frame["path"])
            byte_count = int(frame["byte_count"])
            if relative in frame_sizes and frame_sizes[relative] != byte_count:
                raise RuntimeError("shared frame receipts disagree on byte count")
            frame_sizes[relative] = byte_count
    frame_storage_bytes = sum(frame_sizes.values())
    row_record_storage_bytes = sum(
        _row_path(out, _identity_for(state, int(candidate_index))).stat().st_size
        for state in manifest["states"]
        for candidate_index in state["candidate_indices"]
        if _row_path(out, _identity_for(state, int(candidate_index))).is_file()
    )
    ledger_storage_bytes = len(rows_bytes)
    storage_bytes = (frame_storage_bytes + row_record_storage_bytes
                     + ledger_storage_bytes)
    runtime_total = sum(float(row.get("wall_time_s") or 0.0) for row in ordered)
    payload = {
        "schema": "go2_branch_corpus_v1_2_corpus_identity",
        "pool": manifest["pool"],
        "state_manifest_digest": manifest["state_manifest_digest"],
        "candidate_allocation_manifest_digest":
            manifest["candidate_allocation_manifest_digest"],
        "branch_identity_set_digest": manifest["branch_identity_set_digest"],
        "branch_rows_sha256": branch_rows_sha,
        "branch_row_digests": [row["branch_row_digest"] for row in ordered],
        "state_count": expected_states,
        "attempted_branch_count": len(ordered),
        "valid_branch_count": valid_count,
        "invalid_branch_count": len(ordered) - valid_count,
        "complete": complete,
        "bound_digests": _row_bindings(manifest),
    }
    receipt = {
        "schema": "go2_branch_corpus_v1_2_completion_receipt",
        "status": STATUS,
        "pool": manifest["pool"],
        "complete": complete,
        "states": expected_states,
        "state_count": expected_states,
        "completed_states": completed_states,
        "expected_branches": expected_branches,
        "attempted_branches": len(ordered),
        "attempted_count": len(ordered),
        "rows": len(ordered),
        "valid_branches": valid_count,
        "valid_count": valid_count,
        "invalid_branches": len(ordered) - valid_count,
        "invalid_count": len(ordered) - valid_count,
        "state_manifest_digest": manifest["state_manifest_digest"],
        "candidate_allocation_manifest_digest":
            manifest["candidate_allocation_manifest_digest"],
        "branch_rows_sha256": branch_rows_sha,
        **_row_bindings(manifest),
        "corpus_digest_payload": payload,
        "corpus_digest": canonical_digest(payload),
        "runtime_s_completed_rows": round(runtime_total, 6),
        "runtime_s_this_invocation": round(float(invocation_runtime_s), 6),
        "storage_bytes": storage_bytes,
        "storage_components_bytes": {
            "unique_rendered_frames": frame_storage_bytes,
            "row_records": row_record_storage_bytes,
            "branch_rows_ledger": ledger_storage_bytes,
        },
    }
    return receipt


def _load_exact_compiled_receipt(
        manifest: dict[str, Any], out: Path, ordered: list[dict[str, Any]],
        completed_states: int, rows_text: str) -> dict[str, Any] | None:
    """Return an existing byte-valid ledger/receipt pair, otherwise ``None``."""

    rows_path = out / "branch_rows.jsonl"
    receipt_path = out / "corpus_receipt.json"
    if not rows_path.is_file() or not receipt_path.is_file():
        return None
    try:
        if rows_path.read_bytes() != rows_text.encode():
            return None
        existing = json.loads(receipt_path.read_text())
        runtime = existing.get("runtime_s_this_invocation")
        if (isinstance(runtime, bool) or not isinstance(runtime, (int, float))
                or not math.isfinite(float(runtime)) or float(runtime) < 0.0):
            return None
        expected = _compiled_receipt(
            manifest, out, ordered, completed_states, rows_text, float(runtime))
        if existing != expected:
            return None
        return existing
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _compile_corpus(manifest: dict[str, Any], out: Path,
                    invocation_runtime_s: float) -> dict[str, Any]:
    completed = _completed_rows(manifest, out)
    ordered: list[dict[str, Any]] = []
    completed_states = 0
    for state in manifest["states"]:
        state_rows = []
        for candidate_index in state["candidate_indices"]:
            row = completed.get((state["state_id"], int(candidate_index)))
            if row is not None:
                state_rows.append(row)
                ordered.append(row)
        if len(state_rows) == len(state["candidate_indices"]):
            completed_states += 1
    rows_text = "".join(json.dumps(V1._jsonable(row), sort_keys=True) + "\n"
                        for row in ordered)
    retained = _load_exact_compiled_receipt(
        manifest, out, ordered, completed_states, rows_text)
    if retained is not None:
        return retained

    rows_path = out / "branch_rows.jsonl"
    receipt_path = out / "corpus_receipt.json"
    if not rows_path.is_file() or rows_path.read_bytes() != rows_text.encode():
        if rows_path.exists():
            _preserve_invalid(
                rows_path, out, "superseded-or-invalid-compilation")
        atomic_text(rows_path, rows_text)
    receipt = _compiled_receipt(
        manifest, out, ordered, completed_states, rows_text,
        invocation_runtime_s)
    if receipt_path.exists():
        try:
            existing = json.loads(receipt_path.read_text())
        except (OSError, ValueError, json.JSONDecodeError):
            existing = None
        if existing == receipt:
            return existing
        _preserve_invalid(receipt_path, out, "superseded-or-invalid-compilation")
    atomic_json(receipt_path, receipt)
    return receipt


def _build_smoke_branch_receipt(
        manifest: dict[str, Any], rows: list[dict[str, Any]], *,
        corpus_digest: str, replay_check: dict[str, Any]) -> dict[str, Any]:
    """Build the six-branch replay receipt from already validated rows."""

    state = manifest["states"][0]
    receipt = {
        "schema": "go2_scorer_fit_branch_smoke_receipt_v1_2",
        "status": STATUS,
        "pass": bool(
            len(rows) == 6 and all(row["valid"] for row in rows)
            and replay_check.get("exact_repeat") is True
            and all(len(row["context_frames"]) == 3
                    and len(row["horizon_frames"]) == 4 for row in rows)
        ),
        "state_id": state["state_id"],
        "state_identity_digest": state["state_identity_digest"],
        "branch_identity_digests": sorted(row["branch_identity_digest"]
                                           for row in rows),
        "branch_row_digests": sorted(row["branch_row_digest"] for row in rows),
        "state_manifest_digest": manifest["state_manifest_digest"],
        "corpus_bound_digests": _row_bindings(manifest),
        **_row_bindings(manifest),
        "scorer_contract_v1_2_digest": scorer_contract_digest(),
        "corpus_digest": corpus_digest,
        "render_contract_digest": manifest["render_contract_digest"],
        "textured_v03_renderer_contract_digest":
            manifest["textured_v03_renderer_contract_digest"],
        "preprocess_contract_digest": manifest["preprocess_contract_digest"],
        "preprocessing_digest": manifest["preprocessing_digest"],
        "target_encoder_digest": manifest["target_encoder_digest"],
        "target_encoder_checkpoint_sha256":
            manifest["target_encoder_checkpoint_sha256"],
        "replay_check": replay_check,
    }
    receipt["smoke_branch_receipt_digest"] = canonical_digest(receipt)
    return receipt


def _load_valid_smoke_branch_receipt(
        manifest: dict[str, Any], out: Path,
        rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate and reuse the original exact-replay proof on zero-new smoke."""

    path = out / "smoke_branch_receipt.json"
    if not path.is_file():
        raise RuntimeError(
            "six completed smoke rows exist without a replay receipt; refusing "
            "to downgrade or fabricate the exact-replay check"
        )
    try:
        receipt = json.loads(path.read_text())
        _verify_self_digest(
            receipt, "smoke_branch_receipt_digest", "branch smoke receipt")
        corpus_digest = str(receipt["corpus_digest"])
        if (len(corpus_digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in corpus_digest)):
            raise RuntimeError("branch smoke receipt corpus digest is malformed")
        replay_check = receipt["replay_check"]
        if (not isinstance(replay_check, dict)
                or replay_check.get("state_id") != manifest["states"][0]["state_id"]
                or replay_check.get("exact_repeat") is not True
                or replay_check.get("separate_render_scene_physically_inert") is not True):
            raise RuntimeError("branch smoke replay proof is missing or failed")
        matching = [row for row in rows
                    if row.get("candidate") == replay_check.get("candidate")]
        if (len(matching) != 1
                or matching[0].get("snapshot_digest")
                != replay_check.get("snapshot_digest")):
            raise RuntimeError("branch smoke replay proof no longer matches its row")
        expected = _build_smoke_branch_receipt(
            manifest, rows, corpus_digest=corpus_digest,
            replay_check=replay_check)
        if receipt != expected or receipt.get("pass") is not True:
            raise RuntimeError("branch smoke receipt differs from current rows")
        return receipt
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"branch smoke receipt validation failed: {exc}") from exc


def _final_identifiability_gate(manifest: dict[str, Any], out: Path,
                                receipt: dict[str, Any]) -> dict[str, Any] | None:
    if manifest["pool"] != "final_eval" or not receipt["complete"]:
        return None
    rows = [json.loads(line) for line in (out / "branch_rows.jsonl").read_text().splitlines()
            if line.strip()]
    statistics = V1.identifiability(rows)
    verdict = V1.gate_verdict(statistics)
    report = {
        "schema": "go2_final_evaluation_oracle_identifiability_gate_v1_2",
        "status": STATUS,
        "state_manifest_digest": manifest["state_manifest_digest"],
        "corpus_digest": receipt["corpus_digest"],
        "tie_tolerance": V1.TIE_TOLERANCE,
        "statistics": statistics,
        "gate": verdict,
        "predictor_checkpoint_loading_authorized": bool(verdict["pass"]),
    }
    report["final_gate_digest"] = canonical_digest(report)
    atomic_json(out / "final_gate.json", report)
    return report


def stage_branches(args: argparse.Namespace, *, smoke: bool = False) -> int:
    out = OUT_ROOT / args.pool
    if args.pool == "scorer_fit":
        manifest = load_active_state_manifest_for_consumption(
            out / "state_manifest.json")
    else:
        raw_manifest_path = out / "state_manifest.json"
        manifest_path = _pin_generated_path(
            raw_manifest_path, raw_manifest_path)
        manifest = json.loads(manifest_path.read_text())
        _validate_state_manifest(manifest, args.pool)
    if args.backend != "cpu":
        raise RuntimeError("the frozen qualified branch backend is cpu")
    if smoke and args.pool != "scorer_fit":
        raise RuntimeError("end-to-end smoke is defined only for scorer_fit")
    if not smoke:
        raw_smoke_path = OUT_ROOT / "scorer_fit/smoke_encoding_receipt.json"
        smoke_path = _pin_generated_path(raw_smoke_path, raw_smoke_path)
        if not smoke_path.is_file():
            raise RuntimeError("full branch generation is gated on the encoded six-branch smoke")
        smoke_receipt = json.loads(smoke_path.read_text())
        _verify_self_digest(smoke_receipt, "smoke_receipt_digest", "smoke receipt")
        scorer_fit_manifest = (
            manifest if args.pool == "scorer_fit"
            else load_active_state_manifest_for_consumption(
                OUT_ROOT / "scorer_fit/state_manifest.json")
        )
        if (not smoke_receipt.get("pass")
                or smoke_receipt.get("state_manifest_digest")
                != scorer_fit_manifest["state_manifest_digest"]
                or smoke_receipt.get("scorer_contract_v1_2_digest")
                != scorer_contract_digest()):
            raise RuntimeError("encoded smoke receipt is not valid for this scorer contract")

    completed = _completed_rows(manifest, out)
    states = ([manifest["states"][0]] if smoke else
              manifest["states"][args.state_offset:args.state_offset + args.state_limit])
    frames_dir = out / "frames"
    invocation_started = time.time()
    new_rows = 0
    replay_check: dict[str, Any] | None = None

    if smoke:
        state = manifest["states"][0]
        smoke_rows = [row for row in completed.values()
                      if row["state_id"] == state["state_id"]]
        if len(smoke_rows) == len(state["candidate_indices"]):
            # A zero-new smoke validates its durable proof and never starts
            # Genesis.  The compile call is itself byte-idempotent.
            receipt = _compile_corpus(
                manifest, out, time.time() - invocation_started)
            retained_smoke = _load_valid_smoke_branch_receipt(
                manifest, out, smoke_rows)
            if retained_smoke["corpus_digest"] != receipt["corpus_digest"]:
                retained_smoke = _build_smoke_branch_receipt(
                    manifest, smoke_rows,
                    corpus_digest=receipt["corpus_digest"],
                    replay_check=retained_smoke["replay_check"])
                atomic_json(out / "smoke_branch_receipt.json", retained_smoke)
            print(json.dumps({
                **retained_smoke,
                "recovery": "retained_valid_zero_new_replay_receipt",
            }, indent=2, sort_keys=True))
            return 0

    shared = None

    for entry in states:
        missing = [int(index) for index in entry["candidate_indices"]
                   if (entry["state_id"], int(index)) not in completed]
        if not missing:
            print(f"[branches] retain complete {entry['state_id']}", flush=True)
            continue
        state_started = time.time()
        print(f"[branches] {entry['state_id']} ({entry['scene_id']}); missing={missing}",
              flush=True)
        if shared is None:
            shared = V1._load_shared(args.backend)
        scene_dir = Path(entry["scene_dir"])
        scene_manifest_path = scene_dir / "manifest.json"
        if (file_sha256(scene_manifest_path) != entry["scene_manifest_sha256"]
                or scene_manifest_path.stat().st_size
                != int(entry["scene_manifest_byte_count"])):
            raise RuntimeError("registered scene manifest changed after identity freeze")
        ctx = V1.build_context(scene_dir, seed=int(entry["drive_seed"]),
                               backend=args.backend, shared=shared)
        topology = V12.link_topology(ctx)
        ctx.begin_episode()
        proprio_log: list[list[float]] = []
        control_log: list[list[float]] = []
        context_poses: list[BasePose] = []
        action_context_blocks: list[list[float]] = []
        warmup = int(entry["warmup_blocks"])

        def probe(_tick_idx: int, previous_applied: Sequence[float]) -> None:
            proprio_log.append(proprio_sample(ctx))
            control_log.append(control_sample(previous_applied))

        for block_index in range(warmup):
            driven = drive_block_with_probe(ctx, probe)
            if block_index >= warmup - CONTEXT_SLOTS:
                action_context_blocks.append(action_block_10d(
                    np.asarray(driven.executed, dtype=np.float64)[0]))
                context_poses.append(capture_base_pose(ctx))

        verdict = classify_state(
            ctx, topology,
            requested_stratum=(entry["stratum"]
                               if manifest["pool"] == "scorer_fit" else None))
        redrive_reason: str | None = None
        if isinstance(verdict, str):
            redrive_reason = f"redrive_failed:{verdict}"
        elif len(proprio_log) < PROPRIO_HISTORY:
            redrive_reason = "redrive_failed:short_proprio_history"
        else:
            record, field, _strata = verdict
            redrive_reason = _redrive_mismatch(entry, record, ctx)
        if redrive_reason is not None:
            for candidate_index in missing:
                identity = _identity_for(entry, candidate_index)
                row = _invalid_completed_row(
                    entry, identity, manifest, redrive_reason,
                    time.time() - state_started)
                _write_invalid_attempt_row(out, identity, row, redrive_reason)
            _compile_corpus(manifest, out, time.time() - invocation_started)
            del ctx
            gc.collect()
            raise RuntimeError(
                f"registered state {entry['state_id']} could not be redriven exactly: "
                f"{redrive_reason}")

        raw_manifest = json.loads(scene_manifest_path.read_text())
        import genesis as gs
        renderer = TexturedV03Renderer(ctx, gs=gs, raw_manifest=raw_manifest)
        if renderer.contract_digest != manifest["textured_v03_renderer_contract_digest"]:
            raise RuntimeError("runtime historical renderer contract changed")
        context_frames: list[dict[str, Any]] = []
        for slot, pose in enumerate(context_poses):
            result = renderer.render_pose(pose)
            path = (frames_dir / entry["family"]
                    / f"{entry['state_identity_digest']}_ctx{slot}.png")
            context_frames.append(_frame_receipt(
                result, path, out, index_key="slot", index_value=slot))
        proprio = np.asarray(proprio_log[-PROPRIO_HISTORY:], dtype=np.float32)
        control = np.asarray(control_log[-PROPRIO_HISTORY:], dtype=np.float32)
        if proprio.shape != (PROPRIO_HISTORY, 30) or control.shape != (
                PROPRIO_HISTORY, 2):
            raise RuntimeError(f"planning history shape changed: {proprio.shape}/{control.shape}")
        previous_applied = np.asarray(ctx.runner._last_executed,
                                      dtype=np.float64)[0].tolist()
        snapshot = V1.capture_branch_state(
            ctx, goal=entry["goal"],
            identity={
                "state_id": entry["state_id"],
                "state_identity_digest": entry["state_identity_digest"],
                "scene_id": entry["scene_id"],
                "family": entry["family"],
                "split": entry["split"],
                "block_index": warmup,
                "source_step": entry["source_step"],
                "episode_id": int(entry["episode_id"]),
            })

        for candidate_index in missing:
            identity = _identity_for(entry, candidate_index)
            candidate = V1.CANDIDATE_BANK[candidate_index]
            branch_started = time.time()
            horizon_poses: list[BasePose] = []
            requested_plan, post_slew_plan, action_plan_blocks = (
                candidate_planning_trajectory(candidate, previous_applied))

            def on_block_end(_block_index: int) -> None:
                horizon_poses.append(capture_base_pose(ctx))

            branch = _execute_and_render(ctx, snapshot, candidate, field=field,
                                         topology=topology,
                                         on_block_end=on_block_end)
            scored = V12.score_branch_v12(branch)
            actual_post_slew = branch["post_slew"]
            prefix_plan = post_slew_plan[:len(actual_post_slew)]
            if not np.allclose(np.asarray(actual_post_slew, dtype=np.float64),
                               np.asarray(prefix_plan, dtype=np.float64),
                               rtol=0.0, atol=1e-6):
                raise RuntimeError("runtime post-slew actions disagree with frozen planning reconstruction")
            invalid_reason = None
            if scored is None:
                invalid_reason = ("solver_nan" if branch["nan"]
                                  else "unlocatable_or_unreachable_geodesic")
            horizon_frames: list[dict[str, Any]] = []
            for horizon_index, pose in enumerate(horizon_poses, start=1):
                result = renderer.render_pose(pose)
                path = (frames_dir / entry["family"]
                        / f"{identity['branch_identity_digest']}_h{horizon_index}.png")
                horizon_frames.append(_frame_receipt(
                    result, path, out, index_key="horizon",
                    index_value=horizon_index))
            valid = bool(scored is not None and len(horizon_frames) == HORIZONS)
            if scored is not None and not valid:
                invalid_reason = "truncated_before_h4_render"
            row = {
                "schema": "go2_branch_corpus_v1_2_branch_row",
                "status": STATUS,
                "record_complete": True,
                "pool": args.pool,
                "state_id": entry["state_id"],
                "state_index": int(entry["state_index"]),
                "state_identity_digest": entry["state_identity_digest"],
                "branch_identity_digest": identity["branch_identity_digest"],
                "split_role": entry["split_role"],
                "stratum": entry["stratum"],
                "scene_id": entry["scene_id"],
                "family": entry["family"],
                "split": entry["split"],
                "episode_cluster_id": entry["episode_cluster_id"],
                "episode_id": int(snapshot.identity["episode_id"]),
                "source_step": int(snapshot.identity["source_step"]),
                "candidate": candidate[0],
                "candidate_index": int(candidate_index),
                "primitives": list(candidate[1]),
                "requested": requested_plan,
                "realised_requested_prefix": branch["requested"],
                "post_slew": branch["post_slew"],
                "candidate_post_slew_plan": post_slew_plan,
                "action_blocks": action_plan_blocks,
                "action_context_blocks": action_context_blocks,
                "previous_applied_command": previous_applied,
                "goal": entry["goal"],
                "goal_binding_input": [
                    math.sin(float(entry["goal"]["bearing_body_rad"])),
                    math.cos(float(entry["goal"]["bearing_body_rad"])),
                    float(entry["goal"]["range_m"]),
                ],
                "context_frames": context_frames,
                "horizon_frames": horizon_frames,
                "context_paths": [frame["path"] for frame in context_frames],
                "horizon_paths": [frame["path"] for frame in horizon_frames],
                "proprio": proprio.tolist(),
                "control": control.tolist(),
                "masks": {
                    "context_rgb_valid": [True] * CONTEXT_SLOTS,
                    "observed_proprio_valid": [True] * PROPRIO_HISTORY,
                    "observed_control_valid": [True] * PROPRIO_HISTORY,
                    "future_proprio_available": [False] * HORIZONS,
                    "target_rgb_valid": [index < len(horizon_frames)
                                         for index in range(HORIZONS)],
                },
                "timing": {
                    "command_hz": 10,
                    "ticks_per_block": 5,
                    "seconds_per_block": 0.5,
                    "context_boundary_offsets_blocks": [-2, -1, 0],
                    "target_horizons_blocks": [1, 2, 3, 4],
                },
                "valid": valid,
                "invalid_reason": invalid_reason,
                "blocks_completed": branch["blocks_completed"],
                "truncated_at_block": branch["truncated_at_block"],
                "snapshot_digest": snapshot.digest,
                "state_manifest_digest": manifest["state_manifest_digest"],
                **_row_bindings(manifest),
                "wall_time_s": round(time.time() - branch_started, 6),
                "storage_bytes": sum(frame["byte_count"] for frame in
                                     context_frames + horizon_frames),
            }
            row.update({key: (None if scored is None else scored[key]) for key in (
                "start_geodesic_m", "final_geodesic_m", "progress",
                "contact_fraction", "clearance_cost", "stuck_fraction", "fall",
                "safety", "completion", "utility", "min_clearance_m",
                "evaluation_points")})
            row["branch_row_digest"] = canonical_digest(row)
            _write_row(out, identity, row)
            completed[(entry["state_id"], candidate_index)] = row
            new_rows += 1

            if smoke and replay_check is None:
                repeat = _execute_and_render(ctx, snapshot, candidate, field=field,
                                             topology=topology, on_block_end=None)
                again = V12.score_branch_v12(repeat)
                replay_check = {
                    "state_id": entry["state_id"],
                    "candidate": candidate[0],
                    "snapshot_digest": snapshot.digest,
                    "exact_repeat": bool(
                        scored is not None and again is not None
                        and canonical_digest(scored) == canonical_digest(again)
                        and branch["requested"] == repeat["requested"]
                        and branch["post_slew"] == repeat["post_slew"]),
                    "separate_render_scene_physically_inert": True,
                }
        print(f"    done in {time.time() - state_started:.1f}s", flush=True)
        del renderer, ctx
        gc.collect()

    receipt = _compile_corpus(manifest, out, time.time() - invocation_started)
    if smoke:
        state = manifest["states"][0]
        rows = [row for row in _completed_rows(manifest, out).values()
                if row["state_id"] == state["state_id"]]
        if replay_check is None:
            raise RuntimeError("new smoke branches did not produce an exact-replay check")
        smoke_receipt = _build_smoke_branch_receipt(
            manifest, rows, corpus_digest=receipt["corpus_digest"],
            replay_check=replay_check)
        atomic_json(out / "smoke_branch_receipt.json", smoke_receipt)
        print(json.dumps(smoke_receipt, indent=2, sort_keys=True))
        return 0 if smoke_receipt["pass"] else 1

    if args.pool == "scorer_fit" and receipt["complete"]:
        # The exact-replay proof is first issued against the six-row smoke
        # corpus.  Rebind that unchanged proof once to the immutable complete
        # corpus so the full encoder can validate it fail-closed.
        state = manifest["states"][0]
        smoke_rows = [row for row in _completed_rows(manifest, out).values()
                      if row["state_id"] == state["state_id"]]
        prior_smoke = _load_valid_smoke_branch_receipt(
            manifest, out, smoke_rows)
        refreshed_smoke = _build_smoke_branch_receipt(
            manifest, smoke_rows, corpus_digest=receipt["corpus_digest"],
            replay_check=prior_smoke["replay_check"])
        if refreshed_smoke != prior_smoke:
            atomic_json(out / "smoke_branch_receipt.json", refreshed_smoke)

    gate = _final_identifiability_gate(manifest, out, receipt)
    summary = {
        "pool": args.pool,
        "new_rows": new_rows,
        "attempted_rows": receipt["attempted_branches"],
        "valid_rows": receipt["valid_branches"],
        "complete": receipt["complete"],
        "corpus_digest": receipt["corpus_digest"],
        "final_gate": None if gate is None else gate["gate"],
        "wall_time_s": round(time.time() - invocation_started, 3),
    }
    atomic_json(out / f"branch_summary_{args.state_offset}.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _execute_and_render(ctx, snapshot, candidate, *, field, topology, on_block_end):
    """V12.execute_branch_v12 with an optional per-block render hook."""

    from lewm_worlds.labels.derived import (DerivedLabelComputer, DerivedLabelConfig,
                                            PoseStep)
    V1.restore_branch_state(ctx, snapshot)
    runner = ctx.runner
    goal_cell = int(snapshot.goal["landmark_cell"])
    steps_per_tick = int(runner._policy_steps_per_command_tick)
    label_computer = DerivedLabelComputer(ctx.manifest, config=DerivedLabelConfig())
    episode_id = int(runner.episode_states[0].episode_id)
    state = {"episode_step": int(runner.episode_states[0].episode_step),
             "stamp_ns": int(runner._sim_time_ns)}

    def sample(executed_cmd):
        (x, y), yaw, z = ctx.pose()
        label = label_computer.step(PoseStep(
            timestamp_ns=int(state["stamp_ns"]), env_idx=0, episode_id=episode_id,
            episode_step=int(state["episode_step"]), position_xy_world=(x, y),
            yaw_world_rad=float(yaw),
            last_command=(float(executed_cmd[0]), float(executed_cmd[1]),
                          float(executed_cmd[2]))))
        flags = V1._termination_flags(ctx)
        hit = ctx.scene_graph.locate((x, y))
        located = bool(float(hit.distance_m) <= V1.LOCATE_MAX_DISTANCE_M)
        cell = int(hit.cell_id)
        return {"xy": [x, y], "yaw": yaw, "z": z, "cell_id": cell, "located": located,
                "geodesic_m": float(field.remaining_distance((x, y), cell)
                                    if located else math.inf),
                "at_goal_cell": bool(cell == goal_cell),
                "clearance_m": float(label.clearance_m),
                "stuck": bool(label.stuck_label),
                "disallowed_contacts": int(V12._contact_count(ctx, topology)),
                "terminated": bool(flags["fall"] or flags["out_of_bounds"]
                                   or flags["tipped"]),
                "nan": bool(flags["nan"])}

    start_row = sample(np.asarray(runner._last_executed, dtype=np.float64)[0])
    tick_rows, requested_all, executed_all = [], [], []
    truncated_at_block, nan_seen = None, False

    for block_idx, primitive in enumerate(candidate[1]):
        requested = V1.block_for(primitive)[None, ...]
        executed_block = np.asarray(
            runner._clip_block(np.asarray(requested, dtype=np.float32)).executed,
            dtype=np.float64)

        def after_policy_step(tick_idx, step_idx, _b=executed_block, _i=block_idx):
            if step_idx != steps_per_tick - 1:
                return
            state["episode_step"] += 1
            state["stamp_ns"] += int(runner._command_dt_ns)
            row = sample(_b[0, tick_idx])
            row["block"] = _i
            row["tick"] = int(tick_idx)
            tick_rows.append(row)

        block = runner.execute_requested_block(requested,
                                               after_policy_step=after_policy_step)
        requested_all.append(np.asarray(block.requested)[0].tolist())
        executed_all.append(np.asarray(block.executed)[0].tolist())
        ctx.ticks_executed += runner._block_size
        ctx.episode_ticks += runner._block_size
        ctx.policy_steps += runner._block_size * steps_per_tick
        ctx.last_block_executed = np.asarray(block.executed, dtype=np.float32).copy()
        if on_block_end is not None:
            on_block_end(block_idx)
        if tick_rows and tick_rows[-1]["nan"]:
            nan_seen, truncated_at_block = True, block_idx
            break
        if tick_rows and tick_rows[-1]["terminated"]:
            truncated_at_block = block_idx
            break

    return {"candidate": candidate[0], "primitives": list(candidate[1]),
            "requested": requested_all, "post_slew": executed_all,
            "blocks_completed": len(executed_all),
            "truncated_at_block": truncated_at_block, "nan": nan_seen,
            "start": start_row, "ticks": tick_rows}


def _branch_identity(state: dict[str, Any], candidate_index: int,
                     manifest_bindings: dict[str, Any]) -> dict[str, Any]:
    candidate = V1.CANDIDATE_BANK[candidate_index]
    payload = {
        "schema": "go2_branch_identity_v1_2",
        "pool": manifest_bindings["pool"],
        "state_id": state["state_id"],
        "state_identity_digest": state["state_identity_digest"],
        "scene_id": state["scene_id"],
        "episode_cluster_id": state["episode_cluster_id"],
        "source_step": state["source_step"],
        "goal": state["goal"],
        "candidate_index": int(candidate_index),
        "candidate": candidate[0],
        "primitives": list(candidate[1]),
        "candidate_allocation_amendment_digest":
            manifest_bindings["candidate_allocation_amendment_digest"],
        "candidate_allocation_post_identity_validation_digest":
            manifest_bindings[
                "candidate_allocation_post_identity_validation_digest"
            ],
        "pre_identity_allocation_validation_digest":
            manifest_bindings["pre_identity_allocation_validation_digest"],
        "invalid_scorer_identity_exclusion_digest":
            manifest_bindings["invalid_scorer_identity_exclusion_digest"],
        **{key: manifest_bindings[key] for key in ACTIVE_SELECTOR_BINDING_KEYS},
        **{key: manifest_bindings[key] for key in LAUNCH_BINDING_KEYS},
        "candidate_bank_digest": manifest_bindings["candidate_bank_digest"],
        "oracle_v1_2_digest": manifest_bindings["oracle_v1_2_digest"],
        "scorer_contract_v1_2_digest":
            manifest_bindings["scorer_contract_v1_2_digest"],
        "render_contract_digest": manifest_bindings["render_contract_digest"],
        "textured_v03_renderer_contract_digest":
            manifest_bindings["textured_v03_renderer_contract_digest"],
        "preprocess_contract_digest":
            manifest_bindings["preprocess_contract_digest"],
        "target_encoder_digest": manifest_bindings["target_encoder_digest"],
    }
    return {**payload, "branch_identity_digest": canonical_digest(payload)}


def _validate_state_resolution_transport(
        payload: dict[str, Any], *, expected_pool: str) -> None:
    transport = payload.get("state_resolution_subprocess_transport")
    provenance = payload.get("state_resolution_scene_capture_provenance")
    required_transport = {
        "schema", "one_scene_per_subprocess",
        "atomic_capture_write_before_native_cleanup",
        "return_code_ignored_only_after_valid_capture", "resume_scope",
        "resolver_algorithm_digest", "resolver_cursor_scene_id",
        "scene_capture_count", "scene_capture_provenance_digest",
        "candidate_outcomes_loaded",
    }
    if (
        not isinstance(transport, dict)
        or set(transport) != required_transport
        or transport.get("schema")
        != "go2_branch_corpus_v1_2_state_resolution_transport_v1"
        or transport.get("one_scene_per_subprocess") is not True
        or transport.get("atomic_capture_write_before_native_cleanup") is not True
        or transport.get("return_code_ignored_only_after_valid_capture") is not True
        or transport.get("resume_scope")
        != "MISSING_OR_INVALID_SCENE_CAPTURES_ONLY"
        or transport.get("resolver_algorithm_digest")
        != canonical_digest(STATE_RESOLUTION_REDUCER_CONTRACT)
        or transport.get("candidate_outcomes_loaded") is not False
        or not isinstance(provenance, list)
        or not provenance
        or transport.get("scene_capture_count") != len(provenance)
        or transport.get("scene_capture_provenance_digest")
        != canonical_digest(provenance)
    ):
        raise RuntimeError("state shard scene-resolution transport is malformed")

    family = str(payload["family"])
    out = OUT_ROOT / expected_pool
    live_pool, live_exclusion = scene_pool(expected_pool)
    if family not in live_pool:
        raise RuntimeError("state-resolution family is absent from live pool")
    live_args = argparse.Namespace(
        pool=expected_pool, family=family, backend="cpu")
    found: dict[str, int] | None = None
    required: dict[str, int] | None = None
    chosen_by_identity: dict[str, dict[str, Any]] = {}
    replayed_rejections: dict[str, dict[str, int]] = {}
    provenance_keys = {
        "scene_id", "state_resolution_scene_request_digest",
        "state_resolution_scene_capture_digest", "request_path",
        "request_raw_sha256", "request_byte_count", "capture_path",
        "capture_raw_sha256", "capture_byte_count",
    }
    for row_index, row in enumerate(provenance):
        if not isinstance(row, dict) or set(row) != provenance_keys:
            raise RuntimeError("state-resolution provenance row is malformed")
        request_digest = row.get("state_resolution_scene_request_digest")
        expected_request_path = _state_resolution_request_path(
            out, family, request_digest)
        expected_capture_path = _state_resolution_capture_path(
            out, family, request_digest)
        for raw_path, relative, sha_key, size_key, label in (
            (expected_request_path, row.get("request_path"),
             "request_raw_sha256", "request_byte_count", "request"),
            (expected_capture_path, row.get("capture_path"),
             "capture_raw_sha256", "capture_byte_count", "capture"),
        ):
            relative_path = ROOT / str(relative or "")
            path = _pin_generated_path(relative_path, raw_path)
            if (
                path.is_symlink() or not path.is_file()
                or relative != str(raw_path.relative_to(ROOT))
                or not _is_sha256(row.get(sha_key))
                or row.get(sha_key) != file_sha256(path)
                or row.get(size_key) != path.stat().st_size
            ):
                raise RuntimeError(
                    f"state-resolution {label} provenance bytes changed")
        pinned_request_path = _pin_generated_path(
            expected_request_path, expected_request_path)
        pinned_capture_path = _pin_generated_path(
            expected_capture_path, expected_capture_path)
        request = json.loads(pinned_request_path.read_text())
        capture = json.loads(pinned_capture_path.read_text())
        _validate_state_resolution_scene_request(
            request, args=live_args, out=out, pool=live_pool,
            exclusion=live_exclusion)
        _validate_state_resolution_scene_capture(
            capture, expected_request=request)
        bindings = request.get("state_shard_bindings")
        if (
            request.get("pool") != expected_pool
            or request.get("family") != family
            or request.get("scene_ordinal") != row_index
            or request.get("scene", {}).get("scene_id") != row.get("scene_id")
            or capture.get("scene_id") != row.get("scene_id")
            or capture.get("state_resolution_scene_capture_digest")
            != row.get("state_resolution_scene_capture_digest")
            or not isinstance(bindings, dict)
            or any(payload.get(key) != value for key, value in bindings.items())
        ):
            raise RuntimeError(
                "state-resolution request/capture differs from state shard")
        request_found = request.get("found_before_scene")
        request_required = request.get("required_counts")
        if found is None:
            found = {key: 0 for key in request_required}
            required = dict(request_required)
        if (
            request_found != found
            or request_required != required
            or all(found[key] >= required[key] for key in required)
            or capture.get("worker_failure") is not None
        ):
            raise RuntimeError(
                "state-resolution dynamic quota prefix changed")
        replayed_rejections[str(row["scene_id"])] = dict(
            capture["scene_rejection_reasons"])
        chosen = capture.get("chosen_state")
        if chosen is not None:
            stratum = str(chosen["stratum"])
            found[stratum] += 1
            identity = str(chosen["state_identity_digest"])
            if identity in chosen_by_identity:
                raise RuntimeError("state-resolution capture repeats an identity")
            chosen_by_identity[identity] = chosen
        quota_full = all(found[key] >= required[key] for key in required)
        is_final = row_index == len(provenance) - 1
        if quota_full != is_final:
            raise RuntimeError(
                "state-resolution cursor does not stop at first full quota")
    assert found is not None and required is not None
    if (
        found != required
        or transport.get("resolver_cursor_scene_id")
        != provenance[-1]["scene_id"]
        or replayed_rejections != payload.get("scene_rejection_reasons")
    ):
        raise RuntimeError("state-resolution reducer output changed")
    expected_captured_states = {
        str(state["state_identity_digest"]): state
        for state in payload.get("states", [])
        if not (family == REACHABILITY_REDRIVE_FAMILY
                and state.get("stratum") == "completion_enriched")
    }
    if chosen_by_identity != expected_captured_states:
        raise RuntimeError(
            "state-resolution captured identities differ from final state shard")


def _validate_state_shard(payload: dict[str, Any], path: Path,
                          expected_pool: str) -> None:
    _verify_self_digest(payload, "state_shard_digest", f"state shard {path.name}")
    spec = POOLS[expected_pool]
    expected_keys = {
        "schema", "status", "complete", "pool", "family", "spec",
        "selection", "selection_digest", "scorer_fit_allocation_design_digest",
        "candidate_allocator_contract_digest",
        "candidate_allocation_amendment_digest",
        "pre_identity_allocation_validation_digest",
        "invalid_scorer_identity_exclusion_digest",
        "state_selector_amendment_digest",
        "state_selector_feasibility_receipt_digest",
        *LAUNCH_BINDING_KEYS,
        "candidate_bank_digest", "progress_contract_digest",
        "safety_contract_digest", "oracle_v1_2_digest",
        "scorer_contract_v1_2_digest", "boundary_digest",
        "render_contract_digest", "textured_v03_renderer_contract_digest",
        "preprocess_contract_digest", "preprocessing_digest",
        "target_encoder_digest", "target_encoder_checkpoint_sha256",
        "genesis_backend", "exclusion_binding", "family_allow_list_digest",
        "states", "scene_rejection_reasons",
        "state_resolution_subprocess_transport",
        "state_resolution_scene_capture_provenance", "state_shard_digest",
    }
    if (expected_pool == "scorer_fit"
            and payload.get("family") == REACHABILITY_REDRIVE_FAMILY):
        expected_keys.update({
            "small_prefix_reissue_receipt",
            "small_completion_joint_allocation_search",
        })
    if set(payload) != expected_keys:
        raise RuntimeError(f"state shard {path.name} key surface changed")
    if (payload.get("schema") != "go2_branch_corpus_v1_2_state_shard"
            or payload.get("pool") != expected_pool
            or payload.get("complete") is not True
            or payload.get("spec") != spec
            or payload.get("selection") != SELECTION
            or payload.get("selection_digest") != selection_digest()
            or payload.get("candidate_allocator_contract_digest")
            != ALLOC.allocation_contract_digest()
            or payload.get("candidate_allocation_amendment_digest")
            != ALLOC.allocation_amendment_digest()
            or payload.get("pre_identity_allocation_validation_digest")
            != _load_pre_identity_allocation_validation()[
                "pre_identity_validation_digest"]
            or payload.get("invalid_scorer_identity_exclusion_digest")
            != INVALID_IDS.invalid_identity_exclusion_digest()
            or payload.get("state_selector_amendment_digest")
            != STATE_SELECTOR.state_selector_amendment_digest()
            or payload.get("state_selector_feasibility_receipt_digest")
            != _load_clean_source_launch_receipt()[
                "state_selector_feasibility_receipt_digest"]
            or payload.get("scorer_contract_v1_2_digest")
            != scorer_contract_digest()
            or payload.get("candidate_bank_digest") != V1.bank_digest()
            or payload.get("progress_contract_digest") != progress_digest()
            or payload.get("safety_contract_digest") != safety_digest()
            or payload.get("oracle_v1_2_digest") != v12_oracle_digest()
            or payload.get("render_contract_digest") != render_contract_digest()
            or payload.get("textured_v03_renderer_contract_digest")
            != textured_v03_renderer_contract_digest()
            or payload.get("preprocess_contract_digest")
            != preprocess_contract_digest()
            or payload.get("preprocessing_digest")
            != TARGET_ENCODER["preprocessing_identity_sha256"]
            or payload.get("target_encoder_digest") != target_encoder_digest()
            or payload.get("target_encoder_checkpoint_sha256")
            != TARGET_ENCODER["checkpoint_sha256"]
            or payload.get("boundary_digest") != V1.BOUNDARY_DIGEST
            or payload.get("genesis_backend") != "cpu"):
        raise RuntimeError(f"state shard {path.name} is bound to another contract")
    launch = _load_clean_source_launch_receipt()
    for key in LAUNCH_BINDING_KEYS:
        if payload.get(key) != launch[key]:
            raise RuntimeError(f"state shard {path.name} clean-source {key} mismatch")
    is_parallel_small = (
        expected_pool == "scorer_fit"
        and payload.get("family") == REACHABILITY_REDRIVE_FAMILY)
    if is_parallel_small:
        transport = payload.get("state_resolution_subprocess_transport")
        provenance = payload.get(
            "state_resolution_scene_capture_provenance")
        if (not isinstance(transport, dict)
                or set(transport) != {
                    "schema", "one_scene_per_subprocess",
                    "atomic_capture_write_before_native_cleanup",
                    "return_code_ignored_only_after_valid_capture",
                    "resume_scope", "resolver_algorithm_digest",
                    "resolver_cursor_scene_id", "scene_capture_count",
                    "scene_capture_provenance_digest",
                    "candidate_outcomes_loaded",
                }
                or transport.get("schema")
                != "go2_branch_corpus_v1_2_state_resolution_transport_v1"
                or transport.get("one_scene_per_subprocess") is not True
                or transport.get(
                    "atomic_capture_write_before_native_cleanup") is not True
                or transport.get(
                    "return_code_ignored_only_after_valid_capture") is not True
                or transport.get("resume_scope")
                != "REISSUED_EXACT_SMALL_PREFIX_PLUS_PARALLEL_CERTIFICATE"
                or transport.get("resolver_algorithm_digest")
                != canonical_digest(STATE_RESOLUTION_REDUCER_CONTRACT)
                or transport.get("resolver_cursor_scene_id")
                != payload.get("small_completion_joint_allocation_search", {}).get(
                    "resolver_cursor_scene_id")
                or not isinstance(provenance, list) or len(provenance) != 12
                or transport.get("scene_capture_count") != len(provenance)
                or transport.get("scene_capture_provenance_digest")
                != canonical_digest(provenance)
                or transport.get("candidate_outcomes_loaded") is not False
                or payload.get("small_prefix_reissue_receipt")
                != payload.get("small_completion_joint_allocation_search", {}).get(
                    "small_prefix_reissue_receipt")):
            raise RuntimeError("parallel small terminal transport changed")
    else:
        _validate_state_resolution_transport(payload, expected_pool=expected_pool)
    states = payload.get("states", [])
    if len(states) != spec["states_per_family"]:
        raise RuntimeError(f"state shard {path.name} has the wrong state count")
    if len({state["scene_id"] for state in states}) != len(states):
        raise RuntimeError(f"state shard {path.name} reuses a scene")
    invalid_identity_index = INVALID_IDS.load_invalid_identity_index()
    if payload.get("exclusion_binding", {}).get(
            "invalid_scorer_identity_attempt") != invalid_identity_index.binding():
        raise RuntimeError(f"state shard {path.name} invalid45 binding mismatch")
    INVALID_IDS.assert_disjoint(
        states, label=f"state shard {path.name}", index=invalid_identity_index)
    if any(_state_identity_digest(state) != state.get("state_identity_digest")
           for state in states):
        raise RuntimeError(f"state shard {path.name} has an identity digest mismatch")
    if is_parallel_small:
        inputs = _parallel_small_search_inputs(OUT_ROOT / "scorer_fit")
        if (provenance != inputs["prefix"]["capture_provenance"]
                or payload.get("scene_rejection_reasons")
                != inputs["prefix"]["scene_rejection_reasons"]
                or payload.get("small_prefix_reissue_receipt")
                != inputs["prefix"]["receipt_binding"]
                or [state for state in states
                    if state["stratum"] != "completion_enriched"]
                != sorted(inputs["prefix"]["states"], key=lambda state: (
                    STRATA.index(str(state["stratum"])),
                    str(state["scene_id"])))):
            raise RuntimeError("parallel small shard prefix receipt/replay changed")
        terminal_path = _pin_generated_path(
            OUT_ROOT / "scorer_fit" / PARALLEL_SMALL_TERMINAL_RESULT_NAME,
            OUT_ROOT / "scorer_fit" / PARALLEL_SMALL_TERMINAL_RESULT_NAME)
        if not terminal_path.is_file() or terminal_path.is_symlink():
            raise RuntimeError("parallel small shard terminal result is missing")
        terminal = json.loads(terminal_path.read_text())
        aggregate = _joint_state_order([*inputs["fixed_states"], *states])
        _validate_small_completion_joint_search_receipt(
            manifest={
                "states": aggregate,
                "small_prefix_reissue_receipt":
                    payload["small_prefix_reissue_receipt"],
                "small_completion_joint_allocation_search":
                    payload["small_completion_joint_allocation_search"],
            },
            allocation=dict(terminal.get("allocation", {})),
            replay_live=False)


def _validate_mixed_active_state_shard(
        payload: dict[str, Any], path: Path) -> None:
    """Replay retained anchors and every replacement-scene lexical prefix."""

    family = str(payload.get("family", ""))
    raw_expected_path = _mixed_active_state_shard_path(
        OUT_ROOT / "scorer_fit", family)
    pinned_path = _pin_generated_path(path, raw_expected_path)
    if not pinned_path.is_file() or pinned_path.is_symlink():
        raise RuntimeError("mixed active state shard path is not a regular file")
    if json.loads(pinned_path.read_text()) != payload:
        raise RuntimeError("mixed active state shard differs from pinned bytes")
    _verify_self_digest(
        payload, "state_shard_digest", f"mixed shard {raw_expected_path.name}")
    expected_keys = {
        "schema", "status", "complete", "pool", "family", "spec",
        "selection", "selection_digest", "scorer_fit_allocation_design_digest",
        "candidate_allocator_contract_digest",
        "candidate_allocation_amendment_digest",
        "pre_identity_allocation_validation_digest",
        "invalid_scorer_identity_exclusion_digest",
        "state_selector_amendment_digest",
        "state_selector_feasibility_receipt_digest", *LAUNCH_BINDING_KEYS,
        "candidate_bank_digest", "progress_contract_digest",
        "safety_contract_digest", "oracle_v1_2_digest",
        "scorer_contract_v1_2_digest", "boundary_digest",
        "render_contract_digest", "textured_v03_renderer_contract_digest",
        "preprocess_contract_digest", "preprocessing_digest",
        "target_encoder_digest", "target_encoder_checkpoint_sha256",
        "genesis_backend", "exclusion_binding", "family_allow_list_digest",
        "predecessor_state_shard_binding",
        "retained_predecessor_identity_digests",
        "rejected_predecessor_identity_digests", "replacement_slot_fills",
        "states", "scene_rejection_reasons",
        "mixed_replacement_subprocess_transport",
        "mixed_replacement_scene_capture_provenance", "state_shard_digest",
    }
    if (
        set(payload) != expected_keys
        or payload.get("schema") != MIXED_ACTIVE_STATE_SHARD_SCHEMA
        or payload.get("status") != STATUS
        or payload.get("complete") is not True
        or payload.get("pool") != "scorer_fit"
        or payload.get("spec") != POOLS["scorer_fit"]
        or payload.get("selection") != SELECTION
        or payload.get("selection_digest") != selection_digest()
        or payload.get("state_selector_amendment_digest")
        != STATE_SELECTOR.state_selector_amendment_digest()
        or payload.get("scorer_contract_v1_2_digest") != scorer_contract_digest()
        or payload.get("genesis_backend") != "cpu"
    ):
        raise RuntimeError("mixed active state shard contract changed")
    plan = _mixed_family_replacement_plan(family)
    predecessor_binding = next(
        dict(row) for row in STATE_SELECTOR.PRESERVED_STATE_SHARDS
        if row["family"] == family)
    predecessor_raw_path = ROOT / predecessor_binding["path"]
    predecessor_path = _pin_generated_path(
        predecessor_raw_path, predecessor_raw_path)
    if (
        payload.get("predecessor_state_shard_binding") != predecessor_binding
        or not predecessor_path.is_file() or predecessor_path.is_symlink()
        or file_sha256(predecessor_path) != predecessor_binding["raw_sha256"]
        or predecessor_path.stat().st_size != predecessor_binding["byte_count"]
    ):
        raise RuntimeError("mixed active predecessor shard binding changed")
    pool, exclusion = scene_pool("scorer_fit")
    scenes = pool[family]
    args = argparse.Namespace(pool="scorer_fit", family=family, backend="cpu")
    bindings = _state_shard_bindings(
        args, exclusion, [scene.name for scene in scenes])
    if any(payload.get(key) != value for key, value in bindings.items()):
        raise RuntimeError("mixed active state shard binding changed")
    if (
        payload.get("retained_predecessor_identity_digests")
        != sorted(row["state_identity_digest"] for row in plan["retained_states"])
        or payload.get("rejected_predecessor_identity_digests")
        != plan["rejected_identity_digests"]
    ):
        raise RuntimeError("mixed active disposition changed")
    transport = payload.get("mixed_replacement_subprocess_transport")
    provenance = payload.get("mixed_replacement_scene_capture_provenance")
    if (
        not isinstance(transport, dict)
        or transport.get("schema") != MIXED_REPLACEMENT_TRANSPORT_SCHEMA
        or transport.get("one_scene_per_subprocess") is not True
        or transport.get("atomic_capture_write_before_native_cleanup") is not True
        or transport.get("return_code_ignored_only_after_valid_capture") is not True
        or transport.get("resume_scope")
        != "MISSING_OR_INVALID_REPLACEMENT_SCENE_CAPTURES_ONLY"
        or transport.get("candidate_outcomes_loaded") is not False
        or not isinstance(provenance, list)
        or transport.get("scene_capture_count") != len(provenance)
        or transport.get("scene_capture_provenance_digest")
        != canonical_digest(provenance)
        or not isinstance(transport.get("interval_rows"), list)
        or len(transport["interval_rows"]) != len(plan["interval_groups"])
    ):
        raise RuntimeError("mixed replacement transport changed")

    retained_all, rejected_all, _slots = _mixed_disposition_sets()
    retained_scene_ids = {str(row["scene_id"]) for row in retained_all.values()}
    provenance_cursor = 0
    replacements: list[dict[str, Any]] = []
    replayed_rejections: dict[str, dict[str, int]] = {}
    replayed_interval_rows: list[dict[str, Any]] = []
    for interval_index, interval in enumerate(plan["interval_groups"]):
        candidates = _mixed_replacement_candidate_scenes(
            scenes=scenes, interval=interval,
            retained_scene_ids=retained_scene_ids)
        accepted: list[dict[str, Any]] = []
        scanned_scene_ids: list[str] = []
        while len(accepted) < len(interval["replacement_slots"]):
            candidate_index = len(scanned_scene_ids)
            if candidate_index >= len(candidates):
                raise RuntimeError("mixed active shard claims an exhausted interval passed")
            scene_ordinal, scene_dir = candidates[candidate_index]
            if provenance_cursor >= len(provenance):
                raise RuntimeError("mixed active shard omits a replacement capture")
            row = provenance[provenance_cursor]
            provenance_cursor += 1
            slot = interval["replacement_slots"][len(accepted)]
            request_raw_path = ROOT / str(row.get("request_path", ""))
            capture_raw_path = ROOT / str(row.get("capture_path", ""))
            expected_request_raw_path = _mixed_replacement_request_path(
                OUT_ROOT / "scorer_fit", family,
                str(row.get("mixed_replacement_scene_request_digest", "")))
            expected_capture_raw_path = _mixed_replacement_capture_path(
                OUT_ROOT / "scorer_fit", family,
                str(row.get("mixed_replacement_scene_request_digest", "")))
            request_path = _pin_generated_path(
                request_raw_path, expected_request_raw_path)
            capture_path = _pin_generated_path(
                capture_raw_path, expected_capture_raw_path)
            if (
                not request_path.is_file() or request_path.is_symlink()
                or not capture_path.is_file() or capture_path.is_symlink()
                or row.get("request_raw_sha256") != file_sha256(request_path)
                or row.get("request_byte_count") != request_path.stat().st_size
                or row.get("capture_raw_sha256") != file_sha256(capture_path)
                or row.get("capture_byte_count") != capture_path.stat().st_size
            ):
                raise RuntimeError("mixed replacement provenance bytes changed")
            request = json.loads(request_path.read_text())
            _validate_mixed_replacement_scene_request(
                request, args=args, out=path.parent, pool=pool,
                exclusion=exclusion)
            if (
                request.get("scene_ordinal") != scene_ordinal
                or request.get("scene", {}).get("scene_id") != scene_dir.name
                or request.get("replacement_slot") != slot
                or request.get("accepted_scene_ids_before")
                != [state["scene_id"] for state in accepted]
            ):
                raise RuntimeError("mixed replacement lexical request prefix changed")
            capture = json.loads(capture_path.read_text())
            _validate_mixed_replacement_scene_capture(
                capture, expected_request=request)
            expected_row = _mixed_capture_provenance(
                out=OUT_ROOT / "scorer_fit", request=request, capture=capture,
                interval_index=interval_index)
            if row != expected_row or capture.get("worker_failure") is not None:
                raise RuntimeError("mixed replacement capture provenance changed")
            scanned_scene_ids.append(scene_dir.name)
            replayed_rejections[scene_dir.name] = dict(
                capture["scene_rejection_reasons"])
            if capture["chosen_state"] is not None:
                accepted.append(dict(capture["chosen_state"]))
                replacements.append(dict(capture["chosen_state"]))
        replayed_interval_rows.append({
            "interval_index": interval_index,
            "lower_scene_id_exclusive": interval["lower_scene_id_exclusive"],
            "upper_scene_id_exclusive": interval["upper_scene_id_exclusive"],
            "vacant_ordinals": list(interval["vacant_ordinals"]),
            "replacement_slot_state_ids": [
                row["state_id"] for row in interval["replacement_slots"]],
            "candidate_scene_ids": [scene.name for _ordinal, scene in candidates],
            "scanned_scene_ids": scanned_scene_ids,
            "selected_scene_ids": [state["scene_id"] for state in accepted],
            "stopped_at_first_complete_prefix": True,
        })
    if provenance_cursor != len(provenance):
        raise RuntimeError("mixed active shard appends post-quota captures")
    states = sorted(
        [dict(row) for row in plan["retained_states"]] + replacements,
        key=lambda row: (STRATA.index(str(row["stratum"])), str(row["state_id"])))
    replacement_fills = [{
        "state_id": row["state_id"],
        "state_identity_digest": row["state_identity_digest"],
        "scene_id": row["scene_id"],
        "split_role": row["split_role"],
    } for row in sorted(replacements, key=lambda value: value["state_id"])]
    completion = sorted(
        (row for row in states if row["stratum"] == "completion_enriched"),
        key=lambda row: _completion_state_ordinal(str(row["state_id"])))
    if (
        payload.get("states") != states
        or payload.get("replacement_slot_fills") != replacement_fills
        or payload.get("scene_rejection_reasons") != replayed_rejections
        or transport.get("interval_rows") != replayed_interval_rows
        or len(states) != 15
        or len({row["scene_id"] for row in states}) != 15
        or any(row["state_identity_digest"] in rejected_all for row in states)
        or [row["scene_id"] for row in completion]
        != sorted(row["scene_id"] for row in completion)
    ):
        raise RuntimeError("mixed active state shard differs from live replay")
    INVALID_IDS.assert_disjoint(states, label=f"mixed state shard {family}")


def _assert_archived_fixed_transport_pair(
        *, kind: str, provenance: Mapping[str, Any],
        request_row: Mapping[str, Any], capture_row: Mapping[str, Any],
        old_request: Mapping[str, Any], old_capture: Mapping[str, Any]) -> None:
    """Bind one archived request/capture pair back to its shard provenance."""

    request_key = (
        "mixed_replacement_scene_request_digest" if kind == "mixed" else
        "state_resolution_scene_request_digest")
    capture_key = (
        "mixed_replacement_scene_capture_digest" if kind == "mixed" else
        "state_resolution_scene_capture_digest")
    if (
        request_row.get("kind") != "request"
        or capture_row.get("kind") != "capture"
        or request_row.get("path") != provenance.get("request_path")
        or capture_row.get("path") != provenance.get("capture_path")
        or request_row.get("raw_sha256")
        != provenance.get("request_raw_sha256")
        or request_row.get("byte_count") != provenance.get("request_byte_count")
        or capture_row.get("raw_sha256")
        != provenance.get("capture_raw_sha256")
        or capture_row.get("byte_count") != provenance.get("capture_byte_count")
        or old_request.get(request_key) != provenance.get(request_key)
        or old_capture.get(capture_key) != provenance.get(capture_key)
        or old_capture.get(request_key) != old_request.get(request_key)
        or old_capture.get("request") != old_request
    ):
        raise RuntimeError("archived fixed transport provenance changed")


def _replay_projected_ordinary_fixed_shard(
        *, predecessor: Mapping[str, Any], family: str,
        projected_pairs: Sequence[Mapping[str, Any]]) -> None:
    """Reconstruct the ordinary lexical quota reducer entirely in memory."""

    provenance = predecessor.get("state_resolution_scene_capture_provenance")
    transport = predecessor.get("state_resolution_subprocess_transport")
    expected_transport_keys = {
        "schema", "one_scene_per_subprocess",
        "atomic_capture_write_before_native_cleanup",
        "return_code_ignored_only_after_valid_capture", "resume_scope",
        "resolver_algorithm_digest", "resolver_cursor_scene_id",
        "scene_capture_count", "scene_capture_provenance_digest",
        "candidate_outcomes_loaded",
    }
    if (not isinstance(provenance, list) or not provenance
            or len(projected_pairs) != len(provenance)
            or not isinstance(transport, Mapping)
            or set(transport) != expected_transport_keys
            or transport.get("schema")
            != "go2_branch_corpus_v1_2_state_resolution_transport_v1"
            or transport.get("one_scene_per_subprocess") is not True
            or transport.get("atomic_capture_write_before_native_cleanup")
            is not True
            or transport.get("return_code_ignored_only_after_valid_capture")
            is not True
            or transport.get("resume_scope")
            != "MISSING_OR_INVALID_SCENE_CAPTURES_ONLY"
            or transport.get("resolver_algorithm_digest")
            != canonical_digest(STATE_RESOLUTION_REDUCER_CONTRACT)
            or transport.get("candidate_outcomes_loaded") is not False
            or transport.get("scene_capture_count") != len(provenance)
            or transport.get("scene_capture_provenance_digest")
            != canonical_digest(provenance)):
        raise RuntimeError("ordinary archived reducer transport changed")

    required = dict(POOLS["scorer_fit"]["strata"])
    found = {name: 0 for name in required}
    selected: list[dict[str, Any]] = []
    rejections: dict[str, dict[str, int]] = {}
    identities: set[str] = set()
    for ordinal, (row, projected) in enumerate(
            zip(provenance, projected_pairs, strict=True)):
        request = dict(projected["request"])
        capture = dict(projected["capture"])
        old_request = dict(projected["old_request"])
        old_capture = dict(projected["old_capture"])
        _assert_archived_fixed_transport_pair(
            kind="ordinary", provenance=row,
            request_row=projected["request_row"],
            capture_row=projected["capture_row"],
            old_request=old_request, old_capture=old_capture)
        requested = [name for name in STRATA
                     if found[name] < required[name]]
        scene_id = str(request.get("scene", {}).get("scene_id", ""))
        if (request.get("scene_ordinal") != ordinal
                or row.get("scene_id") != scene_id
                or capture.get("scene_id") != scene_id
                or request.get("found_before_scene") != found
                or request.get("required_counts") != required
                or request.get("requested_strata_in_priority_order")
                != requested
                or not requested
                or capture.get("worker_failure") is not None):
            raise RuntimeError("ordinary archived dynamic quota prefix changed")
        rejections[scene_id] = dict(capture["scene_rejection_reasons"])
        chosen = capture.get("chosen_state")
        if chosen is not None:
            chosen = dict(chosen)
            stratum = str(chosen["stratum"])
            if stratum not in requested or found[stratum] >= required[stratum]:
                raise RuntimeError("ordinary archived reducer exceeded a quota")
            found[stratum] += 1
            identity = str(chosen["state_identity_digest"])
            if identity in identities:
                raise RuntimeError("ordinary archived reducer repeats an identity")
            identities.add(identity)
            selected.append(chosen)
        quota_full = found == required
        if quota_full != (ordinal == len(provenance) - 1):
            raise RuntimeError(
                "ordinary archived reducer did not stop at first full quota")

    selected.sort(key=lambda state: (
        STRATA.index(str(state["stratum"])), str(state["scene_id"])))
    if (found != required
            or transport.get("resolver_cursor_scene_id")
            != provenance[-1].get("scene_id")
            or predecessor.get("states") != selected
            or predecessor.get("scene_rejection_reasons") != rejections):
        raise RuntimeError("ordinary archived reducer output changed")


def _replay_projected_mixed_fixed_shard(
        *, predecessor: Mapping[str, Any], family: str,
        projected_pairs: Sequence[Mapping[str, Any]]) -> None:
    """Reconstruct every mixed interval, slot, and first-complete prefix."""

    provenance = predecessor.get("mixed_replacement_scene_capture_provenance")
    transport = predecessor.get("mixed_replacement_subprocess_transport")
    expected_transport_keys = {
        "schema", "one_scene_per_subprocess",
        "atomic_capture_write_before_native_cleanup",
        "return_code_ignored_only_after_valid_capture", "resume_scope",
        "interval_rows", "scene_capture_count",
        "scene_capture_provenance_digest", "candidate_outcomes_loaded",
    }
    if (not isinstance(provenance, list)
            or len(projected_pairs) != len(provenance)
            or not isinstance(transport, Mapping)
            or set(transport) != expected_transport_keys
            or transport.get("schema") != MIXED_REPLACEMENT_TRANSPORT_SCHEMA
            or transport.get("one_scene_per_subprocess") is not True
            or transport.get("atomic_capture_write_before_native_cleanup")
            is not True
            or transport.get("return_code_ignored_only_after_valid_capture")
            is not True
            or transport.get("resume_scope")
            != "MISSING_OR_INVALID_REPLACEMENT_SCENE_CAPTURES_ONLY"
            or transport.get("candidate_outcomes_loaded") is not False
            or transport.get("scene_capture_count") != len(provenance)
            or transport.get("scene_capture_provenance_digest")
            != canonical_digest(provenance)):
        raise RuntimeError("mixed archived reducer transport changed")

    plan = _mixed_family_replacement_plan(family)
    pool, _exclusion = scene_pool("scorer_fit")
    scenes = pool[family]
    retained_all, rejected_all, _slots = _mixed_disposition_sets()
    retained_scene_ids = {str(row["scene_id"])
                          for row in retained_all.values()}
    cursor = 0
    replacements: list[dict[str, Any]] = []
    rejections: dict[str, dict[str, int]] = {}
    interval_rows: list[dict[str, Any]] = []
    for interval_index, interval in enumerate(plan["interval_groups"]):
        candidates = _mixed_replacement_candidate_scenes(
            scenes=scenes, interval=interval,
            retained_scene_ids=retained_scene_ids)
        accepted: list[dict[str, Any]] = []
        scanned_scene_ids: list[str] = []
        while len(accepted) < len(interval["replacement_slots"]):
            candidate_index = len(scanned_scene_ids)
            if candidate_index >= len(candidates) or cursor >= len(provenance):
                raise RuntimeError("mixed archived interval ends before quota")
            scene_ordinal, scene_dir = candidates[candidate_index]
            row = provenance[cursor]
            projected = projected_pairs[cursor]
            cursor += 1
            request = dict(projected["request"])
            capture = dict(projected["capture"])
            old_request = dict(projected["old_request"])
            old_capture = dict(projected["old_capture"])
            _assert_archived_fixed_transport_pair(
                kind="mixed", provenance=row,
                request_row=projected["request_row"],
                capture_row=projected["capture_row"],
                old_request=old_request, old_capture=old_capture)
            slot = interval["replacement_slots"][len(accepted)]
            accepted_scene_ids = [state["scene_id"] for state in accepted]
            if (row.get("interval_index") != interval_index
                    or row.get("scene_id") != scene_dir.name
                    or row.get("replacement_slot_state_id")
                    != slot["state_id"]
                    or request.get("scene_ordinal") != scene_ordinal
                    or request.get("scene", {}).get("scene_id")
                    != scene_dir.name
                    or request.get("anchor_interval") != {
                        "lower_scene_id_exclusive":
                            interval["lower_scene_id_exclusive"],
                        "upper_scene_id_exclusive":
                            interval["upper_scene_id_exclusive"],
                        "vacant_ordinals": list(interval["vacant_ordinals"]),
                    }
                    or request.get("replacement_slot") != slot
                    or request.get("accepted_scene_ids_before")
                    != accepted_scene_ids
                    or capture.get("scene_id") != scene_dir.name
                    or capture.get("worker_failure") is not None
                    or row.get("selected")
                    != (capture.get("chosen_state") is not None)):
                raise RuntimeError("mixed archived lexical interval prefix changed")
            scanned_scene_ids.append(scene_dir.name)
            rejections[scene_dir.name] = dict(
                capture["scene_rejection_reasons"])
            chosen = capture.get("chosen_state")
            if chosen is not None:
                chosen = dict(chosen)
                if chosen["state_identity_digest"] in rejected_all:
                    raise RuntimeError(
                        "mixed archived reducer restored a rejected identity")
                accepted.append(chosen)
                replacements.append(chosen)
        interval_rows.append({
            "interval_index": interval_index,
            "lower_scene_id_exclusive": interval[
                "lower_scene_id_exclusive"],
            "upper_scene_id_exclusive": interval[
                "upper_scene_id_exclusive"],
            "vacant_ordinals": list(interval["vacant_ordinals"]),
            "replacement_slot_state_ids": [
                row["state_id"] for row in interval["replacement_slots"]],
            "candidate_scene_ids": [
                scene.name for _ordinal, scene in candidates],
            "scanned_scene_ids": scanned_scene_ids,
            "selected_scene_ids": [state["scene_id"] for state in accepted],
            "stopped_at_first_complete_prefix": True,
        })
    if cursor != len(provenance):
        raise RuntimeError("mixed archived reducer appends post-quota captures")

    states = [dict(row) for row in plan["retained_states"]] + replacements
    states.sort(key=lambda row: (
        STRATA.index(str(row["stratum"])), str(row["state_id"])))
    replacement_fills = [{
        "state_id": row["state_id"],
        "state_identity_digest": row["state_identity_digest"],
        "scene_id": row["scene_id"],
        "split_role": row["split_role"],
    } for row in sorted(replacements, key=lambda value: value["state_id"])]
    if (predecessor.get("states") != states
            or predecessor.get("replacement_slot_fills") != replacement_fills
            or predecessor.get("scene_rejection_reasons") != rejections
            or transport.get("interval_rows") != interval_rows
            or predecessor.get("retained_predecessor_identity_digests")
            != sorted(row["state_identity_digest"]
                      for row in plan["retained_states"])
            or predecessor.get("rejected_predecessor_identity_digests")
            != plan["rejected_identity_digests"]):
        raise RuntimeError("mixed archived reducer output changed")


def _revalidate_performance_interrupted_fixed_shard(
        predecessor: dict[str, Any], transport_rows: Sequence[dict[str, Any]],
        successor_bindings: dict[str, Any]) -> bool:
    """Replay archived request/capture semantics under successor bindings.

    Archived bytes remain immutable.  Each request is projected in memory onto
    the current complete state-shard binding, its digest is recomputed, and its
    paired capture is projected onto that exact request before the existing
    structural/scientific validators run.  Only after every row and the final
    selected-state projection pass may the lineage module issue a successor
    wrapper.
    """

    family = str(predecessor.get("family", ""))
    kind = next(
        (row["kind"] for row in PERFORMANCE_INTERRUPTION.FIXED_STATE_SHARDS
         if row["family"] == family), None)
    if kind not in {"ordinary", "mixed"}:
        raise RuntimeError("performance-interrupted shard family changed")
    pool, exclusion = scene_pool("scorer_fit")
    scenes = pool.get(family)
    if scenes is None:
        raise RuntimeError("performance-interrupted family left scene pool")
    args = argparse.Namespace(pool="scorer_fit", family=family, backend="cpu")
    expected_bindings = _state_shard_bindings(
        args, exclusion, [path.name for path in scenes])
    if any(expected_bindings.get(key) != successor_bindings.get(key)
           for key in PERFORMANCE_INTERRUPTION.SUCCESSOR_LINEAGE_KEYS):
        raise RuntimeError("successor replay lineage differs from live bindings")

    by_logical = {str(row["path"]): dict(row) for row in transport_rows}
    provenance_key = (
        "mixed_replacement_scene_capture_provenance" if kind == "mixed" else
        "state_resolution_scene_capture_provenance")
    provenance = predecessor.get(provenance_key)
    if not isinstance(provenance, list) or len(by_logical) != 2 * len(provenance):
        raise RuntimeError("archived transport/provenance cardinality changed")
    projected_pairs: list[dict[str, Any]] = []
    for pair in provenance:
        request_row = by_logical.get(str(pair.get("request_path", "")))
        capture_row = by_logical.get(str(pair.get("capture_path", "")))
        if request_row is None or capture_row is None:
            raise RuntimeError("archived transport pair is incomplete")
        request_path = PERFORMANCE_INTERRUPTION._pin_managed(
            request_row["archive_path"], root=ROOT)
        capture_path = PERFORMANCE_INTERRUPTION._pin_managed(
            capture_row["archive_path"], root=ROOT)
        request = json.loads(request_path.read_text())
        capture = json.loads(capture_path.read_text())
        old_request = dict(request)
        old_capture = dict(capture)
        old_bindings = old_request.get("state_shard_bindings")
        if (not isinstance(old_bindings, dict)
                or set(old_bindings) != set(expected_bindings)):
            raise RuntimeError(
                "archived request state-shard binding surface changed")
        for key in expected_bindings:
            if key in PERFORMANCE_INTERRUPTION.SUCCESSOR_LINEAGE_KEYS:
                continue
            if old_bindings[key] != expected_bindings[key]:
                raise RuntimeError(
                    f"archived request nonlineage binding {key} changed")
        request["state_shard_bindings"] = expected_bindings
        request_key = (
            "mixed_replacement_scene_request_digest" if kind == "mixed" else
            "state_resolution_scene_request_digest")
        request.pop(request_key, None)
        request[request_key] = canonical_digest(request)
        capture["request"] = request
        capture[request_key] = request[request_key]
        capture_key = (
            "mixed_replacement_scene_capture_digest" if kind == "mixed" else
            "state_resolution_scene_capture_digest")
        capture.pop(capture_key, None)
        capture[capture_key] = canonical_digest(capture)
        # The in-memory projection is lineage-only: every nonbinding request
        # field and every non-nested-request capture field remains exact.
        old_request_projection = dict(old_request)
        old_request_projection.pop("state_shard_bindings")
        old_request_projection.pop(request_key)
        new_request_projection = dict(request)
        new_request_projection.pop("state_shard_bindings")
        new_request_projection.pop(request_key)
        if old_request_projection != new_request_projection:
            raise RuntimeError("request scientific projection changed")
        old_capture_projection = dict(old_capture)
        old_capture_projection.pop("request")
        old_capture_projection.pop(request_key)
        old_capture_projection.pop(capture_key)
        new_capture_projection = dict(capture)
        new_capture_projection.pop("request")
        new_capture_projection.pop(request_key)
        new_capture_projection.pop(capture_key)
        if old_capture_projection != new_capture_projection:
            raise RuntimeError("capture scientific projection changed")
        if kind == "mixed":
            _validate_mixed_replacement_scene_request(
                request, args=args, out=OUT_ROOT / "scorer_fit",
                pool=pool, exclusion=exclusion)
            _validate_mixed_replacement_scene_capture(
                capture, expected_request=request)
        else:
            _validate_state_resolution_scene_request(
                request, args=args, out=OUT_ROOT / "scorer_fit",
                pool=pool, exclusion=exclusion)
            _validate_state_resolution_scene_capture(
                capture, expected_request=request)
        if capture.get("worker_failure") is not None:
            raise RuntimeError("archived fixed transport contains worker failure")
        projected_pairs.append({
            "request": request,
            "capture": capture,
            "old_request": old_request,
            "old_capture": old_capture,
            "request_row": request_row,
            "capture_row": capture_row,
        })

    if kind == "ordinary":
        _replay_projected_ordinary_fixed_shard(
            predecessor=predecessor, family=family,
            projected_pairs=projected_pairs)
    else:
        _replay_projected_mixed_fixed_shard(
            predecessor=predecessor, family=family,
            projected_pairs=projected_pairs)
    return True


def _performance_successor_bindings() -> dict[str, Any]:
    """Return the sole current seven-key lineage projection."""

    launch = _load_clean_source_launch_receipt()
    return {
        key: (scorer_contract_digest()
              if key == "scorer_contract_v1_2_digest" else launch[key])
        for key in PERFORMANCE_INTERRUPTION.SUCCESSOR_LINEAGE_KEYS
    }


def stage_reissue_performance_interrupted_fixed_shards() -> int:
    """Reissue seven exact pre-outcome shards after current contract/launch."""

    source = clean_source_binding()
    receipt = (
        PERFORMANCE_INTERRUPTION
        .load_and_validate_performance_interruption_receipt(
            expected_source_repository_commit=str(
                source["source_repository_commit"]),
            expected_clean_source_binding_digest=canonical_digest(source),
            expected_bound_implementations_digest=str(
                source["bound_implementations_digest"]),
            root=ROOT,
        )
    )

    outputs = PERFORMANCE_INTERRUPTION.reissue_fixed_state_shards(
        receipt=receipt,
        revalidate_predecessor=
            _revalidate_performance_interrupted_fixed_shard,
        build_successor_bindings=_performance_successor_bindings,
        outcome_surface_absent=lambda:
            _phase1_outcome_surface_absence_attestation(root=ROOT),
        root=ROOT,
    )
    print(json.dumps({
        "status": PERFORMANCE_INTERRUPTION.REISSUED_SHARD_STATUS,
        "families": sorted(outputs),
        "source_reissued_state_shard_digests": {
            family: payload["source_reissued_state_shard_digest"]
            for family, payload in sorted(outputs.items())
        },
        "scientific_selection_changed": False,
    }, indent=2, sort_keys=True))
    return 0


def stage_reissue_performance_interrupted_small_prefix() -> int:
    """Reissue the exact 12-pair, outcome-free 5G/5S prefix.

    This stage follows contract and launch issuance.  The resulting receipt is
    deliberately *not* added to the launch receipt: it is bound by the search
    plan, terminal joint receipt, small-family shard, and final state manifest.
    """

    receipt = _load_current_performance_interruption_receipt()
    reissue = PERFORMANCE_INTERRUPTION.reissue_small_fixed_prefix(
        performance_receipt=receipt,
        build_successor_bindings=_performance_successor_bindings,
        revalidate_prefix=_revalidate_reissued_small_prefix,
        outcome_surface_absent=lambda:
            _phase1_outcome_surface_absence_attestation(root=ROOT),
        root=ROOT,
    )
    binding = PERFORMANCE_INTERRUPTION.small_prefix_reissue_receipt_binding(
        reissue, root=ROOT)
    print(json.dumps({
        "status": reissue["status"],
        "small_prefix_reissue_receipt": binding,
        "selected_state_projection_digest":
            reissue["selected_state_projection_digest"],
        "resolver_cursor_scene_id": reissue["resolver_cursor_scene_id"],
        "candidate_outcomes_consumed": False,
    }, indent=2, sort_keys=True))
    return 0


def _outcome_generation_started(out: Path) -> bool:
    lexical_root = Path(out)
    pinned_out = _pin_generated_path(lexical_root, lexical_root)
    if pinned_out.is_symlink() or not pinned_out.is_dir():
        raise RuntimeError("outcome surface pool root is not a directory")

    def pin(relative: Path) -> Path:
        pinned = pinned_out / relative
        _assert_unsealed_path(pinned)
        return pinned

    def guarded_tree_has_file(relative: Path) -> bool:
        pinned_root = pin(relative)
        if not pinned_root.exists():
            return False
        if pinned_root.is_symlink():
            raise RuntimeError("outcome surface root is symlinked")
        if pinned_root.is_file():
            return True
        if not pinned_root.is_dir():
            raise RuntimeError("outcome surface root is not regular")
        stack: list[tuple[Path, Path]] = [(relative, pinned_root)]
        while stack:
            logical_directory, pinned_directory = stack.pop()
            for entry in sorted(
                    pinned_directory.iterdir(), key=lambda row: row.name):
                if entry.is_symlink():
                    raise RuntimeError("outcome surface descendant is symlinked")
                logical_entry = logical_directory / entry.name
                pinned_entry = pin(logical_entry)
                if pinned_entry != entry:
                    raise RuntimeError("outcome surface path identity changed")
                if entry.is_file():
                    return True
                if entry.is_dir():
                    stack.append((logical_entry, entry))
                else:
                    raise RuntimeError("outcome surface node is not regular")
        return False

    return bool(
        any(pin(Path(name)).exists() for name in (
            "branch_rows.jsonl", "corpus_receipt.json", "latents_index.json"))
        or guarded_tree_has_file(Path("row_records"))
        or guarded_tree_has_file(Path("frames"))
    )


def _completion_states_for_phase2(
        *, allocation: dict[str, Any], states: Sequence[dict[str, Any]],
        preserved_vectors: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Project exactly 40 allocated completion rows for the combined gate."""

    assigned = {
        str(row["state_identity_digest"]): row
        for row in allocation["assignments"]
    }
    completion_states: list[dict[str, Any]] = []
    for state in states:
        if state["stratum"] != "completion_enriched":
            continue
        assignment = assigned.get(str(state["state_identity_digest"]))
        if assignment is None:
            raise RuntimeError("completion state is absent from candidate allocation")
        vector = _state_completion_rotation_vector(state, preserved_vectors)
        rotation = int(assignment["rotation_index"])
        rotations = vector.get("rotations")
        if (not isinstance(rotations, list) or len(rotations) != 12
                or rotations[rotation].get("candidate_indices")
                != assignment["candidate_indices"]):
            raise RuntimeError("completion rotation evidence/allocation mismatch")
        previous = rotations[rotation].get("previous_applied_command")
        if previous is None:
            raise RuntimeError("completion state lacks previous applied command")
        completion_states.append({
            "state_identity_digest": state["state_identity_digest"],
            "state_id": state["state_id"],
            "family": state["family"],
            "stratum": state["stratum"],
            "candidate_indices": list(assignment["candidate_indices"]),
            "previous_applied_command": list(previous),
            "completion_eligibility": dict(rotations[rotation]),
        })
    if len(completion_states) != 40:
        raise RuntimeError("phase 2 requires exact evidence for 40 completion states")
    return completion_states


def _issue_preserved_state_revalidation(
        out: Path, allocation: dict[str, Any],
        states: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Issue phase 2 only after all 120 identities receive frozen masks."""

    launch = _load_clean_source_launch_receipt()
    completion_states = _completion_states_for_phase2(
        allocation=allocation, states=states,
        preserved_vectors=_phase1_completion_rotation_vectors())
    certify_solve_free = lambda supplied: \
        _certify_parallel_allocation_solve_free(out, supplied)
    expected = STATE_SELECTOR.build_preserved_state_revalidation_receipt_from_solve_free_certified_allocation(
        allocation_manifest=allocation,
        active_states=states,
        completion_states=completion_states,
        certify_allocation_solve_free=certify_solve_free,
        source_repository_commit=str(launch["source_repository_commit"]),
        successor_selection_digest=selection_digest(),
        state_selector_feasibility_receipt_digest=str(
            launch["state_selector_feasibility_receipt_digest"]),
        mixed_precontract_disposition_receipt_digest=str(
            launch["mixed_precontract_disposition_receipt_digest"]),
        root=ROOT,
    )
    raw_path = ROOT / STATE_SELECTOR.PRESERVED_STATE_REVALIDATION_RECEIPT_PATH
    if raw_path.parent != out:
        raise RuntimeError("final preserved-state receipt path escaped scorer-fit pool")
    path = _pin_generated_path(raw_path, raw_path)
    if path.is_file():
        existing = json.loads(path.read_text())
        try:
            STATE_SELECTOR.validate_preserved_state_revalidation_receipt_from_solve_free_certified_allocation(
                existing,
                allocation_manifest=allocation,
                active_states=states,
                certify_allocation_solve_free=certify_solve_free,
                expected_source_commit=str(launch["source_repository_commit"]),
                expected_successor_selection_digest=selection_digest(),
                expected_feasibility_receipt_digest=str(
                    launch["state_selector_feasibility_receipt_digest"]),
                expected_mixed_precontract_disposition_receipt_digest=str(
                    launch["mixed_precontract_disposition_receipt_digest"]),
                root=ROOT,
            )
        except Exception as exc:
            if _outcome_generation_started(out):
                raise RuntimeError(
                    "final preserved-state revalidation changed after outcomes"
                ) from exc
            _preserve_invalid(path, out, "post-allocation-revalidation-invalid")
        else:
            if existing == expected:
                return existing
            if _outcome_generation_started(out):
                raise RuntimeError(
                    "final preserved-state revalidation changed after outcomes"
                )
            _preserve_invalid(path, out, "post-allocation-revalidation-mismatch")
    atomic_json(path, expected)
    return expected


def _build_state_shard_provenance(
        paths: Sequence[Path], shards: Sequence[dict[str, Any]],
        evidence: Sequence[dict[str, Any]] | None = None, *, pool_name: str,
        ) -> list[dict[str, Any]]:
    """Bind active envelope bytes and the validated successor shard separately."""

    if evidence is None:
        derived_evidence: list[dict[str, Any]] = []
        for path, shard in zip(paths, shards, strict=True):
            raw = _active_state_shard_path(
                OUT_ROOT / pool_name, str(shard["family"]), pool=pool_name)
            derived_evidence.append({
                "envelope_schema": str(shard.get("schema", "")),
                "active_path": str(raw.relative_to(ROOT)),
                "active_raw_sha256": file_sha256(path),
                "active_byte_count": path.stat().st_size,
                "source_reissued_state_shard_digest": None,
                "predecessor_state_shard_digest": None,
                "performance_interruption_receipt_digest": None,
                "successor_state_shard_digest": str(
                    shard["state_shard_digest"]),
            })
        evidence = derived_evidence
    by_family = {str(shard["family"]): (path, shard, envelope)
                 for path, shard, envelope in zip(
                     paths, shards, evidence, strict=True)}
    mixed_families = {str(row["family"])
                      for row in STATE_SELECTOR.PRESERVED_STATE_SHARDS}
    rows: list[dict[str, Any]] = []
    for family in sorted(by_family):
        path, shard, envelope = by_family[family]
        raw_path = _active_state_shard_path(
            OUT_ROOT / pool_name, family, pool=pool_name)
        expected_path = _pin_generated_path(raw_path, raw_path)
        if path != expected_path:
            raise RuntimeError(
                "active state-shard canonical path changed before provenance")
        row = {
            "family": family,
            "path": str(raw_path.relative_to(ROOT)),
            "state_shard_digest": str(shard["state_shard_digest"]),
            "raw_sha256": file_sha256(path),
            "byte_count": path.stat().st_size,
            "active_envelope": dict(envelope),
            "selection_provenance": (
                "MIXED_37_RETAINED_8_REPLACED_SELECTOR_AMENDMENT_V2"
                if pool_name == "scorer_fit" and family in mixed_families
                else "SUCCESSOR_SELECTOR_AMENDMENT_V2"
            ),
        }
        rows.append(row)
    if len(rows) != EXPECTED_FAMILIES:
        raise RuntimeError("mixed state-shard provenance must cover eight families")
    return rows


def _validate_state_shard_provenance(
        manifest: dict[str, Any], *, pool: str) -> None:
    rows = manifest.get("state_shard_provenance")
    if not isinstance(rows, list) or len(rows) != EXPECTED_FAMILIES:
        raise RuntimeError("state manifest lacks eight-row shard provenance")
    mixed_families = {str(row["family"])
                      for row in STATE_SELECTOR.PRESERVED_STATE_SHARDS}
    seen: set[str] = set()
    observed_digests: dict[str, str] = {}
    for row in rows:
        family = str(row.get("family", ""))
        if family in seen:
            raise RuntimeError("state-shard provenance repeats a family")
        seen.add(family)
        raw_path = ROOT / str(row.get("path", ""))
        path = _frozen_generated_artifact_path(raw_path)
        expected_path = _frozen_generated_artifact_path(
            _active_state_shard_path(
                OUT_ROOT / pool, family, pool=pool))
        if path != expected_path or not path.is_file() or path.is_symlink():
            raise RuntimeError("state-shard provenance path changed or escapes root")
        loaded_path, payload, envelope = _load_active_state_shard_evidence(
            OUT_ROOT / pool, family, pool=pool)
        expected_provenance = (
            "MIXED_37_RETAINED_8_REPLACED_SELECTOR_AMENDMENT_V2"
            if pool == "scorer_fit" and family in mixed_families
            else "SUCCESSOR_SELECTOR_AMENDMENT_V2"
        )
        if (loaded_path != path
                or row.get("selection_provenance") != expected_provenance
                or row.get("raw_sha256") != file_sha256(path)
                or row.get("byte_count") != path.stat().st_size
                or row.get("active_envelope") != envelope
                or row.get("state_shard_digest")
                != payload.get("state_shard_digest")):
            raise RuntimeError(f"state-shard provenance failed for {family}")
        observed_digests[family] = str(row["state_shard_digest"])
    if (seen != set(manifest.get("state_shard_digests", {}))
            or observed_digests != manifest.get("state_shard_digests")):
        raise RuntimeError("state-shard digest map and provenance disagree")


def merge_states(out: Path) -> int:
    """Merge exactly eight completed shards and freeze all branch identities."""

    pool_name = out.name
    if pool_name not in POOLS:
        raise RuntimeError(f"unknown output pool {pool_name!r}")
    paths: list[Path] = []
    shards: list[dict[str, Any]] = []
    shard_evidence: list[dict[str, Any]] = []
    for family in STATE_SELECTOR.REQUIRED_FAMILIES:
        path, payload, evidence = _load_active_state_shard_evidence(
            out, family, pool=pool_name)
        paths.append(path)
        shards.append(payload)
        shard_evidence.append(evidence)
    families = [str(shard["family"]) for shard in shards]
    if len(set(families)) != EXPECTED_FAMILIES:
        raise RuntimeError("state shards do not represent eight unique families")

    states = [dict(state) for shard in shards for state in shard["states"]]
    states.sort(key=lambda state: (
        state["family"],
        STRATA.index(state["stratum"]) if state["stratum"] in STRATA else 0,
        state["scene_id"],
    ))
    spec = POOLS[pool_name]
    expected_states = EXPECTED_FAMILIES * spec["states_per_family"]
    if len(states) != expected_states:
        raise RuntimeError(f"expected {expected_states} states, found {len(states)}")
    if len({state["scene_id"] for state in states}) != len(states):
        raise RuntimeError("merged state identities are not scene-disjoint")
    if len({state["episode_cluster_id"] for state in states}) != len(states):
        raise RuntimeError("merged state identities are not episode-cluster-disjoint")
    if len({state["state_identity_digest"] for state in states}) != len(states):
        raise RuntimeError("merged state identity digests are not unique")
    if any(not _state_identity_matches_active_or_preserved(state)
           for state in states):
        raise RuntimeError("merged state identities changed across selector phases")
    for index, state in enumerate(states):
        state["state_index"] = index

    common_keys = STATE_SHARD_COMMON_KEYS
    active_shards = shards
    common = {key: active_shards[0][key] for key in common_keys}
    for shard in active_shards[1:]:
        if any(shard[key] != common[key] for key in common_keys):
            raise RuntimeError("state shards contain mixed contract bindings")

    pre_allocation_payload = {
        "schema": "go2_branch_corpus_v1_2_pre_allocation_identity_manifest",
        "pool": pool_name,
        "spec": spec,
        **common,
        "state_identities": [{
            "state_id": state["state_id"],
            "state_identity_digest": state["state_identity_digest"],
            "family": state["family"],
            "stratum": state["stratum"],
            "split_role": state["split_role"],
            "goal_type": state["goal_type"],
        } for state in states],
    }
    pre_allocation_digest = canonical_digest(pre_allocation_payload)

    raw_allocation_path = out / "candidate_allocation_manifest.json"
    allocation_path = _pin_generated_path(
        raw_allocation_path, raw_allocation_path)
    allocation_digest: str
    if pool_name == "scorer_fit":
        small_shard = next(
            (shard for shard in shards
             if shard["family"] == REACHABILITY_REDRIVE_FAMILY), None)
        joint_search = (None if small_shard is None else
                        small_shard.get(
                            "small_completion_joint_allocation_search"))
        if not isinstance(joint_search, dict):
            raise RuntimeError(
                "scorer-fit allocation lacks the small-last joint search receipt")
        raw_terminal = out / PARALLEL_SMALL_TERMINAL_RESULT_NAME
        terminal_path = _pin_generated_path(raw_terminal, raw_terminal)
        if not terminal_path.is_file() or terminal_path.is_symlink():
            raise RuntimeError("parallel terminal result is missing at merge")
        terminal = json.loads(terminal_path.read_text())
        allocation = dict(terminal.get("allocation", {}))
        if allocation.get("source_identity_manifest_digest") \
                != pre_allocation_digest:
            raise RuntimeError(
                "certified allocation source identity digest changed at merge")
        _validate_small_completion_joint_search_receipt(
            manifest={
                "states": states,
                "small_prefix_reissue_receipt":
                    small_shard.get("small_prefix_reissue_receipt"),
                "small_completion_joint_allocation_search": joint_search,
            },
            allocation=allocation,
            replay_live=False)
        if allocation_path.exists() and (
                not allocation_path.is_file() or allocation_path.is_symlink()):
            raise RuntimeError("candidate allocation path is not a regular file")
        if allocation_path.is_file():
            existing = json.loads(allocation_path.read_text())
            if existing != allocation:
                if _outcome_generation_started(out):
                    raise RuntimeError("candidate allocation changed after branch generation")
                _preserve_invalid(allocation_path, out, "allocation-mismatch")
        atomic_json(allocation_path, allocation)
        allocation_digest = allocation["allocation_manifest_digest"]
        assigned = {row["state_id"]: row for row in allocation["assignments"]}
        for state in states:
            state["candidate_indices"] = list(assigned[state["state_id"]][
                "candidate_indices"])
            state["candidate_rotation_index"] = int(assigned[state["state_id"]][
                "rotation_index"])
    else:
        allocation = {
            "schema": "go2_final_eval_all_candidate_allocation_v1_2",
            "source_identity_manifest_digest": pre_allocation_digest,
            "candidate_bank_digest": V1.bank_digest(),
            "assignments": [{
                "state_id": state["state_id"],
                "state_identity_digest": state["state_identity_digest"],
                "candidate_indices": list(range(len(V1.CANDIDATE_BANK))),
            } for state in states],
        }
        allocation["allocation_manifest_digest"] = canonical_digest(allocation)
        if allocation_path.exists() and (
                not allocation_path.is_file() or allocation_path.is_symlink()):
            raise RuntimeError("candidate allocation path is not a regular file")
        if allocation_path.is_file():
            existing = json.loads(allocation_path.read_text())
            if existing != allocation:
                if _outcome_generation_started(out):
                    raise RuntimeError("final allocation changed after branch generation")
                _preserve_invalid(allocation_path, out, "allocation-mismatch")
        atomic_json(allocation_path, allocation)
        allocation_digest = allocation["allocation_manifest_digest"]
        for state in states:
            state["candidate_indices"] = list(range(len(V1.CANDIDATE_BANK)))

    post_identity_validation_digest = (
        allocation.get("post_identity_pre_outcome_validation", {}).get(
            "post_identity_validation_digest"
        )
    )
    if pool_name == "scorer_fit" and not isinstance(
            post_identity_validation_digest, str):
        raise RuntimeError(
            "scorer-fit allocation lacks post-identity/pre-outcome validation"
        )
    if pool_name == "scorer_fit":
        preserved_revalidation = _issue_preserved_state_revalidation(
            out, allocation, states)
    else:
        raw_preserved_path = (
            ROOT / STATE_SELECTOR.PRESERVED_STATE_REVALIDATION_RECEIPT_PATH
        )
        preserved_path = _pin_generated_path(
            raw_preserved_path, raw_preserved_path)
        if not preserved_path.is_file():
            raise RuntimeError(
                "final-evaluation identities require scorer-fit phase-2 revalidation"
            )
        preserved_revalidation = json.loads(preserved_path.read_text())
    manifest_bindings = {
        "pool": pool_name,
        **common,
        "candidate_allocation_post_identity_validation_digest":
            post_identity_validation_digest,
        "preserved_state_revalidation_receipt_digest":
            preserved_revalidation[
                "preserved_state_revalidation_receipt_digest"
            ],
    }
    branch_identities: list[dict[str, Any]] = []
    candidate_counts = {name: 0 for name, _sequence in V1.CANDIDATE_BANK}
    for state in states:
        identities = [_branch_identity(state, int(candidate_index), manifest_bindings)
                      for candidate_index in state["candidate_indices"]]
        state["branch_identities"] = identities
        branch_identities.extend(identities)
        for identity in identities:
            candidate_counts[identity["candidate"]] += 1
    expected_branches = expected_states * spec["candidates_per_state"]
    if len(branch_identities) != expected_branches:
        raise RuntimeError("registered branch identity count mismatch")
    branch_digests = [row["branch_identity_digest"] for row in branch_identities]
    if len(set(branch_digests)) != expected_branches:
        raise RuntimeError("registered branch identities are not unique")
    invalid_identity_disjointness = INVALID_IDS.assert_disjoint(
        states,
        label="merged pre-outcome state and branch identities",
        index=INVALID_IDS.load_invalid_identity_index(),
    )

    # Predecessor exclusion bindings are provenance only.  The active corpus
    # exclusion is the one shared by all five successor shards.
    exclusion_bindings = [shard["exclusion_binding"] for shard in active_shards]
    if any(value != exclusion_bindings[0] for value in exclusion_bindings[1:]):
        raise RuntimeError("state shards disagree on exclusions")
    rejections = {shard["family"]: shard["scene_rejection_reasons"]
                  for shard in shards}
    shard_provenance = _build_state_shard_provenance(
        paths, shards, shard_evidence, pool_name=pool_name)
    manifest = {
        "schema": "go2_branch_corpus_v1_2_state_manifest",
        "status": STATUS,
        "complete": True,
        "pool": pool_name,
        "spec": spec,
        "selection": SELECTION,
        **common,
        "pre_allocation_identity_manifest_digest": pre_allocation_digest,
        "candidate_allocation_manifest_digest": allocation_digest,
        "candidate_allocation_post_identity_validation_digest":
            post_identity_validation_digest,
        "preserved_state_revalidation_receipt_digest":
            preserved_revalidation[
                "preserved_state_revalidation_receipt_digest"
            ],
        "branch_identity_set_digest": canonical_digest(sorted(branch_digests)),
        "exclusion_binding": exclusion_bindings[0],
        "state_shard_digests": {shard["family"]: shard["state_shard_digest"]
                                for shard in shards},
        "state_shard_provenance": shard_provenance,
        "states": states,
        "candidate_appearances": candidate_counts,
        "attempted_branch_count_registered": expected_branches,
        "disjointness": {
            "state_count": expected_states,
            "unique_scene_count": len({state["scene_id"] for state in states}),
            "unique_episode_cluster_count": len({state["episode_cluster_id"]
                                                  for state in states}),
            "unique_state_identity_count": len({state["state_identity_digest"]
                                                 for state in states}),
            "unique_branch_identity_count": len(set(branch_digests)),
            "scene_episode_state_branch_disjoint": True,
            "invalid_scorer_identity_attempt": invalid_identity_disjointness,
        },
        "scene_rejection_reasons": rejections,
        "recovery_provenance": {
            "frozen_phase1_checked_state_identity_count": 45,
            "active_retained_predecessor_state_identity_count": 37,
            "active_replacement_state_identity_count": 8,
            "mixed_precontract_disposition_receipt_digest":
                _load_clean_source_launch_receipt()[
                    "mixed_precontract_disposition_receipt_digest"],
            "frozen_failed_precontract_receipt_binding": dict(
                STATE_SELECTOR.FROZEN_PRESERVED_PRECONTRACT_FAILURE),
            "preserved_state_revalidation_receipt_digest":
                preserved_revalidation[
                    "preserved_state_revalidation_receipt_digest"
                ],
            "valid_predecessor_state_shards": [
                dict(row) for row in STATE_SELECTOR.PRESERVED_STATE_SHARDS
            ],
            "interrupted_attempt_witnesses": [
                witness["path"] for witness in
                INVALID_IDS.INVALID_SCORER_IDENTITY_EXCLUSION["witnesses"]
            ],
            "invalid_scorer_identity_exclusion_digest":
                INVALID_IDS.invalid_identity_exclusion_digest(),
            "invalid_scene_ids_digest": (
                INVALID_IDS.INVALID_SCORER_IDENTITY_EXCLUSION[
                    "derived_identity_bindings"]["scene_ids_digest"]
            ),
            "superseded_pre_run_contract_artifact":
                SUPERSEDED_PRE_RUN_CONTRACT_ARTIFACT,
            "invalid_attempt_decision": (
                "preserved the incomplete three-of-eight-family 45-state "
                "pre-outcome identity attempt under its superseded contract and "
                "selection; no branch, render, latent or outcome existed; the "
                "exact 45 scenes and all descendant identity namespaces are "
                "excluded from this corpus; no invalid artifact is mixed"
            ),
            "valid_paused_identity_decision": (
                "retained 37 byte-bound passing identities from the separate "
                "45-state pre-outcome phase-1 terminal and replaced only its "
                "eight authorized completion slots under exact lexical anchors; "
                "the active 120-state allocation and exact six-candidate masks "
                "were verified by phase 2 before branch identities were issued; "
                "the failed receipt and all predecessor shards remain immutable "
                "provenance only"
            ),
        },
    }
    if pool_name == "scorer_fit":
        manifest["small_completion_joint_allocation_search"] = joint_search
        manifest["small_prefix_reissue_receipt"] = \
            small_shard["small_prefix_reissue_receipt"]
    manifest["state_manifest_digest"] = canonical_digest(manifest)
    raw_manifest_path = out / "state_manifest.json"
    manifest_path = _pin_generated_path(
        raw_manifest_path, raw_manifest_path)
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        if existing != manifest:
            if _outcome_generation_started(out):
                raise RuntimeError("state identity manifest changed after branch generation")
            _preserve_invalid(manifest_path, out, "identity-mismatch")
    atomic_json(manifest_path, manifest)

    from collections import Counter
    print(json.dumps({
        "state_manifest_digest": manifest["state_manifest_digest"],
        "pre_allocation_identity_manifest_digest": pre_allocation_digest,
        "candidate_allocation_manifest_digest": allocation_digest,
        "states": len(states),
        "branches": len(branch_identities),
        "per_family": dict(Counter(state["family"] for state in states)),
        "per_stratum": dict(Counter(state["stratum"] for state in states)),
        "split_roles": dict(Counter(state["split_role"] for state in states)),
        "candidate_appearances": candidate_counts,
    }, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", choices=sorted(POOLS), required=True)
    parser.add_argument("--stage",
                        choices=["allocation-preflight", "selector-feasibility",
                                 "selector-reachability-feasibility",
                                 "record-performance-interruption",
                                 "reissue-performance-fixed-states",
                                 "reissue-performance-small-prefix",
                                 "small-completion-benchmark",
                                 "small-completion-search",
                                 "revalidate-preserved",
                                 "states", "merge-states", "smoke", "branches"],
                        required=True)
    parser.add_argument("--family", default=None,
                        help="resolve one family only; shards merge via merge-states")
    parser.add_argument("--stratum", choices=STRATA, default=None,
                        help="scope selector-feasibility diagnostics to one stratum")
    parser.add_argument(
        "--selector-scene-id", default=None,
        help=argparse.SUPPRESS)
    parser.add_argument(
        "--preserved-state-identity-digest", default=None,
        help=argparse.SUPPRESS)
    parser.add_argument(
        "--phase1-outcome-surface-attestation-digest", default=None,
        help=argparse.SUPPRESS)
    parser.add_argument(
        "--state-resolution-scene-request-digest", default=None,
        help=argparse.SUPPRESS)
    parser.add_argument(
        "--mixed-replacement-scene-request-digest", default=None,
        help=argparse.SUPPRESS)
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--state-offset", type=int, default=0)
    parser.add_argument("--state-limit", type=int, default=10**6)
    args = parser.parse_args()
    if args.state_offset < 0 or args.state_limit < 1:
        raise SystemExit("state offset must be nonnegative and limit positive")
    if (args.selector_scene_id is not None
            and args.stage not in {
                "selector-feasibility", "selector-reachability-feasibility"}):
        raise SystemExit(
            "--selector-scene-id is internal to --stage selector-feasibility")
    if ((args.preserved_state_identity_digest is not None
         or args.phase1_outcome_surface_attestation_digest is not None)
            and args.stage != "revalidate-preserved"):
        raise SystemExit(
            "phase-1 worker arguments are internal to revalidate-preserved")
    if (args.state_resolution_scene_request_digest is not None
            and (args.stage != "states" or args.family is None)):
        raise SystemExit(
            "state-resolution scene request is internal to family states")
    if (args.mixed_replacement_scene_request_digest is not None
            and (args.stage != "states" or args.pool != "scorer_fit"
                 or args.family is None)):
        raise SystemExit(
            "mixed replacement request is internal to scorer-fit family states")
    if (args.state_resolution_scene_request_digest is not None
            and args.mixed_replacement_scene_request_digest is not None):
        raise SystemExit("only one internal state-scene worker may be requested")

    out = OUT_ROOT / args.pool
    # Validate the one permitted generated-root alias before even a nominal
    # mkdir traverses it.  Keep ``out`` lexical for all later custody bindings;
    # the resolved path is used only for this guarded directory creation.
    pinned_out = _pin_generated_path(out, out)
    pinned_out.mkdir(parents=True, exist_ok=True)
    if not pinned_out.is_dir() or pinned_out.is_symlink():
        raise SystemExit("managed pool output path is not a regular directory")
    if args.stage == "allocation-preflight":
        return issue_pre_identity_allocation_validation(out)
    if args.stage == "selector-feasibility":
        return stage_selector_feasibility(args)
    if args.stage == "selector-reachability-feasibility":
        return stage_selector_reachability_feasibility(args)
    if args.stage == "record-performance-interruption":
        if args.pool != "scorer_fit" or args.family is not None:
            raise SystemExit(
                "performance interruption lineage is scorer_fit pool-wide")
        return stage_small_search_performance_interruption()
    if args.stage == "reissue-performance-fixed-states":
        if args.pool != "scorer_fit" or args.family is not None:
            raise SystemExit(
                "fixed-state reissue is scorer_fit pool-wide")
        return stage_reissue_performance_interrupted_fixed_shards()
    if args.stage == "reissue-performance-small-prefix":
        if args.pool != "scorer_fit" or args.family is not None:
            raise SystemExit(
                "small-prefix reissue is scorer_fit pool-wide")
        return stage_reissue_performance_interrupted_small_prefix()
    if args.stage == "small-completion-benchmark":
        if args.pool != "scorer_fit" or args.family is not None:
            raise SystemExit(
                "small completion benchmark is scorer_fit pool-wide")
        return stage_parallel_small_completion_benchmark()
    if args.stage == "small-completion-search":
        if args.pool != "scorer_fit" or args.family is not None:
            raise SystemExit(
                "small completion search is scorer_fit pool-wide")
        return stage_parallel_small_completion_search()
    if args.stage == "revalidate-preserved":
        return stage_preserved_state_precontract_revalidation(args)
    if args.stage == "merge-states":
        return merge_states(out)
    if args.stage == "states":
        if args.family is None:
            raise SystemExit("--stage states requires exactly one --family shard")
        performance_fixed_families = {
            str(row["family"])
            for row in PERFORMANCE_INTERRUPTION.FIXED_STATE_SHARDS
        }
        if (args.pool == "scorer_fit"
                and args.family in performance_fixed_families
                and (args.mixed_replacement_scene_request_digest is not None
                     or args.state_resolution_scene_request_digest is not None)):
            raise SystemExit(
                "performance-reissued fixed families cannot execute scene "
                "workers; restore only with --stage "
                "reissue-performance-fixed-states")
        if args.mixed_replacement_scene_request_digest is not None:
            return stage_mixed_replacement_scene_worker(args)
        if args.state_resolution_scene_request_digest is not None:
            return stage_state_resolution_scene_worker(args)
        if (args.pool == "scorer_fit"
                and args.family == REACHABILITY_REDRIVE_FAMILY):
            raise SystemExit(
                "small_enclosed_maze is issued only by --stage "
                "small-completion-search after prefix reissue and benchmark")
        if (args.pool == "scorer_fit"
                and args.family in performance_fixed_families):
            # These seven scientific selections are immutable inputs to the
            # performance successor.  Never treat a wrapper-validation error
            # or an interrupted archive gap as permission to re-run Genesis.
            # The explicit reissue stage is the only writer for their active
            # paths; this legacy entry point is read-only retention at most.
            try:
                shard_path, existing, evidence = \
                    _load_active_state_shard_evidence(
                        out, str(args.family), pool="scorer_fit")
            except Exception as exc:
                raise RuntimeError(
                    f"fixed family {args.family} must be restored only by "
                    "--stage reissue-performance-fixed-states"
                ) from exc
            print(json.dumps({
                "recovery": "retained_valid_performance_reissued_identity_shard",
                "path": evidence["active_path"],
                "active_envelope_schema": evidence["envelope_schema"],
                "state_shard_digest": existing["state_shard_digest"],
                "states": len(existing["states"]),
            }, indent=2, sort_keys=True))
            return 0
        mixed_family = (
            args.pool == "scorer_fit"
            and args.family in {
                row["family"] for row in STATE_SELECTOR.PRESERVED_STATE_SHARDS
            }
        )
        raw_shard_path = (_mixed_active_state_shard_path(out, args.family)
                          if mixed_family
                          else out / f"state_shard_{args.family}.json")
        shard_path = _pin_generated_path(raw_shard_path, raw_shard_path)
        if shard_path.is_symlink():
            raise RuntimeError("state shard path is symlinked")
        if shard_path.is_file():
            try:
                existing = json.loads(shard_path.read_text())
                if mixed_family:
                    _validate_mixed_active_state_shard(
                        existing, raw_shard_path)
                else:
                    _validate_state_shard(
                        existing, raw_shard_path, args.pool)
                print(json.dumps({
                    "recovery": "retained_valid_completed_identity_shard",
                    "path": str(shard_path),
                    "state_shard_digest": existing["state_shard_digest"],
                    "states": len(existing["states"]),
                }, indent=2, sort_keys=True))
                return 0
            except Exception as exc:
                if _outcome_generation_started(out):
                    raise RuntimeError(
                        f"completed state shard {shard_path.name} or its "
                        "resolution provenance changed after outcomes"
                    ) from exc
                preserved = _preserve_invalid(shard_path, out,
                                              "identity-shard-validation-failed")
                print(f"[recovery] preserved invalid shard {preserved}: {exc}",
                      flush=True)
        manifest = (resolve_mixed_active_family(args)
                    if mixed_family else resolve_states(args))
        atomic_json(shard_path, manifest)
        from collections import Counter
        print(json.dumps({
            "state_shard_digest": manifest["state_shard_digest"],
            "states": len(manifest["states"]),
            "per_family": dict(Counter(s["family"] for s in manifest["states"])),
            "per_stratum": dict(Counter(s["stratum"] for s in manifest["states"])),
            "split_roles": dict(Counter(s["split_role"] for s in manifest["states"])),
        }, indent=2, sort_keys=True))
        return 0
    return stage_branches(args, smoke=args.stage == "smoke")


if __name__ == "__main__":
    raise SystemExit(main())
