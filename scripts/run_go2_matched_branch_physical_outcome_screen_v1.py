#!/usr/bin/env python3
"""Run the one-shot matched-branch physical-outcome screen V1."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import __version__ as PILLOW_VERSION
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks import (  # noqa: E402
    go2_matched_branch_physical_outcome_screen_v1 as evaluator,
)


SCHEMA = "lewm_go2_matched_branch_physical_outcome_screen_v1_result_v1"
TERMINAL_SCHEMA = "lewm_go2_matched_branch_physical_outcome_screen_v1_terminal_v1"
RESERVATION_SCHEMA = (
    "lewm_go2_matched_branch_physical_outcome_screen_v1_reservation_v1"
)
AUTHORITY_SCHEMA = (
    "lewm_go2_matched_branch_physical_outcome_screen_v1_execution_authority_v1"
)
AUTHORITY_STATUS = "AUTHORIZED_ONE_MATCHED_BRANCH_PHYSICAL_OUTCOME_SCREEN_ATTEMPT"
SOURCE_REVIEW_SCHEMA = (
    "lewm_go2_matched_branch_physical_outcome_screen_v1_source_review_v1"
)
SOURCE_REVIEW_STATUS = "PASS_INDEPENDENT_PHYSICAL_OUTCOME_SCREEN_SOURCE_REVIEW"
REPLAY_SCHEMA = "lewm_go2_matched_branch_physical_outcome_screen_v1_replay_v1"
REPLAY_STATUS = "PASS_EXACT_FRESH_PROCESS_PHYSICAL_OUTCOME_REPLAY"

PASS_VISUAL_STATUS = "PASS_VISUAL_PHYSICAL_DYNAMICS_HEADROOM"
PASS_ODOMETRY_STATUS = "PASS_ODOMETRY_ONLY_PHYSICAL_DYNAMICS_HEADROOM"
STOP_STATUS = "STOP_RETAINED_INPUT_PHYSICAL_DYNAMICS_HEADROOM_NOT_ESTABLISHED"
FAIL_STATUS = "FAIL_INFRASTRUCTURE_NO_SCIENTIFIC_DECISION"
TERMINAL_STATUSES = frozenset(
    {PASS_VISUAL_STATUS, PASS_ODOMETRY_STATUS, STOP_STATUS, FAIL_STATUS}
)

PREREGISTRATION = REPO_ROOT / (
    "docs/lewm_go2_matched_branch_physical_outcome_screen_v1_"
    "preregistration_2026-08-03.md"
)
PREREGISTRATION_SHA256 = (
    "6b758b33948ebd621698d47ec01a892c52f473fb6bec930fcdf1cb459fd8da3f"
)
PREREGISTRATION_BYTE_COUNT = 10_369
SOURCE_REVIEW = REPO_ROOT / (
    "docs/lewm_go2_matched_branch_physical_outcome_screen_v1_"
    "source_review_2026-08-03.json"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / (
    ".generated/dev/go2_matched_branch_physical_outcome_screen_v1/attempt_v1"
)
REPLAY_CLI = REPO_ROOT / (
    "scripts/replay_go2_matched_branch_physical_outcome_screen_v1.py"
)

POSTHOC_ROOT = REPO_ROOT / (
    ".generated/dev/lewm-go2-wm-bounded-branch-posthoc-join-admission-v1"
)
POSTHOC_MANIFEST = POSTHOC_ROOT / "manifest.json"
POSTHOC_TERMINAL = POSTHOC_ROOT / "terminal.json"
POSTHOC_TRAIN_ROWS = POSTHOC_ROOT / "train.jsonl"
POSTHOC_EVAL_ROWS = POSTHOC_ROOT / "eval.jsonl"
POSTHOC_TERMINAL_REVIEW = REPO_ROOT / (
    "docs/lewm_go2_world_model_bounded_branch_posthoc_join_admission_v1_"
    "terminal_review_2026-08-02.json"
)
PHYSICS_ROOT = REPO_ROOT / (
    ".generated/dev/lewm-go2-wm-bounded-branch-experiment-integrity-replacement-v1"
)
PHYSICS_RESULT = PHYSICS_ROOT / "physics_result.json"
PHYSICS_RECEIPT_CHECK = PHYSICS_ROOT / "physics_receipt_check.json"
COLLECTION_TERMINAL = PHYSICS_ROOT / "terminal_supervision.json"
COLLECTION_PLAN = REPO_ROOT / (
    "docs/lewm_go2_world_model_bounded_branch_integrity_replacement_v1_"
    "exact_plan_2026-08-02.json"
)
CALIBRATION_RECEIPT = REPO_ROOT / (
    ".generated/dev/lewm-go2-wm-counterfactual-calibration-v3-textured-v03-"
    "posthoc-analysis-v1/calibration_receipt.json"
)
TRAIN_CACHE = REPO_ROOT / (
    ".generated/dev/go2_matched_branch_successor_screen_v1/attempt_v1/"
    "features/dinov2.pt"
)
TRAIN_CACHE_RECEIPT = TRAIN_CACHE.with_suffix(".json")
EVAL_CACHE = REPO_ROOT / (
    ".generated/dev/go2_dinov2_physical_readout_calibration_v1/"
    "attempt_v2_integrity_replacement_v1/dinov2_eval.pt"
)
EVAL_CACHE_RECEIPT = EVAL_CACHE.with_suffix(".json")
PREDECESSOR_TERMINAL_REVIEW = REPO_ROOT / (
    "docs/lewm_go2_dinov2_dense_shared_spatial_readout_calibration_v1_"
    "terminal_review_2026-08-03.json"
)

ROLE_STATE_COUNT = 128
TOTAL_STATE_COUNT = 256
ACTION_COUNT = 9
CONTEXT_COUNT = 3
ROLE_ARTIFACT_COUNT = 1_536
TOKEN_SHAPE = (256, 384)
MAX_TOKEN_NORM_ERROR = 2.0e-3

OUTPUT_NAMES = (
    "reservation.json",
    "physical_outcome_checkpoint.pt",
    "evaluation.json",
    "replay.json",
    "result.json",
    "terminal.json",
)

# This is the exact reviewed implementation closure.  Data files are bound
# separately by ``_fixed_input_bindings_v1``.
SOURCE_PATHS = {
    "counterfactual_contract": REPO_ROOT
    / "lewm/benchmarks/go2_world_model_counterfactual_pilot_v1.py",
    "counterfactual_consumer": REPO_ROOT
    / "lewm/datasets/go2_world_model_counterfactual_pilot_v1.py",
    "prior_calibration_evaluator": REPO_ROOT
    / "lewm/benchmarks/go2_dinov2_physical_readout_calibration_v1.py",
    "dense_predecessor_evaluator": REPO_ROOT
    / "lewm/benchmarks/go2_dinov2_dense_shared_spatial_readout_calibration_v1.py",
    "dense_predecessor_model": REPO_ROOT
    / "lewm/models/go2_dinov2_dense_shared_spatial_readout_calibration_v1.py",
    "counterfactual_action_regret_evaluator": REPO_ROOT
    / "scripts/evaluate_go2_world_model_counterfactual_action_regret_v1.py",
    "physical_outcome_model": REPO_ROOT
    / "lewm/models/go2_matched_branch_physical_outcome_screen_v1.py",
    "physical_outcome_evaluator": REPO_ROOT
    / "lewm/benchmarks/go2_matched_branch_physical_outcome_screen_v1.py",
    "physical_outcome_runner": Path(__file__).resolve(),
    "physical_outcome_replay": REPLAY_CLI,
    "physical_outcome_model_test": REPO_ROOT
    / "lewm/tests/test_go2_matched_branch_physical_outcome_screen_model_v1.py",
    "physical_outcome_evaluator_test": REPO_ROOT
    / "lewm/tests/test_go2_matched_branch_physical_outcome_screen_v1.py",
    "physical_outcome_runner_test": REPO_ROOT
    / "lewm/tests/test_run_go2_matched_branch_physical_outcome_screen_v1.py",
    "physical_outcome_replay_test": REPO_ROOT
    / "lewm/tests/test_replay_go2_matched_branch_physical_outcome_screen_v1.py",
}

SOURCE_REVIEW_FIELDS = frozenset(
    {
        "schema",
        "status",
        "review_date",
        "reviewer",
        "protected_material_opened",
        "preregistration_binding",
        "source_bindings",
        "checks",
        "findings",
    }
)
SOURCE_REVIEW_CHECKS = frozenset(
    {
        "frozen_preregistration_binding_exact",
        "source_bindings_complete_exact_and_committed",
        "all_15_direct_input_bindings_complete_and_exact",
        "all_256_state_receipts_rehashed_without_legacy_live_validation",
        "pre_action_projection_excludes_forbidden_future_fields",
        "train_only_visual_pca_and_normalization",
        "task_action_control_identity_and_regret_frozen",
        "six_member_training_protocol_matches_preregistration",
        "checkpoint_written_before_eval_cache_load_and_evaluation_publication",
        "fresh_process_replay_retrains_and_recomputes",
        "exact_six_file_output_and_failure_terminal_fail_closed",
        "no_rgb_encoder_collection_protected_retry_or_resume_path",
        "focused_tests_passed",
        "compile_and_whitespace_checks_passed",
    }
)

REPLAY_REPRODUCTION_FIELDS = frozenset(
    {
        "checkpoint_exact",
        "pca_identity",
        "normalizer_identities",
        "state_dict_identities",
        "step_counts",
        "predictions",
        "selected_actions",
        "summaries",
        "bootstrap_intervals",
        "gates",
        "verdict",
        "evaluation_identity",
        "exactly_reproduced",
    }
)


class PhysicalOutcomeScreenRunnerError(RuntimeError):
    """Raised when one-shot authority, custody, or output contracts change."""


def canonical_bytes_v1(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PhysicalOutcomeScreenRunnerError(
            "document is not finite canonical JSON"
        ) from error


def _reject_protected(path: Path, *, label: str) -> None:
    for part in path.parts:
        lowered = part.lower()
        if (
            lowered == "sealed_test.json"
            or lowered == "sealed"
            or lowered.startswith("sealed_")
            or lowered in {"heldout", "held_out", "held-out"}
            or lowered.startswith(("heldout_", "held_out_", "held-out-"))
        ):
            raise PhysicalOutcomeScreenRunnerError(
                f"{label} names protected material"
            )


def _safe_path(path: Path, *, label: str, must_exist: bool = True) -> Path:
    selected = Path(os.path.abspath(os.fspath(path)))
    _reject_protected(selected, label=label)
    cursor = Path(selected.anchor)
    for part in selected.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise PhysicalOutcomeScreenRunnerError(f"{label} traverses a symlink")
        if not cursor.exists():
            if must_exist:
                raise PhysicalOutcomeScreenRunnerError(f"{label} is absent")
            break
    if must_exist and not selected.exists():
        raise PhysicalOutcomeScreenRunnerError(f"{label} is absent")
    return selected


def file_binding_v1(path: Path) -> dict[str, Any]:
    selected = _safe_path(path, label="bound file")
    if not selected.is_file():
        raise PhysicalOutcomeScreenRunnerError("bound path is not a file")
    digest = hashlib.sha256()
    size = 0
    with selected.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {"path": str(selected), "sha256": digest.hexdigest(), "byte_count": size}


def _binding(path: Path, sha256: str, byte_count: int) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256,
        "byte_count": byte_count,
    }


def _require_binding(value: object, *, label: str) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"path", "sha256", "byte_count"}
        or not isinstance(value.get("path"), str)
        or not isinstance(value.get("sha256"), str)
        or len(str(value["sha256"])) != 64
        or type(value.get("byte_count")) is not int
        or int(value["byte_count"]) <= 0
    ):
        raise PhysicalOutcomeScreenRunnerError(f"{label} binding is malformed")
    actual = file_binding_v1(Path(str(value["path"])))
    if actual != dict(value):
        raise PhysicalOutcomeScreenRunnerError(f"{label} binding changed")
    return actual


def _read_bound_json(
    path: Path, *, expected_sha256: str, expected_byte_count: int, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = file_binding_v1(path)
    if (
        binding["sha256"] != expected_sha256
        or binding["byte_count"] != expected_byte_count
    ):
        raise PhysicalOutcomeScreenRunnerError(f"{label} caller binding changed")
    try:
        document = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PhysicalOutcomeScreenRunnerError(f"{label} is not valid JSON") from error
    if not isinstance(document, Mapping):
        raise PhysicalOutcomeScreenRunnerError(f"{label} is not a JSON object")
    canonical_bytes_v1(document)
    return dict(document), binding


def _read_new_json(path: Path, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = file_binding_v1(path)
    return _read_bound_json(
        path,
        expected_sha256=binding["sha256"],
        expected_byte_count=binding["byte_count"],
        label=label,
    )


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    raw = canonical_bytes_v1(value) + b"\n"
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


def _save_torch_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o644,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def config_v1() -> dict[str, Any]:
    scientific = evaluator.config_v1()
    if not isinstance(scientific, Mapping):
        raise PhysicalOutcomeScreenRunnerError("evaluator config is not a mapping")
    result = {
        "scientific": dict(scientific),
        "direct_input_file_count": 15,
        "source_file_count": len(SOURCE_PATHS),
        "state_receipt_count": TOTAL_STATE_COUNT,
        "output_inventory": list(OUTPUT_NAMES),
        "replay_count": 1,
        "cpu_float32_training_required": True,
        "rgb_access_permitted": False,
        "encoder_execution_permitted": False,
        "protected_access_permitted": False,
        "retry_or_resume_permitted": False,
        "legacy_task_relevance_validation_permitted": False,
    }
    canonical_bytes_v1(result)
    return result


def _fixed_input_bindings_v1() -> dict[str, dict[str, Any]]:
    return {
        "posthoc_manifest": _binding(
            POSTHOC_MANIFEST,
            "87448995c905107453814a5e7e4cd9968d31cbc0e308513d17bc038c6585f15e",
            11_964,
        ),
        "posthoc_terminal": _binding(
            POSTHOC_TERMINAL,
            "a1590fffc673f7676016bb70d4b4f5530f24b9a49bf05e84dcec6bc1756fbe56",
            1_250,
        ),
        "posthoc_train_rows": _binding(
            POSTHOC_TRAIN_ROWS,
            "edc6f88bb105c39575477fbfbb0224bf0312cf5ee3e90551f86a9c11c2ebb447",
            30_432_624,
        ),
        "posthoc_eval_rows": _binding(
            POSTHOC_EVAL_ROWS,
            "531debbc431f2f8afc83a491b491b8822134c831b16ca4d283fe1e7f4ba07768",
            30_411_588,
        ),
        "posthoc_terminal_review": _binding(
            POSTHOC_TERMINAL_REVIEW,
            "bfd0250357d0f681c674db6c54ea4a8c4d5e617230332383beda3db3e0f38669",
            2_844,
        ),
        "physics_result": _binding(
            PHYSICS_RESULT,
            "25caf0a5d4c69e99559a663aa4cae96fb23ef191ccf34486804c3f2243553314",
            183_320,
        ),
        "physics_receipt_check": _binding(
            PHYSICS_RECEIPT_CHECK,
            "faeb50293bc684e35b6d725b027983ad0110e739db2d7b1aca1926e89a547dc6",
            892,
        ),
        "consumed_collection_terminal": _binding(
            COLLECTION_TERMINAL,
            "f7d2796139645892d22ad6bb99d26caffc2b5c3dcac2a655b1883b299d22bff4",
            12_949,
        ),
        "authorized_collection_plan": _binding(
            COLLECTION_PLAN,
            "8fe34054bb9ae709b6a8ecfea5fdae55c742d1b2e22af3c289d27a77f11c66ef",
            343_973,
        ),
        "calibration_receipt": _binding(
            CALIBRATION_RECEIPT,
            "58d1291ede7ee03a93d68eb7cec80c9322c47cd0b1d5fd1c41bf8f4b49ad484e",
            72_475,
        ),
        "train_cache": _binding(
            TRAIN_CACHE,
            "164f1fef8c859976c93f7fc978e938c6c8f7f9963cf92bb154f51b23d158b34b",
            302_107_682,
        ),
        "train_cache_receipt": _binding(
            TRAIN_CACHE_RECEIPT,
            "e94ec5d188811c44d4cc870e76d1888aa6f30ee6d423557ee9f3e2918a700994",
            1_770,
        ),
        "eval_cache": _binding(
            EVAL_CACHE,
            "00a2e197d98effcd192392f50170648622a7210f954075002dc8b43110c636f8",
            302_106_281,
        ),
        "eval_cache_receipt": _binding(
            EVAL_CACHE_RECEIPT,
            "d3e928cc563beb4dd850f34ca41915b8e5974c6d0b1b182602f3e3f20828421c",
            1_770,
        ),
        "predecessor_dense_dino_terminal_review": _binding(
            PREDECESSOR_TERMINAL_REVIEW,
            "f6ed2d09a407a4cf70097eaa4b2dcffd223e598e4eb59cf8e751997459384020",
            27_120,
        ),
    }


def _bound_document_v1(
    authority: Mapping[str, Any], label: str
) -> dict[str, Any]:
    item = authority["input_bindings"][label]
    document, _ = _read_bound_json(
        Path(item["path"]),
        expected_sha256=item["sha256"],
        expected_byte_count=item["byte_count"],
        label=label.replace("_", " "),
    )
    return document


def _validate_source_review_v1(
    review: Mapping[str, Any],
    *,
    preregistration_binding: Mapping[str, Any],
    source_bindings: Mapping[str, Any],
) -> None:
    reviewer = review.get("reviewer")
    if (
        set(review) != SOURCE_REVIEW_FIELDS
        or review.get("schema") != SOURCE_REVIEW_SCHEMA
        or review.get("status") != SOURCE_REVIEW_STATUS
        or review.get("review_date") != "2026-08-03"
        or review.get("protected_material_opened") is not False
        or review.get("preregistration_binding") != preregistration_binding
        or review.get("source_bindings") != source_bindings
        or review.get("findings") != []
        or not isinstance(reviewer, Mapping)
        or set(reviewer) != {"identity", "independence_basis"}
        or any(not isinstance(value, str) or not value.strip() for value in reviewer.values())
        or not isinstance(review.get("checks"), Mapping)
        or set(review["checks"]) != SOURCE_REVIEW_CHECKS
        or any(value is not True for value in review["checks"].values())
    ):
        raise PhysicalOutcomeScreenRunnerError(
            "independent source review did not pass exactly"
        )


def _read_authority(
    path: Path, *, expected_sha256: str, expected_byte_count: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    authority, authority_binding = _read_bound_json(
        path,
        expected_sha256=expected_sha256,
        expected_byte_count=expected_byte_count,
        label="execution authority",
    )
    required = {
        "schema",
        "status",
        "citable_as_scientific_evidence",
        "development_screen_only",
        "authorizes_physical_outcome_fitting",
        "authorizes_collection",
        "authorizes_rgb_access",
        "authorizes_encoder_execution",
        "authorizes_protected_access",
        "authorizes_retry_or_resume",
        "preregistration_binding",
        "source_review_binding",
        "source_bindings",
        "input_bindings",
        "output_root",
        "environment",
        "config",
        "git_commit",
    }
    if (
        set(authority) != required
        or authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("status") != AUTHORITY_STATUS
        or authority.get("citable_as_scientific_evidence") is not False
        or authority.get("development_screen_only") is not True
        or authority.get("authorizes_physical_outcome_fitting") is not True
        or authority.get("authorizes_collection") is not False
        or authority.get("authorizes_rgb_access") is not False
        or authority.get("authorizes_encoder_execution") is not False
        or authority.get("authorizes_protected_access") is not False
        or authority.get("authorizes_retry_or_resume") is not False
        or authority.get("output_root") != str(DEFAULT_OUTPUT_ROOT.resolve())
        or authority.get("config") != config_v1()
    ):
        raise PhysicalOutcomeScreenRunnerError("execution authority contract changed")
    preregistration = _require_binding(
        authority["preregistration_binding"], label="preregistration"
    )
    if preregistration != _binding(
        PREREGISTRATION, PREREGISTRATION_SHA256, PREREGISTRATION_BYTE_COUNT
    ):
        raise PhysicalOutcomeScreenRunnerError(
            "authority does not bind the frozen preregistration"
        )
    inputs = authority.get("input_bindings")
    if not isinstance(inputs, Mapping) or dict(inputs) != _fixed_input_bindings_v1():
        raise PhysicalOutcomeScreenRunnerError("authority input closure changed")
    for label, item in inputs.items():
        _require_binding(item, label=f"input {label}")
    sources = authority.get("source_bindings")
    if not isinstance(sources, Mapping) or set(sources) != set(SOURCE_PATHS):
        raise PhysicalOutcomeScreenRunnerError("authority source closure changed")
    for label, expected_path in SOURCE_PATHS.items():
        item = _require_binding(sources[label], label=f"source {label}")
        if item["path"] != str(expected_path.resolve()):
            raise PhysicalOutcomeScreenRunnerError(f"source {label} path changed")
    review_binding = _require_binding(
        authority["source_review_binding"], label="source review"
    )
    if review_binding["path"] != str(SOURCE_REVIEW.resolve()):
        raise PhysicalOutcomeScreenRunnerError("source review path changed")
    review, _ = _read_bound_json(
        Path(review_binding["path"]),
        expected_sha256=review_binding["sha256"],
        expected_byte_count=review_binding["byte_count"],
        label="source review",
    )
    _validate_source_review_v1(
        review,
        preregistration_binding=preregistration,
        source_bindings=sources,
    )
    environment = authority.get("environment")
    if (
        not isinstance(environment, Mapping)
        or set(environment) != {"python", "torch", "numpy", "pillow"}
        or environment.get("python") != str(Path(sys.executable).resolve())
        or environment.get("torch") != torch.__version__
        or environment.get("numpy") != np.__version__
        or environment.get("pillow") != PILLOW_VERSION
    ):
        raise PhysicalOutcomeScreenRunnerError("execution environment changed")
    commit = authority.get("git_commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
        or subprocess.run(
            ["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", commit, "HEAD"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        != 0
    ):
        raise PhysicalOutcomeScreenRunnerError(
            "reviewed source commit is not an execution ancestor"
        )
    return authority, authority_binding


def _legacy_binding_to_standard_v1(value: object, *, label: str) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"path", "file_sha256", "byte_count"}
        or not isinstance(value.get("path"), str)
        or not isinstance(value.get("file_sha256"), str)
        or len(str(value["file_sha256"])) != 64
        or type(value.get("byte_count")) is not int
        or int(value["byte_count"]) <= 0
    ):
        raise PhysicalOutcomeScreenRunnerError(f"{label} legacy binding is malformed")
    return {
        "path": str(value["path"]),
        "sha256": str(value["file_sha256"]),
        "byte_count": int(value["byte_count"]),
    }


def _resolve_receipt_binding_v1(
    value: object, *, source_root: Path, label: str
) -> dict[str, Any]:
    binding = _legacy_binding_to_standard_v1(value, label=label)
    relative = Path(str(binding["path"]))
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise PhysicalOutcomeScreenRunnerError(f"{label} path is not strict relative")
    selected = _safe_path(source_root / relative, label=label)
    try:
        selected.relative_to(source_root)
    except ValueError as error:
        raise PhysicalOutcomeScreenRunnerError(f"{label} escapes source root") from error
    binding["path"] = str(selected)
    if file_binding_v1(selected) != binding:
        raise PhysicalOutcomeScreenRunnerError(f"{label} binding changed")
    return binding


def _validate_upstream_route_v1(authority: Mapping[str, Any]) -> None:
    physics = _bound_document_v1(authority, "physics_result")
    receipt_check = _bound_document_v1(authority, "physics_receipt_check")
    terminal = _bound_document_v1(authority, "consumed_collection_terminal")
    calibration = _bound_document_v1(authority, "calibration_receipt")
    predecessor = _bound_document_v1(
        authority, "predecessor_dense_dino_terminal_review"
    )
    expected_counts = {
        "actions": 9,
        "candidate_branches": 2_304,
        "context_frames": 768,
        "roles": {"eval": 128, "train": 128},
        "scenes": 32,
        "sentinel_branches": 0,
        "states": 256,
        "target_frames": 2_304,
        "total_branches": 2_304,
    }
    if (
        physics.get("status") != "PHYSICS_COMPLETE"
        or physics.get("failure") is not None
        or physics.get("observed_counts") != expected_counts
        or physics.get("expected_counts") != expected_counts
        or len(physics.get("state_receipt_bindings", ())) != TOTAL_STATE_COUNT
        or receipt_check.get("status") != "PASS"
        or receipt_check.get("counts") != expected_counts
        or receipt_check.get("rgb_bytes_opened") != 0
        or receipt_check.get("runtime_payloads_opened") != 0
        or terminal.get("status") != "CONSUMED_TERMINAL_FAILURE"
        or terminal.get("authorizes_retry_or_resume") is not False
        or terminal.get("physics_result_binding")
        != {
            "path": str(PHYSICS_RESULT.resolve()),
            "file_sha256": authority["input_bindings"]["physics_result"]["sha256"],
            "byte_count": authority["input_bindings"]["physics_result"]["byte_count"],
        }
        or calibration.get("status") != "COMPLETE"
        or calibration.get("authorizes_retry_or_resume") is not False
        or predecessor.get("status")
        != "PASS_COMPLETE_SCIENTIFIC_STOP_TERMINAL_REVIEW"
        or predecessor.get("protected_material_opened") is not False
        or predecessor.get("findings") != []
    ):
        raise PhysicalOutcomeScreenRunnerError("upstream development route changed")


def _load_state_receipts_v1(
    authority: Mapping[str, Any],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], dict[str, Any]]:
    """Rehash and open the 256 bound state receipts without legacy live checks."""

    plan = _bound_document_v1(authority, "authorized_collection_plan")
    physics = _bound_document_v1(authority, "physics_result")
    plan_binding = _legacy_binding_to_standard_v1(
        physics.get("plan_binding"), label="physics plan"
    )
    if plan_binding != authority["input_bindings"]["authorized_collection_plan"]:
        raise PhysicalOutcomeScreenRunnerError("physics plan binding changed")
    planned_states = plan.get("states")
    receipt_bindings = physics.get("state_receipt_bindings")
    if (
        plan.get("output_root") != str(PHYSICS_ROOT.resolve())
        or not isinstance(planned_states, list)
        or len(planned_states) != TOTAL_STATE_COUNT
        or not isinstance(receipt_bindings, list)
        or len(receipt_bindings) != TOTAL_STATE_COUNT
    ):
        raise PhysicalOutcomeScreenRunnerError("collection state inventory changed")

    train: list[dict[str, Any]] = []
    evaluation: list[dict[str, Any]] = []
    opened_bindings: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, (planned, receipt_binding) in enumerate(
        zip(planned_states, receipt_bindings, strict=True)
    ):
        if not isinstance(planned, Mapping):
            raise PhysicalOutcomeScreenRunnerError("planned state is malformed")
        actual_binding = _resolve_receipt_binding_v1(
            receipt_binding,
            source_root=PHYSICS_ROOT.resolve(),
            label=f"state receipt {index}",
        )
        receipt, _ = _read_bound_json(
            Path(actual_binding["path"]),
            expected_sha256=actual_binding["sha256"],
            expected_byte_count=actual_binding["byte_count"],
            label=f"state receipt {index}",
        )
        state = receipt.get("state")
        context = receipt.get("context")
        branches = receipt.get("branches")
        role = planned.get("role")
        state_id = planned.get("state_id")
        if (
            role not in {"train", "eval"}
            or not isinstance(state_id, str)
            or state_id in seen_ids
            or not isinstance(state, Mapping)
            or state.get("state_id") != state_id
            or state.get("role") != role
            or state.get("scene_id") != planned.get("scene_id")
            or state.get("family") != planned.get("family")
            or state.get("group_index") != planned.get("group_index")
            or state.get("state_index_in_scene")
            != planned.get("state_index_in_scene")
            or receipt.get("status") != "PHYSICS_COMPLETE"
            or not isinstance(context, Mapping)
            or len(context.get("context_base_pose_world_sequence", ()))
            != CONTEXT_COUNT
            or len(context.get("history_executed_blocks", ())) != 2
            or not isinstance(branches, list)
            or len(branches) != ACTION_COUNT
            or sorted(branch.get("action_id") for branch in branches) != list(
                range(ACTION_COUNT)
            )
        ):
            raise PhysicalOutcomeScreenRunnerError(
                f"state receipt {index} does not match the frozen plan"
            )
        seen_ids.add(state_id)
        opened_bindings.append(actual_binding)
        (train if role == "train" else evaluation).append(receipt)
    if len(train) != ROLE_STATE_COUNT or len(evaluation) != ROLE_STATE_COUNT:
        raise PhysicalOutcomeScreenRunnerError("role receipt counts changed")
    audit = {
        "state_receipt_open_count": len(opened_bindings),
        "train_state_count": len(train),
        "eval_state_count": len(evaluation),
        "receipt_binding_identity_sha256": hashlib.sha256(
            canonical_bytes_v1(opened_bindings)
        ).hexdigest(),
        "legacy_task_relevance_validation_called": False,
        "rgb_leaf_open_count": 0,
    }
    return tuple(train), tuple(evaluation), audit


def _expected_artifact_ids_v1(
    receipts: Sequence[Mapping[str, Any]], *, role: str
) -> tuple[str, ...]:
    try:
        ordered = sorted(
            receipts,
            key=lambda receipt: (
                int(receipt["state"]["group_index"]),
                str(receipt["state"]["state_id"]),
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise PhysicalOutcomeScreenRunnerError(
            f"{role} receipt ordering is malformed"
        ) from error
    result: list[str] = []
    for receipt in ordered:
        if receipt["state"].get("role") != role:
            raise PhysicalOutcomeScreenRunnerError(f"{role} receipt crossed role")
        context_ids = tuple(receipt["context"].get("rgb_artifact_ids", ()))
        branches = sorted(receipt["branches"], key=lambda branch: int(branch["action_id"]))
        target_ids = tuple(branch["frame_receipt"].get("artifact_id") for branch in branches)
        if (
            len(context_ids) != CONTEXT_COUNT
            or len(target_ids) != ACTION_COUNT
            or any(not isinstance(value, str) or not value for value in context_ids + target_ids)
        ):
            raise PhysicalOutcomeScreenRunnerError(f"{role} artifact IDs changed")
        result.extend(context_ids)
        result.extend(target_ids)
    if len(result) != ROLE_ARTIFACT_COUNT or len(result) != len(set(result)):
        raise PhysicalOutcomeScreenRunnerError(f"{role} artifact inventory changed")
    return tuple(result)


def _validate_feature_tensor_v1(features: torch.Tensor, *, role: str) -> None:
    if (
        not isinstance(features, torch.Tensor)
        or tuple(features.shape) != (ROLE_ARTIFACT_COUNT, *TOKEN_SHAPE)
        or features.dtype != torch.float16
        or features.device.type != "cpu"
        or not bool(torch.isfinite(features).all())
    ):
        raise PhysicalOutcomeScreenRunnerError(f"{role} feature tensor changed")
    maximum_error = 0.0
    for start in range(0, ROLE_ARTIFACT_COUNT, 64):
        norms = torch.linalg.vector_norm(
            features[start : start + 64].to(dtype=torch.float32), dim=-1
        )
        maximum_error = max(
            maximum_error, float(torch.max(torch.abs(norms - 1.0)).item())
        )
    if maximum_error > MAX_TOKEN_NORM_ERROR:
        raise PhysicalOutcomeScreenRunnerError(
            f"{role} feature token normalization changed"
        )


def _load_feature_cache_v1(
    authority: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    *,
    role: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if role not in {"train", "eval"}:
        raise PhysicalOutcomeScreenRunnerError("cache role changed")
    cache_label = f"{role}_cache"
    receipt_label = f"{role}_cache_receipt"
    cache_binding = authority["input_bindings"][cache_label]
    receipt = _bound_document_v1(authority, receipt_label)
    expected_ids = _expected_artifact_ids_v1(receipts, role=role)
    order_sha256 = hashlib.sha256(canonical_bytes_v1(list(expected_ids))).hexdigest()
    expected_schema = (
        "lewm_go2_matched_branch_successor_feature_cache_receipt_v1"
        if role == "train"
        else "lewm_go2_dinov2_physical_readout_eval_feature_cache_receipt_v1"
    )
    if (
        receipt.get("schema") != expected_schema
        or receipt.get("encoder") != "dinov2"
        or receipt.get("binding") != cache_binding
        or receipt.get("artifact_order_sha256") != order_sha256
        or receipt.get("artifact_count") != ROLE_ARTIFACT_COUNT
        or receipt.get("shape") != [ROLE_ARTIFACT_COUNT, *TOKEN_SHAPE]
        or receipt.get("storage_dtype") != "float16"
        or receipt.get("source_bundle_manifest")
        != {
            "path": authority["input_bindings"]["posthoc_manifest"]["path"],
            "file_sha256": authority["input_bindings"]["posthoc_manifest"]["sha256"],
            "byte_count": authority["input_bindings"]["posthoc_manifest"]["byte_count"],
        }
        or (role == "train" and receipt.get("eval_artifact_open_count") != 0)
        or (role == "eval" and receipt.get("train_artifact_open_count") != 0)
    ):
        raise PhysicalOutcomeScreenRunnerError(f"{role} cache receipt changed")
    # Rehash once more immediately before the safe payload load.
    if file_binding_v1(Path(cache_binding["path"])) != cache_binding:
        raise PhysicalOutcomeScreenRunnerError(f"{role} cache binding changed")
    try:
        payload = torch.load(
            cache_binding["path"], map_location="cpu", weights_only=True
        )
    except Exception as error:
        raise PhysicalOutcomeScreenRunnerError(
            f"{role} cache is not a safe Torch payload"
        ) from error
    expected_payload_schema = (
        "lewm_go2_matched_branch_successor_feature_cache_v1"
        if role == "train"
        else "lewm_go2_dinov2_physical_readout_eval_feature_cache_v1"
    )
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != expected_payload_schema
        or payload.get("encoder") != "dinov2"
        or tuple(payload.get("artifact_ids", ())) != expected_ids
        or (role == "eval" and payload.get("artifact_order_sha256") != order_sha256)
        or not isinstance(payload.get("features"), torch.Tensor)
    ):
        raise PhysicalOutcomeScreenRunnerError(f"{role} cache payload changed")
    features = payload["features"]
    _validate_feature_tensor_v1(features, role=role)
    return {"artifact_ids": expected_ids, "features": features}, receipt


def _execution_bindings_unchanged(
    authority: Mapping[str, Any], *, authority_binding: Mapping[str, Any]
) -> None:
    closure: list[tuple[str, Mapping[str, Any]]] = [
        ("execution authority", authority_binding),
        ("preregistration", authority["preregistration_binding"]),
        ("source review", authority["source_review_binding"]),
    ]
    closure.extend(
        (f"source {label}", item) for label, item in authority["source_bindings"].items()
    )
    closure.extend(
        (f"input {label}", item) for label, item in authority["input_bindings"].items()
    )
    for label, expected in closure:
        if file_binding_v1(Path(str(expected["path"]))) != dict(expected):
            raise PhysicalOutcomeScreenRunnerError(f"{label} changed during execution")


def _configure_deterministic_cpu_training_v1() -> None:
    """Install and verify the preregistered CPU training runtime."""

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    if (
        not torch.are_deterministic_algorithms_enabled()
        or torch.get_num_threads() != 1
    ):
        raise PhysicalOutcomeScreenRunnerError(
            "deterministic one-thread CPU runtime was not established"
        )


def _launch_replay_v1(
    *,
    authority_binding: Mapping[str, Any],
    checkpoint_binding: Mapping[str, Any],
    evaluation_binding: Mapping[str, Any],
) -> None:
    command = [
        sys.executable,
        str(REPLAY_CLI),
        "--authority",
        str(authority_binding["path"]),
        "--expected-authority-sha256",
        str(authority_binding["sha256"]),
        "--expected-authority-byte-count",
        str(authority_binding["byte_count"]),
        "--checkpoint",
        str(checkpoint_binding["path"]),
        "--expected-checkpoint-sha256",
        str(checkpoint_binding["sha256"]),
        "--expected-checkpoint-byte-count",
        str(checkpoint_binding["byte_count"]),
        "--evaluation",
        str(evaluation_binding["path"]),
        "--expected-evaluation-sha256",
        str(evaluation_binding["sha256"]),
        "--expected-evaluation-byte-count",
        str(evaluation_binding["byte_count"]),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise PhysicalOutcomeScreenRunnerError(
            f"fresh replay process failed: {detail[-1000:]}"
        )


def _validate_replay_v1(
    replay: Mapping[str, Any],
    *,
    authority_binding: Mapping[str, Any],
    checkpoint_binding: Mapping[str, Any],
    evaluation_binding: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> None:
    required = {
        "schema",
        "status",
        "citable_as_scientific_evidence",
        "authority_binding",
        "checkpoint_binding",
        "primary_evaluation_binding",
        "recomputed_evaluation",
        "recomputed_verdict",
        "reproduction",
        "protected_material_opened",
        "rgb_access",
        "encoder_execution_count",
    }
    reproduction = replay.get("reproduction")
    expected_replay_verdict = evaluator.verdict_v1(
        evaluation,
        infrastructure_checks_passed=True,
        deterministic_replay_passed=True,
    )
    if (
        set(replay) != required
        or replay.get("schema") != REPLAY_SCHEMA
        or replay.get("status") != REPLAY_STATUS
        or replay.get("citable_as_scientific_evidence") is not False
        or replay.get("authority_binding") != authority_binding
        or replay.get("checkpoint_binding") != checkpoint_binding
        or replay.get("primary_evaluation_binding") != evaluation_binding
        or replay.get("protected_material_opened") is not False
        or replay.get("rgb_access") != {"train": 0, "eval": 0}
        or replay.get("encoder_execution_count") != 0
        or not isinstance(reproduction, Mapping)
        or set(reproduction) != REPLAY_REPRODUCTION_FIELDS
        or any(value is not True for value in reproduction.values())
        or canonical_bytes_v1(replay.get("recomputed_evaluation"))
        != canonical_bytes_v1(evaluation)
        or canonical_bytes_v1(replay.get("recomputed_verdict"))
        != canonical_bytes_v1(expected_replay_verdict)
    ):
        raise PhysicalOutcomeScreenRunnerError(
            "fresh replay did not reproduce the primary evaluation exactly"
        )


def _verdict_status_v1(
    verdict: Mapping[str, Any], *, deterministic_replay_passed: bool
) -> str:
    status = verdict.get("terminal_status")
    if (
        set(verdict) != {"gates", "passed", "terminal_status"}
        or status not in TERMINAL_STATUSES
        or (not deterministic_replay_passed and status != FAIL_STATUS)
        or (deterministic_replay_passed and status == FAIL_STATUS)
    ):
        raise PhysicalOutcomeScreenRunnerError("scientific verdict contract changed")
    return str(status)


def _assert_inventory_v1(output_root: Path, expected: set[str]) -> None:
    observed: list[str] = []
    with os.scandir(output_root) as entries:
        for entry in entries:
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                raise PhysicalOutcomeScreenRunnerError(
                    "attempt root contains a non-file"
                )
            observed.append(entry.name)
    if set(observed) != expected or len(observed) != len(expected):
        raise PhysicalOutcomeScreenRunnerError(
            f"attempt inventory changed: {sorted(observed)}"
        )


def execute_v1(
    authority: Mapping[str, Any], *, authority_binding: Mapping[str, Any]
) -> dict[str, Any]:
    output_root = _safe_path(
        Path(str(authority["output_root"])),
        label="physical outcome output",
        must_exist=False,
    )
    if output_root != DEFAULT_OUTPUT_ROOT.resolve():
        raise PhysicalOutcomeScreenRunnerError("physical outcome output root changed")
    _safe_path(output_root.parent, label="output parent", must_exist=False)
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json_exclusive(
        output_root / "reservation.json",
        {
            "schema": RESERVATION_SCHEMA,
            "authority_binding": dict(authority_binding),
            "attempt_root": str(output_root),
            "owner_pid": os.getpid(),
            "consumes_attempt": True,
        },
    )
    # Authority validation rehashed the entire closure before reservation;
    # rehash it again immediately after attempt consumption and before science.
    _execution_bindings_unchanged(authority, authority_binding=authority_binding)
    _validate_upstream_route_v1(authority)
    train_receipts, eval_receipts, receipt_audit = _load_state_receipts_v1(authority)
    train_cache, train_cache_receipt = _load_feature_cache_v1(
        authority, train_receipts, role="train"
    )
    # Evaluation features are deliberately not loaded until the checkpoint is durable.
    dataset_train = evaluator.build_physical_dataset_v1(
        train_receipts=train_receipts,
        eval_receipts=None,
        train_cache=train_cache,
        eval_cache=None,
    )
    _configure_deterministic_cpu_training_v1()
    implementation_source_binding = authority["source_bindings"][
        "physical_outcome_evaluator"
    ]
    checkpoint = evaluator.fit_primary_checkpoint_v1(
        dataset_train,
        implementation_source_binding=implementation_source_binding,
    )
    if not isinstance(checkpoint, Mapping):
        raise PhysicalOutcomeScreenRunnerError("evaluator returned invalid checkpoint")
    evaluator.validate_checkpoint_v1(
        checkpoint,
        implementation_source_binding=implementation_source_binding,
    )
    checkpoint_path = output_root / "physical_outcome_checkpoint.pt"
    _save_torch_exclusive(checkpoint_path, checkpoint)
    checkpoint_binding = file_binding_v1(checkpoint_path)

    eval_cache, eval_cache_receipt = _load_feature_cache_v1(
        authority, eval_receipts, role="eval"
    )
    dataset = evaluator.build_physical_dataset_v1(
        train_receipts=train_receipts,
        eval_receipts=eval_receipts,
        train_cache=train_cache,
        eval_cache=eval_cache,
    )
    evaluation = evaluator.evaluate_primary_checkpoint_v1(
        checkpoint,
        dataset,
        implementation_source_binding=implementation_source_binding,
    )
    if not isinstance(evaluation, Mapping):
        raise PhysicalOutcomeScreenRunnerError("evaluator returned invalid evaluation")
    evaluation = dict(evaluation)
    canonical_bytes_v1(evaluation)
    evaluation_path = output_root / "evaluation.json"
    _write_json_exclusive(evaluation_path, evaluation)
    evaluation_binding = file_binding_v1(evaluation_path)
    del checkpoint, dataset_train, dataset, train_cache, eval_cache

    _execution_bindings_unchanged(authority, authority_binding=authority_binding)
    _launch_replay_v1(
        authority_binding=authority_binding,
        checkpoint_binding=checkpoint_binding,
        evaluation_binding=evaluation_binding,
    )
    replay, replay_binding = _read_new_json(
        output_root / "replay.json", label="fresh replay"
    )
    _validate_replay_v1(
        replay,
        authority_binding=authority_binding,
        checkpoint_binding=checkpoint_binding,
        evaluation_binding=evaluation_binding,
        evaluation=evaluation,
    )
    _execution_bindings_unchanged(authority, authority_binding=authority_binding)
    verdict = evaluator.verdict_v1(
        evaluation,
        infrastructure_checks_passed=True,
        deterministic_replay_passed=True,
    )
    if not isinstance(verdict, Mapping):
        raise PhysicalOutcomeScreenRunnerError("evaluator returned invalid verdict")
    status = _verdict_status_v1(verdict, deterministic_replay_passed=True)
    report = {
        "schema": SCHEMA,
        "status": status,
        "citable_as_scientific_evidence": False,
        "development_screen_only": True,
        "authorizes_navigation_claim": False,
        "authorizes_new_dense_comparison_preregistration": (
            status == PASS_VISUAL_STATUS
        ),
        "authority_binding": dict(authority_binding),
        "preregistration_binding": dict(authority["preregistration_binding"]),
        "input_bindings": dict(authority["input_bindings"]),
        "artifact_bindings": {
            "physical_outcome_checkpoint": checkpoint_binding,
            "evaluation": evaluation_binding,
            "replay": replay_binding,
        },
        "custody_audit": receipt_audit,
        "cache_receipts": {
            "train": train_cache_receipt,
            "eval": eval_cache_receipt,
        },
        "rgb_access": {"primary": 0, "replay": 0},
        "encoder_execution_count": 0,
        "evaluation": evaluation,
        "replay": {
            "binding": replay_binding,
            "exactly_reproduced": True,
            "fresh_process": True,
            "cache_only": True,
        },
        "verdict": dict(verdict),
    }
    _write_json_exclusive(output_root / "result.json", report)
    _assert_inventory_v1(output_root, set(OUTPUT_NAMES[:-1]))
    terminal = {
        "schema": TERMINAL_SCHEMA,
        "status": status,
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "authorizes_navigation_claim": False,
        "authorizes_new_dense_comparison_preregistration": (
            status == PASS_VISUAL_STATUS
        ),
        "result_binding": file_binding_v1(output_root / "result.json"),
        "deterministic_replay_passed": True,
        "failure": None,
    }
    _write_json_exclusive(output_root / "terminal.json", terminal)
    _assert_inventory_v1(output_root, set(OUTPUT_NAMES))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--expected-authority-sha256", required=True)
    parser.add_argument("--expected-authority-byte-count", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    authority, authority_binding = _read_authority(
        args.authority,
        expected_sha256=args.expected_authority_sha256,
        expected_byte_count=args.expected_authority_byte_count,
    )
    output_root = Path(str(authority["output_root"]))
    existed = output_root.exists()
    try:
        report = execute_v1(authority, authority_binding=authority_binding)
    except Exception as error:
        if (
            not existed
            and output_root.is_dir()
            and not (output_root / "terminal.json").exists()
        ):
            _write_json_exclusive(
                output_root / "terminal.json",
                {
                    "schema": TERMINAL_SCHEMA,
                    "status": FAIL_STATUS,
                    "citable_as_scientific_evidence": False,
                    "authorizes_retry_or_resume": False,
                    "authorizes_navigation_claim": False,
                    "authorizes_new_dense_comparison_preregistration": False,
                    "result_binding": None,
                    "deterministic_replay_passed": False,
                    "failure": {
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    },
                },
            )
        raise
    print(json.dumps({"status": report["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORITY_SCHEMA",
    "AUTHORITY_STATUS",
    "DEFAULT_OUTPUT_ROOT",
    "FAIL_STATUS",
    "OUTPUT_NAMES",
    "PASS_ODOMETRY_STATUS",
    "PASS_VISUAL_STATUS",
    "PREREGISTRATION_BYTE_COUNT",
    "PREREGISTRATION_SHA256",
    "REPLAY_SCHEMA",
    "REPLAY_STATUS",
    "SOURCE_PATHS",
    "SOURCE_REVIEW_CHECKS",
    "SOURCE_REVIEW_SCHEMA",
    "SOURCE_REVIEW_STATUS",
    "STOP_STATUS",
    "PhysicalOutcomeScreenRunnerError",
    "canonical_bytes_v1",
    "config_v1",
    "execute_v1",
    "file_binding_v1",
]
