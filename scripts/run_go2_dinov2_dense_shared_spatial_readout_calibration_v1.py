#!/usr/bin/env python3
"""Run the one-shot dense shared DINO spatial-readout calibration V1."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from PIL import __version__ as PILLOW_VERSION
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks import (  # noqa: E402
    go2_dinov2_dense_shared_spatial_readout_calibration_v1 as evaluator,
)
from lewm.benchmarks import (  # noqa: E402
    go2_dinov2_physical_readout_calibration_integrity_replacement_v1
    as compatibility,
)
from lewm.benchmarks import (  # noqa: E402
    go2_dinov2_physical_readout_calibration_v1 as prior_evaluator,
)
from scripts import (  # noqa: E402
    evaluate_go2_world_model_visual_domain_parity_task_relevance_v1
    as task_relevance,
)
from scripts import (  # noqa: E402
    run_go2_dinov2_physical_readout_calibration_v1 as prior_runner,
)


SCHEMA = (
    "lewm_go2_dinov2_dense_shared_spatial_readout_calibration_v1_result_v1"
)
TERMINAL_SCHEMA = (
    "lewm_go2_dinov2_dense_shared_spatial_readout_calibration_v1_terminal_v1"
)
RESERVATION_SCHEMA = (
    "lewm_go2_dinov2_dense_shared_spatial_readout_calibration_v1_reservation_v1"
)
AUTHORITY_SCHEMA = (
    "lewm_go2_dinov2_dense_shared_spatial_readout_calibration_v1_"
    "execution_authority_v1"
)
AUTHORITY_STATUS = "AUTHORIZED_ONE_DENSE_SHARED_DINO_CALIBRATION_ATTEMPT"
SOURCE_REVIEW_SCHEMA = (
    "lewm_go2_dinov2_dense_shared_spatial_readout_calibration_v1_"
    "source_review_v1"
)
SOURCE_REVIEW_STATUS = "PASS_INDEPENDENT_DENSE_SHARED_CALIBRATION_SOURCE_REVIEW"
COMPATIBILITY_RECEIPT_SCHEMA = (
    "lewm_go2_dinov2_dense_shared_spatial_readout_calibration_v1_"
    "compatibility_receipt_v1"
)
COMPATIBILITY_RECEIPT_STATUS = (
    "PASS_PUBLISHED_BEFORE_STRICT_METADATA_LOADER_RETURN"
)
REPLAY_SCHEMA = (
    "lewm_go2_dinov2_dense_shared_spatial_readout_calibration_v1_replay_v1"
)
REPLAY_STATUS = "PASS_EXACT_FRESH_PROCESS_CACHE_ONLY_REPLAY"

PASS_STATUS = "PASS_DENSE_SHARED_DINO_PHYSICAL_READOUT_HEADROOM_ESTABLISHED"
STOP_STATUS = "STOP_FROZEN_DINO_VISUAL_PLANNING_INTERFACE_NOT_ESTABLISHED"
FAIL_STATUS = "FAIL_INFRASTRUCTURE_NO_SCIENTIFIC_DECISION"
TERMINAL_STATUSES = frozenset({PASS_STATUS, STOP_STATUS, FAIL_STATUS})

PREREGISTRATION = REPO_ROOT / (
    "docs/lewm_go2_dinov2_dense_shared_spatial_readout_calibration_v1_"
    "preregistration_2026-08-03.md"
)
PREREGISTRATION_SHA256 = (
    "630a1bd508629878f6eab1cd4d7839d530e6f9216789bd388f32d4853c2e3f34"
)
PREREGISTRATION_BYTE_COUNT = 17_418
SOURCE_REVIEW = REPO_ROOT / (
    "docs/lewm_go2_dinov2_dense_shared_spatial_readout_calibration_v1_"
    "source_review_2026-08-03.json"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / (
    ".generated/dev/go2_dinov2_dense_shared_spatial_readout_calibration_v1/"
    "attempt_v1"
)
REPLAY_CLI = REPO_ROOT / (
    "scripts/replay_go2_dinov2_dense_shared_spatial_readout_calibration_v1.py"
)

TRAIN_CACHE = REPO_ROOT / (
    ".generated/dev/go2_matched_branch_successor_screen_v1/attempt_v1/"
    "features/dinov2.pt"
)
TRAIN_CACHE_RECEIPT = TRAIN_CACHE.with_suffix(".json")
EVAL_ROOT = REPO_ROOT / (
    ".generated/dev/go2_dinov2_physical_readout_calibration_v1/"
    "attempt_v2_integrity_replacement_v1"
)
EVAL_CACHE = EVAL_ROOT / "dinov2_eval.pt"
EVAL_CACHE_RECEIPT = EVAL_ROOT / "dinov2_eval.json"
PRIOR_RESULT = EVAL_ROOT / "result.json"
PRIOR_TERMINAL = EVAL_ROOT / "terminal.json"
PRIOR_COMPATIBILITY_RECEIPT = EVAL_ROOT / "compatibility_receipt.json"
PRIOR_TERMINAL_REVIEW = REPO_ROOT / (
    "docs/lewm_go2_dinov2_physical_readout_calibration_"
    "integrity_replacement_v1_terminal_review_2026-08-03.json"
)
POSTHOC_ROOT = REPO_ROOT / (
    ".generated/dev/lewm-go2-wm-bounded-branch-posthoc-join-admission-v1"
)
POSTHOC_MANIFEST = POSTHOC_ROOT / "manifest.json"
POSTHOC_TERMINAL = POSTHOC_ROOT / "terminal.json"
POSTHOC_RGB_MANIFEST = POSTHOC_ROOT / "rgb_manifest.json"
POSTHOC_TRAIN_ROWS = POSTHOC_ROOT / "train.jsonl"
POSTHOC_EVAL_ROWS = POSTHOC_ROOT / "eval.jsonl"
POSTHOC_TERMINAL_REVIEW = REPO_ROOT / (
    "docs/lewm_go2_world_model_bounded_branch_posthoc_join_admission_v1_"
    "terminal_review_2026-08-02.json"
)
STORED_TASK_RELEVANCE_RESULT = REPO_ROOT / (
    "docs/lewm_go2_world_model_visual_domain_parity_"
    "task_relevant_input_adequacy_result_v1_2026-08-02.json"
)
STORED_TASK_RELEVANCE_REVIEW = REPO_ROOT / (
    "docs/lewm_go2_world_model_visual_domain_parity_"
    "task_relevant_input_adequacy_independent_review_v1_2026-08-02.json"
)

ROLE_STATE_COUNT = 128
ROLE_ARTIFACT_COUNT = 1_536
TOKEN_SHAPE = (256, 384)
MAX_TOKEN_NORM_ERROR = 2.0e-3

OUTPUT_NAMES = (
    "reservation.json",
    "primary_compatibility_receipt.json",
    "pca_readout_checkpoint.pt",
    "evaluation.json",
    "replay_compatibility_receipt.json",
    "replay.json",
    "result.json",
    "terminal.json",
)

SOURCE_PATHS = {
    "prior_action_regret_evaluator": REPO_ROOT
    / "scripts/evaluate_go2_world_model_counterfactual_action_regret_v1.py",
    "counterfactual_benchmark_contract": REPO_ROOT
    / "lewm/benchmarks/go2_world_model_counterfactual_pilot_v1.py",
    "prior_calibration_evaluator": REPO_ROOT
    / "lewm/benchmarks/go2_dinov2_physical_readout_calibration_v1.py",
    "prior_calibration_evaluator_test": REPO_ROOT
    / "lewm/tests/test_go2_dinov2_physical_readout_calibration_v1.py",
    "predecessor_model": REPO_ROOT
    / "lewm/models/go2_matched_branch_successor_screen_v1.py",
    "pilot_consumer": REPO_ROOT
    / "lewm/datasets/go2_world_model_counterfactual_pilot_v1.py",
    "posthoc_loader": REPO_ROOT
    / "scripts/materialize_go2_world_model_bounded_branch_posthoc_join_admission_v1.py",
    "predecessor_runner": REPO_ROOT
    / "scripts/run_go2_matched_branch_successor_screen_v1.py",
    "prior_calibration_runner": REPO_ROOT
    / "scripts/run_go2_dinov2_physical_readout_calibration_v1.py",
    "prior_calibration_runner_test": REPO_ROOT
    / "lewm/tests/test_run_go2_dinov2_physical_readout_calibration_v1.py",
    "screen_data": REPO_ROOT
    / "lewm/benchmarks/go2_matched_branch_successor_screen_v1.py",
    "compatibility_module": REPO_ROOT
    / (
        "lewm/benchmarks/go2_dinov2_physical_readout_calibration_"
        "integrity_replacement_v1.py"
    ),
    "compatibility_test": REPO_ROOT
    / (
        "lewm/tests/test_go2_dinov2_physical_readout_calibration_"
        "integrity_replacement_v1.py"
    ),
    "task_relevance_evaluator": REPO_ROOT
    / "scripts/evaluate_go2_world_model_visual_domain_parity_task_relevance_v1.py",
    "task_relevance_collector": REPO_ROOT
    / "scripts/collect_go2_world_model_counterfactual_pilot_v1.py",
    "task_relevance_h6_dataset": REPO_ROOT
    / "lewm/datasets/go2_explicit_plan_discounted_successor_state_v27.py",
    "task_relevance_mask_benchmark": REPO_ROOT
    / "lewm/benchmarks/go2_rgb_recurrent_patch_memory_temporal_jepa_v1.py",
    "task_relevance_parity_authority_builder": REPO_ROOT
    / "scripts/build_go2_world_model_visual_domain_parity_authority_v1.py",
    "task_relevance_parity_evaluator": REPO_ROOT
    / "scripts/evaluate_go2_world_model_visual_domain_parity_v1.py",
    "task_relevance_parity_plan_builder": REPO_ROOT
    / "scripts/build_go2_world_model_visual_domain_parity_plan_v1.py",
    "task_relevance_parity_supervisor": REPO_ROOT
    / "scripts/run_go2_world_model_visual_domain_parity_authorized_v1.py",
    "task_relevance_probe": REPO_ROOT
    / "scripts/dev_probe_counterfactual_action_fidelity.py",
    "task_relevance_probe_evaluator": REPO_ROOT
    / "scripts/evaluate_go2_rgb_recurrent_patch_memory_temporal_jepa_v1.py",
    "task_relevance_probe_model": REPO_ROOT
    / "lewm/models/rgb_recurrent_patch_memory_temporal_jepa_v1.py",
    "task_relevance_probe_trainer": REPO_ROOT
    / "scripts/dev_train_temporal_jepa_scaled.py",
    "task_relevance_reference_renderer": REPO_ROOT / "scripts/render_replay_v03.py",
    "task_relevance_graphics_supervisor": REPO_ROOT
    / "scripts/run_go2_world_model_counterfactual_calibration_authorized_v1.py",
    "dense_shared_model": REPO_ROOT
    / "lewm/models/go2_dinov2_dense_shared_spatial_readout_calibration_v1.py",
    "dense_shared_evaluator": REPO_ROOT
    / "lewm/benchmarks/go2_dinov2_dense_shared_spatial_readout_calibration_v1.py",
    "dense_shared_runner": Path(__file__).resolve(),
    "dense_shared_replay": REPLAY_CLI,
    "dense_shared_model_test": REPO_ROOT
    / "lewm/tests/test_go2_dinov2_dense_shared_spatial_readout_calibration_model_v1.py",
    "dense_shared_evaluator_test": REPO_ROOT
    / "lewm/tests/test_go2_dinov2_dense_shared_spatial_readout_calibration_v1.py",
    "dense_shared_runner_test": REPO_ROOT
    / "lewm/tests/test_run_go2_dinov2_dense_shared_spatial_readout_calibration_v1.py",
    "dense_shared_replay_test": REPO_ROOT
    / (
        "lewm/tests/test_replay_go2_dinov2_dense_shared_spatial_readout_"
        "calibration_v1.py"
    ),
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
        "all_35_source_bindings_complete_exact_and_committed",
        "all_16_direct_input_bindings_complete_and_exact",
        "strict_split_root_loader_and_singleton_ssim_admission_preserved",
        "compatibility_replay_uses_bound_prior_evidence_without_rgb_or_encoder",
        "primary_and_replay_compatibility_receipts_precede_loader_return",
        "train_and_eval_cache_loaders_rehash_and_validate_artifact_order",
        "checkpoint_written_before_eval_cache_load",
        "dense_shared_scientific_protocol_matches_preregistration",
        "fresh_process_replay_retrains_and_recomputes_independently",
        "exact_eight_file_output_and_failure_terminal_are_fail_closed",
        "no_rgb_encoder_collection_protected_retry_or_resume_path",
        "focused_tests_passed",
        "compile_and_whitespace_checks_passed",
    }
)

REPLAY_REPRODUCTION_FIELDS = frozenset(
    {
        "pca_identity",
        "state_dict_identities",
        "step_counts",
        "per_seed_scores",
        "ensemble_scores",
        "selected_actions",
        "summaries",
        "bootstrap_intervals",
        "gates",
        "verdict",
        "exactly_reproduced",
    }
)


class DenseSharedCalibrationRunnerError(RuntimeError):
    """Raised when the one-shot authority or custody contract changes."""


def canonical_bytes_v1(value: object) -> bytes:
    """Return the evaluator's finite canonical JSON representation."""

    try:
        raw = evaluator.canonical_bytes_v1(value)
    except Exception as error:
        raise DenseSharedCalibrationRunnerError(
            "document is not finite canonical JSON"
        ) from error
    if not isinstance(raw, bytes):
        raise DenseSharedCalibrationRunnerError(
            "canonical evaluator returned a non-bytes value"
        )
    return raw


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
            raise DenseSharedCalibrationRunnerError(
                f"{label} names protected material"
            )


def _safe_path(path: Path, *, label: str, must_exist: bool = True) -> Path:
    selected = Path(os.path.abspath(os.fspath(path)))
    _reject_protected(selected, label=label)
    cursor = Path(selected.anchor)
    for part in selected.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise DenseSharedCalibrationRunnerError(
                f"{label} traverses a symlink"
            )
        if not cursor.exists():
            if must_exist:
                raise DenseSharedCalibrationRunnerError(f"{label} is absent")
            break
    if must_exist and not selected.exists():
        raise DenseSharedCalibrationRunnerError(f"{label} is absent")
    return selected


def file_binding_v1(path: Path) -> dict[str, Any]:
    selected = _safe_path(path, label="bound file")
    if not selected.is_file():
        raise DenseSharedCalibrationRunnerError("bound path is not a file")
    digest = hashlib.sha256()
    size = 0
    with selected.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {
        "path": str(selected),
        "sha256": digest.hexdigest(),
        "byte_count": size,
    }


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
        raise DenseSharedCalibrationRunnerError(f"{label} binding is malformed")
    actual = file_binding_v1(Path(str(value["path"])))
    if actual != dict(value):
        raise DenseSharedCalibrationRunnerError(f"{label} binding changed")
    return actual


def _read_bound_json(
    path: Path, *, expected_sha256: str, expected_byte_count: int, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = file_binding_v1(path)
    if (
        binding["sha256"] != expected_sha256
        or binding["byte_count"] != expected_byte_count
    ):
        raise DenseSharedCalibrationRunnerError(f"{label} caller binding changed")
    try:
        document = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DenseSharedCalibrationRunnerError(
            f"{label} is not valid JSON"
        ) from error
    if not isinstance(document, Mapping):
        raise DenseSharedCalibrationRunnerError(f"{label} is not a JSON object")
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
        raise DenseSharedCalibrationRunnerError(
            "scientific evaluator config is not a mapping"
        )
    result = {
        "scientific": dict(scientific),
        "direct_input_file_count": 16,
        "source_file_count": 35,
        "output_inventory": list(OUTPUT_NAMES),
        "replay_count": 1,
        "rocm_required": True,
        "rgb_access_permitted": False,
        "encoder_execution_permitted": False,
        "retry_or_resume_permitted": False,
        "compatibility_absolute_tolerance": compatibility.SSIM_ABSOLUTE_TOLERANCE,
        "compatibility_relative_tolerance": compatibility.SSIM_RELATIVE_TOLERANCE,
        "compatibility_allowed_differing_paths": [
            compatibility.SSIM_DOTTED_PATH
        ],
        "compatibility_mode": (
            "bound_prior_evidence_replay_no_rgb_no_encoder"
        ),
    }
    canonical_bytes_v1(result)
    return result


def _fixed_input_bindings_v1() -> dict[str, dict[str, Any]]:
    return {
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
        "prior_calibration_result": _binding(
            PRIOR_RESULT,
            "d87eed0cb8a4912be8fcf0bb2dd582a8394c363ad39cfd9cced8a4f0507a53ee",
            581_557,
        ),
        "prior_calibration_terminal": _binding(
            PRIOR_TERMINAL,
            "5bb8409a085917caee78b404534f5f3bf5537a928f8165793c34ce54a180f0a0",
            575,
        ),
        "prior_terminal_review": _binding(
            PRIOR_TERMINAL_REVIEW,
            "7074779bdc506548d903c0319b74243f2b2934a1888325f813ee52f5a115c679",
            14_382,
        ),
        "prior_compatibility_receipt": _binding(
            PRIOR_COMPATIBILITY_RECEIPT,
            "3bd0f06e2970966a9471f352a76cd6859580336d86a69dec945a989c971e0710",
            3_017,
        ),
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
        "posthoc_terminal_review": _binding(
            POSTHOC_TERMINAL_REVIEW,
            "bfd0250357d0f681c674db6c54ea4a8c4d5e617230332383beda3db3e0f38669",
            2_844,
        ),
        "posthoc_rgb_manifest": _binding(
            POSTHOC_RGB_MANIFEST,
            "5e03afa7665ffef54a1cab5e37135a18d42761bc844ecefacaa433f75a1b1f7e",
            1_880_307,
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
        "stored_task_relevance_result": _binding(
            STORED_TASK_RELEVANCE_RESULT,
            "5094104ac29b4652cd577015c5fbf23b42f0768c78a205cbf07a77d992339ca7",
            94_165,
        ),
        "stored_task_relevance_review": _binding(
            STORED_TASK_RELEVANCE_REVIEW,
            "29eb00a486604824effb56502194855553f87c81a9691d4075a5810273c92ca9",
            2_080,
        ),
    }


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
        or any(
            not isinstance(value, str) or not value.strip()
            for value in reviewer.values()
        )
        or not isinstance(review.get("checks"), Mapping)
        or set(review["checks"]) != SOURCE_REVIEW_CHECKS
        or any(value is not True for value in review["checks"].values())
    ):
        raise DenseSharedCalibrationRunnerError(
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
        "development_calibration_only",
        "authorizes_dense_readout_fitting",
        "authorizes_model_training",
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
        or authority.get("development_calibration_only") is not True
        or authority.get("authorizes_dense_readout_fitting") is not True
        or authority.get("authorizes_model_training") is not False
        or authority.get("authorizes_collection") is not False
        or authority.get("authorizes_rgb_access") is not False
        or authority.get("authorizes_encoder_execution") is not False
        or authority.get("authorizes_protected_access") is not False
        or authority.get("authorizes_retry_or_resume") is not False
        or authority.get("output_root") != str(DEFAULT_OUTPUT_ROOT.resolve())
        or authority.get("config") != config_v1()
    ):
        raise DenseSharedCalibrationRunnerError(
            "execution authority contract changed"
        )
    preregistration = _require_binding(
        authority["preregistration_binding"], label="preregistration"
    )
    if preregistration != _binding(
        PREREGISTRATION, PREREGISTRATION_SHA256, PREREGISTRATION_BYTE_COUNT
    ):
        raise DenseSharedCalibrationRunnerError(
            "authority does not bind the frozen preregistration"
        )
    inputs = authority.get("input_bindings")
    if not isinstance(inputs, Mapping) or dict(inputs) != _fixed_input_bindings_v1():
        raise DenseSharedCalibrationRunnerError("authority input closure changed")
    for label, item in inputs.items():
        _require_binding(item, label=f"input {label}")
    sources = authority.get("source_bindings")
    if not isinstance(sources, Mapping) or set(sources) != set(SOURCE_PATHS):
        raise DenseSharedCalibrationRunnerError("authority source closure changed")
    for label, expected_path in SOURCE_PATHS.items():
        item = _require_binding(sources[label], label=f"source {label}")
        if item["path"] != str(expected_path.resolve()):
            raise DenseSharedCalibrationRunnerError(f"source {label} path changed")
    review_binding = _require_binding(
        authority["source_review_binding"], label="source review"
    )
    if review_binding["path"] != str(SOURCE_REVIEW.resolve()):
        raise DenseSharedCalibrationRunnerError("source review path changed")
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
        or set(environment) != {"python", "torch", "hip", "numpy", "pillow"}
        or environment.get("python") != str(Path(sys.executable).resolve())
        or environment.get("torch") != torch.__version__
        or environment.get("hip") != torch.version.hip
        or environment.get("numpy") != np.__version__
        or environment.get("pillow") != PILLOW_VERSION
    ):
        raise DenseSharedCalibrationRunnerError("execution environment changed")
    commit = authority.get("git_commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
        or subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "merge-base",
                "--is-ancestor",
                commit,
                "HEAD",
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        != 0
    ):
        raise DenseSharedCalibrationRunnerError(
            "reviewed source commit is not an execution ancestor"
        )
    return authority, authority_binding


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


def _validate_prior_route_v1(authority: Mapping[str, Any]) -> None:
    inputs = authority["input_bindings"]
    result = _bound_document_v1(authority, "prior_calibration_result")
    terminal = _bound_document_v1(authority, "prior_calibration_terminal")
    review = _bound_document_v1(authority, "prior_terminal_review")
    receipt = _bound_document_v1(authority, "prior_compatibility_receipt")
    review_bindings = review.get("bindings")
    attempt = review.get("attempt_contract")
    if (
        result.get("schema") != prior_runner.SCHEMA
        or result.get("status") != prior_runner.STOP_STATUS
        or result.get("development_calibration_only") is not True
        or result.get("authorizes_model_training") is not False
        or terminal.get("schema") != prior_runner.TERMINAL_SCHEMA
        or terminal.get("status") != prior_runner.STOP_STATUS
        or terminal.get("deterministic_replay_passed") is not True
        or terminal.get("failure") is not None
        or terminal.get("result_binding")
        != inputs["prior_calibration_result"]
        or receipt.get("schema")
        != (
            "lewm_go2_dinov2_physical_readout_calibration_"
            "integrity_replacement_v1_compatibility_receipt_v1"
        )
        or receipt.get("status") != "PASS_PUBLISHED_BEFORE_CALIBRATION_EVAL_ACCESS"
        or review.get("schema")
        != (
            "lewm_go2_dinov2_physical_readout_calibration_"
            "integrity_replacement_v1_terminal_review_v1"
        )
        or review.get("status") != "PASS_COMPLETE_SCIENTIFIC_STOP_TERMINAL_REVIEW"
        or review.get("protected_material_opened") is not False
        or review.get("findings") != []
        or not isinstance(review_bindings, Mapping)
        or review_bindings.get("result") != inputs["prior_calibration_result"]
        or review_bindings.get("terminal") != inputs["prior_calibration_terminal"]
        or review_bindings.get("compatibility_receipt")
        != inputs["prior_compatibility_receipt"]
        or not isinstance(attempt, Mapping)
        or attempt.get("terminal_status") != prior_runner.STOP_STATUS
        or attempt.get("deterministic_replay_passed") is not True
        or attempt.get("in_attempt_cache_rehash_passed") is not True
        or attempt.get("independent_posthoc_replay_confirmation_passed")
        is not True
    ):
        raise DenseSharedCalibrationRunnerError(
            "prior calibration STOP route is not exact"
        )


def _load_stored_task_relevance_v1(
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    result = _bound_document_v1(authority, "stored_task_relevance_result")
    review = _bound_document_v1(authority, "stored_task_relevance_review")
    result_binding = authority["input_bindings"]["stored_task_relevance_result"]
    expected_review_binding = {
        "path": result_binding["path"],
        "file_sha256": result_binding["sha256"],
        "byte_count": result_binding["byte_count"],
    }
    if (
        review.get("schema")
        != (
            "lewm_go2_world_model_visual_domain_parity_"
            "task_relevant_input_adequacy_independent_review_v1"
        )
        or review.get("status")
        != (
            "PASS_INDEPENDENTLY_REVIEWED_TASK_RELEVANT_INPUT_"
            "ADEQUACY_DEVELOPMENT_ONLY"
        )
        or review.get("authority_granted_by_this_document") is not False
        or review.get("scientific_claim_granted_by_this_document") is not False
        or review.get("remaining_findings") != []
        or review.get("adequacy_result_binding") != expected_review_binding
    ):
        raise DenseSharedCalibrationRunnerError(
            "stored task-relevance review changed"
        )
    return result


def _compatibility_receipt_v1(
    *,
    phase: str,
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, Any],
    admission: Mapping[str, Any],
) -> dict[str, Any]:
    if phase not in {"primary", "replay"}:
        raise DenseSharedCalibrationRunnerError("compatibility phase changed")
    return {
        "schema": COMPATIBILITY_RECEIPT_SCHEMA,
        "status": COMPATIBILITY_RECEIPT_STATUS,
        "phase": phase,
        "citable_as_scientific_evidence": False,
        "publication_stage": (
            f"inside_task_relevance_compatibility_replay_before_{phase}_"
            "strict_loader_return"
        ),
        "authority_binding": dict(authority_binding),
        "preregistration_binding": dict(authority["preregistration_binding"]),
        "prior_terminal_review_binding": dict(
            authority["input_bindings"]["prior_terminal_review"]
        ),
        "prior_compatibility_receipt_binding": dict(
            authority["input_bindings"]["prior_compatibility_receipt"]
        ),
        "stored_task_relevance_result_binding": dict(
            authority["input_bindings"]["stored_task_relevance_result"]
        ),
        "stored_task_relevance_review_binding": dict(
            authority["input_bindings"]["stored_task_relevance_review"]
        ),
        "task_relevance_evaluator_source_binding": dict(
            authority["source_bindings"]["task_relevance_evaluator"]
        ),
        "environment": dict(authority["environment"]),
        "admission": dict(admission),
    }


def _task_relevance_call_bindings_v1(
    stored: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    bindings = stored.get("bindings")
    if not isinstance(bindings, Mapping):
        raise DenseSharedCalibrationRunnerError(
            "stored task-relevance bindings are absent"
        )
    requested: dict[str, Mapping[str, Any]] = {}
    for argument, label in (
        ("parity_result_binding", "parity_result"),
        ("terminal_failure_binding", "terminal_failure"),
        ("progression_analysis_binding", "progression_analysis"),
    ):
        item = bindings.get(label)
        if not isinstance(item, Mapping):
            raise DenseSharedCalibrationRunnerError(
                f"stored task-relevance {label} binding is absent"
            )
        requested[argument] = dict(item)
    return requested


def _replay_prior_compatibility_admission_v1(
    authority: Mapping[str, Any], stored: Mapping[str, Any]
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Replay the bound prior singleton admission without RGB or an encoder."""

    receipt = _bound_document_v1(authority, "prior_compatibility_receipt")
    required = {
        "schema",
        "status",
        "citable_as_scientific_evidence",
        "publication_stage",
        "authority_binding",
        "preregistration_binding",
        "original_failure_review_binding",
        "stored_task_relevance_result_binding",
        "stored_task_relevance_review_binding",
        "task_relevance_evaluator_source_binding",
        "environment",
        "admission",
    }
    admission = receipt.get("admission")
    if (
        set(receipt) != required
        or receipt.get("schema")
        != (
            "lewm_go2_dinov2_physical_readout_calibration_"
            "integrity_replacement_v1_compatibility_receipt_v1"
        )
        or receipt.get("status") != "PASS_PUBLISHED_BEFORE_CALIBRATION_EVAL_ACCESS"
        or receipt.get("citable_as_scientific_evidence") is not False
        or receipt.get("publication_stage")
        != "inside_task_relevance_evaluator_before_outer_loader_acceptance"
        or receipt.get("stored_task_relevance_result_binding")
        != authority["input_bindings"]["stored_task_relevance_result"]
        or receipt.get("stored_task_relevance_review_binding")
        != authority["input_bindings"]["stored_task_relevance_review"]
        or not isinstance(admission, Mapping)
    ):
        raise DenseSharedCalibrationRunnerError(
            "bound prior compatibility receipt changed"
        )
    recomputed = deepcopy(dict(stored))
    measurements = recomputed.get("measurements")
    pixels = measurements.get("pixels") if isinstance(measurements, Mapping) else None
    recomputed_ssim = admission.get(
        "recomputed_minimum_reference_candidate_rgb_ssim"
    )
    if not isinstance(pixels, dict) or not isinstance(recomputed_ssim, float):
        raise DenseSharedCalibrationRunnerError(
            "prior compatibility SSIM evidence changed"
        )
    pixels["minimum_reference_candidate_rgb_ssim"] = recomputed_ssim
    try:
        admitted, replayed_admission = (
            compatibility.admit_task_relevance_result_v1(
                stored=stored, recomputed=recomputed
            )
        )
    except compatibility.CompatibilityAdmissionError as error:
        raise DenseSharedCalibrationRunnerError(
            "prior singleton compatibility admission did not replay exactly"
        ) from error
    if (
        admitted is not stored
        or canonical_bytes_v1(replayed_admission)
        != canonical_bytes_v1(admission)
    ):
        raise DenseSharedCalibrationRunnerError(
            "prior singleton compatibility admission did not replay exactly"
        )
    return admitted, replayed_admission


@contextmanager
def scoped_primary_compatibility_admission_v1(
    authority: Mapping[str, Any],
    *,
    authority_binding: Mapping[str, Any],
) -> Iterator[dict[str, Any]]:
    """Replay prior compatibility inside the one strict-loader call."""

    output_root = Path(str(authority["output_root"]))
    receipt_path = output_root / "primary_compatibility_receipt.json"
    stored = _load_stored_task_relevance_v1(authority)
    admitted_document, replayed_admission = (
        _replay_prior_compatibility_admission_v1(authority, stored)
    )
    expected_call = _task_relevance_call_bindings_v1(stored)
    original_evaluator = task_relevance.evaluate_task_relevance_v1
    original_loader = prior_runner.screen_data.load_bound_posthoc_bundle_v1
    state: dict[str, Any] = {
        "evaluator_calls": 0,
        "loader_calls": 0,
        "receipt_binding": None,
        "admission": None,
    }

    def admitted_evaluator(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        state["evaluator_calls"] += 1
        if (
            state["evaluator_calls"] != 1
            or args
            or kwargs != expected_call
        ):
            raise DenseSharedCalibrationRunnerError(
                "task-relevance compatibility-replay call changed"
            )
        receipt = _compatibility_receipt_v1(
            phase="primary",
            authority=authority,
            authority_binding=authority_binding,
            admission=replayed_admission,
        )
        _write_json_exclusive(receipt_path, receipt)
        state["receipt_binding"] = file_binding_v1(receipt_path)
        state["admission"] = dict(replayed_admission)
        return admitted_document

    def admitted_loader(*args: Any, **kwargs: Any) -> object:
        state["loader_calls"] += 1
        if state["loader_calls"] != 1:
            raise DenseSharedCalibrationRunnerError(
                "strict posthoc loader call count changed"
            )
        bundle = original_loader(*args, **kwargs)
        if state["evaluator_calls"] != 1 or state["receipt_binding"] is None:
            raise DenseSharedCalibrationRunnerError(
                "primary compatibility receipt was not published before loader return"
            )
        return bundle

    task_relevance.evaluate_task_relevance_v1 = admitted_evaluator
    prior_runner.screen_data.load_bound_posthoc_bundle_v1 = admitted_loader
    try:
        yield state
    finally:
        task_relevance.evaluate_task_relevance_v1 = original_evaluator
        prior_runner.screen_data.load_bound_posthoc_bundle_v1 = original_loader


def _feature_plans_v1(
    bundle: object,
) -> tuple[tuple[object, ...], tuple[object, ...], object, object]:
    if getattr(bundle, "access_audit", {}).get("rgb_leaf_open_count") != 0:
        raise DenseSharedCalibrationRunnerError(
            "strict bundle loader opened an RGB leaf"
        )
    groups = getattr(bundle, "groups_by_role", {})
    if not isinstance(groups, Mapping) or set(groups) != {"train", "eval"}:
        raise DenseSharedCalibrationRunnerError("bounded bundle roles changed")
    train_groups = tuple(groups["train"])
    eval_groups = tuple(groups["eval"])
    if len(train_groups) != ROLE_STATE_COUNT or len(eval_groups) != ROLE_STATE_COUNT:
        raise DenseSharedCalibrationRunnerError(
            "bounded bundle state counts changed"
        )
    plans = prior_evaluator.build_calibration_feature_plans_v1(
        train_groups, eval_groups
    )
    train_plan, eval_plan = prior_runner._split_plans(plans)  # noqa: SLF001
    train_ids = prior_runner._artifact_ids(  # noqa: SLF001
        train_plan, role="train"
    )
    eval_ids = prior_runner._artifact_ids(eval_plan, role="eval")  # noqa: SLF001
    if (
        len(train_ids) != ROLE_ARTIFACT_COUNT
        or len(eval_ids) != ROLE_ARTIFACT_COUNT
        or set(train_ids) & set(eval_ids)
    ):
        raise DenseSharedCalibrationRunnerError(
            "role feature plans changed or overlap"
        )
    return train_groups, eval_groups, train_plan, eval_plan


def _validate_feature_tensor_v1(features: torch.Tensor, *, role: str) -> None:
    if (
        not isinstance(features, torch.Tensor)
        or tuple(features.shape) != (ROLE_ARTIFACT_COUNT, *TOKEN_SHAPE)
        or features.dtype != torch.float16
        or features.device.type != "cpu"
        or not bool(torch.isfinite(features).all())
    ):
        raise DenseSharedCalibrationRunnerError(f"{role} feature tensor changed")
    maximum_error = 0.0
    for start in range(0, ROLE_ARTIFACT_COUNT, 64):
        norms = torch.linalg.vector_norm(
            features[start : start + 64].to(dtype=torch.float32), dim=-1
        )
        maximum_error = max(
            maximum_error, float(torch.max(torch.abs(norms - 1.0)).item())
        )
    if maximum_error > MAX_TOKEN_NORM_ERROR:
        raise DenseSharedCalibrationRunnerError(
            f"{role} feature token normalization changed"
        )


def _load_train_cache_v1(
    bundle: object, train_plan: object
) -> tuple[torch.Tensor, dict[str, Any]]:
    try:
        features, receipt = prior_runner._load_train_cache_v1(  # noqa: SLF001
            bundle, train_plan
        )
    except Exception as error:
        raise DenseSharedCalibrationRunnerError(
            "strict train-cache loader failed"
        ) from error
    _validate_feature_tensor_v1(features, role="train")
    return features, receipt


def _load_eval_cache_v1(
    authority: Mapping[str, Any], eval_plan: object
) -> tuple[torch.Tensor, dict[str, Any]]:
    receipt = _bound_document_v1(authority, "eval_cache_receipt")
    try:
        features = prior_runner._load_eval_feature_cache_v1(  # noqa: SLF001
            receipt, eval_plan
        )
    except Exception as error:
        raise DenseSharedCalibrationRunnerError(
            "strict eval-cache loader failed"
        ) from error
    if receipt.get("binding") != authority["input_bindings"]["eval_cache"]:
        raise DenseSharedCalibrationRunnerError(
            "evaluation cache receipt binding changed"
        )
    _validate_feature_tensor_v1(features, role="eval")
    return features, receipt


def _authorized_device_v1() -> torch.device:
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise DenseSharedCalibrationRunnerError(
            "the authorized dense readout calibration requires ROCm"
        )
    return torch.device("cuda")


def _execution_bindings_unchanged(
    authority: Mapping[str, Any], *, authority_binding: Mapping[str, Any]
) -> None:
    closure: list[tuple[str, Mapping[str, Any]]] = [
        ("execution authority", authority_binding),
        ("preregistration", authority["preregistration_binding"]),
        ("source review", authority["source_review_binding"]),
    ]
    closure.extend(
        (f"source {label}", item)
        for label, item in authority["source_bindings"].items()
    )
    closure.extend(
        (f"input {label}", item)
        for label, item in authority["input_bindings"].items()
    )
    for label, expected in closure:
        if file_binding_v1(Path(str(expected["path"]))) != dict(expected):
            raise DenseSharedCalibrationRunnerError(
                f"{label} changed during execution"
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
        raise DenseSharedCalibrationRunnerError(
            f"fresh replay process failed: {detail[-1000:]}"
        )


def _validate_replay_compatibility_receipt_v1(
    receipt: Mapping[str, Any],
    *,
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, Any],
    primary_admission: Mapping[str, Any],
) -> None:
    expected = _compatibility_receipt_v1(
        phase="replay",
        authority=authority,
        authority_binding=authority_binding,
        admission=primary_admission,
    )
    if dict(receipt) != expected:
        raise DenseSharedCalibrationRunnerError(
            "replay compatibility receipt changed"
        )


def _validate_replay_v1(
    replay: Mapping[str, Any],
    *,
    authority_binding: Mapping[str, Any],
    checkpoint_binding: Mapping[str, Any],
    evaluation_binding: Mapping[str, Any],
    replay_compatibility_binding: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> None:
    required = {
        "schema",
        "status",
        "citable_as_scientific_evidence",
        "authority_binding",
        "checkpoint_binding",
        "primary_evaluation_binding",
        "compatibility_receipt_binding",
        "recomputed_evaluation",
        "reproduction",
        "protected_material_opened",
        "rgb_access",
    }
    reproduction = replay.get("reproduction")
    if (
        set(replay) != required
        or replay.get("schema") != REPLAY_SCHEMA
        or replay.get("status") != REPLAY_STATUS
        or replay.get("citable_as_scientific_evidence") is not False
        or replay.get("authority_binding") != authority_binding
        or replay.get("checkpoint_binding") != checkpoint_binding
        or replay.get("primary_evaluation_binding") != evaluation_binding
        or replay.get("compatibility_receipt_binding")
        != replay_compatibility_binding
        or replay.get("protected_material_opened") is not False
        or replay.get("rgb_access") != {"train": 0, "eval": 0}
        or not isinstance(reproduction, Mapping)
        or set(reproduction) != REPLAY_REPRODUCTION_FIELDS
        or any(value is not True for value in reproduction.values())
        or canonical_bytes_v1(replay.get("recomputed_evaluation"))
        != canonical_bytes_v1(evaluation)
    ):
        raise DenseSharedCalibrationRunnerError(
            "fresh replay did not reproduce the primary evaluation exactly"
        )


def _verdict_status_v1(
    verdict: Mapping[str, Any],
    *,
    evaluation: Mapping[str, Any],
    deterministic_replay_passed: bool,
) -> str:
    evaluation_gates = evaluation.get("gates")
    if not isinstance(evaluation_gates, Mapping):
        raise DenseSharedCalibrationRunnerError("evaluation gates are malformed")
    expected_gates = {
        "1_infrastructure_and_custody": {"passed": True},
        **dict(evaluation_gates),
        "7_deterministic_replay": {"passed": deterministic_replay_passed},
    }
    expected_passed = all(
        isinstance(gate, Mapping) and gate.get("passed") is True
        for gate in expected_gates.values()
    )
    expected_status = (
        FAIL_STATUS
        if not deterministic_replay_passed
        else PASS_STATUS if expected_passed else STOP_STATUS
    )
    if (
        set(verdict) != {"gates", "passed", "terminal_status"}
        or verdict.get("gates") != expected_gates
        or verdict.get("passed") is not expected_passed
        or verdict.get("terminal_status") != expected_status
        or expected_status not in TERMINAL_STATUSES
    ):
        raise DenseSharedCalibrationRunnerError(
            "scientific verdict contract changed"
        )
    return expected_status


def _assert_inventory_v1(output_root: Path, expected: set[str]) -> None:
    observed: list[str] = []
    with os.scandir(output_root) as entries:
        for entry in entries:
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                raise DenseSharedCalibrationRunnerError(
                    "attempt root contains a non-file"
                )
            observed.append(entry.name)
    if set(observed) != expected or len(observed) != len(expected):
        raise DenseSharedCalibrationRunnerError(
            f"attempt inventory changed: {sorted(observed)}"
        )


def execute_v1(
    authority: Mapping[str, Any], *, authority_binding: Mapping[str, Any]
) -> dict[str, Any]:
    output_root = _safe_path(
        Path(str(authority["output_root"])),
        label="dense shared calibration output",
        must_exist=False,
    )
    if output_root != DEFAULT_OUTPUT_ROOT.resolve():
        raise DenseSharedCalibrationRunnerError("calibration output root changed")
    _safe_path(
        output_root.parent,
        label="dense shared calibration output parent",
        must_exist=False,
    )
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
    _validate_prior_route_v1(authority)
    with scoped_primary_compatibility_admission_v1(
        authority, authority_binding=authority_binding
    ) as compatibility_state:
        bundle = prior_runner.screen_data.load_bound_posthoc_bundle_v1()
    train_groups, eval_groups, train_plan, eval_plan = _feature_plans_v1(bundle)
    train_features, train_receipt = _load_train_cache_v1(bundle, train_plan)
    device = _authorized_device_v1()
    implementation_source_binding = authority["source_bindings"][
        "dense_shared_evaluator"
    ]
    checkpoint_payload = evaluator.fit_primary_checkpoint_v1(
        train_groups,
        train_features,
        device,
        implementation_source_binding=implementation_source_binding,
    )
    if not isinstance(checkpoint_payload, Mapping):
        raise DenseSharedCalibrationRunnerError(
            "scientific evaluator returned a non-mapping checkpoint"
        )
    checkpoint_path = output_root / "pca_readout_checkpoint.pt"
    _save_torch_exclusive(checkpoint_path, checkpoint_payload)
    checkpoint_binding = file_binding_v1(checkpoint_path)

    # The checkpoint is durable before this function may torch.load the eval cache.
    eval_features, eval_receipt = _load_eval_cache_v1(authority, eval_plan)
    evaluation = evaluator.evaluate_primary_checkpoint_v1(
        checkpoint_payload,
        train_groups,
        eval_groups,
        train_features,
        eval_features,
        device,
        implementation_source_binding=implementation_source_binding,
    )
    if not isinstance(evaluation, Mapping):
        raise DenseSharedCalibrationRunnerError(
            "scientific evaluator returned a non-object evaluation"
        )
    evaluation = dict(evaluation)
    canonical_bytes_v1(evaluation)
    evaluation_path = output_root / "evaluation.json"
    _write_json_exclusive(evaluation_path, evaluation)
    evaluation_binding = file_binding_v1(evaluation_path)
    del checkpoint_payload, train_features, eval_features
    if device.type == "cuda":
        torch.cuda.empty_cache()

    _execution_bindings_unchanged(
        authority, authority_binding=authority_binding
    )
    _launch_replay_v1(
        authority_binding=authority_binding,
        checkpoint_binding=checkpoint_binding,
        evaluation_binding=evaluation_binding,
    )
    replay_receipt, replay_receipt_binding = _read_new_json(
        output_root / "replay_compatibility_receipt.json",
        label="replay compatibility receipt",
    )
    primary_admission = compatibility_state.get("admission")
    if not isinstance(primary_admission, Mapping):
        raise DenseSharedCalibrationRunnerError(
            "primary compatibility admission was not retained"
        )
    _validate_replay_compatibility_receipt_v1(
        replay_receipt,
        authority=authority,
        authority_binding=authority_binding,
        primary_admission=primary_admission,
    )
    replay, replay_binding = _read_new_json(
        output_root / "replay.json", label="fresh replay"
    )
    _validate_replay_v1(
        replay,
        authority_binding=authority_binding,
        checkpoint_binding=checkpoint_binding,
        evaluation_binding=evaluation_binding,
        replay_compatibility_binding=replay_receipt_binding,
        evaluation=evaluation,
    )
    _execution_bindings_unchanged(
        authority, authority_binding=authority_binding
    )
    verdict = evaluator.verdict_v1(
        evaluation,
        infrastructure_checks_passed=True,
        deterministic_replay_passed=True,
    )
    if not isinstance(verdict, Mapping):
        raise DenseSharedCalibrationRunnerError(
            "scientific evaluator returned a non-object verdict"
        )
    status = _verdict_status_v1(
        verdict,
        evaluation=evaluation,
        deterministic_replay_passed=True,
    )
    primary_receipt_binding = file_binding_v1(
        output_root / "primary_compatibility_receipt.json"
    )
    report = {
        "schema": SCHEMA,
        "status": status,
        "citable_as_scientific_evidence": False,
        "development_calibration_only": True,
        "authorizes_model_training": False,
        "authority_binding": dict(authority_binding),
        "preregistration_binding": dict(authority["preregistration_binding"]),
        "source_bundle_manifest": dict(getattr(bundle, "manifest_binding")),
        "input_bindings": dict(authority["input_bindings"]),
        "artifact_bindings": {
            "primary_compatibility_receipt": primary_receipt_binding,
            "pca_readout_checkpoint": checkpoint_binding,
            "evaluation": evaluation_binding,
            "replay_compatibility_receipt": replay_receipt_binding,
            "replay": replay_binding,
        },
        "cache_receipts": {
            "train": train_receipt,
            "eval": eval_receipt,
        },
        "rgb_access": {"primary": 0, "replay": 0},
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
        "authorizes_model_training": False,
        "result_binding": file_binding_v1(output_root / "result.json"),
        "deterministic_replay_passed": True,
        "failure": None,
    }
    _write_json_exclusive(output_root / "terminal.json", terminal)
    _assert_inventory_v1(output_root, set(OUTPUT_NAMES))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--expected-authority-sha256", required=True)
    parser.add_argument("--expected-authority-byte-count", type=int, required=True)
    args = parser.parse_args(argv)
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
                    "authorizes_model_training": False,
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
    "COMPATIBILITY_RECEIPT_SCHEMA",
    "COMPATIBILITY_RECEIPT_STATUS",
    "DEFAULT_OUTPUT_ROOT",
    "DenseSharedCalibrationRunnerError",
    "FAIL_STATUS",
    "OUTPUT_NAMES",
    "PASS_STATUS",
    "PREREGISTRATION_BYTE_COUNT",
    "PREREGISTRATION_SHA256",
    "REPLAY_SCHEMA",
    "REPLAY_STATUS",
    "RESERVATION_SCHEMA",
    "SCHEMA",
    "SOURCE_PATHS",
    "SOURCE_REVIEW_CHECKS",
    "SOURCE_REVIEW_FIELDS",
    "SOURCE_REVIEW_SCHEMA",
    "SOURCE_REVIEW_STATUS",
    "STOP_STATUS",
    "TERMINAL_SCHEMA",
    "canonical_bytes_v1",
    "config_v1",
    "execute_v1",
    "file_binding_v1",
    "scoped_primary_compatibility_admission_v1",
]
