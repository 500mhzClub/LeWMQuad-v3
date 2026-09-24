#!/usr/bin/env python3
"""Issue and execute the prospective one-model small-completion successor.

The five explicit stages preserve the pre-outcome authority boundary:

* ``issue-report`` installs the bounded source/formulation coupling report;
* ``issue-amendment`` installs the clean-source execution amendment;
* ``issue-source-correction`` preserves that immutable V1 authority and
  installs its narrowly source-corrected V2 successor;
* ``issue-preplan-integration-correction`` preserves immutable V2 and installs
  the orthogonal canonical-boundary validation correction; and
* ``solve-and-continue`` validates the synthetic fixtures, opens the frozen
  scientific masks, freezes one exact model plan, performs the one global
  solve, and continues directly through the already-authorised scorer stages
  only when the result is feasible.

Importing this module opens no generated artifact, scientific mask, outcome,
scorer, predictor, frame, latent, or sealed material and invokes no solver.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from scripts import build_go2_branch_corpus_v1_2 as BUILDER
from lewm.oracle import (
    go2_small_completion_global_execution_amendment_v1 as AUTHORITY,
)
from lewm.oracle import go2_small_completion_global_exact_model_v1 as MODEL


PLAN_SCHEMA = "go2_small_completion_global_exact_runner_plan_v1"
PLAN_STATUS = "FROZEN_PRE_SOLVE"
PLAN_SELF_KEY = "global_exact_model_plan_digest"
TERMINAL_SCHEMA = "go2_small_completion_global_exact_runner_terminal_v1"
TERMINAL_STATUS = "PASS"
TERMINAL_SELF_KEY = "global_exact_terminal_result_digest"
INFEASIBILITY_SCHEMA = TERMINAL_SCHEMA
INFEASIBILITY_STATUS = "INFEASIBLE"
INFEASIBILITY_SELF_KEY = TERMINAL_SELF_KEY

_PLAN_LABEL = "global_exact_model_plan"
_TERMINAL_LABEL = "global_exact_terminal_result"
_INFEASIBILITY_LABEL = "global_exact_terminal_infeasibility"

_PRE_MASK_CONTEXT_KEYS = frozenset({
    "predecessor_scientific_input_bindings", "scientific_contract_bindings",
    "preoutcome_input_bindings", "candidate_outcomes_consumed",
    "scientific_masks_accessed", "coupling_report",
    "coupling_report_binding", "execution_amendment",
})
_MASK_CONTEXT_KEYS = _PRE_MASK_CONTEXT_KEYS | {"inputs", "preserved_vectors"}

_RUNNER_PLAN_KEYS = frozenset({
    "schema", "status", "source_repository_commit",
    "coupling_report_digest", "execution_amendment_digest",
    "scientific_contract_bindings_digest",
    "preoutcome_input_bindings_digest",
    "immutable_predecessor_lineage_digest", "fixture_suite_result",
    "fixture_suite_digest", "production_instance_digest",
    "model_execution_plan", "model_execution_plan_digest",
    "selected_execution_method", "external_combination_enumeration",
    "performance_benchmark_run", "v1_or_v2_retried",
    "candidate_outcomes_consumed", "scientific_masks_accessed",
    "scientific_masks_accessed_after_amendment",
    "scientific_masks_are_frozen_model_inputs",
    "downstream_runtime_contracts", "downstream_stage_runtime_roles",
    "downstream_uses_global_solver_interpreter",
    "downstream_metric_in_selection", PLAN_SELF_KEY,
})
_FEASIBLE_TERMINAL_KEYS = frozenset({
    "schema", "status", "global_exact_model_plan_digest",
    "execution_amendment_digest", "model_execution_plan_digest",
    "model_execution_result", "model_execution_result_digest",
    "materialized_allocation_digest", "selected_scene_ids",
    "selected_scene_rows", "elapsed_wall_s", "elapsed_cpu_s",
    "exact_infeasibility_proved", "candidate_outcomes_consumed",
    "scientific_masks_accessed", "branch_labels_read",
    "scorer_or_predictor_accessed", TERMINAL_SELF_KEY,
})
_INFEASIBLE_TERMINAL_KEYS = frozenset({
    "schema", "status", "global_exact_model_plan_digest",
    "execution_amendment_digest", "model_execution_plan_digest",
    "model_execution_result", "model_execution_result_digest",
    "elapsed_wall_s", "elapsed_cpu_s", "exact_infeasibility_proved",
    "scientific_conditions_relaxed", "automatic_selector_revision",
    "candidate_outcomes_consumed", "scientific_masks_accessed",
    "branch_labels_read", "scorer_or_predictor_accessed",
    INFEASIBILITY_SELF_KEY,
})

_QUALIFICATION_CRITERION_KEYS = frozenset({
    "progress_spearman_ge_0.50",
    "safety_auc_ge_0.75",
    "safety_calibration_le_0.10",
    "completion_auc_ge_0.75",
    "completion_calibration_le_0.10",
    "composite_pairwise_ge_0.65",
    "beats_no_latent_baseline_by_0.05",
    "completion_labels_not_degenerate",
})
_QUALIFICATION_SCHEMA = "go2_utility_scorer_v1_2_qualification"
_DEVELOPMENT_RESULT_SCHEMA = (
    "go2_utility_scorer_counterfactual_development_transfer_result_v1_2"
)
_DEVELOPMENT_STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
DOWNSTREAM_VALIDATION_SCHEMA = (
    "go2_small_completion_global_exact_downstream_validation_v1"
)
DOWNSTREAM_VALIDATION_STATUS = "PASS_BOUND_DOWNSTREAM_VALIDATION"
DOWNSTREAM_VALIDATION_SELF_KEY = "downstream_validation_digest"
DOWNSTREAM_RUNTIME_PROBE_SCHEMA = (
    "go2_small_completion_global_exact_downstream_runtime_probe_v1"
)
DOWNSTREAM_RUNTIME_PROBE_STATUS = "PASS_BOUND_DOWNSTREAM_RUNTIME"
DOWNSTREAM_RUNTIME_PROBE_SELF_KEY = "downstream_runtime_probe_digest"
_HEX = frozenset("0123456789abcdef")
RUNNER_RELATIVE_PATH = Path(
    "scripts/run_go2_small_completion_global_exact_v1.py")
_DOWNSTREAM_COMMAND_STAGE_NAMES = (
    "six_branch_smoke",
    "smoke_encoding",
    "full_720_branch_corpus",
    "full_latent_encoding",
    "scorer_training_and_qualification",
    "development_transfer",
)
_DOWNSTREAM_RUNTIME_ROLES = ("genesis", "rocm")

_QUALIFICATION_BASE_KEYS = frozenset({
    "schema", "status", "training_run_digest",
    "scorer_contract_v1_2_digest",
    "frozen_scorer_fit_allocation_design_digest", "corpus_bindings",
    "candidate_allocator_contract_digest",
    "candidate_allocation_amendment_digest",
    "candidate_allocation_post_identity_validation_digest",
    "pre_identity_allocation_validation_digest",
    "invalid_scorer_identity_exclusion_digest", "target_encoder_digest",
    "target_encoder_checkpoint_sha256", "render_contract_digest",
    "textured_v03_renderer_contract_digest", "preprocess_contract_digest",
    "preprocessing_digest", "preprocess", "target_normalisation",
    "fit_states", "calibration_states", "fit_rows", "calibration_rows",
    "scene_disjoint", "label_distributions",
    "completion_prevalence_by_split_and_family", "latent_scorer",
    "no_latent_baseline", "baseline_dominance_pairwise",
    "paired_latent_vs_no_latent_calibration", "no_latent_baseline_package",
    "criterion_details", "criteria", "qualified",
    "qualification_evaluations", "qualification_input",
    "epoch_selection_permitted", "initialisations", "training_receipts",
    "final_state_digests", "scorer_package_sha256",
    "failed_scorer_sha256", "runtime", "storage",
    "predictor_checkpoints_loaded", "qualification_report_digest",
})


class GlobalExactRunnerError(RuntimeError):
    """The prospective authority, exact model, or continuation is invalid."""


CommandRunner = Callable[[Sequence[str], Path], int]
DownstreamValidationInvoker = Callable[
    [str, Path, Path, Any], Mapping[str, Any]]
DownstreamRuntimeProbeInvoker = Callable[
    [str, Path, Path, Any], Mapping[str, Any]]


def _json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    try:
        if pretty:
            return (json.dumps(
                value, indent=2, sort_keys=True, ensure_ascii=True,
                allow_nan=False,
            ) + "\n").encode("utf-8")
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GlobalExactRunnerError("value is not canonical JSON") from exc


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _signed(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = dict(payload)
    if key in result:
        raise GlobalExactRunnerError(f"{key} was supplied before signing")
    result[key] = canonical_digest(result)
    return result


def _without(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    return {name: value for name, value in payload.items() if name != key}


def _is_digest(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in _HEX for character in value))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_managed_file(root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise GlobalExactRunnerError(f"{label} path is malformed")
    root = root.resolve(strict=True)
    raw = Path(value)
    path = raw if raw.is_absolute() else root / raw
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise GlobalExactRunnerError(f"{label} escapes the repository") from exc
    pinned = _pinned_relative(root, relative, label=label)
    if not pinned.is_file() or pinned.is_symlink():
        raise GlobalExactRunnerError(f"{label} is unavailable")
    return pinned


def _runtime_relative(authority: Any, label: str) -> Path:
    matches = [Path(path) for row_label, path, kind
               in authority.NEW_RUNTIME_OUTPUT_PATHS
               if row_label == label and kind == "file"]
    if len(matches) != 1:
        raise GlobalExactRunnerError(
            f"authority runtime path is not unique: {label}")
    return matches[0]


def _is_forbidden_custody_component(component: str) -> bool:
    return (component == "sealed" or component == "sealed_test.json"
            or component.startswith("sealed_"))


def _pinned_relative(root: Path, relative: Path, *, label: str) -> Path:
    """Pin one named repository-relative path without resolving broad trees."""

    root = Path(root).resolve(strict=True)
    relative = Path(relative)
    if relative.is_absolute() or not relative.parts or any(
            part in {"", ".", ".."}
            or _is_forbidden_custody_component(part)
            for part in relative.parts):
        raise GlobalExactRunnerError(f"{label} path is not repository-relative")
    managed_matches = []
    for managed_root in AUTHORITY.MANAGED_GENERATED_ROOTS:
        try:
            relative.relative_to(managed_root)
            managed_matches.append(managed_root)
        except ValueError:
            continue
    if managed_matches:
        if len(managed_matches) != 1:
            raise GlobalExactRunnerError(
                f"{label} matches multiple managed generated roots")
        try:
            return AUTHORITY._pin_generated(root, relative, label=label)
        except AUTHORITY.GlobalExecutionAmendmentError as exc:
            raise GlobalExactRunnerError(
                f"{label} managed generated path is invalid") from exc
    candidate = root / relative
    cursor = root
    for component in relative.parts[:-1]:
        cursor = cursor / component
        if cursor.exists() and cursor.is_symlink():
            raise GlobalExactRunnerError(f"{label} parent is symlinked")
    return candidate


def _load_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    if not path.is_file() or path.is_symlink():
        raise GlobalExactRunnerError(f"{label} is unavailable")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GlobalExactRunnerError(f"{label} JSON is corrupt") from exc
    if not isinstance(value, dict):
        raise GlobalExactRunnerError(f"{label} root is not a mapping")
    return value, raw


def _exclusive_json(path: Path, payload: Mapping[str, Any], *, label: str
                    ) -> dict[str, Any]:
    """Install one immutable JSON object with O_EXCL, fsync, and exact reopen."""

    expected = dict(payload)
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise GlobalExactRunnerError(f"{label} parent is unavailable")
    if path.is_symlink():
        raise GlobalExactRunnerError(f"{label} path is symlinked")
    encoded = _json_bytes(expected, pretty=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o444)
    except FileExistsError as exc:
        raise GlobalExactRunnerError(f"{label} already exists") from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(path, 0o444, follow_symlinks=False)
        directory_fd = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        # Never delete or replace a partial immutable artifact.  Its presence
        # remains an explicit integrity failure for the next invocation.
        raise
    reopened, raw = _load_json(path, label=label)
    if (reopened != expected or raw != encoded
            or stat.S_IMODE(path.stat().st_mode) & 0o222):
        raise GlobalExactRunnerError(f"{label} exact read-only reopen changed")
    return reopened


def _install_or_reopen(path: Path, payload: Mapping[str, Any], *, label: str
                       ) -> dict[str, Any]:
    """Install a plan once, or require an existing plan to be byte-exact."""

    if path.exists() or path.is_symlink():
        reopened, raw = _load_json(path, label=label)
        if reopened != dict(payload) or raw != _json_bytes(dict(payload), pretty=True):
            raise GlobalExactRunnerError(f"existing {label} differs")
        if stat.S_IMODE(path.stat().st_mode) & 0o222:
            raise GlobalExactRunnerError(f"existing {label} is writable")
        return reopened
    return _exclusive_json(path, payload, label=label)


def _require_context(context: Mapping[str, Any], *, masks: bool,
                     authority: Any) -> dict[str, Any]:
    if not isinstance(context, Mapping):
        raise GlobalExactRunnerError("execution context is not a mapping")
    result = dict(context)
    required = {
        "coupling_report", "execution_amendment",
        "scientific_contract_bindings", "preoutcome_input_bindings",
        "candidate_outcomes_consumed", "scientific_masks_accessed",
    }
    if masks:
        required.update({"inputs", "preserved_vectors"})
    if not required.issubset(result):
        raise GlobalExactRunnerError("execution context is incomplete")
    expected_keys = _MASK_CONTEXT_KEYS if masks else _PRE_MASK_CONTEXT_KEYS
    if set(result) != expected_keys:
        raise GlobalExactRunnerError("execution context surface changed")
    report = result["coupling_report"]
    amendment = result["execution_amendment"]
    report_binding = result["coupling_report_binding"]
    v1_authority = (amendment.get("v1_execution_authority")
                    if isinstance(amendment, Mapping) else None)
    v1_amendment = (v1_authority.get("execution_amendment")
                    if isinstance(v1_authority, Mapping) else None)
    v1_report_binding = (v1_authority.get(
        "coupling_report_artifact_binding")
        if isinstance(v1_authority, Mapping) else None)
    v1_amendment_binding = (v1_authority.get(
        "execution_amendment_artifact_binding")
        if isinstance(v1_authority, Mapping) else None)
    source_correction = (amendment.get("source_correction")
                         if isinstance(amendment, Mapping) else None)
    failed_attempt = (amendment.get("failed_attempt_disposition")
                      if isinstance(amendment, Mapping) else None)
    immutable_v2 = (amendment.get("immutable_v2_execution_authority")
                    if isinstance(amendment, Mapping) else None)
    immutable_v2_payload = (immutable_v2.get("payload")
                            if isinstance(immutable_v2, Mapping) else None)
    immutable_v2_binding = (immutable_v2.get("binding")
                            if isinstance(immutable_v2, Mapping) else None)
    post_install_failure = (amendment.get("v2_post_install_reopen_failure")
                            if isinstance(amendment, Mapping) else None)
    preplan_failed_attempt = (amendment.get(
        "post_v2_preplan_failed_attempt_disposition")
        if isinstance(amendment, Mapping) else None)
    preplan_correction = (amendment.get("preplan_integration_correction")
                          if isinstance(amendment, Mapping) else None)
    issuance_boundary = (amendment.get("issuance_boundary")
                         if isinstance(amendment, Mapping) else None)
    if (not isinstance(report, Mapping) or not isinstance(amendment, Mapping)
            or not isinstance(report_binding, Mapping)
            or report.get("classification") != "COUPLED"
            or amendment.get("schema")
            != authority.PREPLAN_INTEGRATION_CORRECTION_SCHEMA
            or amendment.get("status")
            != authority.PREPLAN_INTEGRATION_CORRECTION_STATUS
            or amendment.get("amendment_version") != 2
            or amendment.get("preplan_integration_correction_version") != 1
            or amendment.get("selected_execution_method", {}).get("method")
            != "ONE_GLOBAL_EXACT_FEASIBILITY_MODEL"
            or not isinstance(v1_authority, Mapping)
            or v1_authority.get("coupling_report") != report
            or not isinstance(v1_amendment, Mapping)
            or not isinstance(v1_report_binding, Mapping)
            or not isinstance(v1_amendment_binding, Mapping)
            or v1_report_binding.get("self_digest")
            != authority.ORIGINAL_COUPLING_REPORT_ARTIFACT_BINDING[
                "self_digest"]
            or v1_amendment_binding.get("self_digest")
            != authority.ORIGINAL_EXECUTION_AMENDMENT_ARTIFACT_BINDING[
                "self_digest"]
            or v1_amendment.get(authority.AMENDMENT_SELF_KEY)
            != authority.ORIGINAL_EXECUTION_AMENDMENT_ARTIFACT_BINDING[
                "self_digest"]
            or report_binding.get("coupling_report_digest")
            != report.get(authority.REPORT_SELF_KEY)
            or not isinstance(source_correction, Mapping)
            or amendment.get("source_correction_digest")
            != canonical_digest(source_correction)
            or source_correction.get("scientific_contract_changed") is not False
            or source_correction.get(
                "candidate_outcome_or_downstream_metric_used") is not False
            or not isinstance(failed_attempt, Mapping)
            or dict(failed_attempt)
            != dict(authority.FAILED_SOURCE_TRANSITION_DISPOSITION)
            or amendment.get("failed_attempt_disposition_digest")
            != canonical_digest(failed_attempt)
            or not isinstance(immutable_v2, Mapping)
            or set(immutable_v2) != {"payload", "binding"}
            or not isinstance(immutable_v2_payload, Mapping)
            or immutable_v2_payload.get("schema") != authority.AMENDMENT_V2_SCHEMA
            or immutable_v2_payload.get("status") != authority.AMENDMENT_V2_STATUS
            or immutable_v2_payload.get(authority.AMENDMENT_SELF_KEY)
            != authority.IMMUTABLE_V2_EXECUTION_AMENDMENT_ARTIFACT_BINDING[
                "self_digest"]
            or immutable_v2_binding
            != authority.IMMUTABLE_V2_EXECUTION_AMENDMENT_ARTIFACT_BINDING
            or post_install_failure != authority.V2_POST_INSTALL_REOPEN_FAILURE
            or amendment.get("v2_post_install_reopen_failure_digest")
            != canonical_digest(post_install_failure)
            or preplan_failed_attempt
            != authority.POST_V2_PREPLAN_FAILED_ATTEMPT_DISPOSITION
            or amendment.get(
                "post_v2_preplan_failed_attempt_disposition_digest")
            != canonical_digest(preplan_failed_attempt)
            or not isinstance(preplan_correction, Mapping)
            or amendment.get("preplan_integration_correction_digest")
            != canonical_digest(preplan_correction)
            or preplan_correction.get("scientific_contract_changed") is not False
            or preplan_correction.get(
                "candidate_outcome_or_downstream_metric_used") is not False
            or preplan_correction.get(
                "builder_optional_candidate_projection_changed") is not False
            or not isinstance(issuance_boundary, Mapping)
            or issuance_boundary.get(
                "immutable_v1_and_v2_authorities_preserved") is not True
            or issuance_boundary.get(
                "historical_scientific_masks_accessed") is not True
            or issuance_boundary.get(
                "scientific_masks_accessed_during_this_issuance") is not False
            or issuance_boundary.get(
                "new_attempt_mask_context_started") is not False
            or issuance_boundary.get("candidate_outcomes_consumed") is not False
            or issuance_boundary.get(
                "production_instance_or_model_built") is not False
            or issuance_boundary.get(
                "runner_or_model_plan_written") is not False
            or issuance_boundary.get(
                "scientific_production_solver_invoked") is not False
            or result.get("candidate_outcomes_consumed") is not False
            or result.get("scientific_masks_accessed") is not masks
            or not _is_digest(report.get(authority.REPORT_SELF_KEY))
            or not _is_digest(amendment.get(authority.AMENDMENT_SELF_KEY))):
        raise GlobalExactRunnerError("execution authority boundary changed")
    if masks and "preserved_vectors" not in result:
        raise GlobalExactRunnerError("mask-bearing context lacks frozen vectors")
    if not masks and ({"inputs", "preserved_vectors"} & set(result)):
        raise GlobalExactRunnerError(
            "pre-mask context opened candidate rows or frozen vectors")
    return result


def issue_report(*, builder: Any = BUILDER,
                 authority: Any = AUTHORITY) -> dict[str, Any]:
    report = builder.issue_global_exact_coupling_report()
    if (not isinstance(report, Mapping)
            or report.get("classification") != "COUPLED"
            or report.get("selected_method")
            != "ONE_GLOBAL_EXACT_FEASIBILITY_MODEL"):
        raise GlobalExactRunnerError("issued coupling report is not exact")
    summary = {
        "status": report.get("status"),
        "coupling_classification": report["classification"],
        "selected_method": report["selected_method"],
        "coupling_report_digest": report[authority.REPORT_SELF_KEY],
        "solver_invoked": False,
        "scientific_masks_accessed": False,
        "candidate_outcomes_consumed": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return dict(report)


def issue_amendment(*, builder: Any = BUILDER,
                    authority: Any = AUTHORITY) -> dict[str, Any]:
    amendment = builder.issue_global_exact_execution_amendment()
    method = amendment.get("selected_execution_method", {})
    if (not isinstance(amendment, Mapping)
            or method.get("method") != "ONE_GLOBAL_EXACT_FEASIBILITY_MODEL"
            or amendment.get("issuance_boundary", {}).get("solver_invoked")
            is not False):
        raise GlobalExactRunnerError("issued execution amendment is not exact")
    summary = {
        "status": amendment.get("status"),
        "execution_amendment_digest": amendment[
            authority.AMENDMENT_SELF_KEY],
        "selected_method": method["method"],
        "superseded_external_enumeration": amendment["supersession"][
            "status"],
        "v1_disposition": amendment["v1_disposition"],
        "v2_backend_disposition": amendment["v2_backend_disposition"],
        "solver_invoked": False,
        "scientific_masks_accessed": False,
        "candidate_outcomes_consumed": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return dict(amendment)


def issue_source_correction(
        *, root: Path = ROOT, builder: Any = BUILDER,
        authority: Any = AUTHORITY) -> dict[str, Any]:
    """Issue only the immutable source-corrected successor to V1 authority."""

    historical = (
        builder.load_global_exact_historical_mixed_disposition_authority())
    amendment = authority.issue_execution_amendment_v2(
        root / authority.EXECUTION_AMENDMENT_V2_RELATIVE_PATH,
        historical_mixed_disposition_authority=historical,
        root=root)
    failure = amendment.get("failed_attempt_disposition", {})
    source_correction = amendment.get("source_correction", {})
    v1 = amendment.get("v1_execution_authority", {})
    if (not isinstance(amendment, Mapping)
            or amendment.get("schema") != authority.AMENDMENT_V2_SCHEMA
            or amendment.get("status") != authority.AMENDMENT_V2_STATUS
            or failure != authority.FAILED_SOURCE_TRANSITION_DISPOSITION
            or source_correction.get("scientific_contract_changed") is not False
            or source_correction.get(
                "candidate_outcome_or_downstream_metric_used") is not False
            or v1.get("coupling_report_artifact_binding")
            != authority.ORIGINAL_COUPLING_REPORT_ARTIFACT_BINDING
            or v1.get("execution_amendment_artifact_binding")
            != authority.ORIGINAL_EXECUTION_AMENDMENT_ARTIFACT_BINDING):
        raise GlobalExactRunnerError(
            "issued source-corrected execution authority is not exact")
    summary = {
        "status": amendment["status"],
        "execution_amendment_digest": amendment[
            authority.AMENDMENT_SELF_KEY],
        "historical_source_repository_commit": source_correction[
            "historical_source_repository_commit"],
        "source_repository_commit": amendment["source_repository_commit"],
        "immutable_v1_coupling_report_digest":
            authority.ORIGINAL_COUPLING_REPORT_ARTIFACT_BINDING[
                "self_digest"],
        "immutable_v1_execution_amendment_digest":
            authority.ORIGINAL_EXECUTION_AMENDMENT_ARTIFACT_BINDING[
                "self_digest"],
        "failed_attempt_disposition": failure["disposition"],
        "historical_preoutcome_disposition_read": True,
        "new_candidate_rotation_vectors_read": False,
        "candidate_outcomes_consumed": False,
        "production_plan_or_solver_started": False,
        "scientific_contract_changed": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return dict(amendment)


def issue_preplan_integration_correction(
        *, builder: Any = BUILDER, authority: Any = AUTHORITY,
        ) -> dict[str, Any]:
    """Issue only the active wrapper around the immutable V2 authority."""

    correction = builder.issue_global_exact_preplan_integration_correction()
    if not isinstance(correction, Mapping):
        raise GlobalExactRunnerError(
            "issued preplan integration correction is not exact")
    immutable_v2 = correction.get("immutable_v2_execution_authority", {})
    post_install = correction.get("v2_post_install_reopen_failure", {})
    failed = correction.get(
        "post_v2_preplan_failed_attempt_disposition", {})
    source_correction = correction.get("preplan_integration_correction", {})
    boundary = correction.get("issuance_boundary", {})
    if (correction.get("schema")
            != authority.PREPLAN_INTEGRATION_CORRECTION_SCHEMA
            or correction.get("status")
            != authority.PREPLAN_INTEGRATION_CORRECTION_STATUS
            or correction.get("amendment_version") != 2
            or correction.get("preplan_integration_correction_version") != 1
            or immutable_v2.get("binding")
            != authority.IMMUTABLE_V2_EXECUTION_AMENDMENT_ARTIFACT_BINDING
            or post_install != authority.V2_POST_INSTALL_REOPEN_FAILURE
            or failed != authority.POST_V2_PREPLAN_FAILED_ATTEMPT_DISPOSITION
            or source_correction.get("scientific_contract_changed") is not False
            or source_correction.get(
                "candidate_outcome_or_downstream_metric_used") is not False
            or boundary.get("historical_scientific_masks_accessed") is not True
            or boundary.get(
                "scientific_masks_accessed_during_this_issuance") is not False
            or boundary.get("new_attempt_mask_context_started") is not False
            or boundary.get("candidate_outcomes_consumed") is not False):
        raise GlobalExactRunnerError(
            "issued preplan integration correction is not exact")
    summary = {
        "status": correction["status"],
        "execution_amendment_digest": correction[
            authority.AMENDMENT_SELF_KEY],
        "immutable_v2_execution_amendment_digest": immutable_v2[
            "binding"]["self_digest"],
        "source_repository_commit": correction["source_repository_commit"],
        "post_install_v2_artifact_remains_valid": post_install[
            "v2_artifact_remains_valid"],
        "post_v2_preplan_failure": failed["disposition"],
        "historical_scientific_masks_accessed": True,
        "scientific_masks_accessed_during_this_issuance": False,
        "new_attempt_mask_context_started": False,
        "candidate_outcomes_consumed": False,
        "production_plan_or_solver_started": False,
        "scientific_contract_changed": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return dict(correction)


def _validate_fixture_suite_solve_free(
        fixture_suite: Mapping[str, Any], *, authority: Any, model: Any
        ) -> dict[str, Any]:
    """Validate the persisted synthetic proof without executing its solvers."""

    if not isinstance(fixture_suite, Mapping):
        raise GlobalExactRunnerError("fixture suite is not a mapping")
    result = dict(fixture_suite)
    expected_keys = {
        "schema", "status", "fixture_validation_contract",
        "objective_contract_digest", "solver_contract_digest",
        "solver_runtime_identity", "fixtures",
        "candidate_outcomes_consumed", model.FIXTURE_SUITE_DIGEST_KEY,
    }
    runtime = result.get("solver_runtime_identity")
    fixtures = result.get("fixtures")
    fixture_keys = {
        "fixture_id", "semantic_spec_digest", "model_digest",
        "solver_feasible", "control_feasible",
        "control_valid_assignment_count",
        "deterministic_optimal_objective_value",
        "repeated_runs_identical_bytes", "exact_result_digest",
        "all_returned_constraints_directly_validated",
        "boundary_predicates", "candidate_outcomes_consumed",
    }
    required_ids = authority.FIXTURE_VALIDATION_CONTRACT[
        "required_fixture_ids"]
    try:
        validated_runtime = model.validate_solver_runtime_identity_record(
            runtime)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise GlobalExactRunnerError(
            "persisted fixture solver runtime identity changed") from exc
    if (set(result) != expected_keys
            or result.get("schema") != model.FIXTURE_SUITE_SCHEMA
            or result.get("status")
            != "PASS_MANDATORY_SYNTHETIC_FIXTURE_SUITE"
            or result.get("fixture_validation_contract")
            != authority.FIXTURE_VALIDATION_CONTRACT
            or result.get("objective_contract_digest")
            != model.OBJECTIVE_CONTRACT_DIGEST
            or result.get("solver_contract_digest")
            != model.SOLVER_CONTRACT_DIGEST
            or result.get("candidate_outcomes_consumed") is not False
            or result.get(model.FIXTURE_SUITE_DIGEST_KEY)
            != model.canonical_digest(_without(
                result, model.FIXTURE_SUITE_DIGEST_KEY))
            or result.get(model.FIXTURE_SUITE_DIGEST_KEY)
            != model.FROZEN_FIXTURE_SUITE_RESULT_DIGEST
            or validated_runtime != runtime
            or not isinstance(fixtures, list)
            or [row.get("fixture_id") if isinstance(row, Mapping) else None
                for row in fixtures] != required_ids):
        raise GlobalExactRunnerError("persisted fixture suite changed")
    for row in fixtures:
        if (not isinstance(row, Mapping) or set(row) != fixture_keys
                or not _is_digest(row.get("semantic_spec_digest"))
                or not _is_digest(row.get("model_digest"))
                or not _is_digest(row.get("exact_result_digest"))
                or type(row.get("solver_feasible")) is not bool
                or row.get("solver_feasible") is not row.get("control_feasible")
                or type(row.get("control_valid_assignment_count")) is not int
                or row["control_valid_assignment_count"] < 0
                or row.get("repeated_runs_identical_bytes") is not True
                or row.get("all_returned_constraints_directly_validated")
                is not True
                or row.get("candidate_outcomes_consumed") is not False):
            raise GlobalExactRunnerError("persisted fixture row changed")
    boundary_id = authority.FIXTURE_VALIDATION_CONTRACT[
        "mandatory_boundary_fixture"]["fixture_id"]
    boundary = next(row for row in fixtures
                    if row["fixture_id"] == boundary_id)
    expected_boundary = dict(authority.FIXTURE_VALIDATION_CONTRACT[
        "mandatory_boundary_fixture"])
    expected_boundary.pop("fixture_id")
    if (boundary.get("boundary_predicates") != expected_boundary
            or any(row.get("boundary_predicates") is not None
                   for row in fixtures if row is not boundary)):
        raise GlobalExactRunnerError("persisted boundary fixture changed")
    return result


def _build_runner_plan(*, pre_mask_context: Mapping[str, Any],
                       mask_context: Mapping[str, Any],
                       fixture_suite: Mapping[str, Any],
                       instance: Mapping[str, Any],
                       model_plan: Mapping[str, Any],
                       authority: Any, model: Any,
                       solve_free: bool = False) -> dict[str, Any]:
    report = pre_mask_context["coupling_report"]
    amendment = pre_mask_context["execution_amendment"]
    if (mask_context["coupling_report"] != report
            or mask_context["execution_amendment"] != amendment
            or mask_context["scientific_contract_bindings"]
            != pre_mask_context["scientific_contract_bindings"]
            or mask_context["preoutcome_input_bindings"]
            != pre_mask_context["preoutcome_input_bindings"]):
        raise GlobalExactRunnerError("authority changed while opening masks")
    validated_instance = model.validate_production_instance(instance)
    plan_validator = (model.validate_execution_plan_solve_free
                      if solve_free else model.validate_execution_plan)
    validated_model_plan = plan_validator(validated_instance, model_plan)
    validated_fixtures = (
        _validate_fixture_suite_solve_free(
            fixture_suite, authority=authority, model=model)
        if solve_free else model.validate_fixture_suite_result(fixture_suite))
    continuation = amendment.get("continuation_authority")
    if (not isinstance(continuation, Mapping)
            or continuation.get("downstream_runtime_contracts")
            != authority.DOWNSTREAM_RUNTIME_CONTRACTS
            or continuation.get("downstream_stage_runtime_roles")
            != authority.DOWNSTREAM_STAGE_RUNTIME_ROLES
            or continuation.get("downstream_uses_global_solver_interpreter")
            != {"genesis": False, "rocm": False}):
        raise GlobalExactRunnerError(
            "post-manifest interpreter authority changed")
    return _signed({
        "schema": PLAN_SCHEMA,
        "status": PLAN_STATUS,
        "source_repository_commit": amendment["source_repository_commit"],
        "coupling_report_digest": report[authority.REPORT_SELF_KEY],
        "execution_amendment_digest": amendment[authority.AMENDMENT_SELF_KEY],
        "scientific_contract_bindings_digest": canonical_digest(
            pre_mask_context["scientific_contract_bindings"]),
        "preoutcome_input_bindings_digest": canonical_digest(
            pre_mask_context["preoutcome_input_bindings"]),
        "immutable_predecessor_lineage_digest": amendment[
            "immutable_predecessor_lineage_digest"],
        "fixture_suite_result": validated_fixtures,
        "fixture_suite_digest": validated_fixtures[
            model.FIXTURE_SUITE_DIGEST_KEY],
        "production_instance_digest": model.canonical_digest(
            validated_instance),
        "model_execution_plan": validated_model_plan,
        "model_execution_plan_digest": validated_model_plan[
            model.EXECUTION_PLAN_DIGEST_KEY],
        "selected_execution_method": "ONE_GLOBAL_EXACT_FEASIBILITY_MODEL",
        "external_combination_enumeration": False,
        "performance_benchmark_run": False,
        "v1_or_v2_retried": False,
        "candidate_outcomes_consumed": False,
        "scientific_masks_accessed": True,
        "scientific_masks_accessed_after_amendment": True,
        "scientific_masks_are_frozen_model_inputs": True,
        "downstream_runtime_contracts": json.loads(_json_bytes(
            authority.DOWNSTREAM_RUNTIME_CONTRACTS)),
        "downstream_stage_runtime_roles": dict(
            authority.DOWNSTREAM_STAGE_RUNTIME_ROLES),
        "downstream_uses_global_solver_interpreter": {
            "genesis": False, "rocm": False},
        "downstream_metric_in_selection": False,
    }, PLAN_SELF_KEY)


def _validate_runner_plan(plan: Mapping[str, Any], *,
                          pre_mask_context: Mapping[str, Any],
                          mask_context: Mapping[str, Any],
                          fixture_suite: Mapping[str, Any],
                          instance: Mapping[str, Any], model_plan: Mapping[str, Any],
                          authority: Any, model: Any,
                          solve_free: bool = False) -> dict[str, Any]:
    expected = _build_runner_plan(
        pre_mask_context=pre_mask_context, mask_context=mask_context,
        fixture_suite=fixture_suite, instance=instance, model_plan=model_plan,
        authority=authority, model=model, solve_free=solve_free)
    if (not isinstance(plan, Mapping) or set(plan) != _RUNNER_PLAN_KEYS
            or dict(plan) != expected):
        raise GlobalExactRunnerError("global exact runner plan changed")
    return expected


def validate_runner_plan(
        plan: Mapping[str, Any], *, execution_context: Mapping[str, Any],
        instance: Mapping[str, Any], authority: Any = AUTHORITY,
        model: Any = MODEL) -> dict[str, Any]:
    """Public closed-schema validator used by the lazy-importing finalizer.

    The persisted plan already binds its pre-mask authority inputs.  A live
    post-amendment context must reproduce those same bindings and the exact
    nested fixture/model plans; no generated outcome is needed here.
    """

    context = _require_context(
        execution_context, masks=True, authority=authority)
    if not isinstance(plan, Mapping):
        raise GlobalExactRunnerError("global exact runner plan is not a mapping")
    fixture = _validate_fixture_suite_solve_free(
        plan.get("fixture_suite_result", {}), authority=authority, model=model)
    model_plan = model.validate_execution_plan_solve_free(
        instance, plan.get("model_execution_plan", {}))
    return _validate_runner_plan(
        plan, pre_mask_context=context, mask_context=context,
        fixture_suite=fixture, instance=instance, model_plan=model_plan,
        authority=authority, model=model, solve_free=True)


def _elapsed(start_wall: float, start_cpu: float) -> tuple[float, float]:
    wall = max(0.0, time.monotonic() - start_wall)
    cpu = max(0.0, time.process_time() - start_cpu)
    if not math.isfinite(wall) or not math.isfinite(cpu):
        raise GlobalExactRunnerError("solve elapsed time is not finite")
    return round(wall, 9), round(cpu, 9)


def _build_feasible_terminal(*, plan: Mapping[str, Any],
                             model_result: Mapping[str, Any],
                             materialized: Mapping[str, Any],
                             elapsed_wall_s: float, elapsed_cpu_s: float,
                             model: Any) -> dict[str, Any]:
    selected = materialized.get("selected_scene_ids")
    if (not isinstance(selected, list) or len(selected) != 5
            or len(set(selected)) != 5):
        raise GlobalExactRunnerError("materialized completion scene set changed")
    return _signed({
        "schema": TERMINAL_SCHEMA,
        "status": TERMINAL_STATUS,
        "global_exact_model_plan_digest": plan[PLAN_SELF_KEY],
        "execution_amendment_digest": plan["execution_amendment_digest"],
        "model_execution_plan_digest": plan["model_execution_plan_digest"],
        "model_execution_result": dict(model_result),
        "model_execution_result_digest": model_result[
            model.EXECUTION_RESULT_DIGEST_KEY],
        "materialized_allocation_digest": materialized[
            model.ALLOCATION_RESULT_DIGEST_KEY],
        "selected_scene_ids": list(selected),
        "selected_scene_rows": list(materialized["selected_scene_rows"]),
        "elapsed_wall_s": elapsed_wall_s,
        "elapsed_cpu_s": elapsed_cpu_s,
        "exact_infeasibility_proved": False,
        "candidate_outcomes_consumed": False,
        "scientific_masks_accessed": True,
        "branch_labels_read": False,
        "scorer_or_predictor_accessed": False,
    }, TERMINAL_SELF_KEY)


def _build_infeasible_terminal(*, plan: Mapping[str, Any],
                               model_result: Mapping[str, Any],
                               elapsed_wall_s: float, elapsed_cpu_s: float,
                               model: Any) -> dict[str, Any]:
    return _signed({
        "schema": INFEASIBILITY_SCHEMA,
        "status": INFEASIBILITY_STATUS,
        "global_exact_model_plan_digest": plan[PLAN_SELF_KEY],
        "execution_amendment_digest": plan["execution_amendment_digest"],
        "model_execution_plan_digest": plan["model_execution_plan_digest"],
        "model_execution_result": dict(model_result),
        "model_execution_result_digest": model_result[
            model.EXECUTION_RESULT_DIGEST_KEY],
        "elapsed_wall_s": elapsed_wall_s,
        "elapsed_cpu_s": elapsed_cpu_s,
        "exact_infeasibility_proved": True,
        "scientific_conditions_relaxed": False,
        "automatic_selector_revision": False,
        "candidate_outcomes_consumed": False,
        "scientific_masks_accessed": True,
        "branch_labels_read": False,
        "scorer_or_predictor_accessed": False,
    }, INFEASIBILITY_SELF_KEY)


def _validate_terminal_common(payload: Mapping[str, Any], *,
                              self_key: str, expected_keys: frozenset[str],
                              plan: Mapping[str, Any],
                              instance: Mapping[str, Any], model: Any,
                              solve_free: bool = False,
                              ) -> dict[str, Any]:
    if (not isinstance(payload, Mapping) or set(payload) != expected_keys
            or payload.get(self_key)
            != canonical_digest(_without(payload, self_key))
            or payload.get("global_exact_model_plan_digest")
            != plan[PLAN_SELF_KEY]
            or payload.get("execution_amendment_digest")
            != plan["execution_amendment_digest"]
            or payload.get("model_execution_plan_digest")
            != plan["model_execution_plan_digest"]
            or type(payload.get("elapsed_wall_s")) not in {int, float}
            or type(payload.get("elapsed_cpu_s")) not in {int, float}
            or not math.isfinite(float(payload["elapsed_wall_s"]))
            or not math.isfinite(float(payload["elapsed_cpu_s"]))
            or float(payload["elapsed_wall_s"]) < 0.0
            or float(payload["elapsed_cpu_s"]) < 0.0):
        raise GlobalExactRunnerError("global exact terminal binding changed")
    result_validator = (model.validate_execution_result_solve_free
                        if solve_free else model.validate_execution_result)
    result = result_validator(
        instance, plan["model_execution_plan"],
        payload.get("model_execution_result", {}))
    if (payload.get("model_execution_result_digest")
            != result[model.EXECUTION_RESULT_DIGEST_KEY]):
        raise GlobalExactRunnerError("terminal model result digest changed")
    return dict(payload)


def _validate_feasible_terminal(payload: Mapping[str, Any], *,
                                plan: Mapping[str, Any],
                                instance: Mapping[str, Any], model: Any,
                                solve_free: bool = False,
                                ) -> dict[str, Any]:
    terminal = _validate_terminal_common(
        payload, self_key=TERMINAL_SELF_KEY,
        expected_keys=_FEASIBLE_TERMINAL_KEYS, plan=plan,
        instance=instance, model=model, solve_free=solve_free)
    result = terminal["model_execution_result"]
    if (terminal.get("schema") != TERMINAL_SCHEMA
            or terminal.get("status") != TERMINAL_STATUS
            or result.get("status") != model.EXECUTION_PASS_STATUS
            or terminal.get("exact_infeasibility_proved") is not False
            or terminal.get("candidate_outcomes_consumed") is not False
            or terminal.get("scientific_masks_accessed") is not True
            or terminal.get("branch_labels_read") is not False
            or terminal.get("scorer_or_predictor_accessed") is not False):
        raise GlobalExactRunnerError("feasible terminal status changed")
    materialized = result.get("materialized_allocation")
    if not isinstance(materialized, Mapping):
        raise GlobalExactRunnerError(
            "feasible model terminal lacks its materialized allocation")
    if (terminal.get("materialized_allocation_digest")
            != materialized[model.ALLOCATION_RESULT_DIGEST_KEY]
            or terminal.get("selected_scene_ids")
            != materialized["selected_scene_ids"]
            or terminal.get("selected_scene_rows")
            != materialized["selected_scene_rows"]):
        raise GlobalExactRunnerError("feasible terminal allocation changed")
    return terminal


def _validate_infeasible_terminal(payload: Mapping[str, Any], *,
                                  plan: Mapping[str, Any],
                                  instance: Mapping[str, Any], model: Any,
                                  solve_free: bool = False,
                                  ) -> dict[str, Any]:
    terminal = _validate_terminal_common(
        payload, self_key=INFEASIBILITY_SELF_KEY,
        expected_keys=_INFEASIBLE_TERMINAL_KEYS, plan=plan,
        instance=instance, model=model, solve_free=solve_free)
    result = terminal["model_execution_result"]
    if (terminal.get("schema") != INFEASIBILITY_SCHEMA
            or terminal.get("status") != INFEASIBILITY_STATUS
            or result.get("status") != model.EXECUTION_INFEASIBLE_STATUS
            or result.get("materialized_allocation") is not None
            or terminal.get("exact_infeasibility_proved") is not True
            or terminal.get("scientific_conditions_relaxed") is not False
            or terminal.get("automatic_selector_revision") is not False
            or terminal.get("candidate_outcomes_consumed") is not False
            or terminal.get("scientific_masks_accessed") is not True
            or terminal.get("branch_labels_read") is not False
            or terminal.get("scorer_or_predictor_accessed") is not False):
        raise GlobalExactRunnerError("infeasible terminal status changed")
    return terminal


def validate_runner_terminal(
        terminal: Mapping[str, Any], *,
        execution_context: Mapping[str, Any], instance: Mapping[str, Any],
        runner_plan: Mapping[str, Any], authority: Any = AUTHORITY,
        model: Any = MODEL) -> dict[str, Any]:
    """Validate either mutually exclusive terminal under the frozen plan."""

    context = _require_context(
        execution_context, masks=True, authority=authority)
    plan = validate_runner_plan(
        runner_plan, execution_context=context, instance=instance,
        authority=authority, model=model)
    if not isinstance(terminal, Mapping) or terminal.get("schema") != TERMINAL_SCHEMA:
        raise GlobalExactRunnerError("global exact runner terminal schema changed")
    if terminal.get("status") == TERMINAL_STATUS:
        return _validate_feasible_terminal(
            terminal, plan=plan, instance=instance, model=model,
            solve_free=True)
    if terminal.get("status") == INFEASIBILITY_STATUS:
        return _validate_infeasible_terminal(
            terminal, plan=plan, instance=instance, model=model,
            solve_free=True)
    raise GlobalExactRunnerError("global exact runner terminal status changed")


def _default_command_runner(command: Sequence[str], root: Path) -> int:
    completed = subprocess.run(
        [str(part) for part in command], cwd=root, check=False)
    return int(completed.returncode)


def _runtime_contract(authority: Any, role: str) -> dict[str, Any]:
    contracts = getattr(authority, "DOWNSTREAM_RUNTIME_CONTRACTS", None)
    if (not isinstance(contracts, Mapping)
            or set(contracts) != set(_DOWNSTREAM_RUNTIME_ROLES)
            or role not in contracts
            or not isinstance(contracts[role], Mapping)):
        raise GlobalExactRunnerError(
            "downstream runtime contract surface changed")
    return json.loads(_json_bytes(contracts[role]))


def _runtime_observation_from_contract(contract: Mapping[str, Any]
                                       ) -> dict[str, Any]:
    keys = {
        "python_version", "torch_version", "torch_cuda_runtime",
        "torch_hip_runtime", "accelerator_available",
        "accelerator_device_count", "accelerator_devices",
    }
    if contract.get("role") == "genesis_branch_generation":
        keys.add("genesis_version")
    if not keys.issubset(contract):
        raise GlobalExactRunnerError(
            "downstream runtime observation contract is incomplete")
    return {key: json.loads(_json_bytes(contract[key]))
            for key in sorted(keys)}


def _bound_downstream_interpreters(
        *, root: Path, execution_context: Mapping[str, Any], authority: Any,
        ) -> dict[str, Path]:
    """Pin both authorised downstream runtimes without importing either."""

    context = _require_context(
        execution_context, masks=True, authority=authority)
    continuation = context["execution_amendment"].get(
        "continuation_authority")
    if (not isinstance(continuation, Mapping)
            or continuation.get("downstream_runtime_contracts")
            != authority.DOWNSTREAM_RUNTIME_CONTRACTS
            or continuation.get("downstream_stage_runtime_roles")
            != authority.DOWNSTREAM_STAGE_RUNTIME_ROLES
            or continuation.get("downstream_uses_global_solver_interpreter")
            != {"genesis": False, "rocm": False}):
        raise GlobalExactRunnerError(
            "execution amendment downstream runtime binding changed")
    result: dict[str, Path] = {}
    for role in _DOWNSTREAM_RUNTIME_ROLES:
        contract = _runtime_contract(authority, role)
        interpreter = _pinned_relative(
            root, Path(contract["interpreter_relative_path"]),
            label=f"bound {role} downstream interpreter")
        config = _pinned_relative(
            root, Path(contract["pyvenv_config_relative_path"]),
            label=f"bound {role} downstream pyvenv config")
        byte_count = contract.get("pyvenv_config_byte_count")
        if (not interpreter.exists() or interpreter.is_dir()
                or not os.access(interpreter, os.X_OK)
                or not config.is_file() or config.is_symlink()
                or type(byte_count) is not int or byte_count <= 0
                or config.stat().st_size != byte_count
                or _sha256_file(config) != contract.get(
                    "pyvenv_config_sha256")):
            raise GlobalExactRunnerError(
                f"bound {role} downstream runtime custody changed")
        if interpreter.absolute() == Path(sys.executable).absolute():
            raise GlobalExactRunnerError(
                f"{role} downstream interpreter equals solver interpreter")
        result[role] = interpreter
    return result


def downstream_command_sequence(
        root: Path = ROOT, *,
        downstream_interpreters: Mapping[str, Path] | None = None,
        authority: Any = AUTHORITY) -> list[list[str]]:
    """Return the exact dual-runtime post-manifest command order."""

    if downstream_interpreters is None:
        interpreters = {
            role: root / _runtime_contract(authority, role)[
                "interpreter_relative_path"]
            for role in _DOWNSTREAM_RUNTIME_ROLES
        }
    else:
        if set(downstream_interpreters) != set(_DOWNSTREAM_RUNTIME_ROLES):
            raise GlobalExactRunnerError(
                "downstream interpreter role set changed")
        interpreters = {
            role: Path(downstream_interpreters[role])
            for role in _DOWNSTREAM_RUNTIME_ROLES
        }
    roles = authority.DOWNSTREAM_STAGE_RUNTIME_ROLES
    if (not isinstance(roles, Mapping)
            or set(roles) != set(_DOWNSTREAM_COMMAND_STAGE_NAMES) | {
                "qualification_validation", "development_validation"}
            or [roles[name] for name in _DOWNSTREAM_COMMAND_STAGE_NAMES]
            != ["genesis", "rocm", "genesis", "rocm", "rocm", "rocm"]):
        raise GlobalExactRunnerError(
            "downstream stage runtime routing changed")
    commands = [
        [str(interpreters[roles["six_branch_smoke"]]),
         str(root / "scripts/build_go2_branch_corpus_v1_2.py"),
         "--pool", "scorer_fit", "--stage", "smoke"],
        [str(interpreters[roles["smoke_encoding"]]),
         str(root / "scripts/encode_go2_branch_corpus_v1_2.py"),
         "--pool", "scorer_fit", "--smoke"],
        [str(interpreters[roles["full_720_branch_corpus"]]),
         str(root / "scripts/build_go2_branch_corpus_v1_2.py"),
         "--pool", "scorer_fit", "--stage", "branches"],
        [str(interpreters[roles["full_latent_encoding"]]),
         str(root / "scripts/encode_go2_branch_corpus_v1_2.py"),
         "--pool", "scorer_fit"],
        [str(interpreters[roles["scorer_training_and_qualification"]]),
         str(root / "scripts/train_go2_utility_scorer_v1_2.py"),
         "--pool", "scorer_fit"],
        [str(interpreters[roles["development_transfer"]]),
         str(root /
             "scripts/apply_go2_utility_scorer_to_counterfactual_"
             "development_v1_2.py")],
    ]
    return commands


def _build_downstream_runtime_probe_receipt(
        *, runtime_role: str, observation: Mapping[str, Any], authority: Any,
        ) -> dict[str, Any]:
    contract = _runtime_contract(authority, runtime_role)
    expected_observation = _runtime_observation_from_contract(contract)
    if dict(observation) != expected_observation:
        raise GlobalExactRunnerError(
            f"{runtime_role} downstream runtime identity changed")
    return _signed({
        "schema": DOWNSTREAM_RUNTIME_PROBE_SCHEMA,
        "status": DOWNSTREAM_RUNTIME_PROBE_STATUS,
        "runtime_role": runtime_role,
        "runtime_contract_digest": canonical_digest(contract),
        "interpreter_relative_path": contract[
            "interpreter_relative_path"],
        "pyvenv_config_sha256": contract["pyvenv_config_sha256"],
        "observed_runtime_identity": dict(observation),
    }, DOWNSTREAM_RUNTIME_PROBE_SELF_KEY)


def _validate_downstream_runtime_probe_receipt(
        receipt: Mapping[str, Any], *, runtime_role: str, authority: Any,
        ) -> dict[str, Any]:
    if not isinstance(receipt, Mapping):
        raise GlobalExactRunnerError(
            "downstream runtime probe receipt is not a mapping")
    payload = dict(receipt)
    expected_keys = {
        "schema", "status", "runtime_role", "runtime_contract_digest",
        "interpreter_relative_path", "pyvenv_config_sha256",
        "observed_runtime_identity", DOWNSTREAM_RUNTIME_PROBE_SELF_KEY,
    }
    contract = _runtime_contract(authority, runtime_role)
    if (set(payload) != expected_keys
            or payload.get("schema") != DOWNSTREAM_RUNTIME_PROBE_SCHEMA
            or payload.get("status") != DOWNSTREAM_RUNTIME_PROBE_STATUS
            or payload.get("runtime_role") != runtime_role
            or payload.get("runtime_contract_digest")
            != canonical_digest(contract)
            or payload.get("interpreter_relative_path")
            != contract["interpreter_relative_path"]
            or payload.get("pyvenv_config_sha256")
            != contract["pyvenv_config_sha256"]
            or payload.get(DOWNSTREAM_RUNTIME_PROBE_SELF_KEY)
            != canonical_digest(_without(
                payload, DOWNSTREAM_RUNTIME_PROBE_SELF_KEY))):
        raise GlobalExactRunnerError(
            "downstream runtime probe receipt binding changed")
    expected = _build_downstream_runtime_probe_receipt(
        runtime_role=runtime_role,
        observation=payload.get("observed_runtime_identity", {}),
        authority=authority)
    if payload != expected:
        raise GlobalExactRunnerError(
            "downstream runtime probe receipt is not exact")
    return payload


def _observe_current_downstream_runtime(runtime_role: str) -> dict[str, Any]:
    """Observe one child runtime; this is called only by an internal stage."""

    import torch

    devices: list[dict[str, Any]] = []
    device_count = int(torch.cuda.device_count())
    for index in range(device_count):
        properties = torch.cuda.get_device_properties(index)
        gcn_arch_name = getattr(properties, "gcnArchName", None)
        if gcn_arch_name is None:
            gcn_arch_name = getattr(properties, "gcn_arch_name", None)
        devices.append({
            "index": index,
            "name": str(torch.cuda.get_device_name(index)),
            "capability": list(torch.cuda.get_device_capability(index)),
            "gcn_arch_name": gcn_arch_name,
            "multi_processor_count": int(
                properties.multi_processor_count),
        })
    observation: dict[str, Any] = {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_cuda_runtime": torch.version.cuda,
        "torch_hip_runtime": torch.version.hip,
        "accelerator_available": bool(torch.cuda.is_available()),
        "accelerator_device_count": device_count,
        "accelerator_devices": devices,
    }
    if runtime_role == "genesis":
        # Query distribution metadata without importing Genesis: its normal
        # import path may emit a banner on stdout and would corrupt the
        # closed-JSON probe channel.
        from importlib import metadata

        observation["genesis_version"] = str(
            metadata.version("genesis-world"))
    elif runtime_role != "rocm":
        raise GlobalExactRunnerError("downstream runtime role changed")
    return observation


def _emit_downstream_runtime_probe(
        runtime_role: str, *, root: Path = ROOT,
        authority: Any = AUTHORITY) -> int:
    contract = _runtime_contract(authority, runtime_role)
    expected_interpreter = _pinned_relative(
        root, Path(contract["interpreter_relative_path"]),
        label=f"bound {runtime_role} downstream interpreter")
    config = _pinned_relative(
        root, Path(contract["pyvenv_config_relative_path"]),
        label=f"bound {runtime_role} downstream pyvenv config")
    if (Path(sys.executable).absolute() != expected_interpreter.absolute()
            or not config.is_file() or config.is_symlink()
            or config.stat().st_size
            != contract["pyvenv_config_byte_count"]
            or _sha256_file(config) != contract["pyvenv_config_sha256"]):
        raise GlobalExactRunnerError(
            f"runtime probe is not using the bound {runtime_role} runtime")
    receipt = _build_downstream_runtime_probe_receipt(
        runtime_role=runtime_role,
        observation=_observe_current_downstream_runtime(runtime_role),
        authority=authority)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


def _default_downstream_runtime_probe_invoker(
        runtime_role: str, root: Path, downstream_python: Path,
        authority: Any) -> Mapping[str, Any]:
    stage = f"internal-probe-{runtime_role}-runtime"
    completed = subprocess.run(
        [str(downstream_python), str(root / RUNNER_RELATIVE_PATH),
         "--stage", stage],
        cwd=root, check=False, capture_output=True, text=True, timeout=60)
    if completed.returncode != 0:
        raise GlobalExactRunnerError(
            f"bound {runtime_role} downstream runtime probe failed")
    try:
        payload = json.loads(completed.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise GlobalExactRunnerError(
            f"bound {runtime_role} runtime probe output changed") from exc
    return _validate_downstream_runtime_probe_receipt(
        payload, runtime_role=runtime_role, authority=authority)


def _validate_downstream_manifest_boundary(root: Path) -> dict[str, Any]:
    """Reopen the exact manifest/successor before any scorer metric read."""

    manifest_path = _pinned_relative(
        root,
        Path(".generated/go2_branch_corpus_v1_2/scorer_fit/"
             "state_manifest.json"),
        label="global exact scorer-fit state manifest")
    allocation_path = _pinned_relative(
        root,
        Path(".generated/go2_branch_corpus_v1_2/scorer_fit/"
             "candidate_allocation_manifest.json"),
        label="global exact candidate allocation manifest")
    manifest, _manifest_raw = _load_json(
        manifest_path, label="global exact scorer-fit state manifest")
    allocation, _allocation_raw = _load_json(
        allocation_path, label="global exact candidate allocation manifest")
    try:
        BUILDER.validate_global_exact_allocation_for_consumption(
            manifest, allocation)
        successor = (
            BUILDER.load_global_exact_successor_scorer_contract_for_consumption(
                manifest))
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise GlobalExactRunnerError(
            "downstream global-exact manifest boundary changed") from exc
    expected_keys = {
        "global_exact_execution_amendment_digest",
        "global_exact_successor_scorer_contract_digest",
    }
    if (not isinstance(successor, Mapping)
            or any(not _is_digest(successor.get(key))
                   for key in expected_keys)):
        raise GlobalExactRunnerError(
            "downstream successor scorer binding is malformed")
    historical_scorer_digest = manifest.get("scorer_contract_v1_2_digest")
    if not _is_digest(historical_scorer_digest):
        raise GlobalExactRunnerError(
            "downstream scientific predecessor scorer binding is malformed")
    return {
        **{key: str(successor[key]) for key in sorted(expected_keys)},
        "scientific_predecessor_scorer_contract_v1_2_digest":
            str(historical_scorer_digest),
    }


def _load_qualification(
        root: Path, *, expected_execution_amendment_digest: str,
        expected_successor_contract_digest: str,
        expected_scientific_predecessor_scorer_contract_digest: str,
        ) -> dict[str, Any]:
    # Lazy import: report/amendment/model-plan stages must not initialise the
    # training stack or open any scorer artifact.
    from scripts import train_go2_utility_scorer_v1_2 as trainer

    path = _pinned_relative(
        root, Path(".generated/go2_utility_scorer_v1_2/qualification.json"),
        label="qualification report")
    report, _raw = _load_json(path, label="qualification report")
    digest = report.get("qualification_report_digest")
    criteria = report.get("criteria")
    provenance_keys = frozenset(
        trainer.SCORER_PROVENANCE_BINDING_KEYS
        + trainer.GLOBAL_EXACT_PROVENANCE_BINDING_KEYS)
    expected_keys = _QUALIFICATION_BASE_KEYS | provenance_keys
    if (set(report) != expected_keys
            or report.get("schema") != _QUALIFICATION_SCHEMA
            or report.get("status") != trainer.STATUS
            or not _is_digest(digest)
            or digest != trainer.canonical_digest(
                _without(report, "qualification_report_digest"))
            or report.get("scorer_contract_v1_2_digest")
            != trainer.contract_digest()
            or report.get("current_scorer_contract_v1_2_digest")
            != trainer.contract_digest()
            or report.get("global_exact_execution_amendment_digest")
            != expected_execution_amendment_digest
            or report.get("global_exact_successor_scorer_contract_digest")
            != expected_successor_contract_digest
            or report.get("qualification_evaluations") != 1
            or not isinstance(criteria, Mapping)
            or set(criteria) != _QUALIFICATION_CRITERION_KEYS
            or any(type(value) is not bool for value in criteria.values())
            or report.get("qualified") is not all(criteria.values())
            or report.get("fit_states") != trainer.EXPECTED_FIT_STATES
            or report.get("calibration_states")
            != trainer.EXPECTED_CALIBRATION_STATES
            or report.get("fit_rows") != trainer.EXPECTED_FIT_ROWS
            or report.get("calibration_rows")
            != trainer.EXPECTED_CALIBRATION_ROWS
            or report.get("scene_disjoint") is not True
            or report.get("epoch_selection_permitted") is not False
            or report.get("predictor_checkpoints_loaded") != 0
            or report.get("qualification_input")
            != "scene-disjoint true H=1..4 target latent trajectories"):
        raise GlobalExactRunnerError("qualification terminal is not exact")

    try:
        lineage = trainer.validate_global_exact_scorer_contract_lineage(
            report.get("global_exact_scorer_contract_lineage"))
        provenance = trainer.scorer_provenance_bindings(
            report.get("corpus_bindings", {}))
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise GlobalExactRunnerError(
            "qualification scorer-contract lineage is invalid") from exc
    if (lineage.get(
            "scientific_predecessor_scorer_contract_v1_2_digest")
            != expected_scientific_predecessor_scorer_contract_v1_2_digest
            or lineage.get("current_scorer_contract_v1_2_digest")
            != report.get("current_scorer_contract_v1_2_digest")
            or lineage.get("global_exact_successor_scorer_contract_digest")
            != expected_successor_contract_digest
            or any(report.get(key) != value
                   for key, value in provenance.items())):
        raise GlobalExactRunnerError(
            "qualification scorer-contract lineage binding changed")

    digest_fields = {
        key for key in expected_keys
        if key.endswith("_digest") or key.endswith("_sha256")
    } - {
        "qualification_report_digest", "scorer_package_sha256",
        "failed_scorer_sha256",
    }
    if any(not _is_digest(report.get(key)) for key in digest_fields):
        raise GlobalExactRunnerError(
            "qualification digest binding is malformed")
    if not _is_digest(report.get("training_run_digest")):
        raise GlobalExactRunnerError("qualification run digest is malformed")
    source_commit = report.get("source_repository_commit")
    if (not isinstance(source_commit, str) or len(source_commit) != 40
            or any(character not in _HEX for character in source_commit)):
        raise GlobalExactRunnerError(
            "qualification source commit binding is malformed")
    predecessor = report.get("scientific_predecessor_launch_bindings")
    if (not isinstance(predecessor, Mapping)
            or set(predecessor)
            != set(trainer.SCIENTIFIC_PREDECESSOR_LAUNCH_BINDING_KEYS)
            or not all(
                (isinstance(value, str) and len(value) == 40
                 and all(character in _HEX for character in value))
                if key == "source_repository_commit" else _is_digest(value)
                for key, value in predecessor.items())):
        raise GlobalExactRunnerError(
            "qualification predecessor launch binding changed")

    try:
        recomputed_criteria, recomputed_details, recomputed_dominance = (
            trainer.qualification_criteria(
                report["latent_scorer"]["calibration"],
                report["no_latent_baseline"]["calibration"],
                report["label_distributions"]["fit"]["overall"],
                report["label_distributions"]["calibration"]["overall"],
            )
        )
    except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
        raise GlobalExactRunnerError(
            "qualification metrics cannot reproduce the frozen gate") from exc
    if (dict(criteria) != recomputed_criteria
            or report.get("criterion_details") != recomputed_details
            or report.get("baseline_dominance_pairwise")
            != recomputed_dominance):
        raise GlobalExactRunnerError(
            "qualification verdict differs from frozen criteria")

    baseline = report.get("no_latent_baseline_package")
    baseline_provenance_keys = frozenset(provenance)
    final_state_digests = report.get("final_state_digests")
    baseline_expected_keys = {
        "schema", "status", "complete", "training_run_digest",
        "scorer_contract_v1_2_digest", *baseline_provenance_keys,
        "path", "sha256", "byte_count", "final_state_digest",
        "final_epoch", "epoch_selection", "receipt_digest",
    }
    if (not isinstance(baseline, Mapping)
            or set(baseline) != baseline_expected_keys
            or baseline.get("schema")
            != "go2_utility_no_latent_baseline_receipt_v1_2"
            or baseline.get("status") != trainer.STATUS
            or baseline.get("complete") is not True
            or baseline.get("training_run_digest")
            != report["training_run_digest"]
            or baseline.get("scorer_contract_v1_2_digest")
            != report["scorer_contract_v1_2_digest"]
            or any(baseline.get(key) != report.get(key)
                   for key in baseline_provenance_keys)
            or not _is_digest(baseline.get("receipt_digest"))
            or baseline["receipt_digest"] != trainer.canonical_digest(
                _without(baseline, "receipt_digest"))
            or not _is_digest(baseline.get("sha256"))
            or type(baseline.get("byte_count")) is not int
            or baseline["byte_count"] <= 0
            or not isinstance(final_state_digests, Mapping)
            or baseline.get("final_state_digest")
            != final_state_digests.get("no_latent")
            or type(baseline.get("final_epoch")) is not int
            or baseline["final_epoch"] <= 0
            or baseline.get("epoch_selection")
            != "final_epoch_only_no_selection"):
        raise GlobalExactRunnerError(
            "qualification no-latent baseline receipt changed")
    expected_baseline_relative = Path(
        ".generated/go2_utility_scorer_v1_2") / (
            f"no_latent_baseline_{report['training_run_digest'][:16]}.pt")
    supplied_baseline = Path(baseline["path"])
    supplied_baseline = (supplied_baseline if supplied_baseline.is_absolute()
                         else root / supplied_baseline)
    if supplied_baseline.absolute() != (
            root / expected_baseline_relative).absolute():
        raise GlobalExactRunnerError(
            "qualification no-latent baseline path changed")
    baseline_path = _require_managed_file(
        root, str(supplied_baseline), label="no-latent baseline package")
    if (baseline_path.stat().st_size != baseline["byte_count"]
            or _sha256_file(baseline_path) != baseline["sha256"]):
        raise GlobalExactRunnerError(
            "qualification no-latent baseline bytes changed")

    package_sha = report.get("scorer_package_sha256")
    failed_sha = report.get("failed_scorer_sha256")
    if report["qualified"]:
        if not _is_digest(package_sha) or failed_sha is not None:
            raise GlobalExactRunnerError(
                "qualified scorer package disposition changed")
        package_path = _pinned_relative(
            root, Path(".generated/go2_utility_scorer_v1_2/scorer_package.pt"),
            label="qualified scorer package")
        if (not package_path.is_file() or package_path.is_symlink()
                or _sha256_file(package_path) != package_sha):
            raise GlobalExactRunnerError("qualified scorer package changed")
        receipt_path = _pinned_relative(
            root,
            Path(".generated/go2_utility_scorer_v1_2/"
                 "scorer_package_receipt.json"),
            label="qualified scorer package receipt")
        receipt, _receipt_raw = _load_json(
            receipt_path, label="qualified scorer package receipt")
        if (receipt.get("schema")
                != "go2_utility_scorer_package_receipt_v1_2"
                or receipt.get("status") != trainer.STATUS
                or receipt.get("complete") is not True
                or receipt.get("qualified") is not True
                or receipt.get("scorer_package_sha256") != package_sha
                or receipt.get("training_run_digest")
                != report["training_run_digest"]
                or receipt.get("scorer_contract_v1_2_digest")
                != report["scorer_contract_v1_2_digest"]
                or receipt.get("scorer_package_receipt_digest")
                != trainer.canonical_digest(_without(
                    receipt, "scorer_package_receipt_digest"))):
            raise GlobalExactRunnerError(
                "qualified scorer package receipt changed")
    else:
        if package_sha is not None or not _is_digest(failed_sha):
            raise GlobalExactRunnerError(
                "failed scorer package disposition changed")
        failed_path = _pinned_relative(
            root,
            Path(".generated/go2_utility_scorer_v1_2") /
            f"failed_scorer_{report['training_run_digest'][:16]}.pt",
            label="failed scorer package")
        if (not failed_path.is_file() or failed_path.is_symlink()
                or _sha256_file(failed_path) != failed_sha):
            raise GlobalExactRunnerError("failed scorer package changed")
    return report


def _load_development_result(root: Path) -> dict[str, Any]:
    # Reuse the producing module's exact scorer/spec/result validator instead
    # of treating the exploratory output's self digest as sufficient.
    from scripts import (
        apply_go2_utility_scorer_to_counterfactual_development_v1_2 as app,
    )

    path = _pinned_relative(
        root,
        Path(".generated/go2_utility_scorer_v1_2/"
             "counterfactual_development_transfer_v1_2/result.json"),
        label="development transfer result")
    result, _raw = _load_json(path, label="development transfer result")
    scorer = app.validate_qualified_scorer()
    spec = app.prospective_spec(scorer)
    app.validate_existing_result(
        result, spec["development_transfer_spec_digest"], scorer)
    if (result.get("schema") != _DEVELOPMENT_RESULT_SCHEMA
            or result.get("status") != _DEVELOPMENT_STATUS
            or not _is_digest(result.get("result_digest"))
            or result.get("result_digest")
            != app.legacy_digest(result, ("result_digest",))
            or result.get("complete") is not True):
        raise GlobalExactRunnerError("development transfer result is incomplete")
    return result


def _build_downstream_validation_receipt(
        *, artifact_kind: str, projection: Mapping[str, Any], authority: Any,
        ) -> dict[str, Any]:
    if artifact_kind == "qualification":
        expected_projection_keys = {
            "qualified", "qualification_report_digest",
            "global_exact_execution_amendment_digest",
            "global_exact_successor_scorer_contract_digest",
        }
        valid_projection = (
            type(projection.get("qualified")) is bool
            and all(_is_digest(projection.get(key)) for key in
                    expected_projection_keys - {"qualified"}))
    elif artifact_kind == "development":
        expected_projection_keys = {
            "complete", "development_transfer_result_digest",
            "qualification_report_digest",
            "global_exact_execution_amendment_digest",
            "global_exact_successor_scorer_contract_digest",
        }
        valid_projection = (
            projection.get("complete") is True
            and all(_is_digest(projection.get(key)) for key in
                    expected_projection_keys - {"complete"}))
    else:
        raise GlobalExactRunnerError(
            "downstream validation artifact kind changed")
    if set(projection) != expected_projection_keys or not valid_projection:
        raise GlobalExactRunnerError(
            "downstream validation projection changed")
    runtime_role = "rocm"
    contract = _runtime_contract(authority, runtime_role)
    return _signed({
        "schema": DOWNSTREAM_VALIDATION_SCHEMA,
        "status": DOWNSTREAM_VALIDATION_STATUS,
        "artifact_kind": artifact_kind,
        "runtime_role": runtime_role,
        "runtime_contract_digest": canonical_digest(contract),
        "interpreter_relative_path": contract[
            "interpreter_relative_path"],
        "pyvenv_config_sha256": contract["pyvenv_config_sha256"],
        "validated_projection": dict(projection),
    }, DOWNSTREAM_VALIDATION_SELF_KEY)


def _validate_downstream_validation_receipt(
        receipt: Mapping[str, Any], *, artifact_kind: str, authority: Any,
        ) -> dict[str, Any]:
    if not isinstance(receipt, Mapping):
        raise GlobalExactRunnerError(
            "downstream validation receipt is not a mapping")
    payload = dict(receipt)
    expected_keys = {
        "schema", "status", "artifact_kind", "runtime_role",
        "runtime_contract_digest", "interpreter_relative_path",
        "pyvenv_config_sha256", "validated_projection",
        DOWNSTREAM_VALIDATION_SELF_KEY,
    }
    contract = _runtime_contract(authority, "rocm")
    if (set(payload) != expected_keys
            or payload.get("schema") != DOWNSTREAM_VALIDATION_SCHEMA
            or payload.get("status") != DOWNSTREAM_VALIDATION_STATUS
            or payload.get("artifact_kind") != artifact_kind
            or payload.get("runtime_role") != "rocm"
            or payload.get("runtime_contract_digest")
            != canonical_digest(contract)
            or payload.get("interpreter_relative_path")
            != contract["interpreter_relative_path"]
            or payload.get("pyvenv_config_sha256")
            != contract["pyvenv_config_sha256"]
            or payload.get(DOWNSTREAM_VALIDATION_SELF_KEY)
            != canonical_digest(_without(
                payload, DOWNSTREAM_VALIDATION_SELF_KEY))):
        raise GlobalExactRunnerError(
            "downstream validation receipt binding changed")
    expected = _build_downstream_validation_receipt(
        artifact_kind=artifact_kind,
        projection=payload.get("validated_projection", {}),
        authority=authority)
    if payload != expected:
        raise GlobalExactRunnerError(
            "downstream validation receipt is not exact")
    return payload


def _emit_downstream_validation(
        artifact_kind: str, *, root: Path = ROOT,
        authority: Any = AUTHORITY) -> int:
    contract = _runtime_contract(authority, "rocm")
    expected_interpreter = _pinned_relative(
        root, Path(contract["interpreter_relative_path"]),
        label="bound ROCm downstream interpreter")
    config = _pinned_relative(
        root, Path(contract["pyvenv_config_relative_path"]),
        label="bound ROCm downstream pyvenv config")
    if (Path(sys.executable).absolute() != expected_interpreter.absolute()
            or not config.is_file() or config.is_symlink()
            or config.stat().st_size
            != contract["pyvenv_config_byte_count"]
            or _sha256_file(config)
            != contract["pyvenv_config_sha256"]):
        raise GlobalExactRunnerError(
            "artifact validator is not the bound downstream interpreter")
    _build_downstream_runtime_probe_receipt(
        runtime_role="rocm",
        observation=_observe_current_downstream_runtime("rocm"),
        authority=authority)
    successor = _validate_downstream_manifest_boundary(root)
    qualification = _load_qualification(
        root,
        expected_execution_amendment_digest=successor[
            "global_exact_execution_amendment_digest"],
        expected_successor_contract_digest=successor[
            "global_exact_successor_scorer_contract_digest"],
        expected_scientific_predecessor_scorer_contract_digest=successor[
            "scientific_predecessor_scorer_contract_v1_2_digest"])
    common = {
        "qualification_report_digest": qualification[
            "qualification_report_digest"],
        "global_exact_execution_amendment_digest": qualification[
            "global_exact_execution_amendment_digest"],
        "global_exact_successor_scorer_contract_digest": qualification[
            "global_exact_successor_scorer_contract_digest"],
    }
    if artifact_kind == "qualification":
        projection = {"qualified": qualification["qualified"], **common}
    elif artifact_kind == "development":
        development = _load_development_result(root)
        projection = {
            "complete": True,
            "development_transfer_result_digest": development[
                "result_digest"],
            **common,
        }
    else:
        raise GlobalExactRunnerError(
            "downstream validation artifact kind changed")
    receipt = _build_downstream_validation_receipt(
        artifact_kind=artifact_kind, projection=projection,
        authority=authority)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


def _default_downstream_validation_invoker(
        artifact_kind: str, root: Path, downstream_python: Path,
        authority: Any) -> Mapping[str, Any]:
    stage = f"internal-validate-{artifact_kind}"
    command = [
        str(downstream_python), str(root / RUNNER_RELATIVE_PATH),
        "--stage", stage,
    ]
    completed = subprocess.run(
        command, cwd=root, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise GlobalExactRunnerError(
            f"bound downstream {artifact_kind} validator failed")
    try:
        payload = json.loads(completed.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise GlobalExactRunnerError(
            f"bound downstream {artifact_kind} validator output changed") from exc
    return _validate_downstream_validation_receipt(
        payload, artifact_kind=artifact_kind, authority=authority)


def _continue_downstream(*, root: Path, command_runner: CommandRunner,
                         downstream_interpreters: Mapping[str, Path],
                         runtime_probe_receipts: Mapping[
                             str, Mapping[str, Any]],
                         validation_invoker: DownstreamValidationInvoker,
                         authority: Any,
                         expected_execution_amendment_digest: str,
                         expected_successor_contract_digest: str,
                         ) -> dict[str, Any]:
    if (set(downstream_interpreters) != set(_DOWNSTREAM_RUNTIME_ROLES)
            or set(runtime_probe_receipts)
            != set(_DOWNSTREAM_RUNTIME_ROLES)):
        raise GlobalExactRunnerError(
            "downstream runtime readiness surface changed")
    validated_probes = {
        role: _validate_downstream_runtime_probe_receipt(
            runtime_probe_receipts[role], runtime_role=role,
            authority=authority)
        for role in _DOWNSTREAM_RUNTIME_ROLES
    }
    commands = downstream_command_sequence(
        root, downstream_interpreters=downstream_interpreters,
        authority=authority)
    stage_names = _DOWNSTREAM_COMMAND_STAGE_NAMES
    for stage, command in zip(stage_names, commands, strict=True):
        expected_role = authority.DOWNSTREAM_STAGE_RUNTIME_ROLES[stage]
        if Path(command[0]) != Path(downstream_interpreters[expected_role]):
            raise GlobalExactRunnerError(
                f"downstream stage {stage} escaped its bound runtime")
    probe_digests = {
        role: validated_probes[role][DOWNSTREAM_RUNTIME_PROBE_SELF_KEY]
        for role in _DOWNSTREAM_RUNTIME_ROLES
    }
    completed: list[str] = []
    for index, (stage, command) in enumerate(zip(stage_names, commands, strict=True)):
        return_code = command_runner(command, root)
        if type(return_code) is not int:
            raise GlobalExactRunnerError(
                f"downstream command runner returned no integer for {stage}")
        if stage == "scorer_training_and_qualification":
            validation = _validate_downstream_validation_receipt(
                validation_invoker(
                    "qualification", root,
                    Path(downstream_interpreters["rocm"]), authority),
                artifact_kind="qualification", authority=authority)
            qualification = validation["validated_projection"]
            if (qualification.get("global_exact_execution_amendment_digest")
                    != expected_execution_amendment_digest
                    or qualification.get(
                        "global_exact_successor_scorer_contract_digest")
                    != expected_successor_contract_digest):
                raise GlobalExactRunnerError(
                    "qualification binds a different global-exact successor")
            if return_code == 1 and qualification["qualified"] is False:
                return {
                    "status": "STOP_FROZEN_SCORER_QUALIFICATION_FAILURE",
                    "completed_stages": completed + [stage],
                    "qualification_report_digest": qualification[
                        "qualification_report_digest"],
                    "qualified": False,
                    "development_transfer_started": False,
                    "downstream_runtime_probe_digests": probe_digests,
                    "nothing_running": True,
                }
            if return_code != 0 or qualification["qualified"] is not True:
                raise GlobalExactRunnerError(
                    "scorer training did not produce its exact terminal verdict")
        elif return_code != 0:
            raise GlobalExactRunnerError(
                f"downstream stage {stage} exited {return_code}")
        completed.append(stage)
        if index == len(commands) - 1:
            validation = _validate_downstream_validation_receipt(
                validation_invoker(
                    "development", root,
                    Path(downstream_interpreters["rocm"]), authority),
                artifact_kind="development", authority=authority)
            development = validation["validated_projection"]
            qualification = development
            if (qualification.get("global_exact_execution_amendment_digest")
                    != expected_execution_amendment_digest
                    or qualification.get(
                        "global_exact_successor_scorer_contract_digest")
                    != expected_successor_contract_digest):
                raise GlobalExactRunnerError(
                    "qualified scorer successor binding changed")
            return {
                "status": "COMPLETE_AUTHORISED_DEVELOPMENT_TRANSFER",
                "completed_stages": completed,
                "qualification_report_digest": qualification[
                    "qualification_report_digest"],
                "qualified": True,
                "development_transfer_result_digest": development[
                    "development_transfer_result_digest"],
                "downstream_runtime_probe_digests": probe_digests,
                "final_200_state_corpus_generated": False,
                "nothing_running": True,
            }
    raise AssertionError("downstream command sequence is empty")


def solve_and_continue(*, root: Path = ROOT, builder: Any = BUILDER,
                       authority: Any = AUTHORITY, model: Any = MODEL,
                       command_runner: CommandRunner = _default_command_runner,
                       runtime_probe_invoker: DownstreamRuntimeProbeInvoker =
                       _default_downstream_runtime_probe_invoker,
                       validation_invoker: DownstreamValidationInvoker =
                       _default_downstream_validation_invoker,
                       ) -> tuple[int, dict[str, Any]]:
    """Run or resume the single exact allocation and authorised continuation."""

    # Validate the committed amendment without opening scientific masks.  The
    # mandatory source-only/synthetic fixture proof is completed at this side
    # of the boundary.
    pre_mask = _require_context(
        builder.load_global_exact_execution_context(
            attach_scientific_masks=False),
        masks=False, authority=authority)
    fixture_suite = model.validate_fixture_suite_result(
        model.build_fixture_suite_result())

    # This is the first operation permitted to open the seven frozen vectors.
    mask_context = _require_context(
        builder.load_global_exact_execution_context(
            attach_scientific_masks=True),
        masks=True, authority=authority)
    instance = model.validate_production_instance(
        builder.build_global_exact_production_instance(mask_context))
    model_plan = model.validate_execution_plan(
        instance, model.build_execution_plan(instance))
    proposed_plan = _build_runner_plan(
        pre_mask_context=pre_mask, mask_context=mask_context,
        fixture_suite=fixture_suite, instance=instance,
        model_plan=model_plan, authority=authority, model=model)

    plan_path = _pinned_relative(
        root, _runtime_relative(authority, _PLAN_LABEL),
        label="global exact model plan")
    terminal_path = _pinned_relative(
        root, _runtime_relative(authority, _TERMINAL_LABEL),
        label="global exact terminal result")
    infeasible_path = _pinned_relative(
        root, _runtime_relative(authority, _INFEASIBILITY_LABEL),
        label="global exact terminal infeasibility")
    if terminal_path.is_symlink() or infeasible_path.is_symlink():
        raise GlobalExactRunnerError("global exact terminal path is symlinked")
    if terminal_path.exists() and infeasible_path.exists():
        raise GlobalExactRunnerError("both mutually exclusive terminals exist")
    plan = _install_or_reopen(
        plan_path, proposed_plan, label="global exact model plan")
    _validate_runner_plan(
        plan, pre_mask_context=pre_mask, mask_context=mask_context,
        fixture_suite=fixture_suite, instance=instance,
        model_plan=model_plan, authority=authority, model=model)

    if infeasible_path.exists():
        prior, _raw = _load_json(
            infeasible_path, label="global exact terminal infeasibility")
        terminal = _validate_infeasible_terminal(
            prior, plan=plan, instance=instance, model=model)
        summary = {
            "status": terminal["status"],
            "coupling_classification": "COUPLED",
            "selected_method": "ONE_GLOBAL_EXACT_FEASIBILITY_MODEL",
            "execution_amendment_digest": plan["execution_amendment_digest"],
            "fixture_suite_digest": plan["fixture_suite_digest"],
            "global_exact_model_plan_digest": plan[PLAN_SELF_KEY],
            "global_exact_terminal_infeasibility_digest": terminal[
                INFEASIBILITY_SELF_KEY],
            "exact_infeasibility_proved": True,
            "candidate_outcomes_consumed": False,
            "downstream_started": False,
            "nothing_running": True,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 2, summary

    if terminal_path.exists():
        prior, _raw = _load_json(
            terminal_path, label="global exact terminal result")
        terminal = _validate_feasible_terminal(
            prior, plan=plan, instance=instance, model=model)
    else:
        start_wall = time.monotonic()
        start_cpu = time.process_time()
        model_result = model.validate_execution_result(
            instance, model_plan, model.solve_once(instance, model_plan))
        wall_s, cpu_s = _elapsed(start_wall, start_cpu)
        result_status = model_result.get("status")
        if result_status == model.EXECUTION_INFEASIBLE_STATUS:
            terminal = _build_infeasible_terminal(
                plan=plan, model_result=model_result,
                elapsed_wall_s=wall_s, elapsed_cpu_s=cpu_s, model=model)
            _exclusive_json(
                infeasible_path, terminal,
                label="global exact terminal infeasibility")
            terminal = _validate_infeasible_terminal(
                terminal, plan=plan, instance=instance, model=model)
            summary = {
                "status": terminal["status"],
                "coupling_classification": "COUPLED",
                "selected_method": "ONE_GLOBAL_EXACT_FEASIBILITY_MODEL",
                "execution_amendment_digest": plan[
                    "execution_amendment_digest"],
                "fixture_suite_digest": plan["fixture_suite_digest"],
                "global_exact_model_plan_digest": plan[PLAN_SELF_KEY],
                "global_exact_terminal_infeasibility_digest": terminal[
                    INFEASIBILITY_SELF_KEY],
                "elapsed_wall_s": terminal["elapsed_wall_s"],
                "elapsed_cpu_s": terminal["elapsed_cpu_s"],
                "exact_infeasibility_proved": True,
                "candidate_outcomes_consumed": False,
                "downstream_started": False,
                "nothing_running": True,
            }
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 2, summary
        if result_status != model.EXECUTION_PASS_STATUS:
            raise GlobalExactRunnerError(
                "model terminal does not state exact feasibility")
        materialized = model_result.get("materialized_allocation")
        if not isinstance(materialized, Mapping):
            raise GlobalExactRunnerError(
                "feasible model terminal lacks a materialized allocation")
        terminal = _build_feasible_terminal(
            plan=plan, model_result=model_result, materialized=materialized,
            elapsed_wall_s=wall_s, elapsed_cpu_s=cpu_s, model=model)
        _exclusive_json(
            terminal_path, terminal, label="global exact terminal result")
        terminal = _validate_feasible_terminal(
            terminal, plan=plan, instance=instance, model=model)

    finalized = builder.finalize_global_exact_feasible_allocation(
        execution_context=mask_context, instance=instance,
        execution_plan=plan, execution_result=terminal)
    if not isinstance(finalized, Mapping):
        raise GlobalExactRunnerError("global exact finalizer returned no receipt")
    successor_digest = finalized.get(
        "global_exact_successor_scorer_contract_digest")
    if not _is_digest(successor_digest):
        raise GlobalExactRunnerError(
            "global exact finalizer lacks the successor scorer contract digest")
    downstream_interpreters = _bound_downstream_interpreters(
        root=root, execution_context=mask_context, authority=authority)
    runtime_probe_receipts: dict[str, dict[str, Any]] = {}
    for runtime_role in _DOWNSTREAM_RUNTIME_ROLES:
        runtime_probe_receipts[runtime_role] = (
            _validate_downstream_runtime_probe_receipt(
                runtime_probe_invoker(
                    runtime_role, root,
                    downstream_interpreters[runtime_role], authority),
                runtime_role=runtime_role, authority=authority))
    downstream = _continue_downstream(
        root=root, command_runner=command_runner,
        downstream_interpreters=downstream_interpreters,
        runtime_probe_receipts=runtime_probe_receipts,
        validation_invoker=validation_invoker,
        authority=authority,
        expected_execution_amendment_digest=plan[
            "execution_amendment_digest"],
        expected_successor_contract_digest=successor_digest)
    summary = {
        "status": downstream["status"],
        "coupling_classification": "COUPLED",
        "selected_method": "ONE_GLOBAL_EXACT_FEASIBILITY_MODEL",
        "execution_amendment_digest": plan["execution_amendment_digest"],
        "fixture_suite_digest": plan["fixture_suite_digest"],
        "global_exact_model_plan_digest": plan[PLAN_SELF_KEY],
        "global_exact_terminal_result_digest": terminal[TERMINAL_SELF_KEY],
        "elapsed_wall_s": terminal["elapsed_wall_s"],
        "elapsed_cpu_s": terminal["elapsed_cpu_s"],
        "selected_scene_ids": terminal["selected_scene_ids"],
        "finalization": dict(finalized),
        "downstream": downstream,
        "candidate_outcomes_consumed_for_allocation": False,
        "final_200_state_corpus_generated": False,
        "nothing_running": True,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return (4 if downstream["qualified"] is False else 0), summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", required=True,
        choices=(
            "issue-report", "issue-amendment", "issue-source-correction",
            "issue-preplan-integration-correction",
            "solve-and-continue",
            "internal-probe-genesis-runtime",
            "internal-probe-rocm-runtime",
            "internal-validate-qualification",
            "internal-validate-development",
        ))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.stage == "internal-probe-genesis-runtime":
        return _emit_downstream_runtime_probe("genesis")
    if args.stage == "internal-probe-rocm-runtime":
        return _emit_downstream_runtime_probe("rocm")
    if args.stage == "internal-validate-qualification":
        return _emit_downstream_validation("qualification")
    if args.stage == "internal-validate-development":
        return _emit_downstream_validation("development")
    if args.stage == "issue-report":
        issue_report()
        return 0
    if args.stage == "issue-amendment":
        issue_amendment()
        return 0
    if args.stage == "issue-source-correction":
        issue_source_correction()
        return 0
    if args.stage == "issue-preplan-integration-correction":
        issue_preplan_integration_correction()
        return 0
    code, _summary = solve_and_continue()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
