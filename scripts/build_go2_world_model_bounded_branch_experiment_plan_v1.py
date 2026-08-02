#!/usr/bin/env python3
"""Build the post-calibration, scene-disjoint bounded WM-A pilot plan.

This is a metadata-only builder.  It will not emit a plan unless an exact
calibration receipt says ``FREEZE_PILOT_CONTRACT`` and an independent review
binds that receipt to either successful supervision or the one narrowly
admissible analyzer-integration recovery from an immutable, checker-passed
collection.  It opens ordinary scene metadata named by the caller-bound panel;
it never opens RGB, checkpoints, or protected evaluation material.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as pilot  # noqa: E402
from scripts import analyze_go2_world_model_counterfactual_calibration_v1 as calibration  # noqa: E402
from scripts import build_go2_world_model_counterfactual_calibration_authority_v1 as calibration_authority  # noqa: E402
from scripts import check_go2_world_model_counterfactual_pilot_v1 as receipt_checker  # noqa: E402
from scripts import run_go2_world_model_counterfactual_calibration_authorized_v1 as calibration_supervisor  # noqa: E402
from scripts import evaluate_go2_world_model_bounded_branch_experiment_v1 as evaluation  # noqa: E402
from scripts import evaluate_go2_world_model_visual_domain_parity_v1 as parity_evaluator  # noqa: E402
from scripts import build_go2_world_model_bounded_branch_scene_panel_v1 as panel_selector  # noqa: E402
from scripts.build_go2_world_model_counterfactual_calibration_plan_v1 import (  # noqa: E402
    RUNTIME_CONTRACT_SCHEMA,
    _canonical_manifest_target,
)


PANEL_SCHEMA = panel_selector.PANEL_SCHEMA
TERMINAL_REVIEW_SCHEMA = (
    "lewm_go2_world_model_counterfactual_calibration_terminal_review_v1"
)
ANALYSIS_RECOVERY_REVIEW_SCHEMA = (
    "lewm_go2_world_model_counterfactual_calibration_posthoc_recovery_review_v1"
)
ANALYSIS_RECOVERY_REVIEW_STATUS = (
    "PASS_ANALYZER_ONLY_RECOVERY_FREEZE_PILOT_CONTRACT"
)
ANALYSIS_RECOVERY_TERMINAL_FAILURE = (
    "CalibrationSupervisionError: supervised command exited with status 1"
)
ANALYSIS_RECOVERY_FAILED_TERMINAL_SHA256 = (
    "b601809a8ec318b5848ac9d552619a7577a837e7edc6a707f995d4ba8bff35e1"
)
ANALYSIS_RECOVERY_FAILED_TERMINAL_BYTE_COUNT = 6925
ANALYSIS_RECOVERY_SEMANTIC_CHANGE = (
    "runtime analyzer predicate only: sentinel_audit.passed -> "
    "sentinel_audit.physics_equal"
)
ANALYSIS_RECOVERY_LIMITATIONS = (
    "original_terminal_remains_failed_and_non_citable",
    "posthoc_receipt_does_not_replace_original_terminal",
    "admission_is_calibration_freeze_only",
    "calibration_scenes_and_branches_remain_excluded_from_train_and_eval",
    "no_visual_domain_fidelity_claim",
    "no_retry_resume_refill_overwrite_or_new_execution_authority",
    "fresh_2304_branch_bounded_experiment_and_reviews_still_required",
)
CALIBRATION_TERMINAL_SCHEMA = calibration_supervisor.TERMINAL_SCHEMA
VISUAL_DOMAIN_PARITY_RESULT_SCHEMA = parity_evaluator.RESULT_SCHEMA
VISUAL_DOMAIN_PARITY_RESULT_STATUS = parity_evaluator.PASS_STATUS
VISUAL_DOMAIN_PARITY_REVIEW_SCHEMA = pilot.TEXTURED_V03_PARITY_REVIEW_SCHEMA
VISUAL_DOMAIN_PARITY_REVIEW_STATUS = pilot.TEXTURED_V03_PARITY_REVIEW_PASS_STATUS
VISUAL_DOMAIN_PARITY_EVALUATOR_RELATIVE = parity_evaluator.COMPARISON_CONTRACT[
    "evaluator_source_path"
]
VISUAL_DOMAIN_PARITY_COMPARISON_CONTRACT = copy.deepcopy(
    parity_evaluator.COMPARISON_CONTRACT
)
VISUAL_DOMAIN_PARITY_THRESHOLDS = copy.deepcopy(parity_evaluator.THRESHOLDS)
BOUNDED_RENDER_CONTRACT_V1 = copy.deepcopy(pilot.TEXTURED_V03_RENDER_CONTRACT)
PURPOSE = "bounded_wm_a_pilot"
SCENES_PER_FAMILY_PER_ROLE = 2
STATES_PER_SCENE = 8
ROLE_NAMES = ("train", "eval")
EXPECTED_SCENES = len(pilot.FAMILIES) * len(ROLE_NAMES) * SCENES_PER_FAMILY_PER_ROLE
EXPECTED_STATES = EXPECTED_SCENES * STATES_PER_SCENE
EXPECTED_BRANCHES = EXPECTED_STATES * pilot.ACTION_COUNT
EXPECTED_STORED_FRAMES = EXPECTED_STATES * (pilot.CONTEXT_FRAME_COUNT + pilot.ACTION_COUNT)
HISTORY_PANEL = panel_selector.HISTORY_PANEL
MODEL_PANEL_FREEZE_FIELDS = (
    "progression_analysis_binding",
    "training_result_binding",
    "progression_proxy_routing",
    "checkpoint_panel_bindings",
    "model_observational_scene_ids",
    "model_observational_scene_count",
    "predecessor_terminal_access_binding",
    "predecessor_index_bindings",
    "predecessor_place_manifest_binding",
    "training_pack_role_bindings",
    "training_pack_metadata_bindings",
)
SCENE_PANEL_FREEZE_FIELDS = (
    "scene_panel_binding",
    "scene_panel_schema",
    "scene_selection_contract",
    "scene_corpus_manifest_bindings",
    "scene_inventory_unique_train_scenes",
    "scene_eligible_counts_by_family",
    "scene_excluded_scene_ids_sha256",
    "scene_selection_rows",
)
_ATTEMPT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}$")


class BoundedBranchPlanError(RuntimeError):
    """Raised before invalid metadata can become pilot experiment identity."""


def _calibration_gate_profile(
    receipt: Mapping[str, Any],
) -> dict[str, str]:
    schema = receipt.get("schema")
    if schema == calibration.TEXTURED_V03_CALIBRATION_RECEIPT_SCHEMA:
        return {
            "name": "textured_v03_v3",
            "receipt_schema": calibration.TEXTURED_V03_CALIBRATION_RECEIPT_SCHEMA,
            "purpose": calibration.TEXTURED_V03_CALIBRATION_PURPOSE,
            "terminal_schema": calibration_supervisor.TEXTURED_V03_TERMINAL_SCHEMA,
            "authority_schema": calibration_authority.TEXTURED_V03_AUTHORITY_SCHEMA,
            "authority_status": calibration_authority.TEXTURED_V03_AUTHORITY_STATUS,
        }
    if schema == calibration.CALIBRATION_RECEIPT_SCHEMA:
        return {
            "name": "legacy_v2",
            "receipt_schema": calibration.CALIBRATION_RECEIPT_SCHEMA,
            "purpose": calibration_authority.PURPOSE,
            "terminal_schema": calibration_supervisor.TERMINAL_SCHEMA,
            "authority_schema": calibration_authority.AUTHORITY_SCHEMA,
            "authority_status": calibration_authority.AUTHORITY_STATUS,
        }
    raise BoundedBranchPlanError("calibration receipt version is unsupported")


def _require_calibration_profile_links(
    profile: Mapping[str, str],
    *,
    terminal_schema: object,
    authority_schema: object,
    authority_status: object,
    plan_purpose: object,
    physics_purpose: object,
    check_purpose: object,
) -> None:
    observed = {
        "terminal_schema": terminal_schema,
        "authority_schema": authority_schema,
        "authority_status": authority_status,
        "plan_purpose": plan_purpose,
        "physics_purpose": physics_purpose,
        "check_purpose": check_purpose,
    }
    expected = {
        "terminal_schema": profile["terminal_schema"],
        "authority_schema": profile["authority_schema"],
        "authority_status": profile["authority_status"],
        "plan_purpose": profile["purpose"],
        "physics_purpose": profile["purpose"],
        "check_purpose": profile["purpose"],
    }
    if observed != expected:
        raise BoundedBranchPlanError(
            f"calibration {profile['name']} cross-version evidence mix rejected"
        )


def _require_calibration_parity_identity(
    gate: Mapping[str, Any], parity_freeze: Mapping[str, Any]
) -> None:
    if (
        gate.get("visual_domain_parity_result_binding")
        != parity_freeze.get("result_binding")
        or gate.get("visual_domain_parity_terminal_binding")
        != parity_freeze.get("terminal_binding")
        or gate.get("visual_domain_parity_review_binding")
        != parity_freeze.get("review_binding")
    ):
        raise BoundedBranchPlanError(
            "bounded parity qualification differs from the V3 calibration prerequisite"
        )


def _reject_protected_path(path: Path, *, label: str) -> None:
    for part in Path(path).parts:
        lowered = part.lower()
        if (
            lowered == "sealed_test.json"
            or lowered == "sealed"
            or lowered.startswith("sealed_")
            or lowered in {"heldout", "held_out", "held-out"}
            or lowered.startswith("heldout_")
            or lowered.startswith("held_out_")
            or lowered.startswith("held-out-")
        ):
            raise BoundedBranchPlanError(f"{label} names protected material")


def _binding(value: object, *, label: str) -> dict[str, Any]:
    if isinstance(value, Mapping) and isinstance(value.get("path"), str):
        _reject_protected_path(Path(str(value["path"])), label=label)
    try:
        return pilot.require_binding(value, label=label)
    except (OSError, pilot.PilotContractError) as exc:
        raise BoundedBranchPlanError(str(exc)) from exc


def _validate_analysis_recovery_review(
    review: Mapping[str, Any],
    *,
    receipt: Mapping[str, Any],
    receipt_binding: Mapping[str, Any],
    terminal_binding: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> None:
    required = {
        "schema",
        "status",
        "authority_granted_by_this_document",
        "scientific_claim_granted",
        "failed_terminal_binding",
        "posthoc_calibration_receipt_binding",
        "source_correction",
        "reviewer",
        "reviewed_at",
        "checks",
        "limitations",
        "remaining_findings",
    }
    checks = review.get("checks")
    required_checks = {
        "failed_terminal_preserved",
        "physics_collection_complete",
        "receipt_checker_passed",
        "frozen_analyzer_failure_reproduced",
        "producer_checker_contract_uses_physics_equal",
        "corrected_source_delta_is_analyzer_field_only",
        "posthoc_output_outside_attempt_root",
        "posthoc_receipt_exactly_recomputed",
        "posthoc_receipt_freezes_contract",
        "attempt_root_unchanged",
        "no_retry_resume_refill_or_overwrite",
    }
    reviewer = review.get("reviewer")
    correction = review.get("source_correction")
    required_correction = {
        "failed_source_commit",
        "failed_analyzer_binding",
        "corrected_source_commit",
        "corrected_analyzer_binding",
        "corrected_analyzer_test_binding",
        "semantic_change",
    }
    if (
        set(review) != required
        or review.get("schema") != ANALYSIS_RECOVERY_REVIEW_SCHEMA
        or review.get("status") != ANALYSIS_RECOVERY_REVIEW_STATUS
        or review.get("authority_granted_by_this_document") is not False
        or review.get("scientific_claim_granted") is not False
        or review.get("failed_terminal_binding") != dict(terminal_binding)
        or review.get("posthoc_calibration_receipt_binding")
        != dict(receipt_binding)
        or not isinstance(correction, Mapping)
        or set(correction) != required_correction
        or correction.get("semantic_change") != ANALYSIS_RECOVERY_SEMANTIC_CHANGE
        or not isinstance(correction.get("corrected_source_commit"), str)
        or re.fullmatch(
            r"[0-9a-f]{40}", str(correction.get("corrected_source_commit"))
        ) is None
        or review.get("limitations") != list(ANALYSIS_RECOVERY_LIMITATIONS)
        or review.get("remaining_findings") != []
        or not isinstance(reviewer, Mapping)
        or set(reviewer) != {"identity", "independence_basis"}
        or any(
            not isinstance(reviewer[key], str) or not reviewer[key].strip()
            for key in reviewer
        )
        or not isinstance(checks, Mapping)
        or set(checks) != required_checks
        or any(value is not True for value in checks.values())
    ):
        raise BoundedBranchPlanError(
            "calibration analyzer-recovery review did not pass exactly"
        )
    try:
        datetime.fromisoformat(str(review.get("reviewed_at")).replace("Z", "+00:00"))
    except ValueError as exc:
        raise BoundedBranchPlanError(
            "calibration analyzer-recovery review time is invalid"
        ) from exc

    source_rows = authority.get("source_bindings")
    original_analyzers = [
        row.get("binding")
        for row in source_rows
        if isinstance(source_rows, list)
        and isinstance(row, Mapping)
        and row.get("name") == "calibration_analyzer"
    ] if isinstance(source_rows, list) else []
    corrected_analyzer = _binding(
        correction.get("corrected_analyzer_binding"),
        label="corrected calibration analyzer",
    )
    corrected_test = _binding(
        correction.get("corrected_analyzer_test_binding"),
        label="corrected calibration analyzer test",
    )
    corrected_commit = str(correction.get("corrected_source_commit"))
    collector = calibration_supervisor.collector
    try:
        head = str(collector._git_output("rev-parse", "HEAD"))  # noqa: SLF001
        collector._git_output(  # noqa: SLF001
            "merge-base", "--is-ancestor", corrected_commit, head
        )
        collector._binding_at_commit(  # noqa: SLF001
            corrected_analyzer,
            commit=corrected_commit,
            label="corrected calibration analyzer",
        )
        collector._binding_at_commit(  # noqa: SLF001
            corrected_test,
            commit=corrected_commit,
            label="corrected calibration analyzer test",
        )
    except (pilot.PilotContractError, subprocess.CalledProcessError) as exc:
        raise BoundedBranchPlanError(
            "calibration analyzer-recovery corrected commit changed"
        ) from exc
    if (
        len(original_analyzers) != 1
        or correction.get("failed_source_commit") != authority.get("source_commit")
        or correction.get("failed_analyzer_binding") != original_analyzers[0]
        or corrected_analyzer != receipt.get("analyzer_binding")
        or corrected_analyzer == original_analyzers[0]
        or corrected_test.get("path") != str(
            REPO_ROOT
            / "lewm/tests/test_analyze_go2_world_model_counterfactual_calibration_v1.py"
        )
    ):
        raise BoundedBranchPlanError(
            "calibration analyzer-recovery source identity changed"
        )


def _load_recovery_authority_at_frozen_source(
    authority_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate the consumed authority at its frozen source commit.

    The normal execution loader intentionally requires every source to remain
    byte-identical in the working tree.  The recovery route instead verifies
    every original source at the authority commit, then lets the independent
    recovery review bind the one corrected analyzer separately.
    """

    collector = calibration_supervisor.collector
    try:
        raw_authority, actual_authority = pilot.read_bound_json(
            Path(str(authority_binding["path"])),
            expected_sha256=str(authority_binding["file_sha256"]),
            expected_byte_count=int(authority_binding["byte_count"]),
            label="frozen calibration authority",
        )
        if not isinstance(raw_authority, Mapping):
            raise pilot.PilotContractError("calibration authority must be an object")
        raw_plan_binding = pilot._validate_binding_shape(  # noqa: SLF001
            raw_authority.get("plan_binding"), label="authority plan binding"
        )
        raw_plan, actual_plan = pilot.read_bound_json(
            Path(str(raw_plan_binding["path"])),
            expected_sha256=str(raw_plan_binding["file_sha256"]),
            expected_byte_count=int(raw_plan_binding["byte_count"]),
            label="frozen calibration plan",
        )
        plan = pilot.validate_plan(raw_plan)
        authority = collector._validate_authority_for_plan(  # noqa: SLF001
            raw_authority,
            plan=plan,
            plan_binding=actual_plan,
        )
        review_binding = authority["review_binding"]
        raw_review, actual_review = pilot.read_bound_json(
            Path(str(review_binding["path"])),
            expected_sha256=str(review_binding["file_sha256"]),
            expected_byte_count=int(review_binding["byte_count"]),
            label="frozen calibration source review",
        )
        if actual_review != review_binding:
            raise pilot.PilotContractError("source review binding changed")
        pilot.validate_source_review(raw_review, authority=authority)

        head = str(collector._git_output("rev-parse", "HEAD"))  # noqa: SLF001
        source_commit = str(authority["source_commit"])
        collector._git_output(  # noqa: SLF001
            "merge-base", "--is-ancestor", source_commit, head
        )
        for binding, label in (
            (actual_plan, "frozen calibration plan"),
            (actual_authority, "frozen calibration authority"),
            (actual_review, "frozen calibration source review"),
        ):
            collector._binding_at_commit(  # noqa: SLF001
                binding, commit=head, label=label
            )
        for source in authority["source_bindings"]:
            collector._binding_at_commit(  # noqa: SLF001
                source["binding"],
                commit=source_commit,
                label=f"frozen authority source {source['name']}",
            )
        pilot.require_plan_bindings(plan)
    except (OSError, pilot.PilotContractError, subprocess.CalledProcessError) as exc:
        raise BoundedBranchPlanError(str(exc)) from exc
    return authority, actual_authority, plan, actual_plan


def _validate_calibration_gate(
    receipt: Mapping[str, Any],
    *,
    receipt_binding: Mapping[str, Any],
    terminal: Mapping[str, Any],
    terminal_binding: Mapping[str, Any],
    terminal_review: Mapping[str, Any],
    terminal_review_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Require a reviewed successful sizing run, not merely an analyzer flag."""

    try:
        normalized_receipt = calibration.validate_calibration_receipt_v1(
            receipt, verify_external_bindings=True
        )
    except calibration.CalibrationAnalysisError as exc:
        raise BoundedBranchPlanError(str(exc)) from exc
    profile = _calibration_gate_profile(normalized_receipt)
    if profile["receipt_schema"] != calibration.TEXTURED_V03_CALIBRATION_RECEIPT_SCHEMA:
        raise BoundedBranchPlanError(
            "bounded textured-v03 pilot requires the exact textured-v03 V3 calibration"
        )
    receipt_bound = _binding(receipt_binding, label="calibration receipt")
    terminal_bound = _binding(terminal_binding, label="calibration terminal")
    review_bound = _binding(terminal_review_binding, label="calibration terminal review")
    if normalized_receipt.get("decision") != "FREEZE_PILOT_CONTRACT":
        raise BoundedBranchPlanError("calibration did not freeze the pilot contract")
    required_terminal = {
        "schema",
        "status",
        "citable_as_scientific_evidence",
        "authorizes_retry_or_resume",
        "scientific_verdict_emitted",
        "root_creation_consumes_attempt",
        "reservation_records_consumed_attempt",
        "authority_binding",
        "plan_binding",
        "predecessor_failure_binding",
        "source_commit",
        "attempt_root",
        "wall_elapsed_seconds",
        "wall_ceiling_seconds",
        "phase_receipts",
        "physics_result_binding",
        "receipt_check_binding",
        "calibration_receipt_binding",
        "calibration_decision",
        "gpu_memory_measurement",
        "failure",
        "terminal_reviewer",
        "supervisor_nonce",
    }
    required_terminal.add("visual_domain_parity_prerequisites")
    if not isinstance(terminal, Mapping) or set(terminal) != required_terminal:
        raise BoundedBranchPlanError("calibration terminal fields changed")
    gpu = terminal.get("gpu_memory_measurement")
    required_gpu = {
        "scope",
        "attribution_limitation",
        "vendor_id",
        "device_id",
        "used_counter_path",
        "total_counter_path",
        "sample_interval_seconds",
        "sample_count",
        "read_errors",
        "baseline_used_bytes",
        "peak_used_bytes",
        "peak_delta_above_baseline_bytes",
        "device_total_bytes",
    }
    standard_terminal = terminal.get("status") == "COMPLETE_PENDING_TERMINAL_REVIEW"
    recovery_terminal = (
        terminal.get("status") == "CONSUMED_TERMINAL_FAILURE"
        and isinstance(terminal_review, Mapping)
        and terminal_review.get("schema") == ANALYSIS_RECOVERY_REVIEW_SCHEMA
        and terminal_review.get("status") == ANALYSIS_RECOVERY_REVIEW_STATUS
    )
    common_terminal_failed = (
        terminal.get("schema") != profile["terminal_schema"]
        or terminal.get("citable_as_scientific_evidence") is not False
        or terminal.get("authorizes_retry_or_resume") is not False
        or terminal.get("scientific_verdict_emitted") is not False
        or terminal.get("root_creation_consumes_attempt") is not True
        or terminal.get("reservation_records_consumed_attempt") is not True
        or terminal.get("physics_result_binding")
        != normalized_receipt.get("calibration_collection_receipt")
        or not isinstance(gpu, Mapping)
        or set(gpu) != required_gpu
        or gpu.get("scope") != "selected_device_global_vram_not_process_attributed"
        or gpu.get("read_errors") != 0
    )
    standard_terminal_failed = (
        not standard_terminal
        or terminal.get("calibration_receipt_binding") != receipt_bound
        or terminal.get("calibration_decision") != "FREEZE_PILOT_CONTRACT"
        or terminal.get("failure") is not None
    )
    recovery_terminal_failed = (
        not recovery_terminal
        or terminal_bound.get("file_sha256")
        != ANALYSIS_RECOVERY_FAILED_TERMINAL_SHA256
        or terminal_bound.get("byte_count")
        != ANALYSIS_RECOVERY_FAILED_TERMINAL_BYTE_COUNT
        or terminal.get("calibration_receipt_binding") is not None
        or terminal.get("calibration_decision") is not None
        or terminal.get("failure") != ANALYSIS_RECOVERY_TERMINAL_FAILURE
        or Path(str(receipt_bound["path"])).is_relative_to(
            Path(str(terminal.get("attempt_root")))
        )
    )
    if common_terminal_failed or (
        standard_terminal_failed and recovery_terminal_failed
    ):
        raise BoundedBranchPlanError("calibration terminal did not pass exactly")
    authority_bound = _binding(
        terminal["authority_binding"], label="calibration execution authority"
    )
    plan_bound = _binding(terminal["plan_binding"], label="calibration plan")
    predecessor_bound = _binding(
        terminal["predecessor_failure_binding"],
        label="calibration predecessor failure",
    )
    physics_bound = _binding(
        terminal["physics_result_binding"], label="calibration physics result"
    )
    check_bound = _binding(
        terminal["receipt_check_binding"], label="calibration receipt check"
    )
    if recovery_terminal:
        (
            authority_document,
            actual_authority,
            plan_document,
            actual_plan,
        ) = _load_recovery_authority_at_frozen_source(authority_bound)
    else:
        try:
            (
                authority_document,
                actual_authority,
                plan_document,
                actual_plan,
            ) = calibration_supervisor.load_and_validate_authority(
                Path(str(authority_bound["path"])),
                expected_authority_sha256=str(authority_bound["file_sha256"]),
                expected_authority_byte_count=int(authority_bound["byte_count"]),
            )
        except calibration_supervisor.CalibrationSupervisionError as exc:
            raise BoundedBranchPlanError(
                f"calibration {profile['name']} authority/successor boundary did not validate: "
                f"{exc}"
            ) from exc
    physics_document, actual_physics = pilot.read_bound_json(
        Path(str(physics_bound["path"])),
        expected_sha256=str(physics_bound["file_sha256"]),
        expected_byte_count=int(physics_bound["byte_count"]),
        label="calibration physics result",
    )
    check_document, actual_check = pilot.read_bound_json(
        Path(str(check_bound["path"])),
        expected_sha256=str(check_bound["file_sha256"]),
        expected_byte_count=int(check_bound["byte_count"]),
        label="calibration receipt check",
    )
    normalized_plan = pilot.validate_plan(plan_document)
    calibration_plan_parity = {
        "result_binding": normalized_plan.get(
            "visual_domain_parity_result_binding"
        ),
        "terminal_binding": normalized_plan.get(
            "visual_domain_parity_terminal_binding"
        ),
        "review_binding": normalized_plan.get(
            "visual_domain_parity_review_binding"
        ),
    }
    if (
        normalized_receipt.get("visual_domain_parity_prerequisites")
        != calibration_plan_parity
        or terminal.get("visual_domain_parity_prerequisites")
        != calibration_plan_parity
    ):
        raise BoundedBranchPlanError(
            "calibration receipt parity prerequisites differ from its bound plan"
        )
    expected_check_fields = {
        "schema",
        "status",
        "phase",
        "authority_granted",
        "scientific_claim_granted",
        "runtime_payloads_opened",
        "rgb_bytes_opened",
        "checkpoints_opened",
        "manifest_binding",
        "attempt_id",
        "purpose",
        "counts",
        "roles",
        "can_freeze_pilot_contract",
    }
    phases = terminal.get("phase_receipts")
    _require_calibration_profile_links(
        profile,
        terminal_schema=terminal.get("schema"),
        authority_schema=authority_document.get("schema"),
        authority_status=authority_document.get("status"),
        plan_purpose=normalized_plan.get("purpose"),
        physics_purpose=(
            physics_document.get("purpose")
            if isinstance(physics_document, Mapping)
            else None
        ),
        check_purpose=(
            check_document.get("purpose")
            if isinstance(check_document, Mapping)
            else None
        ),
    )
    expected_phase_count = 3 if recovery_terminal else 4
    if (
        actual_authority != authority_bound
        or not isinstance(authority_document, Mapping)
        or authority_document.get("schema") != profile["authority_schema"]
        or authority_document.get("status") != profile["authority_status"]
        or authority_document.get("scientific_claim_authorized") is not False
        or authority_document.get("plan_binding") != plan_bound
        or authority_document.get("predecessor_failure_binding") != predecessor_bound
        or authority_document.get("source_commit") != terminal.get("source_commit")
        or actual_plan != plan_bound
        or normalized_plan.get("purpose") != profile["purpose"]
        or normalized_plan.get("output_root") != terminal.get("attempt_root")
        or actual_physics != physics_bound
        or not isinstance(physics_document, Mapping)
        or physics_document.get("schema") != pilot.PHYSICS_RESULT_SCHEMA
        or physics_document.get("status") != "PHYSICS_COMPLETE"
        or physics_document.get("purpose") != profile["purpose"]
        or physics_document.get("plan_binding") != plan_bound
        or physics_document.get("authority_binding") != authority_bound
        or physics_document.get("caps") != authority_document.get("caps")
        or physics_document.get("failure") is not None
        or actual_check != check_bound
        or not isinstance(check_document, Mapping)
        or set(check_document) != expected_check_fields
        or check_document.get("schema") != receipt_checker.REPORT_SCHEMA
        or check_document.get("status") != "PASS"
        or check_document.get("phase") != "physics_collection"
        or check_document.get("purpose") != profile["purpose"]
        or check_document.get("manifest_binding") != physics_bound
        or check_document.get("runtime_payloads_opened") is not False
        or check_document.get("rgb_bytes_opened") is not False
        or check_document.get("checkpoints_opened") is not False
        or not isinstance(phases, list)
        or len(phases) != expected_phase_count
        or not isinstance(phases[0], Mapping)
        or phases[0].get("phase") != "graphics_preflight"
        or phases[0].get("status") != "PASS"
        or any(
            not isinstance(phase, Mapping)
            or type(phase.get("exit_code")) is not int
            or phase["exit_code"] != 0
            for phase in phases[1:]
        )
    ):
        raise BoundedBranchPlanError(
            "calibration terminal evidence links did not pass exactly"
        )
    for field in (
        "sample_count",
        "baseline_used_bytes",
        "peak_used_bytes",
        "peak_delta_above_baseline_bytes",
        "device_total_bytes",
    ):
        if type(gpu[field]) is not int or int(gpu[field]) < 0:
            raise BoundedBranchPlanError("calibration GPU measurement is invalid")
    if (
        gpu["sample_count"] <= 0
        or gpu["device_total_bytes"] <= 0
        or gpu["peak_used_bytes"] < gpu["baseline_used_bytes"]
        or gpu["peak_delta_above_baseline_bytes"]
        != gpu["peak_used_bytes"] - gpu["baseline_used_bytes"]
        or gpu["peak_used_bytes"] > gpu["device_total_bytes"]
    ):
        raise BoundedBranchPlanError("calibration GPU measurement is inconsistent")
    elapsed = terminal.get("wall_elapsed_seconds")
    ceiling = terminal.get("wall_ceiling_seconds")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0.0
        or isinstance(ceiling, bool)
        or not isinstance(ceiling, (int, float))
        or not math.isfinite(float(ceiling))
        or float(ceiling) <= 0.0
        or float(elapsed) > float(ceiling)
        or float(ceiling)
        != float(authority_document.get("caps", {}).get("wall_seconds", -1))
    ):
        raise BoundedBranchPlanError("calibration exceeded its wall ceiling")

    if recovery_terminal:
        support = normalized_receipt.get("candidate_branch_support_analysis")
        coverage = (
            support.get("calibrated_discrimination_query_coverage")
            if isinstance(support, Mapping)
            else None
        )
        overall = coverage.get("overall") if isinstance(coverage, Mapping) else None
        per_family = (
            coverage.get("per_family") if isinstance(coverage, Mapping) else None
        )
        technical = normalized_receipt.get("technical_integrity")
        repeatability = normalized_receipt.get("repeatability_analysis")
        visual = normalized_receipt.get("visual_validation")
        if (
            normalized_receipt.get("status") != "COMPLETE"
            or normalized_receipt.get("citable_as_scientific_evidence") is not False
            or normalized_receipt.get("train_eval_scenes_accessed") is not False
            or not isinstance(overall, Mapping)
            or overall.get("eligible_query_count") != 144
            or overall.get("total_query_count") != 144
            or overall.get("passed") is not True
            or not isinstance(per_family, Mapping)
            or set(per_family) != set(pilot.FAMILIES)
            or any(
                not isinstance(row, Mapping)
                or row.get("eligible_query_count") != 18
                or row.get("total_query_count") != 18
                or row.get("passed") is not True
                for row in per_family.values()
            )
            or not isinstance(technical, Mapping)
            or technical.get("receipt_checker_passed") is not True
            or technical.get("sentinel_command_endpoint_and_rgb_exact") is not True
            or technical.get("hard_invalid_frames") != 0
            or not isinstance(repeatability, Mapping)
            or repeatability.get("repeat_controls") != 16
            or repeatability.get("executed_command_tapes_exact") is not True
            or repeatability.get("physical_trajectories_exact") is not True
            or repeatability.get("stored_rgb_exact") is not True
            or not isinstance(visual, Mapping)
            or visual.get("visual_domain_fidelity_claimed") is not False
            or len(normalized_receipt["calibration_contract"]["excluded_scene_ids"])
            != 8
        ):
            raise BoundedBranchPlanError(
                "posthoc calibration receipt did not preserve the exact support gate"
            )
    if recovery_terminal:
        _validate_analysis_recovery_review(
            terminal_review,
            receipt=normalized_receipt,
            receipt_binding=receipt_bound,
            terminal_binding=terminal_bound,
            authority=authority_document,
        )
    else:
        required_review = {
            "schema",
            "status",
            "authority_granted_by_this_document",
            "scientific_claim_granted",
            "terminal_binding",
            "calibration_receipt_binding",
            "decision",
            "reviewer",
            "reviewed_at",
            "checks",
            "remaining_findings",
        }
        if (
            not isinstance(terminal_review, Mapping)
            or set(terminal_review) != required_review
        ):
            raise BoundedBranchPlanError("calibration terminal review fields changed")
        reviewer = terminal_review.get("reviewer")
        checks = terminal_review.get("checks")
        if (
            terminal_review.get("schema") != TERMINAL_REVIEW_SCHEMA
            or terminal_review.get("status") != "PASS_FREEZE_PILOT_CONTRACT"
            or terminal_review.get("authority_granted_by_this_document") is not False
            or terminal_review.get("scientific_claim_granted") is not False
            or terminal_review.get("terminal_binding") != terminal_bound
            or terminal_review.get("calibration_receipt_binding") != receipt_bound
            or terminal_review.get("decision") != "FREEZE_PILOT_CONTRACT"
            or terminal_review.get("remaining_findings") != []
            or not isinstance(reviewer, Mapping)
            or set(reviewer) != {"identity", "independence_basis"}
            or any(
                not isinstance(reviewer[key], str) or not reviewer[key].strip()
                for key in reviewer
            )
            or not isinstance(terminal_review.get("reviewed_at"), str)
            or not terminal_review["reviewed_at"].strip()
            or not isinstance(checks, Mapping)
            or set(checks) != {
                "terminal_complete",
                "receipt_checker_passed",
                "calibration_decision_passed",
                "gpu_sampler_passed",
                "wall_ceiling_passed",
                "no_retry_or_resume",
            }
            or any(value is not True for value in checks.values())
        ):
            raise BoundedBranchPlanError(
                "calibration terminal review did not pass exactly"
            )
    return {
        "calibration_receipt_binding": receipt_bound,
        "calibration_terminal_binding": terminal_bound,
        "calibration_terminal_review_binding": review_bound,
        "visual_domain_parity_result_binding": normalized_plan[
            "visual_domain_parity_result_binding"
        ],
        "visual_domain_parity_terminal_binding": normalized_plan[
            "visual_domain_parity_terminal_binding"
        ],
        "visual_domain_parity_review_binding": normalized_plan[
            "visual_domain_parity_review_binding"
        ],
        "excluded_scene_ids": list(
            normalized_receipt["calibration_contract"]["excluded_scene_ids"]
        ),
        "calibration_wall_seconds": float(elapsed),
        "calibration_stored_rgb_bytes": int(
            normalized_receipt["resource_measurements"]["stored_rgb_png"]["total_bytes"]
        ),
        "calibration_gpu_baseline_used_bytes": int(gpu["baseline_used_bytes"]),
        "calibration_gpu_peak_used_bytes": int(gpu["peak_used_bytes"]),
        "calibration_gpu_peak_delta_bytes": int(
            gpu["peak_delta_above_baseline_bytes"]
        ),
        "selected_device_total_vram_bytes": int(gpu["device_total_bytes"]),
    }


def _finite_number(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise BoundedBranchPlanError(f"{label} must be finite")
    return float(value)


def _closed_interval(
    value: object, *, label: str, lower_bound: float | None = None
) -> tuple[float, float, float]:
    if not isinstance(value, Mapping) or set(value) != {
        "point",
        "lower_95",
        "upper_95",
    }:
        raise BoundedBranchPlanError(f"{label} interval fields changed")
    point = _finite_number(value["point"], label=f"{label} point")
    lower = _finite_number(value["lower_95"], label=f"{label} lower_95")
    upper = _finite_number(value["upper_95"], label=f"{label} upper_95")
    if lower > point or point > upper or (
        lower_bound is not None and lower < lower_bound
    ):
        raise BoundedBranchPlanError(f"{label} interval is inconsistent")
    return point, lower, upper


def _validate_visual_domain_parity_gate_v1(
    result: object,
    *,
    result_binding: Mapping[str, Any],
    review: object,
    review_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute exact 8-scene/32-pose parity before accepting review."""

    result_bound = _binding(result_binding, label="visual-domain parity result")
    review_bound = _binding(
        review_binding, label="visual-domain parity independent review"
    )
    try:
        reopened_result, actual_result = pilot.read_bound_json(
            Path(str(result_bound["path"])),
            expected_sha256=str(result_bound["file_sha256"]),
            expected_byte_count=int(result_bound["byte_count"]),
            label="visual-domain parity result",
        )
        reopened_review, actual_review = pilot.read_bound_json(
            Path(str(review_bound["path"])),
            expected_sha256=str(review_bound["file_sha256"]),
            expected_byte_count=int(review_bound["byte_count"]),
            label="visual-domain parity independent review",
        )
    except (OSError, pilot.PilotContractError) as exc:
        raise BoundedBranchPlanError(str(exc)) from exc
    if (
        actual_result != result_bound
        or actual_review != review_bound
        or reopened_result != result
        or reopened_review != review
    ):
        raise BoundedBranchPlanError(
            "visual-domain parity result/review document binding changed"
        )
    required_result = {
        "schema",
        "status",
        "authority_granted_by_this_document",
        "scientific_claim_granted_by_this_document",
        "development_only",
        "protected_material_opened",
        "comparison_contract",
        "thresholds",
        "measurements",
        "evidence_scene_ids",
        "source_rgb_reference_binding",
        "candidate_rgb_panel_binding",
        "source_producer_lineage",
        "candidate_producer_lineage",
        "candidate_collector_source_binding",
        "candidate_renderer_source_binding",
        "reference_renderer_source_binding",
        "reference_texture_source_binding",
        "evaluator_source_binding",
        "selected_texture_asset_bindings_by_scene",
    }
    if not isinstance(result, Mapping) or set(result) != required_result:
        raise BoundedBranchPlanError("visual-domain parity result fields changed")
    if (
        result.get("schema") != VISUAL_DOMAIN_PARITY_RESULT_SCHEMA
        or result.get("status") != VISUAL_DOMAIN_PARITY_RESULT_STATUS
        or result.get("authority_granted_by_this_document") is not False
        or result.get("scientific_claim_granted_by_this_document") is not False
        or result.get("development_only") is not True
        or result.get("protected_material_opened") is not False
        or result.get("comparison_contract")
        != VISUAL_DOMAIN_PARITY_COMPARISON_CONTRACT
        or result.get("thresholds") != VISUAL_DOMAIN_PARITY_THRESHOLDS
        or not isinstance(result.get("measurements"), Mapping)
        or not parity_evaluator._passes(result["measurements"])  # noqa: SLF001
    ):
        raise BoundedBranchPlanError(
            "visual-domain parity result did not pass the fixed exact contract"
        )
    evidence_scene_ids = result.get("evidence_scene_ids")
    if (
        not isinstance(evidence_scene_ids, list)
        or len(evidence_scene_ids) != len(pilot.FAMILIES)
        or evidence_scene_ids != sorted(set(evidence_scene_ids))
        or any(not isinstance(item, str) or not item for item in evidence_scene_ids)
    ):
        raise BoundedBranchPlanError(
            "visual-domain parity evidence scene inventory is invalid"
        )
    for scene_id in evidence_scene_ids:
        _reject_protected_path(Path(scene_id), label="visual parity evidence scene")

    source_panel_bound = _binding(
        result["source_rgb_reference_binding"],
        label="visual-domain historical source panel",
    )
    candidate_panel_bound = _binding(
        result["candidate_rgb_panel_binding"],
        label="visual-domain candidate panel",
    )
    candidate_collector = _binding(
        result["candidate_collector_source_binding"],
        label="visual-domain candidate collector source",
    )
    candidate_renderer = _binding(
        result["candidate_renderer_source_binding"],
        label="visual-domain candidate renderer source",
    )
    reference_renderer = _binding(
        result["reference_renderer_source_binding"],
        label="visual-domain reference renderer source",
    )
    reference_textures = _binding(
        result["reference_texture_source_binding"],
        label="visual-domain reference texture source",
    )
    evaluator_source = _binding(
        result["evaluator_source_binding"],
        label="visual-domain parity evaluator source",
    )
    exact_sources = {
        "candidate collector": (
            candidate_collector,
            REPO_ROOT / "scripts/collect_go2_world_model_counterfactual_pilot_v1.py",
        ),
        "candidate renderer": (
            candidate_renderer,
            REPO_ROOT / "scripts/render_replay_v03.py",
        ),
        "reference renderer": (
            reference_renderer,
            REPO_ROOT / "scripts/render_replay_v03.py",
        ),
        "reference textures": (
            reference_textures,
            REPO_ROOT / "lewm_genesis/lewm_genesis/textures.py",
        ),
        "parity evaluator": (
            evaluator_source,
            REPO_ROOT / VISUAL_DOMAIN_PARITY_EVALUATOR_RELATIVE,
        ),
    }
    for label, (binding, path) in exact_sources.items():
        if binding != pilot.file_binding(path.resolve()):
            raise BoundedBranchPlanError(
                f"visual-domain {label} source identity changed"
            )
    try:
        source_panel, actual_source_panel = pilot.read_bound_json(
            Path(str(source_panel_bound["path"])),
            expected_sha256=str(source_panel_bound["file_sha256"]),
            expected_byte_count=int(source_panel_bound["byte_count"]),
            label="visual-domain historical source panel",
        )
        candidate_panel, actual_candidate_panel = pilot.read_bound_json(
            Path(str(candidate_panel_bound["path"])),
            expected_sha256=str(candidate_panel_bound["file_sha256"]),
            expected_byte_count=int(candidate_panel_bound["byte_count"]),
            label="visual-domain candidate panel",
        )
        recomputed = parity_evaluator.evaluate_v1(
            source_panel=source_panel,
            source_panel_binding=actual_source_panel,
            candidate_panel=candidate_panel,
            candidate_panel_binding=actual_candidate_panel,
        )
    except (
        OSError,
        pilot.PilotContractError,
        parity_evaluator.VisualDomainParityError,
    ) as exc:
        raise BoundedBranchPlanError(
            f"visual-domain parity recomputation failed: {exc}"
        ) from exc
    if recomputed != dict(result):
        raise BoundedBranchPlanError(
            "visual-domain parity result is not the exact evaluator recomputation"
        )

    required_review = {
        "schema",
        "status",
        "authority_granted_by_this_document",
        "scientific_claim_granted_by_this_document",
        "result_binding",
        "terminal_binding",
        "reviewer",
        "reviewed_at",
        "checks",
        "remaining_findings",
    }
    required_checks = set(pilot.TEXTURED_V03_PARITY_REVIEW_CHECKS)
    checks = review.get("checks") if isinstance(review, Mapping) else None
    reviewer = review.get("reviewer") if isinstance(review, Mapping) else None
    if (
        not isinstance(review, Mapping)
        or set(review) != required_review
        or review.get("schema") != VISUAL_DOMAIN_PARITY_REVIEW_SCHEMA
        or review.get("status") != VISUAL_DOMAIN_PARITY_REVIEW_STATUS
        or review.get("authority_granted_by_this_document") is not False
        or review.get("scientific_claim_granted_by_this_document") is not False
        or review.get("result_binding") != result_bound
        or not isinstance(review.get("terminal_binding"), Mapping)
        or not isinstance(reviewer, Mapping)
        or set(reviewer) != {"identity", "independence_basis"}
        or any(
            not isinstance(reviewer[key], str) or not reviewer[key].strip()
            for key in reviewer
        )
        or not isinstance(checks, Mapping)
        or set(checks) != required_checks
        or any(value is not True for value in checks.values())
        or review.get("remaining_findings") != []
        or not isinstance(review.get("reviewed_at"), str)
        or not review["reviewed_at"].strip()
    ):
        raise BoundedBranchPlanError(
            "visual-domain parity independent review did not pass exactly"
        )
    try:
        reviewed_at = datetime.fromisoformat(
            str(review["reviewed_at"]).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise BoundedBranchPlanError(
            "visual-domain parity reviewed_at is not ISO-8601"
        ) from exc
    if reviewed_at.tzinfo is None:
        raise BoundedBranchPlanError(
            "visual-domain parity reviewed_at must include a timezone"
        )
    terminal_bound = _binding(
        review["terminal_binding"], label="visual-domain parity terminal"
    )
    try:
        pilot.validate_textured_v03_parity_prerequisites(
            result_binding=result_bound,
            terminal_binding=terminal_bound,
            review_binding=review_bound,
        )
    except pilot.PilotContractError as exc:
        raise BoundedBranchPlanError(str(exc)) from exc
    return {
        "result_binding": result_bound,
        "terminal_binding": terminal_bound,
        "review_binding": review_bound,
        "source_rgb_reference_binding": source_panel_bound,
        "candidate_rgb_panel_binding": candidate_panel_bound,
        "source_producer_lineage": copy.deepcopy(result["source_producer_lineage"]),
        "candidate_producer_lineage": copy.deepcopy(
            result["candidate_producer_lineage"]
        ),
        "candidate_collector_source_binding": candidate_collector,
        "candidate_renderer_source_binding": candidate_renderer,
        "reference_renderer_source_binding": reference_renderer,
        "reference_texture_source_binding": reference_textures,
        "evaluator_source_binding": evaluator_source,
        "selected_texture_asset_bindings_by_scene": copy.deepcopy(
            result["selected_texture_asset_bindings_by_scene"]
        ),
        "evidence_scene_ids": list(evidence_scene_ids),
        "comparison_contract": copy.deepcopy(result["comparison_contract"]),
        "thresholds": copy.deepcopy(result["thresholds"]),
        "measurements": copy.deepcopy(result["measurements"]),
    }


def _validate_candidate_render_domain_contract_v1(
    render_contract: object,
) -> None:
    """Reject the current 640x480/downsample path before it can become a plan."""

    if not isinstance(render_contract, Mapping):
        raise BoundedBranchPlanError("candidate render contract is malformed")
    if dict(render_contract) != BOUNDED_RENDER_CONTRACT_V1:
        raise BoundedBranchPlanError(
            "candidate renderer is not the exact versioned native-224 "
            "textured_v03 contract; the 640x480 horizontal-FOV "
            "conversion/downsample path is ineligible"
        )
    reference_renderer = _binding(
        {
            "path": str((REPO_ROOT / "scripts/render_replay_v03.py").resolve()),
            "file_sha256": BOUNDED_RENDER_CONTRACT_V1[
                "historical_renderer_sha256"
            ],
            "byte_count": BOUNDED_RENDER_CONTRACT_V1[
                "historical_renderer_byte_count"
            ],
        },
        label="bounded textured_v03 historical renderer",
    )
    if (
        reference_renderer["file_sha256"]
        != BOUNDED_RENDER_CONTRACT_V1["historical_renderer_sha256"]
    ):
        raise AssertionError("bounded renderer binding changed after validation")


def _textured_v03_texture_asset_bindings_v1() -> list[dict[str, Any]]:
    """Bind the complete, ordered historical textured-v03 asset closure."""

    bindings: list[dict[str, Any]] = []
    repo_root = REPO_ROOT.resolve()
    for relative in pilot.TEXTURED_V03_TEXTURE_RELATIVE_PATHS:
        path = REPO_ROOT / relative
        _reject_protected_path(path, label=f"textured_v03 texture asset {relative}")
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise BoundedBranchPlanError(
                f"textured_v03 texture asset is unavailable: {relative}"
            ) from exc
        try:
            resolved.relative_to(repo_root)
        except ValueError as exc:
            raise BoundedBranchPlanError(
                f"textured_v03 texture asset escaped the repository: {relative}"
            ) from exc
        cursor = repo_root
        for part in Path(relative).parts:
            cursor /= part
            if cursor.is_symlink():
                raise BoundedBranchPlanError(
                    f"textured_v03 texture asset contains a symlink: {relative}"
                )
        if not resolved.is_file() or resolved != path.resolve():
            raise BoundedBranchPlanError(
                f"textured_v03 texture asset is not a regular file: {relative}"
            )
        bindings.append(
            _binding(
                pilot.file_binding(resolved),
                label=f"textured_v03 texture asset {relative}",
            )
        )
    return bindings


def _validate_plan_texture_asset_bindings_v1(
    plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Rehash and prove the plan's ordered textured-v03 asset closure."""

    expected = _textured_v03_texture_asset_bindings_v1()
    if plan.get("texture_asset_bindings") != expected:
        raise BoundedBranchPlanError(
            "bounded plan textured_v03 texture asset closure changed"
        )
    return expected


def _validate_runtime_contract(value: object) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "runtime_bindings",
        "execution_contract",
    } or value.get("schema") != RUNTIME_CONTRACT_SCHEMA:
        raise BoundedBranchPlanError("runtime contract fields/schema changed")
    runtime = value["runtime_bindings"]
    execution = value["execution_contract"]
    if not isinstance(runtime, Mapping) or not isinstance(execution, Mapping):
        raise BoundedBranchPlanError("runtime contract is malformed")
    return (
        {str(name): _binding(binding, label=f"runtime {name}") for name, binding in runtime.items()},
        copy.deepcopy(dict(execution)),
    )


def _validate_model_panel_freeze(
    progression_analysis: Mapping[str, Any],
    *,
    progression_analysis_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze the analyzer-bound 12 checkpoints before branch generation."""

    bound = _binding(
        progression_analysis_binding, label="progression analysis"
    )
    input_result = (
        progression_analysis.get("input_result")
        if isinstance(progression_analysis, Mapping)
        else None
    )
    if (
        not isinstance(progression_analysis, Mapping)
        or not isinstance(input_result, Mapping)
        or not isinstance(input_result.get("path"), str)
    ):
        raise BoundedBranchPlanError("progression analysis is malformed")
    _reject_protected_path(Path(input_result["path"]), label="progression result")
    arm = evaluation.MODEL_ARMS[0]
    seed = evaluation.TRAINING_SEEDS[0]
    checkpoint = (
        Path(str(input_result["path"])).parent
        / f"seed_{seed}"
        / f"{arm}_update_{evaluation.EXPECTED_TERMINAL_UPDATE:06d}.pt"
    )
    try:
        document, separation = evaluation.load_and_validate_progression_analysis_v1(
            Path(str(bound["path"])),
            expected_sha256=str(bound["file_sha256"]),
            expected_byte_count=int(bound["byte_count"]),
            selected_checkpoint=checkpoint,
            expected_arm=arm,
            expected_seed=seed,
            pilot_scene_ids=set(),
        )
    except evaluation.BoundedBranchEvaluationError as exc:
        raise BoundedBranchPlanError(str(exc)) from exc
    if document != dict(progression_analysis):
        raise BoundedBranchPlanError("progression analysis changed")
    return {
        "progression_analysis_binding": bound,
        "training_result_binding": separation["training_result_binding"],
        "progression_proxy_routing": separation["progression_proxy_routing"],
        "checkpoint_panel_bindings": separation["checkpoint_panel_bindings"],
        "model_observational_scene_ids": separation["observational_scene_ids"],
        "model_observational_scene_count": separation["observational_scene_count"],
        "predecessor_terminal_access_binding": separation[
            "predecessor_terminal_access_binding"
        ],
        "predecessor_index_bindings": separation["predecessor_index_bindings"],
        "predecessor_place_manifest_binding": separation[
            "predecessor_place_manifest_binding"
        ],
        "training_pack_metadata_bindings": separation[
            "training_pack_metadata_bindings"
        ],
        "training_pack_role_bindings": separation[
            "training_pack_role_bindings"
        ],
    }


def _validate_panel(value: object, *, excluded_scene_ids: set[str]) -> list[dict[str, Any]]:
    try:
        derived = panel_selector.derive_scene_panel_v1(
            excluded_scene_ids=excluded_scene_ids
        )
    except panel_selector.BoundedBranchScenePanelError as exc:
        raise BoundedBranchPlanError(str(exc)) from exc
    if value != derived:
        raise BoundedBranchPlanError(
            "pilot scene panel is not the deterministic ordinary-TRAIN derivation"
        )
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "selection_contract",
        "corpus_manifest_bindings",
        "inventory_unique_train_scenes",
        "eligible_counts_by_family",
        "excluded_scene_ids_sha256",
        "scenes",
    }:
        raise BoundedBranchPlanError("pilot scene panel fields changed")
    scenes = value.get("scenes")
    if value.get("schema") != PANEL_SCHEMA or not isinstance(scenes, list):
        raise BoundedBranchPlanError("pilot scene panel schema changed")
    if len(scenes) != EXPECTED_SCENES:
        raise BoundedBranchPlanError(f"pilot panel must contain {EXPECTED_SCENES} scenes")
    expected_order = [
        (role, family, scene_slot)
        for role in ROLE_NAMES
        for family in pilot.FAMILIES
        for scene_slot in range(SCENES_PER_FAMILY_PER_ROLE)
    ]
    normalized: list[dict[str, Any]] = []
    seen_scenes: set[str] = set()
    seen_states: set[str] = set()
    for expected, scene in zip(expected_order, scenes, strict=True):
        if not isinstance(scene, Mapping) or set(scene) != {
            "role",
            "family",
            "scene_slot",
            "scene_id",
            "inventory_manifest_sha256",
            "selection_rank",
            "role_allocation_rank",
            "scene_manifest_binding",
            "scene_genesis_binding",
            "selected_texture_asset_bindings",
            "states",
        }:
            raise BoundedBranchPlanError("pilot scene panel entry changed")
        role, family, slot = expected
        scene_id = scene.get("scene_id")
        if (
            scene.get("role") != role
            or scene.get("family") != family
            or scene.get("scene_slot") != slot
            or not isinstance(scene_id, str)
            or not scene_id
            or not isinstance(scene.get("inventory_manifest_sha256"), str)
            or re.fullmatch(
                r"[0-9a-f]{64}", str(scene.get("inventory_manifest_sha256"))
            )
            is None
            or scene_id in seen_scenes
            or scene_id in excluded_scene_ids
        ):
            raise BoundedBranchPlanError("pilot scene order, identity, or exclusion changed")
        manifest_value = scene["scene_manifest_binding"]
        genesis_value = scene["scene_genesis_binding"]
        if (
            not isinstance(manifest_value, Mapping)
            or not isinstance(manifest_value.get("path"), str)
            or not isinstance(genesis_value, Mapping)
            or not isinstance(genesis_value.get("path"), str)
        ):
            raise BoundedBranchPlanError("pilot scene binding pair is malformed")
        manifest_path = Path(str(manifest_value["path"]))
        genesis_path = Path(str(genesis_value["path"]))
        if (
            not manifest_path.is_absolute()
            or not genesis_path.is_absolute()
            or ".." in manifest_path.parts
            or ".." in genesis_path.parts
        ):
            raise BoundedBranchPlanError("pilot scene paths must be canonical absolute paths")
        scene_corpus_root = (REPO_ROOT / ".generated/scene_corpus").absolute()
        try:
            manifest_relative = manifest_path.relative_to(scene_corpus_root)
            genesis_relative = genesis_path.relative_to(scene_corpus_root)
        except ValueError as exc:
            raise BoundedBranchPlanError(
                "pilot scenes must come from the ordinary development scene corpus"
            ) from exc
        for relative, expected_name in (
            (manifest_relative, "manifest.json"),
            (genesis_relative, "genesis_scene.json"),
        ):
            if (
                len(relative.parts) != 5
                or relative.parts[1] != "train"
                or relative.parts[2] != family
                or relative.name != expected_name
            ):
                raise BoundedBranchPlanError(
                    "pilot scenes must use exact ordinary TRAIN family paths"
                )
        manifest = _binding(manifest_value, label=f"scene {scene_id} manifest")
        genesis = _binding(genesis_value, label=f"scene {scene_id} Genesis pack")
        texture_values = scene.get("selected_texture_asset_bindings")
        if not isinstance(texture_values, Mapping) or set(texture_values) != {
            "floor",
            "wall",
            "obstacle",
        }:
            raise BoundedBranchPlanError(
                "pilot scene selected texture-asset fields changed"
            )
        selected_texture_assets = {}
        for category in ("floor", "wall", "obstacle"):
            texture_binding = _binding(
                texture_values[category],
                label=f"scene {scene_id} {category} texture asset",
            )
            category_root = (REPO_ROOT / "assets/textures" / category).resolve()
            try:
                relative_texture = Path(
                    str(texture_binding["path"])
                ).resolve().relative_to(category_root)
            except ValueError as exc:
                raise BoundedBranchPlanError(
                    "pilot selected texture asset escaped its exact category"
                ) from exc
            if (
                len(relative_texture.parts) != 1
                or relative_texture.suffix.lower()
                not in {".jpg", ".jpeg", ".png"}
            ):
                raise BoundedBranchPlanError(
                    "pilot selected texture asset is not a category leaf image"
                )
            selected_texture_assets[category] = texture_binding
        if (
            Path(manifest["path"]).name != "manifest.json"
            or Path(genesis["path"]).name != "genesis_scene.json"
            or Path(manifest["path"]).parent != Path(genesis["path"]).parent
        ):
            raise BoundedBranchPlanError("pilot scene binding pair changed")
        canonical_target = _canonical_manifest_target(
            manifest, scene_id=scene_id, family=family
        )
        manifest_document, manifest_actual = pilot.read_bound_json(
            Path(str(manifest["path"])),
            expected_sha256=str(manifest["file_sha256"]),
            expected_byte_count=int(manifest["byte_count"]),
            label=f"pilot scene {scene_id} manifest",
        )
        if (
            manifest_actual != manifest
            or not isinstance(manifest_document, Mapping)
            or manifest_document.get("split") != "train"
            or manifest_document.get("manifest_sha256")
            != scene.get("inventory_manifest_sha256")
        ):
            raise BoundedBranchPlanError("pilot scene is not ordinary TRAIN metadata")
        states = scene.get("states")
        if not isinstance(states, list) or len(states) != STATES_PER_SCENE:
            raise BoundedBranchPlanError("each pilot scene must contain eight states")
        normalized_states = []
        for state_index, (state, history) in enumerate(zip(states, HISTORY_PANEL, strict=True)):
            if not isinstance(state, Mapping) or set(state) != {
                "state_id",
                "history_action_ids",
            }:
                raise BoundedBranchPlanError("pilot state fields changed")
            state_id = state.get("state_id")
            if (
                not isinstance(state_id, str)
                or not state_id
                or state_id in seen_states
                or state.get("history_action_ids") != list(history)
            ):
                raise BoundedBranchPlanError("pilot state identity/history/target changed")
            seen_states.add(state_id)
            normalized_states.append({
                "state_id": state_id,
                "history_action_ids": list(history),
                "target_xy_m": canonical_target,
            })
        seen_scenes.add(scene_id)
        normalized.append({
            "role": role,
            "family": family,
            "scene_id": scene_id,
            "scene_manifest_binding": manifest,
            "scene_genesis_binding": genesis,
            "selected_texture_asset_bindings": selected_texture_assets,
            "states": normalized_states,
        })
    return normalized


def _validate_scene_panel_receipt_v1(
    value: object,
    *,
    binding: Mapping[str, Any],
    excluded_scene_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reopen one exact selector receipt, rederive it, and freeze its evidence."""

    bound = _binding(binding, label="bounded pilot scene panel")
    try:
        reopened, actual = pilot.read_bound_json(
            Path(str(bound["path"])),
            expected_sha256=str(bound["file_sha256"]),
            expected_byte_count=int(bound["byte_count"]),
            label="bounded pilot scene panel",
        )
    except (OSError, pilot.PilotContractError) as exc:
        raise BoundedBranchPlanError(str(exc)) from exc
    if actual != bound or reopened != value:
        raise BoundedBranchPlanError(
            "bounded pilot scene panel document/binding changed"
        )
    normalized = _validate_panel(
        reopened,
        excluded_scene_ids=excluded_scene_ids,
    )
    assert isinstance(reopened, Mapping)
    freeze = {
        "scene_panel_binding": bound,
        "scene_panel_schema": reopened["schema"],
        "scene_selection_contract": copy.deepcopy(reopened["selection_contract"]),
        "scene_corpus_manifest_bindings": copy.deepcopy(
            reopened["corpus_manifest_bindings"]
        ),
        "scene_inventory_unique_train_scenes": int(
            reopened["inventory_unique_train_scenes"]
        ),
        "scene_eligible_counts_by_family": copy.deepcopy(
            reopened["eligible_counts_by_family"]
        ),
        "scene_excluded_scene_ids_sha256": reopened[
            "excluded_scene_ids_sha256"
        ],
        "scene_selection_rows": copy.deepcopy(reopened["scenes"]),
    }
    if set(freeze) != set(SCENE_PANEL_FREEZE_FIELDS):
        raise AssertionError("scene-panel freeze field declaration changed")
    return normalized, freeze


def _validate_plan_scene_panel_match_v1(
    plan: Mapping[str, Any],
    *,
    normalized_panel: Sequence[Mapping[str, Any]],
) -> None:
    """Prove every planned state is the exact state selected by the receipt."""

    expected_states: list[dict[str, Any]] = []
    for scene in normalized_panel:
        states = scene["states"]
        for state_index, state in enumerate(states):
            expected_states.append(
                {
                    "state_id": state["state_id"],
                    "role": scene["role"],
                    "family": scene["family"],
                    "scene_id": scene["scene_id"],
                    "scene_manifest_binding": scene["scene_manifest_binding"],
                    "scene_genesis_binding": scene["scene_genesis_binding"],
                    "scene_generation": None,
                    "group_index": len(expected_states),
                    "state_index_in_scene": state_index,
                    "history_action_ids": state["history_action_ids"],
                    "candidate_action_ids": list(range(pilot.ACTION_COUNT)),
                    "sentinel_duplicate_action_id": None,
                    "target_xy_m": state["target_xy_m"],
                }
            )
    if len(expected_states) != EXPECTED_STATES or plan.get("states") != expected_states:
        raise BoundedBranchPlanError(
            "bounded plan states do not exactly match the frozen scene-panel receipt"
        )


def build_plan_v1(
    *,
    attempt_id: str,
    output_root: Path,
    scene_panel: Mapping[str, Any],
    scene_panel_binding: Mapping[str, Any],
    visual_domain_parity_result: Mapping[str, Any],
    visual_domain_parity_result_binding: Mapping[str, Any],
    visual_domain_parity_review: Mapping[str, Any],
    visual_domain_parity_review_binding: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
    calibration_receipt: Mapping[str, Any],
    calibration_receipt_binding: Mapping[str, Any],
    calibration_terminal: Mapping[str, Any],
    calibration_terminal_binding: Mapping[str, Any],
    calibration_terminal_review: Mapping[str, Any],
    calibration_terminal_review_binding: Mapping[str, Any],
    progression_analysis: Mapping[str, Any],
    progression_analysis_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if _ATTEMPT.fullmatch(attempt_id) is None:
        raise BoundedBranchPlanError("attempt_id is invalid")
    selected_root = Path(output_root)
    development_root = (REPO_ROOT / ".generated/dev").resolve()
    if (
        not selected_root.is_absolute()
        or not selected_root.resolve().is_relative_to(development_root)
        or selected_root.exists()
        or selected_root.is_symlink()
    ):
        raise BoundedBranchPlanError("output_root must be fresh under .generated/dev")
    gate = _validate_calibration_gate(
        calibration_receipt,
        receipt_binding=calibration_receipt_binding,
        terminal=calibration_terminal,
        terminal_binding=calibration_terminal_binding,
        terminal_review=calibration_terminal_review,
        terminal_review_binding=calibration_terminal_review_binding,
    )
    model_panel = _validate_model_panel_freeze(
        progression_analysis,
        progression_analysis_binding=progression_analysis_binding,
    )
    visual_domain_parity_freeze = _validate_visual_domain_parity_gate_v1(
        visual_domain_parity_result,
        result_binding=visual_domain_parity_result_binding,
        review=visual_domain_parity_review,
        review_binding=visual_domain_parity_review_binding,
    )
    _require_calibration_parity_identity(gate, visual_domain_parity_freeze)
    panel, scene_panel_freeze = _validate_scene_panel_receipt_v1(
        scene_panel,
        binding=scene_panel_binding,
        excluded_scene_ids=(
            set(gate["excluded_scene_ids"])
            | set(model_panel["model_observational_scene_ids"])
            | set(visual_domain_parity_freeze["evidence_scene_ids"])
        ),
    )
    runtime, execution = _validate_runtime_contract(runtime_contract)
    states: list[dict[str, Any]] = []
    for scene in panel:
        for state_index, state in enumerate(scene["states"]):
            states.append({
                "state_id": state["state_id"],
                "role": scene["role"],
                "family": scene["family"],
                "scene_id": scene["scene_id"],
                "scene_manifest_binding": scene["scene_manifest_binding"],
                "scene_genesis_binding": scene["scene_genesis_binding"],
                "scene_generation": None,
                "group_index": len(states),
                "state_index_in_scene": state_index,
                "history_action_ids": state["history_action_ids"],
                "candidate_action_ids": list(range(pilot.ACTION_COUNT)),
                "sentinel_duplicate_action_id": None,
                "target_xy_m": state["target_xy_m"],
            })
    action_catalog = [
        {
            "action_id": action_id,
            "name": name,
            "requested_block": [list(command) for command in pilot.CANONICAL_ACTION_BLOCKS[action_id]],
        }
        for action_id, name in enumerate(pilot.CANONICAL_ACTIONS)
    ]
    render_contract = copy.deepcopy(BOUNDED_RENDER_CONTRACT_V1)
    _validate_candidate_render_domain_contract_v1(render_contract)
    texture_asset_bindings = _textured_v03_texture_asset_bindings_v1()
    plan = pilot.validate_plan({
        "schema": pilot.PLAN_SCHEMA,
        "attempt_id": attempt_id,
        "purpose": PURPOSE,
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "allows_refill": False,
        "allows_overwrite": False,
        "branch_mechanism": pilot.BRANCH_MECHANISM,
        "states_per_scene": STATES_PER_SCENE,
        "history_blocks": pilot.HISTORY_BLOCK_COUNT,
        "output_root": str(selected_root),
        "runtime_bindings": runtime,
        "execution_contract": execution,
        "render_contract": render_contract,
        "texture_asset_bindings": texture_asset_bindings,
        "visual_domain_parity_result_binding": visual_domain_parity_freeze[
            "result_binding"
        ],
        "visual_domain_parity_terminal_binding": visual_domain_parity_freeze[
            "terminal_binding"
        ],
        "visual_domain_parity_review_binding": visual_domain_parity_freeze[
            "review_binding"
        ],
        "action_catalog": action_catalog,
        "states": states,
        "expected_counts": pilot.expected_counts_from_states(states),
    })
    _validate_plan_texture_asset_bindings_v1(plan)
    if (
        plan["expected_counts"]["scenes"] != EXPECTED_SCENES
        or plan["expected_counts"]["states"] != EXPECTED_STATES
        or plan["expected_counts"]["candidate_branches"] != EXPECTED_BRANCHES
        or plan["expected_counts"]["sentinel_branches"] != 0
        or plan["expected_counts"]["target_frames"] + plan["expected_counts"]["context_frames"] != EXPECTED_STORED_FRAMES
        or plan["expected_counts"]["roles"] != {"eval": 128, "train": 128}
    ):
        raise BoundedBranchPlanError("bounded pilot count contract changed")
    _validate_plan_scene_panel_match_v1(plan, normalized_panel=panel)
    return plan, {
        **gate,
        **model_panel,
        **scene_panel_freeze,
        "visual_domain_parity_freeze": visual_domain_parity_freeze,
    }


def _load(path: Path, digest: str, byte_count: int, *, label: str):
    _reject_protected_path(path, label=label)
    try:
        return pilot.read_bound_json(
            path,
            expected_sha256=digest,
            expected_byte_count=byte_count,
            label=label,
        )
    except pilot.PilotContractError as exc:
        raise BoundedBranchPlanError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    for name in (
        "scene-panel",
        "visual-domain-parity-result",
        "visual-domain-parity-review",
        "runtime-contract",
        "calibration-receipt",
        "calibration-terminal",
        "calibration-terminal-review",
        "progression-analysis",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
        parser.add_argument(f"--expected-{name}-sha256", required=True)
        parser.add_argument(f"--expected-{name}-byte-count", required=True, type=int)
    parser.add_argument("--plan-output", required=True, type=Path)
    parser.add_argument("--gate-output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    loaded: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for key, label in (
        ("scene_panel", "bounded pilot scene panel"),
        (
            "visual_domain_parity_result",
            "bounded pilot visual-domain parity result",
        ),
        (
            "visual_domain_parity_review",
            "bounded pilot visual-domain parity independent review",
        ),
        ("runtime_contract", "counterfactual runtime contract"),
        ("calibration_receipt", "calibration receipt"),
        ("calibration_terminal", "calibration terminal"),
        ("calibration_terminal_review", "calibration terminal review"),
        ("progression_analysis", "progression analysis"),
    ):
        loaded[key] = _load(
            getattr(args, key),
            getattr(args, f"expected_{key}_sha256"),
            getattr(args, f"expected_{key}_byte_count"),
            label=label,
        )
    plan, gate = build_plan_v1(
        attempt_id=args.attempt_id,
        output_root=args.output_root,
        scene_panel=loaded["scene_panel"][0],
        scene_panel_binding=loaded["scene_panel"][1],
        visual_domain_parity_result=loaded["visual_domain_parity_result"][0],
        visual_domain_parity_result_binding=loaded[
            "visual_domain_parity_result"
        ][1],
        visual_domain_parity_review=loaded["visual_domain_parity_review"][0],
        visual_domain_parity_review_binding=loaded[
            "visual_domain_parity_review"
        ][1],
        runtime_contract=loaded["runtime_contract"][0],
        calibration_receipt=loaded["calibration_receipt"][0],
        calibration_receipt_binding=loaded["calibration_receipt"][1],
        calibration_terminal=loaded["calibration_terminal"][0],
        calibration_terminal_binding=loaded["calibration_terminal"][1],
        calibration_terminal_review=loaded["calibration_terminal_review"][0],
        calibration_terminal_review_binding=loaded["calibration_terminal_review"][1],
        progression_analysis=loaded["progression_analysis"][0],
        progression_analysis_binding=loaded["progression_analysis"][1],
    )
    gate_document = {
        "schema": "lewm_go2_world_model_bounded_branch_calibration_gate_v1",
        "status": "PASS",
        "authority_granted_by_this_document": False,
        **gate,
    }
    gate_binding = pilot.write_json_exclusive(args.gate_output, gate_document)
    plan_binding = pilot.write_json_exclusive(args.plan_output, plan)
    print(json.dumps({"plan": plan_binding, "calibration_gate": gate_binding}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BoundedBranchPlanError",
    "BOUNDED_RENDER_CONTRACT_V1",
    "EXPECTED_BRANCHES",
    "EXPECTED_SCENES",
    "EXPECTED_STATES",
    "EXPECTED_STORED_FRAMES",
    "HISTORY_PANEL",
    "MODEL_PANEL_FREEZE_FIELDS",
    "PANEL_SCHEMA",
    "SCENE_PANEL_FREEZE_FIELDS",
    "STATES_PER_SCENE",
    "TERMINAL_REVIEW_SCHEMA",
    "VISUAL_DOMAIN_PARITY_COMPARISON_CONTRACT",
    "VISUAL_DOMAIN_PARITY_EVALUATOR_RELATIVE",
    "VISUAL_DOMAIN_PARITY_RESULT_SCHEMA",
    "VISUAL_DOMAIN_PARITY_RESULT_STATUS",
    "VISUAL_DOMAIN_PARITY_REVIEW_SCHEMA",
    "VISUAL_DOMAIN_PARITY_REVIEW_STATUS",
    "VISUAL_DOMAIN_PARITY_THRESHOLDS",
    "build_plan_v1",
]
