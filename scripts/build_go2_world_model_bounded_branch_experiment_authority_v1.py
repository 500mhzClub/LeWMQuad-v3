#!/usr/bin/env python3
"""Build reviewed metadata for one post-calibration bounded WM-A pilot.

The review template is intentionally non-passing.  This helper cannot review
its own source, turn a failed calibration into a pass, or authorize execution
without explicit authorizer metadata and an exact committed plan/gate/review.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as pilot  # noqa: E402
from scripts import collect_go2_world_model_counterfactual_pilot_v1 as runtime_kernel  # noqa: E402
from scripts import evaluate_go2_world_model_bounded_branch_experiment_v1 as evaluation  # noqa: E402
from scripts import build_go2_world_model_bounded_branch_experiment_plan_v1 as plan_builder  # noqa: E402


PURPOSE = "bounded_wm_a_pilot"
GATE_SCHEMA = "lewm_go2_world_model_bounded_branch_calibration_gate_v1"
AUTHORITY_SCHEMA = (
    "lewm_go2_world_model_bounded_branch_experiment_execution_authority_v1"
)
AUTHORITY_STATUS = "AUTHORIZED_ONE_EXACT_2304_BRANCH_BOUNDED_WM_A_PILOT"
MAX_WALL_SECONDS = 28_800.0
MAX_STORED_RGB_BYTES = 2 * 1024**3
WALL_PROJECTION_MULTIPLIER = 20.0
RGB_PROJECTION_MULTIPLIER = 20.0
CALIBRATION_CONCURRENT_LANES = 2 * pilot.CALIBRATION_LANES_PER_STATE
BOUNDED_CONCURRENT_LANES = (
    runtime_kernel.BOUNDED_STATES_PER_SCENE_BATCH * pilot.ACTION_COUNT
)
VRAM_SAFETY_MARGIN_NUMERATOR = 5
VRAM_SAFETY_MARGIN_DENOMINATOR = 4
MINIMUM_WALL_SECONDS = 3_600.0
MINIMUM_STORED_RGB_BYTES = 512 * 1024**2
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")

NEW_SOURCE_PATHS = {
    "bounded_authority_builder": (
        "scripts/build_go2_world_model_bounded_branch_experiment_authority_v1.py"
    ),
    "bounded_plan_builder": (
        "scripts/build_go2_world_model_bounded_branch_experiment_plan_v1.py"
    ),
    "bounded_scene_panel_builder": (
        "scripts/build_go2_world_model_bounded_branch_scene_panel_v1.py"
    ),
    "bounded_plan_canonical_target_helper": (
        "scripts/build_go2_world_model_counterfactual_calibration_plan_v1.py"
    ),
    "bounded_runtime_collector": (
        "scripts/collect_go2_world_model_bounded_branch_experiment_authorized_v1.py"
    ),
    "bounded_external_supervisor": (
        "scripts/run_go2_world_model_bounded_branch_experiment_authorized_v1.py"
    ),
    "bounded_calibration_supervisor_helpers": (
        "scripts/run_go2_world_model_counterfactual_calibration_authorized_v1.py"
    ),
    "bounded_evaluator": (
        "scripts/evaluate_go2_world_model_bounded_branch_experiment_v1.py"
    ),
    "bounded_evaluation_panel_runner": (
        "scripts/run_go2_world_model_bounded_branch_evaluation_panel_v1.py"
    ),
    "bounded_evaluator_base": (
        "scripts/evaluate_go2_world_model_counterfactual_action_regret_v1.py"
    ),
    "bounded_evaluator_probe": "scripts/dev_probe_counterfactual_action_fidelity.py",
    "bounded_evaluator_model": (
        "lewm/models/rgb_recurrent_patch_memory_temporal_jepa_v1.py"
    ),
    "bounded_evaluator_temporal_api": (
        "scripts/evaluate_go2_rgb_recurrent_patch_memory_temporal_jepa_v1.py"
    ),
    "bounded_evaluator_mask_contract": (
        "lewm/benchmarks/go2_rgb_recurrent_patch_memory_temporal_jepa_v1.py"
    ),
    "bounded_evaluator_h6_contract": (
        "lewm/datasets/go2_explicit_plan_discounted_successor_state_v27.py"
    ),
    "bounded_evaluator_snapshot_contract": "scripts/dev_train_temporal_jepa_scaled.py",
    "bounded_visual_domain_reference_renderer": "scripts/render_replay_v03.py",
    "bounded_visual_domain_parity_evaluator": (
        plan_builder.VISUAL_DOMAIN_PARITY_EVALUATOR_RELATIVE
    ),
    "bounded_visual_domain_task_relevance_evaluator": (
        "scripts/evaluate_go2_world_model_visual_domain_parity_task_relevance_v1.py"
    ),
    "bounded_visual_domain_parity_supervisor": (
        "scripts/run_go2_world_model_visual_domain_parity_authorized_v1.py"
    ),
    "bounded_progression_runner": "scripts/dev_train_go2_world_model_progression_v1.py",
    "bounded_progression_analyzer": (
        "scripts/analyze_go2_world_model_progression_v1.py"
    ),
    "bounded_progression_model": "lewm/models/go2_world_model_progression_v1.py",
    "bounded_progression_historical_executor": (
        "scripts/execute_go2_world_model_existing_pool_three_arm_v1.py"
    ),
    "bounded_contract_test": (
        "lewm/tests/test_go2_world_model_bounded_branch_experiment_v1.py"
    ),
    "bounded_lineage_contract_test": (
        "lewm/tests/test_go2_world_model_bounded_branch_lineage_v1.py"
    ),
    "bounded_runtime_boundary_test": (
        "lewm/tests/test_go2_world_model_bounded_branch_runtime_boundary_v1.py"
    ),
    "bounded_evaluation_panel_runner_test": (
        "lewm/tests/test_go2_world_model_bounded_branch_evaluation_panel_runner_v1.py"
    ),
    "bounded_runbook": (
        "docs/lewm_go2_world_model_bounded_branch_experiment_v1_runbook_2026-08-02.md"
    ),
    "pilot_joiner": "scripts/join_go2_world_model_counterfactual_pilot_v1.py",
}

# Importing the public bounded supervisor and evaluation-panel runner loads
# these repository-local modules before an authorized attempt can complete.
# Keep them explicit: a committed authority must bind the implementation that
# Python will actually execute, including transitive model/data dependencies.
BOUNDED_DYNAMIC_IMPORT_SOURCE_PATHS = {
    "bounded_main_pool_census": (
        "lewm/benchmarks/go2_recurrent_jepa_main_pool_census.py"
    ),
    "bounded_single_frame_mask_contract": (
        "lewm/benchmarks/go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1.py"
    ),
    "bounded_place_triplet_dataset": (
        "lewm/datasets/go2_memory_role_place_triplets_v1.py"
    ),
    "bounded_recurrent_h4_dataset": (
        "lewm/datasets/go2_recurrent_h4_rgb_sequences.py"
    ),
    "bounded_recurrent_h4_v2_dataset": (
        "lewm/datasets/go2_recurrent_h4_rgb_sequences_v2.py"
    ),
    "bounded_models_package": "lewm/models/__init__.py",
    "bounded_model_encoders": "lewm/models/encoders.py",
    "bounded_model_lewm": "lewm/models/lewm.py",
    "bounded_model_phase2d_spatial": "lewm/models/phase2d_spatial_lewm.py",
    "bounded_model_predictor": "lewm/models/predictor.py",
    "bounded_model_primitive_affordance": "lewm/models/primitive_affordance.py",
    "bounded_model_single_frame_masked": (
        "lewm/models/rgb_single_frame_multiblock_masked_spatial_jepa_v1.py"
    ),
    "bounded_model_sigreg": "lewm/models/sigreg.py",
    "bounded_model_source_action_utility": "lewm/models/source_action_utility.py",
    "bounded_model_spatial": "lewm/models/spatial_lewm.py",
    "bounded_model_spatial_predictor": "lewm/models/spatial_predictor.py",
    "bounded_calibration_authority_contract": (
        "scripts/build_go2_world_model_counterfactual_calibration_authority_v1.py"
    ),
    "bounded_visual_parity_authority_contract": (
        "scripts/build_go2_world_model_visual_domain_parity_authority_v1.py"
    ),
    "bounded_visual_parity_plan_contract": (
        "scripts/build_go2_world_model_visual_domain_parity_plan_v1.py"
    ),
    "bounded_h6_temporal_frame_packer": "scripts/dev_pack_h6_temporal_frames.py",
    "bounded_single_frame_evaluation_api": (
        "scripts/evaluate_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1.py"
    ),
    "bounded_recurrent_temporal_runner": (
        "scripts/run_go2_rgb_recurrent_patch_memory_temporal_jepa_v1.py"
    ),
}


class BoundedBranchAuthorityError(RuntimeError):
    """Raised before malformed metadata can mint pilot authority."""


def canonical_source_paths_v1() -> dict[str, str]:
    paths = {
        **runtime_kernel.EXPECTED_SOURCE_PATHS,
        **runtime_kernel.NON_SMOKE_SOURCE_PATHS,
        **NEW_SOURCE_PATHS,
        **BOUNDED_DYNAMIC_IMPORT_SOURCE_PATHS,
    }
    paths["external_supervisor"] = NEW_SOURCE_PATHS["bounded_external_supervisor"]
    return dict(sorted(paths.items()))


def _iso8601(value: str, *, label: str) -> None:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise BoundedBranchAuthorityError(f"{label} is not ISO-8601") from exc


def committed_source_bindings_v1(source_commit: str) -> list[dict[str, Any]]:
    if _COMMIT.fullmatch(source_commit) is None:
        raise BoundedBranchAuthorityError("source commit is invalid")
    try:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as exc:
        raise BoundedBranchAuthorityError("source commit is not an ancestor of HEAD") from exc
    rows = []
    for name, relative in canonical_source_paths_v1().items():
        binding = pilot.file_binding(REPO_ROOT / relative)
        try:
            runtime_kernel._binding_at_commit(  # noqa: SLF001
                binding, commit=source_commit, label=f"bounded pilot source {name}"
            )
        except pilot.PilotContractError as exc:
            raise BoundedBranchAuthorityError(str(exc)) from exc
        rows.append({"name": name, "binding": binding})
    return rows


def build_source_review_template_v1(*, source_commit: str) -> dict[str, Any]:
    return {
        "schema": pilot.SOURCE_REVIEW_SCHEMA,
        "status": "PENDING_INDEPENDENT_REVIEW",
        "authority_granted_by_this_document": False,
        "reviewed_source_commit": source_commit,
        "reviewed_source_bindings": committed_source_bindings_v1(source_commit),
        "remaining_findings": ["INDEPENDENT_REVIEW_REQUIRED"],
        "reviewer": {
            "identity": "REVIEWER_MUST_REPLACE",
            "independence_basis": "REVIEWER_MUST_REPLACE",
        },
        "reviewed_at": "REVIEWER_MUST_REPLACE_WITH_ISO8601",
        "review_method": ["REVIEWER_MUST_REPLACE"],
        "test_evidence": ["REVIEWER_MUST_REPLACE"],
        "accepted_limitations": ["REVIEWER_MUST_REPLACE"],
    }


def _validate_gate(
    value: object, *, binding: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    required = {
        "schema",
        "status",
        "authority_granted_by_this_document",
        "calibration_receipt_binding",
        "calibration_terminal_binding",
        "calibration_terminal_review_binding",
        "excluded_scene_ids",
        "calibration_wall_seconds",
        "calibration_stored_rgb_bytes",
        "calibration_gpu_baseline_used_bytes",
        "calibration_gpu_peak_used_bytes",
        "calibration_gpu_peak_delta_bytes",
        "selected_device_total_vram_bytes",
        "visual_domain_parity_result_binding",
        "visual_domain_parity_terminal_binding",
        "visual_domain_parity_review_binding",
        "visual_domain_parity_freeze",
        *plan_builder.MODEL_PANEL_FREEZE_FIELDS,
        *plan_builder.SCENE_PANEL_FREEZE_FIELDS,
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise BoundedBranchAuthorityError("calibration gate fields changed")
    normalized_binding = pilot.require_binding(binding, label="calibration gate")
    if (
        value.get("schema") != GATE_SCHEMA
        or value.get("status") != "PASS"
        or value.get("authority_granted_by_this_document") is not False
    ):
        raise BoundedBranchAuthorityError("calibration gate did not pass")
    loaded = {}
    for key, label in (
        ("calibration_receipt_binding", "calibration receipt"),
        ("calibration_terminal_binding", "calibration terminal"),
        ("calibration_terminal_review_binding", "calibration terminal review"),
        ("progression_analysis_binding", "progression analysis"),
        ("scene_panel_binding", "bounded pilot scene panel"),
    ):
        if isinstance(value[key], Mapping) and isinstance(value[key].get("path"), str):
            try:
                plan_builder._reject_protected_path(  # noqa: SLF001
                    Path(str(value[key]["path"])), label=label
                )
            except plan_builder.BoundedBranchPlanError as exc:
                raise BoundedBranchAuthorityError(str(exc)) from exc
        candidate = pilot.require_binding(value[key], label=label)
        document, actual = pilot.read_bound_json(
            Path(candidate["path"]),
            expected_sha256=str(candidate["file_sha256"]),
            expected_byte_count=int(candidate["byte_count"]),
            label=label,
        )
        if actual != candidate:
            raise BoundedBranchAuthorityError(f"{label} binding changed")
        loaded[key] = (document, actual)
    parity_freeze = value.get("visual_domain_parity_freeze")
    required_parity_freeze = {
        "result_binding",
        "terminal_binding",
        "review_binding",
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
        "evidence_scene_ids",
        "comparison_contract",
        "thresholds",
        "measurements",
    }
    if (
        not isinstance(parity_freeze, Mapping)
        or set(parity_freeze) != required_parity_freeze
    ):
        raise BoundedBranchAuthorityError(
            "visual-domain parity freeze fields changed"
        )
    parity_documents = {}
    for key, label in (
        ("result_binding", "visual-domain parity result"),
        ("terminal_binding", "visual-domain parity terminal"),
        ("review_binding", "visual-domain parity independent review"),
    ):
        try:
            candidate = pilot.require_binding(parity_freeze[key], label=label)
            document, actual = pilot.read_bound_json(
                Path(str(candidate["path"])),
                expected_sha256=str(candidate["file_sha256"]),
                expected_byte_count=int(candidate["byte_count"]),
                label=label,
            )
        except (OSError, pilot.PilotContractError) as exc:
            raise BoundedBranchAuthorityError(str(exc)) from exc
        if actual != candidate:
            raise BoundedBranchAuthorityError(f"{label} binding changed")
        parity_documents[key] = (document, actual)
    try:
        independently_derived = plan_builder._validate_calibration_gate(  # noqa: SLF001
            loaded["calibration_receipt_binding"][0],
            receipt_binding=loaded["calibration_receipt_binding"][1],
            terminal=loaded["calibration_terminal_binding"][0],
            terminal_binding=loaded["calibration_terminal_binding"][1],
            terminal_review=loaded["calibration_terminal_review_binding"][0],
            terminal_review_binding=loaded["calibration_terminal_review_binding"][1],
        )
        model_panel_derived = plan_builder._validate_model_panel_freeze(  # noqa: SLF001
            loaded["progression_analysis_binding"][0],
            progression_analysis_binding=loaded["progression_analysis_binding"][1],
        )
        parity_derived = plan_builder._validate_visual_domain_parity_gate_v1(  # noqa: SLF001
            parity_documents["result_binding"][0],
            result_binding=parity_documents["result_binding"][1],
            review=parity_documents["review_binding"][0],
            review_binding=parity_documents["review_binding"][1],
        )
        normalized_panel, scene_panel_derived = (
            plan_builder._validate_scene_panel_receipt_v1(  # noqa: SLF001
                loaded["scene_panel_binding"][0],
                binding=loaded["scene_panel_binding"][1],
                excluded_scene_ids=(
                    set(independently_derived["excluded_scene_ids"])
                    | set(model_panel_derived["model_observational_scene_ids"])
                    | set(parity_derived["evidence_scene_ids"])
                ),
            )
        )
    except plan_builder.BoundedBranchPlanError as exc:
        raise BoundedBranchAuthorityError(str(exc)) from exc
    independently_derived = {
        **independently_derived,
        **model_panel_derived,
        **scene_panel_derived,
        "visual_domain_parity_freeze": parity_derived,
    }
    if any(independently_derived[key] != value[key] for key in independently_derived):
        raise BoundedBranchAuthorityError("calibration gate derivation changed")
    return {**dict(value), "binding": normalized_binding}, normalized_panel


def projected_caps_v1(gate: Mapping[str, Any]) -> dict[str, float | int]:
    wall = max(
        MINIMUM_WALL_SECONDS,
        math.ceil(float(gate["calibration_wall_seconds"]) * WALL_PROJECTION_MULTIPLIER),
    )
    rgb = max(
        MINIMUM_STORED_RGB_BYTES,
        math.ceil(int(gate["calibration_stored_rgb_bytes"]) * RGB_PROJECTION_MULTIPLIER),
    )
    baseline = int(gate["calibration_gpu_baseline_used_bytes"])
    peak = int(gate["calibration_gpu_peak_used_bytes"])
    delta = int(gate["calibration_gpu_peak_delta_bytes"])
    total = int(gate["selected_device_total_vram_bytes"])
    if (
        min(baseline, peak, delta, total) < 0
        or total <= 0
        or peak < baseline
        or delta != peak - baseline
        or peak > total
    ):
        raise BoundedBranchAuthorityError("calibration VRAM measurements are inconsistent")
    projected_delta_numerator = (
        delta * BOUNDED_CONCURRENT_LANES * VRAM_SAFETY_MARGIN_NUMERATOR
    )
    projected_delta_denominator = (
        CALIBRATION_CONCURRENT_LANES * VRAM_SAFETY_MARGIN_DENOMINATOR
    )
    projected_delta = (
        projected_delta_numerator
        + projected_delta_denominator
        - 1
    ) // projected_delta_denominator
    projected_vram = baseline + projected_delta
    vram_hard_cap = math.floor(total * 0.95)
    if wall > MAX_WALL_SECONDS:
        raise BoundedBranchAuthorityError(
            "calibration projects beyond the bounded pilot wall hard cap"
        )
    if rgb > MAX_STORED_RGB_BYTES:
        raise BoundedBranchAuthorityError(
            "calibration projects beyond the bounded pilot RGB-byte hard cap"
        )
    if projected_vram > vram_hard_cap:
        raise BoundedBranchAuthorityError(
            "calibration projects beyond 95 percent of selected-device VRAM"
        )
    return {
        "minimum_wall_seconds": float(wall),
        "stored_rgb_byte_ceiling": int(rgb),
        "selected_device_vram_byte_ceiling": int(projected_vram),
    }


def _expected_caps(plan: Mapping[str, Any], gate: Mapping[str, Any], wall_seconds: float) -> dict[str, Any]:
    projection = projected_caps_v1(gate)
    if (
        not math.isfinite(float(wall_seconds))
        or wall_seconds < float(projection["minimum_wall_seconds"])
        or wall_seconds > MAX_WALL_SECONDS
    ):
        raise BoundedBranchAuthorityError("authority wall cap violates calibrated bounds")
    return {
        **runtime_kernel._expected_authority_caps(plan),  # noqa: SLF001
        "wall_seconds": float(wall_seconds),
        "stored_rgb_byte_ceiling": int(projection["stored_rgb_byte_ceiling"]),
        "selected_device_vram_byte_ceiling": int(
            projection["selected_device_vram_byte_ceiling"]
        ),
    }


def validate_authority_v1(
    authority: object,
    *,
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    gate: Mapping[str, Any],
    gate_binding: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "schema",
        "status",
        "authority_granted_by_this_document",
        "scientific_claim_authorized",
        "authorizer",
        "issued_at",
        "source_commit",
        "review_binding",
        "plan_binding",
        "calibration_gate_binding",
        "model_panel_freeze",
        "scene_panel_freeze",
        "visual_domain_parity_freeze",
        "source_bindings",
        "attempt",
        "caps",
        "runtime_bindings",
        "execution",
        "evaluation_contract",
        "network_access",
        "external_supervisor",
        "platform_gate_disposition",
    }
    if not isinstance(authority, Mapping) or set(authority) != required:
        raise BoundedBranchAuthorityError("bounded authority fields changed")
    normalized_plan = pilot.validate_plan(plan)
    try:
        plan_builder._validate_plan_texture_asset_bindings_v1(  # noqa: SLF001
            normalized_plan
        )
    except plan_builder.BoundedBranchPlanError as exc:
        raise BoundedBranchAuthorityError(str(exc)) from exc
    normalized_gate, normalized_panel = _validate_gate(
        gate, binding=gate_binding
    )
    plan_builder._validate_plan_scene_panel_match_v1(  # noqa: SLF001
        normalized_plan,
        normalized_panel=normalized_panel,
    )
    plan_builder._validate_candidate_render_domain_contract_v1(  # noqa: SLF001
        normalized_plan["render_contract"]
    )
    if any(
        normalized_plan.get(plan_field)
        != normalized_gate["visual_domain_parity_freeze"][freeze_field]
        for plan_field, freeze_field in (
            ("visual_domain_parity_result_binding", "result_binding"),
            ("visual_domain_parity_terminal_binding", "terminal_binding"),
            ("visual_domain_parity_review_binding", "review_binding"),
        )
    ):
        raise BoundedBranchAuthorityError(
            "plan visual-domain parity bindings changed from the reviewed gate"
        )
    if (
        normalized_plan.get("purpose") != PURPOSE
        or authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("status") != AUTHORITY_STATUS
        or authority.get("authority_granted_by_this_document") is not True
        or authority.get("scientific_claim_authorized") is not False
        or authority.get("network_access") is not False
    ):
        raise BoundedBranchAuthorityError("bounded authority identity/scope changed")
    authorizer = authority.get("authorizer")
    if (
        not isinstance(authorizer, Mapping)
        or set(authorizer) != {"identity", "basis"}
        or any(not isinstance(authorizer[key], str) or not authorizer[key].strip() for key in authorizer)
    ):
        raise BoundedBranchAuthorityError("authority authorizer is invalid")
    _iso8601(str(authority.get("issued_at")), label="issued_at")
    if authority.get("plan_binding") != dict(plan_binding):
        raise BoundedBranchAuthorityError("authority plan binding changed")
    if authority.get("calibration_gate_binding") != dict(gate_binding):
        raise BoundedBranchAuthorityError("authority calibration gate binding changed")
    expected_model_panel_freeze = {
        key: normalized_gate[key]
        for key in plan_builder.MODEL_PANEL_FREEZE_FIELDS
    }
    if authority.get("model_panel_freeze") != expected_model_panel_freeze:
        raise BoundedBranchAuthorityError("authority model-panel freeze changed")
    expected_scene_panel_freeze = {
        key: normalized_gate[key]
        for key in plan_builder.SCENE_PANEL_FREEZE_FIELDS
    }
    if authority.get("scene_panel_freeze") != expected_scene_panel_freeze:
        raise BoundedBranchAuthorityError("authority scene-panel freeze changed")
    if (
        authority.get("visual_domain_parity_freeze")
        != normalized_gate["visual_domain_parity_freeze"]
    ):
        raise BoundedBranchAuthorityError(
            "authority visual-domain parity freeze changed"
        )
    source_commit = authority.get("source_commit")
    sources = authority.get("source_bindings")
    if (
        not isinstance(source_commit, str)
        or _COMMIT.fullmatch(source_commit) is None
        or not isinstance(sources, list)
        or [row.get("name") if isinstance(row, Mapping) else None for row in sources]
        != list(canonical_source_paths_v1())
    ):
        raise BoundedBranchAuthorityError("authority source closure changed")
    for row in sources:
        if not isinstance(row, Mapping) or set(row) != {"name", "binding"}:
            raise BoundedBranchAuthorityError("authority source row changed")
        bound = pilot.require_binding(row["binding"], label=f"source {row['name']}")
        expected = (REPO_ROOT / canonical_source_paths_v1()[str(row["name"])]).resolve()
        if Path(str(bound["path"])).resolve() != expected:
            raise BoundedBranchAuthorityError(f"source path changed for {row['name']}")
    source_by_name = {str(row["name"]): row["binding"] for row in sources}
    parity_freeze = authority["visual_domain_parity_freeze"]
    if (
        parity_freeze["reference_renderer_source_binding"]
        != source_by_name["bounded_visual_domain_reference_renderer"]
        or parity_freeze["reference_renderer_source_binding"]
        != source_by_name["historical_textured_v03_renderer"]
        or parity_freeze["reference_texture_source_binding"]
        != source_by_name["textures"]
        or parity_freeze["candidate_collector_source_binding"]
        != source_by_name["collector"]
        or parity_freeze["candidate_renderer_source_binding"]
        != source_by_name["historical_textured_v03_renderer"]
        or parity_freeze["evaluator_source_binding"]
        != source_by_name["bounded_visual_domain_parity_evaluator"]
        or parity_freeze["evaluator_source_binding"]
        != source_by_name["visual_domain_parity_evaluator"]
    ):
        raise BoundedBranchAuthorityError(
            "visual-domain reference renderer sources are outside the reviewed closure"
        )
    exact_attempt = {
        "id": normalized_plan["attempt_id"],
        "root": normalized_plan["output_root"],
        "maximum_attempts": 1,
        "must_be_absent": True,
        "root_creation_consumes_attempt": True,
        "reservation_records_consumed_attempt": True,
        "retry": False,
        "resume": False,
        "overwrite": False,
        "refill": False,
    }
    if authority.get("attempt") != exact_attempt:
        raise BoundedBranchAuthorityError("authority attempt boundary changed")
    expected_caps = _expected_caps(
        normalized_plan,
        normalized_gate,
        float(authority["caps"].get("wall_seconds", -1)),
    )
    if authority.get("caps") != expected_caps:
        raise BoundedBranchAuthorityError("authority caps changed")
    if (
        authority.get("runtime_bindings") != normalized_plan["runtime_bindings"]
        or authority.get("execution") != normalized_plan["execution_contract"]
        or authority.get("evaluation_contract") != evaluation.evaluation_contract_v1()
    ):
        raise BoundedBranchAuthorityError("runtime/evaluation contract changed")
    disposition = authority.get("platform_gate_disposition")
    if disposition != {
        "platform_hard_gates_resolved": True,
        "scope": PURPOSE,
        "model_panel_frozen_before_generation": True,
        "scene_panel_frozen_before_generation": True,
        "visual_domain_parity_measured_and_independently_reviewed": True,
        "outputs_eligible_for_training_after_receipt_join": False,
        "outputs_eligible_for_preregistered_evaluation_after_terminal_review": True,
        "outputs_eligible_for_scientific_claim": False,
        "authorizes_this_exact_generation": True,
        "authorizes_promotion": False,
    }:
        raise BoundedBranchAuthorityError("platform gate disposition changed")
    supervisor = authority.get("external_supervisor")
    reviewed_supervisor = next(
        row["binding"] for row in sources if row["name"] == "external_supervisor"
    )
    if (
        not isinstance(supervisor, Mapping)
        or set(supervisor) != {"source_binding", "terminal_reviewer"}
        or supervisor.get("source_binding") != reviewed_supervisor
        or not isinstance(supervisor.get("terminal_reviewer"), str)
        or not supervisor["terminal_reviewer"].strip()
    ):
        raise BoundedBranchAuthorityError("external supervisor contract changed")
    return dict(authority)


def build_authority_v1(
    *,
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    gate: Mapping[str, Any],
    gate_binding: Mapping[str, Any],
    review: Mapping[str, Any],
    review_binding: Mapping[str, Any],
    authorizer_identity: str,
    authorizer_basis: str,
    issued_at: str,
    terminal_reviewer: str,
    wall_seconds: float,
) -> dict[str, Any]:
    normalized_plan = pilot.validate_plan(plan)
    try:
        plan_builder._validate_plan_texture_asset_bindings_v1(  # noqa: SLF001
            normalized_plan
        )
    except plan_builder.BoundedBranchPlanError as exc:
        raise BoundedBranchAuthorityError(str(exc)) from exc
    if normalized_plan.get("purpose") != PURPOSE:
        raise BoundedBranchAuthorityError("authority requires a bounded_wm_a_pilot plan")
    normalized_gate, normalized_panel = _validate_gate(
        gate, binding=gate_binding
    )
    plan_builder._validate_plan_scene_panel_match_v1(  # noqa: SLF001
        normalized_plan,
        normalized_panel=normalized_panel,
    )
    plan_builder._validate_candidate_render_domain_contract_v1(  # noqa: SLF001
        normalized_plan["render_contract"]
    )
    if (
        normalized_plan.get("visual_domain_parity_result_binding")
        != normalized_gate["visual_domain_parity_freeze"]["result_binding"]
    ):
        raise BoundedBranchAuthorityError(
            "plan visual-domain parity binding changed from the reviewed gate"
        )
    for value, label in (
        (authorizer_identity, "authorizer identity"),
        (authorizer_basis, "authorizer basis"),
        (terminal_reviewer, "terminal reviewer"),
    ):
        if not isinstance(value, str) or not value.strip():
            raise BoundedBranchAuthorityError(f"{label} is empty")
    _iso8601(issued_at, label="issued_at")
    source_commit = review.get("reviewed_source_commit")
    sources = review.get("reviewed_source_bindings")
    if (
        not isinstance(source_commit, str)
        or _COMMIT.fullmatch(source_commit) is None
        or not isinstance(sources, list)
        or [row.get("name") if isinstance(row, Mapping) else None for row in sources]
        != list(canonical_source_paths_v1())
    ):
        raise BoundedBranchAuthorityError("source review closure changed")
    pilot.validate_source_review(
        review,
        authority={"source_commit": source_commit, "source_bindings": sources},
    )
    supervisor_binding = next(
        row["binding"] for row in sources if row["name"] == "external_supervisor"
    )
    authority = {
        "schema": AUTHORITY_SCHEMA,
        "status": AUTHORITY_STATUS,
        "authority_granted_by_this_document": True,
        "scientific_claim_authorized": False,
        "authorizer": {"identity": authorizer_identity, "basis": authorizer_basis},
        "issued_at": issued_at,
        "source_commit": source_commit,
        "review_binding": dict(review_binding),
        "plan_binding": dict(plan_binding),
        "calibration_gate_binding": dict(gate_binding),
        "model_panel_freeze": {
            key: normalized_gate[key]
            for key in plan_builder.MODEL_PANEL_FREEZE_FIELDS
        },
        "scene_panel_freeze": {
            key: normalized_gate[key]
            for key in plan_builder.SCENE_PANEL_FREEZE_FIELDS
        },
        "visual_domain_parity_freeze": normalized_gate[
            "visual_domain_parity_freeze"
        ],
        "source_bindings": list(sources),
        "attempt": {
            "id": normalized_plan["attempt_id"],
            "root": normalized_plan["output_root"],
            "maximum_attempts": 1,
            "must_be_absent": True,
            "root_creation_consumes_attempt": True,
            "reservation_records_consumed_attempt": True,
            "retry": False,
            "resume": False,
            "overwrite": False,
            "refill": False,
        },
        "caps": _expected_caps(normalized_plan, normalized_gate, wall_seconds),
        "runtime_bindings": dict(normalized_plan["runtime_bindings"]),
        "execution": dict(normalized_plan["execution_contract"]),
        "evaluation_contract": evaluation.evaluation_contract_v1(),
        "network_access": False,
        "external_supervisor": {
            "source_binding": supervisor_binding,
            "terminal_reviewer": terminal_reviewer,
        },
        "platform_gate_disposition": {
            "platform_hard_gates_resolved": True,
            "scope": PURPOSE,
            "model_panel_frozen_before_generation": True,
            "scene_panel_frozen_before_generation": True,
            "visual_domain_parity_measured_and_independently_reviewed": True,
            "outputs_eligible_for_training_after_receipt_join": False,
            "outputs_eligible_for_preregistered_evaluation_after_terminal_review": True,
            "outputs_eligible_for_scientific_claim": False,
            "authorizes_this_exact_generation": True,
            "authorizes_promotion": False,
        },
    }
    return validate_authority_v1(
        authority,
        plan=normalized_plan,
        plan_binding=plan_binding,
        gate=gate,
        gate_binding=gate_binding,
    )


def _read_bound(path: Path, digest: str, byte_count: int, *, label: str):
    try:
        plan_builder._reject_protected_path(path, label=label)  # noqa: SLF001
    except plan_builder.BoundedBranchPlanError as exc:
        raise BoundedBranchAuthorityError(str(exc)) from exc
    if _SHA.fullmatch(digest) is None or byte_count <= 0:
        raise BoundedBranchAuthorityError(f"{label} caller binding is malformed")
    return pilot.read_bound_json(
        path,
        expected_sha256=digest,
        expected_byte_count=byte_count,
        label=label,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    review = subparsers.add_parser("review-template")
    review.add_argument("--source-commit", required=True)
    review.add_argument("--output", required=True, type=Path)
    authority = subparsers.add_parser("authority")
    for name in ("plan", "calibration-gate", "review"):
        authority.add_argument(f"--{name}", required=True, type=Path)
        authority.add_argument(f"--expected-{name}-sha256", required=True)
        authority.add_argument(f"--expected-{name}-byte-count", required=True, type=int)
    authority.add_argument("--authorizer-identity", required=True)
    authority.add_argument("--authorizer-basis", required=True)
    authority.add_argument("--issued-at", required=True)
    authority.add_argument("--terminal-reviewer", required=True)
    authority.add_argument("--wall-seconds", required=True, type=float)
    authority.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "review-template":
        document = build_source_review_template_v1(source_commit=args.source_commit)
    else:
        plan, plan_binding = _read_bound(
            args.plan, args.expected_plan_sha256, args.expected_plan_byte_count,
            label="bounded pilot plan",
        )
        gate, gate_binding = _read_bound(
            args.calibration_gate,
            args.expected_calibration_gate_sha256,
            args.expected_calibration_gate_byte_count,
            label="bounded pilot calibration gate",
        )
        review, review_binding = _read_bound(
            args.review, args.expected_review_sha256, args.expected_review_byte_count,
            label="bounded pilot source review",
        )
        document = build_authority_v1(
            plan=plan,
            plan_binding=plan_binding,
            gate=gate,
            gate_binding=gate_binding,
            review=review,
            review_binding=review_binding,
            authorizer_identity=args.authorizer_identity,
            authorizer_basis=args.authorizer_basis,
            issued_at=args.issued_at,
            terminal_reviewer=args.terminal_reviewer,
            wall_seconds=args.wall_seconds,
        )
    binding = pilot.write_json_exclusive(args.output, document)
    print(json.dumps({"document": binding, "schema": document["schema"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORITY_SCHEMA",
    "AUTHORITY_STATUS",
    "BOUNDED_DYNAMIC_IMPORT_SOURCE_PATHS",
    "BoundedBranchAuthorityError",
    "GATE_SCHEMA",
    "build_authority_v1",
    "build_source_review_template_v1",
    "canonical_source_paths_v1",
    "committed_source_bindings_v1",
    "projected_caps_v1",
    "validate_authority_v1",
]
