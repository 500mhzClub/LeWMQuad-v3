from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from lewm.safety import protected_contact_scope_requirements_review_v1 as stage_a
from scripts import audit_contact_critical_geometry_mismatch_v1 as audit


def _redigest(value: dict[str, object]) -> None:
    value.pop("content_digest", None)
    value["content_digest"] = audit.canonical_digest(value)


def test_exact_diagnostic_vocabulary_and_canonical_workers() -> None:
    assert audit.DIAGNOSTIC_CAUSE_IDS == (
        "CONTACT_CRITICAL_PATCH_UNOBSERVED",
        "POINT_TO_PRIMITIVE_DISTANCE_MISMATCH",
        "GLOBAL_THRESHOLD_HETEROGENEITY",
        "SELF_OCCLUSION_AT_CONTACT_CRITICAL_REGION",
        "UNRESOLVED_GEOMETRIC_MISMATCH",
    )
    assert audit.CANONICAL_WORKERS == audit.DEFAULT_WORKERS == 32
    assert audit.CANONICAL_OUTPUT_RELATIVE_PATH == Path(
        "docs/lewm_contact_critical_geometry_mismatch_audit_v1.json"
    )
    assert audit.MULTI_INDEX_SHA256 == {
        "dual": "1cc9f49821b78a35204baaef30182a580db4ae724f08f80a1e21c4bbc9a8af99",
        "three": "757fa10922da85d2fb1674a42d9b788590db3c82f200e443f4ad2a05c22133ce",
        "diagnostic": "0d3d86696f626840af84364cf0e8d2cba7921498f754c38eb8127a08a2f0beb2",
    }
    assert audit.EXACT_RESULT_SHA256 == (
        "e3eb1a8a64f147eed3bb29d5a778b799f86b45e178c8400b92adf4c253e3f5b4"
    )
    assert audit.EXACT_GEOMETRY_INDEX_SHA256 == (
        "67b473e6d0e0c422e4b2916b800399c98d5ec60383d398ff907fe2b8909a1d7f"
    )


def test_compact_index_binding_never_embeds_record_rows(tmp_path: Path) -> None:
    binding = audit._compact_index_binding(
        path=tmp_path / "index.json",
        sha256=audit.MULTI_INDEX_SHA256["dual"],
        index={
            "content_digest": "a" * 64,
            "states": 176,
            "transitions": 29470,
            "records": [{"large": "payload"}] * 528,
            "conditions": list(audit.DUAL_CONDITION_IDS),
        },
        condition_ids=audit.DUAL_CONDITION_IDS,
        mode_ids=audit.EVIDENCE_MODE_IDS,
    )
    assert set(binding) == {
        "path", "sha256", "content_digest", "states", "transitions",
        "records", "condition_ids", "mode_ids",
    }
    assert binding["sha256"] == audit.MULTI_INDEX_SHA256["dual"]
    assert binding["records"] == 528
    assert not isinstance(binding["records"], list)
    assert binding["condition_ids"] == list(audit.DUAL_CONDITION_IDS)
    assert binding["mode_ids"] == list(audit.EVIDENCE_MODE_IDS)


def test_compact_index_binding_resolves_numeric_axes_and_fails_count_drift(
    tmp_path: Path,
) -> None:
    index = {
        "content_digest": "b" * 64,
        "states": 176,
        "transitions": 29470,
        "records": [{}] * 176,
        "conditions": 4,
        "evidence_modes": 2,
    }
    binding = audit._compact_index_binding(
        path=tmp_path / "single-index.json",
        sha256=audit.SINGLE_MATERIALIZATION_INDEX_SHA256,
        index=index,
        condition_ids=audit.SINGLE_CONDITION_IDS,
        mode_ids=audit.EVIDENCE_MODE_IDS,
    )
    assert binding["condition_ids"] == list(audit.SINGLE_CONDITION_IDS)
    assert binding["mode_ids"] == list(audit.EVIDENCE_MODE_IDS)
    assert binding["sha256"] == audit.SINGLE_MATERIALIZATION_INDEX_SHA256

    changed = dict(index, conditions=3)
    with pytest.raises(audit.AuditError, match="conditions count drift"):
        audit._compact_index_binding(
            path=tmp_path / "single-index.json",
            sha256=audit.SINGLE_MATERIALIZATION_INDEX_SHA256,
            index=changed,
            condition_ids=audit.SINGLE_CONDITION_IDS,
            mode_ids=audit.EVIDENCE_MODE_IDS,
        )


def test_stage_a_barrier_accepts_only_unresolved_unauthorized_scope() -> None:
    receipt = stage_a.build_contract_receipt()
    binding = audit.validate_stage_a_contract(receipt)
    assert binding["primary_classification"] == audit.STAGE_A_PRIMARY
    assert binding["stage_b_authorized"] is False
    assert binding["scope_narrowing_authorized"] is False
    assert binding["protected_links"] == 13
    assert binding["protected_collision_components"] == 27

    changed = copy.deepcopy(receipt)
    changed["stage_b_gate"]["authorized"] = True
    _redigest(changed["stage_b_gate"])
    _redigest(changed)
    with pytest.raises(audit.AuditError, match="Stage B"):
        audit.validate_stage_a_contract(changed)

    changed = copy.deepcopy(receipt)
    changed["scope_decision"]["protected_links_removed"] = ["base"]
    _redigest(changed)
    with pytest.raises(audit.AuditError, match="scope or label"):
        audit.validate_stage_a_contract(changed)


def test_mismatch_classifier_is_multilabel_but_has_one_descriptive_primary() -> None:
    self_blocked = audit.classify_diagnostic_causes(
        oracle_contact=False,
        predicted_contact=True,
        patch_supported=False,
        nominal_fov=True,
        self_occluded=True,
        exact_clearance_m=0.005,
        observed_clearance_m=None,
        threshold_m=0.01,
    )
    assert self_blocked["supported_cause_ids"] == [
        audit.CONTACT_CRITICAL_PATCH_UNOBSERVED,
        audit.SELF_OCCLUSION_AT_CONTACT_CRITICAL_REGION,
    ]
    assert self_blocked["primary_cause"] == (
        audit.SELF_OCCLUSION_AT_CONTACT_CRITICAL_REGION
    )

    point_and_threshold = audit.classify_diagnostic_causes(
        oracle_contact=True,
        predicted_contact=False,
        patch_supported=True,
        nominal_fov=True,
        self_occluded=False,
        exact_clearance_m=-0.001,
        observed_clearance_m=0.02,
        threshold_m=0.01,
    )
    assert point_and_threshold["supported_cause_ids"] == [
        audit.POINT_TO_PRIMITIVE_DISTANCE_MISMATCH,
        audit.GLOBAL_THRESHOLD_HETEROGENEITY,
    ]
    assert point_and_threshold["primary_cause"] == (
        audit.POINT_TO_PRIMITIVE_DISTANCE_MISMATCH
    )

    unresolved = audit.classify_diagnostic_causes(
        oracle_contact=True,
        predicted_contact=False,
        patch_supported=True,
        nominal_fov=True,
        self_occluded=False,
        exact_clearance_m=0.005,
        observed_clearance_m=0.005,
        threshold_m=0.01,
    )
    assert unresolved["supported_cause_ids"] == [
        audit.UNRESOLVED_GEOMETRIC_MISMATCH
    ]
    assert unresolved["scope_change_authorized"] is False


def test_mismatch_classifier_rejects_nonerrors() -> None:
    with pytest.raises(audit.AuditError, match="must be a contact-classification mismatch"):
        audit.classify_diagnostic_causes(
            oracle_contact=True,
            predicted_contact=True,
            patch_supported=True,
            nominal_fov=True,
            self_occluded=False,
            exact_clearance_m=0.0,
            observed_clearance_m=0.0,
            threshold_m=0.01,
        )


def test_coverage_error_reducer_uses_frozen_threshold_and_skips_decision_duplicates() -> None:
    rows = [
        {
            "condition_id": "C", "evidence_mode": "M",
            "error_scope": "CONTACT_CLASSIFICATION",
            "oracle_contact": True, "predicted_contact": False,
            "supporting_origin_ids": ["HEAD"], "nominal_fov": True,
            "robot_self_occlusion": False, "robot_link": "base",
            "exact_clearance_m": -0.001, "observed_clearance_m": 0.02,
            "error_class": "UNRESOLVED",
        },
        {
            "condition_id": "C", "evidence_mode": "M",
            "error_scope": "CONTACT_CLASSIFICATION",
            "oracle_contact": False, "predicted_contact": True,
            "supporting_origin_ids": [], "nominal_fov": True,
            "robot_self_occlusion": True, "robot_link": "FR_hip",
            "exact_clearance_m": 0.01, "observed_clearance_m": None,
            "error_class": "ROBOT_SELF_OCCLUSION",
        },
        {
            "condition_id": "C", "evidence_mode": "M",
            "error_scope": "DECISION_LEVEL",
            "oracle_contact": False, "predicted_contact": False,
        },
    ]
    result = audit.reduce_coverage_error_rows(
        rows, source_id="MULTI_ORIGIN", thresholds={("C", "M"): 0.01},
        multi_origin=True,
    )
    assert result["rows"] == 2
    bucket = result["condition_mode"]["MULTI_ORIGIN|C|M"]
    assert bucket["false_negative_rows"] == 1
    assert bucket["false_positive_rows"] == 1
    assert bucket["patch_unobserved_rows"] == 1
    assert bucket["point_primitive_mismatch_rows"] == 1
    assert bucket["existing_error_classes"] == {
        "ROBOT_SELF_OCCLUSION": 1, "UNRESOLVED": 1
    }
    assert result["supported_cause_counts"][audit.GLOBAL_THRESHOLD_HETEROGENEITY] == 1


def test_per_link_reducer_preserves_support_fraction_and_exact_delta() -> None:
    multi_rows = [
        {
            "condition_id": "C", "evidence_mode": "M", "role": "heldout",
            "link_name": "base", "collision_region": "TRUNK",
            "oracle_contact_link": True, "observation_support": False,
            "unsupported_swept_volume_fraction": 0.25,
            "nominal_fov_inclusion_by_origin": {"HEAD": True},
            "robot_self_occlusion_fraction": 0.5,
            "exact_oracle_minimum_clearance_m": -0.001,
            "minimum_observed_environment_clearance_m": 0.02,
        },
        {
            "condition_id": "C", "evidence_mode": "M", "role": "heldout",
            "link_name": "FR_hip", "collision_region": "HIP",
            "oracle_contact_link": False, "observation_support": True,
            "unsupported_swept_volume_fraction": 0.0,
            "nominal_fov_inclusion_by_origin": {"HEAD": True},
            "robot_self_occlusion_fraction": 0.0,
            "exact_oracle_minimum_clearance_m": 0.1,
            "minimum_observed_environment_clearance_m": 0.1,
        },
    ]
    result = audit.reduce_per_link_rows(
        multi_rows, source_id="MULTI_ORIGIN", multi_origin=True
    )
    assert result["rows"] == 2
    bucket = result["condition_mode_role"]["MULTI_ORIGIN|C|M|heldout"]
    assert bucket["contact_rows"] == 1
    assert bucket["contact_unsupported_any_rows"] == 1
    assert bucket["contact_unsupported_fraction_mean"] == 0.25
    assert bucket["contact_self_occluded_rows"] == 1
    assert bucket["contact_point_primitive_mismatch_rows"] == 0


def test_unsupported_finite_fallback_is_not_a_supported_clearance_residual() -> None:
    bucket = audit._new_event_bucket()
    audit._event_update(
        bucket,
        support=False,
        event_support=False,
        nominal=True,
        direct=False,
        self_occluded=True,
        inherited=False,
        exact=-0.01,
        observed=0.50,
        threshold=0.01,
    )
    assert bucket["deltas"] == []
    assert bucket["supported_observed_clearance_m"] == []
    assert bucket["point_primitive_mismatch_events"] == 0
    assert bucket["patch_unobserved_self_occluded_events"] == 1


def _write_synthetic_npz(path: Path, *, single: bool) -> None:
    transitions, modes, steps, links = 2, 2, 50, 13
    prefix = (transitions, 4, modes, steps, links) if single else (
        transitions, modes, steps, links
    )
    clearance = np.full(prefix, np.inf, np.float32)
    support = np.zeros(prefix, np.uint8)
    event = np.zeros(prefix, np.uint8)
    nominal = np.ones(prefix, np.uint8)
    direct = np.ones(prefix, np.uint8)
    self_occ = np.zeros(prefix, np.uint8)
    inherited = np.zeros(prefix, np.uint8)
    if single:
        clearance[0, :, :, 0, 0] = 0.02
        clearance[0, :, :, 1, 0] = 0.001
        support[0, :, :, 0, 0] = 1
        support[0, :, :, 1, 0] = 1
        event[0, :, :, 0, 0] = 1
    else:
        clearance[0, :, 0, 0] = 0.02
        clearance[0, :, 1, 0] = 0.001
        support[0, :, 0, 0] = 1
        support[0, :, 1, 0] = 1
        event[0, :, 0, 0] = 1
    values: dict[str, np.ndarray] = {
        "clearance_m": clearance,
        "support": support,
        "event_time_support": event,
        "nominal_fov": nominal,
        "direct_visibility": direct,
        "self_occluded": self_occ,
        "finite_scan_support_inherited": inherited,
    }
    if single:
        values.update({
            "frozen_contact": np.asarray([1, 0], np.uint8),
            "oracle_contact_step": np.asarray([0, -1], np.int16),
            "oracle_contact_link": np.asarray([0, -1], np.int16),
            "target_geom_index": np.zeros((transitions, steps, links), np.int16),
            "oracle_clearance_m": np.full((transitions, steps, links), -0.001, np.float32),
        })
    np.savez_compressed(path, **values)


def test_npz_contact_worker_keeps_roles_components_and_calibration_separate(tmp_path: Path) -> None:
    body = tmp_path / "body.npz"
    multi = tmp_path / "multi.npz"
    _write_synthetic_npz(body, single=True)
    _write_synthetic_npz(multi, single=False)
    thresholds = {
        f"{condition}|{mode}": 0.01
        for condition in audit.SINGLE_CONDITION_IDS
        for mode in audit.EVIDENCE_MODE_IDS
    }
    thresholds["DUAL_REALISTIC_L2_SCAN|PLANNING_TIME_CAUSAL_CLOUD"] = 0.01
    thresholds["DUAL_REALISTIC_L2_SCAN|TRUE_FUTURE_OBSERVABILITY_CLOUD"] = 0.01
    result = audit.reduce_state_contact_events({
        "state_id": "state", "role": "calibration", "family": "family",
        "body_path": str(body), "body_sha256": "unused",
        "multi_shards": {
            "DUAL_REALISTIC_L2_SCAN": {"path": str(multi), "sha256": "unused"}
        },
        "shape_names": [f"SHAPE_{index}:box" for index in range(27)],
        "thresholds": thresholds, "verify_shard_hashes": False,
    })
    assert result["frozen_contact_events"] == result["resolved_contact_events"] == 1
    assert result["exact_shapes"] == {"00:SHAPE_0:box": 1}
    assert len(result["condition_mode"]) == 10
    assert all("|calibration" in key for key in result["condition_mode_by_role"])
    assert len(result["calibration_per_link"]) == 10
    assert len(result["calibration_per_shape"]) == 10
    assert all(
        bucket["patch_clearance_with_unsupported_inf"] == pytest.approx([0.001])
        for bucket in result["calibration_per_link"].values()
    )
    assert all(
        bucket["deltas"] == pytest.approx([0.021])
        for bucket in result["per_link"].values()
    )


def test_calibration_threshold_shift_uses_nearest_rank_and_infinite_unsupported() -> None:
    bucket = audit._new_event_bucket()
    for value in (0.001, 0.002, 0.003, 0.004, math.inf):
        audit._event_update(
            bucket,
            support=math.isfinite(value), event_support=math.isfinite(value),
            nominal=True, direct=True, self_occluded=False, inherited=False,
            exact=-0.001, observed=value, threshold=0.0025,
        )
    raw = {"calibration_per_link": {"C|M|base": bucket},
           "calibration_per_shape": {"C|M|00:BASE:box": copy.deepcopy(bucket)}}
    # Reproduce the contact-event finalizer's nearest-rank fields.
    for group in raw.values():
        for item in group.values():
            values = item["patch_clearance_with_unsupported_inf"]
            rank = math.ceil(0.95 * len(values)) - 1
            item["nearest_rank_q95_including_unsupported"] = "POSITIVE_INFINITY"
            item["nearest_rank_q95_rank_zero_based"] = rank
    result = audit.build_calibration_threshold_shift_diagnostic(
        raw, {("C", "M"): 0.0025}
    )
    link = result["per_link"]["C|M|base"]
    assert link["nearest_rank_q95_m"] == "POSITIVE_INFINITY"
    assert link["nearest_rank_q95_rank_zero_based"] == 4
    assert link["q95_minus_frozen_global_threshold_m"] is None
    assert result["interpolation_used"] is False
    assert result["heldout_used"] is False


def test_unresolved_attribution_supports_only_the_conservative_residual_cause() -> None:
    first = {
        "supported_cause_counts": {
            cause: (3 if cause == audit.POINT_TO_PRIMITIVE_DISTANCE_MISMATCH else 0)
            for cause in audit.DIAGNOSTIC_CAUSE_IDS
        }
    }
    result = audit.summarize_diagnostic_cause_evidence(
        (first,), unresolved_exact_contact_attribution_events=9
    )
    assert result["supported_diagnostic_cause_ids"] == [
        audit.POINT_TO_PRIMITIVE_DISTANCE_MISMATCH,
        audit.UNRESOLVED_GEOMETRIC_MISMATCH,
    ]
    assert result["diagnostic_cause_evidence_counts"][
        audit.UNRESOLVED_GEOMETRIC_MISMATCH
    ] == 9
    assert result["diagnostic_cause_evidence_detail"][
        "contact_classification_mismatch_rows"
    ][audit.UNRESOLVED_GEOMETRIC_MISMATCH] == 0
    with pytest.raises(audit.AuditError, match="negative"):
        audit.summarize_diagnostic_cause_evidence(
            (), unresolved_exact_contact_attribution_events=-1
        )


def _minimal_valid_result() -> dict[str, object]:
    modes = {
        f"C{index}|M": {"contact_events": 1} for index in range(21)
    }
    roles = {
        "C0|M|training": {}, "C0|M|calibration": {}, "C0|M|heldout": {}
    }
    core: dict[str, object] = {
        "schema_version": audit.SCHEMA_VERSION,
        "diagnostic_cause_ids": list(audit.DIAGNOSTIC_CAUSE_IDS),
        "supported_diagnostic_cause_ids": [],
        "diagnostic_cause_evidence_counts": {
            cause: 0 for cause in audit.DIAGNOSTIC_CAUSE_IDS
        },
        "runtime_s": 1.0,
        "workers": 32,
        "stage_a_barrier": {
            "primary_classification": audit.STAGE_A_PRIMARY,
            "stage_b_authorized": False,
            "scope_narrowing_authorized": False,
        },
        "exact_contact_event_diagnostic": {
            "condition_mode": modes,
            "condition_mode_by_role": roles,
        },
        "interpretation_boundary": {
            "contact_labels_changed": False,
            "protected_links_or_shapes_removed": False,
            "scope_narrowing_authorized": False,
            "stage_b_authorized": False,
            "raycasting_rerun": False,
            "model_training": False,
            "fresh_panel": False,
            "jepa_g2_memory_navigation_or_routing": False,
            "causes_are_descriptive_not_requirements_authority": True,
        },
        "prohibited_action_counters": {"everything": 0},
    }
    return audit.attach_content_digest(core)


def test_canonical_writer_allows_only_exact_docs_path_and_is_atomic(tmp_path: Path) -> None:
    result = _minimal_valid_result()
    output = tmp_path / audit.CANONICAL_OUTPUT_RELATIVE_PATH
    audit.write_canonical_result(result, output, repo_root=tmp_path)
    assert output.read_bytes().endswith(b"\n")
    persisted = json.loads(output.read_text())
    audit.validate_audit_result(persisted)
    with pytest.raises(audit.AuditError, match="must not already exist"):
        audit.write_canonical_result(result, output, repo_root=tmp_path)
    with pytest.raises(audit.AuditError, match="must be exactly"):
        audit.write_canonical_result(result, tmp_path / "other.json", repo_root=tmp_path)


def test_result_validator_rejects_nonzero_prohibited_counter_and_invented_cause() -> None:
    result = _minimal_valid_result()
    changed = copy.deepcopy(result)
    changed["prohibited_action_counters"]["everything"] = 1
    _redigest(changed)
    with pytest.raises(audit.AuditError, match="counter"):
        audit.validate_audit_result(changed)
    changed = copy.deepcopy(result)
    changed["supported_diagnostic_cause_ids"] = [audit.UNRESOLVED_GEOMETRIC_MISMATCH]
    _redigest(changed)
    with pytest.raises(audit.AuditError, match="actual evidence"):
        audit.validate_audit_result(changed)
