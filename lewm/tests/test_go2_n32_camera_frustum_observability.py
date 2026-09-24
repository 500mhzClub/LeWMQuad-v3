from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from lewm.benchmarks.go2_n32_camera_frustum_observability import (
    ANGULAR_BIN_COUNT,
    AUTHORIZATION_SCHEMA,
    CAMERA_XYZ_BODY_M,
    CLASS_NAMES,
    ENDPOINT_SIDES,
    EXECUTION_BINDING_SHA256,
    FAMILIES,
    FREE_CLASS,
    HALF_FOV_RAD,
    MAPPING_AUDIT_SCHEMA,
    OBSERVABILITY_SUMMARY_SCHEMA,
    OCCUPIED_CLASS,
    RESULT_SCHEMA,
    UNKNOWN_CLASS,
    aggregate_label_observability,
    analyze_frame_labels,
    audit_camera_centered_mapping,
    authorization_decision,
    build_camera_centered_mapping,
    camera_centered_support_mask,
    camera_point_to_bin,
    canonical_json_sha256,
    geometry_contract,
    old_body_column_span_audit,
)


EXPECTED_MAPPING_SHA256 = (
    "2b8cfb9dcf2deeebe7304d64a4a79b1631eb658991108eb3c3149cccf7a7dd4e"
)
EXPECTED_SUPPORT_SHA256 = (
    "026d7654864bea7ae0545bd6448f6def64519a3bedcbc7ea747e7b4b95f82b3a"
)
EXPECTED_SPAN_TABLE_SHA256 = (
    "bfd10e1b1b7a4e1b2497682b41f8886ae610f43b493c85e5fe9133376bf72eaf"
)


def _frame_key(index: int) -> dict[str, object]:
    return {
        "family": FAMILIES[index % len(FAMILIES)],
        "scene_id": f"synthetic_{index:02d}",
        "global_row": index,
        "side": ENDPOINT_SIDES[index % len(ENDPOINT_SIDES)],
    }


def _unknown_target() -> np.ndarray:
    return np.full((64, 64), UNKNOWN_CLASS, dtype=np.uint8)


def _full_mask() -> np.ndarray:
    return np.ones((64, 64), dtype=np.uint8)


def _analyze(index: int, target: np.ndarray | None = None) -> dict:
    key = _frame_key(index)
    return analyze_frame_labels(
        _unknown_target() if target is None else target,
        _full_mask(),
        frame_key=key,
        family=str(key["family"]),
        endpoint_side=str(key["side"]),
    )


def _complete_reports() -> list[dict]:
    return [_analyze(index) for index in range(10)]


def test_frozen_identity_geometry_and_class_schemas_are_exact() -> None:
    geometry = geometry_contract()

    assert EXECUTION_BINDING_SHA256 == (
        "c045a5566e53686ab80fdc86c2de910d312c02c5f03f253dfda13be7a85a16c9"
    )
    assert RESULT_SCHEMA == (
        "lewm_go2_n32_camera_frustum_observability_audit_result_v1"
    )
    assert tuple(geometry["family_order"]) == FAMILIES
    assert tuple(geometry["endpoint_side_order"]) == ENDPOINT_SIDES
    assert tuple(geometry["class_order"]) == CLASS_NAMES
    assert geometry["cartesian_shape"] == [64, 64]
    assert geometry["camera_xyz_body_m"] == [0.326, 0.0, 0.043]
    assert geometry["camera_near_m"] == 0.05
    assert geometry["horizontal_fov_deg"] == 78.323
    assert geometry["range_interval_m"] == [0.0, 6.4]
    assert geometry["mapping_dtype"] == "little_endian_signed_int16"


def test_camera_centered_mapping_has_frozen_hashes_counts_and_injectivity() -> None:
    mapping = build_camera_centered_mapping()
    support = camera_centered_support_mask(mapping)
    audit = audit_camera_centered_mapping(mapping)

    assert mapping.shape == (64, 64, 2)
    assert mapping.dtype == np.int16
    assert support.shape == (64, 64)
    assert support.dtype == np.bool_
    assert audit["schema"] == MAPPING_AUDIT_SCHEMA
    assert audit["mapping_sha256"] == EXPECTED_MAPPING_SHA256
    assert audit["support_mask_sha256"] == EXPECTED_SUPPORT_SHA256
    assert audit["supported_cartesian_cell_count"] == 1990
    assert audit["unsupported_cartesian_cell_count"] == 2106
    assert audit["unique_used_polar_bin_count"] == 1990
    assert audit["unused_polar_bin_count"] == 14394
    assert audit["collision_bin_count"] == 0
    assert audit["collisions"] == []
    assert audit["deterministic"] is True
    assert audit["injective"] is True
    assert audit["passes"] is True


def test_closed_fov_edges_and_half_open_range_are_executable() -> None:
    for sign, expected_bin in ((-1.0, 0), (1.0, 255)):
        forward = CAMERA_XYZ_BODY_M[0] + 1.0
        forward_camera = forward - CAMERA_XYZ_BODY_M[0]
        left = sign * forward_camera * math.tan(HALF_FOV_RAD)
        assert math.atan2(left, forward_camera) == sign * HALF_FOV_RAD
        assert camera_point_to_bin(forward, left) == (12, expected_bin)

        outside_bearing = math.nextafter(
            sign * HALF_FOV_RAD,
            math.copysign(math.inf, sign),
        )
        outside_left = forward_camera * math.tan(outside_bearing)
        assert camera_point_to_bin(forward, outside_left) is None

    assert camera_point_to_bin(CAMERA_XYZ_BODY_M[0] + 0.049, 0.0) is None
    assert camera_point_to_bin(CAMERA_XYZ_BODY_M[0] + 6.4, 0.0) is None
    assert camera_point_to_bin(CAMERA_XYZ_BODY_M[0] + 6.399, 0.0) == (63, 128)


@pytest.mark.parametrize("mutation", ("collision", "partial", "range", "dtype"))
def test_mapping_mutations_fail_closed_with_specific_diagnostics(mutation: str) -> None:
    mapping = build_camera_centered_mapping()
    supported = np.argwhere(np.all(mapping >= 0, axis=-1))
    if mutation == "collision":
        first = tuple(supported[0])
        second = tuple(supported[1])
        mapping[second] = mapping[first]
    elif mutation == "partial":
        row, column = supported[0]
        mapping[row, column, 0] = -1
    elif mutation == "range":
        row, column = supported[0]
        mapping[row, column, 0] = 64
    else:
        mapping = mapping.astype(np.int32)

    audit = audit_camera_centered_mapping(mapping)

    assert audit["passes"] is False
    if mutation == "collision":
        assert audit["collision_bin_count"] == 1
        assert audit["collision_extra_cartesian_count"] == 1
        assert audit["collisions"][0]["multiplicity"] == 2
        assert audit["injective"] is False
    elif mutation == "partial":
        assert audit["partially_mapped_entry_count"] == 1
        assert audit["all_entries_complete"] is False
    elif mutation == "range":
        assert audit["out_of_range_entry_count"] == 1
        assert audit["all_mapped_indices_in_range"] is False
    else:
        assert audit["signed_int16"] is False
        assert audit["nondeterministic_entry_count"] == 0


def test_old_body_column_span_audit_is_exact_and_large() -> None:
    audit = old_body_column_span_audit()
    primary = audit["primary"]
    horizontal = audit["horizontal_only"]

    assert audit["old_column_span_table_sha256"] == EXPECTED_SPAN_TABLE_SHA256
    assert len(primary["columns"]) == ANGULAR_BIN_COUNT
    assert primary == horizontal
    assert primary["summary"]["participating_sample_count"] == 13204
    assert primary["summary"]["columns_with_span_count"] == 244
    assert (
        primary["summary"]["columns_with_fewer_than_two_participants_count"]
        == 12
    )
    assert primary["summary"]["columns_span_ge_8_new_bins"] == 222
    assert primary["summary"]["span_new_angular_bins"]["p50"] == pytest.approx(
        41.05956247946459, rel=0.0, abs=1e-12
    )
    assert primary["summary"]["span_new_angular_bins"]["maximum"] == pytest.approx(
        86.03579205109838, rel=0.0, abs=1e-12
    )


def test_label_support_counts_and_exact_violation_identity() -> None:
    mapping = build_camera_centered_mapping()
    support = camera_centered_support_mask(mapping)
    target = _unknown_target()
    supported_cell = tuple(np.argwhere(support)[0])
    unsupported_cell = tuple(np.argwhere(~support)[0])
    target[supported_cell] = FREE_CLASS
    target[unsupported_cell] = OCCUPIED_CLASS

    report = _analyze(0, target)
    labels = report["label_support"]

    assert labels["total_supervised_label_count"] == 4096
    assert labels["supported_label_count"] == 1990
    assert labels["unsupported_label_count"] == 2106
    assert labels["class_counts"] == {
        "unknown": 4094,
        "free": 1,
        "occupied": 1,
    }
    assert labels["by_class"]["free"] == {
        "total": 1,
        "supported": 1,
        "unsupported": 0,
    }
    assert labels["unsupported_occupied_count"] == 1
    assert labels["passes"] is False
    assert labels["violations"] == [
        {
            "frame_key": _frame_key(0),
            "row": int(unsupported_cell[0]),
            "column": int(unsupported_cell[1]),
            "class_id": OCCUPIED_CLASS,
            "class_name": "occupied",
        }
    ]


@pytest.mark.parametrize(
    ("target", "mask", "message"),
    (
        (np.zeros((63, 64), dtype=np.uint8), _full_mask(), "target must have shape"),
        (
            np.zeros((64, 64), dtype=np.float32),
            _full_mask(),
            "integer dtype",
        ),
        (
            np.full((64, 64), 3, dtype=np.uint8),
            _full_mask(),
            "registered classes",
        ),
        (
            _unknown_target(),
            np.zeros((64, 64), dtype=np.uint8),
            "full 64 x 64",
        ),
        (
            _unknown_target(),
            np.full((64, 64), np.nan),
            "must be finite",
        ),
    ),
)
def test_label_and_supervision_contract_rejects_malformed_arrays(
    target: np.ndarray,
    mask: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        analyze_frame_labels(
            target,
            mask,
            frame_key=_frame_key(0),
            family=FAMILIES[0],
            endpoint_side=ENDPOINT_SIDES[0],
        )


def test_ray_sequences_capture_all_six_directions_and_scalar_incompatibility() -> None:
    mapping = build_camera_centered_mapping()
    target = _unknown_target()
    counts = np.bincount(
        mapping[..., 1][mapping[..., 1] >= 0], minlength=ANGULAR_BIN_COUNT
    )
    angular_bin = int(np.argmax(counts))
    locations = np.argwhere(mapping[..., 1] == angular_bin)
    ordered = sorted(
        (
            int(mapping[row, column, 0]),
            int(row),
            int(column),
        )
        for row, column in locations
    )
    assert len(ordered) >= 7
    sequence = (
        UNKNOWN_CLASS,
        FREE_CLASS,
        OCCUPIED_CLASS,
        UNKNOWN_CLASS,
        OCCUPIED_CLASS,
        FREE_CLASS,
        UNKNOWN_CLASS,
    )
    for (_, row, column), class_id in zip(ordered, sequence):
        target[row, column] = class_id

    report = analyze_frame_labels(
        target,
        _full_mask(),
        frame_key=_frame_key(0),
        family=FAMILIES[0],
        endpoint_side=ENDPOINT_SIDES[0],
        mapping=mapping,
    )
    ray = report["ray_sequences"]["records"][angular_bin]
    summary = report["ray_sequences"]["summary"]

    assert ray["class_sequence"][:7] == list(sequence)
    assert ray["collapsed_class_sequence"][:7] == list(sequence)
    assert ray["transition_count"] >= 6
    assert ray["directed_unequal_transition_counts"] == {
        "unknown_to_free": 1,
        "unknown_to_occupied": 1,
        "free_to_unknown": 1,
        "free_to_occupied": 1,
        "occupied_to_unknown": 1,
        "occupied_to_free": 1,
    }
    assert ray["contains_known_after_unknown"] is True
    assert ray["contains_free_after_occupied"] is True
    assert ray["scalar_first_hit_regular"] is False
    assert summary["sequence_count"] == 256
    assert summary["transition_bucket_counts"]["3_plus"] >= 1
    assert summary["contains_known_after_unknown_count"] >= 1
    assert summary["contains_free_after_occupied_count"] >= 1
    assert summary["scalar_first_hit_irregular_count"] >= 1


def test_aggregate_is_five_family_two_side_ordered_and_hash_sensitive() -> None:
    reports = _complete_reports()
    first = aggregate_label_observability(reports)
    second = aggregate_label_observability(copy.deepcopy(reports))

    assert first == second
    assert first["schema"] == OBSERVABILITY_SUMMARY_SCHEMA
    assert first["frame_count"] == 10
    assert tuple(first["families"]) == FAMILIES
    assert tuple(first["endpoint_sides"]) == ENDPOINT_SIDES
    assert all(first["families"][family]["frame_count"] == 2 for family in FAMILIES)
    assert all(first["endpoint_sides"][side]["frame_count"] == 5 for side in ENDPOINT_SIDES)
    assert first["aggregate"]["label_support"]["class_counts"] == {
        "unknown": 40960,
        "free": 0,
        "occupied": 0,
    }
    assert first["fit_known_target_coverage_gate"]["passes"] is True
    assert len(first["ordered_sequence_summary_records_sha256"]) == 64
    assert len(first["aggregate_transition_tables_sha256"]) == 64
    assert "records" not in first["aggregate"]["ray_sequences"]
    assert all(
        "records" not in first["families"][family]["ray_sequences"]
        for family in FAMILIES
    )
    assert all(
        "records" not in first["endpoint_sides"][side]["ray_sequences"]
        for side in ENDPOINT_SIDES
    )
    transient_records = [
        record
        for report in reports
        for record in report["ray_sequences"]["records"]
    ]
    assert first["ordered_sequence_summary_records_sha256"] == (
        canonical_json_sha256(transient_records)
    )

    reordered = aggregate_label_observability(list(reversed(reports)))
    assert (
        reordered["ordered_sequence_summary_records_sha256"]
        != first["ordered_sequence_summary_records_sha256"]
    )
    assert (
        reordered["aggregate_transition_tables_sha256"]
        == first["aggregate_transition_tables_sha256"]
    )


def test_aggregate_rejects_missing_families_sides_and_duplicate_keys() -> None:
    reports = _complete_reports()
    missing_family = [
        report
        for report in reports
        if report["family"] != "large_enclosed_maze"
    ]
    with pytest.raises(ValueError, match="every registered family"):
        aggregate_label_observability(missing_family)

    one_side = [copy.deepcopy(report) for report in reports]
    for report in one_side:
        report["endpoint_side"] = "current"
    with pytest.raises(ValueError, match="both endpoint sides"):
        aggregate_label_observability(one_side)

    duplicate = copy.deepcopy(reports)
    duplicate[1]["frame_key"] = duplicate[0]["frame_key"]
    with pytest.raises(ValueError, match="frame keys must be unique"):
        aggregate_label_observability(duplicate)


def test_authorization_is_strict_representation_only_and_ambiguity_is_separate() -> None:
    observability = aggregate_label_observability(_complete_reports())
    mapping_audit = audit_camera_centered_mapping()
    decision = authorization_decision(
        provenance_passes=True,
        source_hashes_pass=True,
        reconstruction_passes=True,
        access_reconciliation_passes=True,
        mapping_audit=mapping_audit,
        label_observability=observability,
        rendered_collision_target_ambiguity=True,
    )

    assert decision["schema"] == AUTHORIZATION_SCHEMA
    assert decision["camera_frustum_representation_implementation_authorized"] is True
    assert decision["target_amendment_required_before_model_output"] is True
    assert decision["trained_model_output_authorized"] is False
    assert decision["holdout_access_authorized"] is False
    assert decision["seed_20260711_authorized"] is False
    assert decision["g2_authorized"] is False
    assert decision["runtime_authorized"] is False
    assert decision["promotion_authorized"] is False

    failed_mapping = copy.deepcopy(mapping_audit)
    failed_mapping["passes"] = False
    failed = authorization_decision(
        provenance_passes=True,
        source_hashes_pass=True,
        reconstruction_passes=True,
        access_reconciliation_passes=True,
        mapping_audit=failed_mapping,
        label_observability=observability,
        rendered_collision_target_ambiguity=False,
    )
    assert failed["camera_frustum_representation_implementation_authorized"] is False
    assert failed["target_amendment_required_before_model_output"] is False


def test_authorization_rejects_non_boolean_and_wrong_schema_inputs() -> None:
    observability = aggregate_label_observability(_complete_reports())
    mapping_audit = audit_camera_centered_mapping()
    with pytest.raises(TypeError, match="provenance_passes"):
        authorization_decision(
            provenance_passes=1,  # type: ignore[arg-type]
            source_hashes_pass=True,
            reconstruction_passes=True,
            access_reconciliation_passes=True,
            mapping_audit=mapping_audit,
            label_observability=observability,
            rendered_collision_target_ambiguity=False,
        )

    wrong = copy.deepcopy(observability)
    wrong["schema"] = "wrong"
    with pytest.raises(ValueError, match="observability summary"):
        authorization_decision(
            provenance_passes=True,
            source_hashes_pass=True,
            reconstruction_passes=True,
            access_reconciliation_passes=True,
            mapping_audit=mapping_audit,
            label_observability=wrong,
            rendered_collision_target_ambiguity=False,
        )


def test_canonical_json_hash_rejects_nonfinite_and_is_key_order_invariant() -> None:
    assert canonical_json_sha256({"a": 1, "b": [2, 3]}) == canonical_json_sha256(
        {"b": [2, 3], "a": 1}
    )
    with pytest.raises(ValueError):
        canonical_json_sha256({"invalid": float("nan")})
