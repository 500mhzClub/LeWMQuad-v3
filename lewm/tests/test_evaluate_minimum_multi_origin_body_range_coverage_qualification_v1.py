from __future__ import annotations

import importlib.util
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_minimum_multi_origin_body_range_coverage_qualification_v1",
    ROOT / "scripts/evaluate_minimum_multi_origin_body_range_coverage_qualification_v1.py",
)
assert SPEC is not None and SPEC.loader is not None
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


def _evidence(support: np.ndarray, clearance: float) -> dict[str, np.ndarray]:
    shape = (EVALUATOR.STEPS, EVALUATOR.LINKS)
    supported = np.asarray(support, bool)
    output: dict[str, np.ndarray] = {
        "clearance_m": np.where(supported, clearance, np.inf),
        "support": supported,
        "event_time_support": supported,
        "point_support_count": supported.astype(np.int16),
        "nominal_fov": np.ones(shape, bool),
        "horizontal_fov": np.ones(shape, bool),
        "vertical_fov": np.ones(shape, bool),
        "direct_visibility": supported,
        "self_occluded": ~supported,
        "environment_occluded": np.zeros(shape, bool),
        "near_blind": np.zeros(shape, bool),
        "finite_scan_support_inherited": np.zeros(shape, bool),
        "nearest_ray_index": np.where(supported, 7, -1).astype(np.int32),
        "support_nearest_ray_index": np.where(supported, 7, -1).astype(np.int32),
        "responsible_geom_index": np.zeros(shape, np.int16),
    }
    for field in EVALUATOR.BASE.FLOAT_EVIDENCE_FIELDS:
        output.setdefault(field, np.full(shape, 0.1, np.float64))
    for field in EVALUATOR.BASE.INT16_EVIDENCE_FIELDS:
        output.setdefault(field, np.zeros(shape, np.int16))
    return output


def test_multi_origin_union_preserves_complementary_support_and_provenance() -> None:
    left = np.zeros((EVALUATOR.STEPS, EVALUATOR.LINKS), bool)
    right = np.zeros_like(left)
    left[:, ::2] = True
    right[:, 1::2] = True
    merged = EVALUATOR._merge_origin_evidence(
        (
            ("HEAD_STOCK", _evidence(left, 0.2)),
            ("REAR_TOP_TRUNK", _evidence(right, 0.3)),
        )
    )
    assert merged["support"].all()
    assert np.all(merged["supporting_origin_count"] == 1)
    assert np.all(merged["supporting_origin_bitmask"][:, ::2] == 1)
    assert np.all(merged["supporting_origin_bitmask"][:, 1::2] == 2)
    assert np.all(merged["responsible_origin_index"][:, ::2] == 0)
    assert np.all(merged["responsible_origin_index"][:, 1::2] == 1)


def test_multi_origin_union_marks_only_all_origin_gaps_unsupported() -> None:
    support = np.ones((EVALUATOR.STEPS, EVALUATOR.LINKS), bool)
    support[3, 4] = False
    other = np.zeros_like(support)
    other[3, 4] = True
    merged = EVALUATOR._merge_origin_evidence(
        (
            ("HEAD_STOCK", _evidence(support, 0.2)),
            ("LEFT_UPPER_FLANK", _evidence(other, 0.1)),
        )
    )
    assert merged["support"].all()
    assert merged["clearance_m"][3, 4] == 0.1


def test_condition_execution_maps_only_to_the_selected_count_layout() -> None:
    selection = {
        "selected_dual_layout": "HEAD_STOCK__REAR_TOP_TRUNK",
        "selected_three_origin_layout": (
            "HEAD_STOCK__LEFT_UPPER_FLANK__RIGHT_UPPER_FLANK"
        ),
    }
    dual = EVALUATOR._condition_spec(
        "DUAL_REALISTIC_L2_SCAN", selection
    )
    triple = EVALUATOR._condition_spec(
        "THREE_DENSE_L2_FOV_UPPER_BOUND", selection
    )
    assert dual[0] == selection["selected_dual_layout"]
    assert dual[1] == ("HEAD_STOCK", "REAR_TOP_TRUNK")
    assert dual[2] == "REALISTIC"
    assert triple[0] == selection["selected_three_origin_layout"]
    assert triple[2] == "DENSE_L2_FOV"


def test_report_regions_are_the_frozen_eight_over_thirteen_links() -> None:
    names = (
        "base",
        "FL_hip",
        "FR_hip",
        "RL_hip",
        "RR_hip",
        "FL_thigh",
        "FR_thigh",
        "RL_thigh",
        "RR_thigh",
        "FL_calf",
        "FR_calf",
        "RL_calf",
        "RR_calf",
    )
    masks = EVALUATOR._report_region_masks(names)
    assert tuple(masks) == (
        "trunk",
        "front_left_limb",
        "front_right_limb",
        "rear_left_limb",
        "rear_right_limb",
        "hips",
        "thighs",
        "calves",
    )
    assert all(mask.shape == (13,) and mask.any() for mask in masks.values())


def test_gzip_jsonl_helpers_are_byte_deterministic(tmp_path: Path) -> None:
    paths = [tmp_path / "a.jsonl.gz", tmp_path / "b.jsonl.gz"]
    for path in paths:
        temporary, raw, stream = EVALUATOR._gzip_jsonl_writer(path)
        stream.write(EVALUATOR.canonical_bytes({"z": 1, "a": 2}))
        EVALUATOR._finalize_gzip_jsonl(path, temporary, raw, stream)
    assert paths[0].read_bytes() == paths[1].read_bytes()


def test_emitting_housing_is_exempt_and_other_installed_housing_occludes() -> None:
    base_position = np.asarray([[0.1, -0.2, 0.3]], np.float64)
    base_quaternion = np.asarray([[1.0, 0.0, 0.0, 0.0]], np.float64)
    empty_position = np.empty((1, 0, 3), np.float64)
    empty_quaternion = np.empty((1, 0, 4), np.float64)
    specs, positions, quaternions = EVALUATOR._append_other_sensor_housings(
        emitting_mount_id="HEAD_STOCK",
        installed_mount_ids=("HEAD_STOCK", "LEFT_UPPER_FLANK"),
        specs=(),
        positions=empty_position,
        quaternions=empty_quaternion,
        base_positions=base_position,
        base_quaternions=base_quaternion,
    )
    assert [spec.identity for spec in specs] == [
        "installed_sensor_housing:LEFT_UPPER_FLANK"
    ]
    assert specs[0].data == (0.075, 0.075, 0.065)
    assert positions.shape == (1, 1, 3)
    # The frozen coarse mechanical envelope is trunk-axis-aligned even though
    # the independently selected optical frame is inward/downward.
    assert np.array_equal(quaternions[:, 0], base_quaternion)

    own_specs, own_positions, _own_quaternions = (
        EVALUATOR._append_other_sensor_housings(
            emitting_mount_id="LEFT_UPPER_FLANK",
            installed_mount_ids=("HEAD_STOCK", "LEFT_UPPER_FLANK"),
            specs=(),
            positions=empty_position,
            quaternions=empty_quaternion,
            base_positions=base_position,
            base_quaternions=base_quaternion,
        )
    )
    assert own_specs == []
    assert own_positions.shape == (1, 0, 3)


def test_schema_key_validator_is_fail_closed() -> None:
    EVALUATOR._require_schema_keys({"a": 1, "b": 2}, ("a", "b"), "fixture")
    try:
        EVALUATOR._require_schema_keys({"a": 1}, ("a", "b"), "fixture")
    except RuntimeError as error:
        assert "b" in str(error)
    else:  # pragma: no cover - explicit fail-closed assertion
        raise AssertionError("missing schema key did not fail")


def test_dense_robot_rows_recover_exact_frozen_base_pose() -> None:
    boundary_qpos = np.zeros(19, np.float64)
    boundary_qpos[3] = 1.0
    qpos = np.repeat(boundary_qpos[None], EVALUATOR.STEPS, axis=0)
    qpos[:, 0] = np.arange(1, EVALUATOR.STEPS + 1) * 0.002
    boundary_geom = np.zeros((27, 7), np.float64)
    boundary_geom[:, 3] = 1.0
    geom = np.repeat(boundary_geom[None], EVALUATOR.STEPS, axis=0)
    geom[:, 0, 0] = qpos[:, 0]
    query_indices = np.asarray([0, 3, 3, 50], np.int16)
    all_geom = np.concatenate((boundary_geom[None], geom), axis=0)
    robot_positions = np.repeat(all_geom[query_indices, :, :3], 2, axis=1)
    robot_quaternions = np.repeat(all_geom[query_indices, :, 3:], 2, axis=1)
    # Restore the frozen geometry dimension after deliberately repeating rows.
    robot_positions = robot_positions[:, :27]
    robot_quaternions = robot_quaternions[:, :27]
    base_position, base_quaternion = EVALUATOR._base_pose_rows_for_robot_pose_rows(
        robot_positions,
        robot_quaternions,
        boundary_qpos=boundary_qpos,
        boundary_geom_transform=boundary_geom,
        qpos=qpos,
        geom_transform=geom,
    )
    all_qpos = np.concatenate((boundary_qpos[None], qpos), axis=0)
    assert np.array_equal(base_position, all_qpos[query_indices, :3])
    assert np.array_equal(base_quaternion, all_qpos[query_indices, 3:7])


def test_distinct_transition_uids_force_distinct_realistic_scans(monkeypatch) -> None:
    calls: list[str] = []

    def fake_origin(**kwargs):
        uid = str(kwargs["transition_uid"])
        calls.append(uid)
        evidence = _evidence(
            np.ones((EVALUATOR.STEPS, EVALUATOR.LINKS), bool),
            0.1 if uid == "uid-a" else 0.2,
        )
        phase = EVALUATOR.CONTRACT.derive_multi_origin_scan_phases(
            contract_digest_sha256="12" * 32,
            transition_uid=uid,
            mount_id=str(kwargs["mount_id"]),
        )
        cloud = {
            "phase": phase,
            "ray_count": 6400,
            "environment_return_count": 1,
            "self_return_count": 0,
            "near_blind_count": 0,
            "ground_return_count": 0,
        }
        dense = _evidence(
            np.ones((EVALUATOR.STEPS, EVALUATOR.LINKS), bool), 0.05
        )
        dense["environment_occluded"] = np.zeros_like(dense["support"])
        dense["horizontal_fov"] = np.ones_like(dense["support"])
        dense["vertical_fov"] = np.ones_like(dense["support"])
        dense["near_blind"] = np.zeros_like(dense["support"])
        return evidence, cloud, dense, 0

    monkeypatch.setattr(EVALUATOR, "_origin_realistic_evidence", fake_origin)
    common = {
        "condition_id": "DUAL_REALISTIC_L2_SCAN",
        "layout_id": "HEAD_STOCK__REAR_TOP_TRUNK",
        "mounts": ("HEAD_STOCK",),
        "mode_id": "TRUE_FUTURE_OBSERVABILITY_CLOUD",
        "transition_index": 3,
        "action_representative_transition_index": 3,
        "geometry_representative_transition_index": 3,
        "boundary_qpos": np.zeros(19),
        "boundary_geom_transform": np.zeros((27, 7)),
        "qpos": np.zeros((EVALUATOR.STEPS, 19)),
        "geom_transform": np.zeros((EVALUATOR.STEPS, 27, 7)),
        "environment_boxes": [],
        "robot_specs": [],
        "targets": {},
        "matched_dense_l2_by_mount": {
            "HEAD_STOCK": _evidence(
                np.ones((EVALUATOR.STEPS, EVALUATOR.LINKS), bool), 0.05
            )
        },
    }
    first = EVALUATOR._realistic_layout_evidence(
        **common, transition_uid="uid-a"
    )
    second = EVALUATOR._realistic_layout_evidence(
        **common, transition_uid="uid-b"
    )
    assert calls == ["uid-a", "uid-b"]
    assert first[0]["clearance_m"][0, 0] != second[0]["clearance_m"][0, 0]
    assert (
        first[3][0]["phase"]["phase_digest_sha256"]
        != second[3][0]["phase"]["phase_digest_sha256"]
    )
    assert first[3][0]["render_cache_reused"] is False
    assert second[3][0]["render_cache_reused"] is False
    selection = {
        "selected_dual_layout": "HEAD_STOCK__REAR_TOP_TRUNK",
        "selected_three_origin_layout": (
            "HEAD_STOCK__LEFT_UPPER_FLANK__RIGHT_UPPER_FLANK"
        ),
    }
    assert not EVALUATOR._condition_allows_geometry_evidence_copy(
        "DUAL_REALISTIC_L2_SCAN", selection
    )
    assert EVALUATOR._condition_allows_geometry_evidence_copy(
        "DUAL_DENSE_L2_FOV_UPPER_BOUND", selection
    )


def test_uid_bound_scan_cardinality_is_fail_closed() -> None:
    selection = {
        "selected_dual_layout": "HEAD_STOCK__REAR_TOP_TRUNK",
        "selected_three_origin_layout": (
            "HEAD_STOCK__LEFT_UPPER_FLANK__RIGHT_UPPER_FLANK"
        ),
    }
    scans = EVALUATOR.EXPECTED["transitions"] * 2
    counts = {
        "DUAL_REALISTIC_L2_SCAN": {
            mode: {
                "sensor_scans": scans,
                "unique_rendered_scans": scans,
                "rays": scans * 6400,
            }
            for mode in EVALUATOR.MODES
        }
    }
    EVALUATOR._validate_uid_bound_scan_counts(
        counts, tuple(EVALUATOR.CONTRACT.DUAL_CONDITION_IDS), selection
    )
    counts["DUAL_REALISTIC_L2_SCAN"][EVALUATOR.MODES[0]]["sensor_scans"] -= 1
    try:
        EVALUATOR._validate_uid_bound_scan_counts(
            counts, tuple(EVALUATOR.CONTRACT.DUAL_CONDITION_IDS), selection
        )
    except RuntimeError as error:
        assert "UID-bound" in str(error)
    else:  # pragma: no cover
        raise AssertionError("phase scan cardinality drift did not fail")


def test_state_scan_receipts_bind_frozen_uid_and_phase(monkeypatch) -> None:
    selection = {
        "selected_dual_layout": "HEAD_STOCK__REAR_TOP_TRUNK",
        "selected_three_origin_layout": (
            "HEAD_STOCK__LEFT_UPPER_FLANK__RIGHT_UPPER_FLANK"
        ),
    }
    uids = ("fixture-state:0", "fixture-state:1")
    monkeypatch.setattr(EVALUATOR, "_frozen_transition_uids", lambda _state: uids)
    monkeypatch.setattr(EVALUATOR, "_contract_content_digest", lambda: "12" * 32)
    rows = []
    for transition_index, uid in enumerate(uids):
        for mode_id in EVALUATOR.MODES:
            for mount_id in ("HEAD_STOCK", "REAR_TOP_TRUNK"):
                rows.append(
                    {
                        "transition_index": transition_index,
                        "transition_uid": uid,
                        "condition_id": "DUAL_REALISTIC_L2_SCAN",
                        "evidence_mode": mode_id,
                        "layout_id": selection["selected_dual_layout"],
                        "mount_id": mount_id,
                        "ray_count": 6400,
                        "render_cache_reused": False,
                        "dense_l2_inherited_support_witnesses": 0,
                        "dense_spherical_inherited_support_witnesses": 0,
                        "phase": EVALUATOR.CONTRACT.derive_multi_origin_scan_phases(
                            contract_digest_sha256=EVALUATOR._contract_content_digest(),
                            transition_uid=uid,
                            mount_id=mount_id,
                        ),
                    }
                )
    receipt = {
        "state_id": "fixture-state",
        "transitions": 2,
        "transition_uid_by_index": list(uids),
        "transition_uid_by_index_sha256": EVALUATOR.content_digest(list(uids)),
        "scan_receipts": rows,
    }
    keys = EVALUATOR._validate_state_scan_receipts(
        receipt, tuple(EVALUATOR.CONTRACT.DUAL_CONDITION_IDS), selection
    )
    assert len(keys) == 8
    rows[0]["phase"] = dict(rows[0]["phase"])
    rows[0]["phase"]["horizontal_phase_cycles"] = 0.0
    try:
        EVALUATOR._validate_state_scan_receipts(
            receipt, tuple(EVALUATOR.CONTRACT.DUAL_CONDITION_IDS), selection
        )
    except RuntimeError as error:
        assert "phase" in str(error)
    else:  # pragma: no cover
        raise AssertionError("tampered UID phase did not fail")


def test_condition_npz_schema_checks_required_shape_and_dtype(tmp_path: Path) -> None:
    transitions = 2
    selection = {
        "selected_dual_layout": "HEAD_STOCK__REAR_TOP_TRUNK",
        "selected_three_origin_layout": (
            "HEAD_STOCK__LEFT_UPPER_FLANK__RIGHT_UPPER_FLANK"
        ),
    }
    fused = (transitions, len(EVALUATOR.MODES), EVALUATOR.STEPS, EVALUATOR.LINKS)
    origin = (
        transitions,
        len(EVALUATOR.MODES),
        2,
        EVALUATOR.STEPS,
        EVALUATOR.LINKS,
    )
    arrays = {
        "action_representative_transition": np.arange(transitions, dtype=np.int32),
        "geometry_representative_transition": np.arange(transitions, dtype=np.int32),
        "support": np.zeros(fused, np.uint8),
        "finite_scan_support_inherited": np.zeros(fused, np.uint8),
        "per_origin_support": np.zeros(origin, np.uint8),
        "per_origin_event_time_support": np.zeros(origin, np.uint8),
        "per_origin_nominal_fov": np.zeros(origin, np.uint8),
        "per_origin_direct_visibility": np.zeros(origin, np.uint8),
        "per_origin_self_occluded": np.zeros(origin, np.uint8),
        "per_origin_finite_scan_support_inherited": np.zeros(origin, np.uint8),
        "per_origin_point_support_count": np.zeros(origin, np.int16),
        "per_origin_support_acquisition_index": np.zeros(origin, np.int16),
        "per_origin_nearest_ray_index": np.zeros(origin, np.int32),
        "per_origin_point_age_s": np.zeros(origin, np.float32),
    }
    path = tmp_path / "valid.npz"
    np.savez_compressed(path, **arrays)
    record = {
        "condition_id": "DUAL_DENSE_L2_FOV_UPPER_BOUND",
        "transitions": transitions,
        "shard_path": str(path),
    }
    EVALUATOR._validate_condition_npz_schema(record, selection)
    arrays["per_origin_point_age_s"] = np.zeros(origin, np.float64)
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, **arrays)
    record["shard_path"] = str(bad)
    try:
        EVALUATOR._validate_condition_npz_schema(record, selection)
    except RuntimeError as error:
        assert "shape/dtype" in str(error)
    else:  # pragma: no cover
        raise AssertionError("wrong NPZ dtype did not fail")


def test_training_layout_ledger_roundtrip_and_tamper_detection(
    tmp_path: Path, monkeypatch
) -> None:
    link_names = (
        "base",
        "FL_hip",
        "FL_thigh",
        "FL_calf",
        "FR_hip",
        "FR_thigh",
        "FR_calf",
        "RL_hip",
        "RL_thigh",
        "RL_calf",
        "RR_hip",
        "RR_thigh",
        "RR_calf",
    )
    region_totals = {
        "TRUNK": 50,
        "FRONT_LIMBS": 300,
        "REAR_LIMBS": 300,
        "HIPS_AND_THIGHS": 400,
        "CALVES": 200,
    }
    layout_ids = tuple(EVALUATOR.CONTRACT.PAIR_LAYOUT_IDS) + tuple(
        EVALUATOR.CONTRACT.THREE_LAYOUT_IDS
    )
    layout_row = {
        "transition_support": [1.0],
        "transition_supported_count": [650],
        "transition_total_count": [650],
        "transition_nominal_count": [650],
        "transition_self_occluded_count": [0],
        "transition_region_supported_count": {
            key: [value] for key, value in region_totals.items()
        },
        "transition_region_total_count": {
            key: [value] for key, value in region_totals.items()
        },
        "transition_link_supported_count": [[50] * 13],
        "transition_link_total_count": [[50] * 13],
        "link_supported": [50] * 13,
        "link_total": [50] * 13,
        "supported": 650,
        "total": 650,
        "nominal": 650,
        "self_occluded": 0,
        "region_supported": dict(region_totals),
        "region_total": dict(region_totals),
    }
    records = {
        "fixture-state": {
            "state_id": "fixture-state",
            "family": "fixture-family",
            "protected_link_names": list(link_names),
            "transition_rows": [
                {
                    "transition_index": 0,
                    "level": "current",
                    "current_action_index": 0,
                    "action_index": 0,
                }
            ],
            "layouts": {key: dict(layout_row) for key in layout_ids},
        }
    }
    monkeypatch.setattr(EVALUATOR, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setitem(EVALUATOR.EXPECTED, "training_states", 1)
    binding = EVALUATOR._persist_training_layout_evidence(
        records, ("fixture-state",), layout_ids
    )
    score = {
        "layout_id": "",
        "minimum_body_region_support": 1.0,
        "transition_support_p05": 1.0,
        "rear_limb_support": 1.0,
        "calf_support": 1.0,
        "overall_mean_support": 1.0,
        "self_occluded_fraction": 0.0,
        "minimum_family_support": 1.0,
        "region_support": {key: 1.0 for key in region_totals},
        "per_link_support": {key: 1.0 for key in link_names},
        "per_link_counts": {
            key: {"supported_witnesses": 50, "total_witnesses": 50}
            for key in link_names
        },
        "per_family_support": {"fixture-family": 1.0},
        "transition_count": 1,
    }
    scores = {
        layout_id: {**score, "layout_id": layout_id} for layout_id in layout_ids
    }
    selection = {
        "scores": scores,
        "selected_dual_layout": EVALUATOR.CONTRACT.PAIR_LAYOUT_IDS[0],
        "selected_three_origin_layout": EVALUATOR.CONTRACT.THREE_LAYOUT_IDS[0],
    }
    EVALUATOR._validate_training_layout_evidence_rows(binding, selection)
    tampered = dict(binding)
    tampered["role"] = "heldout"
    tampered_core = dict(tampered)
    tampered_core.pop("content_digest")
    tampered["content_digest"] = EVALUATOR.content_digest(tampered_core)
    try:
        EVALUATOR._validate_training_layout_evidence_binding(tampered)
    except RuntimeError as error:
        assert "binding drift" in str(error)
    else:  # pragma: no cover
        raise AssertionError("tampered training-layout binding did not fail")


def test_layout_reuse_is_exact_geometry_only_and_requires_no_contact_arrays() -> None:
    qpos = np.zeros((3, 2, 19), np.float32)
    link = np.zeros((3, 2, 13, 7), np.float32)
    geom = np.zeros((3, 2, 27, 7), np.float32)
    # Row 1 is byte-identical to row 0.  Row 2 differs geometrically despite
    # sharing the same deployable-action partition and boundary digest.
    qpos[2, 0, 0] = np.float32(1e-6)
    shard = SimpleNamespace(
        transition_count=3,
        arrays={
            "qpos": qpos,
            "link_transform": link,
            "geom_transform": geom,
        },
    )
    action_map = SimpleNamespace(copies_by_representative={0: (0, 1, 2)})
    groups = EVALUATOR._build_label_free_layout_reuse_groups(
        shard,
        action_map,
        ("same-boundary", "same-boundary", "same-boundary"),
    )
    assert groups == {0: (0, 1), 2: (2,)}
    assert EVALUATOR.LABEL_FREE_LAYOUT_REUSE_FIELDS == (
        "qpos",
        "link_transform",
        "geom_transform",
    )


def test_layout_reuse_never_crosses_boundary_or_action_partition() -> None:
    arrays = {
        "qpos": np.zeros((3, 1, 19), np.float32),
        "link_transform": np.zeros((3, 1, 13, 7), np.float32),
        "geom_transform": np.zeros((3, 1, 27, 7), np.float32),
    }
    shard = SimpleNamespace(transition_count=3, arrays=arrays)
    action_map = SimpleNamespace(copies_by_representative={0: (0, 1), 2: (2,)})
    groups = EVALUATOR._build_label_free_layout_reuse_groups(
        shard,
        action_map,
        ("boundary-a", "boundary-b", "boundary-a"),
    )
    assert groups == {0: (0,), 1: (1,), 2: (2,)}


def test_sparse_realistic_evidence_is_enriched_by_matched_dense_attribution(
    monkeypatch,
) -> None:
    support = np.zeros((EVALUATOR.STEPS, EVALUATOR.LINKS), bool)
    support[4, 7] = True
    sparse = _evidence(support, 0.12)
    for field in (
        "nominal_fov",
        "horizontal_fov",
        "vertical_fov",
        "direct_visibility",
        "self_occluded",
        "environment_occluded",
        "near_blind",
    ):
        sparse.pop(field)
    cloud = {
        "points": np.zeros((1, 3), np.float64),
        "object_index": np.zeros(1, np.int16),
        "ray_index": np.zeros(1, np.int32),
        "point_time_s": np.zeros(1),
        "point_range_m": np.ones(1),
        "ray_count": 6400,
        "environment_return_count": 1,
        "self_return_count": 0,
        "near_blind_count": 0,
        "ground_return_count": 0,
    }
    monkeypatch.setattr(EVALUATOR, "_render_realistic_origin", lambda **_kwargs: cloud)
    monkeypatch.setattr(
        EVALUATOR.BASE, "sparse_per_link_evidence", lambda **_kwargs: sparse
    )
    dense = _evidence(np.zeros_like(support), np.inf)
    values, _cloud, inherited_dense, inherited = (
        EVALUATOR._origin_realistic_evidence(
            mount_id="HEAD_STOCK",
            mode_id="TRUE_FUTURE_OBSERVABILITY_CLOUD",
            transition_uid="fixture-uid",
            boundary_qpos=np.zeros(19),
            boundary_geom_transform=np.zeros((27, 7)),
            qpos=np.zeros((EVALUATOR.STEPS, 19)),
            geom_transform=np.zeros((EVALUATOR.STEPS, 27, 7)),
            environment_boxes=[],
            robot_specs=[],
            targets={"geom_index": np.zeros((EVALUATOR.STEPS, EVALUATOR.LINKS))},
            installed_mount_ids=("HEAD_STOCK",),
            matched_dense_l2_evidence=dense,
        )
    )
    assert inherited == 1
    assert values["nominal_fov"].all()
    assert inherited_dense["support"][4, 7]
    assert inherited_dense["finite_scan_support_inherited"][4, 7]
    merged = EVALUATOR._merge_origin_evidence((("HEAD_STOCK", values),))
    assert merged["support"][4, 7]


def test_support_hierarchy_unions_uid_phases_and_is_clearance_monotone() -> None:
    first_support = np.zeros((EVALUATOR.STEPS, EVALUATOR.LINKS), bool)
    second_support = np.zeros_like(first_support)
    first_support[1, 2] = True
    second_support[3, 4] = True
    first_realistic = _evidence(first_support, 0.2)
    second_realistic = _evidence(second_support, 0.1)
    first_l2 = _evidence(first_support, 0.2)
    second_l2 = _evidence(second_support, 0.1)
    first_l2["finite_scan_support_inherited"][1, 2] = True
    second_l2["finite_scan_support_inherited"][3, 4] = True
    union_l2 = _evidence(first_support | second_support, 0.15)
    union_l2["clearance_m"][1, 2] = 0.2
    union_l2["clearance_m"][3, 4] = 0.1
    union_l2["finite_scan_support_inherited"] = first_support | second_support
    sphere = _evidence(np.zeros_like(first_support), np.inf)
    records = {
        ("DUAL_REALISTIC_L2_SCAN", EVALUATOR.MODES[1], 0): (
            [("HEAD_STOCK", first_realistic)],
            [("HEAD_STOCK", first_l2)],
            [{"dense_l2_inherited_support_witnesses": 1}],
        ),
        ("DUAL_REALISTIC_L2_SCAN", EVALUATOR.MODES[1], 1): (
            [("HEAD_STOCK", second_realistic)],
            [("HEAD_STOCK", second_l2)],
            [{"dense_l2_inherited_support_witnesses": 1}],
        ),
    }
    dense_l2, dense_sphere, summary = (
        EVALUATOR._finalize_support_hierarchy_for_geometry_group(
            realistic_condition_id="DUAL_REALISTIC_L2_SCAN",
            mode_id=EVALUATOR.MODES[1],
            copies=(0, 1),
            realistic_records=records,
            dense_l2_uid_union=[("HEAD_STOCK", union_l2)],
            dense_spherical_base=[("HEAD_STOCK", sphere)],
        )
    )
    assert dense_l2[0][1]["support"].sum() == 2
    assert dense_sphere[0][1]["support"].sum() == 2
    assert summary["pass"]
    assert summary["per_origin_subset_violations"] == 0
    assert summary["clearance_monotonicity_violations"] == 0
    assert records[("DUAL_REALISTIC_L2_SCAN", EVALUATOR.MODES[1], 0)][2][0][
        "dense_spherical_inherited_support_witnesses"
    ] == 1


def test_support_hierarchy_replaces_overlapping_larger_clearance() -> None:
    support = np.zeros((EVALUATOR.STEPS, EVALUATOR.LINKS), bool)
    support[8, 6] = True
    upper = _evidence(support, 0.30)
    lower = _evidence(support, 0.08)
    inherited = EVALUATOR._inherit_supported_evidence(
        upper, lower, source_is_finite_scan=True
    )
    # The witness was already supported, so this is not a newly supported
    # cell, but the union must retain the lower clearance and its finite-scan
    # provenance to preserve the upper-bound monotonicity contract.
    assert inherited == 0
    assert upper["clearance_m"][8, 6] == 0.08
    assert upper["finite_scan_support_inherited"][8, 6]
    assert upper["support"][8, 6]


def test_execution_plan_remains_bound_to_freeze_commit_after_head_moves(
    tmp_path: Path,
) -> None:
    materialization = tmp_path / "dual_index.json"
    threshold = tmp_path / "dual_thresholds.json"
    materialization.write_bytes(b"materialization\n")
    threshold.write_bytes(b"thresholds\n")
    selection = {
        "selected_dual_layout": "HEAD_STOCK__REAR_TOP_TRUNK",
        "selected_three_origin_layout": (
            "HEAD_STOCK__REAR_TOP_TRUNK__LEFT_UPPER_FLANK"
        ),
    }
    dual_pass = True
    executed = list(EVALUATOR.CONTRACT.DUAL_CONDITION_IDS)
    skipped = {
        condition_id: list(EVALUATOR.MODES)
        for condition_id in tuple(EVALUATOR.CONTRACT.THREE_CONDITION_IDS)
        + (EVALUATOR.DIAGNOSTIC_CONDITION,)
    }
    receipt = {
        "schema": "minimum_multi_origin_conditional_execution_plan_v1",
        "experiment_id": EVALUATOR.EXPERIMENT,
        "source_freeze_commit": "frozen-source-commit",
        "selected_dual_layout_id": selection["selected_dual_layout"],
        "selected_three_layout_id": selection["selected_three_origin_layout"],
        "dual_realistic_pass": dual_pass,
        "regression_complete": True,
        "dual_complete": True,
        "dual_realistic_gate_pass": dual_pass,
        "three_required": False,
        "three_origin_executed": False,
        "three_complete": False,
        "all_four_diagnostic_required": False,
        "all_four_diagnostic_executed": False,
        "all_four_diagnostic_complete": False,
        "executed_condition_ids": executed,
        "phase_order": ["dual"],
        "executed_condition_modes": {
            condition_id: list(EVALUATOR.MODES) for condition_id in executed
        },
        "skipped_condition_modes": skipped,
        "materialization_indices": {
            "dual": {
                "path": str(materialization),
                "sha256": EVALUATOR.sha256_file(materialization),
            }
        },
        "threshold_freezes": {
            "dual": {
                "path": str(threshold),
                "sha256": EVALUATOR.sha256_file(threshold),
            }
        },
        "conditional_rule": "fixture",
        "conditional_flow_valid": True,
        "pass": True,
    }
    receipt["content_digest"] = EVALUATOR.content_digest(receipt)
    gates = {"DUAL_REALISTIC_L2_SCAN": {"pass": True}}

    EVALUATOR._validate_execution_plan_receipt(
        receipt,
        gates,
        selection,
        expected_source_freeze_commit="frozen-source-commit",
    )
    try:
        EVALUATOR._validate_execution_plan_receipt(
            receipt,
            gates,
            selection,
            expected_source_freeze_commit="later-result-commit",
        )
    except RuntimeError as error:
        assert "execution-plan" in str(error)
    else:  # pragma: no cover
        raise AssertionError("execution plan accepted a different source freeze commit")
