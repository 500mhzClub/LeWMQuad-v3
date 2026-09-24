import inspect
import math

import numpy as np
import pytest

from lewm.safety import body_centric_range_coverage_v1 as geometry
from lewm.safety import minimum_multi_origin_body_range_coverage_v1 as subject


IDENTITY = (1.0, 0.0, 0.0, 0.0)


def _primitive(
    identity: str,
    kind: str,
    data: tuple[float, ...],
    position: tuple[float, float, float],
    *,
    geom_index: int,
    link_name: str,
) -> geometry.RobotPrimitive:
    return geometry.RobotPrimitive(
        identity=identity,
        kind=kind,
        data=data,
        position_xyz_m=position,
        quaternion_wxyz=IDENTITY,
        geom_index=geom_index,
        link_index=geom_index,
        link_name=link_name,
    )


def _nominal_geometry() -> tuple[tuple[geometry.RobotPrimitive, ...], tuple[geometry.RobotPrimitive, ...]]:
    trunk = _primitive(
        "base:00",
        "box",
        (0.3762, 0.0935, 0.114),
        (0.0, 0.0, 0.0),
        geom_index=0,
        link_name="base",
    )
    protected = (
        trunk,
        _primitive("FL_hip", "sphere", (0.03,), (0.20, 0.08, -0.10), geom_index=3, link_name="FL_hip"),
        _primitive("RL_thigh", "sphere", (0.03,), (-0.20, 0.08, -0.10), geom_index=4, link_name="RL_thigh"),
        _primitive("FL_calf", "capsule", (0.025, 0.15), (0.20, 0.08, -0.25), geom_index=5, link_name="FL_calf"),
        _primitive("RR_calf", "capsule", (0.025, 0.15), (-0.20, -0.08, -0.25), geom_index=6, link_name="RR_calf"),
    )
    return (trunk,), protected


def _origin(
    mount: str,
    support: tuple[bool, ...],
    *,
    event: tuple[float, ...] | None = None,
    timestamp: tuple[float, ...] | None = None,
    clearance: tuple[float, ...] | None = None,
    self_return: tuple[bool, ...] | None = None,
) -> subject.OriginSupportEvidence:
    count = len(support)
    support_array = np.asarray(support, dtype=bool)
    self_array = np.asarray(self_return or (False,) * count, dtype=bool)
    physical = support_array | self_array
    time = np.asarray(timestamp or tuple(0.002 * index for index in range(count)))
    return subject.OriginSupportEvidence(
        mount_identity=mount,
        witness_identity=tuple(f"w{index}" for index in range(count)),
        event_time_s=np.asarray(event or tuple(0.01 * index for index in range(count))),
        observation_support=support_array,
        nominal_fov_inclusion=np.ones(count, dtype=bool),
        direct_visibility_after_self_occlusion=support_array,
        self_return=self_array,
        acquisition_timestamp_s=np.where(physical, time, np.nan),
        ray_or_point_index=np.where(physical, np.arange(count), -1),
        point_range_m=np.where(physical, 1.0 + np.arange(count), np.nan),
        minimum_clearance_m=np.where(
            support_array,
            np.asarray(clearance or tuple(0.1 + 0.01 * index for index in range(count))),
            np.nan,
        ),
    )


def test_exact_four_mounts_derive_from_frozen_trunk_and_clear_housing():
    trunk, _protected = _nominal_geometry()
    envelope = subject.derive_trunk_envelope(trunk)
    assert envelope.minimum_xyz_m == (-0.1881, -0.04675, -0.057)
    assert envelope.maximum_xyz_m == (0.1881, 0.04675, 0.057)

    mounts = subject.derive_mount_candidates(trunk)
    assert tuple(row.mount_id for row in mounts) == subject.MOUNT_IDS
    expected = {
        subject.HEAD_STOCK: (0.28945, 0.0, -0.046825),
        subject.REAR_TOP_TRUNK: (-0.1254, 0.0, 0.0995),
        subject.LEFT_UPPER_FLANK: (0.0, 0.09425, 0.0285),
        subject.RIGHT_UPPER_FLANK: (0.0, -0.09425, 0.0285),
    }
    for row in mounts:
        np.testing.assert_allclose(row.translation_body_xyz_m, expected[row.mount_id], rtol=0.0, atol=2e-17)
        assert row.parent_link == "base"
    left = np.asarray(mounts[2].translation_body_xyz_m)
    right = np.asarray(mounts[3].translation_body_xyz_m)
    assert np.array_equal(right, left * np.asarray((1.0, -1.0, 1.0)))

    housing_half = 0.5 * np.asarray(subject.DEFAULT_HOUSING_FULL_EXTENTS_XYZ_M)
    assert mounts[1].translation_body_xyz_m[2] - housing_half[2] >= 0.057 + 0.01 - 1e-15
    assert mounts[2].translation_body_xyz_m[1] - housing_half[1] >= 0.04675 + 0.01 - 1e-15
    assert mounts[3].translation_body_xyz_m[1] + housing_half[1] <= -0.04675 - 0.01 + 1e-15


def test_head_stock_transform_and_four_pole_orientation_library_are_exact():
    trunk, _protected = _nominal_geometry()
    mounts = subject.derive_mount_candidates(trunk)
    head = subject.orientation_library(mounts[0])
    assert len(head) == 1 and head[0].orientation_id == subject.STOCK
    np.testing.assert_allclose(
        head[0].quaternion_body_wxyz,
        subject.rpy_quaternion_wxyz((0.0, 2.8782, 0.0)),
        rtol=0.0,
        atol=0.0,
    )

    for hardpoint in mounts[1:]:
        orientations = subject.orientation_library(hardpoint)
        assert tuple(row.orientation_id for row in orientations) == subject.SUPPLEMENTAL_ORIENTATION_IDS
        outward = np.asarray(hardpoint.outward_body_xyz)
        expected_poles = (
            np.asarray((0.0, 0.0, 1.0)),
            np.asarray((0.0, 0.0, -1.0)),
            (outward - np.asarray((0.0, 0.0, 1.0))) / math.sqrt(2.0),
            (-outward - np.asarray((0.0, 0.0, 1.0))) / math.sqrt(2.0),
        )
        for pose, pole in zip(orientations, expected_poles, strict=True):
            rotation = geometry.rotation_matrix_wxyz(pose.quaternion_body_wxyz)
            np.testing.assert_allclose(rotation[:, 2], pole, rtol=0.0, atol=2e-15)
            projected_x = np.asarray((1.0, 0.0, 0.0)) - pole[0] * pole
            if np.linalg.norm(projected_x) <= 1e-12:
                projected_x = np.asarray((0.0, 1.0, 0.0)) - pole[1] * pole
            projected_x /= np.linalg.norm(projected_x)
            np.testing.assert_allclose(rotation[:, 0], projected_x, rtol=0.0, atol=2e-15)
            np.testing.assert_allclose(np.cross(rotation[:, 0], rotation[:, 1]), rotation[:, 2], atol=2e-15)


def test_static_orientation_selection_is_deterministic_label_free_and_uses_frozen_tuple():
    trunk, protected = _nominal_geometry()
    mounts = subject.derive_mount_candidates(trunk)
    first = subject.score_and_select_mount_orientations(mounts, protected)
    second = subject.score_and_select_mount_orientations(mounts, protected)
    assert [row.selected.pose.orientation_id for row in first] == [
        row.selected.pose.orientation_id for row in second
    ]
    rank = {name: index for index, name in enumerate(subject.SUPPLEMENTAL_ORIENTATION_IDS)}
    rank[subject.STOCK] = 0
    for selection in first:
        assert 1 <= len(selection.candidates) <= 4
        for score in selection.candidates:
            assert score.nominal_witness_count == int(score.nominal_mask.sum())
            assert score.self_occluded_witness_count == int(score.self_occluded_mask.sum())
            if score.nominal_witness_count:
                assert not score.zero_nominal_fail_closed
                assert score.self_occlusion_fraction == float(
                    score.self_occluded_witness_count / score.nominal_witness_count
                )
            else:
                assert score.zero_nominal_fail_closed
                assert score.self_occlusion_fraction == 1.0
        expected = min(
            selection.candidates,
            key=lambda item: (
                -item.visible_fraction,
                item.self_occlusion_fraction,
                -item.calf_visibility_fraction,
                -item.rear_limb_visibility_fraction,
                rank[item.pose.orientation_id],
            ),
        )
        assert selection.selected.pose.orientation_id == expected.pose.orientation_id

    public_parameters = {
        parameter
        for function in (
            subject.score_static_orientation,
            subject.score_and_select_mount_orientations,
            subject.build_mount_library_receipt,
        )
        for parameter in inspect.signature(function).parameters
    }
    assert not ({"label", "labels", "contact", "outcome", "heldout"} & public_parameters)


def test_mount_library_receipt_is_canonical_and_binds_every_candidate_score():
    trunk, protected = _nominal_geometry()
    first = subject.build_mount_library_receipt(trunk, protected)
    second = subject.build_mount_library_receipt(trunk, protected)
    assert geometry.canonical_json_bytes(first) == geometry.canonical_json_bytes(second)
    assert first["content_digest"] == geometry.canonical_digest(
        {key: value for key, value in first.items() if key != "content_digest"}
    )
    assert first["contact_outcomes_used"] is False
    assert [row["mount_id"] for row in first["mount_candidates"]] == list(subject.MOUNT_IDS)
    assert len(first["orientation_selections"]) == 4
    assert [len(row["candidates"]) for row in first["orientation_selections"]] == [1, 4, 4, 4]
    assert [row["layout_id"] for row in first["layouts"]] == list(subject.LAYOUT_IDS)
    supplemental = first["housing_clearance_validation"][1:]
    assert all(row["pass"] for row in supplemental)
    assert all(row["protected_primitive_count"] == len(protected) for row in supplemental)
    assert all(
        row["minimum_protected_geometry_clearance_m"]
        >= row["required_clearance_m"] - 1e-12
        for row in supplemental
    )
    assert first["housing_occlusion_frame"] == (
        "BODY_AXIS_ALIGNED_TRUNK_FRAME_INDEPENDENT_OF_OPTICAL_RAY_FRAME"
    )


def test_no_arg_mount_receipt_uses_only_frozen_static_nominal_source_geometry():
    primitives = subject.frozen_static_nominal_go2_primitives()
    assert len(primitives) == 27
    assert [row.geom_index for row in primitives] == list(range(27))
    assert len({(row.link_index, row.link_name) for row in primitives}) == 13
    assert {row.link_name for row in primitives} == {
        "base",
        "FL_hip", "FL_thigh", "FL_calf",
        "FR_hip", "FR_thigh", "FR_calf",
        "RL_hip", "RL_thigh", "RL_calf",
        "RR_hip", "RR_thigh", "RR_calf",
    }
    assert subject.FROZEN_NOMINAL_STANCE_RAD == tuple(
        float(item)
        for item in np.asarray(
            (0.0, 0.0, 0.0, 0.0, 0.9, 0.9, 0.9, 0.9, -1.8, -1.8, -1.8, -1.8),
            dtype=np.float32,
        )
    )
    receipt = subject.build_mount_library_receipt()
    source = receipt["nominal_geometry_source"]
    assert source["source_class"] == "FROZEN_STATIC_NOMINAL_GO2_URDF_GEOMETRY"
    assert source["nominal_stance_source_path"] == "lewm_genesis/lewm_genesis/rollout.py"
    assert source["nominal_stance_source_sha256"] == subject.NOMINAL_STANCE_SOURCE_SHA256
    assert source["genesis_go2_urdf_sha256"] == subject.GENESIS_GO2_URDF_SHA256
    assert source["primitive_count"] == 27
    assert source["protected_link_count"] == 13
    assert receipt["orientation_selections"][0]["selected_pose"]["rpy_body_rad"] == [0.0, 2.8782, 0.0]
    assert geometry.canonical_json_bytes(receipt) == geometry.canonical_json_bytes(
        subject.build_mount_library_receipt()
    )


def test_mount_receipt_requires_both_or_neither_explicit_geometry_inputs():
    trunk, protected = _nominal_geometry()
    with pytest.raises(ValueError, match="together"):
        subject.build_mount_library_receipt(trunk_primitives=trunk)
    with pytest.raises(ValueError, match="together"):
        subject.build_mount_library_receipt(protected_primitives=protected)


def test_only_exact_three_pairs_and_three_triples_are_enumerated_in_fixed_order():
    layouts = subject.enumerate_layout_candidates()
    assert tuple(row.layout_id for row in layouts) == subject.LAYOUT_IDS
    assert [row.origin_count for row in layouts] == [2, 2, 2, 3, 3, 3]
    assert [row.mount_ids for row in layouts] == [row[1] for row in subject.LAYOUT_DEFINITIONS]
    with pytest.raises(ValueError, match="exactly"):
        subject.enumerate_layout_candidates(subject.MOUNT_IDS[:-1])


def test_multi_origin_union_is_complementary_and_keeps_mount_timestamp_provenance():
    left = _origin(
        subject.LEFT_UPPER_FLANK,
        (True, False, True),
        event=(0.05, 0.05, 0.05),
        timestamp=(0.01, 0.02, 0.04),
        clearance=(0.20, 0.30, 0.08),
    )
    right = _origin(
        subject.RIGHT_UPPER_FLANK,
        (False, True, True),
        event=(0.05, 0.05, 0.05),
        timestamp=(0.03, 0.02, 0.03),
        clearance=(0.30, 0.10, 0.09),
    )
    union = subject.union_multi_origin_support((left, right))
    assert union.observation_support.tolist() == [True, True, True]
    assert union.unsupported.tolist() == [False, False, False]
    assert union.support_count.tolist() == [1, 1, 2]
    assert union.support_mount_identity == (
        subject.LEFT_UPPER_FLANK,
        subject.RIGHT_UPPER_FLANK,
        subject.LEFT_UPPER_FLANK,
    )
    assert union.minimum_clearance_mount_identity == (
        subject.LEFT_UPPER_FLANK,
        subject.RIGHT_UPPER_FLANK,
        subject.LEFT_UPPER_FLANK,
    )
    np.testing.assert_allclose(union.support_timestamp_s, (0.01, 0.02, 0.04), atol=0.0)
    np.testing.assert_allclose(union.support_point_age_s, (0.04, 0.03, 0.01), atol=1e-18)
    assert union.supporting_origin_mask.tolist() == [
        [True, False, True],
        [False, True, True],
    ]


def test_support_provenance_uses_latest_past_then_earliest_future_and_fixed_origin_tie():
    head = _origin(
        subject.HEAD_STOCK,
        (True, True, True),
        event=(0.05, 0.05, 0.05),
        timestamp=(0.01, 0.06, 0.02),
        clearance=(0.2, 0.2, 0.2),
    )
    rear = _origin(
        subject.REAR_TOP_TRUNK,
        (True, True, True),
        event=(0.05, 0.05, 0.05),
        timestamp=(0.04, 0.07, 0.02),
        clearance=(0.1, 0.1, 0.1),
    )
    union = subject.union_multi_origin_support((head, rear))
    assert union.support_mount_identity == (
        subject.REAR_TOP_TRUNK,
        subject.HEAD_STOCK,
        subject.HEAD_STOCK,
    )
    np.testing.assert_allclose(union.support_timestamp_s, (0.04, 0.06, 0.02), atol=0.0)
    assert union.minimum_clearance_mount_identity == (subject.REAR_TOP_TRUNK,) * 3


def test_self_return_is_retained_for_audit_but_excluded_from_union_support():
    head = _origin(
        subject.HEAD_STOCK,
        (False,),
        event=(0.05,),
        timestamp=(0.02,),
        self_return=(True,),
    )
    rear = _origin(
        subject.REAR_TOP_TRUNK,
        (False,),
        event=(0.05,),
        timestamp=(0.03,),
    )
    union = subject.union_multi_origin_support((head, rear))
    assert union.self_return_by_origin.tolist() == [[True], [False]]
    assert union.acquisition_timestamp_by_origin_s[0, 0] == 0.02
    assert not union.observation_support[0]
    assert union.unsupported[0]
    assert union.support_mount_identity == (None,)
    with pytest.raises(ValueError, match="non-self"):
        subject.OriginSupportEvidence(
            mount_identity=subject.HEAD_STOCK,
            witness_identity=("w",),
            event_time_s=np.asarray((0.0,)),
            observation_support=np.asarray((True,)),
            nominal_fov_inclusion=np.asarray((True,)),
            direct_visibility_after_self_occlusion=np.asarray((True,)),
            self_return=np.asarray((True,)),
            acquisition_timestamp_s=np.asarray((0.0,)),
            ray_or_point_index=np.asarray((0,)),
            point_range_m=np.asarray((1.0,)),
            minimum_clearance_m=np.asarray((0.1,)),
        )


def test_phase_is_contract_transition_mount_bound_repeatable_and_independent():
    from lewm.safety import minimum_multi_origin_body_range_coverage_qualification_v1_contract as contract

    digest = "01" * 32
    first = subject.derive_scan_phase(digest, "state:a/current:3", subject.HEAD_STOCK)
    again = subject.derive_scan_phase(digest, "state:a/current:3", subject.HEAD_STOCK)
    other_mount = subject.derive_scan_phase(digest, "state:a/current:3", subject.REAR_TOP_TRUNK)
    other_transition = subject.derive_scan_phase(digest, "state:a/current:4", subject.HEAD_STOCK)
    assert first == again
    assert len({first.phase_digest_sha256, other_mount.phase_digest_sha256, other_transition.phase_digest_sha256}) == 3
    assert 0.0 <= first.azimuth_phase_cycles < 1.0
    assert 0.0 <= first.vertical_phase_cycles < 1.0
    authority = contract.derive_multi_origin_scan_phases(
        contract_digest_sha256=digest,
        transition_uid="state:a/current:3",
        mount_id=subject.HEAD_STOCK,
    )
    assert first.phase_digest_sha256 == authority["phase_digest_sha256"]
    assert first.horizontal_phase_cycles == authority["horizontal_phase_cycles"]
    assert first.vertical_phase_cycles == authority["vertical_phase_cycles"]


def _support_for_layout(
    layout_id: str,
    *,
    transition_support: tuple[float, ...] = (1.0, 0.8, 0.6, 0.4),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    mounts = subject.LAYOUT_MOUNTS[layout_id]
    links = ("base", "FL_thigh", "RL_thigh", "FL_calf", "RR_calf")
    support = np.zeros((len(mounts), len(transition_support), 50, len(links)), dtype=bool)
    total = 50 * len(links)
    for transition, fraction in enumerate(transition_support):
        count = int(round(fraction * total))
        flattened = support[0, transition].reshape(-1)
        flattened[:count] = True
    self_occlusion = ~support.copy()
    nominal = np.ones_like(support)
    return support, nominal, self_occlusion, links


def test_layout_metrics_use_complete_training_transition_x_50_x_link_denominators_and_linear_p5():
    layout_id = subject.LAYOUT_IDS[0]
    support, nominal, self_occlusion, links = _support_for_layout(layout_id)
    metrics = subject.compute_layout_support_metrics(
        layout_id,
        subject.LAYOUT_MOUNTS[layout_id],
        support,
        nominal,
        self_occlusion,
        links,
        role="training",
    )
    expected_transition = support.any(axis=0).reshape(4, -1).mean(axis=1)
    assert metrics.transitions == 4
    assert metrics.physics_steps == 50
    assert metrics.protected_links == len(links)
    assert metrics.p5_transition_support == np.percentile(expected_transition, 5.0, method="linear")
    assert set(metrics.body_region_support) == set(subject.BODY_REGION_IDS)
    assert metrics.minimum_body_region_support == min(metrics.body_region_support.values())
    assert metrics.rear_limb_support == metrics.body_region_support[subject.REAR_LIMBS]
    assert metrics.calf_support == metrics.body_region_support[subject.CALVES]
    expected_self_occlusion = float(
        ((~support.any(axis=0)) & self_occlusion.all(axis=0)).sum()
        / nominal.any(axis=0).sum()
    )
    assert metrics.self_occlusion_fraction == expected_self_occlusion
    with pytest.raises(ValueError, match="training"):
        subject.compute_layout_support_metrics(
            layout_id,
            subject.LAYOUT_MOUNTS[layout_id],
            support,
            nominal,
            self_occlusion,
            links,
            role="development_held_out",
        )
    with pytest.raises(ValueError, match="50"):
        subject.compute_layout_support_metrics(
            layout_id,
            subject.LAYOUT_MOUNTS[layout_id],
            support[:, :, :-1],
            nominal[:, :, :-1],
            self_occlusion[:, :, :-1],
            links,
            role="training",
        )
    with pytest.raises(ValueError, match="zero nominal"):
        subject.compute_layout_support_metrics(
            layout_id,
            subject.LAYOUT_MOUNTS[layout_id],
            np.zeros_like(support),
            np.zeros_like(nominal),
            self_occlusion,
            links,
            role="training",
        )


def _metric(layout_id: str, values: tuple[float, float, float, float, float, float]) -> subject.LayoutSupportMetrics:
    return subject.LayoutSupportMetrics(
        layout_id=layout_id,
        role="training",
        transitions=2,
        physics_steps=50,
        protected_links=5,
        minimum_body_region_support=values[0],
        p5_transition_support=values[1],
        rear_limb_support=values[2],
        calf_support=values[3],
        overall_mean_support=values[4],
        self_occlusion_fraction=values[5],
        body_region_support={region: values[0] for region in subject.BODY_REGION_IDS},
    )


def test_pair_and_triple_selection_follow_full_lexicographic_tuple_and_fixed_id():
    baseline = (0.8, 0.7, 0.6, 0.5, 0.9, 0.2)
    pair_rows = [_metric(layout_id, baseline) for layout_id in subject.LAYOUT_IDS[:3]]
    assert subject.select_layout_lexicographically(pair_rows, origin_count=2).layout_id == subject.LAYOUT_IDS[0]
    pair_rows[1] = _metric(subject.LAYOUT_IDS[1], (0.81, 0.0, 0.0, 0.0, 0.0, 1.0))
    assert subject.select_layout_lexicographically(pair_rows, origin_count=2).layout_id == subject.LAYOUT_IDS[1]

    triple_rows = [_metric(layout_id, baseline) for layout_id in subject.LAYOUT_IDS[3:]]
    triple_rows[2] = _metric(subject.LAYOUT_IDS[5], (0.8, 0.7, 0.6, 0.5, 0.9, 0.1))
    assert subject.select_layout_lexicographically(triple_rows, origin_count=3).layout_id == subject.LAYOUT_IDS[5]
    selected = subject.select_pair_and_triple_layouts(tuple(pair_rows + triple_rows))
    assert selected["pair"].layout_id == subject.LAYOUT_IDS[1]
    assert selected["triple"].layout_id == subject.LAYOUT_IDS[5]


def test_two_ply_helpers_fail_closed_make_ties_positive_and_abstain():
    assert subject.sensor_contact_decision(0.1, True, 0.1)
    assert subject.sensor_contact_decision(math.inf, False, 0.1)
    assert not subject.sensor_contact_decision(0.1000001, True, 0.1)
    assert subject.predicted_safe_action_count((True, False, True)) == 1
    assert subject.predicted_safe_action_count((True, True)) == 0
    assert subject.admit_current_action(False, 1)
    assert not subject.admit_current_action(True, 1)
    assert not subject.admit_current_action(False, 0)


def test_complete_named_synthetic_fixture_receipt_passes_and_is_byte_identical():
    expected = {
        "clear full-body sweep",
        "front trunk contact",
        "side trunk contact",
        "rear trunk contact",
        "front-limb contact",
        "rear-limb contact",
        "calf contact",
        "one origin occluded another observes",
        "complementary L/R flank",
        "complementary head/rear",
        "near blind",
        "between scan samples",
        "synchronized overlap",
        "independent phases",
        "one safe successor",
        "zero safe successors",
        "threshold tie",
        "abstention",
        "H3",
    }
    first = subject.synthetic_fixture_receipt()
    second = subject.synthetic_fixture_receipt()
    alias = subject.run_fixtures()
    assert first["pass"]
    assert set(first["fixtures"]) == expected
    assert all(row["pass"] for row in first["fixtures"].values())
    assert first["requirements"]["byte-identical receipt regeneration"]["pass"]
    assert geometry.canonical_json_bytes(first) == geometry.canonical_json_bytes(second)
    assert geometry.canonical_json_bytes(first) == geometry.canonical_json_bytes(alias)
    assert first["content_digest"] == geometry.canonical_digest(
        {key: value for key, value in first.items() if key != "content_digest"}
    )


def test_fixture_raw_evidence_is_complete_and_self_digesting():
    raw = subject.run_fixtures()["raw_fixture_evidence"]
    core = dict(raw)
    declared = core.pop("content_digest")
    assert declared == geometry.canonical_digest(core)
    assert set(raw["complete_raw_ray_queries"]) == {
        "near_blind",
        "between_scan_samples",
        "one_origin_occluded_another_observes",
    }
    for name, row in raw["complete_raw_ray_queries"].items():
        assert "environment_boxes" in row
        assert "robot_primitives" in row
        assert "near_m" in row and "far_m" in row
        if name == "one_origin_occluded_another_observes":
            assert "blocked_direction_world" in row
            assert "observed_direction_world" in row
            assert row["timestamps_s"] == [0.0, 0.0]
        else:
            assert len(row["directions_world"]) == len(row["timestamps_s"])
            assert len(row["first_hits"]["rows"]) == len(
                row["directions_world"]
            )
    contact_queries = raw["reconstructible_noncloud_inputs"]["contact_queries"]
    assert len(contact_queries) == 6
    assert all(
        "environment_box" in row and "signed_clearance_m" in row
        for row in contact_queries.values()
    )
