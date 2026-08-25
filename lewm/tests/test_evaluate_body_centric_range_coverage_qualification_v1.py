from __future__ import annotations

import numpy as np

from lewm.safety import body_centric_range_coverage_corpus_v1 as corpus
from lewm.safety import body_centric_range_coverage_v1 as geometry
from scripts import evaluate_body_centric_range_coverage_qualification_v1 as evaluator


def test_frozen_exact_geometry_preflight_totals() -> None:
    context = corpus.load_corpus_context(evaluator.ROOT)
    receipt = evaluator.exact_geometry_materialization_corpus_audit(context, corpus)
    assert receipt["pass"] is True
    assert receipt["action_representatives"] == 13_385
    assert receipt["geometry_representatives"] == 13_584
    assert receipt["nonexact_independent_rows"] == 199
    assert receipt["affected_states"] == 81
    assert receipt["affected_states_by_role"] == {
        "training": 58,
        "calibration": 12,
        "heldout": 11,
    }
    assert receipt["exact_reused_pairs"] == 15_886


def test_failed_attempt_custody_and_fresh_restart_binding() -> None:
    receipt = evaluator.validate_prospective_execution_amendment(
        evaluator.CONTRACT.build_contract(),
        canonical_files_before_preflight=(),
    )
    assert receipt["pass"] is True
    assert receipt["canonical_output_fresh"] is True
    assert receipt["prior_state_shards_reused"] is False
    assert receipt["failed_attempt_manifest_rows"] == 328


def test_sparse_evaluator_accepts_canonical_uppercase_environment_hits() -> None:
    qpos = np.zeros(19, dtype=np.float64)
    qpos[:7] = (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    wall = geometry.OrientedBox(
        "wall",
        (2.0, 0.0, 1.25),
        (0.1, 5.0, 5.0),
        object_index=0,
    )
    cloud = evaluator.render_sparse_cloud(
        condition_id="CURRENT_SPARSE_RANGE_BASELINE",
        mode_id="PLANNING_TIME_CAUSAL_CLOUD",
        state_id="fixture",
        transition_identity="fixture/current/0",
        boundary_snapshot_digest="1" * 64,
        boundary_qpos=qpos,
        boundary_geom_transform=np.empty((0, 7), dtype=np.float64),
        qpos=np.repeat(qpos[None], evaluator.STEPS, axis=0),
        geom_transform=np.empty((evaluator.STEPS, 0, 7), dtype=np.float64),
        environment_boxes=[wall],
        robot_specs=[],
    )
    assert cloud["environment_return_count"] > 0
    assert cloud["points"].shape[0] == cloud["environment_return_count"]
    assert cloud["self_return_count"] == 0


def test_audit_rank_binds_complete_prospective_stratum() -> None:
    first = evaluator.audit_selection_sha256(
        transition_uid="state/current/0",
        role="calibration",
        family="maze",
        transition_kind="current",
    )
    second = evaluator.audit_selection_sha256(
        transition_uid="state/current/0",
        role="calibration",
        family="maze",
        transition_kind="successor",
    )
    assert len(first) == 64
    assert first != second


def test_error_witness_tracks_decision_cause_and_preserves_unresolved_link() -> None:
    sensor = np.full((evaluator.STEPS, evaluator.LINKS), 1.0)
    support = np.ones_like(sensor, dtype=bool)
    support[7, 4] = False
    support[2, 9] = False
    oracle = np.full_like(sensor, 2.0)
    oracle[7, 4] = 0.2
    oracle[2, 9] = 0.1
    false_positive = evaluator._select_error_witness(
        oracle_contact=False,
        predicted_contact=True,
        sensor_clearance=sensor,
        support=support,
        oracle_clearance=oracle,
        oracle_contact_step=-1,
        oracle_contact_link=-1,
    )
    assert false_positive == {
        "step": 2,
        "link": 9,
        "link_resolved": True,
        "mechanism": "UNSUPPORTED_SWEEP_RISK",
    }

    false_negative = evaluator._select_error_witness(
        oracle_contact=True,
        predicted_contact=False,
        sensor_clearance=sensor,
        support=np.ones_like(support),
        oracle_clearance=oracle,
        oracle_contact_step=7,
        oracle_contact_link=-1,
    )
    assert false_negative["step"] == 7
    assert false_negative["link_resolved"] is False
    assert false_negative["mechanism"] == "FROZEN_ORACLE_CONTACT"


def test_dense_same_object_near_face_is_occlusion_not_direct_visibility() -> None:
    qpos = np.zeros((evaluator.STEPS, 19), dtype=np.float64)
    qpos[:, :7] = (-2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    boundary = qpos[0].copy()
    target = np.zeros((evaluator.STEPS, evaluator.LINKS, 3), dtype=np.float64)
    target[..., 0] = 0.5
    target[..., 2] = 0.067
    values = evaluator.dense_per_link_evidence(
        condition_id="DENSE_BODY_CENTRIC_SINGLE_ORIGIN",
        mode_id="TRUE_FUTURE_OBSERVABILITY_CLOUD",
        boundary_qpos=boundary,
        boundary_geom_transform=np.empty((0, 7), dtype=np.float64),
        qpos=qpos,
        geom_transform=np.empty((evaluator.STEPS, 0, 7), dtype=np.float64),
        environment_boxes=[
            geometry.OrientedBox(
                "box", (0.0, 0.0, 0.067), (0.5, 0.5, 0.5), object_index=0
            )
        ],
        robot_specs=[],
        targets={
            "target_points": target,
            "object_index": np.zeros((evaluator.STEPS, evaluator.LINKS), np.int16),
            "clearance_m": np.ones((evaluator.STEPS, evaluator.LINKS), np.float64),
        },
    )
    assert not values["support"].any()
    assert values["environment_occluded"].all()
    assert not values["direct_visibility"].any()


def test_sparse_event_time_support_rejects_returns_after_event() -> None:
    specs = [
        geometry.RobotPrimitiveSpec(
            identity=f"link-{index}",
            kind="sphere",
            data=(0.1,),
            local_position_xyz_m=(0.0, 0.0, 0.0),
            local_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            geom_index=index,
            link_index=index,
            link_name=f"link-{index}",
        )
        for index in range(evaluator.LINKS)
    ]
    transforms = np.zeros(
        (evaluator.STEPS, evaluator.LINKS, 7), dtype=np.float64
    )
    transforms[..., 3] = 1.0
    targets = np.zeros((evaluator.STEPS, evaluator.LINKS, 3), np.float64)
    targets[..., 0] = 1.0
    values = evaluator.sparse_per_link_evidence(
        cloud={
            "points": np.asarray([[1.0, 0.0, 0.0]]),
            "object_index": np.asarray([0], np.int16),
            "ray_index": np.asarray([5], np.int32),
            "point_time_s": np.asarray([0.05]),
            "point_range_m": np.asarray([1.0]),
        },
        robot_specs=specs,
        geom_transform=transforms,
        qpos=np.concatenate(
            (
                np.zeros((evaluator.STEPS, 3), np.float64),
                np.tile(
                    np.asarray([1.0, 0.0, 0.0, 0.0], np.float64),
                    (evaluator.STEPS, 1),
                ),
                np.zeros((evaluator.STEPS, 12), np.float64),
            ),
            axis=1,
        ),
        targets={
            "target_points": targets,
            "object_index": np.zeros((evaluator.STEPS, evaluator.LINKS), np.int16),
        },
        evaluation_time_s=np.arange(1, evaluator.STEPS + 1) * 0.002,
    )
    assert not values["event_time_support"][:24].any()
    assert values["event_time_support"][24:].all()
    assert np.all(values["support"])


def _dense_accumulation_fixture(*, release_step: int | None) -> dict[str, np.ndarray]:
    qpos = np.zeros((evaluator.STEPS, 19), dtype=np.float64)
    qpos[:, 3] = 1.0
    boundary_qpos = qpos[0].copy()
    boundary_geom = np.asarray([[0.45, 0.0, 0.067, 1.0, 0.0, 0.0, 0.0]])
    geom = np.repeat(boundary_geom[None], evaluator.STEPS, axis=0)
    if release_step is None:
        # The boundary is visible but every in-transition endpoint is blocked.
        boundary_geom[0, :3] = (0.0, 2.0, 0.067)
    else:
        # Only the chosen later endpoint exposes the target.
        geom[release_step:, 0, :3] = (0.0, 2.0, 0.067)
    target = np.zeros((evaluator.STEPS, evaluator.LINKS, 3), dtype=np.float64)
    target[..., 0] = 0.9
    target[..., 2] = 0.067
    return evaluator.dense_per_link_evidence(
        condition_id="DENSE_BODY_CENTRIC_SINGLE_ORIGIN",
        mode_id="TRUE_FUTURE_OBSERVABILITY_CLOUD",
        boundary_qpos=boundary_qpos,
        boundary_geom_transform=boundary_geom,
        qpos=qpos,
        geom_transform=geom,
        environment_boxes=[
            geometry.OrientedBox(
                "wall", (1.0, 0.0, 0.067), (0.1, 1.0, 1.0), object_index=0
            )
        ],
        robot_specs=[
            geometry.RobotPrimitiveSpec(
                identity="occluder",
                kind="sphere",
                data=(0.2,),
                local_position_xyz_m=(0.0, 0.0, 0.0),
                local_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
                geom_index=0,
                link_index=0,
                link_name="base",
            )
        ],
        targets={
            "target_points": target,
            "object_index": np.zeros((evaluator.STEPS, evaluator.LINKS), np.int16),
            "clearance_m": np.full((evaluator.STEPS, evaluator.LINKS), 0.3),
        },
    )


def test_dense_true_future_accumulates_boundary_visible_witness() -> None:
    values = _dense_accumulation_fixture(release_step=None)
    assert values["support"].all()
    assert values["event_time_support"].all()
    assert np.all(values["support_acquisition_index"] == 0)
    assert np.all(values["support_point_age_s"] > 0.0)


def test_dense_true_future_preserves_later_only_negative_age() -> None:
    values = _dense_accumulation_fixture(release_step=evaluator.STEPS - 1)
    assert values["support"].all()
    assert not values["event_time_support"][0].any()
    assert np.all(values["support_acquisition_index"][0] == evaluator.STEPS)
    assert np.all(values["support_point_age_s"][0] < 0.0)


def test_platform_host_housing_exemption_is_exact_and_ray_only() -> None:
    specs = [
        geometry.RobotPrimitiveSpec(
            identity=f"base:{index:02d}",
            kind="sphere",
            data=(0.1,),
            local_position_xyz_m=(0.0, 0.0, 0.0),
            local_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            geom_index=index,
            link_index=0,
            link_name="base",
        )
        for index in (1, 2, 3)
    ]
    positions = np.zeros((4, 3, 3), np.float64)
    quaternions = np.zeros((4, 3, 4), np.float64)
    quaternions[..., 0] = 1.0
    retained, retained_positions, retained_quaternions = (
        evaluator.self_occlusion_geometry_for_condition(
            "DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT",
            specs,
            positions,
            quaternions,
        )
    )
    assert [spec.geom_index for spec in retained] == [3]
    assert retained_positions.shape == (4, 1, 3)
    assert retained_quaternions.shape == (4, 1, 4)
    body_specs, _, _ = evaluator.self_occlusion_geometry_for_condition(
        "DENSE_BODY_CENTRIC_SINGLE_ORIGIN", specs, positions, quaternions
    )
    assert [spec.geom_index for spec in body_specs] == [1, 2, 3]


def test_dense_platform_inherits_every_realistic_local_witness() -> None:
    shape = (evaluator.STEPS, evaluator.LINKS)
    dense = {
        "support": np.zeros(shape, bool),
        "event_time_support": np.zeros(shape, bool),
        "clearance_m": np.full(shape, np.inf),
        "responsible_object_index": np.full(shape, -1, np.int16),
        "nearest_ray_index": np.full(shape, -1, np.int32),
        "point_age_s": np.full(shape, np.nan),
        "nearest_point_range_m": np.full(shape, np.nan),
        "support_nearest_ray_index": np.full(shape, -1, np.int32),
        "support_point_age_s": np.full(shape, np.nan),
        "support_nearest_range_m": np.full(shape, np.nan),
        "support_object_index": np.full(shape, -1, np.int16),
        "point_support_count": np.zeros(shape, np.int16),
        "responsible_geom_index": np.full(shape, -1, np.int16),
        "obstacle_direction_body_rad": np.full(shape, np.nan),
        "nominal_fov": np.zeros(shape, bool),
        "horizontal_fov": np.zeros(shape, bool),
        "vertical_fov": np.zeros(shape, bool),
        "direct_visibility": np.zeros(shape, bool),
        "self_occluded": np.ones(shape, bool),
        "environment_occluded": np.ones(shape, bool),
        "near_blind": np.ones(shape, bool),
        "self_occluder_geom_index": np.zeros(shape, np.int16),
        "self_occluder_link_index": np.zeros(shape, np.int16),
        "finite_scan_support_inherited": np.zeros(shape, bool),
        "responsible_acquisition_index": np.zeros(shape, np.int16),
        "support_acquisition_index": np.zeros(shape, np.int16),
    }
    sparse = {key: np.array(value, copy=True) for key, value in dense.items()}
    sparse["support"][3, 4] = True
    sparse["event_time_support"][3, 4] = True
    sparse["clearance_m"][3, 4] = 0.12
    sparse["responsible_object_index"][3, 4] = 7
    sparse["support_object_index"][3, 4] = 7
    sparse["nearest_ray_index"][3, 4] = 99
    sparse["support_nearest_ray_index"][3, 4] = 99
    sparse["point_age_s"][3, 4] = 0.001
    sparse["support_point_age_s"][3, 4] = 0.001
    sparse["nearest_point_range_m"][3, 4] = 2.0
    sparse["support_nearest_range_m"][3, 4] = 2.0
    assert evaluator.inherit_realistic_support_into_dense_platform(dense, sparse) == 1
    assert dense["support"][3, 4]
    assert dense["finite_scan_support_inherited"][3, 4]
    assert dense["nearest_ray_index"][3, 4] == 99
