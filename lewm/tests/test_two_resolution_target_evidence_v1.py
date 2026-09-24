from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
import pytest

from lewm.benchmarks.go2_g3_exact_physical_equivalence_v2 import (
    _build_projected_snapshot,
)
from lewm.planning.revisioned_physical_configuration_memory import (
    MapFrameIdentity,
    PhysicalLabel,
    SnapshotBindingError,
)
from lewm.planning.two_resolution_configuration_projection_v2 import (
    ConfigurationComponentV2,
    TwoResolutionConfigurationPlannerV2,
    TwoResolutionConfigurationProjectionV2,
)
from lewm.planning.two_resolution_target_evidence_v1 import (
    PRODUCTION_G3_V2_COMPONENT_SOURCE,
    PRODUCTION_G3_V2_SNAPSHOT_SOURCE,
    PRODUCTION_G5_TWO_RESOLUTION_EVIDENCE_ISSUER,
    PRODUCTION_V5_CAMERA_CALIBRATION_SHA256,
    PRODUCTION_V5_CHECKPOINT_FILE_SHA256,
    PRODUCTION_V5_RAW_OUTCOME_SOURCE,
    PRODUCTION_V5_RUNNER_EXECUTION_IDENTITY,
    PhysicalCellProbabilityV1,
    SyntheticV5TargetOutcomeIssuerV1,
    TwoResolutionNegativeTargetEvidenceV1,
    TwoResolutionPositiveTargetEvidenceV1,
    TwoResolutionTargetEvidenceBindingError,
    TwoResolutionTargetEvidenceIssuerV1,
    TwoResolutionTargetEvidenceRejectedError,
    TwoResolutionTargetEvidenceReplayError,
    V5RunnerExecutionIdentityV1,
    physical_cell_to_configuration_cell,
    require_production_two_resolution_target_evidence_issuer,
)
from lewm_worlds.manifest import CameraValidityConstraints, SceneManifest, SpawnSpec


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


RUNNER_IDENTITY = V5RunnerExecutionIdentityV1(
    entrypoint_wrapper_file_sha256=_hash("synthetic-v5-runner-wrapper"),
    captured_launcher_file_sha256=_hash("synthetic-v5-captured-launcher"),
    captured_core_file_sha256=_hash("synthetic-v5-captured-core"),
)
CHECKPOINT_SHA256 = _hash("synthetic-v5-checkpoint")
CALIBRATION_SHA256 = _hash("synthetic-v5-camera-calibration")


def _manifest(scene_id: str, *, origin: tuple[float, float]) -> SceneManifest:
    return SceneManifest(
        scene_id=scene_id,
        family="unit",
        difficulty_tier="unit",
        topology_seed=1,
        visual_seed=2,
        physics_seed=3,
        world_bounds_xy_m=(
            (origin[0], origin[1]),
            (origin[0] + 3.2, origin[1] + 3.2),
        ),
        spawn=SpawnSpec(
            xyz_m=(origin[0] + 1.5, origin[1] + 1.5, 0.35),
            quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        ),
        graph_nodes=(),
        graph_edges=(),
        obstacles=(),
        landmarks=(),
        camera_constraints=CameraValidityConstraints(
            min_wall_thickness_m=0.08,
            near_m=0.05,
            far_m=200.0,
            min_camera_clearance_m=0.10,
        ),
        split="train",
    )


def _g3(
    scene_id: str = "g5-two-resolution",
    *,
    origin: tuple[float, float] = (-1.6, -1.6),
    unknown_physical_cells: frozenset[tuple[int, int]] = frozenset(),
    component_start: tuple[int, int] = (15, 15),
):
    labels = np.full((64, 64), int(PhysicalLabel.FREE), dtype=np.uint8)
    for cell in unknown_physical_cells:
        labels[cell] = int(PhysicalLabel.UNKNOWN)
    memory, snapshot, planner = _build_projected_snapshot(
        _manifest(scene_id, origin=origin),
        labels,
        origin_xy_m=origin,
        configuration_shape=(32, 32),
    )
    assert isinstance(planner, TwoResolutionConfigurationPlannerV2)
    projection = getattr(planner, "_projection")
    assert isinstance(projection, TwoResolutionConfigurationProjectionV2)
    component = planner.connected_component(snapshot, component_start)
    assert component_start in component.cells
    return memory, projection, snapshot, planner, component


def _outcome(
    source: SyntheticV5TargetOutcomeIssuerV1,
    snapshot,
    *,
    kind: str = "negative",
    unknown_cells: frozenset[tuple[int, int]] = frozenset(),
    target_cells: frozenset[tuple[int, int]] = frozenset({(40, 40)}),
    visible_cells: frozenset[tuple[int, int]] | None = None,
    probability: tuple[PhysicalCellProbabilityV1, ...] | None = None,
    checkpoint_sha256: str = CHECKPOINT_SHA256,
    calibration_sha256: str = CALIBRATION_SHA256,
    runner_identity: V5RunnerExecutionIdentityV1 = RUNNER_IDENTITY,
):
    if probability is None:
        probability = (
            PhysicalCellProbabilityV1((30, 30), 0.9),
            PhysicalCellProbabilityV1((30, 31), 0.8),
            PhysicalCellProbabilityV1((31, 30), 0.7),
            PhysicalCellProbabilityV1((31, 31), 0.6),
        )
    if visible_cells is None:
        visible_cells = frozenset(row.cell for row in probability)
    unlocalized = 0.0
    if kind == "positive":
        probability = (
            PhysicalCellProbabilityV1((30, 30), 0.4),
            PhysicalCellProbabilityV1((31, 30), 0.3),
            PhysicalCellProbabilityV1((32, 30), 0.2),
        )
        visible_cells = frozenset(row.cell for row in probability)
        unlocalized = 0.1
    free = frozenset(visible_cells)
    return source.issue(
        snapshot=snapshot,
        outcome_kind=kind,
        target_id="blue",
        pose_timestamp_ns=17,
        runner_execution_identity=runner_identity,
        checkpoint_file_sha256=checkpoint_sha256,
        raw_outcome_file_sha256=_hash(f"raw-{kind}-{source!r}"),
        camera_calibration_sha256=calibration_sha256,
        pose_provenance_sha256=_hash("synthetic-pose"),
        free_physical_cells=free,
        unknown_physical_cells=unknown_cells,
        target_physical_cells=target_cells,
        visible_physical_cells=visible_cells,
        physical_probability=probability,
        unlocalized_probability=unlocalized,
        confidence=0.85,
    )


def _stack(*, kind: str = "negative"):
    memory, projection, snapshot, planner, component = _g3()
    source = SyntheticV5TargetOutcomeIssuerV1(_synthetic_test_fixture=True)
    outcome = _outcome(source, snapshot, kind=kind)
    issuer = TwoResolutionTargetEvidenceIssuerV1(
        projection=projection,
        planner=planner,
        outcome_source=source,
        runner_execution_identity=RUNNER_IDENTITY,
        checkpoint_file_sha256=CHECKPOINT_SHA256,
        camera_calibration_sha256=CALIBRATION_SHA256,
        _synthetic_test_fixture=True,
    )
    return memory, projection, snapshot, planner, component, source, outcome, issuer


def test_positive_evidence_aggregates_physical_mass_into_configuration_cells() -> None:
    _, _, snapshot, _, component, _, outcome, issuer = _stack(kind="positive")
    context = issuer.issue_context(snapshot, component, outcome)
    evidence = issuer.open_writer(context).issue_positive()
    assert type(evidence) is TwoResolutionPositiveTargetEvidenceV1
    assert [(row.cell, row.value) for row in evidence.localized_distribution] == [
        ((15, 15), pytest.approx(0.7)),
        ((16, 15), pytest.approx(0.2)),
    ]
    assert evidence.unlocalized_probability == pytest.approx(0.1)
    assert evidence.posterior_cell_size_m == pytest.approx(0.10)
    assert context.posterior_cell_size_m == pytest.approx(0.10)
    assert context.physical_cell_size_m == pytest.approx(0.05)
    assert evidence.production_eligible is False


def test_negative_evidence_requires_and_uses_all_four_physical_children() -> None:
    _, _, snapshot, _, component, _, outcome, issuer = _stack()
    context = issuer.issue_context(snapshot, component, outcome)
    evidence = issuer.open_writer(context).issue_negative()
    assert type(evidence) is TwoResolutionNegativeTargetEvidenceV1
    assert [(row.cell, row.value) for row in evidence.visible_detection_probability] == [
        ((15, 15), pytest.approx(0.6)),
    ]
    assert evidence.posterior_cell_size_m == pytest.approx(0.10)
    assert evidence.production_eligible is False


def test_high_index_conversion_cannot_confuse_physical_and_configuration_cells() -> None:
    physical = MapFrameIdentity(
        session_id="large-physical",
        origin_xy_m=(-4.0, -3.0),
        cell_size_m=0.05,
        frame_id="physical",
    )
    configuration = MapFrameIdentity(
        session_id="large-configuration",
        origin_xy_m=(-4.0, -3.0),
        cell_size_m=0.10,
        frame_id="configuration",
    )
    converted = physical_cell_to_configuration_cell(
        physical_frame=physical,
        configuration_frame=configuration,
        physical_shape=(256, 192),
        configuration_shape=(128, 96),
        physical_cell=(122, 86),
    )
    assert converted == (61, 43)
    assert converted != (122, 86)


def test_context_binds_exact_live_g3_and_v5_identities_and_excludes_target_cells() -> None:
    _, _, snapshot, _, component, _, outcome, issuer = _stack()
    context = issuer.issue_context(snapshot, component, outcome)
    assert context.configuration_snapshot_sha256 == snapshot.content_sha256
    assert context.configuration_component_sha256 == component.content_sha256
    assert context.physical_map_frame.content_sha256 == snapshot.physical_map_frame_sha256
    assert (
        context.configuration_map_frame.content_sha256
        == snapshot.configuration_map_frame_sha256
    )
    assert context.physical_revision == snapshot.physical_revision
    assert context.configuration_revision == snapshot.configuration_revision
    assert context.free_support_sha256 == snapshot.free_support_sha256
    assert context.occupied_support_sha256 == snapshot.occupied_support_sha256
    assert context.runner_execution_identity_sha256 == RUNNER_IDENTITY.content_sha256
    assert context.checkpoint_file_sha256 == CHECKPOINT_SHA256
    assert context.camera_calibration_sha256 == CALIBRATION_SHA256
    assert context.raw_outcome_file_sha256 == outcome.raw_outcome_file_sha256
    assert context.raw_outcome_content_sha256 == outcome.raw_outcome_content_sha256
    assert (20, 20) in context.excluded_target_configuration_cells
    assert (20, 20) not in context.candidate_domain
    assert context.candidate_domain <= component.cells


@pytest.mark.parametrize("kind", ["positive", "negative"])
def test_evidence_kind_is_not_interchangeable(kind: str) -> None:
    _, _, snapshot, _, component, _, outcome, issuer = _stack(kind=kind)
    context = issuer.issue_context(snapshot, component, outcome)
    writer = issuer.open_writer(context)
    with pytest.raises(TwoResolutionTargetEvidenceRejectedError):
        (writer.issue_negative() if kind == "positive" else writer.issue_positive())


def test_negative_partial_physical_cell_visibility_is_rejected() -> None:
    _, _, snapshot, _, component, source, _, issuer = _stack()
    probability = (
        PhysicalCellProbabilityV1((30, 30), 0.9),
        PhysicalCellProbabilityV1((30, 31), 0.8),
        PhysicalCellProbabilityV1((31, 30), 0.7),
    )
    outcome = _outcome(source, snapshot, probability=probability)
    context = issuer.issue_context(snapshot, component, outcome)
    with pytest.raises(TwoResolutionTargetEvidenceRejectedError, match="all four"):
        issuer.open_writer(context).issue_negative()


@pytest.mark.parametrize("bad_kind", ["unknown", "target"])
def test_unknown_or_target_physical_evidence_is_rejected(bad_kind: str) -> None:
    _, _, snapshot, _, component, source, _, issuer = _stack()
    evidence_cell = (30, 30)
    unknown = frozenset({evidence_cell}) if bad_kind == "unknown" else frozenset()
    targets = (
        frozenset({evidence_cell, (40, 40)})
        if bad_kind == "target"
        else frozenset({(40, 40)})
    )
    outcome = _outcome(
        source,
        snapshot,
        unknown_cells=unknown,
        target_cells=targets,
    )
    with pytest.raises(TwoResolutionTargetEvidenceRejectedError, match="UNKNOWN or target"):
        issuer.issue_context(snapshot, component, outcome)


def test_physical_cell_mapping_to_configuration_unknown_is_rejected() -> None:
    boundary = (30, 30)
    _, projection, snapshot, planner, component = _g3(
        unknown_physical_cells=frozenset({boundary}),
        component_start=(24, 24),
    )
    assert snapshot.state((15, 15)) is PhysicalLabel.UNKNOWN
    source = SyntheticV5TargetOutcomeIssuerV1(_synthetic_test_fixture=True)
    issuer = TwoResolutionTargetEvidenceIssuerV1(
        projection=projection,
        planner=planner,
        outcome_source=source,
        runner_execution_identity=RUNNER_IDENTITY,
        checkpoint_file_sha256=CHECKPOINT_SHA256,
        camera_calibration_sha256=CALIBRATION_SHA256,
        _synthetic_test_fixture=True,
    )
    outcome = _outcome(
        source,
        snapshot,
        probability=(PhysicalCellProbabilityV1(boundary, 0.9),),
        visible_cells=frozenset({boundary}),
    )
    with pytest.raises(TwoResolutionTargetEvidenceRejectedError, match="UNKNOWN"):
        issuer.issue_context(snapshot, component, outcome)


@pytest.mark.parametrize(
    ("checkpoint", "calibration"),
    [(_hash("wrong-checkpoint"), CALIBRATION_SHA256),
     (CHECKPOINT_SHA256, _hash("wrong-calibration"))],
)
def test_wrong_checkpoint_or_calibration_is_rejected(
    checkpoint: str,
    calibration: str,
) -> None:
    _, _, snapshot, _, component, source, _, issuer = _stack()
    outcome = _outcome(
        source,
        snapshot,
        checkpoint_sha256=checkpoint,
        calibration_sha256=calibration,
    )
    with pytest.raises(TwoResolutionTargetEvidenceBindingError, match="checkpoint/calibration"):
        issuer.issue_context(snapshot, component, outcome)


def test_wrong_runner_execution_identity_is_rejected() -> None:
    _, _, snapshot, _, component, source, _, issuer = _stack()
    wrong = V5RunnerExecutionIdentityV1(
        _hash("wrong-wrapper"),
        _hash("wrong-launcher"),
        _hash("wrong-core"),
    )
    outcome = _outcome(source, snapshot, runner_identity=wrong)
    with pytest.raises(TwoResolutionTargetEvidenceBindingError, match="runner execution"):
        issuer.issue_context(snapshot, component, outcome)


def test_wrong_frame_and_origin_bound_outcome_is_rejected() -> None:
    _, _, snapshot, _, component, source, _, issuer = _stack()
    _, _, other_snapshot, _, _ = _g3(
        "other-frame",
        origin=(-2.0, -2.0),
    )
    outcome = _outcome(source, other_snapshot)
    with pytest.raises(TwoResolutionTargetEvidenceBindingError, match="snapshot revision/frame"):
        issuer.issue_context(snapshot, component, outcome)


def test_wrong_revision_bound_outcome_is_rejected() -> None:
    _, projection, snapshot, planner, _, source, _, issuer = _stack()
    old_outcome = _outcome(source, snapshot)
    new_snapshot = projection.project()
    new_component = planner.connected_component(new_snapshot, (15, 15))
    with pytest.raises(TwoResolutionTargetEvidenceBindingError, match="snapshot revision/frame"):
        issuer.issue_context(new_snapshot, new_component, old_outcome)


def test_wrong_two_to_one_shape_or_origin_is_rejected_before_conversion() -> None:
    physical = MapFrameIdentity("physical", (0.0, 0.0), 0.05, "physical")
    configuration = MapFrameIdentity("configuration", (0.0, 0.0), 0.10, "configuration")
    with pytest.raises(TwoResolutionTargetEvidenceBindingError, match="exactly 2x"):
        physical_cell_to_configuration_cell(
            physical_frame=physical,
            configuration_frame=configuration,
            physical_shape=(65, 64),
            configuration_shape=(32, 32),
            physical_cell=(30, 30),
        )
    shifted = MapFrameIdentity("shifted", (0.05, 0.0), 0.10, "configuration")
    with pytest.raises(TwoResolutionTargetEvidenceBindingError, match="shared origin"):
        physical_cell_to_configuration_cell(
            physical_frame=physical,
            configuration_frame=shifted,
            physical_shape=(64, 64),
            configuration_shape=(32, 32),
            physical_cell=(30, 30),
        )


def test_snapshot_component_outcome_and_context_copies_are_rejected() -> None:
    _, _, snapshot, _, component, source, outcome, issuer = _stack()
    with pytest.raises(SnapshotBindingError, match="exact live"):
        issuer.issue_context(copy.copy(snapshot), component, outcome)
    with pytest.raises(SnapshotBindingError, match="exact live"):
        issuer.issue_context(snapshot, copy.copy(component), outcome)
    with pytest.raises(TwoResolutionTargetEvidenceBindingError, match="exact live"):
        issuer.issue_context(snapshot, component, copy.copy(outcome))
    context = issuer.issue_context(snapshot, component, outcome)
    with pytest.raises(TwoResolutionTargetEvidenceBindingError, match="exact live"):
        issuer.open_writer(copy.copy(context))


def test_wrong_support_component_clone_is_rejected() -> None:
    _, _, snapshot, _, component, _, outcome, issuer = _stack()
    wrong_component = replace(
        component,
        free_support_sha256=_hash("wrong-free-support"),
    )
    assert type(wrong_component) is ConfigurationComponentV2
    with pytest.raises(SnapshotBindingError, match="exact live"):
        issuer.issue_context(snapshot, wrong_component, outcome)


def test_outcome_context_writer_and_evidence_are_single_use() -> None:
    _, _, snapshot, _, component, _, outcome, issuer = _stack()
    context = issuer.issue_context(snapshot, component, outcome)
    with pytest.raises(TwoResolutionTargetEvidenceReplayError, match="already consumed"):
        issuer.issue_context(snapshot, component, outcome)
    writer = issuer.open_writer(context)
    with pytest.raises(TwoResolutionTargetEvidenceReplayError, match="single writer"):
        issuer.open_writer(context)
    evidence = writer.issue_negative()
    with pytest.raises(TwoResolutionTargetEvidenceReplayError, match="single-use"):
        writer.issue_negative()
    issuer.consume_evidence(evidence)
    with pytest.raises(TwoResolutionTargetEvidenceReplayError, match="already consumed"):
        issuer.consume_evidence(evidence)
    with pytest.raises(TwoResolutionTargetEvidenceBindingError, match="exact live"):
        issuer.consume_evidence(copy.copy(evidence))


def test_context_becomes_stale_when_g3_projection_advances() -> None:
    _, projection, snapshot, _, component, _, outcome, issuer = _stack()
    context = issuer.issue_context(snapshot, component, outcome)
    projection.project()
    with pytest.raises(SnapshotBindingError):
        issuer.open_writer(context)


def test_production_identities_are_none_and_synthetic_authority_is_explicit() -> None:
    assert (
        PRODUCTION_G3_V2_SNAPSHOT_SOURCE,
        PRODUCTION_G3_V2_COMPONENT_SOURCE,
        PRODUCTION_V5_RUNNER_EXECUTION_IDENTITY,
        PRODUCTION_V5_CHECKPOINT_FILE_SHA256,
        PRODUCTION_V5_RAW_OUTCOME_SOURCE,
        PRODUCTION_V5_CAMERA_CALIBRATION_SHA256,
        PRODUCTION_G5_TWO_RESOLUTION_EVIDENCE_ISSUER,
    ) == (None,) * 7
    with pytest.raises(PermissionError, match="identities are unset"):
        require_production_two_resolution_target_evidence_issuer()
    with pytest.raises(PermissionError, match="test-only"):
        SyntheticV5TargetOutcomeIssuerV1()
    _, projection, _, planner, _ = _g3()
    source = SyntheticV5TargetOutcomeIssuerV1(_synthetic_test_fixture=True)
    with pytest.raises(PermissionError, match="no production"):
        TwoResolutionTargetEvidenceIssuerV1(
            projection=projection,
            planner=planner,
            outcome_source=source,
            runner_execution_identity=RUNNER_IDENTITY,
            checkpoint_file_sha256=CHECKPOINT_SHA256,
            camera_calibration_sha256=CALIBRATION_SHA256,
        )


def test_issuer_and_writer_are_noncopyable_and_no_accelerator_surface_exists() -> None:
    _, _, snapshot, _, component, source, outcome, issuer = _stack()
    with pytest.raises(TypeError, match="non-copyable"):
        copy.copy(source)
    with pytest.raises(TypeError, match="non-copyable"):
        copy.copy(issuer)
    context = issuer.issue_context(snapshot, component, outcome)
    writer = issuer.open_writer(context)
    with pytest.raises(TypeError, match="non-copyable"):
        copy.copy(writer)
    source_path = __import__(
        "lewm.planning.two_resolution_target_evidence_v1",
        fromlist=["ignored"],
    ).__file__
    assert source_path is not None
    text = Path(source_path).read_text(encoding="utf-8")
    for forbidden in ("torch", "cuda", "rocm", "numpy", "checkpoint load"):
        assert forbidden not in text.lower()
