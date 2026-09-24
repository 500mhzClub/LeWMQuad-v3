from __future__ import annotations

import copy
from dataclasses import replace
import hashlib

import numpy as np
import pytest

from lewm.benchmarks.go2_g3_exact_physical_equivalence_v2 import (
    _build_projected_snapshot,
)
from lewm.planning.revisioned_physical_configuration_memory import (
    PhysicalLabel,
    StaleSnapshotError,
)
from lewm.planning.two_resolution_reversible_target_belief_v1 import (
    TwoResolutionReversibleTargetBeliefMemoryV1,
    TwoResolutionTargetMemoryBindingError,
    TwoResolutionTargetMemoryConfigV1,
    TwoResolutionTargetMemoryReplayError,
    require_production_two_resolution_target_memory,
)
from lewm.planning.two_resolution_target_evidence_v1 import (
    PhysicalCellProbabilityV1,
    SyntheticV5TargetOutcomeIssuerV1,
    TwoResolutionTargetEvidenceIssuerV1,
    V5RunnerExecutionIdentityV1,
)
from lewm_worlds.manifest import CameraValidityConstraints, SceneManifest, SpawnSpec


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


RUNNER = V5RunnerExecutionIdentityV1(
    entrypoint_wrapper_file_sha256=_hash("posterior-runner-wrapper"),
    captured_launcher_file_sha256=_hash("posterior-runner-launcher"),
    captured_core_file_sha256=_hash("posterior-runner-core"),
)
CHECKPOINT = _hash("posterior-checkpoint")
CALIBRATION = _hash("posterior-calibration")


def _manifest() -> SceneManifest:
    return SceneManifest(
        scene_id="two-resolution-posterior",
        family="unit",
        difficulty_tier="unit",
        topology_seed=1,
        visual_seed=2,
        physics_seed=3,
        world_bounds_xy_m=((-1.6, -1.6), (1.6, 1.6)),
        spawn=SpawnSpec(
            xyz_m=(0.0, 0.0, 0.35),
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


def _stack():
    labels = np.full((64, 64), int(PhysicalLabel.FREE), dtype=np.uint8)
    _, snapshot, planner = _build_projected_snapshot(
        _manifest(),
        labels,
        origin_xy_m=(-1.6, -1.6),
        configuration_shape=(32, 32),
    )
    projection = getattr(planner, "_projection")
    component = planner.connected_component(snapshot, (15, 15))
    outcome_source = SyntheticV5TargetOutcomeIssuerV1(
        _synthetic_test_fixture=True
    )
    evidence_issuer = TwoResolutionTargetEvidenceIssuerV1(
        projection=projection,
        planner=planner,
        outcome_source=outcome_source,
        runner_execution_identity=RUNNER,
        checkpoint_file_sha256=CHECKPOINT,
        camera_calibration_sha256=CALIBRATION,
        _synthetic_test_fixture=True,
    )
    memory = TwoResolutionReversibleTargetBeliefMemoryV1(
        evidence_issuer=evidence_issuer,
        _synthetic_development_fixture=True,
    )
    return snapshot, component, outcome_source, evidence_issuer, memory


def _issue(
    snapshot,
    component,
    outcome_source,
    evidence_issuer,
    *,
    kind: str,
    rows: tuple[PhysicalCellProbabilityV1, ...] | None = None,
    confidence: float = 0.85,
):
    if rows is None:
        if kind == "positive":
            rows = (
                PhysicalCellProbabilityV1((30, 30), 0.4),
                PhysicalCellProbabilityV1((31, 30), 0.3),
                PhysicalCellProbabilityV1((32, 30), 0.2),
            )
        else:
            rows = (
                PhysicalCellProbabilityV1((30, 30), 0.9),
                PhysicalCellProbabilityV1((30, 31), 0.8),
                PhysicalCellProbabilityV1((31, 30), 0.7),
                PhysicalCellProbabilityV1((31, 31), 0.6),
            )
    visible = frozenset(row.cell for row in rows)
    sequence = getattr(outcome_source, "_sequence") + 1
    outcome = outcome_source.issue(
        snapshot=snapshot,
        outcome_kind=kind,
        target_id="blue",
        pose_timestamp_ns=17 + sequence,
        runner_execution_identity=RUNNER,
        checkpoint_file_sha256=CHECKPOINT,
        raw_outcome_file_sha256=_hash(f"posterior-raw-{sequence}"),
        camera_calibration_sha256=CALIBRATION,
        pose_provenance_sha256=_hash(f"posterior-pose-{sequence}"),
        free_physical_cells=visible,
        unknown_physical_cells=frozenset(),
        target_physical_cells=frozenset({(40, 40)}),
        visible_physical_cells=visible,
        physical_probability=rows,
        unlocalized_probability=0.1 if kind == "positive" else 0.0,
        confidence=confidence,
    )
    context = evidence_issuer.issue_context(snapshot, component, outcome)
    writer = evidence_issuer.open_writer(context)
    evidence = writer.issue_positive() if kind == "positive" else writer.issue_negative()
    return context, evidence


def test_positive_negative_and_later_positive_are_reversible_on_configuration_grid() -> None:
    snapshot, component, source, issuer, memory = _stack()
    positive_context, positive = _issue(
        snapshot, component, source, issuer, kind="positive"
    )
    first = memory.apply(positive_context, positive)
    first_mass = {row.cell: row.value for row in first.cell_mass}
    assert first_mass[(15, 15)] == pytest.approx(0.2975)
    assert first_mass[(16, 15)] == pytest.approx(0.085)
    assert first.unlocalized_mass == pytest.approx(0.6175)
    assert first.configuration_map_frame.cell_size_m == pytest.approx(0.10)
    assert first.physical_map_frame.cell_size_m == pytest.approx(0.05)

    negative_context, negative = _issue(
        snapshot, component, source, issuer, kind="negative"
    )
    second = memory.apply(negative_context, negative)
    second_mass = {row.cell: row.value for row in second.cell_mass}
    assert second_mass[(15, 15)] == pytest.approx(first_mass[(15, 15)] * 0.49)
    assert second_mass[(16, 15)] == pytest.approx(first_mass[(16, 15)])
    assert second.unlocalized_mass > first.unlocalized_mass

    later_context, later = _issue(
        snapshot, component, source, issuer, kind="positive"
    )
    third = memory.apply(later_context, later)
    third_mass = {row.cell: row.value for row in third.cell_mass}
    assert third_mass[(15, 15)] > second_mass[(15, 15)]
    assert third.positive_evidence_count == 2
    assert third.negative_evidence_count == 1
    assert sum(third_mass.values()) + third.unlocalized_mass == pytest.approx(1.0)


def test_evidence_is_single_use_and_snapshot_is_exact_instance_and_revision_bound() -> None:
    snapshot, component, source, issuer, memory = _stack()
    context, evidence = _issue(snapshot, component, source, issuer, kind="positive")
    first = memory.apply(context, evidence)
    memory.assert_current_snapshot(first)
    with pytest.raises(TypeError):
        copy.copy(first)
    with pytest.raises(TwoResolutionTargetMemoryBindingError):
        memory.assert_current_snapshot(replace(first))
    with pytest.raises(TwoResolutionTargetMemoryReplayError):
        memory.apply(context, evidence)

    context2, evidence2 = _issue(snapshot, component, source, issuer, kind="negative")
    memory.apply(context2, evidence2)
    with pytest.raises(StaleSnapshotError):
        memory.assert_current_snapshot(first)


def test_separated_modes_remain_separate_deterministic_hypotheses() -> None:
    snapshot, component, source, issuer, memory = _stack()
    rows = (
        PhysicalCellProbabilityV1((30, 30), 0.4),
        PhysicalCellProbabilityV1((31, 30), 0.3),
        PhysicalCellProbabilityV1((50, 50), 0.2),
    )
    context, evidence = _issue(
        snapshot,
        component,
        source,
        issuer,
        kind="positive",
        rows=rows,
    )
    posterior = memory.apply(context, evidence)
    hypotheses = memory.hypotheses(posterior)
    assert [hypothesis.cells for hypothesis in hypotheses] == [
        frozenset({(15, 15)}),
        frozenset({(25, 25)}),
    ]
    assert hypotheses[0].mass > hypotheses[1].mass
    assert hypotheses[0].peak_cell == (15, 15)
    assert hypotheses[1].peak_cell == (25, 25)


def test_repeated_negative_updates_never_delete_a_mode() -> None:
    snapshot, component, source, issuer, memory = _stack()
    context, evidence = _issue(snapshot, component, source, issuer, kind="positive")
    posterior = memory.apply(context, evidence)
    for _ in range(24):
        context, evidence = _issue(
            snapshot,
            component,
            source,
            issuer,
            kind="negative",
            confidence=1.0,
        )
        posterior = memory.apply(context, evidence)
    mass = {row.cell: row.value for row in posterior.cell_mass}
    assert mass[(15, 15)] >= memory._config.posterior_mass_floor
    assert mass[(15, 15)] > 0.0
    assert mass[(16, 15)] > 0.0


def test_wrong_binding_and_copied_evidence_fail_before_posterior_mutation() -> None:
    snapshot, component, source, issuer, memory = _stack()
    context, evidence = _issue(snapshot, component, source, issuer, kind="positive")
    copied = replace(evidence)
    with pytest.raises(TwoResolutionTargetMemoryBindingError):
        memory.apply(context, copied)
    assert memory._revision == 0
    assert memory._unlocalized["blue"] == 1.0

    wrong_context = replace(context, checkpoint_file_sha256=_hash("wrong"))
    with pytest.raises(Exception):
        memory.apply(wrong_context, evidence)
    assert memory._revision == 0


def test_authority_is_explicitly_development_only_and_production_fails_closed() -> None:
    _, _, _, issuer, _ = _stack()
    with pytest.raises(PermissionError):
        TwoResolutionReversibleTargetBeliefMemoryV1(evidence_issuer=issuer)
    with pytest.raises(PermissionError):
        require_production_two_resolution_target_memory()
    config = TwoResolutionTargetMemoryConfigV1()
    assert config.to_dict()["production_promotion_authorized"] is False
    assert config.to_dict()["hardware_execution_authorized"] is False

