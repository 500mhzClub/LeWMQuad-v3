"""Tests for held-out navigation benchmark integrity helpers."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from lewm.benchmarks.generalization_protocol import (
    AuditedSceneRecord,
    SceneDisjointManifests,
    SceneSplitCounts,
    StrictClaimObservation,
    audited_scene_record,
    build_scene_disjoint_manifests,
    fixed_spawn_audit_config_from_geometry_contract,
    reachable_area_normalized_coverage,
    strict_ground_truth_claim,
    summarize_strict_ground_truth_claims,
    verify_scene_disjoint_manifests,
    write_scene_disjoint_manifests,
)
from lewm.planning.geometry_contract import load_geometry_contract
from lewm_worlds.fixed_spawn_audit import audit_fixed_spawn
from lewm_worlds.manifest import (
    BoxObject,
    CameraValidityConstraints,
    GraphNode,
    SceneManifest,
    SpawnSpec,
    manifest_sha256,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def geometry_contract():
    return load_geometry_contract(
        repository_root=REPOSITORY_ROOT,
        verify_sources=False,
    )


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _record(
    family: str,
    index: int,
    *,
    audit_config: dict[str, object],
    fully_reachable: bool = True,
) -> AuditedSceneRecord:
    scene_id = f"{family}_{index:02d}"
    return AuditedSceneRecord(
        scene_id=scene_id,
        family=family,
        topology_seed=index,
        source_split="candidate",
        manifest_sha256=_digest(f"manifest:{scene_id}"),
        audit_sha256=_digest(f"audit:{scene_id}"),
        audit_config=audit_config,
        fully_reachable=fully_reachable,
        reachable_area_m2=12.5,
        beacon_count=4,
        beacons_with_preferred_standoff=3,
        failure_reason="" if fully_reachable else "unreachable",
    )


def _open_manifest() -> SceneManifest:
    beacon = BoxObject(
        object_id="beacon",
        kind="landmark",
        center_xyz_m=(0.9, 0.0, 0.3),
        size_xyz_m=(0.2, 0.2, 0.6),
        yaw_rad=0.0,
        material_id="test",
    )
    return SceneManifest(
        scene_id="coverage_toy",
        family="test",
        difficulty_tier="test",
        topology_seed=0,
        visual_seed=0,
        physics_seed=0,
        world_bounds_xy_m=((-2.0, -2.0), (2.0, 2.0)),
        spawn=SpawnSpec(
            xyz_m=(0.0, 0.0, 0.375),
            quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        ),
        graph_nodes=(GraphNode(0, (0.0, 0.0), 1.0),),
        graph_edges=(),
        obstacles=(),
        landmarks=(beacon,),
        camera_constraints=CameraValidityConstraints(0.08, 0.05, 20.0, 0.10),
    )


def test_strict_claim_uses_inclusive_true_radius_and_line_of_sight() -> None:
    on_boundary = StrictClaimObservation("a", (0.0, 0.0), (1.2, 0.0), True)
    outside = StrictClaimObservation("b", (0.0, 0.0), (1.200001, 0.0), True)
    occluded = StrictClaimObservation("c", (0.0, 0.0), (1.0, 0.0), False)

    assert strict_ground_truth_claim(on_boundary, claim_radius_m=1.2).accepted
    assert not strict_ground_truth_claim(outside, claim_radius_m=1.2).accepted
    assert not strict_ground_truth_claim(occluded, claim_radius_m=1.2).accepted


def test_strict_claim_summary_counts_each_target_once() -> None:
    observations = (
        StrictClaimObservation("a", (0.0, 0.0), (1.0, 0.0), True),
        StrictClaimObservation("a", (0.1, 0.0), (1.0, 0.0), True),
        StrictClaimObservation("b", (0.0, 0.0), (0.5, 0.0), False),
    )

    summary = summarize_strict_ground_truth_claims(
        observations,
        claim_radius_m=1.2,
    )

    assert summary.observation_count == 3
    assert summary.accepted_observation_count == 2
    assert summary.claimed_target_ids == ("a",)


def test_audited_record_binds_manifest_and_audit_content(geometry_contract) -> None:
    manifest = _open_manifest()
    audit = audit_fixed_spawn(
        manifest,
        config=fixed_spawn_audit_config_from_geometry_contract(geometry_contract),
    )

    record = audited_scene_record(manifest, audit)

    assert record.manifest_sha256 == manifest_sha256(manifest)
    assert len(record.audit_sha256) == 64
    assert record.audit_config == audit.to_dict()["config"]
    assert record.fully_reachable
    assert record.beacons_with_preferred_standoff == 1


def test_coverage_uses_unique_reachable_10cm_cells(geometry_contract) -> None:
    config = fixed_spawn_audit_config_from_geometry_contract(geometry_contract)
    audit = audit_fixed_spawn(_open_manifest(), config=config)
    first = min(audit.coverage_reachable_cells)
    cell_size = config.coverage_cell_size_m

    def center(cell: tuple[int, int]) -> tuple[float, float]:
        return (
            audit.coverage_grid_origin_xy_m[0] + (cell[0] + 0.5) * cell_size,
            audit.coverage_grid_origin_xy_m[1] + (cell[1] + 0.5) * cell_size,
        )

    samples = (center(first), center(first))
    metric = reachable_area_normalized_coverage(samples, audit=audit)

    assert metric.pose_sample_count == 2
    assert metric.unique_pose_cell_count == 1
    assert metric.unique_swept_cell_count == 1
    assert metric.visited_reachable_cell_count == 1
    assert metric.reachable_cell_count == audit.coverage_reachable_cell_count
    assert metric.visited_reachable_area_m2 == pytest.approx(0.01)
    assert metric.fraction == pytest.approx(1 / audit.coverage_reachable_cell_count)

    outside = reachable_area_normalized_coverage(
        ((20.0, 20.0), (20.0, 20.0)),
        audit=audit,
    )
    assert outside.unique_pose_cell_count == 1
    assert outside.unique_swept_cell_count == 1
    assert outside.visited_reachable_cell_count == 0


def test_coverage_supercovers_cells_between_sparse_poses(geometry_contract) -> None:
    config = fixed_spawn_audit_config_from_geometry_contract(geometry_contract)
    audit = audit_fixed_spawn(_open_manifest(), config=config)
    cells = set(audit.coverage_reachable_cells)
    start = next(
        (x, y)
        for y in range(audit.coverage_grid_shape[1])
        for x in range(audit.coverage_grid_shape[0] - 3)
        if all((x + offset, y) in cells for offset in range(4))
    )
    cell_size = config.coverage_cell_size_m

    def center(cell: tuple[int, int]) -> tuple[float, float]:
        return (
            audit.coverage_grid_origin_xy_m[0] + (cell[0] + 0.5) * cell_size,
            audit.coverage_grid_origin_xy_m[1] + (cell[1] + 0.5) * cell_size,
        )

    metric = reachable_area_normalized_coverage(
        (center(start), center((start[0] + 3, start[1]))),
        audit=audit,
    )

    assert metric.pose_sample_count == 2
    assert metric.unique_pose_cell_count == 2
    assert metric.unique_swept_cell_count == 4
    assert metric.visited_reachable_cell_count == 4
    assert metric.visited_reachable_area_m2 == pytest.approx(0.04)


def test_split_is_order_independent_disjoint_and_sealed(geometry_contract) -> None:
    config = asdict(fixed_spawn_audit_config_from_geometry_contract(geometry_contract))
    records = [
        *(_record("alpha", index, audit_config=config) for index in range(5)),
        *(_record("beta", index, audit_config=config) for index in range(5)),
        _record("alpha", 99, audit_config=config, fully_reachable=False),
    ]
    allocations = {
        "alpha": SceneSplitCounts(validation=1, sealed_test=1),
        "beta": SceneSplitCounts(validation=1, sealed_test=1),
    }

    first = build_scene_disjoint_manifests(
        records,
        benchmark_id="generalization-v1",
        split_seed=20260709,
        geometry_contract=geometry_contract,
        allocations=allocations,
    )
    second = build_scene_disjoint_manifests(
        list(reversed(records)),
        benchmark_id="generalization-v1",
        split_seed=20260709,
        geometry_contract=geometry_contract,
        allocations=allocations,
    )

    assert first == second
    assert verify_scene_disjoint_manifests(first)["passes"]
    assert len(first.development["train_scenes"]) == 6
    assert len(first.development["validation_scenes"]) == 2
    assert len(first.development["excluded_scenes"]) == 1
    assert len(first.sealed_test["scenes"]) == 2
    assert "split_seed" not in first.development
    development_json = json.dumps(first.development, sort_keys=True)
    for scene in first.sealed_test["scenes"]:
        assert scene["scene_id"] not in development_json
    assert (
        first.development["geometry_contract_sha256"]
        == geometry_contract.sha256
        == first.sealed_test["geometry_contract_sha256"]
    )


def test_split_verifier_rejects_tampered_sealed_membership(
    geometry_contract,
) -> None:
    config = asdict(fixed_spawn_audit_config_from_geometry_contract(geometry_contract))
    records = [
        _record("alpha", index, audit_config=config)
        for index in range(3)
    ]
    manifests = build_scene_disjoint_manifests(
        records,
        benchmark_id="generalization-v1",
        split_seed=11,
        geometry_contract=geometry_contract,
        allocations={"alpha": SceneSplitCounts(validation=1, sealed_test=1)},
    )
    tampered_payload = copy.deepcopy(dict(manifests.sealed_test))
    tampered_payload["scenes"][0]["scene_id"] = "substituted-scene"
    tampered = SceneDisjointManifests(
        development=manifests.development,
        sealed_test=tampered_payload,
    )

    verification = verify_scene_disjoint_manifests(tampered)

    assert not verification["passes"]
    assert "sealed_payload_commitment_mismatch" in verification["errors"]
    assert "development_commitment_mismatch" in verification["errors"]


def test_split_rejects_duplicate_topology_and_wrong_geometry(geometry_contract) -> None:
    correct = asdict(fixed_spawn_audit_config_from_geometry_contract(geometry_contract))
    duplicate = [
        _record("alpha", 1, audit_config=correct),
        AuditedSceneRecord(
            **{
                **asdict(_record("alpha", 2, audit_config=correct)),
                "topology_seed": 1,
            }
        ),
    ]
    with pytest.raises(ValueError, match="duplicate family/topology seed"):
        build_scene_disjoint_manifests(
            duplicate,
            benchmark_id="generalization-v1",
            split_seed=1,
            geometry_contract=geometry_contract,
            allocations={"alpha": SceneSplitCounts(validation=0, sealed_test=0)},
        )

    wrong = dict(correct)
    wrong["body_radius_m"] = 0.24
    with pytest.raises(ValueError, match="does not match the geometry contract"):
        build_scene_disjoint_manifests(
            [_record("alpha", 1, audit_config=wrong)],
            benchmark_id="generalization-v1",
            split_seed=1,
            geometry_contract=geometry_contract,
            allocations={"alpha": SceneSplitCounts(validation=0, sealed_test=0)},
        )


def test_separate_manifest_writer_refuses_overwrite(
    tmp_path: Path,
    geometry_contract,
) -> None:
    config = asdict(fixed_spawn_audit_config_from_geometry_contract(geometry_contract))
    manifests = build_scene_disjoint_manifests(
        [_record("alpha", 1, audit_config=config)],
        benchmark_id="generalization-v1",
        split_seed=3,
        geometry_contract=geometry_contract,
        allocations={"alpha": SceneSplitCounts(validation=0, sealed_test=0)},
    )
    development_path = tmp_path / "development.json"
    sealed_path = tmp_path / "sealed" / "test.json"

    write_scene_disjoint_manifests(
        manifests,
        development_path=development_path,
        sealed_test_path=sealed_path,
    )

    assert development_path.is_file()
    assert sealed_path.is_file()
    with pytest.raises(FileExistsError):
        write_scene_disjoint_manifests(
            manifests,
            development_path=development_path,
            sealed_test_path=sealed_path,
        )
