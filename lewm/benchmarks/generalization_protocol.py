"""Integrity helpers for scene-disjoint navigation generalization benchmarks."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from lewm.planning.geometry_contract import GeometryContract
from lewm_worlds.fixed_spawn_audit import (
    FixedSpawnAuditConfig,
    FixedSpawnAuditReport,
)
from lewm_worlds.manifest import SceneManifest, manifest_sha256


_DEVELOPMENT_SCHEMA = "lewm_navigation_development_manifest_v0"
_SEALED_TEST_SCHEMA = "lewm_navigation_sealed_test_manifest_v0"
_SPLIT_RANK_DOMAIN = "lewm-navigation-scene-split-v0"
_SCENE_ROLE_COMMITMENT_DOMAIN = "lewm-navigation-scene-role-v1"


@dataclass(frozen=True)
class StrictClaimObservation:
    """Ground-truth state at one proposed beacon-claim event."""

    target_id: str
    robot_xy_m: tuple[float, float]
    target_xy_m: tuple[float, float]
    line_of_sight: bool


@dataclass(frozen=True)
class StrictClaimResult:
    """Result of applying the physical claim contract to one observation."""

    target_id: str
    distance_m: float
    within_claim_radius: bool
    line_of_sight: bool
    accepted: bool


@dataclass(frozen=True)
class StrictClaimSummary:
    """Unique-target aggregate for a sequence of proposed claims."""

    observation_count: int
    accepted_observation_count: int
    claimed_target_ids: tuple[str, ...]


@dataclass(frozen=True)
class ReachableCoverageMetric:
    """Coverage normalized by the fixed spawn's reachable component."""

    pose_sample_count: int
    unique_pose_cell_count: int
    unique_swept_cell_count: int
    visited_reachable_cell_count: int
    reachable_cell_count: int
    visited_reachable_area_m2: float
    reachable_area_m2: float
    fraction: float


@dataclass(frozen=True)
class SceneSplitCounts:
    """Explicit per-family allocation; the remainder is training data."""

    validation: int
    sealed_test: int

    def validate(self) -> None:
        if self.validation < 0 or self.sealed_test < 0:
            raise ValueError("scene split counts must be non-negative")


@dataclass(frozen=True)
class AuditedSceneRecord:
    """Immutable identity and fixed-spawn audit summary for one candidate."""

    scene_id: str
    family: str
    topology_seed: int
    source_split: str | None
    manifest_sha256: str
    audit_sha256: str
    audit_config: Mapping[str, Any]
    fully_reachable: bool
    reachable_area_m2: float
    beacon_count: int
    beacons_with_preferred_standoff: int
    failure_reason: str
    physical_eligible: bool | None = None
    physical_eligibility_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "scene_id": self.scene_id,
            "family": self.family,
            "topology_seed": self.topology_seed,
            "source_split": self.source_split,
            "manifest_sha256": self.manifest_sha256,
            "audit_sha256": self.audit_sha256,
            "fully_reachable": self.fully_reachable,
            "reachable_area_m2": self.reachable_area_m2,
            "beacon_count": self.beacon_count,
            "beacons_with_preferred_standoff": (
                self.beacons_with_preferred_standoff
            ),
            "failure_reason": self.failure_reason,
        }
        if self.physical_eligible is not None:
            payload["physical_eligible"] = self.physical_eligible
            payload["physical_eligibility_sha256"] = (
                self.physical_eligibility_sha256
            )
        return payload


@dataclass(frozen=True)
class SceneDisjointManifests:
    """Separate development payload and committed sealed-test payload."""

    development: Mapping[str, Any]
    sealed_test: Mapping[str, Any]


def build_hashed_scene_role_commitment(
    manifests: SceneDisjointManifests,
) -> dict[str, Any]:
    """Commit split roles without copying raw scene IDs outside manifests.

    Downstream dataset builders can hash a candidate ID under each role and
    reject development or sealed matches without ever opening the sealed
    manifest.  The benchmark ID is part of the hash domain, preventing tokens
    from being compared across independently frozen benchmarks by accident.
    """

    development = dict(manifests.development)
    sealed = dict(manifests.sealed_test)
    benchmark_id = str(development.get("benchmark_id", ""))
    if not benchmark_id or benchmark_id != str(sealed.get("benchmark_id", "")):
        raise ValueError("scene-role commitment requires matching benchmark IDs")
    role_entries = {
        "train": list(development.get("train_scenes", [])),
        "development": list(development.get("validation_scenes", [])),
        "sealed_test": list(sealed.get("scenes", [])),
        "excluded": list(development.get("excluded_scenes", [])),
    }

    def token(role: str, scene_id: str) -> str:
        encoded = "\0".join(
            (_SCENE_ROLE_COMMITMENT_DOMAIN, benchmark_id, role, scene_id)
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    role_tokens: dict[str, list[str]] = {}
    plain_hashes: dict[str, list[str]] = {}
    seen_ids: set[str] = set()
    for role, entries in role_entries.items():
        tokens: list[str] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ValueError(f"malformed {role} scene entry")
            scene_id = str(entry.get("scene_id", ""))
            if not scene_id or scene_id in seen_ids:
                raise ValueError("scene roles require unique non-empty scene IDs")
            seen_ids.add(scene_id)
            tokens.append(token(role, scene_id))
        role_tokens[role] = sorted(tokens)
        plain_hashes[role] = sorted(
            hashlib.sha256(str(entry["scene_id"]).encode("utf-8")).hexdigest()
            for entry in entries
        )
    set_hashes = {
        role: {
            "role_tokens_sha256": _sha256_payload(
                {"tokens": role_tokens[role]}
            ),
            "scene_id_sha256": _sha256_payload(
                {"scene_id_sha256": plain_hashes[role]}
            ),
        }
        for role in role_tokens
    }
    core = {
        "schema": "lewm_navigation_hashed_scene_roles_v1",
        "hash_domain": _SCENE_ROLE_COMMITMENT_DOMAIN,
        "benchmark_id": benchmark_id,
        "geometry_contract_sha256": development.get(
            "geometry_contract_sha256"
        ),
        "sealed_test_commitment_sha256": sealed.get("commitment_sha256"),
        "roles": role_tokens,
        "scene_id_sha256_by_role": plain_hashes,
        "set_sha256_by_role": set_hashes,
        "counts": {role: len(tokens) for role, tokens in role_tokens.items()},
    }
    return {**core, "content_sha256": _sha256_payload(core)}


def scene_role_token(
    scene_id: str,
    *,
    role: str,
    benchmark_id: str,
) -> str:
    """Hash one scene identity exactly as the role commitment does."""

    if not scene_id or not role or not benchmark_id:
        raise ValueError("scene_id, role, and benchmark_id must be non-empty")
    encoded = "\0".join(
        (_SCENE_ROLE_COMMITMENT_DOMAIN, benchmark_id, role, scene_id)
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def scene_id_sha256(scene_id: str) -> str:
    """Return the plain one-way ID used by leak-screening membership files."""

    if not scene_id:
        raise ValueError("scene_id must be non-empty")
    return hashlib.sha256(scene_id.encode("utf-8")).hexdigest()


def strict_ground_truth_claim(
    observation: StrictClaimObservation,
    *,
    claim_radius_m: float,
) -> StrictClaimResult:
    """Apply an inclusive, zero-tolerance physical claim-radius check."""

    radius = float(claim_radius_m)
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError("claim_radius_m must be non-negative")
    points = (*observation.robot_xy_m, *observation.target_xy_m)
    if len(observation.robot_xy_m) != 2 or len(observation.target_xy_m) != 2:
        raise ValueError("claim coordinates must be two-dimensional")
    if not all(math.isfinite(float(value)) for value in points):
        raise ValueError("claim coordinates must be finite")
    distance = math.dist(observation.robot_xy_m, observation.target_xy_m)
    within_radius = distance <= radius
    return StrictClaimResult(
        target_id=str(observation.target_id),
        distance_m=float(distance),
        within_claim_radius=bool(within_radius),
        line_of_sight=bool(observation.line_of_sight),
        accepted=bool(within_radius and observation.line_of_sight),
    )


def summarize_strict_ground_truth_claims(
    observations: Iterable[StrictClaimObservation],
    *,
    claim_radius_m: float,
) -> StrictClaimSummary:
    """Count each physically verified target at most once."""

    results = [
        strict_ground_truth_claim(item, claim_radius_m=claim_radius_m)
        for item in observations
    ]
    claimed = tuple(sorted({item.target_id for item in results if item.accepted}))
    return StrictClaimSummary(
        observation_count=len(results),
        accepted_observation_count=sum(item.accepted for item in results),
        claimed_target_ids=claimed,
    )


def reachable_area_normalized_coverage(
    trajectory_xy_m: Iterable[tuple[float, float]],
    *,
    audit: FixedSpawnAuditReport,
) -> ReachableCoverageMetric:
    """Score the swept reference path over audited reachable coverage cells."""

    if audit.coverage_reachable_cell_count <= 0:
        raise ValueError("cannot normalize coverage with an empty reachable component")
    if audit.coverage_reachable_cell_count != len(audit.coverage_reachable_cells):
        raise ValueError("coverage audit count does not match its reachable cell set")
    positions = tuple(
        (float(position[0]), float(position[1])) for position in trajectory_xy_m
    )
    if not all(math.isfinite(value) for position in positions for value in position):
        raise ValueError("trajectory coordinates must be finite")
    pose_cells = {audit.world_to_coverage_grid(position) for position in positions}
    swept_cells = set(pose_cells)
    for start_xy_m, end_xy_m in zip(positions, positions[1:]):
        swept_cells.update(
            supercover_segment_cells(
                start_xy_m,
                end_xy_m,
                origin_xy_m=audit.coverage_grid_origin_xy_m,
                cell_size_m=audit.config.coverage_cell_size_m,
            )
        )
    visited = swept_cells & set(audit.coverage_reachable_cells)
    cell_area = float(audit.config.coverage_cell_size_m**2)
    denominator = int(audit.coverage_reachable_cell_count)
    numerator = len(visited)
    return ReachableCoverageMetric(
        pose_sample_count=len(positions),
        unique_pose_cell_count=len(pose_cells),
        unique_swept_cell_count=len(swept_cells),
        visited_reachable_cell_count=numerator,
        reachable_cell_count=denominator,
        visited_reachable_area_m2=float(numerator * cell_area),
        reachable_area_m2=float(denominator * cell_area),
        fraction=float(numerator / denominator),
    )


def audited_scene_record(
    manifest: SceneManifest,
    audit: FixedSpawnAuditReport,
) -> AuditedSceneRecord:
    """Bind a scene's immutable content identity to its audit result."""

    if manifest.scene_id != audit.scene_id:
        raise ValueError(
            f"manifest/audit scene mismatch: {manifest.scene_id} != {audit.scene_id}"
        )
    audit_payload = audit.to_dict()
    return AuditedSceneRecord(
        scene_id=str(manifest.scene_id),
        family=str(manifest.family),
        topology_seed=int(manifest.topology_seed),
        source_split=manifest.split,
        manifest_sha256=str(manifest_sha256(manifest)),
        audit_sha256=_sha256_payload(audit_payload),
        audit_config=dict(audit_payload["config"]),
        fully_reachable=bool(audit.fully_reachable),
        reachable_area_m2=float(audit.coverage_reachable_area_m2),
        beacon_count=len(audit.beacons),
        beacons_with_preferred_standoff=sum(
            beacon.preferred_standoff_reachable for beacon in audit.beacons
        ),
        failure_reason=str(audit.failure_reason),
    )


def build_scene_disjoint_manifests(
    records: Sequence[AuditedSceneRecord],
    *,
    benchmark_id: str,
    split_seed: int,
    geometry_contract: GeometryContract,
    allocations: Mapping[str, SceneSplitCounts],
) -> SceneDisjointManifests:
    """Create deterministic family-stratified development and sealed manifests.

    Only scenes passing the fixed-spawn audit are eligible.  The sealed-test
    payload is separate; the development payload exposes only its count and a
    SHA-256 commitment.  Explicit counts avoid fraction-rounding drift as the
    candidate pool changes.
    """

    if not benchmark_id.strip():
        raise ValueError("benchmark_id must be non-empty")
    _validate_sha256(geometry_contract.sha256, name="geometry_contract.sha256")
    if not records:
        raise ValueError("at least one audited scene record is required")
    _validate_scene_records(records)

    families = {record.family for record in records}
    if set(allocations) != families:
        missing = sorted(families - set(allocations))
        extra = sorted(set(allocations) - families)
        raise ValueError(
            "allocations must match candidate families; "
            f"missing={missing} extra={extra}"
        )
    for allocation in allocations.values():
        allocation.validate()

    audit_configs = {_canonical_json(dict(record.audit_config)) for record in records}
    if len(audit_configs) != 1:
        raise ValueError(
            "all candidate scenes must use the same fixed-spawn audit config"
        )
    audit_config = json.loads(next(iter(audit_configs)))
    expected_audit_config = asdict(
        fixed_spawn_audit_config_from_geometry_contract(geometry_contract)
    )
    if audit_config != expected_audit_config:
        raise ValueError(
            "candidate audit config does not match the geometry contract"
        )
    allocation_payload = {
        family: asdict(allocations[family]) for family in sorted(allocations)
    }
    candidate_pool_sha256 = _candidate_pool_sha256(
        record.to_dict() for record in records
    )

    train: list[AuditedSceneRecord] = []
    validation: list[AuditedSceneRecord] = []
    sealed: list[AuditedSceneRecord] = []
    excluded = [record for record in records if not record.fully_reachable]
    eligible = [record for record in records if record.fully_reachable]
    for family in sorted(families):
        family_records = [record for record in eligible if record.family == family]
        family_records.sort(
            key=lambda record: (
                _scene_rank(record, split_seed=int(split_seed)),
                record.scene_id,
            )
        )
        allocation = allocations[family]
        required = int(allocation.validation + allocation.sealed_test)
        if required > len(family_records):
            raise ValueError(
                f"family {family!r} has {len(family_records)} eligible scenes but "
                f"requires {required} validation+sealed-test scenes"
            )
        sealed.extend(family_records[: allocation.sealed_test])
        validation.extend(
            family_records[
                allocation.sealed_test : allocation.sealed_test + allocation.validation
            ]
        )
        train.extend(family_records[required:])

    sort_key = lambda record: (record.family, record.scene_id)
    train.sort(key=sort_key)
    validation.sort(key=sort_key)
    sealed.sort(key=sort_key)
    excluded.sort(key=sort_key)

    sealed_core = {
        "schema": _SEALED_TEST_SCHEMA,
        "benchmark_id": str(benchmark_id),
        "split_seed": int(split_seed),
        "geometry_contract_sha256": str(geometry_contract.sha256),
        "audit_config": audit_config,
        "split_allocations": allocation_payload,
        "candidate_pool_sha256": candidate_pool_sha256,
        "scenes": [record.to_dict() for record in sealed],
    }
    commitment = _sha256_payload(sealed_core)
    sealed_payload = {**sealed_core, "commitment_sha256": commitment}
    development_payload = {
        "schema": _DEVELOPMENT_SCHEMA,
        "benchmark_id": str(benchmark_id),
        "geometry_contract_sha256": str(geometry_contract.sha256),
        "audit_config": audit_config,
        "split_allocations": allocation_payload,
        "candidate_pool_sha256": candidate_pool_sha256,
        "train_scenes": [record.to_dict() for record in train],
        "validation_scenes": [record.to_dict() for record in validation],
        "excluded_scenes": [record.to_dict() for record in excluded],
        "sealed_test": {
            "schema": _SEALED_TEST_SCHEMA,
            "commitment_sha256": commitment,
            "scene_count": len(sealed),
            "scene_count_by_family": {
                family: sum(record.family == family for record in sealed)
                for family in sorted(families)
            },
        },
    }
    manifests = SceneDisjointManifests(
        development=development_payload,
        sealed_test=sealed_payload,
    )
    verification = verify_scene_disjoint_manifests(manifests)
    if not verification["passes"]:
        raise AssertionError(
            f"generated invalid split manifests: {verification['errors']}"
        )
    return manifests


def verify_scene_disjoint_manifests(
    manifests: SceneDisjointManifests,
) -> dict[str, Any]:
    """Verify the test commitment and scene/topology disjointness."""

    development = dict(manifests.development)
    sealed_test = dict(manifests.sealed_test)
    errors: list[str] = []
    if development.get("schema") != _DEVELOPMENT_SCHEMA:
        errors.append("development_schema_mismatch")
    if sealed_test.get("schema") != _SEALED_TEST_SCHEMA:
        errors.append("sealed_test_schema_mismatch")
    if development.get("benchmark_id") != sealed_test.get("benchmark_id"):
        errors.append("benchmark_id_mismatch")
    if (
        development.get("geometry_contract_sha256")
        != sealed_test.get("geometry_contract_sha256")
    ):
        errors.append("geometry_contract_sha256_mismatch")
    if development.get("audit_config") != sealed_test.get("audit_config"):
        errors.append("audit_config_mismatch")
    if development.get("split_allocations") != sealed_test.get(
        "split_allocations"
    ):
        errors.append("split_allocations_mismatch")
    if development.get("candidate_pool_sha256") != sealed_test.get(
        "candidate_pool_sha256"
    ):
        errors.append("candidate_pool_sha256_mismatch")
    sealed_core = {
        key: value
        for key, value in sealed_test.items()
        if key != "commitment_sha256"
    }
    actual_commitment = _sha256_payload(sealed_core)
    expected_commitment = (
        development.get("sealed_test", {}).get("commitment_sha256")
    )
    if sealed_test.get("commitment_sha256") != actual_commitment:
        errors.append("sealed_payload_commitment_mismatch")
    if expected_commitment != actual_commitment:
        errors.append("development_commitment_mismatch")

    split_entries = {
        "train": list(development.get("train_scenes", [])),
        "validation": list(development.get("validation_scenes", [])),
        "sealed_test": list(sealed_test.get("scenes", [])),
        "excluded": list(development.get("excluded_scenes", [])),
    }
    seen_ids: dict[str, str] = {}
    seen_topologies: dict[tuple[str, int], str] = {}
    for split_name, entries in split_entries.items():
        for entry in entries:
            if not isinstance(entry, Mapping):
                errors.append(f"malformed_scene_entry:{split_name}")
                continue
            scene_id = str(entry.get("scene_id", ""))
            family = str(entry.get("family", ""))
            try:
                topology_seed = int(entry["topology_seed"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"malformed_topology_seed:{split_name}:{scene_id}")
                continue
            topology = (family, topology_seed)
            if not scene_id or not family:
                errors.append(f"malformed_scene_identity:{split_name}")
                continue
            expected_reachable = split_name != "excluded"
            if bool(entry.get("fully_reachable")) != expected_reachable:
                errors.append(
                    f"reachability_split_mismatch:{split_name}:{scene_id}"
                )
            if scene_id in seen_ids:
                errors.append(
                    f"scene_overlap:{scene_id}:{seen_ids[scene_id]}:{split_name}"
                )
            else:
                seen_ids[scene_id] = split_name
            if topology in seen_topologies:
                errors.append(
                    f"topology_overlap:{topology}:"
                    f"{seen_topologies[topology]}:{split_name}"
                )
            else:
                seen_topologies[topology] = split_name

    declared_count = development.get("sealed_test", {}).get("scene_count")
    if declared_count != len(split_entries["sealed_test"]):
        errors.append("sealed_test_count_mismatch")
    declared_count_by_family = development.get("sealed_test", {}).get(
        "scene_count_by_family"
    )
    actual_count_by_family: dict[str, int] = {
        str(family): 0
        for family in (
            declared_count_by_family
            if isinstance(declared_count_by_family, Mapping)
            else ()
        )
    }
    for entry in split_entries["sealed_test"]:
        if isinstance(entry, Mapping):
            family = str(entry.get("family", ""))
            actual_count_by_family[family] = actual_count_by_family.get(family, 0) + 1
    if declared_count_by_family != actual_count_by_family:
        errors.append("sealed_test_family_count_mismatch")

    all_entries = [entry for entries in split_entries.values() for entry in entries]
    if all(isinstance(entry, Mapping) for entry in all_entries):
        actual_pool_sha256 = _candidate_pool_sha256(all_entries)
        if development.get("candidate_pool_sha256") != actual_pool_sha256:
            errors.append("candidate_pool_contents_mismatch")
    else:
        actual_pool_sha256 = None
    return {
        "schema": "lewm_navigation_scene_split_verification_v0",
        "passes": not errors,
        "errors": errors,
        "counts": {name: len(entries) for name, entries in split_entries.items()},
        "actual_commitment_sha256": actual_commitment,
        "actual_candidate_pool_sha256": actual_pool_sha256,
    }


def write_scene_disjoint_manifests(
    manifests: SceneDisjointManifests,
    *,
    development_path: Path,
    sealed_test_path: Path,
    overwrite: bool = False,
) -> None:
    """Write the two payloads separately, refusing accidental replacement."""

    for path in (development_path, sealed_test_path):
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    development_path.write_text(_pretty_json(manifests.development), encoding="utf-8")
    sealed_test_path.write_text(_pretty_json(manifests.sealed_test), encoding="utf-8")


def fixed_spawn_audit_config_from_geometry_contract(
    contract: GeometryContract,
) -> FixedSpawnAuditConfig:
    """Derive the audit configuration from the versioned geometry contract."""

    if contract.coverage.normalization != (
        "fixed_spawn_reachable_configuration_space_cells"
    ):
        raise ValueError(
            "coverage normalization must be "
            "fixed_spawn_reachable_configuration_space_cells"
        )
    return FixedSpawnAuditConfig(
        cell_size_m=float(contract.configuration_space.oracle_cell_size_m),
        coverage_cell_size_m=float(contract.coverage.cell_size_m),
        body_radius_m=float(
            contract.configuration_space.body_inflation_radius_m
        ),
        claim_radius_m=float(contract.visibility_and_claim.claim_radius_m),
        standoff_m=float(contract.visibility_and_claim.standoff_m),
        standoff_candidates=int(
            contract.visibility_and_claim.standoff_candidates
        ),
        minimum_navigable_corridor_width_m=float(
            contract.visibility_and_claim.minimum_navigable_corridor_width_m
        ),
        minimum_navigable_standoffs_per_beacon=1,
        require_line_of_sight=bool(
            contract.visibility_and_claim.require_line_of_sight_for_scene_validity
        ),
        connectivity=int(contract.configuration_space.connectivity),
        allow_diagonal_corner_cutting=bool(
            contract.configuration_space.allow_diagonal_corner_cutting
        ),
        treat_landmarks_as_obstacles=bool(
            contract.configuration_space.landmarks_are_obstacles
        ),
        treat_distractors_as_obstacles=bool(
            contract.configuration_space.distractors_are_obstacles
        ),
    )


def _validate_scene_records(records: Sequence[AuditedSceneRecord]) -> None:
    ids: set[str] = set()
    topologies: set[tuple[str, int]] = set()
    for record in records:
        if not record.scene_id or not record.family:
            raise ValueError("scene_id and family must be non-empty")
        _validate_sha256(record.manifest_sha256, name="manifest_sha256")
        _validate_sha256(record.audit_sha256, name="audit_sha256")
        if (
            not math.isfinite(record.reachable_area_m2)
            or record.reachable_area_m2 < 0.0
        ):
            raise ValueError("reachable_area_m2 must be finite and non-negative")
        if record.beacon_count < 0:
            raise ValueError("beacon_count must be non-negative")
        if not (
            0
            <= record.beacons_with_preferred_standoff
            <= record.beacon_count
        ):
            raise ValueError(
                "beacons_with_preferred_standoff must be within beacon_count"
            )
        if record.fully_reachable and record.failure_reason:
            raise ValueError("fully reachable scenes may not have a failure reason")
        if not record.fully_reachable and not record.failure_reason:
            raise ValueError("excluded scenes must have a failure reason")
        if (record.physical_eligible is None) != (
            record.physical_eligibility_sha256 is None
        ):
            raise ValueError(
                "physical eligibility flag and SHA-256 must be provided together"
            )
        if record.physical_eligibility_sha256 is not None:
            _validate_sha256(
                record.physical_eligibility_sha256,
                name="physical_eligibility_sha256",
            )
            if record.fully_reachable and not record.physical_eligible:
                raise ValueError(
                    "fully reachable physical scenes must pass physical eligibility"
                )
        if record.scene_id in ids:
            raise ValueError(f"duplicate scene_id: {record.scene_id}")
        ids.add(record.scene_id)
        topology = (record.family, int(record.topology_seed))
        if topology in topologies:
            raise ValueError(f"duplicate family/topology seed: {topology}")
        topologies.add(topology)


def _scene_rank(record: AuditedSceneRecord, *, split_seed: int) -> str:
    payload = "\0".join(
        (
            _SPLIT_RANK_DOMAIN,
            str(int(split_seed)),
            record.family,
            record.scene_id,
            record.manifest_sha256,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_sha256(value: str, *, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


def _candidate_pool_sha256(entries: Iterable[Mapping[str, Any]]) -> str:
    ordered = sorted((dict(entry) for entry in entries), key=_canonical_json)
    return _sha256_payload({"scenes": ordered})


def supercover_segment_cells(
    start_xy_m: tuple[float, float],
    end_xy_m: tuple[float, float],
    *,
    origin_xy_m: tuple[float, float],
    cell_size_m: float,
) -> frozenset[tuple[int, int]]:
    """Return every grid cell touched by a world-frame line segment."""

    x0 = (float(start_xy_m[0]) - origin_xy_m[0]) / cell_size_m
    y0 = (float(start_xy_m[1]) - origin_xy_m[1]) / cell_size_m
    x1 = (float(end_xy_m[0]) - origin_xy_m[0]) / cell_size_m
    y1 = (float(end_xy_m[1]) - origin_xy_m[1]) / cell_size_m
    x, y = int(math.floor(x0)), int(math.floor(y0))
    end_x, end_y = int(math.floor(x1)), int(math.floor(y1))
    cells: set[tuple[int, int]] = {(x, y)}
    if (x, y) == (end_x, end_y):
        return frozenset(cells)

    dx, dy = x1 - x0, y1 - y0
    step_x = 1 if dx > 0.0 else -1 if dx < 0.0 else 0
    step_y = 1 if dy > 0.0 else -1 if dy < 0.0 else 0
    t_delta_x = math.inf if step_x == 0 else abs(1.0 / dx)
    t_delta_y = math.inf if step_y == 0 else abs(1.0 / dy)
    next_boundary_x = float(x + 1 if step_x > 0 else x)
    next_boundary_y = float(y + 1 if step_y > 0 else y)
    t_max_x = (
        math.inf if step_x == 0 else (next_boundary_x - x0) / dx
    )
    t_max_y = (
        math.inf if step_y == 0 else (next_boundary_y - y0) / dy
    )

    while (x, y) != (end_x, end_y):
        if math.isclose(t_max_x, t_max_y, rel_tol=0.0, abs_tol=1e-12):
            cells.add((x + step_x, y))
            cells.add((x, y + step_y))
            x += step_x
            y += step_y
            t_max_x += t_delta_x
            t_max_y += t_delta_y
        elif t_max_x < t_max_y:
            x += step_x
            t_max_x += t_delta_x
        else:
            y += step_y
            t_max_y += t_delta_y
        cells.add((x, y))
    return frozenset(cells)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _pretty_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


__all__ = [
    "AuditedSceneRecord",
    "ReachableCoverageMetric",
    "SceneDisjointManifests",
    "SceneSplitCounts",
    "StrictClaimObservation",
    "StrictClaimResult",
    "StrictClaimSummary",
    "audited_scene_record",
    "build_hashed_scene_role_commitment",
    "build_scene_disjoint_manifests",
    "fixed_spawn_audit_config_from_geometry_contract",
    "reachable_area_normalized_coverage",
    "scene_role_token",
    "scene_id_sha256",
    "strict_ground_truth_claim",
    "summarize_strict_ground_truth_claims",
    "supercover_segment_cells",
    "verify_scene_disjoint_manifests",
    "write_scene_disjoint_manifests",
]
