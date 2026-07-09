"""Scene-disjoint paired RGB/BEV navigation data with strict provenance.

The labels in this module are privileged offline targets.  Runtime consumers
receive RGB only; no simulator depth or privileged geometry is part of a row's
model input.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from lewm.planning.geometry_contract import GeometryContract
from lewm_worlds.manifest import (
    SceneManifest,
    manifest_sha256,
    parse_scene_manifest_dict,
)
from lewm_worlds.planning_grid import InflatedOccupancyGrid


UNKNOWN_CLASS = 0
FREE_CLASS = 1
OCCUPIED_CLASS = 2
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class DatasetContractError(ValueError):
    """Raised when source data violates the paired-navigation contract."""


class ForbiddenSceneError(DatasetContractError):
    """Raised before scene artifacts are opened when a held-out scene is present."""


class ProvenanceError(DatasetContractError):
    """Raised when a recorded artifact no longer matches its digest."""


@dataclass(frozen=True)
class LocalGridGeometry:
    """Base-egocentric Cartesian grid; rows are forward and columns are left.

    Minima are *cell edges*.  This distinction is persisted in dataset
    metadata because differentiable warps consume the corresponding center
    ranges.
    """

    rows: int = 64
    cols: int = 64
    cell_size_m: float = 0.10
    forward_min_edge_m: float = -1.0
    left_min_edge_m: float = -3.2

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("grid rows and columns must be positive")
        if not math.isfinite(self.cell_size_m) or self.cell_size_m <= 0.0:
            raise ValueError("grid cell_size_m must be positive")
        if not math.isfinite(self.forward_min_edge_m):
            raise ValueError("forward_min_edge_m must be finite")
        if not math.isfinite(self.left_min_edge_m):
            raise ValueError("left_min_edge_m must be finite")

    @property
    def forward_max_edge_m(self) -> float:
        return self.forward_min_edge_m + self.rows * self.cell_size_m

    @property
    def left_max_edge_m(self) -> float:
        return self.left_min_edge_m + self.cols * self.cell_size_m

    @property
    def forward_center_range_m(self) -> tuple[float, float]:
        half = 0.5 * self.cell_size_m
        return (
            round(self.forward_min_edge_m + half, 12),
            round(self.forward_max_edge_m - half, 12),
        )

    @property
    def left_center_range_m(self) -> tuple[float, float]:
        half = 0.5 * self.cell_size_m
        return (
            round(self.left_min_edge_m + half, 12),
            round(self.left_max_edge_m - half, 12),
        )

    def forward_centers_m(self) -> np.ndarray:
        return self.forward_min_edge_m + (
            np.arange(self.rows, dtype=np.float64) + 0.5
        ) * self.cell_size_m

    def left_centers_m(self) -> np.ndarray:
        return self.left_min_edge_m + (
            np.arange(self.cols, dtype=np.float64) + 0.5
        ) * self.cell_size_m

    def local_cell(self, forward_m: float, left_m: float) -> tuple[int, int] | None:
        row = int(
            math.floor(
                (float(forward_m) - self.forward_min_edge_m) / self.cell_size_m
            )
        )
        col = int(
            math.floor((float(left_m) - self.left_min_edge_m) / self.cell_size_m)
        )
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return row, col
        return None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "shape": [self.rows, self.cols],
            "cell_size_m": self.cell_size_m,
            "forward_edge_range_m": [
                self.forward_min_edge_m,
                self.forward_max_edge_m,
            ],
            "left_edge_range_m": [self.left_min_edge_m, self.left_max_edge_m],
            "forward_center_range_m": list(self.forward_center_range_m),
            "left_center_range_m": list(self.left_center_range_m),
            "array_axes": {
                "row": "base_forward_increasing",
                "column": "base_left_increasing",
            },
            "base_frame_axes": {
                "forward": "+x_base_link",
                "left": "+y_base_link",
            },
            "bounds_are": "cell_edges",
        }


DEFAULT_LOCAL_GRID = LocalGridGeometry()


@dataclass(frozen=True)
class CameraObservation:
    position_xyz_m: tuple[float, float, float]
    lookat_xyz_m: tuple[float, float, float]
    horizontal_fov_deg: float
    near_m: float


@dataclass(frozen=True)
class SceneRenderSource:
    """Explicit paths for one rendered scene.

    ``scene_id`` is intentionally duplicated outside the artifacts.  The
    builder checks its hash against every labeled held-out commitment before
    opening the manifest, frame plan, or images.
    """

    scene_id: str
    scene_manifest_path: Path
    render_plan_path: Path
    family: str | None = None
    rgb_dir: Path | None = None
    frames_jsonl_path: Path | None = None
    rendered_frames_jsonl_path: Path | None = None


@dataclass(frozen=True)
class SceneIdCommitmentSet:
    """One labeled opaque membership set and its source provenance."""

    label: str
    scene_id_sha256: frozenset[str]
    source_path: Path | None = None
    source_file_sha256: str | None = None
    source_format: str = "newline_sha256"
    source_schema: str | None = None
    source_role: str | None = None
    source_content_sha256: str | None = None
    source_role_set_sha256: str | None = None
    benchmark_id: str | None = None

    def __post_init__(self) -> None:
        label = str(self.label).strip()
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
            raise ValueError(
                f"invalid commitment label {self.label!r}; "
                "use letters, digits, '.', '-', '_'"
            )
        object.__setattr__(self, "label", label)
        if not self.scene_id_sha256:
            raise ValueError(f"scene-ID commitment set is empty: {label}")
        for digest in self.scene_id_sha256:
            if not _SHA256_RE.fullmatch(str(digest)):
                raise ValueError(f"invalid scene-ID SHA-256 commitment: {digest!r}")
        if self.source_path is not None:
            object.__setattr__(self, "source_path", self.source_path.resolve())
            if not _SHA256_RE.fullmatch(str(self.source_file_sha256 or "")):
                raise ValueError(f"missing source file SHA-256 for {label}")
        for name, digest in (
            ("source_content_sha256", self.source_content_sha256),
            ("source_role_set_sha256", self.source_role_set_sha256),
        ):
            if digest is not None and not _SHA256_RE.fullmatch(str(digest)):
                raise ValueError(f"invalid {name} for {label}: {digest!r}")

    def to_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "count": len(self.scene_id_sha256),
            "commitment_set_sha256": canonical_json_sha256(
                sorted(self.scene_id_sha256)
            ),
            "source_format": self.source_format,
        }
        if self.source_path is not None:
            metadata["file"] = str(self.source_path)
            metadata["file_sha256"] = self.source_file_sha256
        if self.source_schema is not None:
            metadata["source_schema"] = self.source_schema
        if self.source_role is not None:
            metadata["source_role"] = self.source_role
        if self.source_content_sha256 is not None:
            metadata["source_content_sha256"] = self.source_content_sha256
        if self.source_role_set_sha256 is not None:
            metadata["source_role_set_sha256"] = self.source_role_set_sha256
        if self.benchmark_id is not None:
            metadata["benchmark_id"] = self.benchmark_id
        return metadata


@dataclass(frozen=True)
class SceneIdExclusions:
    """Generic labeled held-out scene commitments used as a pre-open guard."""

    sets: tuple[SceneIdCommitmentSet, ...]

    def __post_init__(self) -> None:
        if not self.sets:
            raise ValueError("at least one labeled scene-ID commitment set is required")
        labels = [item.label for item in self.sets]
        if len(labels) != len(set(labels)):
            duplicates = sorted(
                label for label, count in Counter(labels).items() if count > 1
            )
            raise ValueError(
                "duplicate scene-ID commitment labels: " + ", ".join(duplicates)
            )

    @property
    def union(self) -> frozenset[str]:
        return frozenset(
            digest for item in self.sets for digest in item.scene_id_sha256
        )

    def assert_allowed(self, scene_id: str) -> None:
        digest = scene_id_sha256(scene_id)
        labels = sorted(
            item.label for item in self.sets if digest in item.scene_id_sha256
        )
        if labels:
            # Do not place the raw held-out identity in logs or persisted errors.
            raise ForbiddenSceneError(
                f"scene SHA-256 {digest} belongs to excluded set(s): "
                + ", ".join(labels)
            )

    def merged(self, other: "SceneIdExclusions") -> "SceneIdExclusions":
        return SceneIdExclusions(self.sets + other.sets)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "comparison": "sha256(utf8(scene_id))",
            "union_count": len(self.union),
            "union_commitment_set_sha256": canonical_json_sha256(
                sorted(self.union)
            ),
            "sets": {
                item.label: item.to_metadata()
                for item in sorted(self.sets, key=lambda value: value.label)
            },
            "raw_forbidden_scene_ids_persisted": False,
        }


@dataclass(frozen=True)
class V3SceneExclusions:
    """Backward-compatible in-memory alias for the original v3 API."""

    development_scene_id_sha256: frozenset[str]
    sealed_scene_id_sha256: frozenset[str]

    def as_generic(self) -> SceneIdExclusions:
        return SceneIdExclusions(
            (
                SceneIdCommitmentSet(
                    label="v3_development",
                    scene_id_sha256=self.development_scene_id_sha256,
                    source_format="legacy_in_memory",
                ),
                SceneIdCommitmentSet(
                    label="v3_sealed",
                    scene_id_sha256=self.sealed_scene_id_sha256,
                    source_format="legacy_in_memory",
                ),
            )
        )

    def __post_init__(self) -> None:
        self.as_generic()

    def assert_allowed(self, scene_id: str) -> None:
        self.as_generic().assert_allowed(scene_id)

    def to_metadata(self) -> dict[str, Any]:
        generic = self.as_generic().to_metadata()
        development = generic["sets"]["v3_development"]
        sealed = generic["sets"]["v3_sealed"]
        return {
            "development_commitment_count": development["count"],
            "development_commitment_set_sha256": development[
                "commitment_set_sha256"
            ],
            "sealed_commitment_count": sealed["count"],
            "sealed_commitment_set_sha256": sealed["commitment_set_sha256"],
            "comparison": generic["comparison"],
            "raw_forbidden_scene_ids_persisted": False,
        }


@dataclass(frozen=True)
class PrimitiveTransition:
    current: Mapping[str, Any]
    next: Mapping[str, Any]
    primitive: str
    duration_s: float
    frame_window: tuple[Mapping[str, Any], ...] = ()


def _transition_stream_key(
    transition: PrimitiveTransition,
) -> tuple[str, int, str, int]:
    episode = transition.current["episode"]
    return (
        transition.primitive,
        int(transition.current.get("env_index", 0)),
        str(episode.get("episode_id")),
        int(episode.get("reset_count", 0)),
    )


def _transition_time_key(transition: PrimitiveTransition) -> tuple[Any, ...]:
    episode = transition.current["episode"]
    return (
        int(transition.current.get("env_index", 0)),
        str(episode.get("episode_id")),
        int(episode.get("reset_count", 0)),
        int(episode["episode_step"]),
        int(transition.current["timestamp_ns"]),
        int(transition.current["frame_index"]),
    )


def _transition_hash(
    transition: PrimitiveTransition,
    *,
    scene_id: str,
    seed: str,
) -> str:
    current_pose = _base_xy_yaw(transition.current)
    return canonical_json_sha256(
        {
            "seed": seed,
            "scene_id": scene_id,
            "primitive": transition.primitive,
            "stream": _transition_stream_key(transition),
            "current_frame_index": int(transition.current["frame_index"]),
            "next_frame_index": int(transition.next["frame_index"]),
            "current_timestamp_ns": int(transition.current["timestamp_ns"]),
            "base_xy_yaw": current_pose,
        }
    )


def select_primitive_transitions(
    transitions: Sequence[PrimitiveTransition],
    *,
    scene_id: str,
    max_transitions: int = 512,
    seed: str = "go2_paired_navigation_selection_v1",
) -> tuple[list[PrimitiveTransition], dict[str, Any]]:
    """Hash-rank and round-robin primitive/env/episode strata before labeling."""

    limit = int(max_transitions)
    if limit <= 0:
        raise ValueError("max_transitions must be positive")
    strata: dict[
        tuple[str, int, str, int], list[PrimitiveTransition]
    ] = defaultdict(list)
    for transition in transitions:
        strata[_transition_stream_key(transition)].append(transition)
    for bucket in strata.values():
        bucket.sort(
            key=lambda item: _transition_hash(item, scene_id=scene_id, seed=seed)
        )

    by_primitive: dict[
        str,
        list[tuple[tuple[str, int, str, int], list[PrimitiveTransition]]],
    ] = defaultdict(list)
    for stratum, bucket in strata.items():
        by_primitive[stratum[0]].append((stratum, bucket))
    primitive_sequences: dict[str, list[PrimitiveTransition]] = {}
    for primitive, buckets in by_primitive.items():
        buckets.sort(
            key=lambda item: canonical_json_sha256(
                {"seed": seed, "scene_id": scene_id, "stratum": item[0]}
            )
        )
        sequence: list[PrimitiveTransition] = []
        depth = 0
        while True:
            added = False
            for _stratum, bucket in buckets:
                if depth < len(bucket):
                    sequence.append(bucket[depth])
                    added = True
            if not added:
                break
            depth += 1
        primitive_sequences[primitive] = sequence

    primitive_order = sorted(
        primitive_sequences,
        key=lambda primitive: canonical_json_sha256(
            {"seed": seed, "scene_id": scene_id, "primitive": primitive}
        ),
    )
    selected: list[PrimitiveTransition] = []
    depth = 0
    while len(selected) < limit:
        added = False
        for primitive in primitive_order:
            sequence = primitive_sequences[primitive]
            if depth < len(sequence):
                selected.append(sequence[depth])
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        depth += 1
    selected.sort(key=_transition_time_key)

    candidate_by_primitive = Counter(item.primitive for item in transitions)
    selected_by_primitive = Counter(item.primitive for item in selected)
    metadata = {
        "seed": seed,
        "method": "hash_rank_within_primitive_env_episode_strata_then_round_robin",
        "max_transitions": limit,
        "candidate_count": len(transitions),
        "selected_count": len(selected),
        "stratum_count": len(strata),
        "candidate_by_primitive": dict(sorted(candidate_by_primitive.items())),
        "selected_by_primitive": dict(sorted(selected_by_primitive.items())),
    }
    return selected, metadata


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def scene_id_sha256(scene_id: str) -> str:
    return hashlib.sha256(str(scene_id).encode("utf-8")).hexdigest()


def read_scene_id_commitments(path: Path) -> frozenset[str]:
    """Read newline-delimited SHA-256 commitments, never raw scene IDs."""

    commitments: set[str] = set()
    for line_number, raw in enumerate(path.read_text().splitlines(), start=1):
        value = raw.strip().lower()
        if not value or value.startswith("#"):
            continue
        if not _SHA256_RE.fullmatch(value):
            raise ValueError(f"{path}:{line_number}: expected a SHA-256 digest")
        commitments.add(value)
    return frozenset(commitments)


def _structured_scene_role_sets(
    *,
    label_prefix: str,
    path: Path,
    payload: Mapping[str, Any],
    file_sha256: str,
    excluded_roles: Sequence[str],
) -> tuple[SceneIdCommitmentSet, ...]:
    schema = str(payload.get("schema", ""))
    if schema != "lewm_navigation_hashed_scene_roles_v1":
        raise DatasetContractError(
            f"unsupported scene-role commitment schema in {path}: {schema!r}"
        )
    expected_content_sha256 = str(payload.get("content_sha256", ""))
    core = dict(payload)
    core.pop("content_sha256", None)
    actual_content_sha256 = canonical_json_sha256(core)
    if expected_content_sha256 != actual_content_sha256:
        raise ProvenanceError(
            f"scene-role commitment content hash mismatch for {path}: "
            f"expected {expected_content_sha256!r}, got {actual_content_sha256}"
        )
    benchmark_id = str(payload.get("benchmark_id", ""))
    if not benchmark_id:
        raise DatasetContractError(
            f"scene-role commitment has no benchmark ID: {path}"
        )
    memberships = payload.get("scene_id_sha256_by_role")
    role_tokens = payload.get("roles")
    counts = payload.get("counts")
    declared_set_hashes = payload.get("set_sha256_by_role")
    if not isinstance(memberships, Mapping):
        raise DatasetContractError(
            f"scene-role commitment lacks plain membership sets: {path}"
        )
    if not isinstance(role_tokens, Mapping) or not isinstance(counts, Mapping):
        raise DatasetContractError(f"malformed scene-role commitment: {path}")
    if not isinstance(declared_set_hashes, Mapping):
        raise DatasetContractError(
            f"scene-role commitment lacks per-role set commitments: {path}"
        )

    parsed_memberships: dict[str, frozenset[str]] = {}
    for role, raw_values in memberships.items():
        role_name = str(role)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", role_name):
            raise DatasetContractError(
                f"invalid role label in scene-role commitment: {role_name!r}"
            )
        if not isinstance(raw_values, list):
            raise DatasetContractError(
                f"scene-role membership {role_name!r} is not a list"
            )
        values = [str(value).lower() for value in raw_values]
        if values != sorted(values) or len(values) != len(set(values)):
            raise DatasetContractError(
                f"scene-role membership {role_name!r} must be sorted and unique"
            )
        if any(not _SHA256_RE.fullmatch(value) for value in values):
            raise DatasetContractError(
                f"scene-role membership {role_name!r} contains a non-SHA-256 value"
            )
        if int(counts.get(role_name, -1)) != len(values):
            raise DatasetContractError(
                f"scene-role membership count mismatch for {role_name!r}"
            )
        raw_tokens = role_tokens.get(role_name)
        if not isinstance(raw_tokens, list):
            raise DatasetContractError(
                f"scene-role token set {role_name!r} is not a list"
            )
        tokens = [str(value).lower() for value in raw_tokens]
        if (
            tokens != sorted(tokens)
            or len(tokens) != len(set(tokens))
            or any(not _SHA256_RE.fullmatch(value) for value in tokens)
            or len(tokens) != len(values)
        ):
            raise DatasetContractError(
                f"malformed domain-separated token set for {role_name!r}"
            )
        declared = declared_set_hashes.get(role_name)
        if not isinstance(declared, Mapping):
            raise DatasetContractError(
                f"missing declared set hashes for role {role_name!r}"
            )
        expected_plain_hash = canonical_json_sha256(
            {"scene_id_sha256": values}
        )
        expected_token_hash = canonical_json_sha256({"tokens": tokens})
        if str(declared.get("scene_id_sha256", "")) != expected_plain_hash:
            raise ProvenanceError(
                f"plain membership commitment mismatch for role {role_name!r}"
            )
        if str(declared.get("role_tokens_sha256", "")) != expected_token_hash:
            raise ProvenanceError(
                f"role-token commitment mismatch for role {role_name!r}"
            )
        parsed_memberships[role_name] = frozenset(values)

    requested = tuple(dict.fromkeys(str(role) for role in excluded_roles))
    if not requested:
        raise ValueError("at least one structured exclusion role is required")
    result: list[SceneIdCommitmentSet] = []
    for role in requested:
        values = parsed_memberships.get(role)
        if values is None:
            raise DatasetContractError(
                f"scene-role commitment is missing required role {role!r}: {path}"
            )
        declared = declared_set_hashes[role]
        result.append(
            SceneIdCommitmentSet(
                label=f"{label_prefix}.{role}",
                scene_id_sha256=values,
                source_path=path,
                source_file_sha256=file_sha256,
                source_format="hashed_scene_roles_json",
                source_schema=schema,
                source_role=role,
                source_content_sha256=actual_content_sha256,
                source_role_set_sha256=str(declared["scene_id_sha256"]),
                benchmark_id=benchmark_id,
            )
        )
    return tuple(result)


def load_scene_id_exclusions(
    commitment_files: Sequence[tuple[str, Path]],
    *,
    structured_excluded_roles: Sequence[str] = ("development", "sealed_test"),
) -> SceneIdExclusions:
    """Load labeled newline sets or hashed-role JSON commitments.

    A newline file contributes exactly ``LABEL``. A structured benchmark role
    artifact contributes ``LABEL.development`` and ``LABEL.sealed_test`` by
    default. Its training and bookkeeping roles are verified but deliberately
    not excluded from dataset construction.
    """

    if not commitment_files:
        raise ValueError("at least one labeled scene-ID commitment file is required")
    sets: list[SceneIdCommitmentSet] = []
    seen_input_labels: set[str] = set()
    for raw_label, raw_path in commitment_files:
        label = str(raw_label).strip()
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
            raise ValueError(
                f"invalid commitment label {raw_label!r}; "
                "use letters, digits, '.', '-', '_'"
            )
        if label in seen_input_labels:
            raise ValueError(f"duplicate commitment input label: {label}")
        seen_input_labels.add(label)
        path = Path(raw_path).resolve()
        raw_text = path.read_text(encoding="utf-8")
        file_sha256 = sha256_file(path)
        if raw_text.lstrip().startswith("{"):
            try:
                payload = json.loads(raw_text)
            except json.JSONDecodeError as exc:
                raise DatasetContractError(
                    f"invalid scene-role commitment JSON: {path}"
                ) from exc
            if not isinstance(payload, Mapping):
                raise DatasetContractError(
                    f"scene-role commitment must be a JSON object: {path}"
                )
            sets.extend(
                _structured_scene_role_sets(
                    label_prefix=label,
                    path=path,
                    payload=payload,
                    file_sha256=file_sha256,
                    excluded_roles=structured_excluded_roles,
                )
            )
        else:
            sets.append(
                SceneIdCommitmentSet(
                    label=label,
                    scene_id_sha256=read_scene_id_commitments(path),
                    source_path=path,
                    source_file_sha256=file_sha256,
                    source_format="newline_sha256",
                )
            )
    return SceneIdExclusions(tuple(sets))


def deterministic_scene_split(
    scene_ids: Iterable[str],
    *,
    validation_fraction: float = 0.15,
    seed: str = "go2_paired_navigation_v1",
) -> dict[str, str]:
    """Assign stable train/validation splits using only the scene identity."""

    fraction = float(validation_fraction)
    if not 0.0 <= fraction < 1.0:
        raise ValueError("validation_fraction must lie in [0, 1)")
    unique = sorted(set(map(str, scene_ids)))
    result: dict[str, str] = {}
    denominator = float(1 << 256)
    for scene_id in unique:
        digest = hashlib.sha256(f"{seed}\0{scene_id}".encode("utf-8")).digest()
        score = int.from_bytes(digest, "big") / denominator
        result[scene_id] = "validation" if score < fraction else "train"
    return result


DATASET_ROLES = (
    "train",
    "checkpoint_selection",
    "probability_calibration",
    "g2_evaluation",
)


def deterministic_family_role_split(
    scene_families: Mapping[str, str],
    *,
    role_scenes_per_family: int,
    seed: str = "go2_paired_navigation_roles_v1",
) -> dict[str, str]:
    """Assign fixed held-out roles independently within every scene family.

    The assignment depends only on the declared family, scene identity, and
    seed.  Each family contributes exactly ``role_scenes_per_family`` scenes
    to checkpoint selection, probability calibration, and untouched G2, in
    that order; all remaining scenes are training scenes.
    """

    count = int(role_scenes_per_family)
    if count <= 0 or count != role_scenes_per_family:
        raise ValueError("role_scenes_per_family must be a positive integer")
    normalized: dict[str, str] = {}
    for raw_scene_id, raw_family in scene_families.items():
        scene_id = str(raw_scene_id).strip()
        family = str(raw_family).strip()
        if not scene_id or not family:
            raise ValueError("scene IDs and declared families must be nonempty")
        if scene_id in normalized:
            raise ValueError(f"duplicate scene identity in role split: {scene_id!r}")
        normalized[scene_id] = family
    if not normalized:
        raise ValueError("at least one scene is required for a family-role split")

    by_family: dict[str, list[str]] = defaultdict(list)
    for scene_id, family in normalized.items():
        by_family[family].append(scene_id)
    minimum = 3 * count + 1
    assignments: dict[str, str] = {}
    heldout_roles = DATASET_ROLES[1:]
    for family, family_scenes in sorted(by_family.items()):
        if len(family_scenes) < minimum:
            raise ValueError(
                f"family {family!r} has {len(family_scenes)} scenes; at least "
                f"{minimum} are required for {count} scene(s) in each held-out "
                "role and a nonempty training role"
            )
        ranked = sorted(
            family_scenes,
            key=lambda scene_id: (
                hashlib.sha256(
                    f"{seed}\0{family}\0{scene_id}".encode("utf-8")
                ).digest(),
                scene_id,
            ),
        )
        cursor = 0
        for role in heldout_roles:
            for scene_id in ranked[cursor : cursor + count]:
                assignments[scene_id] = role
            cursor += count
        for scene_id in ranked[cursor:]:
            assignments[scene_id] = "train"
    return dict(sorted(assignments.items()))


def relative_se2_current_frame(
    current_xy_yaw: Sequence[float],
    next_xy_yaw: Sequence[float],
) -> np.ndarray:
    """Return next-base ``(forward, left, yaw)`` in the current base frame."""

    x0, y0, yaw0 = map(float, current_xy_yaw)
    x1, y1, yaw1 = map(float, next_xy_yaw)
    dx = x1 - x0
    dy = y1 - y0
    cos_yaw = math.cos(yaw0)
    sin_yaw = math.sin(yaw0)
    forward = cos_yaw * dx + sin_yaw * dy
    left = -sin_yaw * dx + cos_yaw * dy
    dyaw = math.atan2(math.sin(yaw1 - yaw0), math.cos(yaw1 - yaw0))
    return np.asarray([forward, left, dyaw], dtype=np.float32)


def _world_points_to_grid_free(
    grid: InflatedOccupancyGrid,
    xs: np.ndarray,
    ys: np.ndarray,
) -> np.ndarray:
    origin_x, origin_y = grid.origin_xy
    cell = grid.cell_size_m
    ix = np.floor((xs - origin_x) / cell).astype(np.int64)
    iy = np.floor((ys - origin_y) / cell).astype(np.int64)
    inside = (ix >= 0) & (ix < grid.shape[0]) & (iy >= 0) & (iy < grid.shape[1])
    free = np.zeros(xs.shape, dtype=bool)
    free[inside] = grid.free_mask[ix[inside], iy[inside]]
    return free


def label_camera_visible_configuration_grid(
    configuration_grid: InflatedOccupancyGrid,
    *,
    physical_visibility_grid: InflatedOccupancyGrid,
    base_xy_yaw: Sequence[float],
    camera: CameraObservation,
    local_grid: LocalGridGeometry = DEFAULT_LOCAL_GRID,
    ray_step_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label visible configuration occupancy using physical line of sight.

    Visibility uses the recorded camera origin and horizontal look direction.
    Rays stop at their first *physical* obstacle sample in the uninflated
    visibility grid; cells behind that sample remain UNKNOWN.  A visible target
    is classified FREE/OCCUPIED in the body-inflated configuration grid.  The
    distinction is essential: body clearance is a navigation target, not an
    RGB occluder.  This intentionally does not consume rendered depth.
    """

    if configuration_grid.inflation_m <= 0.0:
        raise DatasetContractError(
            "configuration_grid must use positive body inflation"
        )
    if physical_visibility_grid.inflation_m != 0.0:
        raise DatasetContractError(
            "physical_visibility_grid must use zero obstacle inflation"
        )

    base_x, base_y, base_yaw = map(float, base_xy_yaw)
    if not all(math.isfinite(value) for value in (base_x, base_y, base_yaw)):
        raise DatasetContractError("base xy/yaw must be finite")
    camera_xy = np.asarray(camera.position_xyz_m[:2], dtype=np.float64)
    if not np.isfinite(camera_xy).all():
        raise DatasetContractError("camera position must be finite")
    look_xy = np.asarray(camera.lookat_xyz_m[:2], dtype=np.float64) - camera_xy
    look_norm = float(np.linalg.norm(look_xy))
    if not math.isfinite(look_norm) or look_norm <= 1e-9:
        raise DatasetContractError("camera horizontal look direction is degenerate")
    look_xy /= look_norm
    fov_deg = float(camera.horizontal_fov_deg)
    if not 0.0 < fov_deg < 180.0:
        raise DatasetContractError("horizontal camera FOV must lie in (0, 180) degrees")
    near_m = float(camera.near_m)
    if not math.isfinite(near_m) or near_m < 0.0:
        raise DatasetContractError("camera near plane must be finite and non-negative")

    forward, left = np.meshgrid(
        local_grid.forward_centers_m(),
        local_grid.left_centers_m(),
        indexing="ij",
    )
    cos_yaw = math.cos(base_yaw)
    sin_yaw = math.sin(base_yaw)
    world_x = base_x + cos_yaw * forward - sin_yaw * left
    world_y = base_y + sin_yaw * forward + cos_yaw * left
    rel_x = world_x - camera_xy[0]
    rel_y = world_y - camera_xy[1]
    distances = np.hypot(rel_x, rel_y)
    safe_distances = np.maximum(distances, 1e-12)
    ray_x = rel_x / safe_distances
    ray_y = rel_y / safe_distances
    dot = ray_x * look_xy[0] + ray_y * look_xy[1]
    cross = look_xy[0] * ray_y - look_xy[1] * ray_x
    bearing = np.arctan2(cross, dot)
    eligible = (
        np.isfinite(distances)
        & (distances >= near_m)
        & (np.abs(bearing) <= math.radians(fov_deg) * 0.5 + 1e-12)
    )

    target_free = _world_points_to_grid_free(configuration_grid, world_x, world_y)
    first_hit_distance = np.full(distances.shape, np.inf, dtype=np.float64)
    first_hit_row = np.full(distances.shape, -1, dtype=np.int16)
    first_hit_col = np.full(distances.shape, -1, dtype=np.int16)

    step = (
        min(
            0.5 * physical_visibility_grid.cell_size_m,
            0.25 * local_grid.cell_size_m,
        )
        if ray_step_m is None
        else float(ray_step_m)
    )
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("ray_step_m must be positive")
    max_distance = float(np.max(distances[eligible])) if np.any(eligible) else near_m
    sample_count = max(0, int(math.ceil(max(0.0, max_distance - near_m) / step)))
    unresolved = eligible.copy()

    # Sampling at <= half the physical raster size cannot jump across a full
    # obstacle cell. Endpoint occupancy is handled separately.
    for sample_index in range(sample_count):
        sample_distance = near_m + sample_index * step
        active = unresolved & (sample_distance < distances - 1e-9)
        if not np.any(active):
            continue
        sample_x = camera_xy[0] + ray_x[active] * sample_distance
        sample_y = camera_xy[1] + ray_y[active] * sample_distance
        occupied = ~_world_points_to_grid_free(
            physical_visibility_grid, sample_x, sample_y
        )
        if not np.any(occupied):
            continue
        active_indices = np.flatnonzero(active)
        hit_indices = active_indices[occupied]
        hit_x = sample_x[occupied]
        hit_y = sample_y[occupied]
        dx = hit_x - base_x
        dy = hit_y - base_y
        hit_forward = cos_yaw * dx + sin_yaw * dy
        hit_left = -sin_yaw * dx + cos_yaw * dy
        hit_rows = np.floor(
            (hit_forward - local_grid.forward_min_edge_m) / local_grid.cell_size_m
        ).astype(np.int64)
        hit_cols = np.floor(
            (hit_left - local_grid.left_min_edge_m) / local_grid.cell_size_m
        ).astype(np.int64)
        flat_distance = first_hit_distance.ravel()
        flat_row = first_hit_row.ravel()
        flat_col = first_hit_col.ravel()
        flat_distance[hit_indices] = sample_distance
        inside_local = (
            (hit_rows >= 0)
            & (hit_rows < local_grid.rows)
            & (hit_cols >= 0)
            & (hit_cols < local_grid.cols)
        )
        flat_row[hit_indices[inside_local]] = hit_rows[inside_local]
        flat_col[hit_indices[inside_local]] = hit_cols[inside_local]
        unresolved.ravel()[hit_indices] = False

    labels = np.full((local_grid.rows, local_grid.cols), UNKNOWN_CLASS, dtype=np.uint8)
    clear_to_center = eligible & np.isinf(first_hit_distance)
    labels[clear_to_center & target_free] = FREE_CLASS
    labels[clear_to_center & ~target_free] = OCCUPIED_CLASS
    rows, cols = np.indices(labels.shape)
    first_hit_is_target = (
        eligible
        & np.isfinite(first_hit_distance)
        & (first_hit_row == rows)
        & (first_hit_col == cols)
    )
    labels[first_hit_is_target] = OCCUPIED_CLASS
    supervision_mask = np.isfinite(world_x) & np.isfinite(world_y)
    observed_mask = labels != UNKNOWN_CLASS
    return labels, supervision_mask, observed_mask


def occupancy_label_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    supervision_mask: np.ndarray,
    observed_mask: np.ndarray,
) -> dict[str, float]:
    """Score known classes separately from UNKNOWN admission/hallucination.

    FREE/OCCUPIED IoU is restricted to known ground-truth cells.  UNKNOWN
    behavior is reported separately, preventing a large occluded region from
    inflating or dominating traversability metrics.
    """

    prediction = np.asarray(prediction)
    target = np.asarray(target)
    supervision_mask = np.asarray(supervision_mask, dtype=bool)
    observed_mask = np.asarray(observed_mask, dtype=bool)
    if not (
        prediction.shape
        == target.shape
        == supervision_mask.shape
        == observed_mask.shape
    ):
        raise ValueError("prediction, target, and masks must have identical shapes")
    if np.any(observed_mask & ~supervision_mask):
        raise ValueError("observed_mask must be a subset of supervision_mask")
    if np.any(observed_mask != ((target != UNKNOWN_CLASS) & supervision_mask)):
        raise ValueError("observed_mask must identify supervised non-UNKNOWN targets")

    known = observed_mask
    unknown = supervision_mask & ~observed_mask

    def ratio(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else float("nan")

    result: dict[str, float] = {
        "known_cell_accuracy": ratio(
            int(np.count_nonzero((prediction == target) & known)),
            int(np.count_nonzero(known)),
        ),
        "unknown_recall": ratio(
            int(np.count_nonzero((prediction == UNKNOWN_CLASS) & unknown)),
            int(np.count_nonzero(unknown)),
        ),
        "unknown_admission_rate_on_known_truth": ratio(
            int(np.count_nonzero((prediction == UNKNOWN_CLASS) & known)),
            int(np.count_nonzero(known)),
        ),
        "known_hallucination_rate_on_unknown_truth": ratio(
            int(np.count_nonzero((prediction != UNKNOWN_CLASS) & unknown)),
            int(np.count_nonzero(unknown)),
        ),
    }
    for name, class_id in (("free", FREE_CLASS), ("occupied", OCCUPIED_CLASS)):
        intersection = int(
            np.count_nonzero((prediction == class_id) & (target == class_id) & known)
        )
        union = int(
            np.count_nonzero(((prediction == class_id) | (target == class_id)) & known)
        )
        result[f"{name}_iou_on_known_truth"] = ratio(intersection, union)
    return result


def iter_primitive_transitions(
    frames: Iterable[Mapping[str, Any]],
    *,
    time_tolerance_s: float = 2e-4,
    stats: Counter[str] | None = None,
) -> Iterator[PrimitiveTransition]:
    """Yield complete, reset-safe primitive transitions from interleaved frames."""

    counters: Counter[str] = stats if stats is not None else Counter()
    streams: dict[tuple[int, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for frame in frames:
        episode = frame.get("episode")
        if not isinstance(episode, Mapping):
            counters["frames_missing_episode"] += 1
            continue
        key = (
            int(frame.get("env_index", 0)),
            str(episode.get("episode_id")),
            int(episode.get("reset_count", 0)),
        )
        streams[key].append(frame)

    for stream_key in sorted(streams):
        stream = sorted(
            streams[stream_key],
            key=lambda frame: (
                int(frame["episode"]["episode_step"]),
                int(frame["timestamp_ns"]),
                int(frame["frame_index"]),
            ),
        )
        seen_steps: set[int] = set()
        for frame in stream:
            step = int(frame["episode"]["episode_step"])
            if step in seen_steps:
                raise DatasetContractError(
                    f"duplicate episode step {step} in stream {stream_key}"
                )
            seen_steps.add(step)

        index = 0
        while index < len(stream):
            context = stream[index].get("command_context")
            if not isinstance(context, Mapping):
                counters["frames_missing_command_context"] += 1
                index += 1
                continue
            primitive = str(context.get("primitive_name", ""))
            sequence_id = context.get("sequence_id")
            if not primitive or sequence_id is None:
                counters["frames_incomplete_command_context"] += 1
                index += 1
                continue
            signature = (str(sequence_id), primitive)
            run_end = index + 1
            while run_end < len(stream):
                other = stream[run_end].get("command_context")
                if not isinstance(other, Mapping):
                    break
                other_signature = (
                    str(other.get("sequence_id")),
                    str(other.get("primitive_name", "")),
                )
                if other_signature != signature:
                    break
                run_end += 1

            block_size = int(context.get("block_size", 0))
            command_dt_s = float(context.get("command_dt_s", 0.0))
            if block_size <= 0 or not math.isfinite(command_dt_s) or command_dt_s <= 0.0:
                counters["runs_invalid_block_timing"] += 1
                index = run_end
                continue
            if run_end - index != block_size:
                counters["runs_not_exactly_one_block"] += 1
                index = run_end
                continue
            if run_end >= len(stream):
                counters["runs_without_next_frame"] += 1
                index = run_end
                continue

            window = stream[index : run_end + 1]
            steps = [int(frame["episode"]["episode_step"]) for frame in window]
            timestamps = [int(frame["timestamp_ns"]) for frame in window]
            if any(right != left + 1 for left, right in zip(steps, steps[1:])):
                counters["runs_nonconsecutive_episode_steps"] += 1
                index = run_end
                continue
            expected_duration_s = block_size * command_dt_s
            if abs(expected_duration_s - 0.5) > time_tolerance_s:
                counters["runs_not_half_second"] += 1
                index = run_end
                continue
            actual_duration_s = (timestamps[-1] - timestamps[0]) * 1e-9
            if abs(actual_duration_s - expected_duration_s) > time_tolerance_s:
                counters["runs_wrong_duration"] += 1
                index = run_end
                continue
            expected_tick_s = command_dt_s
            tick_deltas = [
                (right - left) * 1e-9 for left, right in zip(timestamps, timestamps[1:])
            ]
            if any(abs(delta - expected_tick_s) > time_tolerance_s for delta in tick_deltas):
                counters["runs_irregular_tick_timing"] += 1
                index = run_end
                continue
            counters["primitive_transitions"] += 1
            yield PrimitiveTransition(
                current=stream[index],
                next=stream[run_end],
                primitive=primitive,
                duration_s=actual_duration_s,
                frame_window=tuple(window),
            )
            index = run_end


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise DatasetContractError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise DatasetContractError(f"expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetContractError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(payload, dict):
                raise DatasetContractError(f"expected object at {path}:{line_number}")
            yield payload


def _resolve_path(parent: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() and path.exists():
        return path.resolve()
    candidates = [parent / path, parent / path.name]
    if path.parent.name:
        candidates.append(parent / path.parent.name / path.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _base_xy_yaw(frame: Mapping[str, Any]) -> tuple[float, float, float]:
    try:
        position = frame["base_pose_world"]["position"]
        yaw = frame["base_rpy_rad"]["yaw"]
        return float(position["x"]), float(position["y"]), float(yaw)
    except (KeyError, TypeError, ValueError) as exc:
        raise DatasetContractError("frame is missing a valid base pose/yaw") from exc


def _transition_configuration_rejection(
    transition: PrimitiveTransition,
    configuration_grid: InflatedOccupancyGrid,
) -> str | None:
    """Return a stable rejection code for an unsafe command window."""

    window = transition.frame_window or (transition.current, transition.next)
    if len(window) < 2:
        return "transitions_rejected_incomplete_configuration_window"
    points: list[tuple[float, float]] = []
    try:
        for frame in window:
            x, y, yaw = _base_xy_yaw(frame)
            if not all(math.isfinite(value) for value in (x, y, yaw)):
                return "transitions_rejected_invalid_configuration_pose"
            points.append((x, y))
    except DatasetContractError:
        return "transitions_rejected_invalid_configuration_pose"

    for point in points:
        clearance = configuration_grid.configuration_clearance_m(point)
        if math.isnan(clearance) or clearance < 0.0:
            return "transitions_rejected_negative_configuration_clearance"
    for current, next_point in zip(points, points[1:]):
        if not configuration_grid.has_free_line(current, next_point):
            return "transitions_rejected_nonfree_configuration_segment"
    return None


def _camera_observation(
    frame: Mapping[str, Any],
    *,
    horizontal_fov_deg: float,
    near_m: float,
) -> CameraObservation:
    try:
        camera = frame["camera_pose_world"]
        position = tuple(float(value) for value in camera["position"])
        lookat = tuple(float(value) for value in camera["lookat"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DatasetContractError("frame is missing a valid recorded camera pose") from exc
    if len(position) != 3 or len(lookat) != 3:
        raise DatasetContractError("camera position/lookat must contain three values")
    return CameraObservation(
        position_xyz_m=position,
        lookat_xyz_m=lookat,
        horizontal_fov_deg=float(horizontal_fov_deg),
        near_m=float(near_m),
    )


def _frame_key(frame: Mapping[str, Any]) -> tuple[int, int]:
    return int(frame["frame_index"]), int(frame.get("env_index", 0))


def _image_for_frame(
    frame: Mapping[str, Any],
    *,
    rendered: Mapping[str, Any] | None,
    rgb_dir: Path | None,
    rendered_parent: Path | None,
) -> Path | None:
    if rendered is not None:
        if not bool(rendered.get("camera_valid", True)):
            return None
        value = rendered.get("rgb_path")
        if value:
            assert rendered_parent is not None
            path = _resolve_path(rendered_parent, str(value))
            if path.is_file():
                return path
    if rgb_dir is None:
        return None
    frame_index, env_index = _frame_key(frame)
    candidate = rgb_dir / f"frame_{frame_index:06d}_env_{env_index:02d}.png"
    return candidate.resolve() if candidate.is_file() else None


def _effective_frame(
    planned: Mapping[str, Any], rendered: Mapping[str, Any] | None
) -> dict[str, Any]:
    merged = dict(planned)
    if rendered is not None and rendered.get("camera_pose_world") is not None:
        merged["camera_pose_world"] = rendered["camera_pose_world"]
    return merged


def _source_scene(
    source: SceneRenderSource,
    *,
    geometry_contract: GeometryContract,
) -> tuple[
    SceneManifest,
    dict[str, Any],
    Path,
    list[dict[str, Any]],
    dict[tuple[int, int], dict[str, Any]],
    dict[str, str],
]:
    manifest_path = source.scene_manifest_path.resolve()
    render_plan_path = source.render_plan_path.resolve()
    manifest_payload = _read_json(manifest_path)
    manifest = parse_scene_manifest_dict(manifest_payload)
    if manifest.scene_id != source.scene_id:
        raise DatasetContractError(
            f"source scene ID {source.scene_id!r} does not match manifest {manifest.scene_id!r}"
        )
    if source.family is not None and manifest.family != source.family:
        raise DatasetContractError(
            f"source family {source.family!r} does not match manifest "
            f"{manifest.family!r} for scene {source.scene_id!r}"
        )
    canonical_manifest_hash = manifest_sha256(manifest)
    plan = _read_json(render_plan_path)
    if str(plan.get("schema")) != "lewm_render_replay_plan_v0":
        raise DatasetContractError(f"unsupported render plan schema in {render_plan_path}")
    if str(plan.get("scene_id")) != source.scene_id:
        raise DatasetContractError("render plan scene ID does not match source")
    if str(plan.get("manifest_sha256")) != canonical_manifest_hash:
        raise DatasetContractError("render plan manifest hash does not match manifest")
    camera_plan = plan.get("camera")
    if not isinstance(camera_plan, Mapping):
        raise DatasetContractError("render plan is missing camera geometry")
    if str(camera_plan.get("fov_axis")) != "horizontal":
        raise DatasetContractError("only horizontal camera FOV plans are supported")
    fov_deg = float(camera_plan["fov_deg"])
    near_m = float(camera_plan["near_m"])
    if not math.isclose(
        fov_deg, geometry_contract.camera.horizontal_fov_deg, abs_tol=1e-3
    ):
        raise DatasetContractError("render-plan FOV differs from geometry contract")
    if not math.isclose(near_m, geometry_contract.camera.near_m, abs_tol=1e-6):
        raise DatasetContractError("render-plan near plane differs from geometry contract")

    frames_path = (
        source.frames_jsonl_path.resolve()
        if source.frames_jsonl_path is not None
        else _resolve_path(render_plan_path.parent, str(plan["frames_jsonl"]))
    )
    frames = list(_read_jsonl(frames_path))
    for frame in frames:
        episode = frame.get("episode")
        if isinstance(episode, Mapping):
            episode_manifest_hash = episode.get("manifest_sha256")
            if (
                episode_manifest_hash is not None
                and str(episode_manifest_hash) != canonical_manifest_hash
            ):
                raise DatasetContractError("frame episode manifest hash does not match manifest")

    rendered_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    if source.rendered_frames_jsonl_path is not None:
        planned_by_key = {_frame_key(frame): frame for frame in frames}
        for rendered in _read_jsonl(source.rendered_frames_jsonl_path.resolve()):
            key = _frame_key(rendered)
            if key in rendered_by_key:
                raise DatasetContractError(f"duplicate rendered frame key {key}")
            planned = planned_by_key.get(key)
            if planned is None:
                raise DatasetContractError(f"rendered frame key {key} is absent from frame plan")
            if int(rendered["timestamp_ns"]) != int(planned["timestamp_ns"]):
                raise DatasetContractError(
                    f"rendered frame timestamp does not match frame plan for key {key}"
                )
            rendered_by_key[key] = rendered

    hashes = {
        "geometry_contract_sha256": geometry_contract.sha256,
        "geometry_contract_file_sha256": sha256_file(geometry_contract.source_path),
        "scene_manifest_sha256": canonical_manifest_hash,
        "scene_manifest_file_sha256": sha256_file(manifest_path),
        "frame_plan_sha256": sha256_file(render_plan_path),
        "frames_jsonl_sha256": sha256_file(frames_path),
    }
    if source.rendered_frames_jsonl_path is not None:
        hashes["rendered_frames_jsonl_sha256"] = sha256_file(
            source.rendered_frames_jsonl_path.resolve()
        )
    return manifest, plan, frames_path, frames, rendered_by_key, hashes


def _np_unicode(values: Sequence[Any]) -> np.ndarray:
    strings = [str(value) for value in values]
    width = max((len(value) for value in strings), default=1)
    return np.asarray(strings, dtype=f"<U{width}")


def build_paired_navigation_dataset(
    *,
    sources: Sequence[SceneRenderSource],
    output_dir: Path,
    geometry_contract: GeometryContract,
    scene_exclusions: SceneIdExclusions | None = None,
    v3_exclusions: V3SceneExclusions | None = None,
    local_grid: LocalGridGeometry = DEFAULT_LOCAL_GRID,
    validation_fraction: float = 0.15,
    split_seed: str = "go2_paired_navigation_v1",
    role_scenes_per_family: int | None = None,
    allow_role_transition_shortfall: bool = False,
    max_transitions_per_scene: int = 512,
    selection_seed: str = "go2_paired_navigation_selection_v1",
) -> dict[str, Any]:
    """Build scene-sharded G2 rows from RGB frame plans and exact geometry."""

    if not sources:
        raise ValueError("at least one scene source is required")
    if scene_exclusions is None and v3_exclusions is None:
        raise ValueError("held-out scene-ID exclusions are required")
    exclusions = (
        scene_exclusions
        if scene_exclusions is not None
        else v3_exclusions.as_generic()  # type: ignore[union-attr]
    )
    if scene_exclusions is not None and v3_exclusions is not None:
        exclusions = scene_exclusions.merged(v3_exclusions.as_generic())
    scene_ids = [str(source.scene_id) for source in sources]

    # This is the first operation over candidate scene identities. Reject
    # before duplicate diagnostics or any scene-owned artifact access.
    for scene_id in scene_ids:
        exclusions.assert_allowed(scene_id)
    if len(scene_ids) != len(set(scene_ids)):
        raise DatasetContractError("each scene may appear in at most one source")

    if not math.isclose(
        local_grid.cell_size_m,
        geometry_contract.configuration_space.online_cell_size_m,
        abs_tol=1e-12,
    ):
        raise DatasetContractError(
            "local label cell size must equal the geometry contract online cell size"
        )
    if role_scenes_per_family is None:
        roles: dict[str, str] | None = None
        splits = deterministic_scene_split(
            scene_ids,
            validation_fraction=validation_fraction,
            seed=split_seed,
        )
    else:
        missing_family = sorted(
            source.scene_id
            for source in sources
            if source.family is None or not str(source.family).strip()
        )
        if missing_family:
            raise DatasetContractError(
                "family-role splitting requires a declared source family for "
                "every scene; missing: " + ", ".join(missing_family)
            )
        roles = deterministic_family_role_split(
            {source.scene_id: str(source.family) for source in sources},
            role_scenes_per_family=role_scenes_per_family,
            seed=split_seed,
        )
        splits = {
            scene_id: "train" if role == "train" else "validation"
            for scene_id, role in roles.items()
        }
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / "scenes"
    shard_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "rows.jsonl"

    total_stats: Counter[str] = Counter()
    source_provenance: list[dict[str, Any]] = []
    shard_records: list[dict[str, Any]] = []
    image_hash_cache: dict[Path, str] = {}
    image_commitments: list[tuple[str, str]] = []
    global_row = 0

    with index_path.open("w") as index_stream:
        for source in sorted(sources, key=lambda item: item.scene_id):
            manifest, plan, frames_path, frames, rendered_by_key, hashes = _source_scene(
                source,
                geometry_contract=geometry_contract,
            )
            camera_plan = plan["camera"]
            fov_deg = float(camera_plan["fov_deg"])
            near_m = float(camera_plan["near_m"])
            rendered_parent = (
                source.rendered_frames_jsonl_path.resolve().parent
                if source.rendered_frames_jsonl_path is not None
                else None
            )
            rgb_dir = source.rgb_dir.resolve() if source.rgb_dir is not None else None
            configuration_occupancy = InflatedOccupancyGrid(
                manifest,
                cell_size_m=geometry_contract.configuration_space.oracle_cell_size_m,
                inflation_m=geometry_contract.configuration_space.body_inflation_radius_m,
                treat_landmarks_as_obstacles=(
                    geometry_contract.configuration_space.landmarks_are_obstacles
                ),
                treat_distractors_as_obstacles=(
                    geometry_contract.configuration_space.distractors_are_obstacles
                ),
            )
            transition_stats: Counter[str] = Counter()
            raw_transitions = list(
                iter_primitive_transitions(frames, stats=transition_stats)
            )
            candidate_transitions: list[PrimitiveTransition] = []
            configuration_rejections: Counter[str] = Counter()
            for transition in raw_transitions:
                configuration_rejection = _transition_configuration_rejection(
                    transition, configuration_occupancy
                )
                if configuration_rejection is not None:
                    transition_stats[configuration_rejection] += 1
                    configuration_rejections[configuration_rejection] += 1
                    continue
                transition_stats["transitions_configuration_valid"] += 1
                current_key = _frame_key(transition.current)
                next_key = _frame_key(transition.next)
                rendered_current = rendered_by_key.get(current_key)
                rendered_next = rendered_by_key.get(next_key)
                if source.rendered_frames_jsonl_path is not None and (
                    rendered_current is None or rendered_next is None
                ):
                    transition_stats["transitions_missing_rendered_metadata"] += 1
                    continue
                current_image = _image_for_frame(
                    transition.current,
                    rendered=rendered_current,
                    rgb_dir=rgb_dir,
                    rendered_parent=rendered_parent,
                )
                next_image = _image_for_frame(
                    transition.next,
                    rendered=rendered_next,
                    rgb_dir=rgb_dir,
                    rendered_parent=rendered_parent,
                )
                if current_image is None or next_image is None:
                    transition_stats["transitions_missing_or_invalid_rgb"] += 1
                    continue
                candidate_transitions.append(transition)
            transitions, selection_metadata = select_primitive_transitions(
                candidate_transitions,
                scene_id=source.scene_id,
                max_transitions=max_transitions_per_scene,
                seed=selection_seed,
            )
            selection_metadata["raw_transition_count"] = len(raw_transitions)
            selection_metadata["configuration_validity"] = {
                "screened_transition_count": len(raw_transitions),
                "accepted_transition_count": int(
                    transition_stats["transitions_configuration_valid"]
                ),
                "rejected_transition_count": int(sum(configuration_rejections.values())),
                "rejection_counts": dict(sorted(configuration_rejections.items())),
                "selected_transition_count": len(transitions),
                "selected_rejection_count": 0,
                "frame_scope": "complete_command_window_including_post_command_frame",
                "pose_test": "exact_configuration_clearance_m_gte_0",
                "segment_test": "raster_has_free_line_between_adjacent_frames",
            }
            total_stats.update(transition_stats)
            total_stats["raw_transitions"] += len(raw_transitions)
            total_stats["candidate_transitions"] += len(candidate_transitions)
            total_stats["selected_transitions_before_labeling"] += len(transitions)

            # Physical-visibility rasterization and per-frame raycasts remain
            # after cheap render eligibility and deterministic row selection.
            physical_visibility_occupancy = InflatedOccupancyGrid(
                manifest,
                cell_size_m=geometry_contract.configuration_space.oracle_cell_size_m,
                inflation_m=0.0,
                treat_landmarks_as_obstacles=(
                    geometry_contract.configuration_space.landmarks_are_obstacles
                ),
                treat_distractors_as_obstacles=(
                    geometry_contract.configuration_space.distractors_are_obstacles
                ),
            )

            label_cache: dict[
                tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]
            ] = {}
            labels_current: list[np.ndarray] = []
            labels_next: list[np.ndarray] = []
            supervision_current: list[np.ndarray] = []
            supervision_next: list[np.ndarray] = []
            observed_current: list[np.ndarray] = []
            observed_next: list[np.ndarray] = []
            odometry: list[np.ndarray] = []
            scene_rows: list[dict[str, Any]] = []

            for transition in transitions:
                current_key = _frame_key(transition.current)
                next_key = _frame_key(transition.next)
                rendered_current = rendered_by_key.get(current_key)
                rendered_next = rendered_by_key.get(next_key)
                current_image = _image_for_frame(
                    transition.current,
                    rendered=rendered_current,
                    rgb_dir=rgb_dir,
                    rendered_parent=rendered_parent,
                )
                next_image = _image_for_frame(
                    transition.next,
                    rendered=rendered_next,
                    rgb_dir=rgb_dir,
                    rendered_parent=rendered_parent,
                )
                if current_image is None or next_image is None:
                    raise DatasetContractError(
                        "selected transition lost RGB eligibility during one build"
                    )
                current_frame = _effective_frame(transition.current, rendered_current)
                next_frame = _effective_frame(transition.next, rendered_next)

                for key, frame in ((current_key, current_frame), (next_key, next_frame)):
                    if key not in label_cache:
                        label_cache[key] = label_camera_visible_configuration_grid(
                            configuration_occupancy,
                            physical_visibility_grid=physical_visibility_occupancy,
                            base_xy_yaw=_base_xy_yaw(frame),
                            camera=_camera_observation(
                                frame,
                                horizontal_fov_deg=fov_deg,
                                near_m=near_m,
                            ),
                            local_grid=local_grid,
                        )
                current_label, current_supervision, current_observed = label_cache[
                    current_key
                ]
                next_label, next_supervision, next_observed = label_cache[next_key]
                relative_pose = relative_se2_current_frame(
                    _base_xy_yaw(current_frame), _base_xy_yaw(next_frame)
                )
                for image_path in (current_image, next_image):
                    if image_path not in image_hash_cache:
                        image_hash_cache[image_path] = sha256_file(image_path)
                current_image_hash = image_hash_cache[current_image]
                next_image_hash = image_hash_cache[next_image]
                image_commitments.extend(
                    (
                        (str(current_image), current_image_hash),
                        (str(next_image), next_image_hash),
                    )
                )

                labels_current.append(current_label)
                labels_next.append(next_label)
                supervision_current.append(current_supervision)
                supervision_next.append(next_supervision)
                observed_current.append(current_observed)
                observed_next.append(next_observed)
                odometry.append(relative_pose)
                episode = transition.current["episode"]
                next_episode = transition.next["episode"]
                scene_rows.append(
                    {
                        "schema": "lewm_go2_paired_navigation_row_v2",
                        "global_row": global_row,
                        "scene_id": source.scene_id,
                        "family": manifest.family,
                        "dataset_split": splits[source.scene_id],
                        **(
                            {"dataset_role": roles[source.scene_id]}
                            if roles is not None
                            else {}
                        ),
                        "source_split": plan.get("split"),
                        "env_index": int(transition.current.get("env_index", 0)),
                        "episode_id": str(episode.get("episode_id")),
                        "reset_count": int(episode.get("reset_count", 0)),
                        "current_episode_step": int(episode["episode_step"]),
                        "next_episode_step": int(next_episode["episode_step"]),
                        "current_frame_index": current_key[0],
                        "next_frame_index": next_key[0],
                        "current_timestamp_ns": int(transition.current["timestamp_ns"]),
                        "next_timestamp_ns": int(transition.next["timestamp_ns"]),
                        "transition_duration_s": transition.duration_s,
                        "primitive": transition.primitive,
                        "source_transition_configuration_validated": True,
                        "relative_se2_current_frame": relative_pose.tolist(),
                        "current_image_path": str(current_image),
                        "next_image_path": str(next_image),
                        "current_image_sha256": current_image_hash,
                        "next_image_sha256": next_image_hash,
                        "geometry_contract_sha256": hashes["geometry_contract_sha256"],
                        "scene_manifest_sha256": hashes["scene_manifest_sha256"],
                        "frame_plan_sha256": hashes["frame_plan_sha256"],
                        "frames_jsonl_sha256": hashes["frames_jsonl_sha256"],
                    }
                )
                global_row += 1

            if (
                roles is not None
                and not allow_role_transition_shortfall
                and len(scene_rows) < int(max_transitions_per_scene)
            ):
                raise DatasetContractError(
                    f"direct-role scene {source.scene_id!r} produced "
                    f"{len(scene_rows)} rows, fewer than the requested "
                    f"{int(max_transitions_per_scene)}; pass "
                    "allow_role_transition_shortfall=True when the "
                    "preregistered contract caps rows per scene ('at most N') "
                    "so the shortfall is an explicit, recorded decision"
                )

            if not scene_rows:
                total_stats["scenes_without_rows"] += 1
            else:
                shard_name = f"scene_{scene_id_sha256(source.scene_id)[:16]}.npz"
                shard_path = shard_dir / shard_name
                np.savez_compressed(
                    shard_path,
                    current_labels=np.stack(labels_current).astype(np.uint8, copy=False),
                    next_labels=np.stack(labels_next).astype(np.uint8, copy=False),
                    current_supervision_mask=np.stack(supervision_current).astype(
                        bool, copy=False
                    ),
                    next_supervision_mask=np.stack(supervision_next).astype(
                        bool, copy=False
                    ),
                    current_observed_mask=np.stack(observed_current).astype(
                        bool, copy=False
                    ),
                    next_observed_mask=np.stack(observed_next).astype(bool, copy=False),
                    relative_se2_current_frame=np.stack(odometry).astype(
                        np.float32, copy=False
                    ),
                    primitive=_np_unicode([row["primitive"] for row in scene_rows]),
                    current_image_path=_np_unicode(
                        [row["current_image_path"] for row in scene_rows]
                    ),
                    next_image_path=_np_unicode(
                        [row["next_image_path"] for row in scene_rows]
                    ),
                    current_image_sha256=_np_unicode(
                        [row["current_image_sha256"] for row in scene_rows]
                    ),
                    next_image_sha256=_np_unicode(
                        [row["next_image_sha256"] for row in scene_rows]
                    ),
                )
                shard_hash = sha256_file(shard_path)
                shard_records.append(
                    {
                        "scene_id": source.scene_id,
                        "path": str(shard_path),
                        "sha256": shard_hash,
                        "rows": len(scene_rows),
                    }
                )
                for shard_row, row in enumerate(scene_rows):
                    row["label_shard_path"] = str(shard_path)
                    row["label_shard_sha256"] = shard_hash
                    row["label_shard_row"] = shard_row
                    index_stream.write(json.dumps(row, sort_keys=True) + "\n")
                total_stats["rows_written"] += len(scene_rows)

            source_paths = {
                "scene_manifest": str(source.scene_manifest_path.resolve()),
                "render_plan": str(source.render_plan_path.resolve()),
                "frames_jsonl": str(frames_path),
            }
            if source.rendered_frames_jsonl_path is not None:
                source_paths["rendered_frames_jsonl"] = str(
                    source.rendered_frames_jsonl_path.resolve()
                )
            source_provenance.append(
                {
                    "scene_id": source.scene_id,
                    "family": manifest.family,
                    "dataset_split": splits[source.scene_id],
                    **(
                        {"dataset_role": roles[source.scene_id]}
                        if roles is not None
                        else {}
                    ),
                    "paths": source_paths,
                    "hashes": hashes,
                    "rows": len(scene_rows),
                    "transition_stats": dict(sorted(transition_stats.items())),
                    "selection": selection_metadata,
                }
            )

    if global_row == 0:
        raise DatasetContractError("no valid paired-navigation rows were produced")
    index_hash = sha256_file(index_path)
    split_counts = Counter(splits[record["scene_id"]] for record in source_provenance)
    manifest_payload: dict[str, Any] = {
        "schema": "lewm_go2_paired_navigation_dataset_v2",
        "row_count": global_row,
        "scene_count": len(source_provenance),
        "scene_split_counts": dict(sorted(split_counts.items())),
        "split": {
            "unit": "scene_id",
            "seed": split_seed,
            **(
                {
                    "validation_fraction": float(validation_fraction),
                    "assignment": "sha256(seed + NUL + scene_id) threshold",
                }
                if roles is None
                else {
                    "validation_fraction": None,
                    "assignment": "direct_family_role_contract",
                }
            ),
        },
        "exclusions": exclusions.to_metadata(),
        "local_grid": local_grid.to_metadata(),
        "label_semantics": {
            "unknown": UNKNOWN_CLASS,
            "free": FREE_CLASS,
            "occupied": OCCUPIED_CLASS,
            "supervision_mask": "finite local-grid cells; includes UNKNOWN targets",
            "observed_mask": "supervision_mask and label != UNKNOWN",
            "unknown_is_supervised": True,
            "target_occupancy_space": "body_inflated_configuration_space",
            "visibility_occlusion_space": "uninflated_physical_obstacle_occupancy",
            "visibility": "recorded_camera_horizontal_fov_first_physical_obstacle",
            "configuration_inflation_radius_m": (
                geometry_contract.configuration_space.body_inflation_radius_m
            ),
            "visibility_inflation_radius_m": 0.0,
            "visibility_raster_cell_size_m": (
                geometry_contract.configuration_space.oracle_cell_size_m
            ),
            "privileged_input_at_runtime": False,
        },
        "transition_contract": {
            "one_complete_command_block": True,
            "nominal_duration_s": 0.5,
            "grouping": ["env_index", "episode_id", "reset_count"],
            "source_configuration_validity": {
                "required": True,
                "frame_scope": (
                    "complete_command_window_including_post_command_frame"
                ),
                "pose_test": "exact_configuration_clearance_m_gte_0",
                "segment_test": (
                    "raster_has_free_line_between_adjacent_frames"
                ),
                "configuration_inflation_radius_m": (
                    geometry_contract.configuration_space.body_inflation_radius_m
                ),
                "configuration_raster_cell_size_m": (
                    geometry_contract.configuration_space.oracle_cell_size_m
                ),
            },
            "relative_se2": [
                "next_forward_in_current_base",
                "next_left_in_current_base",
                "wrapped_delta_yaw",
            ],
        },
        "row_selection": {
            "selection_seed": selection_seed,
            "max_transitions_per_scene": int(max_transitions_per_scene),
            "occurs_before_label_raycast": True,
            "occurs_after_configuration_validity_filter": True,
            "method": "hash_rank_within_primitive_env_episode_strata_then_round_robin",
        },
        "geometry_contract": {
            "path": str(geometry_contract.source_path),
            "sha256": geometry_contract.sha256,
            "file_sha256": sha256_file(geometry_contract.source_path),
            "oracle_cell_size_m": geometry_contract.configuration_space.oracle_cell_size_m,
            "body_inflation_radius_m": (
                geometry_contract.configuration_space.body_inflation_radius_m
            ),
        },
        "index": {
            "path": str(index_path),
            "sha256": index_hash,
        },
        "image_set_sha256": canonical_json_sha256(sorted(set(image_commitments))),
        "shards": shard_records,
        "sources": source_provenance,
        "stats": dict(sorted(total_stats.items())),
    }
    if roles is not None:
        role_scene_ids = {
            role: sorted(
                scene_id for scene_id, assigned_role in roles.items()
                if assigned_role == role
            )
            for role in DATASET_ROLES
        }
        family_role_scene_counts: dict[str, dict[str, int]] = {}
        family_role_row_counts: dict[str, dict[str, int]] = {}
        role_row_counts = {role: 0 for role in DATASET_ROLES}
        for record in source_provenance:
            family = str(record["family"])
            role = str(record["dataset_role"])
            family_role_scene_counts.setdefault(
                family, {item: 0 for item in DATASET_ROLES}
            )[role] += 1
            family_role_row_counts.setdefault(
                family, {item: 0 for item in DATASET_ROLES}
            )[role] += int(record["rows"])
            role_row_counts[role] += int(record["rows"])
        manifest_payload["scene_roles"] = {
            "schema": "lewm_go2_family_scene_roles_v1",
            "unit": "scene_id",
            "seed": split_seed,
            "role_scenes_per_family": int(role_scenes_per_family),
            "assignment": (
                "sha256(seed + NUL + family + NUL + scene_id) ascending; "
                "fixed checkpoint_selection, probability_calibration, "
                "g2_evaluation slices; remainder train"
            ),
            "assignments": roles,
            "assignments_sha256": canonical_json_sha256(roles),
            "scene_counts": {
                role: len(scene_ids_for_role)
                for role, scene_ids_for_role in role_scene_ids.items()
            },
            "row_counts": dict(sorted(role_row_counts.items())),
            "family_scene_counts": {
                family: dict(sorted(counts.items()))
                for family, counts in sorted(family_role_scene_counts.items())
            },
            "family_row_counts": {
                family: dict(sorted(counts.items()))
                for family, counts in sorted(family_role_row_counts.items())
            },
            "scene_id_sha256_commitments": {
                role: canonical_json_sha256(
                    sorted(scene_id_sha256(scene_id) for scene_id in scene_ids_for_role)
                )
                for role, scene_ids_for_role in role_scene_ids.items()
            },
            "label_independent": True,
            "transition_shortfall_allowed": bool(allow_role_transition_shortfall),
        }
    if v3_exclusions is not None:
        manifest_payload["v3_exclusions"] = v3_exclusions.to_metadata()
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n")
    return manifest_payload


def verify_dataset_provenance(
    dataset_manifest_path: Path,
    *,
    verify_images: bool = True,
) -> dict[str, int]:
    """Re-hash every recorded source, shard, index, and optionally every RGB."""

    payload = _read_json(dataset_manifest_path)
    if payload.get("schema") != "lewm_go2_paired_navigation_dataset_v2":
        raise ProvenanceError("unsupported paired-navigation dataset schema")
    checked = Counter()

    def check(path_value: str, expected: str, kind: str) -> None:
        path = Path(path_value)
        if not path.is_file():
            raise ProvenanceError(f"missing {kind}: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ProvenanceError(
                f"{kind} hash mismatch for {path}: expected {expected}, got {actual}"
            )
        checked[kind] += 1

    geometry = payload["geometry_contract"]
    check(geometry["path"], geometry["file_sha256"], "geometry_contract")
    checked_exclusion_files: set[tuple[str, str]] = set()
    exclusions = payload.get("exclusions", {})
    for exclusion in exclusions.get("sets", {}).values():
        if "file" not in exclusion:
            continue
        item = (str(exclusion["file"]), str(exclusion["file_sha256"]))
        if item in checked_exclusion_files:
            continue
        checked_exclusion_files.add(item)
        check(item[0], item[1], "exclusion_commitment")
    index = payload["index"]
    check(index["path"], index["sha256"], "index")
    for shard in payload.get("shards", []):
        check(shard["path"], shard["sha256"], "shard")
    for source in payload.get("sources", []):
        paths = source["paths"]
        hashes = source["hashes"]
        check(paths["scene_manifest"], hashes["scene_manifest_file_sha256"], "scene_manifest")
        check(paths["render_plan"], hashes["frame_plan_sha256"], "frame_plan")
        check(paths["frames_jsonl"], hashes["frames_jsonl_sha256"], "frames_jsonl")
        if "rendered_frames_jsonl" in paths:
            check(
                paths["rendered_frames_jsonl"],
                hashes["rendered_frames_jsonl_sha256"],
                "rendered_frames_jsonl",
            )
    if verify_images:
        seen_images: set[tuple[str, str]] = set()
        for row in _read_jsonl(Path(index["path"])):
            for prefix in ("current", "next"):
                item = (row[f"{prefix}_image_path"], row[f"{prefix}_image_sha256"])
                if item in seen_images:
                    continue
                seen_images.add(item)
                check(item[0], item[1], "image")
        if canonical_json_sha256(sorted(seen_images)) != payload["image_set_sha256"]:
            raise ProvenanceError("image-set commitment does not match row index")
    return dict(sorted(checked.items()))


def load_source_index(path: Path) -> list[SceneRenderSource]:
    """Load an explicit JSONL source index; directory discovery is deliberate absent."""

    parent = path.resolve().parent

    def source_path(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else parent / candidate

    sources: list[SceneRenderSource] = []
    for row in _read_jsonl(path):
        sources.append(
            SceneRenderSource(
                scene_id=str(row["scene_id"]),
                scene_manifest_path=source_path(str(row["scene_manifest_path"])),
                render_plan_path=source_path(str(row["render_plan_path"])),
                family=(str(row["family"]) if row.get("family") else None),
                rgb_dir=(
                    source_path(str(row["rgb_dir"])) if row.get("rgb_dir") else None
                ),
                frames_jsonl_path=(
                    source_path(str(row["frames_jsonl_path"]))
                    if row.get("frames_jsonl_path")
                    else None
                ),
                rendered_frames_jsonl_path=(
                    source_path(str(row["rendered_frames_jsonl_path"]))
                    if row.get("rendered_frames_jsonl_path")
                    else None
                ),
            )
        )
    return sources


__all__ = [
    "CameraObservation",
    "DatasetContractError",
    "DEFAULT_LOCAL_GRID",
    "DATASET_ROLES",
    "ForbiddenSceneError",
    "FREE_CLASS",
    "LocalGridGeometry",
    "OCCUPIED_CLASS",
    "PrimitiveTransition",
    "ProvenanceError",
    "SceneIdCommitmentSet",
    "SceneIdExclusions",
    "SceneRenderSource",
    "UNKNOWN_CLASS",
    "V3SceneExclusions",
    "build_paired_navigation_dataset",
    "canonical_json_sha256",
    "deterministic_scene_split",
    "deterministic_family_role_split",
    "iter_primitive_transitions",
    "label_camera_visible_configuration_grid",
    "load_scene_id_exclusions",
    "load_source_index",
    "read_scene_id_commitments",
    "relative_se2_current_frame",
    "occupancy_label_metrics",
    "select_primitive_transitions",
    "scene_id_sha256",
    "sha256_file",
    "verify_dataset_provenance",
]
