"""Scene-disjoint paired RGB/BEV navigation data with strict provenance.

The labels in this module are privileged offline targets.  Runtime consumers
receive RGB only; no simulator depth or privileged geometry is part of a row's
model input.
"""
from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import re
import shutil
from collections import Counter, defaultdict
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from lewm.planning.geometry_contract import GeometryContract
from lewm_worlds.manifest import (
    BoxObject,
    SceneManifest,
    manifest_sha256,
    parse_scene_manifest_dict,
)
from lewm_worlds.planning_grid import InflatedOccupancyGrid


UNKNOWN_CLASS = 0
FREE_CLASS = 1
OCCUPIED_CLASS = 2
LABEL_CONTRACT_CENTER_VISIBLE_V2 = "center_visible_configuration_v2"
LABEL_CONTRACT_OBSERVABLE_PHYSICAL_V3 = "observable_physical_occupancy_v3"
OBSERVABLE_FOOTPRINT_RADIUS_M = 0.47
OBSERVABLE_CAMERA_NATIVE_ASPECT_HEIGHT_OVER_WIDTH = 3.0 / 4.0
OBSERVABLE_GROUND_PLANE_Z_M = 0.0
SUPPORTED_LABEL_CONTRACTS = (
    LABEL_CONTRACT_CENTER_VISIBLE_V2,
    LABEL_CONTRACT_OBSERVABLE_PHYSICAL_V3,
)
_DATASET_SCHEMA_BY_LABEL_CONTRACT = {
    LABEL_CONTRACT_CENTER_VISIBLE_V2: "lewm_go2_paired_navigation_dataset_v2",
    LABEL_CONTRACT_OBSERVABLE_PHYSICAL_V3: (
        "lewm_go2_paired_navigation_dataset_v3"
    ),
}
_ROW_SCHEMA_BY_LABEL_CONTRACT = {
    LABEL_CONTRACT_CENTER_VISIBLE_V2: "lewm_go2_paired_navigation_row_v2",
    LABEL_CONTRACT_OBSERVABLE_PHYSICAL_V3: "lewm_go2_paired_navigation_row_v3",
}
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
    vertical_fov_deg: float | None = None
    up_xyz: tuple[float, float, float] = (0.0, 0.0, 1.0)
    ground_plane_z_m: float = OBSERVABLE_GROUND_PLANE_Z_M
    image_width_px: int = 224
    image_height_px: int = 168
    obstacle_ray_stride_px: int = 2


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
    render_summary_path: Path | None = None


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


def _validated_build_provenance(
    payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate the self-contained v3 dataset build identity record."""

    if not isinstance(payload, Mapping):
        raise DatasetContractError(
            "observable physical v3 requires explicit build_provenance"
        )
    record = json.loads(json.dumps(dict(payload)))
    if record.get("schema") != "lewm_experiment_manifest_v1":
        raise DatasetContractError("unsupported dataset build provenance schema")
    if not str(record.get("created_at_utc", "")).strip():
        raise DatasetContractError("dataset build provenance lacks creation time")
    if not str(record.get("run_command", "")).strip():
        raise DatasetContractError("dataset build provenance lacks invocation")
    config = record.get("config")
    if (
        not isinstance(config, Mapping)
        or record.get("config_sha256") != canonical_json_sha256(config)
    ):
        raise DatasetContractError("dataset build provenance config hash mismatch")
    git = record.get("git")
    if (
        not isinstance(git, Mapping)
        or not re.fullmatch(r"[0-9a-f]{40}", str(git.get("commit", "")))
        or not _SHA256_RE.fullmatch(str(git.get("dirty_diff_sha256", "")))
    ):
        raise DatasetContractError("dataset build provenance lacks git identity")
    inputs = record.get("inputs")
    required_inputs = {
        "builder_source",
        "dataset_source",
        "source_index",
        "render_audit_contract",
    }
    if not isinstance(inputs, Mapping) or not required_inputs.issubset(inputs):
        raise DatasetContractError("dataset build provenance lacks required inputs")
    for name, artifact in inputs.items():
        if (
            not isinstance(artifact, Mapping)
            or not str(artifact.get("path", "")).strip()
            or not _SHA256_RE.fullmatch(str(artifact.get("sha256", "")))
        ):
            raise DatasetContractError(
                f"invalid dataset build provenance input: {name}"
            )
    geometry = record.get("dependencies", {}).get("geometry_contract")
    if (
        not isinstance(geometry, Mapping)
        or not str(geometry.get("path", "")).strip()
        or not _SHA256_RE.fullmatch(str(geometry.get("sha256", "")))
    ):
        raise DatasetContractError(
            "dataset build provenance lacks geometry dependency"
        )
    return record


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


def _raycast_camera_visible_physical_grid(
    physical_visibility_grid: InflatedOccupancyGrid,
    *,
    base_xy_yaw: Sequence[float],
    camera: CameraObservation,
    local_grid: LocalGridGeometry = DEFAULT_LOCAL_GRID,
    ray_step_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Raycast the camera into an uninflated physical occupancy raster.

    Visibility uses the recorded camera origin and horizontal look direction.
    Rays stop at their first *physical* obstacle sample in the uninflated
    visibility grid; cells behind that sample remain UNKNOWN.  This privileged
    offline raycast intentionally does not consume rendered depth.
    """

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

    target_physical_free = _world_points_to_grid_free(
        physical_visibility_grid, world_x, world_y
    )
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
    labels[clear_to_center & target_physical_free] = FREE_CLASS
    labels[clear_to_center & ~target_physical_free] = OCCUPIED_CLASS
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
    return labels, supervision_mask, observed_mask, world_x, world_y


def label_camera_visible_physical_grid(
    physical_visibility_grid: InflatedOccupancyGrid,
    *,
    base_xy_yaw: Sequence[float],
    camera: CameraObservation,
    local_grid: LocalGridGeometry = DEFAULT_LOCAL_GRID,
    ray_step_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return camera-observable physical FREE/OCCUPIED/UNKNOWN labels."""

    labels, supervision, observed, _, _ = _raycast_camera_visible_physical_grid(
        physical_visibility_grid,
        base_xy_yaw=base_xy_yaw,
        camera=camera,
        local_grid=local_grid,
        ray_step_m=ray_step_m,
    )
    return labels, supervision, observed


def vertical_fov_from_horizontal(
    horizontal_fov_deg: float,
    *,
    image_width: int,
    image_height: int,
) -> float:
    """Return rectilinear vertical FOV from horizontal FOV and native aspect."""

    horizontal = float(horizontal_fov_deg)
    width = int(image_width)
    height = int(image_height)
    if not 0.0 < horizontal < 180.0:
        raise ValueError("horizontal_fov_deg must lie in (0, 180)")
    if width <= 0 or height <= 0:
        raise ValueError("native image dimensions must be positive")
    return math.degrees(
        2.0
        * math.atan(
            math.tan(math.radians(horizontal) * 0.5) * float(height) / float(width)
        )
    )


def _camera_basis(
    camera: CameraObservation,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    position = np.asarray(camera.position_xyz_m, dtype=np.float64)
    lookat = np.asarray(camera.lookat_xyz_m, dtype=np.float64)
    up_hint = np.asarray(camera.up_xyz, dtype=np.float64)
    if position.shape != (3,) or lookat.shape != (3,) or up_hint.shape != (3,):
        raise DatasetContractError("camera position/lookat/up must contain three values")
    if not (
        np.isfinite(position).all()
        and np.isfinite(lookat).all()
        and np.isfinite(up_hint).all()
    ):
        raise DatasetContractError("camera position/lookat/up must be finite")
    forward = lookat - position
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm <= 1e-9:
        raise DatasetContractError("camera look direction is degenerate")
    forward /= forward_norm
    right = np.cross(forward, up_hint)
    right_norm = float(np.linalg.norm(right))
    if right_norm <= 1e-9:
        raise DatasetContractError("camera up is parallel to its look direction")
    right /= right_norm
    up = np.cross(right, forward)
    up /= float(np.linalg.norm(up))
    horizontal = float(camera.horizontal_fov_deg)
    vertical = camera.vertical_fov_deg
    if not 0.0 < horizontal < 180.0:
        raise DatasetContractError("horizontal camera FOV must lie in (0, 180) degrees")
    if vertical is None or not 0.0 < float(vertical) < 180.0:
        raise DatasetContractError(
            "observable physical labels require a vertical camera FOV in (0, 180)"
        )
    return (
        position,
        forward,
        right,
        up,
        math.tan(math.radians(horizontal) * 0.5),
        math.tan(math.radians(float(vertical)) * 0.5),
    )


def _rectilinear_frustum_mask(
    points_xyz: np.ndarray,
    camera: CameraObservation,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape [N, 3]")
    position, forward, right, up, tan_h, tan_v = _camera_basis(camera)
    relative = points - position[None, :]
    forward_cam = relative @ forward
    right_cam = relative @ right
    up_cam = relative @ up
    finite = np.isfinite(relative).all(axis=1)
    near = float(camera.near_m)
    if not math.isfinite(near) or near < 0.0:
        raise DatasetContractError("camera near plane must be finite and non-negative")
    visible = (
        finite
        & (forward_cam > near)
        & (np.abs(right_cam) <= forward_cam * tan_h + 1e-12)
        & (np.abs(up_cam) <= forward_cam * tan_v + 1e-12)
    )
    return visible, np.linalg.norm(relative, axis=1)


def _ray_box_entry_distances(
    camera_xyz: np.ndarray,
    directions_xyz: np.ndarray,
    box: BoxObject,
) -> np.ndarray:
    """Vectorized ray entry distance for one yaw-oriented 3D box."""

    center = np.asarray(box.center_xyz_m, dtype=np.float64)
    half = 0.5 * np.asarray(box.size_xyz_m, dtype=np.float64)
    rotation = _box_rotation_matrix(box)
    origin_delta = np.asarray(camera_xyz, dtype=np.float64) - center
    origin = rotation.T @ origin_delta
    directions = np.asarray(directions_xyz, dtype=np.float64)
    local = directions @ rotation
    t_min = np.full(directions.shape[0], -np.inf, dtype=np.float64)
    t_max = np.full(directions.shape[0], np.inf, dtype=np.float64)
    valid = np.ones(directions.shape[0], dtype=bool)
    for axis in range(3):
        component = local[:, axis]
        parallel = np.abs(component) <= 1e-12
        valid &= ~(parallel & (abs(origin[axis]) > half[axis] + 1e-12))
        nonparallel = ~parallel
        low = np.full(component.shape, -np.inf, dtype=np.float64)
        high = np.full(component.shape, np.inf, dtype=np.float64)
        low[nonparallel] = (
            -half[axis] - origin[axis]
        ) / component[nonparallel]
        high[nonparallel] = (
            half[axis] - origin[axis]
        ) / component[nonparallel]
        swap = low > high
        low[swap], high[swap] = high[swap], low[swap]
        t_min = np.maximum(t_min, low)
        t_max = np.minimum(t_max, high)
    entry = np.maximum(t_min, 0.0)
    valid &= t_max + 1e-12 >= entry
    return np.where(valid, entry, np.inf)


def _box_rotation_matrix(box: BoxObject) -> np.ndarray:
    roll = float(box.roll_rad)
    pitch = float(box.pitch_rad)
    yaw = float(box.yaw_rad)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation_x = np.asarray(
        ((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr)),
        dtype=np.float64,
    )
    rotation_y = np.asarray(
        ((cp, 0.0, sp), (0.0, 1.0, 0.0), (-sp, 0.0, cp)),
        dtype=np.float64,
    )
    rotation_z = np.asarray(
        ((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    return rotation_z @ rotation_y @ rotation_x


def _first_box_hits(
    points_xyz: np.ndarray,
    camera: CameraObservation,
    obstacle_boxes: Sequence[BoxObject],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points_xyz, dtype=np.float64)
    frustum, distances = _rectilinear_frustum_mask(points, camera)
    position = np.asarray(camera.position_xyz_m, dtype=np.float64)
    nonzero = distances > 1e-12
    nearest_distance = np.full(points.shape[0], np.inf, dtype=np.float64)
    nearest_index = np.full(points.shape[0], -1, dtype=np.int32)
    active = np.flatnonzero(frustum & nonzero)
    if active.size == 0:
        return frustum, distances, nearest_distance, nearest_index
    directions = (
        points[active] - position[None, :]
    ) / distances[active, None]
    active_nearest = np.full(active.size, np.inf, dtype=np.float64)
    active_index = np.full(active.size, -1, dtype=np.int32)
    for box_index, box in enumerate(obstacle_boxes):
        entry = _ray_box_entry_distances(position, directions, box)
        closer = entry < active_nearest - 1e-10
        active_nearest[closer] = entry[closer]
        active_index[closer] = box_index
    nearest_distance[active] = active_nearest
    nearest_index[active] = active_index
    return frustum, distances, nearest_distance, nearest_index


def _visible_obstacle_camera_ray_witnesses_xy(
    camera: CameraObservation,
    obstacle_boxes: Sequence[BoxObject],
) -> np.ndarray:
    """Cast a bounded pinhole lattice and return exact nearest box-hit xy."""

    if not obstacle_boxes:
        return np.zeros((0, 2), dtype=np.float64)
    width = int(camera.image_width_px)
    height = int(camera.image_height_px)
    stride = int(camera.obstacle_ray_stride_px)
    if width <= 0 or height <= 0 or stride <= 0:
        raise DatasetContractError("camera image dimensions and ray stride must be positive")
    position, forward, right, up, tan_h, tan_v = _camera_basis(camera)
    pixel_x = np.arange(0, width, stride, dtype=np.float64) + 0.5 * stride
    pixel_y = np.arange(0, height, stride, dtype=np.float64) + 0.5 * stride
    pixel_x = np.minimum(pixel_x, width - 0.5)
    pixel_y = np.minimum(pixel_y, height - 0.5)
    normalized_x = (2.0 * pixel_x / float(width) - 1.0) * tan_h
    normalized_y = (1.0 - 2.0 * pixel_y / float(height)) * tan_v
    grid_x, grid_y = np.meshgrid(normalized_x, normalized_y, indexing="xy")
    directions = (
        forward[None, :]
        + grid_x.ravel()[:, None] * right[None, :]
        + grid_y.ravel()[:, None] * up[None, :]
    )
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    nearest = np.full(directions.shape[0], np.inf, dtype=np.float64)
    for box in obstacle_boxes:
        nearest = np.minimum(
            nearest,
            _ray_box_entry_distances(position, directions, box),
        )
    valid = np.isfinite(nearest) & (nearest > float(camera.near_m))
    if not np.any(valid):
        return np.zeros((0, 2), dtype=np.float64)
    first_hits = position[None, :] + directions[valid] * nearest[valid, None]
    return np.unique(np.round(first_hits[:, :2], decimals=12), axis=0)


def derive_configuration_labels_from_fused_physical_raster(
    physical_labels: np.ndarray,
    *,
    physical_x_centers_m: np.ndarray,
    physical_y_centers_m: np.ndarray,
    configuration_world_x_m: np.ndarray,
    configuration_world_y_m: np.ndarray,
    footprint_radius_m: float,
    physical_cell_size_m: float,
) -> np.ndarray:
    """Derive evaluation-only configuration classes after memory fusion.

    This function is never used to construct a per-frame v3 training target.
    FREE uses every world-aligned fused physical cell whose closed square
    intersects the closed footprint disc.  OCCUPIED requires the stronger
    witness that an observed occupied raster *center* lies inside the disc;
    a cell that only touches the disc can therefore withhold FREE but cannot
    create a false OCCUPIED target.  This keeps 0.05 m obstacles between
    0.10 m output centers in scope without expanding the collision radius.
    """

    physical = np.asarray(physical_labels)
    x_centers = np.asarray(physical_x_centers_m, dtype=np.float64)
    y_centers = np.asarray(physical_y_centers_m, dtype=np.float64)
    config_x = np.asarray(configuration_world_x_m, dtype=np.float64)
    config_y = np.asarray(configuration_world_y_m, dtype=np.float64)
    if physical.shape != (x_centers.size, y_centers.size):
        raise ValueError("physical label shape does not match its center axes")
    if config_x.shape != config_y.shape:
        raise ValueError("configuration world-coordinate arrays must match")
    if not np.isin(physical, (UNKNOWN_CLASS, FREE_CLASS, OCCUPIED_CLASS)).all():
        raise ValueError("physical_labels contains an unsupported class")
    radius = float(footprint_radius_m)
    cell_size = float(physical_cell_size_m)
    if radius <= 0.0 or cell_size <= 0.0:
        raise ValueError("footprint radius and physical cell size must be positive")
    half_cell = 0.5 * cell_size
    labels = np.full(config_x.shape, UNKNOWN_CLASS, dtype=np.uint8)
    for output_index in np.ndindex(config_x.shape):
        center_x = float(config_x[output_index])
        center_y = float(config_y[output_index])
        x_candidates = np.flatnonzero(
            np.abs(x_centers - center_x) <= radius + half_cell + 1e-12
        )
        y_candidates = np.flatnonzero(
            np.abs(y_centers - center_y) <= radius + half_cell + 1e-12
        )
        if x_candidates.size == 0 or y_candidates.size == 0:
            continue
        dx = np.maximum(
            np.abs(x_centers[x_candidates] - center_x) - half_cell, 0.0
        )
        dy = np.maximum(
            np.abs(y_centers[y_candidates] - center_y) - half_cell, 0.0
        )
        intersects = (
            dx[:, None] * dx[:, None] + dy[None, :] * dy[None, :]
            <= radius * radius + 1e-12
        )
        support = physical[np.ix_(x_candidates, y_candidates)][intersects]
        center_inside_disc = (
            (x_centers[x_candidates, None] - center_x) ** 2
            + (y_centers[None, y_candidates] - center_y) ** 2
            <= radius * radius + 1e-12
        )
        candidate_labels = physical[np.ix_(x_candidates, y_candidates)]
        if np.any(
            (candidate_labels == OCCUPIED_CLASS) & center_inside_disc
        ):
            labels[output_index] = OCCUPIED_CLASS
        elif support.size > 0 and np.all(support == FREE_CLASS):
            labels[output_index] = FREE_CLASS
    return labels


def observable_physical_labels_from_raster(
    physical_labels: np.ndarray,
    *,
    physical_x_centers_m: np.ndarray,
    physical_y_centers_m: np.ndarray,
    output_world_x_m: np.ndarray,
    output_world_y_m: np.ndarray,
    output_yaw_rad: float,
    physical_cell_size_m: float,
    output_cell_size_m: float,
    visible_obstacle_first_hit_xy_m: np.ndarray | None = None,
) -> np.ndarray:
    """Conservatively aggregate 0.05 m physical evidence to output cells.

    FREE requires every world-aligned source cell square intersecting the
    rotated output square to carry visible-free evidence. OCCUPIED requires an
    exact first-visible obstacle-surface hit inside the output square. Neither
    class is obtained by inflating an obstacle to the robot footprint.
    """

    physical = np.asarray(physical_labels)
    x_centers = np.asarray(physical_x_centers_m, dtype=np.float64)
    y_centers = np.asarray(physical_y_centers_m, dtype=np.float64)
    output_x = np.asarray(output_world_x_m, dtype=np.float64)
    output_y = np.asarray(output_world_y_m, dtype=np.float64)
    if physical.shape != (x_centers.size, y_centers.size):
        raise ValueError("physical label shape does not match its center axes")
    if output_x.shape != output_y.shape:
        raise ValueError("output world-coordinate arrays must match")
    if not np.isin(physical, (UNKNOWN_CLASS, FREE_CLASS, OCCUPIED_CLASS)).all():
        raise ValueError("physical_labels contains an unsupported class")
    source_half = 0.5 * float(physical_cell_size_m)
    output_half = 0.5 * float(output_cell_size_m)
    if source_half <= 0.0 or output_half <= 0.0:
        raise ValueError("physical and output cell sizes must be positive")
    yaw = float(output_yaw_rad)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    output_u = np.asarray((cos_yaw, sin_yaw), dtype=np.float64)
    output_v = np.asarray((-sin_yaw, cos_yaw), dtype=np.float64)
    world_extent = output_half * (abs(cos_yaw) + abs(sin_yaw)) + source_half
    labels = np.full(output_x.shape, UNKNOWN_CLASS, dtype=np.uint8)
    for output_index in np.ndindex(output_x.shape):
        center_x = float(output_x[output_index])
        center_y = float(output_y[output_index])
        x_candidates = np.flatnonzero(
            np.abs(x_centers - center_x) <= world_extent + 1e-12
        )
        y_candidates = np.flatnonzero(
            np.abs(y_centers - center_y) <= world_extent + 1e-12
        )
        if x_candidates.size == 0 or y_candidates.size == 0:
            continue
        candidate_x, candidate_y = np.meshgrid(
            x_centers[x_candidates], y_centers[y_candidates], indexing="ij"
        )
        dx = candidate_x - center_x
        dy = candidate_y - center_y
        along_u = dx * output_u[0] + dy * output_u[1]
        along_v = dx * output_v[0] + dy * output_v[1]
        intersects = (
            (np.abs(along_u) <= output_half + source_half * (
                abs(output_u[0]) + abs(output_u[1])
            ) + 1e-12)
            & (np.abs(along_v) <= output_half + source_half * (
                abs(output_v[0]) + abs(output_v[1])
            ) + 1e-12)
            & (np.abs(dx) <= world_extent + 1e-12)
            & (np.abs(dy) <= world_extent + 1e-12)
        )
        support = physical[np.ix_(x_candidates, y_candidates)][intersects]
        if support.size > 0 and np.all(support == FREE_CLASS):
            labels[output_index] = FREE_CLASS

    witnesses = (
        np.zeros((0, 2), dtype=np.float64)
        if visible_obstacle_first_hit_xy_m is None
        else np.asarray(visible_obstacle_first_hit_xy_m, dtype=np.float64)
    )
    if witnesses.ndim != 2 or witnesses.shape[1] != 2:
        raise ValueError("visible obstacle witnesses must have shape [N, 2]")
    if witnesses.size:
        flat_x = output_x.ravel()
        flat_y = output_y.ravel()
        witnessed = np.zeros(flat_x.size, dtype=bool)
        for start in range(0, witnesses.shape[0], 512):
            batch = witnesses[start : start + 512]
            dx = batch[None, :, 0] - flat_x[:, None]
            dy = batch[None, :, 1] - flat_y[:, None]
            forward = cos_yaw * dx + sin_yaw * dy
            left = -sin_yaw * dx + cos_yaw * dy
            witnessed |= np.any(
                (np.abs(forward) <= output_half + 1e-12)
                & (np.abs(left) <= output_half + 1e-12),
                axis=1,
            )
        labels.ravel()[witnessed] = OCCUPIED_CLASS
    return labels


def _output_cells_intersect_collision_geometry(
    output_x: np.ndarray,
    output_y: np.ndarray,
    *,
    output_yaw_rad: float,
    output_cell_size_m: float,
    obstacle_boxes: Sequence[BoxObject],
) -> np.ndarray:
    """Conservative 2D overlap veto for physical FREE output cells."""

    veto = np.zeros(output_x.shape, dtype=bool)
    output_half = 0.5 * float(output_cell_size_m)
    yaw = float(output_yaw_rad)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    for box in obstacle_boxes:
        rotation = _box_rotation_matrix(box)
        half = 0.5 * np.asarray(box.size_xyz_m, dtype=np.float64)
        local_corners = np.asarray(
            [
                (sx * half[0], sy * half[1], sz * half[2])
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ],
            dtype=np.float64,
        )
        world_corners = (
            local_corners @ rotation.T
            + np.asarray(box.center_xyz_m, dtype=np.float64)[None, :]
        )
        x_low, y_low = np.min(world_corners[:, :2], axis=0)
        x_high, y_high = np.max(world_corners[:, :2], axis=0)
        box_center_x = 0.5 * (x_low + x_high)
        box_center_y = 0.5 * (y_low + y_high)
        box_half_x = 0.5 * (x_high - x_low)
        box_half_y = 0.5 * (y_high - y_low)
        dx = output_x - box_center_x
        dy = output_y - box_center_y
        along_u = cos_yaw * dx + sin_yaw * dy
        along_v = -sin_yaw * dx + cos_yaw * dy
        veto |= (
            (np.abs(dx) <= box_half_x + output_half * (
                abs(cos_yaw) + abs(sin_yaw)
            ) + 1e-12)
            & (np.abs(dy) <= box_half_y + output_half * (
                abs(cos_yaw) + abs(sin_yaw)
            ) + 1e-12)
            & (np.abs(along_u) <= output_half + box_half_x * abs(cos_yaw)
                + box_half_y * abs(sin_yaw) + 1e-12)
            & (np.abs(along_v) <= output_half + box_half_x * abs(sin_yaw)
                + box_half_y * abs(cos_yaw) + 1e-12)
        )
    return veto


def post_memory_configuration_morphology_metadata(
    *,
    radius_m: float,
    physical_cell_size_m: float,
) -> dict[str, Any]:
    support_contract = {
        "schema": "lewm_post_memory_configuration_morphology_v1",
        "radius_m": float(radius_m),
        "memory_cell_size_m": float(physical_cell_size_m),
        "input": "multi_view_fused_physical_FREE_OCCUPIED_UNKNOWN_belief",
        "applied_during_per_frame_supervision": False,
        "footprint_shape": "closed_yaw_invariant_disc",
        "free_support_inclusion_rule": (
            "configuration FREE only when every fused physical belief cell "
            "whose closed square intersects the disc is FREE"
        ),
        "occupied_witness_inclusion_rule": (
            "configuration OCCUPIED when a fused OCCUPIED cell center lies "
            "within radius_m"
        ),
        "otherwise": "UNKNOWN",
        "support_frame": "online_memory_map",
        "support_is_pose_dependent": True,
        "exact_support_reconstruction_inputs": [
            "online memory map origin",
            "queried configuration center",
            "radius_m",
            "memory_cell_size_m",
        ],
    }
    return {
        **support_contract,
        "support_contract_sha256": canonical_json_sha256(support_contract),
    }


def _observable_physical_raster_and_output_labels(
    physical_visibility_grid: InflatedOccupancyGrid,
    *,
    rendered_obstacle_boxes: Sequence[BoxObject],
    collision_obstacle_boxes: Sequence[BoxObject],
    base_xy_yaw: Sequence[float],
    camera: CameraObservation,
    local_grid: LocalGridGeometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive full-frustum physical evidence at the 0.10 m output resolution."""

    base_x, base_y, base_yaw = map(float, base_xy_yaw)
    forward, left = np.meshgrid(
        local_grid.forward_centers_m(),
        local_grid.left_centers_m(),
        indexing="ij",
    )
    cos_yaw = math.cos(base_yaw)
    sin_yaw = math.sin(base_yaw)
    config_x = base_x + cos_yaw * forward - sin_yaw * left
    config_y = base_y + sin_yaw * forward + cos_yaw * left

    cell_size = float(physical_visibility_grid.cell_size_m)
    half_cell = 0.5 * cell_size
    output_half = 0.5 * float(local_grid.cell_size_m)
    origin_x, origin_y = physical_visibility_grid.origin_xy
    x_low = float(np.min(config_x)) - output_half - half_cell
    x_high = float(np.max(config_x)) + output_half + half_cell
    y_low = float(np.min(config_y)) - output_half - half_cell
    y_high = float(np.max(config_y)) + output_half + half_cell
    ix_low = int(math.floor((x_low - origin_x) / cell_size - 0.5)) - 1
    ix_high = int(math.ceil((x_high - origin_x) / cell_size - 0.5)) + 1
    iy_low = int(math.floor((y_low - origin_y) / cell_size - 0.5)) - 1
    iy_high = int(math.ceil((y_high - origin_y) / cell_size - 0.5)) + 1
    ix = np.arange(ix_low, ix_high + 1, dtype=np.int64)
    iy = np.arange(iy_low, iy_high + 1, dtype=np.int64)
    x_centers = origin_x + (ix.astype(np.float64) + 0.5) * cell_size
    y_centers = origin_y + (iy.astype(np.float64) + 0.5) * cell_size
    world_x, world_y = np.meshgrid(x_centers, y_centers, indexing="ij")
    inside = (
        (ix[:, None] >= 0)
        & (ix[:, None] < physical_visibility_grid.shape[0])
        & (iy[None, :] >= 0)
        & (iy[None, :] < physical_visibility_grid.shape[1])
    )
    physical_free = np.zeros(inside.shape, dtype=bool)
    valid_rows, valid_cols = np.nonzero(inside)
    physical_free[valid_rows, valid_cols] = physical_visibility_grid.free_mask[
        ix[valid_rows], iy[valid_cols]
    ]
    physical_labels = np.full(inside.shape, UNKNOWN_CLASS, dtype=np.uint8)

    free_rows, free_cols = np.nonzero(inside & physical_free)
    if free_rows.size:
        free_center_x = world_x[free_rows, free_cols]
        free_center_y = world_y[free_rows, free_cols]
        sample_offsets = np.asarray(
            (
                (0.0, 0.0),
                (-half_cell, -half_cell),
                (-half_cell, half_cell),
                (half_cell, -half_cell),
                (half_cell, half_cell),
            ),
            dtype=np.float64,
        )
        floor_points = np.column_stack(
            (
                (free_center_x[:, None] + sample_offsets[None, :, 0]).ravel(),
                (free_center_y[:, None] + sample_offsets[None, :, 1]).ravel(),
                np.full(free_rows.size * sample_offsets.shape[0],
                        float(camera.ground_plane_z_m)),
            )
        )
        in_frustum, distances, nearest, _ = _first_box_hits(
            floor_points, camera, rendered_obstacle_boxes
        )
        directly_visible_floor = (
            in_frustum & (nearest >= distances - 1e-9)
        ).reshape(free_rows.size, sample_offsets.shape[0]).all(axis=1)
        physical_labels[
            free_rows[directly_visible_floor], free_cols[directly_visible_floor]
        ] = FREE_CLASS

    visible_obstacle_xy = _visible_obstacle_camera_ray_witnesses_xy(
        camera,
        rendered_obstacle_boxes,
    )
    labels = observable_physical_labels_from_raster(
        physical_labels,
        physical_x_centers_m=x_centers,
        physical_y_centers_m=y_centers,
        output_world_x_m=config_x,
        output_world_y_m=config_y,
        output_yaw_rad=base_yaw,
        physical_cell_size_m=cell_size,
        output_cell_size_m=local_grid.cell_size_m,
        visible_obstacle_first_hit_xy_m=visible_obstacle_xy,
    )
    # Exact collision geometry is a fail-closed veto only. Render-omitted or
    # sub-raster objects can downgrade a proposed physical FREE cell to
    # UNKNOWN, but privileged geometry never creates a supervised known class.
    collision_overlap = _output_cells_intersect_collision_geometry(
        config_x,
        config_y,
        output_yaw_rad=base_yaw,
        output_cell_size_m=local_grid.cell_size_m,
        obstacle_boxes=collision_obstacle_boxes,
    )
    labels[(labels == FREE_CLASS) & collision_overlap] = UNKNOWN_CLASS
    supervision = np.isfinite(config_x) & np.isfinite(config_y)
    return labels, supervision, labels != UNKNOWN_CLASS


def label_camera_visible_configuration_grid(
    configuration_grid: InflatedOccupancyGrid,
    *,
    physical_visibility_grid: InflatedOccupancyGrid,
    base_xy_yaw: Sequence[float],
    camera: CameraObservation,
    local_grid: LocalGridGeometry = DEFAULT_LOCAL_GRID,
    ray_step_m: float | None = None,
    label_contract: str = LABEL_CONTRACT_CENTER_VISIBLE_V2,
    obstacle_boxes: Sequence[BoxObject] | None = None,
    collision_obstacle_boxes: Sequence[BoxObject] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label camera-observable navigation occupancy under a named contract.

    The v2 compatibility contract classifies a visible candidate center in the
    pre-inflated oracle grid. The v3 contract instead supervises observable
    physical FREE/OCCUPIED/UNKNOWN at the 0.10 m output resolution. The fixed
    0.47 m robot morphology is deliberately deferred until after multi-view
    memory fusion and is not a per-frame training target.
    """

    if configuration_grid.inflation_m <= 0.0:
        raise DatasetContractError(
            "configuration_grid must use positive body inflation"
        )
    if label_contract not in SUPPORTED_LABEL_CONTRACTS:
        raise DatasetContractError(f"unsupported label contract: {label_contract!r}")
    if (
        label_contract == LABEL_CONTRACT_OBSERVABLE_PHYSICAL_V3
        and not math.isclose(
            configuration_grid.inflation_m,
            OBSERVABLE_FOOTPRINT_RADIUS_M,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise DatasetContractError(
            "observable physical v3 requires the canonical 0.47 m "
            "configuration-space radius"
        )

    if label_contract == LABEL_CONTRACT_OBSERVABLE_PHYSICAL_V3:
        if obstacle_boxes is None:
            raise DatasetContractError(
                "observable physical labels require exact rendered scene boxes"
            )
        return _observable_physical_raster_and_output_labels(
            physical_visibility_grid,
            rendered_obstacle_boxes=obstacle_boxes,
            collision_obstacle_boxes=(
                obstacle_boxes
                if collision_obstacle_boxes is None
                else collision_obstacle_boxes
            ),
            base_xy_yaw=base_xy_yaw,
            camera=camera,
            local_grid=local_grid,
        )

    physical, supervision, _, world_x, world_y = (
        _raycast_camera_visible_physical_grid(
            physical_visibility_grid,
            base_xy_yaw=base_xy_yaw,
            camera=camera,
            local_grid=local_grid,
            ray_step_m=ray_step_m,
        )
    )
    target_free = _world_points_to_grid_free(configuration_grid, world_x, world_y)
    labels = physical.copy()
    visible_physical_free = physical == FREE_CLASS
    labels[visible_physical_free & ~target_free] = OCCUPIED_CLASS
    observed = labels != UNKNOWN_CLASS
    return labels, supervision, observed


# Canonical name for new callers. The historical name remains for v2 API
# compatibility, where the output genuinely is per-frame configuration space.
label_camera_visible_navigation_grid = label_camera_visible_configuration_grid


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
    vertical_fov_deg: float | None = None,
    ground_plane_z_m: float = OBSERVABLE_GROUND_PLANE_Z_M,
    require_recorded_up: bool = False,
    image_width_px: int = 224,
    image_height_px: int = 168,
    obstacle_ray_stride_px: int = 2,
) -> CameraObservation:
    try:
        camera = frame["camera_pose_world"]
        position = tuple(float(value) for value in camera["position"])
        lookat = tuple(float(value) for value in camera["lookat"])
        if require_recorded_up and "up" not in camera:
            raise KeyError("up")
        up = tuple(float(value) for value in camera.get("up", (0.0, 0.0, 1.0)))
    except (KeyError, TypeError, ValueError) as exc:
        raise DatasetContractError("frame is missing a valid recorded camera pose") from exc
    if len(position) != 3 or len(lookat) != 3 or len(up) != 3:
        raise DatasetContractError("camera position/lookat/up must contain three values")
    return CameraObservation(
        position_xyz_m=position,
        lookat_xyz_m=lookat,
        horizontal_fov_deg=float(horizontal_fov_deg),
        near_m=float(near_m),
        vertical_fov_deg=(
            None if vertical_fov_deg is None else float(vertical_fov_deg)
        ),
        up_xyz=up,
        ground_plane_z_m=float(ground_plane_z_m),
        image_width_px=int(image_width_px),
        image_height_px=int(image_height_px),
        obstacle_ray_stride_px=int(obstacle_ray_stride_px),
    )


def _physical_obstacle_boxes(
    manifest: SceneManifest,
    *,
    treat_landmarks_as_obstacles: bool,
    treat_distractors_as_obstacles: bool,
) -> tuple[BoxObject, ...]:
    boxes: list[BoxObject] = [*manifest.walls, *manifest.obstacles]
    if treat_landmarks_as_obstacles:
        boxes.extend(manifest.landmarks)
    if treat_distractors_as_obstacles and manifest.visual_randomization is not None:
        boxes.extend(manifest.visual_randomization.distractor_objects)
    return tuple(boxes)


def _render_object_records(manifest: SceneManifest) -> list[dict[str, Any]]:
    groups: tuple[tuple[str, Sequence[BoxObject]], ...] = (
        ("wall", manifest.walls),
        ("obstacle", manifest.obstacles),
        ("landmark", manifest.landmarks),
        (
            "distractor",
            ()
            if manifest.visual_randomization is None
            else manifest.visual_randomization.distractor_objects,
        ),
    )
    records: list[dict[str, Any]] = []
    for group, boxes in groups:
        for box in boxes:
            records.append(
                {
                    "group": group,
                    "object_id": str(box.object_id),
                    "kind": str(box.kind),
                    "center_xyz_m": [float(value) for value in box.center_xyz_m],
                    "size_xyz_m": [float(value) for value in box.size_xyz_m],
                    "rpy_rad": [
                        float(box.roll_rad),
                        float(box.pitch_rad),
                        float(box.yaw_rad),
                    ],
                    "material_id": str(box.material_id),
                }
            )
    return sorted(records, key=lambda item: (item["group"], item["object_id"]))


def _validated_v04_render(
    source: SceneRenderSource,
    *,
    manifest: SceneManifest,
    plan: Mapping[str, Any],
    frames_path: Path,
    frames: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[tuple[int, int], dict[str, Any]], float]:
    summary_path = source.render_summary_path
    if summary_path is None:
        raise DatasetContractError(
            "observable physical v3 requires render_summary_path from the "
            "corrected source index"
        )
    summary_path = summary_path.resolve()
    summary = _read_json(summary_path)
    if str(summary.get("schema")) != "lewm_rendered_vision_v04":
        raise DatasetContractError(
            "observable physical v3 requires lewm_rendered_vision_v04 RGB"
        )
    if str(summary.get("render_status")) != "complete":
        raise DatasetContractError("v04 render summary is not complete")
    if str(summary.get("scene_id")) != manifest.scene_id:
        raise DatasetContractError("v04 render summary scene identity mismatch")

    resolution = summary.get("resolution_wh")
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or any(int(value) <= 0 for value in resolution)
    ):
        raise DatasetContractError("v04 render summary has invalid resolution_wh")
    width, height = map(int, resolution)
    if not math.isclose(
        float(height) / float(width),
        OBSERVABLE_CAMERA_NATIVE_ASPECT_HEIGHT_OVER_WIDTH,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise DatasetContractError("observable physical v3 requires 4:3 v04 RGB")
    camera = summary.get("camera_projection")
    if not isinstance(camera, Mapping):
        raise DatasetContractError("v04 render summary lacks camera_projection")
    expected_horizontal = float(plan["camera"]["fov_deg"])
    expected_near = float(plan["camera"]["near_m"])
    expected_vertical = vertical_fov_from_horizontal(
        expected_horizontal,
        image_width=width,
        image_height=height,
    )
    projection_checks = (
        str(camera.get("model")) == "pinhole",
        str(camera.get("renderer_fov_axis")) == "vertical",
        bool(camera.get("runtime_rectification_required", True)) is False,
        math.isclose(
            float(camera.get("horizontal_fov_deg", math.nan)),
            expected_horizontal,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        math.isclose(
            float(camera.get("vertical_fov_deg", math.nan)),
            expected_vertical,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        math.isclose(
            float(camera.get("near_m", math.nan)),
            expected_near,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
    )
    if not all(projection_checks):
        raise DatasetContractError("v04 camera projection contract mismatch")

    expected_records = _render_object_records(manifest)
    expected_ids = sorted(record["object_id"] for record in expected_records)
    if len(expected_ids) != len(set(expected_ids)):
        raise DatasetContractError("manifest collision object IDs must be unique")
    parity = summary.get("object_parity")
    if not isinstance(parity, Mapping):
        raise DatasetContractError("v04 render summary lacks object_parity")
    if (
        str(parity.get("schema")) != "lewm_render_object_parity_v1"
        or parity.get("rendered_groups")
        != ["wall", "obstacle", "landmark", "distractor"]
        or bool(parity.get("collision_distractors_rendered")) is not True
        or bool(parity.get("full_box_roll_pitch_yaw_rendered")) is not True
        or int(parity.get("rendered_object_count", -1)) != len(expected_records)
        or parity.get("rendered_object_ids") != expected_ids
        or str(parity.get("rendered_object_ids_sha256"))
        != canonical_json_sha256(expected_ids)
        or str(parity.get("rendered_object_records_sha256"))
        != canonical_json_sha256(expected_records)
    ):
        raise DatasetContractError("v04 rendered/collision object parity mismatch")

    summary_source = summary.get("source")
    if not isinstance(summary_source, Mapping):
        raise DatasetContractError("v04 render summary lacks source provenance")
    expected_sources = {
        "plan": (source.render_plan_path.resolve(), sha256_file(source.render_plan_path)),
        "frames_jsonl": (frames_path.resolve(), sha256_file(frames_path)),
        "scene_manifest": (
            source.scene_manifest_path.resolve(),
            sha256_file(source.scene_manifest_path),
        ),
    }
    for name, (expected_path, expected_sha) in expected_sources.items():
        item = summary_source.get(name)
        if (
            not isinstance(item, Mapping)
            or Path(str(item.get("path", ""))).resolve() != expected_path
            or str(item.get("sha256", "")) != expected_sha
        ):
            raise DatasetContractError(f"v04 {name} provenance mismatch")

    selection = summary.get("frame_selection")
    if not isinstance(selection, Mapping):
        raise DatasetContractError("v04 render summary lacks frame_selection")
    selection_path = Path(str(selection.get("path", ""))).resolve()
    if (
        not selection_path.is_file()
        or sha256_file(selection_path) != str(selection.get("sha256", ""))
    ):
        raise DatasetContractError("v04 frame-selection provenance mismatch")
    selection_payload = _read_json(selection_path)
    if (
        str(selection_payload.get("schema"))
        != "lewm_go2_selected_render_frames_v1"
        or str(selection_payload.get("scene_id")) != manifest.scene_id
        or str(selection_payload.get("frame_key_set_sha256", ""))
        != str(selection.get("frame_key_set_sha256", ""))
    ):
        raise DatasetContractError("v04 frame-selection commitment mismatch")

    planned_by_key = {_frame_key(frame): frame for frame in frames}
    rendered_rows = summary.get("rendered_frames")
    if not isinstance(rendered_rows, list) or not rendered_rows:
        raise DatasetContractError("v04 render summary has no rendered frames")
    rendered_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    normalized_rows: list[dict[str, Any]] = []
    for raw in rendered_rows:
        if not isinstance(raw, Mapping):
            raise DatasetContractError("v04 rendered-frame record must be an object")
        key = _frame_key(raw)
        planned = planned_by_key.get(key)
        if planned is None or key in rendered_by_key:
            raise DatasetContractError("v04 rendered-frame key mismatch or duplicate")
        if int(raw.get("timestamp_ns", -1)) != int(planned["timestamp_ns"]):
            raise DatasetContractError("v04 rendered-frame timestamp mismatch")
        image_path = (
            summary_path.parent
            / "rgb"
            / f"frame_{key[0]:06d}_env_{key[1]:02d}.png"
        ).resolve()
        image_sha = str(raw.get("image_sha256", ""))
        if not image_path.is_file() or sha256_file(image_path) != image_sha:
            raise DatasetContractError("v04 rendered-image hash mismatch")
        normalized = {
            "frame_index": key[0],
            "env_index": key[1],
            "timestamp_ns": int(raw["timestamp_ns"]),
            "image_sha256": image_sha,
        }
        normalized_rows.append(normalized)
        rendered_by_key[key] = {
            **normalized,
            "camera_valid": True,
            "rgb_path": str(image_path),
        }
    normalized_rows.sort(key=lambda item: (item["frame_index"], item["env_index"]))
    if (
        len(normalized_rows) != int(summary.get("frame_count", -1))
        or canonical_json_sha256(normalized_rows)
        != str(summary.get("rendered_image_set_sha256", ""))
    ):
        raise DatasetContractError("v04 rendered-frame set commitment mismatch")
    selected_keys = selection_payload.get("frame_keys")
    if selected_keys != [list(key) for key in sorted(rendered_by_key)]:
        raise DatasetContractError("v04 selected and rendered frame keys differ")
    return summary, rendered_by_key, expected_vertical


def _validated_render_audit_contract(
    audit_path: Path,
    *,
    source_index_path: Path,
    sources: Sequence[SceneRenderSource],
) -> dict[str, Any]:
    audit_path = audit_path.resolve()
    source_index_path = source_index_path.resolve()
    audit = _read_json(audit_path)
    core = dict(audit)
    declared_content_sha = str(core.pop("content_sha256", ""))
    if (
        str(audit.get("schema")) != "lewm_go2_selected_render_audit_v1"
        or declared_content_sha != canonical_json_sha256(core)
    ):
        raise DatasetContractError("invalid corrected-render audit content contract")
    index_record = audit.get("output_source_index")
    if (
        not isinstance(index_record, Mapping)
        or Path(str(index_record.get("path", ""))).resolve() != source_index_path
        or str(index_record.get("sha256", "")) != sha256_file(source_index_path)
    ):
        raise DatasetContractError(
            "render audit output source-index path/SHA does not match builder input"
        )
    camera = audit.get("camera_projection")
    expected_vertical = vertical_fov_from_horizontal(
        78.323, image_width=224, image_height=168
    )
    if (
        not isinstance(camera, Mapping)
        or camera.get("resolution_wh") != [224, 168]
        or not math.isclose(
            float(camera.get("horizontal_fov_deg", math.nan)),
            78.323,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            float(camera.get("vertical_fov_deg", math.nan)),
            expected_vertical,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            float(camera.get("near_m", math.nan)),
            0.05,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or bool(camera.get("runtime_rectification_required", True)) is not False
    ):
        raise DatasetContractError("render audit camera projection contract mismatch")
    object_contract = audit.get("object_contract")
    if (
        not isinstance(object_contract, Mapping)
        or object_contract.get("rendered_groups")
        != ["wall", "obstacle", "landmark", "distractor"]
        or bool(object_contract.get("collision_distractors_rendered")) is not True
        or bool(object_contract.get("full_box_roll_pitch_yaw_rendered")) is not True
    ):
        raise DatasetContractError("render audit object contract mismatch")
    if (
        bool(audit.get("g2_row_metadata_read")) is not True
        or bool(audit.get("g2_image_bytes_hashed_for_integrity")) is not True
        or bool(audit.get("g2_images_decoded_or_inspected")) is not False
        or bool(audit.get("g2_image_content_metrics_computed")) is not False
        or bool(audit.get("g2_label_shards_opened")) is not False
        or bool(audit.get("g2_model_outputs_opened")) is not False
    ):
        raise DatasetContractError("render audit G2-contact contract mismatch")
    if int(audit.get("scene_count", -1)) != len(sources):
        raise DatasetContractError("render audit scene count differs from builder input")

    indexed_sources = load_source_index(source_index_path)

    def source_identity(item: SceneRenderSource) -> tuple[Any, ...]:
        def resolved(value: Path | None) -> str:
            return "" if value is None else str(value.resolve())

        return (
            item.scene_id,
            item.family,
            resolved(item.scene_manifest_path),
            resolved(item.render_plan_path),
            resolved(item.rgb_dir),
            resolved(item.frames_jsonl_path),
            resolved(item.rendered_frames_jsonl_path),
            resolved(item.render_summary_path),
        )

    if sorted(map(source_identity, indexed_sources)) != sorted(
        map(source_identity, sources)
    ):
        raise DatasetContractError(
            "builder sources are not exactly the render-audited source index"
        )
    audits = audit.get("scene_audits")
    if not isinstance(audits, list) or len(audits) != len(sources):
        raise DatasetContractError("render audit scene commitments are incomplete")
    audit_by_scene_hash = {
        str(item.get("scene_id_sha256", "")): item
        for item in audits
        if isinstance(item, Mapping)
    }
    if len(audit_by_scene_hash) != len(sources):
        raise DatasetContractError("render audit scene commitments are not unique")
    for source in sources:
        item = audit_by_scene_hash.get(scene_id_sha256(source.scene_id))
        if (
            item is None
            or source.render_summary_path is None
            or str(item.get("summary_sha256", ""))
            != sha256_file(source.render_summary_path)
        ):
            raise DatasetContractError(
                "render audit per-scene summary commitment mismatch"
            )
    return {
        "schema": str(audit["schema"]),
        "path": str(audit_path),
        "file_sha256": sha256_file(audit_path),
        "content_sha256": declared_content_sha,
        "output_source_index": {
            "path": str(source_index_path),
            "sha256": sha256_file(source_index_path),
        },
        "camera_projection": dict(camera),
        "object_contract": dict(object_contract),
        "scene_count": int(audit["scene_count"]),
        "frame_count": int(audit["frame_count"]),
        "g2_contact": {
            "row_metadata_read": True,
            "image_bytes_hashed_for_integrity": True,
            "images_decoded_or_inspected": False,
            "image_content_metrics_computed": False,
            "label_shards_opened": False,
            "model_outputs_opened": False,
        },
    }


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


@dataclass(frozen=True)
class _SceneBuildRequest:
    source: SceneRenderSource
    geometry_contract: GeometryContract
    local_grid: LocalGridGeometry
    dataset_split: str
    dataset_role: str | None
    max_transitions_per_scene: int
    selection_seed: str
    label_contract: str
    expected_v04_resolution: tuple[int, int] | None
    allow_role_transition_shortfall: bool


@dataclass
class _SceneBuildResult:
    scene_id: str
    rows: list[dict[str, Any]]
    arrays: dict[str, np.ndarray] | None
    source_provenance: dict[str, Any]
    stats: dict[str, int]
    image_commitments: list[tuple[str, str]]
    observable_native_resolution: tuple[int, int] | None
    observable_vertical_fov: float | None


def _build_paired_navigation_scene(
    request: _SceneBuildRequest,
) -> _SceneBuildResult:
    """Build one scene without assigning global rows or writing final artifacts."""

    source = request.source
    geometry_contract = request.geometry_contract
    local_grid = request.local_grid
    label_contract = request.label_contract
    manifest, plan, frames_path, frames, rendered_by_key, hashes = _source_scene(
        source,
        geometry_contract=geometry_contract,
    )
    camera_plan = plan["camera"]
    fov_deg = float(camera_plan["fov_deg"])
    near_m = float(camera_plan["near_m"])
    vertical_fov_deg: float | None = None
    image_width_px = 224
    image_height_px = 168
    v04_summary: dict[str, Any] | None = None
    observable_native_resolution: tuple[int, int] | None = None
    if label_contract == LABEL_CONTRACT_OBSERVABLE_PHYSICAL_V3:
        v04_summary, rendered_by_key, vertical_fov_deg = _validated_v04_render(
            source,
            manifest=manifest,
            plan=plan,
            frames_path=frames_path,
            frames=frames,
        )
        native_width, native_height = map(int, v04_summary["resolution_wh"])
        observable_native_resolution = (native_width, native_height)
        if observable_native_resolution != request.expected_v04_resolution:
            raise DatasetContractError(
                "per-scene v04 resolution differs from render audit"
            )
        image_width_px = native_width
        image_height_px = native_height
        hashes["render_summary_file_sha256"] = sha256_file(
            source.render_summary_path  # type: ignore[arg-type]
        )
    rendered_parent = (
        source.render_summary_path.resolve().parent
        if v04_summary is not None
        else (
            source.rendered_frames_jsonl_path.resolve().parent
            if source.rendered_frames_jsonl_path is not None
            else None
        )
    )
    rgb_dir = (
        None
        if v04_summary is not None
        else (source.rgb_dir.resolve() if source.rgb_dir is not None else None)
    )
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
    raw_transitions = list(iter_primitive_transitions(frames, stats=transition_stats))
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
        if (
            source.rendered_frames_jsonl_path is not None or v04_summary is not None
        ) and (rendered_current is None or rendered_next is None):
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
        max_transitions=request.max_transitions_per_scene,
        seed=request.selection_seed,
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
    scene_stats: Counter[str] = Counter(transition_stats)
    scene_stats["raw_transitions"] += len(raw_transitions)
    scene_stats["candidate_transitions"] += len(candidate_transitions)
    scene_stats["selected_transitions_before_labeling"] += len(transitions)

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
    physical_obstacle_boxes = _physical_obstacle_boxes(
        manifest,
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
    image_hash_cache: dict[Path, str] = {}
    image_commitments: list[tuple[str, str]] = []

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
                label_cache[key] = label_camera_visible_navigation_grid(
                    configuration_occupancy,
                    physical_visibility_grid=physical_visibility_occupancy,
                    base_xy_yaw=_base_xy_yaw(frame),
                    camera=_camera_observation(
                        frame,
                        horizontal_fov_deg=fov_deg,
                        near_m=near_m,
                        vertical_fov_deg=vertical_fov_deg,
                        require_recorded_up=(
                            label_contract
                            == LABEL_CONTRACT_OBSERVABLE_PHYSICAL_V3
                        ),
                        image_width_px=image_width_px,
                        image_height_px=image_height_px,
                        obstacle_ray_stride_px=2,
                    ),
                    local_grid=local_grid,
                    label_contract=label_contract,
                    obstacle_boxes=physical_obstacle_boxes,
                    collision_obstacle_boxes=physical_obstacle_boxes,
                )
        current_label, current_supervision, current_observed = label_cache[current_key]
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
                "schema": _ROW_SCHEMA_BY_LABEL_CONTRACT[label_contract],
                "scene_id": source.scene_id,
                "family": manifest.family,
                "dataset_split": request.dataset_split,
                **(
                    {"dataset_role": request.dataset_role}
                    if request.dataset_role is not None
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
                **(
                    {"label_contract": label_contract}
                    if label_contract == LABEL_CONTRACT_OBSERVABLE_PHYSICAL_V3
                    else {}
                ),
            }
        )

    if (
        request.dataset_role is not None
        and not request.allow_role_transition_shortfall
        and len(scene_rows) < request.max_transitions_per_scene
    ):
        raise DatasetContractError(
            f"direct-role scene {source.scene_id!r} produced {len(scene_rows)} rows, "
            f"fewer than the requested {request.max_transitions_per_scene}; pass "
            "allow_role_transition_shortfall=True when the preregistered contract "
            "caps rows per scene ('at most N') so the shortfall is an explicit, "
            "recorded decision"
        )

    arrays: dict[str, np.ndarray] | None = None
    if not scene_rows:
        scene_stats["scenes_without_rows"] += 1
    else:
        arrays = {
            "current_labels": np.stack(labels_current).astype(np.uint8, copy=False),
            "next_labels": np.stack(labels_next).astype(np.uint8, copy=False),
            "current_supervision_mask": np.stack(supervision_current).astype(
                bool, copy=False
            ),
            "next_supervision_mask": np.stack(supervision_next).astype(
                bool, copy=False
            ),
            "current_observed_mask": np.stack(observed_current).astype(
                bool, copy=False
            ),
            "next_observed_mask": np.stack(observed_next).astype(bool, copy=False),
            "relative_se2_current_frame": np.stack(odometry).astype(
                np.float32, copy=False
            ),
            "primitive": _np_unicode([row["primitive"] for row in scene_rows]),
            "current_image_path": _np_unicode(
                [row["current_image_path"] for row in scene_rows]
            ),
            "next_image_path": _np_unicode(
                [row["next_image_path"] for row in scene_rows]
            ),
            "current_image_sha256": _np_unicode(
                [row["current_image_sha256"] for row in scene_rows]
            ),
            "next_image_sha256": _np_unicode(
                [row["next_image_sha256"] for row in scene_rows]
            ),
        }
        scene_stats["rows_written"] += len(scene_rows)

    source_paths = {
        "scene_manifest": str(source.scene_manifest_path.resolve()),
        "render_plan": str(source.render_plan_path.resolve()),
        "frames_jsonl": str(frames_path),
    }
    if source.rendered_frames_jsonl_path is not None:
        source_paths["rendered_frames_jsonl"] = str(
            source.rendered_frames_jsonl_path.resolve()
        )
    if v04_summary is not None:
        source_paths["render_summary"] = str(
            source.render_summary_path.resolve()  # type: ignore[union-attr]
        )
    source_provenance = {
        "scene_id": source.scene_id,
        "family": manifest.family,
        "dataset_split": request.dataset_split,
        **(
            {"dataset_role": request.dataset_role}
            if request.dataset_role is not None
            else {}
        ),
        "paths": source_paths,
        "hashes": hashes,
        "rows": len(scene_rows),
        "transition_stats": dict(sorted(transition_stats.items())),
        "selection": selection_metadata,
    }
    return _SceneBuildResult(
        scene_id=source.scene_id,
        rows=scene_rows,
        arrays=arrays,
        source_provenance=source_provenance,
        stats=dict(scene_stats),
        image_commitments=image_commitments,
        observable_native_resolution=observable_native_resolution,
        observable_vertical_fov=(
            None if vertical_fov_deg is None else round(vertical_fov_deg, 12)
        ),
    )


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
    label_contract: str = LABEL_CONTRACT_CENTER_VISIBLE_V2,
    source_index_path: Path | None = None,
    render_audit_contract_path: Path | None = None,
    build_provenance: Mapping[str, Any] | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Build scene-sharded G2 rows from RGB frame plans and exact geometry."""

    if not sources:
        raise ValueError("at least one scene source is required")
    workers = int(workers)
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if label_contract not in SUPPORTED_LABEL_CONTRACTS:
        raise DatasetContractError(f"unsupported label contract: {label_contract!r}")
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
    if (
        label_contract == LABEL_CONTRACT_OBSERVABLE_PHYSICAL_V3
        and not math.isclose(
            geometry_contract.configuration_space.body_inflation_radius_m,
            OBSERVABLE_FOOTPRINT_RADIUS_M,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise DatasetContractError(
            "observable physical v3 requires the canonical 0.47 m "
            "geometry contract"
        )
    if label_contract == LABEL_CONTRACT_OBSERVABLE_PHYSICAL_V3 and (
        not geometry_contract.configuration_space.landmarks_are_obstacles
        or not geometry_contract.configuration_space.distractors_are_obstacles
    ):
        raise DatasetContractError(
            "observable physical v3 requires landmarks and distractors in the "
            "collision/render parity set"
        )
    dataset_schema = _DATASET_SCHEMA_BY_LABEL_CONTRACT[label_contract]
    row_schema = _ROW_SCHEMA_BY_LABEL_CONTRACT[label_contract]
    render_audit_contract: dict[str, Any] | None = None
    validated_build_provenance: dict[str, Any] | None = None
    if label_contract == LABEL_CONTRACT_OBSERVABLE_PHYSICAL_V3:
        if source_index_path is None or render_audit_contract_path is None:
            raise DatasetContractError(
                "observable physical v3 requires source_index_path and "
                "render_audit_contract_path"
            )
        render_audit_contract = _validated_render_audit_contract(
            render_audit_contract_path,
            source_index_path=source_index_path,
            sources=sources,
        )
        validated_build_provenance = _validated_build_provenance(build_provenance)
        provenance_config = validated_build_provenance["config"]
        expected_config_paths = {
            "source_index": str(source_index_path.resolve()),
            "render_audit_contract": str(render_audit_contract_path.resolve()),
            "geometry_contract": str(geometry_contract.source_path.resolve()),
            "output_dir": str(output_dir.resolve()),
        }
        if str(provenance_config.get("label_contract", "")) != label_contract:
            raise DatasetContractError(
                "dataset build provenance label contract mismatch"
            )
        if int(provenance_config.get("workers", -1)) != workers:
            raise DatasetContractError("dataset build provenance workers mismatch")
        for key, expected_path in expected_config_paths.items():
            if str(provenance_config.get(key, "")) != expected_path:
                raise DatasetContractError(
                    f"dataset build provenance {key} mismatch"
                )
        provenance_inputs = validated_build_provenance["inputs"]
        audited_source_index = render_audit_contract["output_source_index"]
        expected_input_records = {
            "source_index": (
                audited_source_index["path"],
                audited_source_index["sha256"],
            ),
            "render_audit_contract": (
                render_audit_contract["path"],
                render_audit_contract["file_sha256"],
            ),
        }
        for name, (expected_path, expected_sha256) in expected_input_records.items():
            artifact = provenance_inputs[name]
            if (
                str(artifact.get("path", "")) != str(expected_path)
                or str(artifact.get("sha256", "")) != str(expected_sha256)
            ):
                raise DatasetContractError(
                    f"dataset build provenance {name} input mismatch"
                )
        provenance_geometry = validated_build_provenance["dependencies"][
            "geometry_contract"
        ]
        if (
            str(provenance_geometry.get("path", ""))
            != str(geometry_contract.source_path.resolve())
            or str(provenance_geometry.get("sha256", ""))
            != sha256_file(geometry_contract.source_path)
        ):
            raise DatasetContractError(
                "dataset build provenance geometry dependency mismatch"
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
    output_dir_preexisted = output_dir.exists()
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / "scenes"
    shard_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "rows.jsonl"

    total_stats: Counter[str] = Counter()
    source_provenance: list[dict[str, Any]] = []
    shard_records: list[dict[str, Any]] = []
    image_commitments: list[tuple[str, str]] = []
    observable_native_resolutions: set[tuple[int, int]] = set()
    observable_vertical_fovs: set[float] = set()
    global_row = 0

    expected_v04_resolution = (
        tuple(
            map(
                int,
                render_audit_contract["camera_projection"]["resolution_wh"],
            )
        )
        if render_audit_contract is not None
        else None
    )
    requests = [
        _SceneBuildRequest(
            source=source,
            geometry_contract=geometry_contract,
            local_grid=local_grid,
            dataset_split=splits[source.scene_id],
            dataset_role=(None if roles is None else roles[source.scene_id]),
            max_transitions_per_scene=int(max_transitions_per_scene),
            selection_seed=selection_seed,
            label_contract=label_contract,
            expected_v04_resolution=expected_v04_resolution,
            allow_role_transition_shortfall=allow_role_transition_shortfall,
        )
        for source in sorted(sources, key=lambda item: item.scene_id)
    ]

    def commit_scene_result(
        result: _SceneBuildResult,
        *,
        expected_scene_id: str,
        index_stream: Any,
    ) -> None:
        nonlocal global_row
        if result.scene_id != expected_scene_id:
            raise DatasetContractError(
                "scene worker result order/identity mismatch: "
                f"expected {expected_scene_id!r}, got {result.scene_id!r}"
            )
        if int(result.source_provenance.get("rows", -1)) != len(result.rows):
            raise DatasetContractError(
                "scene worker row count disagrees with source provenance"
            )
        total_stats.update(result.stats)
        image_commitments.extend(result.image_commitments)
        if result.observable_native_resolution is not None:
            observable_native_resolutions.add(result.observable_native_resolution)
        if result.observable_vertical_fov is not None:
            observable_vertical_fovs.add(result.observable_vertical_fov)

        scene_rows = result.rows
        if scene_rows:
            if result.arrays is None:
                raise DatasetContractError(
                    "scene worker returned rows without label arrays"
                )
            shard_name = f"scene_{scene_id_sha256(result.scene_id)[:16]}.npz"
            shard_path = shard_dir / shard_name
            np.savez_compressed(shard_path, **result.arrays)
            shard_hash = sha256_file(shard_path)
            shard_records.append(
                {
                    "scene_id": result.scene_id,
                    "path": str(shard_path),
                    "sha256": shard_hash,
                    "rows": len(scene_rows),
                }
            )
            for shard_row, row in enumerate(scene_rows):
                if str(row.get("schema", "")) != row_schema:
                    raise DatasetContractError(
                        "scene worker returned an unexpected row schema"
                    )
                row["global_row"] = global_row
                row["label_shard_path"] = str(shard_path)
                row["label_shard_sha256"] = shard_hash
                row["label_shard_row"] = shard_row
                index_stream.write(json.dumps(row, sort_keys=True) + "\n")
                global_row += 1
        elif result.arrays is not None:
            raise DatasetContractError(
                "scene worker returned label arrays without rows"
            )
        source_provenance.append(result.source_provenance)

    try:
        with index_path.open("w") as index_stream:
            if workers == 1:
                for request in requests:
                    commit_scene_result(
                        _build_paired_navigation_scene(request),
                        expected_scene_id=request.source.scene_id,
                        index_stream=index_stream,
                    )
            else:
                executor = ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=multiprocessing.get_context("spawn"),
                )
                futures: dict[int, Future[_SceneBuildResult]] = {}
                next_submit = 0
                next_consume = 0
                max_in_flight = min(len(requests), workers * 2)
                try:
                    while next_submit < max_in_flight:
                        futures[next_submit] = executor.submit(
                            _build_paired_navigation_scene,
                            requests[next_submit],
                        )
                        next_submit += 1
                    while next_consume < len(requests):
                        result = futures.pop(next_consume).result()
                        if next_submit < len(requests):
                            futures[next_submit] = executor.submit(
                                _build_paired_navigation_scene,
                                requests[next_submit],
                            )
                            next_submit += 1
                        commit_scene_result(
                            result,
                            expected_scene_id=requests[next_consume].source.scene_id,
                            index_stream=index_stream,
                        )
                        next_consume += 1
                except BaseException:
                    for future in futures.values():
                        future.cancel()
                    raise
                finally:
                    executor.shutdown(wait=True, cancel_futures=True)
    except BaseException:
        shutil.rmtree(output_dir, ignore_errors=True)
        if output_dir_preexisted:
            output_dir.mkdir(parents=True, exist_ok=True)
        raise

    if global_row == 0:
        raise DatasetContractError("no valid paired-navigation rows were produced")
    index_hash = sha256_file(index_path)
    split_counts = Counter(splits[record["scene_id"]] for record in source_provenance)
    label_semantics: dict[str, Any]
    if label_contract == LABEL_CONTRACT_OBSERVABLE_PHYSICAL_V3:
        if len(observable_vertical_fovs) != 1:
            raise DatasetContractError(
                "observable sources must share one rectified vertical camera FOV"
            )
        post_memory_morphology = post_memory_configuration_morphology_metadata(
            radius_m=geometry_contract.configuration_space.body_inflation_radius_m,
            physical_cell_size_m=(
                geometry_contract.configuration_space.online_cell_size_m
            ),
        )
        post_memory_morphology.update(
            {
                "radius_source": (
                    "geometry_contract.configuration_space."
                    "body_inflation_radius_m"
                ),
                "memory_cell_size_source": (
                    "geometry_contract.configuration_space.online_cell_size_m"
                ),
            }
        )
        vertical_fov = next(iter(observable_vertical_fovs))
        aggregation_contract = {
            "schema": "lewm_observable_physical_aggregation_v1",
            "source_cell_size_m": (
                geometry_contract.configuration_space.oracle_cell_size_m
            ),
            "output_cell_size_m": local_grid.cell_size_m,
            "free_rule": (
                "every world-aligned source cell square intersecting the "
                "base-yaw-aligned output square has direct visible-free "
                "ground evidence at its center and all four corners"
            ),
            "occupied_rule": (
                "a sampled camera-lattice ray's exact nearest 3D manifest-box "
                "hit xy lies inside the base-yaw-aligned output square"
            ),
            "known_class_precedence": "OCCUPIED_then_FREE_else_UNKNOWN",
            "collision_geometry_veto": (
                "a conservative full-RPY projected-box/output-square overlap "
                "downgrades proposed FREE to UNKNOWN and never creates a "
                "known class"
            ),
        }
        aggregation_contract["contract_sha256"] = canonical_json_sha256(
            aggregation_contract
        )
        label_semantics = {
            "label_contract": label_contract,
            "unknown": UNKNOWN_CLASS,
            "free": FREE_CLASS,
            "occupied": OCCUPIED_CLASS,
            "supervision_mask": "finite local-grid cells; includes UNKNOWN targets",
            "observed_mask": "supervision_mask and label != UNKNOWN",
            "unknown_is_supervised": True,
            "target_occupancy_space": "observable_physical_occupancy",
            "per_frame_configuration_classes_supervised": False,
            "classes": {
                "unknown": UNKNOWN_CLASS,
                "free": FREE_CLASS,
                "occupied": OCCUPIED_CLASS,
            },
            "visibility": {
                "model": "recorded_camera_full_rectilinear_frustum_first_3d_hit",
                "rendered_depth_consumed": False,
                "pose_source": "recorded camera_pose_world position/lookat/up",
                "horizontal_fov_deg": geometry_contract.camera.horizontal_fov_deg,
                "vertical_fov_deg": vertical_fov,
                "native_resolutions": [
                    list(resolution)
                    for resolution in sorted(observable_native_resolutions)
                ],
                "near_m": geometry_contract.camera.near_m,
                "frustum_predicate": (
                    "forward_cam > near and abs(right_cam/forward_cam) <= "
                    "tan(horizontal_fov/2) and abs(up_cam/forward_cam) <= "
                    "tan(vertical_fov/2)"
                ),
                "box_transform": "Rz(yaw) @ Ry(pitch) @ Rx(roll)",
                "free_surface": {
                    "kind": "ground_plane_center_and_four_cell_corners",
                    "z_m": OBSERVABLE_GROUND_PLANE_Z_M,
                    "must_have_no_earlier_3d_box_hit": True,
                },
                "occupied_surface": {
                    "kind": "exact_nearest_hit_on_full_RPY_manifest_box",
                    "sampling": "rectified_camera_pixel_lattice",
                    "camera_pixel_stride": 2,
                    "exhaustive_surface_visibility_claimed": False,
                    "missed_hits_are_conservatively_UNKNOWN": True,
                    "ground_point_visibility_required": False,
                },
            },
            "physical_aggregation": aggregation_contract,
            "renderer_contract": {
                "summary_schema": "lewm_rendered_vision_v04",
                "object_parity_schema": "lewm_render_object_parity_v1",
                "all_collision_objects_rendered": True,
                "full_box_roll_pitch_yaw_rendered": True,
                "sparse_frame_and_image_hashes_verified": True,
            },
            "post_memory_configuration_derivation": post_memory_morphology,
            "post_memory_configuration_derivation_is_evaluation_only": True,
            "configuration_inflation_radius_m": (
                geometry_contract.configuration_space.body_inflation_radius_m
            ),
            "privileged_input_at_runtime": False,
        }
    else:
        # Keep the v2 metadata stable: the explicit build option must not
        # silently rewrite the meaning or identity of existing v2 artifacts.
        label_semantics = {
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
        }
    manifest_payload: dict[str, Any] = {
        "schema": dataset_schema,
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
        "label_semantics": label_semantics,
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
    if render_audit_contract is not None:
        manifest_payload["render_audit_contract"] = render_audit_contract
    if validated_build_provenance is not None:
        manifest_payload["build_provenance"] = validated_build_provenance
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
    roles: Iterable[str] | None = None,
    scene_ids: Iterable[str] | None = None,
) -> dict[str, int]:
    """Re-hash global or explicitly scoped paired-navigation artifacts.

    Global verification remains the default.  A scoped verification accepts
    either direct dataset ``roles`` or explicit ``scene_ids``, never both.  It
    always verifies the shared geometry, exclusions, and row-index commitment,
    but opens source, label-shard, and RGB artifacts only for selected scenes.
    """

    payload = _read_json(dataset_manifest_path)
    dataset_schema = str(payload.get("schema", ""))
    supported_dataset_schemas = set(_DATASET_SCHEMA_BY_LABEL_CONTRACT.values())
    if dataset_schema not in supported_dataset_schemas:
        raise ProvenanceError("unsupported paired-navigation dataset schema")
    expected_row_schema = (
        "lewm_go2_paired_navigation_row_v3"
        if dataset_schema == "lewm_go2_paired_navigation_dataset_v3"
        else "lewm_go2_paired_navigation_row_v2"
    )
    if dataset_schema == "lewm_go2_paired_navigation_dataset_v3":
        semantics = payload.get("label_semantics")
        if (
            not isinstance(semantics, Mapping)
            or str(semantics.get("label_contract"))
            != LABEL_CONTRACT_OBSERVABLE_PHYSICAL_V3
            or str(semantics.get("target_occupancy_space"))
            != "observable_physical_occupancy"
            or bool(semantics.get("per_frame_configuration_classes_supervised", True))
            is not False
            or bool(
                semantics.get(
                    "post_memory_configuration_derivation_is_evaluation_only",
                    False,
                )
            )
            is not True
        ):
            raise ProvenanceError("invalid observable-physical v3 label semantics")
    if roles is not None and scene_ids is not None:
        raise ProvenanceError("provenance scope must use roles or scene_ids, not both")
    if isinstance(roles, (str, bytes)) or isinstance(scene_ids, (str, bytes)):
        raise ProvenanceError("provenance roles and scene_ids must be collections")

    sources = payload.get("sources", [])
    if not isinstance(sources, list) or not sources:
        raise ProvenanceError("dataset manifest must contain source provenance")
    source_by_scene: dict[str, Mapping[str, Any]] = {}
    for source in sources:
        if not isinstance(source, Mapping):
            raise ProvenanceError("source provenance entries must be objects")
        source_scene_id = str(source.get("scene_id", "")).strip()
        if not source_scene_id or source_scene_id in source_by_scene:
            raise ProvenanceError(
                f"source scene IDs must be nonempty and unique: {source_scene_id!r}"
            )
        source_by_scene[source_scene_id] = source

    selected_scene_ids: set[str] | None = None
    selected_roles: tuple[str, ...] | None = None
    if roles is not None:
        selected_roles = tuple(dict.fromkeys(str(role) for role in roles))
        if not selected_roles:
            raise ProvenanceError("provenance role scope must be nonempty")
        invalid_roles = sorted(set(selected_roles) - set(DATASET_ROLES))
        if invalid_roles:
            raise ProvenanceError(f"unknown provenance roles: {invalid_roles}")
        role_contract = payload.get("scene_roles")
        if not isinstance(role_contract, Mapping):
            raise ProvenanceError(
                "role-scoped provenance requires a direct scene-role contract"
            )
        if role_contract.get("schema") != "lewm_go2_family_scene_roles_v1":
            raise ProvenanceError("unsupported scene-role provenance schema")
        assignments = role_contract.get("assignments")
        if not isinstance(assignments, Mapping):
            raise ProvenanceError("scene-role contract lacks assignments")
        normalized_assignments = {
            str(scene_id): str(role) for scene_id, role in assignments.items()
        }
        if canonical_json_sha256(normalized_assignments) != str(
            role_contract.get("assignments_sha256", "")
        ):
            raise ProvenanceError("scene-role assignment commitment mismatch")
        if set(normalized_assignments) != set(source_by_scene):
            raise ProvenanceError("scene-role assignments do not match source scenes")
        invalid_assignments = sorted(
            set(normalized_assignments.values()) - set(DATASET_ROLES)
        )
        if invalid_assignments:
            raise ProvenanceError(
                f"scene-role assignments contain unknown roles: {invalid_assignments}"
            )
        empty_roles = sorted(
            set(selected_roles) - set(normalized_assignments.values())
        )
        if empty_roles:
            raise ProvenanceError(f"provenance roles select no scenes: {empty_roles}")
        expected_scene_counts = {
            role: sum(assigned == role for assigned in normalized_assignments.values())
            for role in DATASET_ROLES
        }
        if role_contract.get("scene_counts") != expected_scene_counts:
            raise ProvenanceError("scene-role count commitment mismatch")
        for source_scene_id, source in source_by_scene.items():
            if str(source.get("dataset_role", "")) != normalized_assignments[
                source_scene_id
            ]:
                raise ProvenanceError(
                    f"source role disagrees with assignment: {source_scene_id}"
                )
        selected_scene_ids = {
            scene_id
            for scene_id, role in normalized_assignments.items()
            if role in selected_roles
        }
    elif scene_ids is not None:
        selected_scene_ids = {str(scene_id) for scene_id in scene_ids}
        if not selected_scene_ids or "" in selected_scene_ids:
            raise ProvenanceError("provenance scene scope must be nonempty")
        missing_scenes = sorted(selected_scene_ids - set(source_by_scene))
        if missing_scenes:
            raise ProvenanceError(
                f"provenance scope contains unknown scenes: {missing_scenes}"
            )

    if selected_scene_ids is not None and not selected_scene_ids:
        raise ProvenanceError("provenance scope selects no scenes")
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
    if dataset_schema == "lewm_go2_paired_navigation_dataset_v3":
        audit_record = payload.get("render_audit_contract")
        if not isinstance(audit_record, Mapping):
            raise ProvenanceError("v3 dataset lacks render-audit contract")
        check(
            str(audit_record.get("path", "")),
            str(audit_record.get("file_sha256", "")),
            "render_audit_contract",
        )
        audited_index = audit_record.get("output_source_index")
        if not isinstance(audited_index, Mapping):
            raise ProvenanceError("render-audit record lacks source index")
        check(
            str(audited_index.get("path", "")),
            str(audited_index.get("sha256", "")),
            "render_audit_source_index",
        )
        audit_payload = _read_json(Path(str(audit_record["path"])))
        audit_core = dict(audit_payload)
        audit_content = str(audit_core.pop("content_sha256", ""))
        if (
            audit_content != canonical_json_sha256(audit_core)
            or audit_content != str(audit_record.get("content_sha256", ""))
            or audit_payload.get("output_source_index") != audited_index
        ):
            raise ProvenanceError("render-audit content commitment mismatch")
        try:
            build_record = _validated_build_provenance(
                payload.get("build_provenance")
            )
        except DatasetContractError as exc:
            raise ProvenanceError(str(exc)) from exc
        build_config = build_record["config"]
        expected_build_paths = {
            "source_index": str(audited_index["path"]),
            "render_audit_contract": str(audit_record["path"]),
            "geometry_contract": str(geometry["path"]),
            "output_dir": str(dataset_manifest_path.resolve().parent),
        }
        if str(build_config.get("label_contract", "")) != (
            LABEL_CONTRACT_OBSERVABLE_PHYSICAL_V3
        ):
            raise ProvenanceError("dataset build label contract mismatch")
        for key, expected_path in expected_build_paths.items():
            if str(build_config.get(key, "")) != expected_path:
                raise ProvenanceError(f"dataset build {key} mismatch")
        for artifact in build_record["inputs"].values():
            check(
                str(artifact["path"]),
                str(artifact["sha256"]),
                "build_input",
            )
        build_geometry = build_record["dependencies"]["geometry_contract"]
        if (
            str(build_geometry["path"]) != str(geometry["path"])
            or str(build_geometry["sha256"]) != str(geometry["file_sha256"])
        ):
            raise ProvenanceError("dataset build geometry dependency mismatch")
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

    shards = payload.get("shards", [])
    if not isinstance(shards, list):
        raise ProvenanceError("dataset shards must be a list")
    shard_by_scene: dict[str, Mapping[str, Any]] = {}
    shard_path_owners: dict[Path, str] = {}
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise ProvenanceError("shard provenance entries must be objects")
        shard_scene_id = str(shard.get("scene_id", "")).strip()
        if shard_scene_id not in source_by_scene or shard_scene_id in shard_by_scene:
            raise ProvenanceError(
                f"shard scene IDs must name one unique source: {shard_scene_id!r}"
            )
        shard_path = Path(str(shard.get("path", ""))).resolve()
        prior_owner = shard_path_owners.setdefault(shard_path, shard_scene_id)
        if prior_owner != shard_scene_id:
            raise ProvenanceError(
                f"label shard path is shared across scenes: {shard_path}"
            )
        shard_by_scene[shard_scene_id] = shard

    indexed_rows = list(_read_jsonl(Path(index["path"])))
    if len(indexed_rows) != int(payload.get("row_count", -1)):
        raise ProvenanceError("row-index count disagrees with dataset manifest")
    indexed_row_counts: Counter[str] = Counter()
    indexed_shard_rows: dict[str, set[int]] = defaultdict(set)
    image_path_owners: dict[Path, set[str]] = defaultdict(set)
    for row in indexed_rows:
        if str(row.get("schema", "")) != expected_row_schema:
            raise ProvenanceError(
                "row schema disagrees with paired-navigation dataset schema"
            )
        if (
            dataset_schema == "lewm_go2_paired_navigation_dataset_v3"
            and str(row.get("label_contract", ""))
            != LABEL_CONTRACT_OBSERVABLE_PHYSICAL_V3
        ):
            raise ProvenanceError("v3 row lacks observable-physical label contract")
        row_scene_id = str(row.get("scene_id", ""))
        if row_scene_id not in source_by_scene:
            raise ProvenanceError(
                f"row index names unknown source scene: {row_scene_id!r}"
            )
        source = source_by_scene[row_scene_id]
        if "dataset_role" in source or "dataset_role" in row:
            if str(row.get("dataset_role", "")) != str(
                source.get("dataset_role", "")
            ):
                raise ProvenanceError(
                    f"row role disagrees with source role: {row_scene_id}"
                )
        shard = shard_by_scene.get(row_scene_id)
        if shard is None:
            raise ProvenanceError(f"indexed scene lacks a label shard: {row_scene_id}")
        if str(row.get("label_shard_path", "")) != str(shard.get("path", "")):
            raise ProvenanceError(
                f"row label-shard path disagrees with scene shard: {row_scene_id}"
            )
        if str(row.get("label_shard_sha256", "")) != str(
            shard.get("sha256", "")
        ):
            raise ProvenanceError(
                f"row label-shard hash disagrees with scene shard: {row_scene_id}"
            )
        shard_row = int(row.get("label_shard_row", -1))
        if shard_row < 0 or shard_row in indexed_shard_rows[row_scene_id]:
            raise ProvenanceError(
                f"invalid or duplicate label-shard row for scene: {row_scene_id}"
            )
        indexed_shard_rows[row_scene_id].add(shard_row)
        indexed_row_counts[row_scene_id] += 1
        for prefix in ("current", "next"):
            image_path_owners[
                Path(str(row[f"{prefix}_image_path"])).resolve()
            ].add(row_scene_id)
    for source_scene_id, source in source_by_scene.items():
        declared_rows = int(source.get("rows", -1))
        if indexed_row_counts[source_scene_id] != declared_rows:
            raise ProvenanceError(
                f"source row count disagrees with index: {source_scene_id}"
            )
        shard = shard_by_scene.get(source_scene_id)
        if shard is not None and int(shard.get("rows", -1)) != declared_rows:
            raise ProvenanceError(
                f"shard row count disagrees with source: {source_scene_id}"
            )
        if indexed_shard_rows[source_scene_id] != set(range(declared_rows)):
            raise ProvenanceError(
                f"label-shard row indices are incomplete: {source_scene_id}"
            )
    aliased_images = sorted(
        str(path) for path, owners in image_path_owners.items() if len(owners) > 1
    )
    if aliased_images:
        raise ProvenanceError(
            "RGB paths are shared across scene roles: " + ", ".join(aliased_images)
        )

    required_shard_scenes = {
        scene_id
        for scene_id, source in source_by_scene.items()
        if int(source.get("rows", 0)) > 0
        and (selected_scene_ids is None or scene_id in selected_scene_ids)
    }
    missing_shards = sorted(required_shard_scenes - set(shard_by_scene))
    if missing_shards:
        raise ProvenanceError(f"selected scenes lack label shards: {missing_shards}")
    for shard_scene_id, shard in shard_by_scene.items():
        if selected_scene_ids is None or shard_scene_id in selected_scene_ids:
            check(str(shard["path"]), str(shard["sha256"]), "shard")

    for source_scene_id, source in source_by_scene.items():
        if selected_scene_ids is not None and source_scene_id not in selected_scene_ids:
            continue
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
        if "render_summary" in paths:
            check(
                paths["render_summary"],
                hashes["render_summary_file_sha256"],
                "render_summary",
            )
    if verify_images:
        seen_images: set[tuple[str, str]] = set()
        for row in indexed_rows:
            row_scene_id = str(row.get("scene_id", ""))
            if selected_scene_ids is not None and row_scene_id not in selected_scene_ids:
                continue
            for prefix in ("current", "next"):
                item = (row[f"{prefix}_image_path"], row[f"{prefix}_image_sha256"])
                if item in seen_images:
                    continue
                seen_images.add(item)
                check(item[0], item[1], "image")
        if (
            selected_scene_ids is None
            and canonical_json_sha256(sorted(seen_images))
            != payload["image_set_sha256"]
        ):
            raise ProvenanceError("image-set commitment does not match row index")
    if selected_scene_ids is not None:
        checked["selected_scene"] = len(selected_scene_ids)
        if selected_roles is not None:
            checked["selected_role"] = len(selected_roles)
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
                render_summary_path=(
                    source_path(str(row["render_summary_path"]))
                    if row.get("render_summary_path")
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
    "LABEL_CONTRACT_CENTER_VISIBLE_V2",
    "LABEL_CONTRACT_OBSERVABLE_PHYSICAL_V3",
    "LocalGridGeometry",
    "OBSERVABLE_FOOTPRINT_RADIUS_M",
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
    "label_camera_visible_navigation_grid",
    "label_camera_visible_physical_grid",
    "load_scene_id_exclusions",
    "load_source_index",
    "read_scene_id_commitments",
    "relative_se2_current_frame",
    "occupancy_label_metrics",
    "derive_configuration_labels_from_fused_physical_raster",
    "post_memory_configuration_morphology_metadata",
    "select_primitive_transitions",
    "scene_id_sha256",
    "sha256_file",
    "verify_dataset_provenance",
    "vertical_fov_from_horizontal",
]
