"""Read-only adapter for the frozen two-ply/per-link geometry corpus.

This module is deliberately an adapter, not a simulator.  It reads only the
artifacts frozen by ``TWO_PLY_SUCCESSOR_TRANSITION_CORPUS_REPAIRED_V1`` and
``EXPLICIT_PER_LINK_GEOMETRIC_MICRO_STATE_UPPER_BOUND_V1``.  In particular it
does not import Genesis, advance a solver, synthesize a state, or read any
sealed/G2 namespace.

The geometry shards retain every historical action row.  The deployable
decision contract, however, identifies actions by the controller and applied
one-tick command.  :func:`build_applied_action_copy_map` therefore supplies a
representative/copy map while retaining a mapping entry for every original
transition.  This is the only safe way to materialize expensive sensor
evidence once for physically identical transitions and still emit all 29,470
frozen output rows.

Public integration API
----------------------

``load_corpus_context(root) -> FrozenCorpus``
    Validates the corpus, split, action-contract, and geometry-index hashes and
    cardinalities.  The context contains metadata only; it does not load a
    geometry shard or snapshot.

``load_state(context, state_id) -> LoadedFrozenState``
    Loads exactly one state.  The returned object contains source/H3 rows,
    transition identities in shard order, the 27-shape contract, scene OBBs,
    the representative/copy map, the predecessor shard, and exact snapshot
    boundaries.  Important shapes are:

    * ``boundary_qpos``: ``(19,)`` current planning-boundary generalized state;
    * ``boundary_geom_transform``: ``(27, 7)`` current analytical collision
      transforms as ``xyz + quaternion(wxyz)``;
    * ``boundaries.current.raw_geom_transform``: ``(27, 7)`` exact last-27
      Genesis geometry-table transforms;
    * ``shard.arrays['link_transform']``: ``(T, 50, 13, 7)``;
    * ``shard.arrays['geom_transform']``: ``(T, 50, 27, 7)``;
    * ``shard.arrays['qpos']``: ``(T, 50, 19)``;
    * ``transition_rows`` and
      ``action_copy_map.representative_by_transition``: both length ``T``.

    Successor boundary qpos and geometry are keyed by frozen current action in
    ``successor_boundary_qpos`` and ``successor_boundary_geom_transform``.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import pickle
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


EXPERIMENT = "BODY_CENTRIC_RANGE_COVERAGE_QUALIFICATION_V1"
PREDECESSOR_EXPERIMENT = "EXPLICIT_PER_LINK_GEOMETRIC_MICRO_STATE_UPPER_BOUND_V1"

SOURCE_LINEAGE = "10b3a190d506830e6a87e04a0f1c832b92295bd7"
COMPLETED_RESULT_COMMIT = "034c2fb902997ac29e2742fc4ddc2c28ad1706b6"
CORPUS_LOGICAL_DIGEST = "e41d9926cb7f0f9e1158d09a88b746547806411950b9cac7ebe58aa500a92223"
CORPUS_INDEX_SHA256 = "c1055724ebffd71c67b5424b4e447d223a828d3bc4da01369486bff73ad265f0"
SPLIT_SHA256 = "eb2b41ca3ca4d4f7d2d2fc41495944e306e39798ede8865dd0904fa6c3d88021"
ACTION_CONTRACT_SHA256 = "cf8df092e8eff61d04348ebfe22b5e6a0cd31b5f39a4e05e45242752b3e5dc06"
REPAIRED_ROW_LEDGER_SHA256 = "63726e042e793d06784236b9dcc37c3844c798b8f526d03e4f19517186d5cc94"
GEOMETRY_INDEX_SHA256 = "67b473e6d0e0c422e4b2916b800399c98d5ec60383d398ff907fe2b8909a1d7f"

CORPUS_INDEX_RELATIVE = Path(
    ".generated/two_ply_successor_transition_corpus_repaired_v1/corpus_index.json"
)
SPLIT_RELATIVE = Path(
    ".generated/two_ply_successor_transition_corpus_repaired_v1/"
    "development_internal_calibration_repaired_v1.json"
)
ACTION_CONTRACT_RELATIVE = Path(
    ".generated/two_ply_successor_transition_corpus_repaired_v1/"
    "canonical_fourteen_action_contract.json"
)
GEOMETRY_INDEX_RELATIVE = Path(
    ".generated/explicit_per_link_geometric_micro_state_upper_bound_v1/geometry_index.json"
)

EXPECTED_STATES = 176
EXPECTED_TRANSITIONS = 29_470
EXPECTED_CURRENT_TRANSITIONS = 2_464
EXPECTED_SUCCESSOR_TRANSITIONS = 27_006
EXPECTED_PHYSICS_STEPS = 1_473_500
EXPECTED_ROLE_COUNTS = MappingProxyType(
    {"training": 128, "calibration": 24, "heldout": 24}
)
EXPECTED_CURRENT_REPRESENTATIVES = 1_594
EXPECTED_SUCCESSOR_REPRESENTATIVES = 11_791
PHYSICS_STEPS_PER_TRANSITION = 50
PROTECTED_LINK_COUNT = 13
COLLISION_SHAPE_COUNT = 27

ROLE_SPLIT_KEYS = MappingProxyType(
    {
        "training": "development_training_state_ids",
        "calibration": "internal_calibration_state_ids",
        "heldout": "development_heldout_state_ids",
    }
)
ROLE_LONG_NAMES = MappingProxyType(
    {
        "training": "training",
        "calibration": "internal_calibration",
        "heldout": "development_heldout",
    }
)

PROTECTED_LINK_NAMES = (
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


def body_region_for_link(link_name: str) -> str:
    """Return the prospectively fixed coverage region for a protected link."""

    name = str(link_name)
    if name == "base":
        return "trunk"
    if name.endswith("_calf"):
        return "calf"
    if name.startswith(("FL_", "FR_")):
        return "front_limb"
    if name.startswith(("RL_", "RR_")):
        return "rear_limb"
    raise CorpusBindingError(f"unrecognized protected link: {name}")


BODY_REGION_BY_LINK = MappingProxyType(
    {name: body_region_for_link(name) for name in PROTECTED_LINK_NAMES}
)

_LINK_POS_KEY = "Scene._sim._coupler.rigid_solver.links_state.pos"
_LINK_QUAT_KEY = "Scene._sim._coupler.rigid_solver.links_state.quat"
_GEOM_POS_KEY = "Scene._sim._coupler.rigid_solver.geoms_state.pos"
_GEOM_QUAT_KEY = "Scene._sim._coupler.rigid_solver.geoms_state.quat"
_QPOS_KEY = "Scene._sim._coupler.rigid_solver.qpos"

_REQUIRED_SHARD_FIELDS = (
    "frozen_contact_label",
    "physics_timestamp_s",
    "qpos",
    "link_transform",
    "geom_transform",
    "native_contact",
    "native_robot_link",
    "native_other_link",
    "native_penetration",
    "exact_contact",
    "exact_contact_count",
    "exact_robot_link",
    "exact_other_link",
    "exact_robot_geom",
    "exact_other_geom",
    "exact_penetration",
    "exact_manifold_position",
    "exact_manifold_normal",
    "transition_level",
    "transition_current_action",
    "transition_action",
)


class CorpusBindingError(RuntimeError):
    """The local predecessor evidence does not match its frozen binding."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_content_digest(value: Mapping[str, Any], *, omit: Iterable[str] = ()) -> str:
    excluded = set(omit)
    payload = {key: item for key, item in value.items() if key not in excluded}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CorpusBindingError(f"cannot read frozen JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise CorpusBindingError(f"frozen JSON is not an object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CorpusBindingError(message)


def _immutable_array(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    output = np.array(value, dtype=dtype, copy=True)
    output.setflags(write=False)
    return output


def _as_envless_array(value: Any, trailing: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim >= 2 and array.shape[-2] == 1 and array.shape[-1] == trailing:
        array = array[..., 0, :]
    if array.ndim != 2 or array.shape[1] != trailing:
        raise CorpusBindingError(
            f"snapshot transform has shape {array.shape}, expected (N, {trailing})"
        )
    return array


def _resolve_frozen_path(root: Path, recorded: str | Path) -> Path:
    """Resolve a predecessor path after a repository/cache relocation.

    Only the two authorized cache namespaces are accepted.  There is no
    generic path escape through this adapter.
    """

    path = Path(recorded).expanduser()
    if path.is_file():
        return path
    text = path.as_posix()
    generated_marker = "/.generated/"
    if generated_marker in text:
        candidate = root / ".generated" / text.split(generated_marker, 1)[1]
        if candidate.is_file():
            return candidate
    cache_marker = "/.cache/lewm_go2_temporal_v03/"
    if cache_marker in text:
        candidate = (
            Path.home()
            / ".cache/lewm_go2_temporal_v03"
            / text.split(cache_marker, 1)[1]
        )
        if candidate.is_file():
            return candidate
    raise CorpusBindingError(f"missing frozen predecessor artifact: {recorded}")


def applied_action_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Frozen deployable identity: controller plus 1e-7 applied command."""

    command = row.get("applied_action")
    if not isinstance(command, Sequence) or isinstance(command, (str, bytes)):
        raise CorpusBindingError("transition lacks an applied_action sequence")
    if len(command) != 3:
        raise CorpusBindingError(f"applied_action must have three values, got {command}")
    values = tuple(round(float(item), 7) for item in command)
    if not all(math.isfinite(item) for item in values):
        raise CorpusBindingError("applied_action contains a non-finite value")
    return (str(row["controller"]), *values)


@dataclass(frozen=True)
class AppliedActionCopyMap:
    """Physical-transition representatives with full historical row custody."""

    state_id: str
    representative_by_transition: tuple[int, ...]
    copies_by_representative: Mapping[int, tuple[int, ...]]
    current_action_representative: Mapping[int, int]
    current_representative_transitions: tuple[int, ...]
    successor_representative_transitions: tuple[int, ...]
    unique_successor_prefix_actions: tuple[int, ...]

    @property
    def transition_count(self) -> int:
        return len(self.representative_by_transition)

    @property
    def representative_count(self) -> int:
        return len(self.copies_by_representative)

    @property
    def current_representative_count(self) -> int:
        return len(self.current_representative_transitions)

    @property
    def successor_representative_count(self) -> int:
        return len(self.successor_representative_transitions)

    def representative_for(self, transition_index: int) -> int:
        return self.representative_by_transition[int(transition_index)]


@dataclass(frozen=True)
class SceneOBBSet:
    object_names: tuple[str, ...]
    centers_m: np.ndarray
    half_extents_m: np.ndarray
    yaw_rad: np.ndarray

    def __len__(self) -> int:
        return len(self.object_names)


@dataclass(frozen=True)
class BoundaryTransforms:
    snapshot_digest: str
    boundary: Mapping[str, Any]
    qpos: np.ndarray
    link_global_indices: tuple[int, ...]
    link_names: tuple[str, ...]
    link_transform: np.ndarray
    raw_geom_global_indices: tuple[int, ...]
    raw_geom_transform: np.ndarray
    contract_geom_transform: np.ndarray


@dataclass(frozen=True)
class StateBoundaryTransforms:
    state_id: str
    current: BoundaryTransforms
    successors: Mapping[int, BoundaryTransforms]


@dataclass(frozen=True)
class GeometryShard:
    state_id: str
    arrays: Mapping[str, np.ndarray]
    transition_rows: tuple[Mapping[str, Any], ...]
    identity_rows: tuple[Mapping[str, Any], ...]
    geometry_receipt: Mapping[str, Any]

    @property
    def transition_count(self) -> int:
        return len(self.transition_rows)

    def transition_index(
        self, level: str, current_action_index: int, action_index: int
    ) -> int:
        target = (str(level), int(current_action_index), int(action_index))
        for row in self.transition_rows:
            identity = (
                str(row["level"]),
                int(row["current_action_index"]),
                int(row["action_index"]),
            )
            if identity == target:
                return int(row["transition_index"])
        raise KeyError(target)


@dataclass(frozen=True)
class LoadedFrozenState:
    """One streaming unit for the sensor-coverage materializer."""

    state_id: str
    scene_id: str
    family: str
    role: str
    role_long_name: str
    source_record: Mapping[str, Any]
    geometry_record: Mapping[str, Any]
    current_rows: tuple[Mapping[str, Any], ...]
    successor_rows: tuple[Mapping[str, Any], ...]
    transition_rows: tuple[Mapping[str, Any], ...]
    scene_boxes: SceneOBBSet
    geometry_contract: tuple[Mapping[str, Any], ...]
    protected_link_names: tuple[str, ...]
    body_region_by_link: Mapping[str, str]
    action_copy_map: AppliedActionCopyMap
    shard: GeometryShard
    boundaries: StateBoundaryTransforms

    @property
    def boundary_qpos(self) -> np.ndarray:
        return self.boundaries.current.qpos

    @property
    def boundary_geom_transform(self) -> np.ndarray:
        return self.boundaries.current.contract_geom_transform

    @property
    def boundary_raw_geom_transform(self) -> np.ndarray:
        return self.boundaries.current.raw_geom_transform

    @property
    def successor_boundary_qpos(self) -> Mapping[int, np.ndarray]:
        return MappingProxyType(
            {action: boundary.qpos for action, boundary in self.boundaries.successors.items()}
        )

    @property
    def successor_boundary_geom_transform(self) -> Mapping[int, np.ndarray]:
        return MappingProxyType(
            {
                action: boundary.contract_geom_transform
                for action, boundary in self.boundaries.successors.items()
            }
        )


@dataclass(frozen=True)
class FrozenCorpus:
    root: Path
    corpus_index: Mapping[str, Any]
    split: Mapping[str, Any]
    action_contract: Mapping[str, Any]
    geometry_index: Mapping[str, Any]
    corpus_records_by_id: Mapping[str, Mapping[str, Any]]
    geometry_records_by_id: Mapping[str, Mapping[str, Any]]
    role_by_state_id: Mapping[str, str]
    state_ids: tuple[str, ...]
    binding_receipt: Mapping[str, Any]

    def role_for_state(self, state_id: str, *, long_name: bool = False) -> str:
        role = self.role_by_state_id[str(state_id)]
        return ROLE_LONG_NAMES[role] if long_name else role

    def state_ids_for_role(self, role: str) -> tuple[str, ...]:
        aliases = {
            "internal_calibration": "calibration",
            "development_heldout": "heldout",
        }
        short = aliases.get(str(role), str(role))
        if short not in ROLE_SPLIT_KEYS:
            raise KeyError(role)
        return tuple(
            state_id
            for state_id in self.state_ids
            if self.role_by_state_id[state_id] == short
        )

    def corpus_record(self, state_id: str) -> Mapping[str, Any]:
        return self.corpus_records_by_id[str(state_id)]

    def geometry_record(self, state_id: str) -> Mapping[str, Any]:
        return self.geometry_records_by_id[str(state_id)]

    def action_copy_map(self, state_id: str) -> AppliedActionCopyMap:
        return build_applied_action_copy_map(
            self.corpus_record(state_id), self.geometry_record(state_id)
        )

    def transition_identity_rows(
        self, state_id: str
    ) -> tuple[Mapping[str, Any], ...]:
        return transition_identity_rows(self, state_id)

    def iter_transition_identity_rows(
        self, *, role: str | None = None
    ) -> Iterator[Mapping[str, Any]]:
        allowed = None if role is None else set(self.state_ids_for_role(role))
        for state_id in self.state_ids:
            if allowed is None or state_id in allowed:
                yield from self.transition_identity_rows(state_id)

    def scene_obbs(self, state_id: str) -> SceneOBBSet:
        return scene_obbs(self.geometry_record(state_id))

    def collision_shape_contract(
        self, state_id: str
    ) -> tuple[Mapping[str, Any], ...]:
        return collision_shape_contract(self.geometry_record(state_id))

    def protected_link_names(self, state_id: str) -> tuple[str, ...]:
        names = tuple(
            str(row["link_name"])
            for row in self.collision_shape_contract(state_id)
            if int(row["geom_index"])
            == min(
                int(candidate["geom_index"])
                for candidate in self.collision_shape_contract(state_id)
                if int(candidate["link_index"]) == int(row["link_index"])
            )
        )
        _require(names == PROTECTED_LINK_NAMES, f"{state_id}: protected link order drift")
        return names

    def load_geometry_shard(
        self, state_id: str, *, fields: Iterable[str] | None = None
    ) -> GeometryShard:
        return load_geometry_shard(self, state_id, fields=fields)

    def load_boundary_transforms(self, state_id: str) -> StateBoundaryTransforms:
        return load_boundary_transforms(self, state_id)


def _validate_source_and_geometry_rows(
    corpus_record: Mapping[str, Any], geometry_record: Mapping[str, Any]
) -> None:
    state_id = str(corpus_record["state_id"])
    _require(str(geometry_record["state_id"]) == state_id, f"{state_id}: receipt mismatch")
    _require(corpus_record.get("status") == "PASS", f"{state_id}: corpus state not PASS")
    _require(geometry_record.get("status") == "PASS", f"{state_id}: geometry state not PASS")
    current = {int(row["action_index"]): row for row in corpus_record["current_rows"]}
    successors = {
        int(row["current_action_index"]): row for row in corpus_record["successor_rows"]
    }
    rows = list(geometry_record["transition_rows"])
    _require(
        [int(row["transition_index"]) for row in rows] == list(range(len(rows))),
        f"{state_id}: non-contiguous transition indices",
    )
    for metadata in rows:
        level = str(metadata["level"])
        action = int(metadata["action_index"])
        current_action = int(metadata["current_action_index"])
        if level == "current":
            _require(current_action == -1 and action in current, f"{state_id}: bad current row")
            source = current[action]
            frozen = bool(source["current_contact"])
        elif level == "successor":
            _require(current_action in successors, f"{state_id}: orphan successor prefix")
            candidates = {
                int(row["action_index"]): row
                for row in successors[current_action]["next_actions"]
            }
            _require(action in candidates, f"{state_id}: orphan successor action")
            source = candidates[action]
            frozen = bool(source["contact"])
        else:
            raise CorpusBindingError(f"{state_id}: invalid transition level {level}")
        _require(
            applied_action_key(source) == applied_action_key(metadata),
            f"{state_id}: applied-action alignment failed at transition "
            f"{metadata['transition_index']}",
        )
        _require(
            bool(metadata["frozen_contact"]) == frozen,
            f"{state_id}: frozen contact alignment failed",
        )
    expected = len(current) + sum(len(row["next_actions"]) for row in successors.values())
    _require(len(rows) == expected, f"{state_id}: transition cardinality mismatch")


def _role_map(split: Mapping[str, Any], state_ids: Sequence[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for role, key in ROLE_SPLIT_KEYS.items():
        values = split.get(key)
        _require(isinstance(values, list), f"split missing {key}")
        _require(len(values) == EXPECTED_ROLE_COUNTS[role], f"{role} role count drift")
        for state_id in values:
            identity = str(state_id)
            _require(identity not in output, f"split role overlap at {identity}")
            output[identity] = role
    _require(set(output) == set(state_ids), "split is not an exact state partition")
    return output


def _validate_global_cardinalities(
    corpus: Mapping[str, Any], geometry: Mapping[str, Any]
) -> None:
    _require(corpus.get("status") == "PASS", "repaired corpus is not PASS")
    _require(geometry.get("status") == "PASS", "geometry index is not PASS")
    _require(int(geometry.get("states", -1)) == EXPECTED_STATES, "state count drift")
    _require(
        int(geometry.get("transitions", -1)) == EXPECTED_TRANSITIONS,
        "transition count drift",
    )
    _require(
        int(geometry.get("materialized_current_transitions", -1))
        == EXPECTED_CURRENT_TRANSITIONS,
        "current transition count drift",
    )
    _require(
        int(geometry.get("materialized_successor_transitions", -1))
        == EXPECTED_SUCCESSOR_TRANSITIONS,
        "successor transition count drift",
    )
    _require(
        int(geometry.get("physics_steps", -1)) == EXPECTED_PHYSICS_STEPS,
        "physics-step count drift",
    )
    inventory = corpus.get("inventory", {})
    _require(set(inventory) == set(EXPECTED_ROLE_COUNTS), "corpus role inventory drift")
    states = sum(int(inventory[role]["states"]) for role in EXPECTED_ROLE_COUNTS)
    current = sum(
        int(inventory[role]["current_transitions"]) for role in EXPECTED_ROLE_COUNTS
    )
    successor = sum(
        int(inventory[role]["successor_action_transitions"])
        for role in EXPECTED_ROLE_COUNTS
    )
    _require(states == EXPECTED_STATES, "corpus inventory state count drift")
    _require(current == EXPECTED_CURRENT_TRANSITIONS, "corpus inventory current count drift")
    _require(
        successor == EXPECTED_SUCCESSOR_TRANSITIONS,
        "corpus inventory successor count drift",
    )
    audit = geometry.get("deployable_action_audit", {})
    _require(
        audit.get("classification") == "DEPLOYABLE_MICRO_ACTION_CONTRACT_ALIGNED",
        "deployable action audit is not aligned",
    )
    _require(audit.get("duplicate_outcome_inconsistencies") == [], "duplicate outcome drift")
    _require(
        int(audit.get("current_entries", -1)) == EXPECTED_CURRENT_TRANSITIONS,
        "deployable current-entry count drift",
    )
    _require(
        int(audit.get("unique_current_entries", -1))
        == EXPECTED_CURRENT_REPRESENTATIVES,
        "deployable unique-current count drift",
    )
    _require(
        int(audit.get("unique_successor_entries_for_unique_prefixes", -1))
        == EXPECTED_SUCCESSOR_REPRESENTATIVES,
        "deployable unique-successor count drift",
    )


def load_frozen_corpus(root: str | Path | None = None) -> FrozenCorpus:
    """Load and verify the frozen development corpus without opening G2."""

    repository = (
        Path(root).expanduser().resolve()
        if root is not None
        else Path(__file__).resolve().parents[2]
    )
    corpus_path = repository / CORPUS_INDEX_RELATIVE
    split_path = repository / SPLIT_RELATIVE
    action_path = repository / ACTION_CONTRACT_RELATIVE
    geometry_path = repository / GEOMETRY_INDEX_RELATIVE
    expected_hashes = {
        corpus_path: CORPUS_INDEX_SHA256,
        split_path: SPLIT_SHA256,
        action_path: ACTION_CONTRACT_SHA256,
        geometry_path: GEOMETRY_INDEX_SHA256,
    }
    observed_hashes: dict[str, str] = {}
    for path, expected in expected_hashes.items():
        _require(path.is_file(), f"missing frozen binding: {path}")
        observed = sha256_file(path)
        _require(observed == expected, f"SHA-256 mismatch for {path}: {observed}")
        observed_hashes[str(path.relative_to(repository))] = observed

    corpus = _read_json(corpus_path)
    split = _read_json(split_path)
    action = _read_json(action_path)
    geometry = _read_json(geometry_path)
    _require(
        corpus.get("corpus_logical_digest") == CORPUS_LOGICAL_DIGEST,
        "corpus logical digest drift",
    )
    _require(
        geometry.get("experiment") == PREDECESSOR_EXPERIMENT,
        "wrong predecessor geometry experiment",
    )
    _require(geometry.get("source_commit") == SOURCE_LINEAGE, "geometry source drift")
    bindings = geometry.get("bindings", {})
    _require(
        bindings.get("corpus_logical_digest") == CORPUS_LOGICAL_DIGEST,
        "geometry/corpus logical binding drift",
    )
    _require(
        bindings.get("corpus_index_sha256") == CORPUS_INDEX_SHA256,
        "geometry/corpus index binding drift",
    )
    _require(
        bindings.get("action_contract_sha256") == ACTION_CONTRACT_SHA256,
        "geometry/action binding drift",
    )
    _require(
        bindings.get("predecessor_row_ledger_sha256") == REPAIRED_ROW_LEDGER_SHA256,
        "repaired row-ledger binding drift",
    )
    _require(corpus.get("split", {}).get("sha256") == SPLIT_SHA256, "split binding drift")
    _require(
        corpus.get("action_contract", {}).get("sha256") == ACTION_CONTRACT_SHA256,
        "corpus action binding drift",
    )
    _require(
        split.get("content_digest")
        == json_content_digest(split, omit=("content_digest",)),
        "split content digest drift",
    )
    _require(
        action.get("action_bank_source_digest")
        == json_content_digest(action, omit=("action_bank_source_digest",)),
        "action-contract content digest drift",
    )
    _validate_global_cardinalities(corpus, geometry)

    corpus_records = list(corpus.get("records", []))
    geometry_records = list(geometry.get("records", []))
    _require(len(corpus_records) == EXPECTED_STATES, "corpus record count drift")
    _require(len(geometry_records) == EXPECTED_STATES, "geometry record count drift")
    corpus_by_id = {str(row["state_id"]): row for row in corpus_records}
    geometry_by_id = {str(row["state_id"]): row for row in geometry_records}
    _require(len(corpus_by_id) == EXPECTED_STATES, "duplicate corpus state identity")
    _require(len(geometry_by_id) == EXPECTED_STATES, "duplicate geometry state identity")
    state_ids = tuple(str(row["state_id"]) for row in corpus_records)
    _require(set(corpus_by_id) == set(geometry_by_id), "corpus/geometry state mismatch")
    roles = _role_map(split, state_ids)

    transition_count = 0
    current_representatives = 0
    successor_representatives = 0
    for state_id in state_ids:
        corpus_record = corpus_by_id[state_id]
        geometry_record = geometry_by_id[state_id]
        _validate_source_and_geometry_rows(corpus_record, geometry_record)
        _require(
            int(geometry_record.get("physics_steps_per_transition", -1))
            == PHYSICS_STEPS_PER_TRANSITION,
            f"{state_id}: physics-step cardinality drift",
        )
        _require(
            int(geometry_record.get("relevant_collision_shapes", -1))
            == COLLISION_SHAPE_COUNT,
            f"{state_id}: collision-shape cardinality drift",
        )
        _require(
            int(geometry_record.get("robot_links", -1)) == PROTECTED_LINK_COUNT,
            f"{state_id}: protected-link cardinality drift",
        )
        action_map = build_applied_action_copy_map(corpus_record, geometry_record)
        transition_count += action_map.transition_count
        current_representatives += action_map.current_representative_count
        successor_representatives += action_map.successor_representative_count
    _require(transition_count == EXPECTED_TRANSITIONS, "row custody count drift")
    _require(
        current_representatives == EXPECTED_CURRENT_REPRESENTATIVES,
        "unique current-action count drift",
    )
    _require(
        successor_representatives == EXPECTED_SUCCESSOR_REPRESENTATIVES,
        "unique successor-action count drift",
    )

    receipt = {
        "schema": "body_centric_range_coverage_frozen_corpus_binding_v1",
        "source_lineage": SOURCE_LINEAGE,
        "completed_result_commit": COMPLETED_RESULT_COMMIT,
        "corpus_logical_digest": CORPUS_LOGICAL_DIGEST,
        "hashes": observed_hashes,
        "states": EXPECTED_STATES,
        "transitions": transition_count,
        "roles": dict(EXPECTED_ROLE_COUNTS),
        "current_representatives": current_representatives,
        "successor_representatives": successor_representatives,
        "sealed_or_g2_opened": False,
    }
    receipt["content_digest"] = json_content_digest(receipt)
    return FrozenCorpus(
        root=repository,
        corpus_index=MappingProxyType(corpus),
        split=MappingProxyType(split),
        action_contract=MappingProxyType(action),
        geometry_index=MappingProxyType(geometry),
        corpus_records_by_id=MappingProxyType(corpus_by_id),
        geometry_records_by_id=MappingProxyType(geometry_by_id),
        role_by_state_id=MappingProxyType(roles),
        state_ids=state_ids,
        binding_receipt=MappingProxyType(receipt),
    )


def build_applied_action_copy_map(
    corpus_record: Mapping[str, Any], geometry_record: Mapping[str, Any]
) -> AppliedActionCopyMap:
    """Map every frozen row to its unique deployable physical representative.

    Successor rows are keyed by both the representative applied current action
    and the applied next action.  This collapses duplicate current prefixes as
    well as duplicate next actions, matching the predecessor's audited
    ``unique successor entries for unique prefixes`` contract.
    """

    state_id = str(corpus_record["state_id"])
    metadata_rows = list(geometry_record["transition_rows"])
    by_identity = {
        (
            str(row["level"]),
            int(row["current_action_index"]),
            int(row["action_index"]),
        ): row
        for row in metadata_rows
    }
    current_rows = list(corpus_record["current_rows"])
    current_representative_by_key: dict[tuple[Any, ...], int] = {}
    current_action_representative: dict[int, int] = {}
    for row in current_rows:
        action = int(row["action_index"])
        representative = current_representative_by_key.setdefault(
            applied_action_key(row), action
        )
        current_action_representative[action] = representative

    successor_by_prefix = {
        int(row["current_action_index"]): row for row in corpus_record["successor_rows"]
    }
    representative_by_transition = [-1] * len(metadata_rows)
    current_representative_transitions: list[int] = []
    successor_representative_transitions: list[int] = []

    for row in current_rows:
        action = int(row["action_index"])
        target_action = current_action_representative[action]
        transition = int(by_identity[("current", -1, action)]["transition_index"])
        representative = int(
            by_identity[("current", -1, target_action)]["transition_index"]
        )
        representative_by_transition[transition] = representative
        if transition == representative:
            current_representative_transitions.append(transition)

    successor_lookup: dict[tuple[int, tuple[Any, ...]], tuple[int, int]] = {}
    unique_prefixes = tuple(
        action
        for action in dict.fromkeys(current_action_representative.values())
        if action in successor_by_prefix
    )
    for prefix in unique_prefixes:
        for row in successor_by_prefix[prefix]["next_actions"]:
            key = (prefix, applied_action_key(row))
            action = int(row["action_index"])
            identity = ("successor", prefix, action)
            transition = int(by_identity[identity]["transition_index"])
            if key not in successor_lookup:
                successor_lookup[key] = (action, transition)
                successor_representative_transitions.append(transition)

    for original_prefix, successor in successor_by_prefix.items():
        representative_prefix = current_action_representative[original_prefix]
        _require(
            representative_prefix in successor_by_prefix,
            f"{state_id}: duplicate successor prefix lacks representative",
        )
        for row in successor["next_actions"]:
            action = int(row["action_index"])
            transition = int(
                by_identity[("successor", original_prefix, action)]["transition_index"]
            )
            key = (representative_prefix, applied_action_key(row))
            _require(key in successor_lookup, f"{state_id}: successor copy key missing")
            representative_by_transition[transition] = successor_lookup[key][1]

    _require(
        all(value >= 0 for value in representative_by_transition),
        f"{state_id}: incomplete representative map",
    )
    copies: dict[int, list[int]] = {}
    for transition, representative in enumerate(representative_by_transition):
        copies.setdefault(representative, []).append(transition)
    for representative, rows in copies.items():
        _require(representative in rows, f"{state_id}: representative is not self-bound")
    frozen_copies = MappingProxyType(
        {key: tuple(value) for key, value in sorted(copies.items())}
    )
    return AppliedActionCopyMap(
        state_id=state_id,
        representative_by_transition=tuple(representative_by_transition),
        copies_by_representative=frozen_copies,
        current_action_representative=MappingProxyType(current_action_representative),
        current_representative_transitions=tuple(current_representative_transitions),
        successor_representative_transitions=tuple(successor_representative_transitions),
        unique_successor_prefix_actions=unique_prefixes,
    )


def _source_rows(
    corpus_record: Mapping[str, Any],
) -> dict[tuple[str, int, int], Mapping[str, Any]]:
    output: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    for row in corpus_record["current_rows"]:
        output[("current", -1, int(row["action_index"]))] = row
    for successor in corpus_record["successor_rows"]:
        prefix = int(successor["current_action_index"])
        for row in successor["next_actions"]:
            output[("successor", prefix, int(row["action_index"]))] = row
    return output


def transition_identity_rows(
    corpus: FrozenCorpus, state_id: str
) -> tuple[Mapping[str, Any], ...]:
    """Return all frozen output identities in exact shard-array order."""

    source_receipt = corpus.corpus_record(state_id)
    geometry_receipt = corpus.geometry_record(state_id)
    sources = _source_rows(source_receipt)
    copy_map = corpus.action_copy_map(state_id)
    rows: list[Mapping[str, Any]] = []
    for metadata in geometry_receipt["transition_rows"]:
        transition = int(metadata["transition_index"])
        identity = (
            str(metadata["level"]),
            int(metadata["current_action_index"]),
            int(metadata["action_index"]),
        )
        source = sources[identity]
        frozen = bool(
            source["current_contact"]
            if identity[0] == "current"
            else source["contact"]
        )
        output: dict[str, Any] = {
            "state_index": int(source_receipt["state_index"]),
            "state_id": str(state_id),
            "scene_id": str(source_receipt["scene_id"]),
            "family": str(source_receipt["family"]),
            "role": corpus.role_for_state(state_id),
            "role_long_name": corpus.role_for_state(state_id, long_name=True),
            "transition_index": transition,
            "identity": str(metadata["identity"]),
            "level": identity[0],
            "current_action_index": identity[1],
            "action_index": identity[2],
            "candidate_name": str(source["candidate_name"]),
            "controller": str(source["controller"]),
            "requested_action": tuple(float(item) for item in source["requested_action"]),
            "applied_action": tuple(float(item) for item in source["applied_action"]),
            "applied_action_key": applied_action_key(source),
            "representative_transition_index": copy_map.representative_for(transition),
            "is_physical_representative": copy_map.representative_for(transition)
            == transition,
            "frozen_contact": frozen,
            "frozen_first_contact_step": source.get("first_contact_step"),
            "native_replay_contact": bool(metadata["native_contact"]),
            "native_first_contact_step": metadata.get("native_first_contact_step"),
            "exact_contact": bool(metadata["exact_contact"]),
            "exact_first_contact_step": metadata.get("exact_first_contact_step"),
        }
        if identity[0] == "current":
            for field in (
                "h3_progress_m",
                "h3_heading_improvement_rad",
                "decision_progress_m",
                "immediate_progress_m",
                "successor_safe_action_count",
                "successor_viable",
                "successor_identity",
                "successor_snapshot_digest",
            ):
                output[field] = source.get(field)
        rows.append(MappingProxyType(output))
    _require(len(rows) == int(geometry_receipt["transitions"]), "identity row count drift")
    return tuple(rows)


def scene_obbs(geometry_record: Mapping[str, Any]) -> SceneOBBSet:
    rows = list(geometry_record["scene_collision_primitives"])
    names = tuple(str(row["object"]) for row in rows)
    centers = _immutable_array([row["center"] for row in rows], dtype=np.float64)
    halves = _immutable_array([row["half_extent"] for row in rows], dtype=np.float64)
    yaws = _immutable_array([row["yaw_rad"] for row in rows], dtype=np.float64)
    _require(centers.shape == (len(rows), 3), "scene OBB center shape drift")
    _require(halves.shape == (len(rows), 3), "scene OBB half-extent shape drift")
    _require(yaws.shape == (len(rows),), "scene OBB yaw shape drift")
    _require(np.isfinite(centers).all(), "scene OBB centers are not finite")
    _require(np.isfinite(halves).all() and np.all(halves > 0), "invalid OBB extent")
    _require(np.isfinite(yaws).all(), "scene OBB yaws are not finite")
    return SceneOBBSet(names, centers, halves, yaws)


def collision_shape_contract(
    geometry_record: Mapping[str, Any]
) -> tuple[Mapping[str, Any], ...]:
    rows = tuple(geometry_record["collision_shape_contract"])
    _require(len(rows) == COLLISION_SHAPE_COUNT, "collision-shape count drift")
    _require(
        [int(row["geom_index"]) for row in rows]
        == list(range(COLLISION_SHAPE_COUNT)),
        "collision geometry indices are not canonical",
    )
    local_indices = sorted({int(row["link_index"]) for row in rows})
    _require(local_indices == list(range(PROTECTED_LINK_COUNT)), "protected link index drift")
    for index, expected_name in enumerate(PROTECTED_LINK_NAMES):
        names = {
            str(row["link_name"])
            for row in rows
            if int(row["link_index"]) == index
        }
        _require(names == {expected_name}, f"protected link name drift at {index}: {names}")
    return rows


def _requested_shard_fields(fields: Iterable[str] | None) -> tuple[str, ...]:
    if fields is None:
        return ()
    ordered = tuple(dict.fromkeys(str(field) for field in fields))
    missing = set(_REQUIRED_SHARD_FIELDS) - set(ordered)
    return tuple((*ordered, *sorted(missing)))


def load_geometry_shard(
    corpus: FrozenCorpus,
    state_id: str,
    *,
    fields: Iterable[str] | None = None,
) -> GeometryShard:
    """Load one predecessor shard and prove JSON/array row alignment."""

    receipt = corpus.geometry_record(state_id)
    path = _resolve_frozen_path(corpus.root, receipt["shard_path"])
    observed_sha = sha256_file(path)
    _require(observed_sha == receipt["shard_sha256"], f"{state_id}: shard SHA drift")
    selected = _requested_shard_fields(fields)
    with np.load(path, allow_pickle=False) as archive:
        available = set(archive.files)
        _require(
            set(_REQUIRED_SHARD_FIELDS).issubset(available),
            f"{state_id}: required shard arrays missing",
        )
        names = archive.files if not selected else selected
        _require(set(names).issubset(available), f"{state_id}: requested shard field missing")
        arrays = {name: _immutable_array(archive[name]) for name in names}
    rows = tuple(MappingProxyType(dict(row)) for row in receipt["transition_rows"])
    transitions = len(rows)
    expected_shapes = {
        "frozen_contact_label": (transitions,),
        "physics_timestamp_s": (transitions, PHYSICS_STEPS_PER_TRANSITION),
        "qpos": (transitions, PHYSICS_STEPS_PER_TRANSITION, 19),
        "link_transform": (
            transitions,
            PHYSICS_STEPS_PER_TRANSITION,
            PROTECTED_LINK_COUNT,
            7,
        ),
        "geom_transform": (
            transitions,
            PHYSICS_STEPS_PER_TRANSITION,
            COLLISION_SHAPE_COUNT,
            7,
        ),
        "native_contact": (transitions, PHYSICS_STEPS_PER_TRANSITION),
        "exact_contact": (transitions, PHYSICS_STEPS_PER_TRANSITION),
        "transition_level": (transitions,),
        "transition_current_action": (transitions,),
        "transition_action": (transitions,),
    }
    for name, shape in expected_shapes.items():
        _require(arrays[name].shape == shape, f"{state_id}: {name} shape drift")
    for row in rows:
        index = int(row["transition_index"])
        _require(str(arrays["transition_level"][index]) == str(row["level"]), "level drift")
        _require(
            int(arrays["transition_current_action"][index])
            == int(row["current_action_index"]),
            "current-action array drift",
        )
        _require(
            int(arrays["transition_action"][index]) == int(row["action_index"]),
            "action array drift",
        )
        _require(
            bool(arrays["frozen_contact_label"][index])
            == bool(row["frozen_contact"]),
            "frozen-contact array drift",
        )
    return GeometryShard(
        state_id=str(state_id),
        arrays=MappingProxyType(arrays),
        transition_rows=rows,
        identity_rows=corpus.transition_identity_rows(state_id),
        geometry_receipt=receipt,
    )


def _first_true(value: np.ndarray) -> int | None:
    indices = np.flatnonzero(np.asarray(value, dtype=bool))
    return None if not len(indices) else int(indices[0])


def _name_for_global_index(names: Mapping[str, Any], index: int) -> str | None:
    if int(index) < 0:
        return None
    return str(names.get(str(int(index)), "unresolved"))


def contact_attribution(shard: GeometryShard, transition_index: int) -> Mapping[str, Any]:
    """Expose independent frozen, native-replay, and exact contact custody."""

    index = int(transition_index)
    _require(0 <= index < shard.transition_count, "transition index out of range")
    arrays = shard.arrays
    metadata = shard.transition_rows[index]
    identity_row = shard.identity_rows[index]
    link_names = shard.geometry_receipt["link_names"]
    object_names = shard.geometry_receipt["object_names"]
    native_step = _first_true(arrays["native_contact"][index])
    exact_step = _first_true(arrays["exact_contact"][index])

    def native_value(field: str, default: Any = None) -> Any:
        if native_step is None:
            return default
        value = arrays[field][index, native_step]
        return value.item() if isinstance(value, np.generic) else value

    def exact_value(field: str, default: Any = None) -> Any:
        if exact_step is None:
            return default
        value = arrays[field][index, exact_step]
        return value.item() if isinstance(value, np.generic) else value

    native_link_index = int(native_value("native_robot_link", -1))
    native_link_name = _name_for_global_index(link_names, native_link_index)
    exact_link_index = int(exact_value("exact_robot_link", -1))
    exact_link_name = _name_for_global_index(link_names, exact_link_index)
    exact_position = exact_value("exact_manifold_position")
    exact_normal = exact_value("exact_manifold_normal")
    output = {
        "state_id": shard.state_id,
        "transition_index": index,
        "identity": metadata["identity"],
        "frozen": {
            "contact": bool(arrays["frozen_contact_label"][index]),
            "first_contact_step": identity_row.get("frozen_first_contact_step"),
            "attribution_source": "repaired row label; link attribution is not imputed",
        },
        "native_replay": {
            "contact": native_step is not None,
            "first_contact_step": native_step,
            "robot_link_index": None if native_step is None else native_link_index,
            "robot_link_name": native_link_name,
            "body_region": None
            if native_link_name not in BODY_REGION_BY_LINK
            else BODY_REGION_BY_LINK[native_link_name],
            "other_link_index": None
            if native_step is None
            else int(native_value("native_other_link", -1)),
            "other_link_name": _name_for_global_index(
                object_names, int(native_value("native_other_link", -1))
            ),
            "penetration_m": None
            if native_step is None
            else float(native_value("native_penetration", math.nan)),
        },
        "exact": {
            "contact": exact_step is not None,
            "first_contact_step": exact_step,
            "robot_link_index": None if exact_step is None else exact_link_index,
            "robot_link_name": exact_link_name,
            "body_region": None
            if exact_link_name not in BODY_REGION_BY_LINK
            else BODY_REGION_BY_LINK[exact_link_name],
            "other_link_index": None
            if exact_step is None
            else int(exact_value("exact_other_link", -1)),
            "other_link_name": _name_for_global_index(
                object_names, int(exact_value("exact_other_link", -1))
            ),
            "robot_geom_index": None
            if exact_step is None
            else int(exact_value("exact_robot_geom", -1)),
            "other_geom_index": None
            if exact_step is None
            else int(exact_value("exact_other_geom", -1)),
            "contact_count": 0
            if exact_step is None
            else int(exact_value("exact_contact_count", 0)),
            "penetration_m": None
            if exact_step is None
            else float(exact_value("exact_penetration", math.nan)),
            "manifold_position_m": None
            if exact_position is None
            else tuple(float(item) for item in np.asarray(exact_position)),
            "manifold_normal": None
            if exact_normal is None
            else tuple(float(item) for item in np.asarray(exact_normal)),
        },
    }
    return MappingProxyType(output)


class _SnapshotShell:
    __slots__ = (
        "solver_state",
        "step_index",
        "last_actions",
        "harness",
        "rng",
        "counters",
        "goal",
        "identity",
        "boundary",
        "digest",
    )


class _EpisodeStateShell:
    pass


class _DiscardedTorchStorage:
    """Inert placeholder for an unused tensor storage in a frozen snapshot."""

    __slots__ = ("serialized_bytes",)

    def __init__(self, serialized_bytes: bytes) -> None:
        self.serialized_bytes = len(serialized_bytes)


class _DiscardedTorchTensor:
    """Inert placeholder for unused goal-record tensors; never array-like."""

    __slots__ = ("shape",)

    def __init__(self, shape: Any) -> None:
        self.shape = tuple(int(value) for value in shape)


def _discard_torch_storage(serialized_bytes: bytes) -> _DiscardedTorchStorage:
    return _DiscardedTorchStorage(serialized_bytes)


def _discard_torch_tensor(
    _storage: _DiscardedTorchStorage,
    _storage_offset: int,
    size: Any,
    _stride: Any,
    _requires_grad: bool,
    _backward_hooks: Any,
    _metadata: Any = None,
) -> _DiscardedTorchTensor:
    return _DiscardedTorchTensor(size)


class _FrozenSnapshotUnpickler(pickle.Unpickler):
    """Unpickle only the hash-verified predecessor snapshot container.

    Mapping the two project classes to inert shells prevents importing the
    simulation runner or Genesis. NumPy arrays retain their normal loaders.
    Frozen Torch tensors occur only in the unused goal record and are mapped
    to inert shells, avoiding any Torch import or tensor allocation.
    """

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) == ("run_go2_oracle_branch_pilot_v1", "BranchSnapshot"):
            return _SnapshotShell
        if (module, name) == ("lewm_genesis.lewm_contract", "EpisodeState"):
            return _EpisodeStateShell
        if (module, name) == ("numpy._core.numeric", "_frombuffer"):
            # Compatibility with the NumPy 2.x module path used at freeze time.
            numeric = getattr(np, "_core", np.core).numeric
            return numeric._frombuffer
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _discard_torch_tensor
        if (module, name) == ("torch.storage", "_load_from_bytes"):
            return _discard_torch_storage
        allowed = {
            ("numpy", "dtype"),
            ("collections", "Counter"),
            ("collections", "OrderedDict"),
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"disallowed snapshot global: {module}.{name}")
        return super().find_class(module, name)


def _rotation(quaternion: Sequence[float]) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    _require(norm > 0 and math.isfinite(norm), "invalid frozen quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _quaternion_multiply(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    """Reproduce the predecessor's frozen compose function byte-for-byte.

    The predecessor intentionally remains the geometry lineage for this
    qualification.  Its x component omitted the conventional ``-az*by``
    term; correcting that here would silently change the 27-shape contract.
    This compatibility function is therefore not exported as general
    quaternion algebra.
    """

    aw, ax, ay, az = np.asarray(a, dtype=np.float64)
    bw, bx, by, bz = np.asarray(b, dtype=np.float64)
    return np.asarray(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def instantiate_collision_contract(
    link_transform: np.ndarray, contract: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    """Instantiate the predecessor's exact 27-shape analytical lineage."""

    links = np.asarray(link_transform, dtype=np.float64)
    _require(links.shape == (PROTECTED_LINK_COUNT, 7), "boundary link shape drift")
    output = np.empty((len(contract), 7), dtype=np.float64)
    for geometry_index, row in enumerate(contract):
        link_index = int(row["link_index"])
        parent_position = links[link_index, :3]
        parent_quaternion = links[link_index, 3:]
        output[geometry_index, :3] = parent_position + _rotation(
            parent_quaternion
        ) @ np.asarray(row["local_pos"], dtype=np.float64)
        output[geometry_index, 3:] = _quaternion_multiply(
            parent_quaternion, row["local_quat"]
        )
    output.setflags(write=False)
    return output


def _extract_boundary(
    snapshot: Any,
    geometry_record: Mapping[str, Any],
    contract: Sequence[Mapping[str, Any]],
) -> BoundaryTransforms:
    state = snapshot.solver_state
    for key in (_QPOS_KEY, _LINK_POS_KEY, _LINK_QUAT_KEY, _GEOM_POS_KEY, _GEOM_QUAT_KEY):
        _require(key in state, f"snapshot is missing {key}")
    raw_qpos = np.asarray(state[_QPOS_KEY])
    if raw_qpos.shape == (19, 1):
        raw_qpos = raw_qpos[:, 0]
    elif raw_qpos.shape == (1, 19):
        raw_qpos = raw_qpos[0]
    _require(raw_qpos.shape == (19,), f"snapshot qpos shape drift: {raw_qpos.shape}")
    qpos = _immutable_array(raw_qpos, dtype=np.float32)
    all_link_positions = _as_envless_array(state[_LINK_POS_KEY], 3)
    all_link_quaternions = _as_envless_array(state[_LINK_QUAT_KEY], 4)
    all_geom_positions = _as_envless_array(state[_GEOM_POS_KEY], 3)
    all_geom_quaternions = _as_envless_array(state[_GEOM_QUAT_KEY], 4)
    _require(
        len(all_link_positions) == len(all_link_quaternions),
        "snapshot link transform length drift",
    )
    _require(
        len(all_geom_positions) == len(all_geom_quaternions),
        "snapshot geometry transform length drift",
    )
    link_start = len(all_link_positions) - PROTECTED_LINK_COUNT
    geom_start = len(all_geom_positions) - COLLISION_SHAPE_COUNT
    _require(link_start >= 0 and geom_start >= 0, "snapshot robot suffix missing")
    link_indices = tuple(range(link_start, len(all_link_positions)))
    geom_indices = tuple(range(geom_start, len(all_geom_positions)))
    global_names = geometry_record["link_names"]
    link_names = tuple(_name_for_global_index(global_names, index) for index in link_indices)
    _require(link_names == PROTECTED_LINK_NAMES, "snapshot last-13 link suffix drift")
    links = _immutable_array(
        np.concatenate(
            (all_link_positions[link_start:], all_link_quaternions[link_start:]), axis=1
        ),
        dtype=np.float32,
    )
    raw_geometries = _immutable_array(
        np.concatenate(
            (all_geom_positions[geom_start:], all_geom_quaternions[geom_start:]), axis=1
        ),
        dtype=np.float32,
    )
    _require(raw_geometries.shape == (COLLISION_SHAPE_COUNT, 7), "last-27 geom drift")
    _require(np.isfinite(qpos).all(), "boundary qpos is not finite")
    _require(np.isfinite(links).all(), "boundary links are not finite")
    _require(np.isfinite(raw_geometries).all(), "boundary geometries are not finite")
    analytical = instantiate_collision_contract(links, contract)
    boundary = MappingProxyType(dict(snapshot.boundary))
    return BoundaryTransforms(
        snapshot_digest=str(snapshot.digest),
        boundary=boundary,
        qpos=qpos,
        link_global_indices=link_indices,
        link_names=tuple(str(name) for name in link_names),
        link_transform=links,
        raw_geom_global_indices=geom_indices,
        raw_geom_transform=raw_geometries,
        contract_geom_transform=analytical,
    )


def load_boundary_transforms(
    corpus: FrozenCorpus, state_id: str
) -> StateBoundaryTransforms:
    """Read exact current/successor boundaries from the gzip snapshot.

    The operation verifies the immutable snapshot SHA before unpickling and
    extracts arrays only.  No runtime context is created and no solver method
    is called.
    """

    source = corpus.corpus_record(state_id)
    geometry = corpus.geometry_record(state_id)
    path = _resolve_frozen_path(corpus.root, source["snapshot_path"])
    _require(sha256_file(path) == source["snapshot_sha256"], f"{state_id}: snapshot SHA drift")
    try:
        with gzip.open(path, "rb") as stream:
            snapshots = _FrozenSnapshotUnpickler(stream).load()
    except (OSError, pickle.UnpicklingError, ModuleNotFoundError) as error:
        raise CorpusBindingError(f"cannot load frozen snapshot {state_id}: {error}") from error
    _require(set(snapshots) == {"current", "successors"}, "snapshot container drift")
    contract = collision_shape_contract(geometry)
    current = _extract_boundary(snapshots["current"], geometry, contract)
    _require(
        current.snapshot_digest == source["current_boundary"]["snapshot_digest"],
        f"{state_id}: current snapshot digest drift",
    )
    successor_rows = {
        int(row["current_action_index"]): row for row in source["successor_rows"]
    }
    snapshots_by_action = {int(key): value for key, value in snapshots["successors"].items()}
    _require(
        set(snapshots_by_action) == set(successor_rows),
        f"{state_id}: successor snapshot key drift",
    )
    successors: dict[int, BoundaryTransforms] = {}
    for action, snapshot in snapshots_by_action.items():
        boundary = _extract_boundary(snapshot, geometry, contract)
        expected_digest = successor_rows[action]["boundary"]["snapshot_digest"]
        _require(
            boundary.snapshot_digest == expected_digest,
            f"{state_id}: successor {action} snapshot digest drift",
        )
        successors[action] = boundary
    return StateBoundaryTransforms(
        state_id=str(state_id),
        current=current,
        successors=MappingProxyType(dict(sorted(successors.items()))),
    )


def load_corpus_context(root: str | Path | None = None) -> FrozenCorpus:
    """Integration alias for :func:`load_frozen_corpus`."""

    return load_frozen_corpus(root)


def load_state(
    context: FrozenCorpus,
    state_id: str,
    *,
    shard_fields: Iterable[str] | None = (),
) -> LoadedFrozenState:
    """Load one complete, read-only state for streaming evaluation."""

    identity = str(state_id)
    source = context.corpus_record(identity)
    geometry = context.geometry_record(identity)
    shard = context.load_geometry_shard(identity, fields=shard_fields)
    boundaries = context.load_boundary_transforms(identity)
    rows = context.transition_identity_rows(identity)
    action_map = context.action_copy_map(identity)
    _require(
        len(rows) == shard.transition_count == action_map.transition_count,
        f"{identity}: loaded-state row custody mismatch",
    )
    return LoadedFrozenState(
        state_id=identity,
        scene_id=str(source["scene_id"]),
        family=str(source["family"]),
        role=context.role_for_state(identity),
        role_long_name=context.role_for_state(identity, long_name=True),
        source_record=source,
        geometry_record=geometry,
        current_rows=tuple(MappingProxyType(dict(row)) for row in source["current_rows"]),
        successor_rows=tuple(
            MappingProxyType(dict(row)) for row in source["successor_rows"]
        ),
        transition_rows=rows,
        scene_boxes=context.scene_obbs(identity),
        geometry_contract=context.collision_shape_contract(identity),
        protected_link_names=context.protected_link_names(identity),
        body_region_by_link=BODY_REGION_BY_LINK,
        action_copy_map=action_map,
        shard=shard,
        boundaries=boundaries,
    )


def _quaternion_max_difference(a: np.ndarray, b: np.ndarray) -> float:
    first = np.max(np.abs(a - b), axis=-1)
    second = np.max(np.abs(a + b), axis=-1)
    return float(np.max(np.minimum(first, second), initial=0.0))


def validate_boundary_alignment(
    corpus: FrozenCorpus,
    state_id: str,
    current_action_index: int,
    *,
    shard: GeometryShard | None = None,
    boundaries: StateBoundaryTransforms | None = None,
    atol: float = 2e-6,
) -> Mapping[str, Any]:
    """Validate last-27 extraction and a persisted successor endpoint.

    ``geom_transform`` is the predecessor's 27-shape analytical contract, not
    Genesis' raw geometry quaternion table.  It is therefore compared with
    ``contract_geom_transform``; the raw last-27 transform remains separately
    exposed for self-occlusion/raycast consumers.
    """

    geometry_shard = shard or corpus.load_geometry_shard(
        state_id, fields=("link_transform", "geom_transform")
    )
    state_boundaries = boundaries or corpus.load_boundary_transforms(state_id)
    action = int(current_action_index)
    _require(action in state_boundaries.successors, f"{state_id}: no successor {action}")
    transition = geometry_shard.transition_index("current", -1, action)
    endpoint = state_boundaries.successors[action]
    observed_links = geometry_shard.arrays["link_transform"][transition, -1]
    observed_geometries = geometry_shard.arrays["geom_transform"][transition, -1]
    link_position = float(np.max(np.abs(observed_links[:, :3] - endpoint.link_transform[:, :3])))
    link_quaternion = _quaternion_max_difference(
        observed_links[:, 3:], endpoint.link_transform[:, 3:]
    )
    geometry_position = float(
        np.max(np.abs(observed_geometries[:, :3] - endpoint.contract_geom_transform[:, :3]))
    )
    geometry_quaternion = _quaternion_max_difference(
        observed_geometries[:, 3:], endpoint.contract_geom_transform[:, 3:]
    )
    last_27 = (
        endpoint.raw_geom_transform.shape == (COLLISION_SHAPE_COUNT, 7)
        and endpoint.raw_geom_global_indices
        == tuple(
            range(
                endpoint.raw_geom_global_indices[0],
                endpoint.raw_geom_global_indices[0] + COLLISION_SHAPE_COUNT,
            )
        )
    )
    passed = (
        last_27
        and link_position <= atol
        and link_quaternion <= atol
        and geometry_position <= atol
        and geometry_quaternion <= atol
    )
    output = {
        "schema": "body_centric_range_boundary_alignment_v1",
        "state_id": str(state_id),
        "current_action_index": action,
        "transition_index": transition,
        "raw_boundary_geometry_is_last_27": bool(last_27),
        "link_position_max_abs_m": link_position,
        "link_quaternion_sign_invariant_max_abs": link_quaternion,
        "contract_geometry_position_max_abs_m": geometry_position,
        "contract_geometry_quaternion_sign_invariant_max_abs": geometry_quaternion,
        "atol": float(atol),
        "pass": bool(passed),
    }
    output["content_digest"] = json_content_digest(output)
    return MappingProxyType(output)


def validate_representative_geometry(
    shard: GeometryShard, copy_map: AppliedActionCopyMap
) -> Mapping[str, Any]:
    """Prove that every proposed copy has an identical physical trajectory."""

    arrays = shard.arrays
    fields = ("qpos", "link_transform", "geom_transform", "native_contact", "exact_contact")
    for field in fields:
        _require(field in arrays, f"representative validation requires {field}")
    mismatches: list[dict[str, Any]] = []
    for representative, copies in copy_map.copies_by_representative.items():
        for transition in copies:
            for field in fields:
                if not np.array_equal(arrays[field][representative], arrays[field][transition]):
                    mismatches.append(
                        {
                            "representative": representative,
                            "transition": transition,
                            "field": field,
                        }
                    )
    output = {
        "schema": "body_centric_range_applied_action_copy_validation_v1",
        "state_id": shard.state_id,
        "transitions": copy_map.transition_count,
        "representatives": copy_map.representative_count,
        "mismatches": mismatches,
        "pass": not mismatches,
    }
    output["content_digest"] = json_content_digest(output)
    return MappingProxyType(output)


__all__ = [
    "ACTION_CONTRACT_SHA256",
    "AppliedActionCopyMap",
    "BODY_REGION_BY_LINK",
    "BoundaryTransforms",
    "COLLISION_SHAPE_COUNT",
    "COMPLETED_RESULT_COMMIT",
    "CORPUS_INDEX_SHA256",
    "CORPUS_LOGICAL_DIGEST",
    "CorpusBindingError",
    "EXPECTED_CURRENT_TRANSITIONS",
    "EXPECTED_PHYSICS_STEPS",
    "EXPECTED_ROLE_COUNTS",
    "EXPECTED_STATES",
    "EXPECTED_SUCCESSOR_TRANSITIONS",
    "EXPECTED_TRANSITIONS",
    "FrozenCorpus",
    "GEOMETRY_INDEX_SHA256",
    "GeometryShard",
    "LoadedFrozenState",
    "PHYSICS_STEPS_PER_TRANSITION",
    "PROTECTED_LINK_COUNT",
    "PROTECTED_LINK_NAMES",
    "REPAIRED_ROW_LEDGER_SHA256",
    "ROLE_LONG_NAMES",
    "SOURCE_LINEAGE",
    "SPLIT_SHA256",
    "SceneOBBSet",
    "StateBoundaryTransforms",
    "applied_action_key",
    "body_region_for_link",
    "build_applied_action_copy_map",
    "collision_shape_contract",
    "contact_attribution",
    "instantiate_collision_contract",
    "json_content_digest",
    "load_corpus_context",
    "load_boundary_transforms",
    "load_frozen_corpus",
    "load_geometry_shard",
    "load_state",
    "scene_obbs",
    "sha256_file",
    "transition_identity_rows",
    "validate_boundary_alignment",
    "validate_representative_geometry",
]
