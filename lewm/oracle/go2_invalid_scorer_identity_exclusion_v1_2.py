"""Frozen exclusion for the abandoned 45-state scorer-fit identity attempt.

The three witnesses named below were selected before any candidate outcome was
generated.  They are scientifically ineligible, but their identities remain a
permanent exclusion.  This module deliberately reads only those three exact
files, verifies their byte bindings and self digests, and projects the physical
identity namespaces that later corpus stages must keep disjoint.

Scene exclusion is the primary rule.  Because the branch design permits only
one state per scene, excluding a witnessed scene also excludes every episode
cluster, state, observation and branch descended from that scene.  The more
specific projections are retained and checked as independent defence in depth;
they do not weaken the scene-level rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]

SUPERSEDED_SCORER_CONTRACT_DIGEST = (
    "a016cadbcdb4c90297871bee1c202bb5751b4567a38a39ba3a7126f3d91d9cba"
)
SUPERSEDED_SELECTION_DIGEST = (
    "207b7681abc24a88bb98f8f271c7648987ca90e68657c989a91ad668ec4e8559"
)

INVALID_SCORER_IDENTITY_EXCLUSION: dict[str, Any] = {
    "schema": "go2_invalid_scorer_identity_exclusion_v1_2",
    "status": "PRESERVED_INVALID_PRE_OUTCOME_IDENTITIES",
    "reason": (
        "abandoned partial scorer-fit identity selection under a superseded "
        "contract; only three of eight family shards existed and the set "
        "overlaps frozen factorial scenes"
    ),
    "outcomes_generated": False,
    "reuse_permitted": False,
    "superseded_scorer_contract_digest": SUPERSEDED_SCORER_CONTRACT_DIGEST,
    "superseded_selection_digest": SUPERSEDED_SELECTION_DIGEST,
    "witnesses": [
        {
            "path": (
                ".generated/go2_branch_corpus_v1_2_interrupted_a016cadb/"
                "scorer_fit/state_shard_large_enclosed_maze.json"
            ),
            "family": "large_enclosed_maze",
            "byte_count": 27_565,
            "sha256": (
                "9504b507ecf55b0aa64976669035e5e687c45cbe72c0fe02366a7e8938f5cf0c"
            ),
            "self_digest": (
                "09837fb3907a6b9dc7fa358a169a05bceba2f0b3e19354f97fc31538eacede30"
            ),
            "state_count": 15,
        },
        {
            "path": (
                ".generated/go2_branch_corpus_v1_2_interrupted_a016cadb/"
                "scorer_fit/state_shard_loop_alias_stress.json"
            ),
            "family": "loop_alias_stress",
            "byte_count": 22_002,
            "sha256": (
                "689b5abb3d5a4edaf5d3247d0d46283afb4e5a8c425ac7442c49077d10c6f8c1"
            ),
            "self_digest": (
                "f1ba0a54b2cee8e553513fe2c18d422dc783261cbdae58c0e90050ab520d0bda"
            ),
            "state_count": 15,
        },
        {
            "path": (
                ".generated/go2_branch_corpus_v1_2_interrupted_a016cadb/"
                "scorer_fit/state_shard_medium_enclosed_maze.json"
            ),
            "family": "medium_enclosed_maze",
            "byte_count": 26_370,
            "sha256": (
                "69372c6c177d346e3c9c67a58066a357b6d5f334f39762fa5cd2efb5d6a5eaa2"
            ),
            "self_digest": (
                "2610c6123ecfdc0d332618205ea8eb498f6cf5f523c68f014f0e893b4a131be2"
            ),
            "state_count": 15,
        },
    ],
    "derived_identity_bindings": {
        "scene_count": 45,
        "scene_ids_digest": (
            "5d5c4fef96e5132ad443c4fbd2778ad7d13fb9190328a498ca56490d53e041fe"
        ),
        "episode_cluster_count": 45,
        "episode_cluster_ids_digest": (
            "d04722aef2676cf7fc644ac5f684ff6ec8796ed6c6771481c06084a5c6b3ba0d"
        ),
        "physical_state_count": 45,
        "physical_state_keys_digest": (
            "31cdb80d0229204fa6b7ca8fd9e31574d60a33658c1973e7a0b0d8ab5e105fc2"
        ),
        "snapshot_observation_count": 45,
        "snapshot_observation_keys_digest": (
            "ade2b9ccdc26acc982cf38d80eb2526df3ce118a3e942fe3796fc04013b27be0"
        ),
        "registered_branch_count": 270,
        "registered_branch_keys_digest": (
            "3ae32d95d5863f55bc197a92f9e76d2face12ebc0234feaef82f0f49e16dfed4"
        ),
    },
    "exclusion_semantics": {
        "primary": "exclude every witnessed scene",
        "episode_cluster": "exclude every episode cluster in a witnessed scene",
        "state": "exclude every physical state in a witnessed scene",
        "observation": "exclude every observation in a witnessed scene",
        "branch": "exclude every candidate branch in a witnessed scene",
        "one_state_per_scene_contract_required": True,
    },
}


def _digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def invalid_identity_exclusion_digest() -> str:
    """Return the immutable amendment digest without reading runtime artefacts."""

    return _digest(INVALID_SCORER_IDENTITY_EXCLUSION)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class InvalidIdentityIndex:
    """Verified identity projections recovered from the three exact witnesses."""

    scene_ids: frozenset[str]
    episode_cluster_ids: frozenset[str]
    physical_state_keys: frozenset[tuple[str, int, int]]
    snapshot_observation_keys: frozenset[tuple[str, int]]
    registered_branch_keys: frozenset[tuple[str, int, int, int]]

    def binding(self) -> dict[str, Any]:
        frozen = INVALID_SCORER_IDENTITY_EXCLUSION["derived_identity_bindings"]
        return {
            "invalid_scorer_identity_exclusion_digest":
                invalid_identity_exclusion_digest(),
            **frozen,
            "witnesses_verified": True,
            "transitive_scene_descendant_exclusion": True,
        }


@lru_cache(maxsize=None)
def load_invalid_identity_index(root: Path | None = None) -> InvalidIdentityIndex:
    """Verify the exact preserved shards and recover their identity projections."""

    custody_root = ROOT if root is None else Path(root)
    states: list[dict[str, Any]] = []
    families: set[str] = set()
    for witness in INVALID_SCORER_IDENTITY_EXCLUSION["witnesses"]:
        path = custody_root / str(witness["path"])
        if not path.is_file():
            raise RuntimeError(f"invalid-identity witness is missing: {path}")
        raw = path.read_bytes()
        if (len(raw) != int(witness["byte_count"])
                or _sha256(raw) != witness["sha256"]):
            raise RuntimeError(f"invalid-identity witness byte binding failed: {path}")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid-identity witness is malformed: {path}") from exc
        expected_self = _digest({key: value for key, value in payload.items()
                                 if key != "state_manifest_digest"})
        if (payload.get("schema") != "go2_branch_corpus_v1_2_state_manifest"
                or payload.get("pool") != "scorer_fit"
                or payload.get("genesis_backend") != "cpu"
                or payload.get("scorer_contract_v1_2_digest")
                != SUPERSEDED_SCORER_CONTRACT_DIGEST
                or payload.get("selection_digest") != SUPERSEDED_SELECTION_DIGEST
                or payload.get("state_manifest_digest") != witness["self_digest"]
                or expected_self != witness["self_digest"]):
            raise RuntimeError(f"invalid-identity witness provenance failed: {path}")
        shard_states = payload.get("states")
        if not isinstance(shard_states, list) \
                or len(shard_states) != int(witness["state_count"]):
            raise RuntimeError(f"invalid-identity witness state count failed: {path}")
        shard_families = {str(state.get("family")) for state in shard_states}
        if shard_families != {str(witness["family"])}:
            raise RuntimeError(f"invalid-identity witness family failed: {path}")
        families.update(shard_families)
        states.extend(shard_states)

    if len(families) != len(INVALID_SCORER_IDENTITY_EXCLUSION["witnesses"]):
        raise RuntimeError("invalid-identity witnesses do not represent three families")

    scenes = frozenset(str(state["scene_id"]) for state in states)
    clusters = frozenset(
        f"{state['scene_id']}/env0/ep{int(state['episode_id'])}" for state in states
    )
    physical_states = frozenset(
        (str(state["scene_id"]), int(state["episode_id"]),
         int(state["source_step"])) for state in states
    )
    observations = frozenset(
        (str(state["scene_id"]), int(state["source_step"])) for state in states
    )
    branches = frozenset(
        (str(state["scene_id"]), int(state["episode_id"]),
         int(state["source_step"]), int(candidate))
        for state in states for candidate in state["candidate_indices"]
    )
    index = InvalidIdentityIndex(
        scene_ids=scenes,
        episode_cluster_ids=clusters,
        physical_state_keys=physical_states,
        snapshot_observation_keys=observations,
        registered_branch_keys=branches,
    )
    expected = INVALID_SCORER_IDENTITY_EXCLUSION["derived_identity_bindings"]
    observed = {
        "scene_count": len(index.scene_ids),
        "scene_ids_digest": _digest(sorted(index.scene_ids)),
        "episode_cluster_count": len(index.episode_cluster_ids),
        "episode_cluster_ids_digest": _digest(sorted(index.episode_cluster_ids)),
        "physical_state_count": len(index.physical_state_keys),
        "physical_state_keys_digest": _digest(
            sorted([list(key) for key in index.physical_state_keys])),
        "snapshot_observation_count": len(index.snapshot_observation_keys),
        "snapshot_observation_keys_digest": _digest(
            sorted([list(key) for key in index.snapshot_observation_keys])),
        "registered_branch_count": len(index.registered_branch_keys),
        "registered_branch_keys_digest": _digest(
            sorted([list(key) for key in index.registered_branch_keys])),
    }
    if observed != expected:
        raise RuntimeError("invalid-identity witness projection binding failed")
    return index


def disjointness_report(
        records: Iterable[Mapping[str, Any]],
        index: InvalidIdentityIndex | None = None) -> dict[str, Any]:
    """Check scene and descendant identity projections for state/branch records."""

    invalid = load_invalid_identity_index() if index is None else index
    overlap: dict[str, set[Any]] = {
        "scene": set(),
        "episode_cluster": set(),
        "physical_state": set(),
        "snapshot_observation": set(),
        "registered_branch": set(),
    }
    count = 0
    for record in records:
        count += 1
        scene = str(record.get("scene_id") or "")
        if scene in invalid.scene_ids:
            overlap["scene"].add(scene)
        cluster = record.get("episode_cluster_id")
        if isinstance(cluster, str) and cluster in invalid.episode_cluster_ids:
            overlap["episode_cluster"].add(cluster)
        try:
            episode = int(record["episode_id"])
            source_step = int(record["source_step"])
        except (KeyError, TypeError, ValueError):
            episode = source_step = -1
        state_key = (scene, episode, source_step)
        if state_key in invalid.physical_state_keys:
            overlap["physical_state"].add(state_key)
        observation_key = (scene, source_step)
        if observation_key in invalid.snapshot_observation_keys:
            overlap["snapshot_observation"].add(observation_key)
        candidates: list[int] = []
        if isinstance(record.get("candidate_indices"), list):
            candidates.extend(int(value) for value in record["candidate_indices"])
        if isinstance(record.get("candidate_index"), int):
            candidates.append(int(record["candidate_index"]))
        for candidate in candidates:
            branch_key = (scene, episode, source_step, candidate)
            if branch_key in invalid.registered_branch_keys:
                overlap["registered_branch"].add(branch_key)

    serialised = {
        name: sorted([list(value) if isinstance(value, tuple) else value
                      for value in values], key=lambda value: json.dumps(value))
        for name, values in overlap.items()
    }
    return {
        "invalid_scorer_identity_exclusion_digest":
            invalid_identity_exclusion_digest(),
        "records_checked": count,
        "overlap_counts": {name: len(values) for name, values in overlap.items()},
        "overlaps": serialised,
        "scene_cluster_state_observation_branch_disjoint":
            not any(overlap.values()),
    }


def assert_disjoint(
        records: Iterable[Mapping[str, Any]], *, label: str,
        index: InvalidIdentityIndex | None = None) -> dict[str, Any]:
    """Return a durable report or reject any invalid-attempt identity reuse."""

    report = disjointness_report(records, index=index)
    if not report["scene_cluster_state_observation_branch_disjoint"]:
        failed = [name for name, count in report["overlap_counts"].items() if count]
        raise RuntimeError(
            f"{label} reuses preserved invalid scorer identities at {failed}"
        )
    return report
